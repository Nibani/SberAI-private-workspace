"""Portable delivery checks for the immutable temporal experiment.

The registered producer, original verifier and all scientific bytes remain
hash-pinned. Recomputed NumPy/SciPy floating summaries use abs/rel1e-12 to
accommodate reduction/quantile implementations across platforms. This numerical
comparison tolerance is not a statistical threshold: structure, integer counts,
booleans, selection and every scientific gate still compare exactly. Every
non-bitexact float is printed, including accepted differences. The first Linux
failure lacked values, so its cause remains unconfirmed until this diagnostic
is run there.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scripts.benchmark_temporal_changes import ROOT, sha, select, summarize
from scripts.logical_storage import LogicalStorage
from scripts.verify_temporal_changes import read
from sbercluster.temporal_changes import projected_tracking, detect_mean_changes

FLOAT_ABS_TOL = 1e-12
FLOAT_REL_TOL = 1e-12


def compare_cells(saved, recomputed):
    """Strict tree comparison with diagnostics for finite computed floats only."""
    differences=[]; failures=[]

    def walk(a,b,path):
        if type(a) is not type(b):
            failures.append({'path':path,'reason':'type differs','saved':a,'recomputed':b})
        elif isinstance(a,dict):
            if a.keys()!=b.keys():
                failures.append({'path':path,'reason':'keys differ',
                                 'saved':list(a),'recomputed':list(b)})
            else:
                for key in a: walk(a[key],b[key],f'{path}.{key}')
        elif isinstance(a,list):
            if len(a)!=len(b):
                failures.append({'path':path,'reason':'length differs','saved':len(a),'recomputed':len(b)})
            else:
                for i,(x,y) in enumerate(zip(a,b)): walk(x,y,f'{path}[{i}]')
        elif isinstance(a,float):
            if not math.isfinite(a) or not math.isfinite(b):
                failures.append({'path':path,'reason':'nonfinite float','saved':repr(a),'recomputed':repr(b)})
            elif a.hex()!=b.hex():
                delta=abs(a-b); allowed=max(FLOAT_ABS_TOL,FLOAT_REL_TOL*max(abs(a),abs(b)))
                row={'path':path,'saved':a,'recomputed':b,'saved_hex':a.hex(),'recomputed_hex':b.hex(),
                     'absolute_difference':delta,'allowed_difference':allowed,'within_tolerance':delta<=allowed}
                differences.append(row)
                if delta>allowed: failures.append(row)
        elif a!=b:
            failures.append({'path':path,'reason':'exact value differs','saved':a,'recomputed':b})

    walk(saved,recomputed,'cells')
    return {'status':'PASS' if not failures else 'FAIL','absolute_tolerance':FLOAT_ABS_TOL,
            'relative_tolerance':FLOAT_REL_TOL,'non_bitexact_floats':differences,
            'max_absolute_difference':max((r['absolute_difference'] for r in differences),default=0.),
            'failures':failures}


def verify(out):
    storage=LogicalStorage(ROOT)
    reg=read(out/'registration.json'); cfg=reg['protocol']
    assert sha(out/'registration.json')==(out/'registration.sha256').read_text().strip()
    # Includes immutable scripts/verify_temporal_changes.py, not this delivery
    # checker. No registered pin is dropped, amended or replaced.
    for p,h in {**reg['code_sha256'],**reg['input_sha256']}.items():
        assert hashlib.sha256(storage.read_bytes(p)).hexdigest()==h, f'Frozen input/code changed: {p}'
    selected=read(out/'selected.json')
    assert sha(out/'selected.json')==(out/'selected.sha256').read_text().strip()
    assert read(out/'evaluation-start.json')['selected_sha256']==sha(out/'selected.json')
    assert selected['tune_records_sha256']==sha(out/'tune-per-seed.csv')
    tune=[]; evaluation=[]
    for phase,seeds in [('tune',reg['tune_seeds']),('evaluation',reg['evaluation_seeds'])]:
        for seed in seeds:
            path=out/'checkpoints'/f'{phase}-{seed}.json'
            point=(storage.read_json(path.resolve().relative_to(ROOT).as_posix())
                   if path.resolve().is_relative_to(ROOT) else read(path))
            rows=point['rows']
            expected=12*2*6 if phase=='tune' else 12*2
            assert len(rows)==expected
            assert all(r['seed']==seed and r['phase']==phase for r in rows)
            assert {(r['cell'],r['channel']) for r in rows}=={(c,ch) for c in cfg['cells'] for ch in cfg['channels']}
            (tune if phase=='tune' else evaluation).extend(rows)
    recomputed=select(tune,cfg)
    assert recomputed['selection']==selected['selection']
    assert recomputed['null_checks']==selected['null_checks']
    saved=read(out/'summary.json'); summary=summarize(evaluation,cfg)
    assert saved['status']=='COMPLETE'
    assert read(out/'invariants.json')['status']=='PASS'
    assert saved['deterministic_invariants']==read(out/'invariants.json')
    comparison=compare_cells(saved['cells'],summary['cells'])
    print(json.dumps({'temporal_float_comparison':comparison},ensure_ascii=False,allow_nan=False),flush=True)
    assert comparison['status']=='PASS', 'Temporal cells differ beyond numerical compatibility: see exact diagnostic paths above'
    assert saved['synthetic_promotion_gate']==summary['synthetic_promotion_gate']
    fixture=np.load(out/'real-constant-fixture.npz')
    const=projected_tracking(fixture['reference'],np.repeat(fixture['reference'][:,None],24,axis=1),
                             fixture['centers'],fixture['graph_labels'])
    assert const['persistent_changes']==0
    for row in fixture['reference']:
        assert detect_mean_changes(np.repeat(row[None],24,axis=0),1,3)['boundaries']==[]
    real=read(out/'real-correction.json'); entities=real['entities']
    assert len(entities)==2016 and list(entities)==fixture['ids'].tolist()
    assert real['summary']['constant_fixture_corrected_projection_flags']==0
    assert real['summary']['constant_fixture_historical_graph_false_flags']==48
    assert real['summary']==saved['real_correction']
    assert all(len(e['projected_labels'])==24 and len(e['observation_mask'])==24 for e in entities.values())
    assert sum(e['persistent_projection_change'] for e in entities.values())==real['summary']['actual_persistent_projection_changes']
    with (out/'real-monthly-projections.csv').open(encoding='utf-8',newline='') as stream:
        table=list(csv.DictReader(stream))
    assert len(table)==2016*24
    for r in table:
        e=entities[r['entity_id']]; t=real['periods'].index(r['period'])
        assert int(r['projected_label'])==e['projected_labels'][t]
        assert int(r['projected_baseline_label'])==e['projected_baseline_label']
    for p,h in read(out/'source-manifest.json')['inputs'].items():
        assert hashlib.sha256(storage.read_bytes(p)).hexdigest()==h
    return {'verification':'PASS','synthetic_promotion_gate':saved['synthetic_promotion_gate'],
            'selected':selected['selection'],'real_summary':real['summary'],
            'float_comparison':comparison}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/temporal-v4')
    args=parser.parse_args(); print(json.dumps(verify(args.output),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
