"""Validate immutable temporal artifacts; failed scientific gates remain valid evidence."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scripts.benchmark_temporal_changes import ROOT, sha, select, summarize
from sbercluster.temporal_changes import projected_tracking, detect_mean_changes


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def verify(out):
    reg=read(out/'registration.json'); cfg=reg['protocol']
    assert sha(out/'registration.json')==(out/'registration.sha256').read_text().strip()
    for p,h in {**reg['code_sha256'],**reg['input_sha256']}.items():
        assert sha(ROOT/p)==h, f'Frozen input/code changed: {p}'
    selected=read(out/'selected.json')
    assert sha(out/'selected.json')==(out/'selected.sha256').read_text().strip()
    assert read(out/'evaluation-start.json')['selected_sha256']==sha(out/'selected.json')
    assert selected['tune_records_sha256']==sha(out/'tune-per-seed.csv')
    tune=[]; evaluation=[]
    for phase,seeds in [('tune',reg['tune_seeds']),('evaluation',reg['evaluation_seeds'])]:
        for seed in seeds:
            point=read(out/'checkpoints'/f'{phase}-{seed}.json')
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
    assert saved['cells']==summary['cells']
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
    for p,h in read(out/'source-manifest.json')['inputs'].items(): assert sha(ROOT/p)==h
    return {'verification':'PASS','synthetic_promotion_gate':saved['synthetic_promotion_gate'],
            'selected':selected['selection'],'real_summary':real['summary']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/temporal-v4')
    args=parser.parse_args(); print(json.dumps(verify(args.output),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
