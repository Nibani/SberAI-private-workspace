"""Frozen temporal primary protocol: pilot, register, tune, evaluate, real export.

Run once into a new directory: python -m scripts.benchmark_temporal_changes.
No evaluation outcomes are generated until selected.json and its SHA are saved.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
from scipy.stats import beta
from sbercluster.temporal_changes import (detect_mean_changes, projected_tracking,
    fit_calendar, temporal_channels, train_variance, penalty_value, match_boundaries)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'configs/temporal_changes.json'
CODE_FILES = ['sbercluster/temporal_changes.py','scripts/benchmark_temporal_changes.py',
              'scripts/verify_temporal_changes.py','tests/test_temporal_changes.py',
              'docs/TEMPORAL_CHANGES.md','requirements-temporal.txt','configs/temporal_changes.json']


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csv_write(path, rows):
    if not rows: return Path(path).write_text('',encoding='utf-8')
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        fields=list(dict.fromkeys(key for row in rows for key in row))
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows(rows)
def rng(seed,purpose,*extra): return np.random.default_rng(np.random.SeedSequence([seed,purpose,*extra]))


def generate(seed, cell, cfg):
    """Truth is returned separately; detector receives only the observed array."""
    n,t,d=cfg['n'],cfg['t'],cfg['d']
    means=rng(seed,1).normal(size=(n,d))
    innovations=rng(seed,2).normal(size=(n,t,d))
    rho=0 if cell=='N0_iid' else .5
    noise=innovations.copy()
    for j in range(1,t): noise[:,j]=rho*noise[:,j-1]+np.sqrt(1-rho*rho)*innovations[:,j]
    phase=np.arange(t)*2*np.pi/12
    seasonal=np.zeros((t,d)); seasonal[:,0]=np.sin(phase); seasonal[:,1]=np.cos(phase)
    if cell=='N5_boundary_null': means[:,0]=0
    # N2 is the same repeated common seasonal null; other AR cells also share
    # the annual calendar and exactly the same development observations.
    x=means[:,None]+seasonal[None]+noise
    affected=rng(seed,3).choice(n,n//4,replace=False)
    truth={i:[] for i in range(n)}
    if cell=='N3_common_shift2':
        x[:,12:,0]+=2
        for i in range(n): truth[i]=[12]
    elif cell=='N4_spike4': x[affected,12,0]+=4
    elif cell in ['P025','P05','P1','P2']:
        delta={'P025':.25,'P05':.5,'P1':1,'P2':2}[cell]
        x[affected,12:,0]+=delta
        for i in affected: truth[int(i)]=[12]
    elif cell=='P6_return':
        x[affected,12:18,0]+=1
        for i in affected: truth[int(i)]=[12,18]
    elif cell=='C3_short_censored':
        x[affected,21:,0]+=2
        for i in affected: truth[int(i)]=[21]
    return x,truth,affected


def null_cell(cell,channel):
    return cell.startswith('N') and not (cell=='N3_common_shift2' and channel=='absolute')


def score_panel(signal,truth,cell,channel,c,variance,representative,min_size=3):
    penalty=penalty_value(c,variance,signal.shape[2],signal.shape[1])
    events=[]; false_objects=[]; all_tp=all_pred=all_true=per_tp=per_pred=per_true=0
    errors=[]; per_errors=[]; false_boundaries=0; entries=returns=0
    for i,row in enumerate(signal):
        result=detect_mean_changes(row,penalty,min_size)
        true=truth[i] if not (cell=='N3_common_shift2' and channel=='relative') else []
        eligible=[] if cell=='C3_short_censored' else true
        matches=match_boundaries(result['boundaries'],true)
        pmatches=match_boundaries(result['persistent_boundaries'],eligible)
        all_tp+=len(matches); all_pred+=len(result['boundaries']); all_true+=len(true)
        per_tp+=len(pmatches); per_pred+=len(result['persistent_boundaries']); per_true+=len(eligible)
        false_objects.append(bool(len(result['persistent_boundaries'])-len(pmatches)))
        false_boundaries+=len(result['boundaries'])-len(matches)
        errors.extend(abs(p-q) for p,q in matches); per_errors.extend(abs(p-q) for p,q in pmatches)
        entries+=sum(q==12 for p,q in pmatches); returns+=sum(q==18 for p,q in pmatches)
        for event in result['events']:
            events.append({'entity_index':i,**event,'effect_vector':json.dumps(event['effect_vector'])})
    metric={'penalty':penalty,'train_variance':variance,
        'false_persistent_object_fraction':float(np.mean(false_objects)),
        'representative':int(representative),'representative_false':int(false_objects[representative]),
        'false_boundaries_per_object_year':false_boundaries/(len(signal)*signal.shape[1]/12),
        'all_tp':all_tp,'all_predicted':all_pred,'all_truth':all_true,
        'persistent_tp':per_tp,'persistent_predicted':per_pred,'persistent_truth':per_true,
        'all_recall':all_tp/all_true if all_true else None,
        'all_precision':all_tp/all_pred if all_pred else None,
        'persistent_recall':per_tp/per_true if per_true else None,
        'persistent_precision':per_tp/per_pred if per_pred else None,
        'mean_absolute_onset_error':float(np.mean(errors)) if errors else None,
        'persistent_mean_absolute_onset_error':float(np.mean(per_errors)) if per_errors else None,
        'persistent_entry_matches':entries,'persistent_return_matches':returns,
        'censored_truth':sum(bool(v) for v in truth.values()) if cell=='C3_short_censored' else 0,
        'mean_effect_norm':float(np.mean([e['effect_norm'] for e in events])) if events else None}
    return metric,events


def seed_run(seed, cfg, policies, phase, deadline=None):
    rows=[]; events=[]; calibration=[]
    for ci,cell in enumerate(cfg['cells']):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('Technical pilot time budget reached between cells')
        raw,truth,affected=generate(seed,cell,cfg)
        offsets=fit_calendar(raw)
        channels=temporal_channels(raw,offsets)
        representative=int(rng(seed,4,ci).integers(cfg['n']))
        for channel in cfg['channels']:
            y=channels[channel]; variance=train_variance(y)
            calibration.append({'seed':seed,'cell':cell,'channel':channel,'variance':variance,
                                'calendar':offsets.tolist(),'representative':representative,
                                'affected':affected.tolist()})
            cs=policies[channel]
            if not cs:
                rows.append({'phase':phase,'seed':seed,'cell':cell,'channel':channel,'c':None,
                             'status':'NO_CALIBRATED_CANDIDATE','null_cell':null_cell(cell,channel)})
            for c in cs:
                metric,ev=score_panel(y,truth,cell,channel,c,variance,representative,cfg['min_segment_length'])
                rows.append({'phase':phase,'seed':seed,'cell':cell,'channel':channel,'c':c,
                             'status':'COMPLETE','null_cell':null_cell(cell,channel),**metric})
                events.extend({'phase':phase,'seed':seed,'cell':cell,'channel':channel,'c':c,**e} for e in ev)
    return rows,events,calibration


def technical_invariants():
    """Small deterministic fixtures, independent of all tune/evaluation seeds."""
    ref=np.array([[0.],[1.],[1.1],[2.1]])
    x=np.repeat(ref[:,None],24,axis=1); centers=np.array([[.55],[1.55]])
    graph=np.array([0,1,0,1]); result=projected_tracking(ref,x,centers,graph)
    assert result['persistent_changes']==0
    assert result['projected_baseline'].tolist()==[0,0,1,1]
    for row in x: assert detect_mean_changes(row,1,3)['boundaries']==[]
    shifted=np.zeros((24,2)); shifted[12:,0]=5
    assert detect_mean_changes(shifted,1,3)['persistent_boundaries']==[12]
    calendar=np.column_stack([np.sin(np.arange(12)*2*np.pi/12),np.cos(np.arange(12)*2*np.pi/12)])
    means=np.arange(18,dtype=float).reshape(9,2)
    raw=means[:,None]+np.tile(calendar,(2,1))[None]
    fitted=fit_calendar(raw); channels=temporal_channels(raw,fitted)
    np.testing.assert_allclose(channels['absolute'],np.repeat(means[:,None],24,axis=1),atol=1e-13)
    changed=raw.copy(); changed[:,12:]+=[3.,-2.]
    shock=temporal_channels(changed,fitted)
    np.testing.assert_allclose(shock['relative'],channels['relative'],atol=1e-13)
    np.testing.assert_allclose(shock['common_drift'][12:]-channels['common_drift'][12:],np.tile([3.,-2.],(12,1)),atol=1e-13)
    np.testing.assert_array_equal(fit_calendar(changed),fitted)
    assert train_variance(changed)==train_variance(raw)
    perm=np.array([2,0,3,1])
    other=projected_tracking(ref[perm],x[perm],centers,graph[perm])
    np.testing.assert_array_equal(other['projected_labels'],result['projected_labels'][perm])
    for invalid in [np.array([[np.nan],[0.]]),np.array([[np.inf],[0.]])]:
        try: detect_mean_changes(invalid,1,3)
        except ValueError: pass
        else: raise AssertionError('Nonfinite fixture accepted')
    return {'status':'PASS','constant_final_features':True,'exact_persistent_shift':True,
            'raw_calendar_correction':True,'common_shock_preserved_and_relative_removed':True,
            'evaluation_calibration_isolation':True,'projection_permutation_equivariance':True,
            'invalid_input_explicit':True,'numeric_tolerance':1e-13}


def pilot_run(out,cfg,max_seconds):
    """Bounded single fixture run, never reading real scientific inputs."""
    begin=time.monotonic(); manifest={p:sha(ROOT/p) for p in CODE_FILES}
    invariants=technical_invariants(); save(out/'invariants.json',invariants)
    try:
        rows,_,_=seed_run(cfg['fixture_seed'],cfg,{ch:[1] for ch in cfg['channels']},
                          'fixture_pilot',deadline=begin+max_seconds)
    except TimeoutError:
        result={'status':'INCOMPLETE','reason':'technical pilot time budget',
                'seconds':time.monotonic()-begin,'code_sha256':manifest}
        save(out/'pilot.json',result); raise SystemExit('INCOMPLETE technical pilot; tune/eval not started')
    elapsed=time.monotonic()-begin
    result={'status':'COMPLETE','fixture_seed':cfg['fixture_seed'],'seconds':elapsed,
        'segmentations':cfg['n']*len(cfg['cells'])*len(cfg['channels']),
        'records':len(rows),'purpose':'technical runtime/schema only; not selection or evaluation',
        'estimated_primary_seconds':elapsed*(16*6+64),'code_sha256':manifest,
        'deterministic_invariants':invariants,'max_seconds':max_seconds,
        'timeout_scope':'checked between twelve small cells; Python imports are outside this timer'}
    save(out/'pilot.json',result)
    return result


def select(tune,cfg):
    selected={}; checks=[]
    for channel in cfg['channels']:
        chosen=None
        for c in cfg['c_grid']:
            pass_cells=[]
            for cell in cfg['cells']:
                if not null_cell(cell,channel): continue
                subset=[r for r in tune if r['channel']==channel and r['cell']==cell and r['c']==c]
                fpr=float(np.mean([r['false_persistent_object_fraction'] for r in subset]))
                ok=len(subset)==len(cfg['tune_seeds']) and fpr<=.05
                checks.append({'channel':channel,'c':c,'cell':cell,'mean_fpr':fpr,'pass':ok})
                pass_cells.append(ok)
            if chosen is None and all(pass_cells): chosen=c
        selected[channel]=chosen
    return {'selection':selected,'null_checks':checks,'rule':cfg['calibration'],
            'status':'SELECTED' if all(c is not None for c in selected.values()) else 'NO_CALIBRATED_CANDIDATE'}


def summarize(rows,cfg):
    curves=[]; gates=[]
    boot=rng(cfg['bootstrap_seed'],5).integers(0,cfg['eval_seed_count'],size=(10000,cfg['eval_seed_count']))
    for channel in cfg['channels']:
        for cell in cfg['cells']:
            subset=[r for r in rows if r['channel']==channel and r['cell']==cell]
            if any(r['status']!='COMPLETE' for r in subset):
                curves.append({'channel':channel,'cell':cell,'status':'NO_CALIBRATED_CANDIDATE'})
                gates.append(False); continue
            item={'channel':channel,'cell':cell,'status':'COMPLETE','n_seeds':len(subset),'c':subset[0]['c']}
            for metric in ['false_persistent_object_fraction','all_recall','persistent_recall',
                           'all_precision','persistent_precision','mean_absolute_onset_error',
                           'persistent_mean_absolute_onset_error','false_boundaries_per_object_year','mean_effect_norm']:
                values=[r[metric] for r in subset]
                if all(v is None for v in values): item[metric]=None; item[metric+'_ci95']=None; continue
                a=np.array([np.nan if v is None else v for v in values],float)
                item[metric]=float(np.nanmean(a))
                samples=a[boot]; counts=np.isfinite(samples).sum(axis=1)
                means=np.divide(np.nansum(samples,axis=1),counts,out=np.full(10000,np.nan),where=counts>0)
                item[metric+'_ci95']=np.nanpercentile(means,[2.5,97.5]).tolist()
            failures=sum(r['representative_false'] for r in subset); n=len(subset)
            upper=float(beta.ppf(.95,failures+1,n-failures)) if failures<n else 1.
            item['representative_failures']=failures; item['representative_cp_upper95']=upper
            item['null_cell']=null_cell(cell,channel)
            item['fpr_gate']=not item['null_cell'] or (item['false_persistent_object_fraction']<=.05 and upper<=.05)
            if item['null_cell']: gates.append(item['fpr_gate'])
            curves.append(item)
    return {'scope':cfg['scope'],'status':'COMPLETE','cells':curves,
            'synthetic_promotion_gate':'PASS' if all(gates) else 'FAIL',
            'economic_ground_truth':False,'prospective_monitoring_validated':False,
            'uncertainty':cfg['uncertainty'],'stop':cfg['stop']}


def export_real(out, cfg, selected):
    from sbercluster import panel as P
    import pandas as pd
    panel=P.read_panel(ROOT/'data/v12/panel.csv.gz')
    model=np.load(ROOT/'reports/v1.2/model.npz')
    if not np.array_equal(model['ids'],panel.ids): raise ValueError('Frozen model ID order mismatch')
    labels=pd.read_csv(ROOT/'reports/v1.2/labels.csv',dtype={'monthly_types':str}).set_index('entity_id').loc[panel.ids]
    graph=labels['type_2023'].to_numpy(int)
    # Historical model.npz predates serializing level_center. Its level used
    # contemporaneous median log-total of this same frozen complete cohort.
    # Reconstruct only that legacy projection coordinate; new CP calibration
    # remains the separate first12-month intensity transform below.
    legacy_level_reconstructed='level_center' not in model.files
    level_center=(np.median(np.log(panel.totals),axis=0) if legacy_level_reconstructed
                  else model['level_center'])
    scaler=P.Scaler(model['structure_center'],model['scale'],12,level_center)
    xm=P.monthly_attributes(panel,scaler); ref=P.annual_profile(xm,slice(0,12))
    common=P.national_component(xm)-np.median(ref,axis=0)
    adjusted=xm-common[None]
    projection=projected_tracking(ref,adjusted,model['centers'],graph)
    if not np.array_equal(projection['projected_labels'],np.array([[int(c) for c in str(s)] for s in labels['monthly_types']])):
        raise ValueError('Historical monthly labels differ: refusing silent transform drift')
    constants=projected_tracking(ref,np.repeat(ref[:,None],24,axis=1),model['centers'],graph)
    historical_false=int(np.sum(constants['projected_labels'][:,-1]!=graph))
    # Separate intensity-only CP calibration uses ONLY 2023 for all fitted values.
    logs=np.log(panel.shares); train=logs[:,:12].reshape(-1,5)
    center=np.median(train,axis=0); scale=np.subtract(*np.percentile(train,[75,25],axis=0))
    raw=(logs-center)/scale/np.sqrt(5)
    calendar=fit_calendar(raw); channels=temporal_channels(raw,calendar)
    entities={}; table=[]; cp_counts={}
    for i,entity_id in enumerate(panel.ids):
        entities[str(entity_id)]={'graph_label':int(graph[i]),
            'projected_baseline_label':int(projection['projected_baseline'][i]),
            'projected_labels':projection['projected_labels'][i].tolist(),
            'projection_disagreement':bool(projection['projection_disagreement'][i]),
            'distance_margin':projection['distance_margin'][i].tolist(),
            'boundary_crossed':projection['boundary_crossed'][i].tolist(),
            'persistent_projection_change':bool(projection['persistent_change_mask'][i]),
            'observation_status':'complete_24_observations_fixed_2016_cohort',
            'observation_mask':[True]*24,'cp':{}}
    calibration={'center':center.tolist(),'scale':scale.tolist(),'calendar':calendar.tolist(),
                 'fitted_period':'2023-01..2023-12','intensity_coordinates':5,'level_included':False,
                 'boundary_identity_limit':'Canonical IDs and original2016 geometry fixed; no independent monthly territorial-boundary verification.',
                 'channels':{}}
    for channel in cfg['channels']:
        variance=train_variance(channels[channel]); c=selected['selection'][channel]
        calibration['channels'][channel]={'variance':variance,'c':c}
        persistent=boundaries=0
        for i,entity_id in enumerate(panel.ids):
            if c is None:
                result={'status':'NO_CALIBRATED_CANDIDATE','boundaries':[], 'events':[],'persistent_boundaries':[]}
            else:
                result=detect_mean_changes(channels[channel][i],penalty_value(c,variance,5),3)
                result['status']='RETROSPECTIVE_RESEARCH_CANDIDATE'
            entities[str(entity_id)]['cp'][channel]=result
            persistent+=bool(result['persistent_boundaries']); boundaries+=len(result['boundaries'])
        cp_counts[channel]={'entities_with_persistent_candidates':persistent,'boundaries':boundaries,
                            'status':'RETROSPECTIVE_RESEARCH_CANDIDATE' if c is not None else 'NO_CALIBRATED_CANDIDATE'}
    for entity_id,e in entities.items():
        for t,period in enumerate(panel.periods):
            table.append({'entity_id':entity_id,'period':period,'graph_label':e['graph_label'],
                          'projected_baseline_label':e['projected_baseline_label'],
                          'projected_label':e['projected_labels'][t],'distance_margin':e['distance_margin'][t],
                          'projection_disagreement':e['projection_disagreement'],
                          'boundary_crossed':e['boundary_crossed'][t],
                          'persistent_projection_change_at_end':e['persistent_projection_change']})
    csv_write(out/'real-monthly-projections.csv',table)
    save(out/'real-calibration.json',calibration)
    real={'revision':cfg['protocol_id'],'periods':list(panel.periods),'entities':entities,
          'common_drift':{'projection_v12_six_coordinates':common.tolist(),
                          'cp_five_intensity_coordinates':channels['common_drift'].tolist()},
          'summary':{'n_entities':len(panel.ids),'n_months':24,'retrospective':True,
            'constant_fixture_historical_graph_false_flags':historical_false,
            'constant_fixture_corrected_projection_flags':constants['persistent_changes'],
            'projection_disagreements':int(projection['projection_disagreement'].sum()),
            'actual_persistent_projection_changes':projection['persistent_changes'],
            'historical_actual_persistent_flags':int(labels['persistent_change'].sum()),
            'cp':cp_counts,'economic_ground_truth':False,
            'interpretation':'Projected boundary crossings and fitted profile segments; no confirmed economic transitions.'}}
    save(out/'real-correction.json',real)
    np.savez_compressed(out/'real-constant-fixture.npz',ids=panel.ids,reference=ref,
        centers=model['centers'],graph_labels=graph,projected_baseline=constants['projected_baseline'])
    save(out/'source-manifest.json',{'inputs':{p:sha(ROOT/p) for p in
        ['data/v12/panel.csv.gz','reports/v1.2/model.npz','reports/v1.2/labels.csv','configs/v12.json']},
        'reference_cohort_ids_sha256':hashlib.sha256('\n'.join(panel.ids).encode()).hexdigest(),
        'projection_transform':'Pinned v12 scaler, first12 median reference, monthly cohort median shift; six coordinates. Unchanged monthly labels; corrected baseline only.',
        'legacy_level_reference_reconstructed_from_pinned_panel':legacy_level_reconstructed,
        'cp_transform':'New train2023 frozen log-intensity median/IQR/sqrt5 and calendar; absolute and transductive relative channels.',
        'scope':'2016 fixed canonical IDs,24 observed months2023-2024; historical2024 reused; no real event labels.'})
    return real['summary']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'reports/temporal-v4')
    parser.add_argument('--pilot-only',action='store_true',help='Fixture only: no registration, tune, evaluation or real inputs')
    parser.add_argument('--pilot-max-seconds',type=float,default=120,help='Pilot timer checked between cells, excluding Python imports')
    parser.add_argument('--pilot-report',type=Path,help='Reuse completed pilot with identical code/config hashes')
    args=parser.parse_args(); out=args.output
    if args.pilot_max_seconds<=0 or not np.isfinite(args.pilot_max_seconds): raise SystemExit('Invalid pilot budget')
    if args.pilot_only and args.pilot_report is not None: raise SystemExit('Pilot-only cannot reuse a pilot')
    if out.exists(): raise SystemExit('Output exists: refusing overwrite of registered outcomes')
    out.mkdir(parents=True); cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
    if importlib.metadata.version('ruptures')!='1.1.10': raise RuntimeError('Pinned ruptures1.1.10 required')
    start=time.monotonic()
    # Technical fixture pilot is not a tune/evaluation seed and cannot select c.
    if args.pilot_report is None:
        pilot=pilot_run(out,cfg,args.pilot_max_seconds)
    else:
        pilot=json.loads(args.pilot_report.read_text(encoding='utf-8'))
        if pilot['status']!='COMPLETE' or pilot['code_sha256']!={p:sha(ROOT/p) for p in CODE_FILES}:
            raise SystemExit('Pilot incomplete or code/config drift: no tune started')
        save(out/'pilot.json',pilot); save(out/'invariants.json',pilot['deterministic_invariants'])
    if args.pilot_only:
        print(json.dumps(pilot,ensure_ascii=False)); return
    registration={'protocol':cfg,'code_sha256':{p:sha(ROOT/p) for p in CODE_FILES},
        'config_sha256':sha(CONFIG),'purpose_map':cfg['purpose_ids'],
        'input_sha256':{p:sha(ROOT/p) for p in ['data/v12/panel.csv.gz',
            'reports/v1.2/model.npz','reports/v1.2/labels.csv','configs/v12.json']},
        'tune_seeds':cfg['tune_seeds'],
        'evaluation_seeds':list(range(cfg['eval_seed_start'],cfg['eval_seed_start']+cfg['eval_seed_count'])),
        'environment':{'python':sys.version,'platform':platform.platform(),
            'versions':{p:importlib.metadata.version(p) for p in ['numpy','scipy','ruptures']}},
        'scorer_revision':'zero-based-start-boundaries-v1-matching-dp',
        'pre_fit_exclusions':'m6 and broad stresses deferred before fitting by frozen config',
        'missing_selection_behavior':'No selected policy => all registered eval cells marked NO_CALIBRATED_CANDIDATE, never PASS; no post-outcome extended grid.'}
    save(out/'registration.json',registration)
    (out/'registration.sha256').write_text(sha(out/'registration.json')+'\n',encoding='ascii')
    all_tune=[]; all_eval=[]; all_events=[]
    (out/'checkpoints').mkdir()
    for phase,seeds,policies in [('tune',cfg['tune_seeds'],{ch:cfg['c_grid'] for ch in cfg['channels']}),
                              ('evaluation',registration['evaluation_seeds'],None)]:
        if phase=='evaluation':
            selected=select(all_tune,cfg); selected['tune_records_sha256']=sha(out/'tune-per-seed.csv')
            selected['registration_sha256']=sha(out/'registration.json')
            save(out/'selected.json',selected)
            (out/'selected.sha256').write_text(sha(out/'selected.json')+'\n',encoding='ascii')
            save(out/'evaluation-start.json',{'selected_sha256':sha(out/'selected.json'),
                 'registration_sha256':sha(out/'registration.json'),'eval_seeds':seeds})
            policies={ch:[] if c is None else [c] for ch,c in selected['selection'].items()}
        for seed in seeds:
            if time.monotonic()-start>3600:
                save(out/'summary.json',{'status':'INCOMPLETE','reason':'registered60min resource budget',
                    'completed_tune_seeds':len(all_tune)//(12*2*6),'completed_eval_records':len(all_eval)})
                raise SystemExit('INCOMPLETE: resource budget reached; checkpoints retained')
            rows,events,calibration=seed_run(seed,cfg,policies,phase)
            save(out/'checkpoints'/f'{phase}-{seed}.json',{'rows':rows,'events':events,'calibration':calibration})
            (all_tune if phase=='tune' else all_eval).extend(rows); all_events.extend(events)
            import psutil
            with (out/'resource-monitor.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps({'phase':phase,'seed':seed,'elapsed_seconds':time.monotonic()-start,
                    'rss_bytes':psutil.Process().memory_info().rss})+'\n')
            print(f'{phase} {seed}: {len(rows)} records, elapsed {time.monotonic()-start:.1f}s',flush=True)
        csv_write(out/f'{phase if phase=="tune" else "eval"}-per-seed.csv',all_tune if phase=='tune' else all_eval)
    csv_write(out/'events.csv',all_events)
    summary=summarize(all_eval,cfg)
    summary['deterministic_invariants']=pilot['deterministic_invariants']
    summary['real_correction']=export_real(out,cfg,selected)
    summary['runtime_seconds']=time.monotonic()-start
    summary['selected']=selected['selection']
    summary['registration_sha256']=sha(out/'registration.json'); summary['selected_sha256']=sha(out/'selected.json')
    save(out/'summary.json',summary); save(out/'curves.json',summary['cells'])
    print(json.dumps({'status':summary['status'],'synthetic_promotion_gate':summary['synthetic_promotion_gate'],
                      'selected':selected['selection'],'seconds':summary['runtime_seconds']},ensure_ascii=False))


if __name__=='__main__': main()
