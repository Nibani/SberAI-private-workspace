"""Validate a completed fixed experiment, independently redoing saved arithmetic."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from sbercluster import conditional_profiles as cp
from sbercluster.metrics import all_metrics
from scripts.run_conditional_profiles import ROOT, CODE, sha, load_inputs


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def close(a,b):
    if isinstance(a, dict):
        assert set(a)==set(b), (set(a)-set(b),set(b)-set(a))
        for k in a:
            close(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):
            close(x,y)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        assert np.isfinite(a) and np.isclose(a,b,rtol=1e-9,atol=1e-10),(a,b)
    else:
        assert a==b,(a,b)


def restore_transform(model):
    class Predictor:
        def predict(self, design):
            return design @ model['ridge_coef'].T + model['ridge_intercept']
    return cp.ConditionalTransform(model['statuses'].tolist(),
        (model['basis_median'],model['basis_mean'],model['basis_scale']),Predictor(),
        model['center'],model['scale'],model['active'])


def check_transform(model, level, shares, status, labor, train):
    tf=restore_transform(model)
    known=sorted(set(str(s) for s in status[train] if str(s)))
    assert tf.statuses==known
    basis=np.column_stack([level[train],level[train]**2])
    for a,b in zip(tf.basis_scaling,cp.standardize_fit(basis)):
        assert np.allclose(a,b,atol=1e-12)
    d=np.column_stack([cp.standardize_apply(basis,tf.basis_scaling),cp.status_design(status[train],known)])
    centered=d-d.mean(axis=0)
    coefficient=np.linalg.solve(centered.T@centered+10*np.eye(d.shape[1]),centered.T@(shares[train]-shares[train].mean(axis=0))).T
    assert np.allclose(coefficient,model['ridge_coef'],atol=1e-9)
    assert np.allclose(shares[train].mean(axis=0)-d.mean(axis=0)@coefficient.T,model['ridge_intercept'],atol=1e-9)
    raw=np.column_stack([tf.residual(level[train],shares[train],status[train]),labor[train]])
    assert np.allclose(tf.center,np.median(raw,axis=0))
    scale=np.quantile(raw,.75,axis=0)-np.quantile(raw,.25,axis=0)
    assert np.allclose(tf.scale,scale)
    assert np.array_equal(tf.active,scale>0)
    coords=tf.coordinates(level,shares,status,labor)
    assert np.allclose(coords,model['coordinates'])
    assert np.allclose(cp.block_attributes(coords,tf.active),model['attributes'])
    return tf


def verify(output):
    summary=read(output/'summary.json')
    assert summary['status']=='COMPLETE','Incomplete experiments never pass'
    manifest=read(output/'results-manifest.json')
    required=['registration.json','registration.sha256','cohort.csv','predictions.csv','types.csv','labels.csv',
              'graph-controls.json','stability.json','summary.json','atlas.json','report.md','models.npz',*[f'fold-{f}.npz' for f in range(5)]]
    assert set(required)<=set(manifest)
    for path,digest in manifest.items():
        assert (output/path).is_file() and sha(output/path)==digest,path
    reg=read(output/'registration.json')
    assert sha(output/'registration.json')==(output/'registration.sha256').read_text().strip()
    config_path=ROOT/reg['config_path']
    assert sha(config_path)==reg['config_sha256']
    config=read(config_path)
    assert reg['protocol_sha256']==config['protocol_sha256']==sha(ROOT/config['protocol_path'])
    assert reg['protocol_sha256']=='5310933e84f7e40e5b3c0387fdcfede9f62b12b8b70f01467324cba928336e08'
    assert config['seed']==20261009 and config['ridge_alpha']==10 and config['k']==4 and config['n_init']==20
    assert config['graph_alpha']==.5 and config['max_sweeps']==50 and config['neighbors']==15
    assert config['null_draws']==199 and config['null_seed']==20261109
    assert config['bootstrap_draws']==1999 and config['bootstrap_seed']==20261209 and config['runtime_seconds']==3600
    assert set(reg['code_sha256'])==set(CODE)
    for path,digest in {**reg['code_sha256'],**reg['inputs_sha256']}.items():
        assert sha(ROOT/path)==digest,path
    assert reg['registered_before_first_fit'] is True
    data=load_inputs(config)
    p=data['panel']; idx=np.where(data['labor_valid'])[0]
    ids=p.ids[idx]; regions=data['regions'][idx]; labor=data['labor'][idx]; status=data['status'][idx]
    level=np.median(np.log(p.totals[idx,:12]),axis=1); shares=np.median(p.shares[idx,:12]*100,axis=1)
    outcome=data['outcome'][idx]
    wage23=np.log(np.where(data['wage23'][idx]>0,data['wage23'][idx],np.nan))
    wage24=np.log(np.where(data['wage24'][idx]>0,data['wage24'][idx],np.nan))
    pop=np.log(np.where(data['population'][idx]>0,data['population'][idx],np.nan))
    assert reg['model_ids']==ids.tolist() and reg['outcome_ids']==ids[outcome].tolist()
    assert reg['fold_regions']==sorted(set(regions)) and reg['folds']==cp.region_folds(regions)
    with (output/'predictions.csv').open(encoding='utf-8',newline='') as stream:
        rows=list(csv.DictReader(stream))
    for row in rows:
        for col in ['prediction','target','log_wage_2023']:
            assert np.isfinite(float(row[col]))
    by_method={m:{r['entity_id']:r for r in rows if r['method']==m} for m in cp.METHODS}
    assert len(rows)==int(outcome.sum())*13
    for m in cp.METHODS:
        assert len([r for r in rows if r['method']==m])==len(by_method[m])==int(outcome.sum())
        assert set(by_method[m])==set(ids[outcome])
    stability=read(output/'stability.json')
    assert stability['status']=='COMPLETE'
    for fold in range(5):
        model=np.load(output/f'fold-{fold}.npz',allow_pickle=False)
        train=np.array([reg['folds'][r]!=fold for r in regions]); test=~train
        assert np.array_equal(model['train'],train)
        assert np.array_equal(model['supervised'],train&outcome) and np.array_equal(model['eligible'],test&outcome)
        assert not set(regions[train])&set(regions[test])
        assert reg['fold_model_ids'][str(fold)]=={'train':ids[train].tolist(),'test':ids[test].tolist()}
        tf=check_transform(model,level,shares,status,labor,train)
        x=model['attributes']; coords=model['coordinates']
        for labels,centers in [('train_labels','centers'),('train_graph_labels','graph_centers')]:
            expected=np.array([x[train][model[labels]==k].mean(axis=0) for k in range(4)])
            assert np.allclose(model[centers],expected)
        q=cp.nearest(x,model['centers']); qg=cp.nearest(x,model['graph_centers'])
        assert np.array_equal(q,model['projection']) and np.array_equal(qg,model['graph_projection'])
        controls=np.column_stack([level,pop,cp.status_design(status,tf.statuses)])
        quartile=np.searchsorted(np.quantile(level[train],[.25,.5,.75]),level,side='right')
        designs=cp.predictive_designs(controls,coords,wage23,q,qg,quartile)
        supervised=train&outcome; eligible=test&outcome
        growth=float(np.mean(wage24[supervised]-wage23[supervised]))
        fd=stability['fold_diagnostics'][fold]
        assert fd['train_ids']==ids[train].tolist() and fd['test_ids']==ids[test].tolist()
        assert fd['supervised_train_ids']==ids[supervised].tolist()
        close(fd['attribute_support'],cp.support(model['train_labels'],regions[train]))
        close(fd['graph_label_support'],cp.support(model['train_graph_labels'],regions[train]))
        close(fd['graph_projection_support'],cp.support(qg[train],regions[train]))
        for m in cp.METHODS:
            pred=wage23[eligible] if m=='PERSIST' else wage23[eligible]+growth if m=='GROWTH' else cp.ridge_predict(designs[m][supervised],wage24[supervised],designs[m][eligible],cp.continuous_columns(m,controls.shape[1]))
            for i,v in zip(np.where(eligible)[0],pred):
                row=by_method[m][ids[i]]
                assert row['region']==regions[i] and int(row['fold'])==fold
                assert np.isclose(float(row['target']),wage24[i],atol=1e-12)
                assert np.isclose(float(row['log_wage_2023']),wage23[i],atol=1e-12)
                assert np.isclose(float(row['prediction']),v,atol=1e-10)
    external=cp.external_arithmetic(rows)
    for k,v in external.items():
        close(v,summary[k])
    model=np.load(output/'models.npz',allow_pickle=False)
    tf=check_transform(model,level,shares,status,labor,np.ones(len(ids),bool))
    with (output/'labels.csv').open(encoding='utf-8',newline='') as stream:
        labels=list(csv.DictReader(stream))
    assert [r['entity_id'] for r in labels]==ids.tolist()
    z=np.array([int(r['attribute_fitted_label']) for r in labels]); zg=np.array([int(r['graph_fitted_label']) for r in labels])
    assert np.array_equal(cp.nearest(model['attributes'],model['centers']),[int(r['projected_type']) for r in labels])
    qg=cp.nearest(model['attributes'],model['graph_centers'])
    assert np.array_equal(qg,[int(r['graph_projected_type']) for r in labels])
    unit23,rms,_=cp.temporal_vectors(tf,np.log(p.totals[idx,:12]),p.shares[idx,:12]*100,status,regions)
    unit24,_,_=cp.temporal_vectors(tf,np.log(p.totals[idx,12:]),p.shares[idx,12:]*100,status,regions,rms=rms)
    assert np.allclose(unit23,model['temporal_unit2023']) and np.allclose(unit24,model['temporal_unit2024'])
    edges=cp.directed_temporal_graph(unit23,ids)
    graph=read(output/'graph-controls.json')
    calculated=cp.graph_controls(edges,model['attributes'],ids,regions,level,unit24)
    for k,v in calculated.items():
        close(v,graph[k])
    for seed in stability['seed_runs']:
        close(seed['ARI'],float(adjusted_rand_score(z,seed['labels'])))
        close({k:seed[k] for k in ['sizes','regions','admissible']},cp.support(np.array(seed['labels']),regions))
    assert [r['seed'] for r in stability['seed_runs']]==list(range(cp.SEED,cp.SEED+10))
    p10=float(np.quantile([r['ARI'] for r in stability['seed_runs']],.1))
    close(stability['seed_ari_p10'],p10)
    counts={r:int(np.sum(regions==r)) for r in set(regions)}
    largest=sorted(counts,key=lambda r:(-counts[r],r))[:10]
    assert reg['largest_regions']==largest and [r['region'] for r in stability['leave_region_out']]==largest
    for r in stability['leave_region_out']:
        keep=regions!=r['region']
        assert r['ids']==ids[keep].tolist()
        close(r['ARI'],float(adjusted_rand_score(z[keep],r['labels'])))
    min_ari=min(r['ARI'] for r in stability['leave_region_out'])
    close(stability['leave_region_min_ari'],min_ari)
    quarters=[]
    for start in range(0,24,3):
        cc=tf.coordinates(np.median(np.log(p.totals[idx,start:start+3]),axis=1),np.median(p.shares[idx,start:start+3]*100,axis=1),status,labor)
        quarters.append(cp.nearest(cp.block_attributes(cc,tf.active),model['centers']))
    assert np.array_equal(np.array(quarters).T,stability['quarter_labels'])
    assert stability['constant_projected_changes']==0
    admissible=bool(cp.support(z,regions)['admissible'] and all(r['attribute_support']['admissible'] for r in stability['fold_diagnostics']) and p10>=.7 and min_ari>=.7)
    graph_admissible=bool(cp.support(zg,regions)['admissible'] and cp.support(qg,regions)['admissible'] and all(r['graph_label_support']['admissible'] and r['graph_projection_support']['admissible'] for r in stability['fold_diagnostics']))
    assert summary['attribute_admissible']==stability['attribute_admissible']==admissible
    assert summary['graph_admissible']==stability['graph_admissible']==graph_admissible
    assert summary['type_promoted']==bool(admissible and external['type_external_increment'])
    assert summary['graph_temporal_increment']==graph['temporal_increment'] and summary['graph_control_status']==graph['control_status']
    assert summary['graph_economic_increment']==bool(graph_admissible and graph['temporal_increment'] and all(v['lower97_5']>.05 for v in external['graph_contrasts'].values()))
    a=cp.union_graph(edges,len(ids))
    close(summary['metrics']['attribute'],all_metrics(model['attributes'],z,a))
    close(summary['metrics']['graph'],all_metrics(model['attributes'],zg,a))
    atlas=read(output/'atlas.json'); close(atlas['summary'],summary)
    assert set(atlas['entities'])==set(p.ids)
    assert len(atlas['types'])==4
    for i,key in enumerate(p.ids):
        entity=atlas['entities'][key]
        if not data['labor_valid'][i]:
            assert entity['type'] is None and entity['projected_type'] is None and entity['missing_fields']
        else:
            j=int(np.where(ids==key)[0][0])
            assert entity['type']==int(z[j]) and entity['projected_type']==int(cp.nearest(model['attributes'][j:j+1],model['centers'])[0])
    assert summary['runtime_seconds_after_imports']<=3600
    return {'status':'PASS','entities':len(ids),'outcome_ids':int(outcome.sum()),'methods':13,
            'verified':'registration, roles, fold calibration and predictions, losses/bootstrap, graph controls/dyadic bounds, stability, all ICVI, atlas'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/conditional-v4')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(verify(args.output.resolve()),ensure_ascii=False))


if __name__=='__main__':
    main()
