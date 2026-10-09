"""Execute the single registered conditional-v4 experiment, retaining every outcome."""
from __future__ import annotations
import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback
import platform

import numpy as np
import scipy
import sklearn
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

from sbercluster import conditional_profiles as cp
from sbercluster.joint import fit_graph_regularized_kmeans
from sbercluster.metrics import all_metrics
from sbercluster.panel import read_panel

ROOT = Path(__file__).resolve().parents[1]
CODE = ['sbercluster/conditional_profiles.py', 'scripts/run_conditional_profiles.py',
        'scripts/verify_conditional_profiles.py', 'sbercluster/panel.py',
        'sbercluster/joint.py', 'sbercluster/metrics.py']
WHITELIST = ['entity_id', 'region_name', 'municipal_district_type', 'municipal_district_status',
             'population_total', *cp.LABOR_FIELDS, 'wage_total',
             'year_from', 'year_to', 'change_id_from', 'change_id_to',
             'source_boundary_history_flag', 'registry_changed_2023_to_year']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    def native_scalar(v):
        if isinstance(v,np.generic):
            return v.item()
        raise TypeError(f'Unsupported JSON type {type(v).__name__}')
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=native_scalar) + '\n', encoding='utf-8')


def csv_write(path, rows, fields=None):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def economic(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        result = {}
        for row in csv.DictReader(stream):
            key = row['entity_id']
            if key in result:
                raise ValueError('Duplicate economic ID')
            result[key] = {c: row.get(c, '') for c in WHITELIST}
    return result


def number(row, field):
    try:
        return float(row.get(field, ''))
    except (ValueError, TypeError):
        return np.nan


def peak_memory_bytes():
    if os.name != 'nt':
        return None
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),
                  *[(name,ctypes.c_size_t) for name in ['PeakWorkingSetSize','WorkingSetSize',
                  'QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage',
                  'QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]]
    counters=Counters(); counters.cb=ctypes.sizeof(counters)
    kernel=ctypes.windll.kernel32; kernel.GetCurrentProcess.restype=ctypes.c_void_p
    get=ctypes.windll.psapi.GetProcessMemoryInfo
    get.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong]
    return int(counters.PeakWorkingSetSize) if get(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb) else None


def load_inputs(config):
    paths = {k: ROOT / v for k, v in config['inputs'].items()}
    if sha(paths['cohort2024']) != config['cohort2024_sha256']:
        raise ValueError('2024 bytes differ from recovered input')
    panel = read_panel(paths['panel'])
    econ23, econ24 = economic(paths['cohort2023']), economic(paths['cohort2024'])
    with paths['bridge'].open(encoding='utf-8-sig', newline='') as stream:
        bridge_ids = [r['entity_id'] for r in csv.DictReader(stream)]
    if set(bridge_ids) != set(panel.ids) or len(set(bridge_ids)) != len(bridge_ids):
        raise ValueError('Bridge ID coverage mismatch')
    if set(panel.ids) != set(econ23) or set(panel.ids) != set(econ24):
        raise ValueError('Economic ID coverage mismatch')
    jobs = np.array([number(econ23[i], cp.LABOR_FIELDS[0]) for i in panel.ids])
    share = np.array([number(econ23[i], cp.LABOR_FIELDS[1]) for i in panel.ids])
    labor_valid = np.isfinite(jobs) & (jobs > 0) & np.isfinite(share)
    population = np.array([number(econ23[i], 'population_total') for i in panel.ids])
    wage23 = np.array([number(econ23[i], 'wage_total') for i in panel.ids])
    wage24 = np.array([number(econ24[i], 'wage_total') for i in panel.ids])
    outcome = labor_valid & np.isfinite(population) & (population > 0) & np.isfinite(wage23) & (wage23 > 0) & np.isfinite(wage24) & (wage24 > 0)
    status = np.array([econ23[i]['municipal_district_type'] for i in panel.ids])
    regions = np.array([econ23[i]['region_name'] for i in panel.ids])
    if not np.array_equal(regions, panel.regions):
        raise ValueError('2023 economic/panel region disagreement')
    labor = np.column_stack([np.log(np.where(jobs > 0, jobs, np.nan)), share])
    return dict(panel=panel, econ23=econ23, econ24=econ24, paths=paths, labor=labor,
                labor_valid=labor_valid, population=population, wage23=wage23, wage24=wage24,
                outcome=outcome, status=status, regions=regions, jobs=jobs, share=share)


def run(config_path, output, failure_receipt=None):
    started = time.perf_counter()  # module imports explicitly outside budget
    config = json.loads(config_path.read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=False)  # no replacement of any attempt
    if failure_receipt is not None:
        (output/'failure-receipt.json').write_bytes(failure_receipt.read_bytes())
    def tick(stage=''):
        if time.perf_counter() - started > config['runtime_seconds']:
            raise TimeoutError('3600 second post-import experiment deadline')
        if stage:
            line=datetime.now(timezone.utc).isoformat(timespec='seconds')+' '+stage
            print(line, flush=True)
            with (output/'execution.log').open('a',encoding='utf-8') as stream:
                stream.write(line+'\n')
    try:
        data = load_inputs(config)
        p = data['panel']
        model_idx = np.where(data['labor_valid'])[0]
        ids, regions = p.ids[model_idx], data['regions'][model_idx]
        status, labor = data['status'][model_idx], data['labor'][model_idx]
        monthly_level, monthly_share = np.log(p.totals[model_idx]), p.shares[model_idx]*100
        level, shares = np.median(monthly_level[:, :12], axis=1), np.median(monthly_share[:, :12], axis=1)
        folds = cp.region_folds(regions)
        outcome = data['outcome'][model_idx]
        wage23 = np.log(np.where(data['wage23'][model_idx]>0,data['wage23'][model_idx],np.nan))
        wage24 = np.log(np.where(data['wage24'][model_idx]>0,data['wage24'][model_idx],np.nan))
        population = np.log(np.where(data['population'][model_idx]>0,data['population'][model_idx],np.nan))
        counts = {r: int(np.sum(regions == r)) for r in sorted(set(regions))}
        largest = sorted(counts, key=lambda r: (-counts[r], r))[:10]
        protocol_path = ROOT / config['protocol_path']
        if sha(protocol_path) != config['protocol_sha256']:
            raise ValueError('Frozen protocol hash drift')
        registration = {'revision': config['revision'], 'protocol_sha256': sha(protocol_path),
            'config_path': str(config_path.relative_to(ROOT)).replace('\\', '/'), 'config_sha256': sha(config_path),
            'code_sha256': {v: sha(ROOT/v) for v in CODE},
            'inputs_sha256': {str(v.relative_to(ROOT)).replace('\\','/'): sha(v) for v in data['paths'].values()},
            'fold_regions': sorted(set(regions)), 'folds': folds,
            'model_ids': ids.tolist(), 'outcome_ids': ids[outcome].tolist(),
            'fold_model_ids': {str(f): {'train': ids[np.array([folds[r] != f for r in regions])].tolist(),
                                      'test': ids[np.array([folds[r] == f for r in regions])].tolist()} for f in range(5)},
            'largest_regions': largest, 'registered_before_first_fit': True,
            'input_roles': {'unsupervised': '2023 labor and spending only; wages and legacy labels excluded',
                            'supervised_train': 'wage24 training regions', 'endpoint': 'wage24 heldout regions'},
            'mechanical_choices': config['mechanical_choices'], 'runtime_budget_after_imports': 3600,
            'runtime_versions':{'python':platform.python_version(),'numpy':np.__version__,
                                'scipy':scipy.__version__,'sklearn':sklearn.__version__},
            'process_id':os.getpid(),'registered_utc':datetime.now(timezone.utc).isoformat()}
        dump(output/'registration.json', registration)
        (output/'registration.sha256').write_text(sha(output/'registration.json')+'\n', encoding='ascii')
        cohort = []
        for i, key in enumerate(p.ids):
            missing = [f for f, v in zip(cp.LABOR_FIELDS, [data['jobs'][i], data['share'][i]]) if not np.isfinite(v) or (f == cp.LABOR_FIELDS[0] and v <= 0)]
            changed = {f: [data['econ23'][key].get(f, ''), data['econ24'][key].get(f, '')] for f in ['region_name','municipal_district_type','year_from','year_to','change_id_from','change_id_to'] if data['econ23'][key].get(f, '') != data['econ24'][key].get(f, '')}
            cohort.append({'entity_id': key, 'region': data['regions'][i], 'model_eligible': bool(data['labor_valid'][i]),
                           'outcome_eligible': bool(data['outcome'][i]), 'missing_labor_fields': json.dumps(missing),
                           'missing_or_nonpositive_wage23': not(np.isfinite(data['wage23'][i]) and data['wage23'][i]>0),
                           'missing_or_nonpositive_wage24': not(np.isfinite(data['wage24'][i]) and data['wage24'][i]>0),
                           'missing_or_nonpositive_population23': not(np.isfinite(data['population'][i]) and data['population'][i]>0),
                           'registration_changes': json.dumps(changed, ensure_ascii=False)})
        csv_write(output/'cohort.csv', cohort)
        tick(f'Registration saved before fit: {len(ids)} model IDs / {len(set(regions))} regions; {int(outcome.sum())} outcome IDs.')
        predictions, fold_diagnostics = [], []
        for fold in range(5):
            tick(f'Outer fold {fold+1}/5')
            train = np.array([folds[r] != fold for r in regions])
            test = ~train
            tf = cp.fit_transform(level[train], shares[train], status[train], labor[train])
            coords = tf.coordinates(level, shares, status, labor)
            x = cp.block_attributes(coords, tf.active)
            cb = cp.fit_codebook(x[train], labor[train], tf.residual(level[train], shares[train], status[train]))
            q = cp.nearest(x, cb['centers'])
            unit, rms, graph_diag = cp.temporal_vectors(tf, monthly_level[train, :12], monthly_share[train, :12], status[train], regions[train])
            edges = cp.directed_temporal_graph(unit, ids[train])
            zraw, potts = fit_graph_regularized_kmeans(x[train], cp.union_graph(edges, int(train.sum())), cb['labels'], alpha=.5, seed=cp.SEED, max_sweeps=50)
            zg, centers_g, _ = cp.canonical_codebook(x[train], zraw, labor[train], tf.residual(level[train], shares[train], status[train]))
            qg = cp.nearest(x, centers_g)
            controls = np.column_stack([level, population, cp.status_design(status, tf.statuses)])
            quartile = np.searchsorted(np.quantile(level[train], [.25,.5,.75]), level, side='right')
            designs = cp.predictive_designs(controls, coords, wage23, q, qg, quartile)
            supervised = train & outcome
            eligible = test & outcome
            growth = float(np.mean(wage24[supervised]-wage23[supervised]))
            np.savez_compressed(output/f'fold-{fold}.npz', ids=ids, coordinates=coords, attributes=x,
                centers=cb['centers'], graph_centers=centers_g, train_labels=cb['labels'], train_graph_labels=zg,
                train_graph_raw_labels=zraw, train_edges=edges, active=tf.active, center=tf.center, scale=tf.scale,
                ridge_coef=tf.ridge.coef_, ridge_intercept=tf.ridge.intercept_, basis_median=tf.basis_scaling[0],
                basis_mean=tf.basis_scaling[1], basis_scale=tf.basis_scaling[2], statuses=np.array(tf.statuses),
                projection=q, graph_projection=qg, train=train, eligible=eligible, supervised=supervised)
            for method in cp.METHODS:
                if method == 'PERSIST':
                    predicted = wage23[eligible]
                elif method == 'GROWTH':
                    predicted = wage23[eligible]+growth
                else:
                    predicted = cp.ridge_predict(designs[method][supervised], wage24[supervised], designs[method][eligible], cp.continuous_columns(method, controls.shape[1]))
                for i, pred in zip(np.where(eligible)[0], predicted):
                    predictions.append({'entity_id': ids[i], 'region': regions[i], 'fold': fold, 'method': method,
                                        'target': float(wage24[i]), 'prediction': float(pred), 'log_wage_2023': float(wage23[i])})
            fold_diagnostics.append({'fold': fold, 'attribute_support': cp.support(cb['labels'], regions[train]),
                'graph_label_support': cp.support(zg, regions[train]), 'graph_projection_support': cp.support(qg[train], regions[train]),
                'graph_label_projection_disagreement': int(np.sum(zg != qg[train])),
                'omitted_coordinates': np.where(~tf.active)[0].tolist(), 'train_ids': ids[train].tolist(),
                'test_ids': ids[test].tolist(), 'supervised_train_ids': ids[supervised].tolist(),
                'growth_log_increment': growth, 'graph_transform': graph_diag, 'potts': potts})
            csv_write(output/'predictions.checkpoint.csv', predictions)
            dump(output/'folds.checkpoint.json', fold_diagnostics)
        csv_write(output/'predictions.csv', predictions)
        external = cp.external_arithmetic(predictions)
        tick('Five outer folds completed; fitting the fixed full descriptive model.')
        tf = cp.fit_transform(level, shares, status, labor)
        residual = tf.residual(level, shares, status)
        coords = tf.coordinates(level, shares, status, labor)
        x = cp.block_attributes(coords, tf.active)
        cb = cp.fit_codebook(x, labor, residual)
        unit23, rms, graph_diag23 = cp.temporal_vectors(tf, monthly_level[:, :12], monthly_share[:, :12], status, regions)
        unit24, _, graph_diag24 = cp.temporal_vectors(tf, monthly_level[:, 12:], monthly_share[:, 12:], status, regions, rms=rms)
        edges = cp.directed_temporal_graph(unit23, ids)
        a = cp.union_graph(edges, len(ids))
        rawg, potts = fit_graph_regularized_kmeans(x, a, cb['labels'], alpha=.5, seed=cp.SEED, max_sweeps=50)
        zg, centers_g, order_g = cp.canonical_codebook(x, rawg, labor, residual)
        qg = cp.nearest(x, centers_g)
        tick('Full model completed; running the 199 frozen null graphs and endpoint controls.')
        graph = cp.graph_controls(edges, x, ids, regions, level, unit24, tick=tick)
        graph.update({'transform2023': graph_diag23, 'transform2024': graph_diag24,
                      'node_order': ids.tolist(), 'temporal_unit2024': unit24.tolist(), 'level2023': level.tolist(),
                      'attributes2023': x.tolist(), 'regions': regions.tolist(), 'potts': potts,
                      'full_label_support': cp.support(zg, regions), 'full_projection_support': cp.support(qg, regions),
                      'label_projection_disagreement': int(np.sum(zg != qg))})
        dump(output/'graph-controls.json', graph)
        seed_runs = []
        for seed in range(cp.SEED, cp.SEED+10):
            tick(f'Fixed seed stability {seed-cp.SEED+1}/10')
            fit = cb if seed == cp.SEED else cp.fit_codebook(x, labor, residual, seed)
            seed_runs.append({'seed': seed, 'ARI': float(adjusted_rand_score(cb['labels'], fit['labels'])),
                              **cp.support(fit['labels'], regions), 'labels': fit['labels'].tolist()})
        leave_region = []
        for r in largest:
            tick(f'Leave-region stability: {r}')
            keep = regions != r
            tr = cp.fit_transform(level[keep], shares[keep], status[keep], labor[keep])
            cc = tr.coordinates(level[keep], shares[keep], status[keep], labor[keep])
            xx = cp.block_attributes(cc, tr.active)
            fit = cp.fit_codebook(xx, labor[keep], tr.residual(level[keep], shares[keep], status[keep]))
            leave_region.append({'region': r, 'n': int(keep.sum()),
                'ARI': float(adjusted_rand_score(cb['labels'][keep], fit['labels'])),
                'ids': ids[keep].tolist(), 'labels': fit['labels'].tolist(), **cp.support(fit['labels'], regions[keep])})
        quarter_labels = []
        for start in range(0,24,3):
            qlevel = np.median(monthly_level[:, start:start+3], axis=1)
            qshares = np.median(monthly_share[:, start:start+3], axis=1)
            qx = cp.block_attributes(tf.coordinates(qlevel, qshares, status, labor), tf.active)
            quarter_labels.append(cp.nearest(qx, cb['centers']))
        quarters = np.array(quarter_labels).T
        constant_q = cp.nearest(x, cb['centers'])
        constant_changes = int(np.sum(np.repeat(constant_q[:,None],24,axis=1) != constant_q[:,None]))
        full_support = cp.support(cb['labels'], regions)
        p10 = float(np.quantile([v['ARI'] for v in seed_runs], .1))
        min_ari = min(v['ARI'] for v in leave_region)
        attribute_admissible = bool(full_support['admissible'] and all(v['attribute_support']['admissible'] for v in fold_diagnostics) and p10 >= .70 and min_ari >= .70)
        graph_admissible = bool(graph['full_label_support']['admissible'] and graph['full_projection_support']['admissible'] and all(v['graph_label_support']['admissible'] and v['graph_projection_support']['admissible'] for v in fold_diagnostics))
        stability = {'status':'COMPLETE', 'seed_runs': seed_runs, 'seed_ari_p10': p10,
                     'leave_region_out': leave_region, 'leave_region_min_ari': min_ari,
                     'full_attribute_support': full_support, 'attribute_admissible': attribute_admissible,
                     'graph_admissible': graph_admissible, 'fold_diagnostics': fold_diagnostics,
                     'quarter_labels': quarters.tolist(), 'quarter_names':[f'{y}-Q{q}' for y in [2023,2024] for q in [1,2,3,4]],
                     'constant_projected_changes': constant_changes}
        dump(output/'stability.json', stability)
        tick('Computing the six existing ICVI definitions and named TurboMQ.')
        metrics = {'attribute': all_metrics(x, cb['labels'], a), 'graph': all_metrics(x, zg, a),
                   'dependence': 'Graph-dependent internal indices; MQ definition remains pending, NewmanQ and TurboMQ are named candidates.'}
        types = []
        colors = ['#6b927d','#b99066','#7a8fad','#a77d98']
        for k in range(4):
            mask = cb['labels'] == k
            profile = {'organization_jobs_per_resident': float(np.median(data['jobs'][model_idx][mask])),
                       'OPQ_headcount_share': float(np.median(labor[mask,1])),
                       'conditional_percentages': np.median(residual[mask],axis=0).tolist()}
            types.append({'id':k, 'name':f'Профиль {k+1}: рабочих мест {profile["organization_jobs_per_resident"]:.2f} на жителя, O/P/Q {profile["OPQ_headcount_share"]*100:.0f}%',
                          'color':colors[k], 'n':int(mask.sum()), 'regions':len(set(regions[mask])), 'profile':profile})
        csv_write(output/'types.csv', [{'id':t['id'],'name':t['name'],'n':t['n'],'regions':t['regions'],'profile':json.dumps(t['profile'])} for t in types])
        labels = [{'entity_id': key,'attribute_fitted_label':int(cb['labels'][i]),'attribute_original_label':int(cb['raw'][i]),
                   'projected_type':int(cb['q'][i]),'graph_fitted_label':int(zg[i]),'graph_original_label':int(rawg[i]),
                   'graph_projected_type':int(qg[i]), 'quarter_labels':json.dumps(quarters[i].tolist())} for i,key in enumerate(ids)]
        csv_write(output/'labels.csv',labels)
        np.savez_compressed(output/'models.npz', ids=ids, attributes=x, coordinates=coords, residual=residual,
            annual_level=level, annual_shares=shares, labor=labor, regions=regions, centers=cb['centers'],
            graph_centers=centers_g, active=tf.active, center=tf.center, scale=tf.scale,
            ridge_coef=tf.ridge.coef_, ridge_intercept=tf.ridge.intercept_, basis_median=tf.basis_scaling[0],
            basis_mean=tf.basis_scaling[1], basis_scale=tf.basis_scaling[2], statuses=np.array(tf.statuses),
            temporal_unit2023=unit23, temporal_unit2024=unit24, temporal_rms=rms)
        graph_economic = bool(graph_admissible and graph['temporal_increment'] and all(v['lower97_5'] > .05 for v in external['graph_contrasts'].values()))
        summary = {'revision':'conditional-v4', 'status':'COMPLETE', 'model_n':len(ids), 'model_regions':len(set(regions)),
            'outcome_n':int(outcome.sum()), 'outcome_regions':len(set(regions[outcome])), 'uncovered_n':int((~data['labor_valid']).sum()),
            'attribute_admissible':attribute_admissible,'descriptive_admissible':attribute_admissible,'graph_admissible':graph_admissible,
            'type_promoted':bool(attribute_admissible and external['type_external_increment']),
            'graph_economic_increment':graph_economic, 'graph_temporal_increment':graph['temporal_increment'],
            'graph_control_status':graph['control_status'], 'metrics':metrics, **external,
            'runtime_seconds_after_imports':time.perf_counter()-started,
            'process_measurement':{'pid':os.getpid(),'threadpool_limit':1,'peak_working_set_bytes':peak_memory_bytes()},
            'interpretation':'Retrospective geographical wage24 estimation; 2024 was previously inspected. Organization jobs are not all employment; O/P/Q is industry, not ownership. No economic event or future forecast claim.'}
        entities = {}
        positions = {key:i for i,key in enumerate(ids)}
        distances = np.sort(np.sum((x[:,None]-cb['centers'][None])**2,axis=2),axis=1)
        for i,key in enumerate(p.ids):
            if key in positions:
                j=positions[key]
                entities[key]={'type':int(cb['labels'][j]), 'projected_type':int(cb['q'][j]), 'missing_fields':[],
                    'labor_values':{'organization_jobs_per_resident':float(data['jobs'][i]), 'OPQ_headcount_share':float(data['share'][i])}, 'profile_residuals':residual[j].tolist(),
                    'margin':float(distances[j,1]-distances[j,0]), 'quarter_stability':float(np.mean(quarters[j]==cb['q'][j]))}
            else:
                entities[key]={'type':None,'projected_type':None,'missing_fields':json.loads(cohort[i]['missing_labor_fields']),
                    'labor_values':{'organization_jobs_per_resident':float(data['jobs'][i]) if np.isfinite(data['jobs'][i]) else None,
                                    'OPQ_headcount_share':float(data['share'][i]) if np.isfinite(data['share'][i]) else None},
                    'profile_residuals':None,'margin':None,'quarter_stability':None,
                    'spending_percentages2023':np.median(p.shares[i,:12]*100,axis=0).tolist()}
        neighbors = {key:[str(ids[j]) for source,j in edges if source==i][:15] for i,key in enumerate(ids)}
        atlas={'revision':'conditional-v4','summary':summary,'types':types,'entities':entities,
               'graph':{'summary':{k:graph[k] for k in ['control_status','temporal_increment','graph_score','static_score','sources','nodes']},
                        'neighbors':neighbors, 'definition':'Positive cosine synchronization of conditional monthly spending; no flows.'},
               'external_results':{k:summary[k] for k in ['losses','type_contrast','graph_contrasts','type_external_increment','graph_economic_increment']}}
        dump(output/'atlas.json',atlas)
        tick('All required operations completed; recording truthful promotion gates.')
        dump(output/'summary.json',summary)
        report = (f'# Условные трудовые и расходные профили\n\nОпыт завершён: {len(ids)} МО / {len(set(regions))} регионов; внешний исход {int(outcome.sum())} МО / {len(set(regions[outcome]))} регионов. Без полной трудовой информации: {summary["uncovered_n"]} МО.\n\n'
          f'Прибавка типов к PQUAD: {external["type_contrast"]["estimate"]:.6%}; нижняя граница 97.5%: {external["type_contrast"]["lower97_5"]:.6%}. Внешний gate: {external["type_external_increment"]}; описательная допустимость: {attribute_admissible}; продвижение: {summary["type_promoted"]}.\n\n'
          f'Граф: контроль {graph["control_status"]}, временная прибавка {graph["temporal_increment"]}, экономическая прибавка {graph_economic}. Минимальный ARI удаления региона {min_ari:.6f}; seed p10 {p10:.6f}.\n\n'
          'Это географическая ретроспективная оценка зарплаты 2024, которая уже изучалась ранее. Она не является прогнозом из 2023 или нетронутым подтверждением. Четыре типа — фиксированный бюджет описания. Отрицательные и неопределённые исходы сохранены. O/P/Q обозначает отрасли, а не собственника. Все сравнения и исходы — в summary.json, graph-controls.json, stability.json, predictions.csv.\n')
        (output/'report.md').write_text(report,encoding='utf-8')
        # Final receipt is written last; COMPLETE alone cannot fake a completed directory.
        dump(output/'results-manifest.json',{p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
        print(json.dumps({k:summary[k] for k in ['status','model_n','outcome_n','type_contrast','type_promoted','graph_control_status','graph_economic_increment','runtime_seconds_after_imports']},ensure_ascii=False),flush=True)
    except Exception as exc:
        dump(output/'summary.json',{'status':'INCOMPLETE','error':type(exc).__name__+': '+str(exc),
                                  'runtime_seconds_after_imports':time.perf_counter()-started})
        (output/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=ROOT/'configs/conditional_profiles.json')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/conditional-v4')
    parser.add_argument('--failure-receipt',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.config.resolve(),args.output.resolve(),args.failure_receipt)


if __name__=='__main__':
    main()
