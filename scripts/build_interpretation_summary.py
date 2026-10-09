"""Build compact economic facts from the frozen first-run result, without refit."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/competition-enhancement'

def rows(name):
    with (OUT/name).open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))

def load(name):
    return json.loads((OUT/name).read_text('utf-8'))

def main():
    summaries = rows('mirkin/coverage.csv')
    intervals = {(r['group'],r['feature']):r for r in rows('mirkin/bootstrap-intervals.csv')}
    stability = {(r['group'],r['feature']):r for r in rows('mirkin/stability.csv') if float(r['threshold'])==0.3}
    chosen = {('0','total_rub'),('0','horeca_pct'),('0','wage_total'),('2','horeca_pct'),('3','marketplaces_pct'),('3','market_access'),('3','wage_total'),('3','mining_share')}
    feature_facts = []
    for r in summaries:
        key = r['group'],r['feature']
        if key not in chosen:
            continue
        interval, stable = intervals[key],stability[key]
        feature_facts.append({'group':int(r['group']),'feature':r['feature'],'year':int(r['year']),
            'n_group':int(r['n_group']),'N_group':int(r['N_group']),'n_global':int(r['n_global']),'N_global':int(r['N_global']),
            'mean_group':float(r['mean_group']),'mean_global':float(r['mean_global']),
            'relative_deviation':float(r['relative_deviation']),'coverage_group':float(r['coverage_group']),
            'regions_group':int(r['regions_group']),'small_global_denominator':r['small_denominator']=='True',
            'relative_deviation_lower95':float(interval['relative_deviation_lower95']),
            'relative_deviation_upper95':float(interval['relative_deviation_upper95']),
            'leave_one_region_out_minimum':float(stable['minimum_d']),
            'leave_one_region_out_maximum':float(stable['maximum_d']),
            'leave_one_region_out_sign_changes':int(stable['sign_changes']),
            'descriptive_only':True})
    values = rows('economic_cases/case-values.csv')
    neighbors = rows('economic_cases/case-neighbors.csv')
    cases = []
    for selected in rows('economic_cases/selected-cases.csv'):
        entity = selected['entity_id']
        cv = {(int(r['year']),r['feature']):(float(r['value']) if r['value'] else None) for r in values if r['case_entity_id']==entity}
        c = {'entity_id':entity,'name':selected['name'],'region_code':int(selected['region_code']),
            'group_2023':int(selected['group_2023']),'group_name':selected['group_name'],
            'group_2024_adjusted':int(selected['group_2024_adjusted']),
            'group_2024_absolute':int(selected['group_2024_absolute']),
            'persistent_change_flag':selected['persistent_change']=='True',
            'selection_rule':selected['selection_rule'],
            'spending2023_mean_monthly_rub':cv[2023,'total_rub'],
            'spending2024_mean_monthly_rub':cv[2024,'total_rub'],
            'nominal_spending_growth_percent':100*(cv[2024,'total_rub']/cv[2023,'total_rub']-1),
            'marketplace2023_percent':cv[2023,'marketplaces_pct'],
            'marketplace2024_percent':cv[2024,'marketplaces_pct'],
            'marketplace_change_percentage_points':cv[2024,'marketplaces_pct']-cv[2023,'marketplaces_pct'],
            'wage2023_organization_rub':cv[2023,'wage_total'],'population2023_municipal':cv[2023,'population_total'],
            'market_access2024_points':cv[2024,'market_access'],
            'missing_sector_features':[f for f in ('agriculture_share','manufacturing_share','mining_share','public_share') if cv[2023,f] is None]}
        comparisons = []
        for factor in (None,1.1,1.25,1.5,2.0):
            part = [r for r in neighbors if r['case_entity_id']==entity and ((factor is None and r['comparison']=='all_X23_15') or (factor is not None and r['expense_factor_limit'] and float(r['expense_factor_limit'])==factor))]
            first = min(part,key=lambda r:int(r['neighbor_rank'])) if part else None
            comparisons.append({'symmetric_expense_factor_limit':factor,'neighbors_returned':len(part),
                'same_region_neighbors':sum(r['same_region']=='True' for r in part),
                'same_group_neighbors':sum(r['same_group']=='True' for r in part),
                'first_neighbor_entity_id':first['neighbor_entity_id'] if first else None,
                'first_neighbor_name':first['neighbor_name'] if first else None,
                'first_neighbor_distance':float(first['distance_frozen_X23']) if first else None,
                'first_neighbor_expense_factor':float(first['symmetric_mean_expense_factor']) if first else None})
        c['comparison_sensitivity'] = comparisons
        cases.append(c)
    result = {'status':'COMPLETED_DESCRIPTIVE','source':'unchanged first frozen scientific execution',
        'mirkin_formula':'(group arithmetic mean - overall arithmetic mean)/overall arithmetic mean',
        'mirkin_display_threshold_30percent_is_not_p_value':True,
        'primary_analogue_experiment':load('economic_cases/analogue-test-summary.json'),
        'analogue_identity':load('economic_cases/identity.json'),
        'representative_feature_facts':feature_facts,'economic_cases':cases,
        'four_russias_primary':load('four_russias/status.json')['details']['primary_metrics'],
        'four_russias_is_municipal_proxy_not_truth':True,
        'mobility':load('mobility/status.json')['details'],
        'limitations':['No causal, monetary-benefit or future-validation claim.',
            'Global groups and scaler selected using historical 2023–2024 spending.',
            'Sector and external averages use different observed coverage; missing values were not imputed.',
            'Case comparisons use spending profiles and legal type; they do not establish industrial specialization or a policy effect.',
            'Persistent change is a frozen algorithm flag; same-profile consistency diagnostics are reported separately by the science owner.']}
    (OUT/'interpretation').mkdir(parents=True,exist_ok=True)
    (OUT/'interpretation/summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    delivered = json.loads((OUT/'interpretation/provenance/execute-completed.json').read_text('utf-8'))
    inventory = dict(delivered['output_files_sha256'])
    owned = list((OUT/'interpretation').rglob('*'))
    owned += [Path(__file__),ROOT/'scripts/verify_interpretation.py',ROOT/'scripts/competition_interpretation.py']
    for path in sorted(owned):
        if path.is_file() and path.name != 'artifact-sha256.json':
            inventory[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'first_scientific_run_exit_code':delivered['exit_code'],
        'first_scientific_run_source_sha256':delivered['source_sha256'],
        'manifest_excludes_itself':True,'files_sha256':inventory}
    (OUT/'interpretation/artifact-sha256.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'cases':[c['entity_id'] for c in cases],
        'mirkin_representative_facts':len(feature_facts),'artifact_hash_count':len(inventory)},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
