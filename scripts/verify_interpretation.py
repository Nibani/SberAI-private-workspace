"""Verify the first frozen interpretation run from delivered CSVs and hashes.

Uses only the standard library; never fits a model, opens 2025 outcomes or
repeats scientific execution or writes delivered artifacts. Results go to stdout.
Run: python -B -X utf8 -m scripts.verify_interpretation --check
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/competition-enhancement'

def read(relative):
    with (OUT / relative).open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        raise AssertionError(f'{label}: {actual} != {expected}')

def quantile(values, p):
    ordered = sorted(values)
    position = (len(ordered)-1)*p
    lower = int(position)
    fraction = position-lower
    return ordered[lower] if fraction == 0 else ordered[lower]*(1-fraction)+ordered[lower+1]*fraction

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify and print results without writing the historical verification artifact.')
    parser.parse_args()
    completed = json.loads((OUT/'interpretation/provenance/execute-completed.json').read_text('utf-8'))
    prepared = json.loads((OUT/'interpretation/provenance/prepared.json').read_text('utf-8'))
    assert completed['exit_code'] == 0
    assert completed['outcome_years'] == [2023,2024]
    verified = {}
    for name, expected in {**prepared['frozen_files_sha256'], **completed['output_files_sha256']}.items():
        actual = digest(ROOT/name)
        if actual != expected:
            raise AssertionError('Frozen file hash mismatch: '+name)
        verified[name] = actual
    for folder in ('mirkin','mobility','economic_cases','four_russias'):
        status = json.loads((OUT/folder/'status.json').read_text('utf-8'))
        for name, expected in status['files'].items():
            assert digest(ROOT/name) == expected, name
    values, global_values = defaultdict(list), defaultdict(list)
    input_rows = read('mirkin-input.csv')
    for row in input_rows:
        if row['value']:
            value = float(row['value'])
            if math.isfinite(value):
                values[(row['group'],row['feature'])].append(value)
                global_values[row['feature']].append(value)
    summaries = read('mirkin-summary.csv')
    assert len(input_rows) == 2016*13 and len(summaries) == 4*13
    for row in summaries:
        part, whole = values[(row['group'],row['feature'])], global_values[row['feature']]
        assert len(part) == int(row['n_group']) and len(whole) == int(row['n_global'])
        gm, overall = math.fsum(part)/len(part), math.fsum(whole)/len(whole)
        close(gm,float(row['mean_group']),'group mean')
        close(overall,float(row['mean_global']),'overall mean')
        close((gm-overall)/overall,float(row['relative_deviation']),'Mirkin deviation')
    coverage = read('mirkin/coverage.csv')
    for row in coverage:
        close(int(row['n_group'])/int(row['N_group']),float(row['coverage_group']),'group coverage')
        close(int(row['n_global'])/int(row['N_global']),float(row['coverage_global']),'global coverage')
    summary = json.loads((OUT/'economic_cases/analogue-test-summary.json').read_text('utf-8'))
    predictions = read('economic_cases/analogue-test-predictions.csv')
    regions = read('economic_cases/analogue-test-by-region.csv')
    split = read('economic_cases/analogue-split.csv')
    assert len(predictions) == 903 and len(regions) == 36
    assert sum(r['role']=='donor' for r in split) == 37
    assert sum(r['role']=='evaluation' for r in split) == 36
    donor_region_ids = {r['region_code'] for r in split if r['role']=='donor'}
    assert not donor_region_ids.intersection(r['region_code'] for r in predictions)
    by_region = defaultdict(list)
    for row in predictions:
        by_region[row['region_code']].append(row)
        observed = float(row['observed_growth_pct'])
        for method in ('continuous15','type15'):
            close(abs(observed-float(row[method+'_prediction_pct'])),float(row[method+'_AE_pp']),'individual AE')
    for row in regions:
        part = by_region[row['region_code']]
        assert len(part) == int(row['n_entities'])
        for method in ('continuous15','type15'):
            close(math.fsum(float(p[method+'_AE_pp']) for p in part)/len(part),float(row[method+'_MAE_pp']),'region MAE')
    mc = math.fsum(float(r['continuous15_MAE_pp']) for r in regions)/36
    mt = math.fsum(float(r['type15_MAE_pp']) for r in regions)/36
    close(mc,summary['continuous15_region_mean_MAE_pp'],'primary continuous MAE')
    close(mt,summary['type15_region_mean_MAE_pp'],'primary restricted MAE')
    close((mc-mt)/mc,summary['relative_MAE_improvement'],'primary effect')
    boot = read('economic_cases/analogue-test-bootstrap.csv')
    assert len(boot) == 9999
    valid = []
    for row in boot:
        if row['valid'] == 'True':
            c,t = float(row['continuous15_region_mean_MAE_pp']),float(row['type15_region_mean_MAE_pp'])
            effect = float(row['relative_MAE_improvement'])
            close((c-t)/c,effect,'bootstrap effect')
            valid.append(effect)
    assert len(valid) == summary['valid_bootstrap_draws'] == 9999
    close(quantile(valid,0.025),summary['relative_MAE_improvement_lower95'],'lower percentile')
    close(quantile(valid,0.975),summary['relative_MAE_improvement_upper95'],'upper percentile')
    assert summary['operational_criterion_lower95_above_5percent'] is False
    case_rows = read('economic_cases/selected-cases.csv')
    assert [r['entity_id'] for r in case_rows] == ['tid_196','tid_616','tid_18']
    assert all(r['persistent_change']=='False' for r in case_rows)
    mobility = json.loads((OUT/'mobility/status.json').read_text('utf-8'))
    assert mobility['status']=='UNAVAILABLE' and mobility['details']['approved_matches']==0
    identity = json.loads((OUT/'economic_cases/identity.json').read_text('utf-8'))
    assert identity['same_neighbor_function'] and identity['max_absolute_prediction_difference']<1e-10
    small = [(r['group'],r['feature']) for r in coverage if r['small_denominator']=='True']
    low_coverage = [(r['group'],r['feature']) for r in coverage if r['coverage_below_half']=='True']
    result = {'status':'PASS','scientific_execution_repeated':False,
        'verified_frozen_files':len(verified),'mirkin_input_rows':len(input_rows),'mirkin_summary_rows':len(summaries),
        'evaluation_entities':len(predictions),'evaluation_regions':len(regions),'paired_bootstrap_draws':len(boot),
        'small_global_denominator_flags':small,'group_feature_coverage_below_half':low_coverage,
        'analogue_primary_summary':summary,'mobility_status':mobility['status'],
        'execute_completed_sha256':digest(OUT/'interpretation/provenance/execute-completed.json'),
        'verification_source_sha256':digest(Path(__file__))}
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
