"""Compare spending level with saved region-held-out wage predictions."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sbercluster.panel import read_panel

ROOT = Path(__file__).resolve().parents[1]


def calculate():
    sources = ['data/v12/panel.csv.gz', 'reports/v1.2/strict-region-validation/strict_predictions.csv']
    panel = read_panel(ROOT / sources[0])
    frame = pd.read_csv(ROOT / sources[1])
    frame = frame.loc[frame.indicator == 'log_wage'].copy()
    # Each territory's feature uses its own 2023 observations only.
    level = np.median(np.log(panel.totals[:, :12]), axis=1)
    frame['level'] = frame.entity_id.map(dict(zip(map(str, panel.ids), level)))
    assert frame.entity_id.is_unique and np.isfinite(frame[['observed', 'level']]).all().all()
    y = frame.observed.to_numpy()
    regions = frame.region.to_numpy()
    x = np.column_stack([np.ones(len(frame)), frame.level.to_numpy()])
    pred = np.empty(len(frame))
    for region in np.unique(regions):
        test = regions == region
        beta = np.linalg.lstsq(x[~test], y[~test], rcond=None)[0]
        pred[test] = x[test] @ beta
    def r2(values):
        return float(1 - np.sum((y - values)**2) / np.sum((y - y.mean())**2))
    return {'n': len(frame), 'regions': len(np.unique(regions)),
            'level_r2': r2(pred), 'types_r2': r2(frame.type_prediction.to_numpy()),
            'admin_r2': r2(frame.admin_prediction.to_numpy()),
            'feature': 'Median of log monthly All categories expenses, January–December 2023',
            'outcome': 'Log organization wage, 2023; same observed cohort as strict type validation',
            'procedure': 'OLS with intercept; each region excluded from regression training; no cross-territory feature fitting',
            'scope': 'Retrospective comparison. Incremental value of types above spending level is not evaluated.',
            'sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources},
            'command': 'python -m scripts.check_wage_level'}


if __name__ == '__main__':
    result = calculate()
    output = ROOT / 'reports/v1.2/wage_level_comparison.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: result[k] for k in ('n', 'regions', 'level_r2', 'types_r2', 'admin_r2')}))
