"""Attach current verified results and a deterministic presentation layout.

The PCA layout is a display of the existing six-dimensional profile. It neither
changes any scientific feature nor supplies an additional economic network.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAYOUT_SHA256 = '74b9bcdac3a98c112ce3ce66cc5ab4201814885513108fbe39f5e3c9d513ab24'


def calculate_story_layout(entities: list[dict]) -> dict:
    x = np.asarray([row['v12_features'] for row in entities], dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 6 or not np.isfinite(x).all():
        raise ValueError('The story requires the finite original six-coordinate profile')
    centered = x - x.mean(axis=0)
    values, vectors = np.linalg.eigh(centered.T @ centered)
    order = np.argsort(values)[::-1]
    vectors = vectors[:, order[:2]]
    for column in range(2):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        if vectors[pivot, column] < 0:
            vectors[:, column] *= -1
    xy = centered @ vectors
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    normalized = (xy-lo)/np.maximum(hi-lo,1e-12)
    variance = np.maximum(values[order],0)
    return {
        'method': 'PCA2 of existing six-dimensional profiles; display only',
        'scope': 'presentation only; edges remain nearest neighbours in six dimensions',
        'entity_ids': [row['id'] for row in entities],
        'coordinates': normalized.tolist(),
        'components': vectors.T.tolist(),
        'explained_variance_ratio': (variance[:2]/max(float(variance.sum()),1e-12)).tolist(),
        'input_sha256': hashlib.sha256(x.tobytes(order='C')).hexdigest(),
        'coordinate_bounds': [lo.tolist(),hi.tolist()],
    }


def story_layout(entities: list[dict]) -> dict:
    """Use the frozen display coordinates; BLAS final bits must not alter a web build."""
    raw = (ROOT / 'data/atlas-story-layout.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != LAYOUT_SHA256:
        raise ValueError('Frozen presentation layout has changed')
    layout = json.loads(raw)
    x = np.asarray([row['v12_features'] for row in entities], dtype='<f8')
    if (x.ndim != 2 or x.shape[1] != 6 or not np.isfinite(x).all()
            or layout['entity_ids'] != [row['id'] for row in entities]
            or layout['input_sha256'] != hashlib.sha256(x.tobytes(order='C')).hexdigest()):
        raise ValueError('Frozen presentation layout does not match the six-coordinate profiles')
    return layout


def attach(payload: dict) -> dict:
    """Enrich a fresh payload without changing legacy model fields or map paths."""
    payload.setdefault('contest',{})['story_layout'] = story_layout(payload['entities'])
    temporal = ROOT / 'reports/temporal-v4/real-correction.json'
    if temporal.is_file():
        from scripts.current_evidence_web import load_temporal
        real, summary = load_temporal(temporal)
        ids = {str(entity['id']) for entity in payload['entities']}
        if ids != set(real['entities']):
            raise ValueError('Current temporal results do not cover the exact atlas cohort')
        fields = ('graph_label', 'projected_baseline_label', 'projected_labels',
                  'projection_disagreement', 'distance_margin', 'boundary_crossed',
                  'persistent_projection_change', 'observation_status')
        compact = {}
        for entity_id, row in real['entities'].items():
            compact[entity_id] = {key: row[key] for key in fields}
            compact[entity_id]['cp'] = {
                channel: {key: value.get(key) for key in
                          ('status', 'boundaries', 'persistent_boundaries')}
                for channel, value in row['cp'].items()
            }
        payload['contest']['temporal_current'] = {
            'revision': real['revision'], 'periods': real['periods'],
            'summary': real['summary'], 'entities': compact,
            'synthetic_promotion_gate': summary['synthetic_promotion_gate'],
        }
    economic_path = ROOT / 'reports/conditional-v4/atlas.json'
    if economic_path.is_file():
        economics = json.loads(economic_path.read_text(encoding='utf-8'))
        summary = json.loads(economic_path.with_name('summary.json').read_text(encoding='utf-8'))
        ids = {str(entity['id']) for entity in payload['entities']}
        if summary.get('status') != 'COMPLETE' or ids != set(economics['entities']):
            raise ValueError('Current economic results are incomplete or cover a different cohort')
        payload['contest']['economics_current'] = economics
    return payload
