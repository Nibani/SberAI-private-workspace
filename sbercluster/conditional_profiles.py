"""Fixed labor-aware conditional prototype experiment. No wage enters codebooks."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import t
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.metrics import adjusted_rand_score

METHODS = 'PERSIST GROWTH C C4 CG FULL QUAD PFULL PQUAD PFULL4 PQUAD4 PQUADG CQ'.split()
SEED = 20261009
LABOR_FIELDS = ['covered_organization_jobs_per_resident',
                'public_administration_education_health_headcount_share']


def region_folds(regions):
    ordered = sorted(set(regions), key=lambda r: hashlib.sha256(
        ('conditional-v4-region|' + r).encode('utf-8')).hexdigest())
    return {r: i % 5 for i, r in enumerate(ordered)}


def standardize_fit(x):
    x = np.asarray(x, float)
    med = np.nanmedian(x, axis=0)
    if not np.isfinite(med).all():
        raise ValueError('All-missing training covariate')
    filled = np.where(np.isfinite(x), x, med)
    mean, scale = filled.mean(axis=0), filled.std(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    return med, mean, scale


def standardize_apply(x, fit, missing=False):
    med, mean, scale = fit
    x = np.asarray(x, float)
    finite = np.isfinite(x)
    z = (np.where(finite, x, med) - mean) / scale
    return np.column_stack([z, ~finite]) if missing else z


def status_design(status, known):
    status = np.asarray(status, str)
    return np.column_stack([*(status == s for s in known), ~np.isin(status, known)]).astype(float)


def continuous_columns(method, control_width):
    if method in ('C', 'C4', 'CG', 'CQ'):
        return [0, 1]
    if 'QUAD' in method:
        return list(range(54)) + ([control_width + 52] if method.startswith('P') else [])
    return [0, 1, *range(control_width, control_width + 7)] + ([control_width + 7] if method.startswith('P') else [])


def ridge_predict(train_x, train_y, test_x, continuous=None):
    continuous = list(range(train_x.shape[1])) if continuous is None else continuous
    categorical = [i for i in range(train_x.shape[1]) if i not in continuous]
    scaling = standardize_fit(train_x[:, continuous])
    # Indicators remain literal 0/1. Only continuous columns are standardized.
    a = np.column_stack([standardize_apply(train_x[:, continuous], scaling, missing=True), train_x[:, categorical]])
    b = np.column_stack([standardize_apply(test_x[:, continuous], scaling, missing=True), test_x[:, categorical]])
    return Ridge(alpha=10., fit_intercept=True).fit(a, train_y).predict(b)


@dataclass
class ConditionalTransform:
    statuses: list
    basis_scaling: tuple
    ridge: object
    center: np.ndarray
    scale: np.ndarray
    active: np.ndarray

    def residual(self, level, shares, status):
        level = np.asarray(level)
        shape = level.shape
        flat = level.ravel()
        s = np.repeat(status, shape[1]) if level.ndim == 2 else status
        basis = standardize_apply(np.column_stack([flat, flat ** 2]), self.basis_scaling)
        design = np.column_stack([basis, status_design(s, self.statuses)])
        return (np.asarray(shares).reshape(-1, 5) - self.ridge.predict(design)).reshape(*shape, 5)

    def coordinates(self, level, shares, status, labor):
        residual = self.residual(level, shares, status)
        raw = np.column_stack([residual, labor])
        coords = np.zeros_like(raw)
        coords[:, self.active] = (raw[:, self.active] - self.center[self.active]) / self.scale[self.active]
        return coords


def fit_transform(level, shares, status, labor):
    """Only declared 2023 inputs. There is intentionally no outcome argument."""
    statuses = sorted(set(str(s) for s in status if str(s)))
    scaling = standardize_fit(np.column_stack([level, level ** 2]))
    basis = standardize_apply(np.column_stack([level, level ** 2]), scaling)
    design = np.column_stack([basis, status_design(status, statuses)])
    ridge = Ridge(alpha=10., fit_intercept=True).fit(design, shares)
    raw = np.column_stack([shares - ridge.predict(design), labor])
    center = np.median(raw, axis=0)
    scale = np.quantile(raw, .75, axis=0) - np.quantile(raw, .25, axis=0)
    active = scale > 0
    return ConditionalTransform(statuses, scaling, ridge, center, scale, active)


def block_attributes(coords, active):
    pieces = []
    for block in (np.arange(5), np.arange(5, 7)):
        columns = block[active[block]]
        if len(columns):
            pieces.append(coords[:, columns] * np.sqrt(.5 / len(columns)))
    if not pieces:
        raise ValueError('No active clustering coordinates')
    return np.column_stack(pieces)


def nearest(x, centers):
    return np.argmin(np.sum((x[:, None] - centers[None]) ** 2, axis=2), axis=1)


def canonical_codebook(x, raw_labels, labor, residual):
    k = 4
    profiles = [np.median(np.column_stack([labor, residual])[raw_labels == c], axis=0) for c in range(k)]
    order = sorted(range(k), key=lambda c: (*profiles[c].tolist(), c))
    remap = np.argsort(order)
    labels = remap[raw_labels]
    centers = np.array([x[raw_labels == c].mean(axis=0) for c in order])
    return labels, centers, order


def fit_codebook(x, labor, residual, seed=SEED):
    raw = KMeans(4, n_init=20, random_state=seed, algorithm='lloyd').fit_predict(x)
    labels, centers, order = canonical_codebook(x, raw, labor, residual)
    return {'raw': raw, 'labels': labels, 'centers': centers, 'order': order,
            'q': nearest(x, centers)}


def support(labels, regions):
    sizes = [int(np.sum(labels == k)) for k in range(4)]
    counts = [len(set(regions[labels == k])) for k in range(4)]
    return {'sizes': sizes, 'regions': counts,
            'admissible': min(sizes) >= .05 * len(labels) and min(counts) >= 5}


def temporal_vectors(transform, levels, shares, status, regions, rms=None):
    residual = transform.residual(levels, shares, status)
    centered = residual - residual.mean(axis=1, keepdims=True)
    # Exact time constants are exact zero, avoiding normalization of roundoff.
    constant = np.ptp(residual, axis=1) == 0
    centered = np.where(constant[:, None, :], 0., centered)
    unique = sorted(set(regions))
    waves = {}
    for r in unique:
        reference = centered[regions == r] if np.sum(regions == r) >= 10 else centered
        waves[r] = reference.mean(axis=0)
    # mean_h(A_h,t - mean_t A_h,t) = regional monthly mean - regional grand mean.
    values = centered - np.array([waves[r] for r in regions])
    if rms is None:
        rms = np.sqrt(np.mean(values ** 2, axis=(0, 1)))
    active = rms > 0
    vectors = (values[:, :, active] / rms[active]).reshape(len(values), -1)
    norms = np.linalg.norm(vectors, axis=1)
    unit = np.divide(vectors, norms[:, None], out=np.zeros_like(vectors), where=norms[:, None] > 1e-12)
    return unit, rms, {'fallback_regions': [r for r in unique if np.sum(regions == r) < 10],
                       'zero_norm_nodes': int(np.sum(norms <= 1e-12)),
                       'omitted_channels': np.where(~active)[0].tolist()}


def directed_temporal_graph(unit, ids):
    cosine = unit @ unit.T
    edges = []
    for i in range(len(ids)):
        candidates = np.where((cosine[i] > 0) & (np.arange(len(ids)) != i))[0]
        chosen = sorted(candidates, key=lambda j: (-cosine[i, j], str(ids[j])))[:15]
        edges.extend((i, int(j)) for j in chosen)
    return np.asarray(edges, int).reshape(-1, 2)


def union_graph(edges, n):
    a = csr_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
    return a.maximum(a.T)


def static_graph(x, ids, degree):
    edges = []
    for i in range(len(ids)):
        distances = np.sum((x - x[i]) ** 2, axis=1)
        chosen = sorted((j for j in range(len(ids)) if j != i), key=lambda j: (distances[j], str(ids[j])))[:degree[i]]
        edges.extend((i, int(j)) for j in chosen)
    return np.asarray(edges, int).reshape(-1, 2)


def source_weights(edges, n):
    degree = np.bincount(edges[:, 0], minlength=n)
    sources = np.count_nonzero(degree)
    return 1. / (sources * degree[edges[:, 0]])


def weighted_edges(edges, weights):
    result = {}
    for (i, j), w in zip(edges, weights):
        edge = (int(min(i, j)), int(max(i, j)))
        result[edge] = result.get(edge, 0.) + float(w)
    return result


def dyadic_bound(g, h, score_matrix, regions):
    """Frozen dyadic sandwich, region endpoint dependence and t correction."""
    cg = sum(w * score_matrix[i, j] for (i, j), w in g.items())
    ch = sum(w * score_matrix[i, j] for (i, j), w in h.items()) if h else 0.
    touched = sorted(set(regions[i] for edge in (g.keys() | h.keys()) for i in edge))
    u = {r: 0. for r in touched}
    pairs = {}
    for i, j in g.keys() | h.keys():
        phi = g.get((i, j), 0.) * (score_matrix[i, j] - cg)
        if h:
            phi -= h.get((i, j), 0.) * (score_matrix[i, j] - ch)
        a, b = regions[i], regions[j]
        u[a] += phi
        if b != a:
            u[b] += phi
            pair = tuple(sorted((a, b)))
            pairs[pair] = pairs.get(pair, 0.) + phi
    r = len(touched)
    variance = r / (r - 1) * (sum(v*v for v in u.values()) - sum(v*v for v in pairs.values())) if r > 1 else None
    valid = r >= 20 and variance is not None and variance > 0
    return {'estimate': float(cg - ch), 'variance': variance, 'regions': r, 'valid': valid,
            'lower97_5': float(cg - ch - t.ppf(.975, r-1)*np.sqrt(variance)) if valid else None}


def graph_controls(edges, x, ids, regions, level, unit24, tick=lambda: None):
    n = len(ids)
    degree = np.bincount(edges[:, 0], minlength=n)
    static = static_graph(x, ids, degree)
    weights = source_weights(edges, n)
    scores = unit24 @ unit24.T
    g = weighted_edges(edges, weights)
    h = weighted_edges(static, source_weights(static, n))
    cg = float(sum(w*scores[i, j] for (i, j), w in g.items()))
    cs = float(sum(w*scores[i, j] for (i, j), w in h.items()))
    quartile = np.searchsorted(np.quantile(level, [.25, .5, .75]), level, side='right')
    strata = {}
    for i in range(n):
        strata.setdefault((str(regions[i]), int(quartile[i]), int(degree[i])), []).append(i)
    rng = np.random.default_rng(20261109)
    original = set(map(tuple, edges.tolist()))
    nulls, changed, digests, mean_h, permutations = [], [], [], {}, []
    for _ in range(199):
        tick()
        permutation = np.arange(n)
        for members in strata.values():
            permutation[members] = rng.permutation(members)
        e = permutation[edges]
        assert np.array_equal(np.bincount(e[:, 0], minlength=n), degree)
        assert np.all(e[:, 0] != e[:, 1])
        w = source_weights(e, n)
        nh = weighted_edges(e, w)
        nulls.append(float(sum(v*scores[i, j] for (i, j), v in nh.items())))
        changed.append(1-len(original & set(map(tuple, e.tolist())))/len(original))
        digests.append(hashlib.sha256(np.array(sorted(map(tuple, e.tolist())), dtype='<i8').tobytes()).hexdigest())
        permutations.append(permutation.tolist())
        for pair, v in nh.items():
            mean_h[pair] = mean_h.get(pair, 0.) + v / 199
    bounds = {'graph': dyadic_bound(g, {}, scores, regions),
              'graph_minus_static': dyadic_bound(g, h, scores, regions),
              'graph_minus_mean_null': dyadic_bound(g, mean_h, scores, regions)}
    validity = np.median(changed) >= .8 and len(set(digests)) >= 180
    valid_bounds = all(b['valid'] for b in bounds.values())
    increment = bool(validity and valid_bounds and bounds['graph']['lower97_5'] > 0
                     and bounds['graph_minus_static']['lower97_5'] > .05
                     and bounds['graph_minus_mean_null']['lower97_5'] > .05
                     and cg > np.quantile(nulls, .975))
    loro = []
    for r in sorted(set(regions)):
        tick()
        candidates = [edges, static, *(np.array(p, int)[edges] for p in permutations)]
        remaining = [e[(regions[e[:, 0]] != r) & (regions[e[:, 1]] != r)] for e in candidates]
        common = np.ones(n, bool)
        for e in remaining:
            common &= np.bincount(e[:, 0], minlength=n) > 0
        filtered = [e[common[e[:, 0]]] for e in remaining]
        sufficient = int(common.sum()) >= 20
        scores_removed = [float(np.sum(source_weights(e, n)*scores[e[:, 0], e[:, 1]])) if sufficient else None for e in filtered]
        row = {'region': str(r), 'common_sources': int(common.sum()), 'common_source_ids': ids[common].tolist(),
               'status': 'COMPLETE' if sufficient else 'UNDETERMINED',
               'graph': scores_removed[0], 'static': scores_removed[1],
               'null_scores': scores_removed[2:],
               'mean_null': float(np.mean(scores_removed[2:])) if sufficient else None,
               'removed_both_endpoints': True}
        loro.append(row)
    return {'status': 'COMPLETE', 'control_status': 'VALID' if validity and valid_bounds else 'UNDETERMINED',
            'temporal_increment': increment, 'graph_score': cg, 'static_score': cs,
            'null_scores': nulls, 'null_permutations': permutations, 'null_graph_digests': digests,
            'changed_edge_fractions': changed, 'unique_null_graphs': len(set(digests)),
            'singleton_strata': sum(len(v)==1 for v in strata.values()),
            'singleton_nodes': sum(len(v) for v in strata.values() if len(v)==1),
            'directed_edges': edges.tolist(), 'static_edges': static.tolist(),
            'sources': int(np.count_nonzero(degree)), 'nodes': n,
            'static_overlap': len(original & set(map(tuple, static.tolist())))/len(original),
            'bounds': bounds, 'leave_region_out': loro,
            'inference': 'Approximate dyadic dependence conditional on graphs; no exact randomization p-value.'}


def quadratic(x):
    return np.column_stack([x, *(x[:, i]*x[:, j] for i in range(x.shape[1]) for j in range(i, x.shape[1]))])


def predictive_designs(controls, continuous, wage23, q, qg, quartile):
    c, full = controls, np.column_stack([controls, continuous])
    # Interactions concern all continuous FULL columns; administrative dummies stay additive.
    # Caller puts the two continuous controls first, followed by administrative indicators.
    numeric = np.column_stack([controls[:, :2], continuous])
    quad = np.column_stack([quadratic(numeric), controls[:, 2:]])
    pfull, pquad = np.column_stack([full, wage23]), np.column_stack([quad, wage23])
    one = np.eye(4)
    return {'C': c, 'C4': np.column_stack([c, one[q]]), 'CG': np.column_stack([c, one[qg]]),
            'FULL': full, 'QUAD': quad, 'PFULL': pfull, 'PQUAD': pquad,
            'PFULL4': np.column_stack([pfull, one[q]]), 'PQUAD4': np.column_stack([pquad, one[q]]),
            'PQUADG': np.column_stack([pquad, one[qg]]), 'CQ': np.column_stack([c, one[quartile]])}


def external_arithmetic(rows):
    regions = sorted(set(r['region'] for r in rows))
    losses, vectors = {}, {}
    for method in METHODS:
        selected = [r for r in rows if r['method'] == method]
        errors = {r: [] for r in regions}
        for row in selected:
            errors[row['region']].append((float(row['target'])-float(row['prediction']))**2)
        vectors[method] = np.array([np.mean(errors[r]) for r in regions])
        losses[method] = {'region_mse': float(vectors[method].mean()),
                          'municipality_mse': float(np.mean([e for r in regions for e in errors[r]])),
                          'log_mae': float(np.mean([abs(float(r['target'])-float(r['prediction'])) for r in selected]))}
    draws = np.random.default_rng(20261209).integers(0, len(regions), (1999, len(regions)))
    def contrast(candidate, baseline):
        values = 1-vectors[candidate][draws].mean(axis=1)/vectors[baseline][draws].mean(axis=1)
        return {'estimate': float(1-vectors[candidate].mean()/vectors[baseline].mean()),
                'lower97_5': float(np.quantile(values, .025))}
    type_contrast = contrast('PQUAD4', 'PQUAD')
    graph_contrasts = {'versus_type': contrast('PQUADG', 'PQUAD4'), 'versus_continuous': contrast('PQUADG', 'PQUAD')}
    flag = type_contrast['lower97_5'] > .05 and all(losses['PQUAD4']['region_mse'] <= losses[m]['region_mse'] for m in ('PERSIST', 'GROWTH', 'PFULL'))
    return {'losses': losses, 'type_contrast': type_contrast, 'type_external_increment': bool(flag),
            'graph_contrasts': graph_contrasts,
            'compressed_contrasts': {'C4_vs_C': contrast('C4', 'C'), 'C4_vs_CQ': contrast('C4', 'CQ')},
            'region_losses': {m: dict(zip(regions, v.tolist())) for m, v in vectors.items()}}
