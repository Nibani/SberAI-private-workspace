"""Candidate edge rules for the municipal network and their common evaluation.

Every rule turns a pairwise closeness score into the same sparse structure: each
municipality keeps its k closest partners and the two directed lists are joined
(union symmetrization), so rules differ only in *what* closeness means:

=====================  ==========================================================
euclid_structure       distance between five-coordinate spending structures
euclid_profile         distance between six-coordinate profiles (structure+level)
cosine_spending        cosine similarity of ruble spending vectors by category
corr_total             Pearson correlation of relative spending-level series
corr_multivariate      mean correlation of six relative series
lagged_corr_total      max correlation over leads/lags up to ``max_lag`` months
dtw_total              dynamic time warping distance (Sakoe-Chiba band)
geographic             great-circle distance between municipality centroids
road                   SberIndex road distances (published k=15 graph)
=====================  ==========================================================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, triu
from scipy.spatial.distance import cdist


def knn_union(score: np.ndarray, k: int) -> csr_matrix:
    """Binary symmetric k-nearest graph from a dense closeness score (higher = closer).

    Ties are broken by index through a stable sort; the diagonal is excluded.
    """
    score = np.array(score, dtype=float, copy=True)
    n = len(score)
    if score.shape != (n, n) or not 0 < k < n:
        raise ValueError("Square score matrix and 0 < k < n required")
    np.fill_diagonal(score, -np.inf)
    score[np.isnan(score)] = -np.inf
    order = np.argsort(-score, axis=1, kind="stable")[:, :k]
    finite = np.isfinite(np.take_along_axis(score, order, axis=1))
    rows = np.repeat(np.arange(n), k).reshape(n, k)[finite]
    a = csr_matrix((np.ones(len(rows)), (rows, order[finite])), shape=(n, n))
    a = a.maximum(a.T)
    a.setdiag(0)
    a.eliminate_zeros()
    return a


def pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    """Exact distances (scipy cdist), so neighbour order matches ``graph.knn_graph``."""
    x = np.asarray(x, dtype=float)
    return cdist(x, x)


def cosine_similarity(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norm == 0):
        raise ValueError("Cosine similarity is undefined for a zero vector")
    unit = x / norm
    return unit @ unit.T


def _unit_rows(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(centered, axis=1, keepdims=True)
    return np.divide(centered, norm, out=np.zeros_like(centered), where=norm > 0)


def pearson_matrix(series: np.ndarray) -> np.ndarray:
    """Correlation of row time series; a constant series correlates 0 with everything."""
    unit = _unit_rows(np.asarray(series, dtype=float))
    return unit @ unit.T


def multivariate_correlation(series: np.ndarray) -> np.ndarray:
    """Mean Pearson correlation over attributes for (N, T, P) series."""
    series = np.asarray(series, dtype=float)
    blocks = [_unit_rows(series[:, :, p]) for p in range(series.shape[2])]
    stacked = np.concatenate(blocks, axis=1) / np.sqrt(series.shape[2])
    return stacked @ stacked.T


def lagged_correlation(series: np.ndarray, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Maximum correlation over leads and lags and the lag that attains it.

    ``lag[i, j] = l > 0`` means that series i leads series j by l months:
    corr(x_i(t), x_j(t + l)) is the largest value among |l| <= max_lag.
    """
    series = np.asarray(series, dtype=float)
    n, length = series.shape
    if not 0 <= max_lag < length - 2:
        raise ValueError("max_lag must leave at least three overlapping months")
    best = np.full((n, n), -np.inf)
    lag = np.zeros((n, n), dtype=int)
    for shift in range(max_lag + 1):
        corr = _unit_rows(series[:, :length - shift]) @ _unit_rows(series[:, shift:]).T
        for value, signed in ((corr, shift), (corr.T, -shift)):
            better = value > best
            best[better] = value[better]
            lag[better] = signed
    return best, lag


def dtw_distances(series: np.ndarray, band: int, block: int = 256) -> np.ndarray:
    """All-pairs DTW between z-normalized series with a Sakoe-Chiba band.

    Classic recursion C(p, q) = (a_p - b_q)^2 + min(C(p-1, q), C(p, q-1), C(p-1, q-1)),
    restricted to |p - q| <= band; the distance is sqrt(C(T-1, T-1)).
    """
    series = np.asarray(series, dtype=float)
    std = series.std(axis=1, keepdims=True)
    z = (series - series.mean(axis=1, keepdims=True)) / np.where(std > 0, std, 1)
    n, length = z.shape
    out = np.empty((n, n))
    for start in range(0, n, block):
        a = z[start:start + block]
        previous: dict[int, np.ndarray] = {}
        for p in range(length):
            current: dict[int, np.ndarray] = {}
            for q in range(max(0, p - band), min(length, p + band + 1)):
                cost = (a[:, p][:, None] - z[:, q][None, :]) ** 2
                candidates = [c for c in (current.get(q - 1), previous.get(q), previous.get(q - 1)) if c is not None]
                current[q] = cost if not candidates else cost + np.minimum.reduce(candidates)
            previous = current
        out[start:start + block] = np.sqrt(previous[length - 1])
    return out


def haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError("Every municipality needs coordinates")
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    h = np.sin(dlat / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(h, 0, 1)))


# ----------------------------------------------------------------- evaluation
def assortativity(adjacency: csr_matrix, values: np.ndarray) -> float | None:
    """Newman's scalar assortativity: Pearson correlation of values across edge ends."""
    a = triu(adjacency, k=1).tocoo()
    left = np.concatenate([values[a.row], values[a.col]])
    right = np.concatenate([values[a.col], values[a.row]])
    ok = np.isfinite(left) & np.isfinite(right)
    if ok.sum() < 10 or np.std(left[ok]) == 0:
        return None
    return float(np.corrcoef(left[ok], right[ok])[0, 1])


def within_group_residual(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Deviation from the group mean (region fixed effect removed); NaN stays NaN."""
    series = pd.Series(np.asarray(values, dtype=float))
    return (series - series.groupby(np.asarray(groups)).transform("mean")).to_numpy()


def same_group_share(adjacency: csr_matrix, groups: np.ndarray) -> float:
    a = triu(adjacency, k=1).tocoo()
    return float(np.mean(groups[a.row] == groups[a.col]))


def edge_jaccard(first: csr_matrix, second: csr_matrix) -> float:
    a = triu(first, k=1).tocsr()
    b = triu(second, k=1).tocsr()
    a.data[:] = 1
    b.data[:] = 1
    intersection = a.multiply(b).nnz
    return float(intersection / (a.nnz + b.nnz - intersection))


def evaluate_graph(adjacency: csr_matrix, external: pd.DataFrame, regions: np.ndarray,
                   reference: csr_matrix | None = None, later: csr_matrix | None = None) -> dict:
    """Edge homophily on independent data, regional concentration and stability."""
    row = {"edges": int(triu(adjacency, k=1).nnz),
           "same_region_share": same_group_share(adjacency, regions),
           "stability_jaccard": None if later is None else edge_jaccard(adjacency, later),
           "overlap_with_reference": None if reference is None else edge_jaccard(adjacency, reference)}
    raw, within = [], []
    for column in external.columns:
        values = external[column].to_numpy(dtype=float)
        r_raw = assortativity(adjacency, values)
        r_within = assortativity(adjacency, within_group_residual(values, regions))
        row[f"assort_{column}"] = r_raw
        row[f"assort_within_region_{column}"] = r_within
        raw.append(r_raw)
        within.append(r_within)
    row["assort_mean"] = float(np.mean([v for v in raw if v is not None]))
    row["assort_within_region_mean"] = float(np.mean([v for v in within if v is not None]))
    return row
