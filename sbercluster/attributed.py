"""Clustering methods for the attributed municipal network and their comparison.

Three families are compared on the same six attributes X and the same weighted
graph W (the selected edge rule):

* attributes only: KMeans, Ward, Gaussian mixture;
* network only: spectral clustering and Leiden with a resolution searched for K;
* attributes + network: a KEFRiN-style K-means on features plus adjacency rows
  (Shalileh & Mirkin, 2022) and the Potts-regularized K-means of
  ``sbercluster.joint`` that minimizes  SSE/TSS + alpha * cut/mass.

Rankings over the six internal validity indices are aggregated with the Copeland
rule: method A beats B when it is better on more indices than it is worse.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, triu
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from .graph import knn_graph
from .joint import fit_graph_regularized_kmeans
from .metrics import all_metrics

ICVI = (("SW", 1), ("CH", 1), ("S_Dbw", -1), ("AVI", 1), ("AVU", -1), ("MQ", 1))


def binary(adjacency: csr_matrix) -> csr_matrix:
    b = csr_matrix(adjacency, dtype=float, copy=True)
    b.data[:] = 1.0
    return b


def leiden_fixed_k(weights: csr_matrix, k: int, seed: int = 1729, steps: int = 40) -> tuple[np.ndarray, dict]:
    """Leiden (RB configuration) with a bisected resolution that yields k communities if possible."""
    import igraph as ig
    import leidenalg as la
    upper = triu(weights, k=1).tocoo()
    graph = ig.Graph(n=weights.shape[0], edges=list(zip(upper.row.tolist(), upper.col.tolist())))
    edge_weights = upper.data.tolist()
    low, high, best = 1e-4, 2.0, None
    for _ in range(steps):
        resolution = (low + high) / 2
        partition = la.find_partition(graph, la.RBConfigurationVertexPartition, weights=edge_weights,
                                      resolution_parameter=resolution, seed=seed, n_iterations=-1)
        found = len(partition)
        if best is None or abs(found - k) < abs(best[0] - k):
            best = (found, np.array(partition.membership), resolution)
        if found == k:
            break
        low, high = (low, resolution) if found > k else (resolution, high)
    return best[1], {"communities": int(best[0]), "resolution": float(best[2])}


def kefrin_like(x: np.ndarray, adjacency: csr_matrix, k: int, seed: int = 1729) -> np.ndarray:
    """K-means on [centered features, w * centered adjacency rows] with equal total scatter."""
    features = x - x.mean(axis=0)
    rows = binary(adjacency).toarray()
    rows -= rows.mean()
    weight = np.sqrt(np.square(features).sum() / np.square(rows).sum())
    return KMeans(k, n_init=20, random_state=seed).fit_predict(np.hstack([features, weight * rows]))


def fit_method(method: str, x: np.ndarray, weights: csr_matrix, k: int, seed: int = 1729,
               kmeans_labels: np.ndarray | None = None) -> np.ndarray:
    """Return labels 0..K-1 of one named method; ``joint_<alpha>`` starts from KMeans."""
    if method == "kmeans":
        return KMeans(k, n_init=50, random_state=seed).fit_predict(x)
    if method == "ward":
        return AgglomerativeClustering(k, linkage="ward").fit_predict(x)
    if method == "gmm":
        return GaussianMixture(k, covariance_type="full", n_init=10, random_state=seed).fit(x).predict(x)
    if method == "spectral_graph":
        return SpectralClustering(k, affinity="precomputed", assign_labels="cluster_qr",
                                  random_state=seed).fit_predict(weights)
    if method == "leiden_graph":
        return leiden_fixed_k(weights, k, seed)[0]
    if method == "kefrin":
        return kefrin_like(x, weights, k, seed)
    if method.startswith("joint_"):
        start = kmeans_labels if kmeans_labels is not None else fit_method("kmeans", x, weights, k, seed)
        labels, _ = fit_graph_regularized_kmeans(x, weights, start, alpha=float(method.split("_", 1)[1]),
                                                 seed=seed, max_sweeps=200)
        return labels
    raise ValueError(f"Unknown method: {method}")


def centroids(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.array([x[labels == c].mean(axis=0) for c in np.unique(labels)])


def nearest(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    d2 = np.square(x[:, None, :] - centers[None, :, :]).sum(axis=2)
    return np.argmin(d2, axis=1)


def icvi(x: np.ndarray, labels: np.ndarray, weights: csr_matrix) -> dict:
    """Six ICVI with MQ = Newman modularity on the binary graph (v1.1 convention)."""
    m = all_metrics(x, labels, binary(weights))
    return {"min_size": m["min_cluster_size"], "SW": m["SW"], "CH": m["CH"], "S_Dbw": m["S_Dbw"],
            "AVI": m["AVI"], "AVU": m["AVU"], "MQ": m["NewmanQ"], "TurboMQ": m["TurboMQ"]}


def copeland(table: pd.DataFrame, criteria=ICVI) -> pd.Series:
    """Copeland score over criteria; a missing value loses every comparison it enters."""
    values = np.array([[np.nan if pd.isna(table.iloc[i][c]) else s * float(table.iloc[i][c])
                        for c, s in criteria] for i in range(len(table))])
    values = np.where(np.isnan(values), -np.inf, values)
    scores = []
    for i in range(len(values)):
        score = 0
        for j in range(len(values)):
            if i != j:
                wins = int(np.sum(values[i] > values[j]))
                losses = int(np.sum(values[i] < values[j]))
                score += (wins > losses) - (wins < losses)
        scores.append(score)
    return pd.Series(scores, index=table.index, name="copeland")


def _dummies(labels: np.ndarray) -> np.ndarray:
    _, codes = np.unique(labels, return_inverse=True)
    out = np.zeros((len(labels), codes.max() + 1))
    out[np.arange(len(labels)), codes] = 1
    return out


def _r2(y: np.ndarray, design: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    centered = y - y.mean()
    return float(1 - resid @ resid / (centered @ centered))


def external_validity(labels: np.ndarray, external: pd.DataFrame, regions: np.ndarray) -> dict:
    """Share of variance of independent indicators explained by the types.

    ``eta2`` is the plain R^2 of type dummies; ``partial_r2_within_region`` is the
    additional R^2 over region fixed effects, i.e. what the types explain inside
    regions, relative to the variance left after regions.
    """
    out = {}
    for column in external.columns:
        y = external[column].to_numpy(dtype=float)
        ok = np.isfinite(y)
        types, regs = _dummies(labels[ok]), _dummies(regions[ok])
        base = _r2(y[ok], regs)
        both = _r2(y[ok], np.hstack([regs, types[:, 1:]]))
        out[f"eta2_{column}"] = _r2(y[ok], types)
        # Undefined when regions already explain everything.
        out[f"partial_r2_within_region_{column}"] = (both - base) / (1 - base) if 1 - base > 1e-12 else np.nan
    out["eta2_mean"] = float(np.nanmean([v for k, v in out.items() if k.startswith("eta2_")]))
    out["partial_r2_within_region_mean"] = float(np.nanmean([v for k, v in out.items() if k.startswith("partial_")]))
    return out


def leave_one_region_out_r2(external: pd.DataFrame, designs: dict[str, np.ndarray],
                            regions: np.ndarray) -> pd.DataFrame:
    """Out-of-region R^2: every region is predicted by a model fitted on all other regions."""
    rows = []
    for column in external.columns:
        y = external[column].to_numpy(dtype=float)
        ok = np.isfinite(y)
        row = {"indicator": column, "n": int(ok.sum())}
        for name, design in designs.items():
            prediction = np.full(len(y), np.nan)
            for region in np.unique(regions[ok]):
                test, train = ok & (regions == region), ok & (regions != region)
                beta, *_ = np.linalg.lstsq(design[train], y[train], rcond=None)
                prediction[test] = design[test] @ beta
            resid = y[ok] - prediction[ok]
            centered = y[ok] - y[ok].mean()
            row[name] = float(1 - resid @ resid / (centered @ centered))
        rows.append(row)
    return pd.DataFrame(rows)


def stability(method: str, monthly: np.ndarray, labels: np.ndarray, k: int, graph_k: int,
              draws: int, seed: int, development: slice = slice(0, 12)) -> dict:
    """ARI of refits after (a) resampling development months and (b) 80% territory subsamples."""
    rng = np.random.default_rng(seed)
    months = np.arange(development.start, development.stop)
    x = np.median(monthly[:, development], axis=1)
    month_ari, territory_ari = [], []
    for draw in range(draws):
        chosen = rng.choice(months, len(months), replace=True)
        xb = np.median(monthly[:, chosen], axis=1)
        wb, _ = knn_graph(xb, k=graph_k)
        month_ari.append(adjusted_rand_score(labels, fit_method(method, xb, wb, k, seed=draw)))
        subset = np.sort(rng.choice(len(x), int(0.8 * len(x)), replace=False))
        ws, _ = knn_graph(x[subset], k=graph_k)
        sub_labels = fit_method(method, x[subset], ws, k, seed=draw)
        full = nearest(x, centroids(x[subset], sub_labels))
        territory_ari.append(adjusted_rand_score(labels, full))
    return {"month_bootstrap_ari_mean": float(np.mean(month_ari)),
            "month_bootstrap_ari_p10": float(np.percentile(month_ari, 10)),
            "territory_subsample_ari_mean": float(np.mean(territory_ari)),
            "territory_subsample_ari_p10": float(np.percentile(territory_ari, 10))}
