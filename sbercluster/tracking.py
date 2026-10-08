"""Tracking types over time on the monthly attributed networks.

Two complementary views:

1. Frozen prototypes. Every month each municipality is assigned to the nearest
   2023 type center after the national monthly wave is removed. This yields a
   label sequence per municipality, monthly transition counts, a pooled Markov
   matrix with its stationary distribution, and *persistent* changes (a new
   type held for the last ``window`` months).
2. Free re-clustering. Each month is clustered again from scratch and matched
   to the 2023 types by the Hungarian assignment of maximal overlap, so no drift
   accumulates. Following MONIC (Spiliopoulou et al., 2006) a type *survives*
   when its match keeps at least ``survival`` of its members and the match is
   not much larger (Jaccard >= ``survival``); it is *absorbed* when most members
   stay together inside a much larger group; it *splits* when its members spread
   over two or more groups with at least a quarter each; otherwise it *disappears*.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans


def assign_monthly(monthly: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Nearest-center label for every municipality and month (N, T)."""
    d2 = np.square(monthly[:, :, None, :] - centers[None, None, :, :]).sum(axis=3)
    return np.argmin(d2, axis=2)


def transition_counts(before: np.ndarray, after: np.ndarray, k: int) -> np.ndarray:
    counts = np.zeros((k, k), dtype=int)
    np.add.at(counts, (before, after), 1)
    return counts


def markov(labels: np.ndarray, k: int) -> dict:
    """Pooled one-month transition matrix and its stationary distribution."""
    counts = sum(transition_counts(labels[:, t], labels[:, t + 1], k) for t in range(labels.shape[1] - 1))
    matrix = counts / counts.sum(axis=1, keepdims=True)
    values, vectors = np.linalg.eig(matrix.T)
    stationary = np.real(vectors[:, np.argmin(np.abs(values - 1))])
    stationary = stationary / stationary.sum()
    return {"counts": counts.tolist(), "matrix": matrix.tolist(), "stationary": stationary.tolist(),
            "expected_months_in_type": (1 / (1 - np.diag(matrix))).tolist()}


def persistence(labels: np.ndarray, base: np.ndarray, window: int = 6) -> dict:
    """How often municipalities keep their annual type, and persistent changes at the end."""
    switches = (np.diff(labels, axis=1) != 0).sum(axis=1)
    tail = labels[:, -window:]
    stable_tail = np.all(tail == tail[:, :1], axis=1)
    changed = stable_tail & (tail[:, 0] != base)
    return {"constant_share": float(np.mean(switches == 0)),
            "mean_switches": float(switches.mean()),
            "months_in_annual_type_share": float(np.mean(labels == base[:, None])),
            "persistent_change_window_months": window,
            "persistent_changes": int(changed.sum()),
            "persistent_change_mask": changed,
            "persistent_new_type": np.where(changed, tail[:, 0], -1)}


def jaccard_matrix(first: np.ndarray, second: np.ndarray, k1: int, k2: int) -> np.ndarray:
    inter = transition_counts(first, second, max(k1, k2))[:k1, :k2].astype(float)
    size1 = np.bincount(first, minlength=k1)[:, None]
    size2 = np.bincount(second, minlength=k2)[None, :]
    union = size1 + size2 - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def monic_events(base: np.ndarray, aligned: np.ndarray, k: int, survival: float) -> list[str]:
    """MONIC-style event of every base type relative to an aligned partition."""
    counts = transition_counts(base, aligned, k)
    sizes = np.maximum(counts.sum(axis=1), 1)
    jaccard = np.diag(jaccard_matrix(base, aligned, k, k))
    events = []
    for c in range(k):
        spread = counts[c] / sizes[c]
        if spread[c] >= survival and jaccard[c] >= survival:
            events.append("survival")
        elif spread.max() >= survival:
            events.append("absorption")
        elif np.sum(spread >= 0.25) >= 2:
            events.append("split")
        else:
            events.append("disappearance")
    return events


def align_to(base: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Relabel ``labels`` so that cluster c overlaps base type c as much as possible."""
    rows, cols = linear_sum_assignment(-transition_counts(base, labels, k))
    mapping = dict(zip(cols.tolist(), rows.tolist()))
    return np.array([mapping[c] for c in labels])


def free_reclustering(monthly: np.ndarray, base: np.ndarray, survival: float = 0.5,
                      seed: int = 1729, n_init: int = 20) -> dict:
    """Cluster every month independently, align to the base types and record events."""
    k = int(base.max() + 1)
    months, labels = [], []
    previous = None
    for t in range(monthly.shape[1]):
        raw = KMeans(k, n_init=n_init, random_state=seed).fit_predict(monthly[:, t])
        aligned = align_to(base, raw, k)
        counts = transition_counts(base, aligned, k)
        months.append({"month": t, "sizes": np.bincount(aligned, minlength=k).tolist(),
                       "jaccard_with_base": np.diag(jaccard_matrix(base, aligned, k, k)).tolist(),
                       "kept_share": (np.diag(counts) / np.maximum(counts.sum(axis=1), 1)).tolist(),
                       "jaccard_with_previous_month": None if previous is None else
                       np.diag(jaccard_matrix(previous, aligned, k, k)).tolist(),
                       "events": monic_events(base, aligned, k, survival)})
        labels.append(aligned)
        previous = aligned
    return {"months": months, "labels": np.stack(labels, axis=1)}


def summarize_events(result: dict, k: int, periods: tuple) -> list[dict]:
    """Per type: in how many months of each year it survives and its median Jaccard."""
    rows = []
    for c in range(k):
        for year, months in (("2023", range(0, 12)), ("2024", range(12, 24))):
            events = [result["months"][t]["events"][c] for t in months]
            jaccard = [result["months"][t]["jaccard_with_base"][c] for t in months]
            rows.append({"type": c, "year": year, "survival_months": events.count("survival"),
                         "absorption_months": events.count("absorption"), "split_months": events.count("split"),
                         "disappearance_months": events.count("disappearance"),
                         "median_jaccard_with_base": float(np.median(jaccard)),
                         "min_jaccard_with_base": float(np.min(jaccard))})
    return rows
