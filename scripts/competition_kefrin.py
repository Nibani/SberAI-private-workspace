"""Euclidean KEFRiN equations with an explicit equal-scatter comparison wrapper.

Implemented independently from Shalileh and Mirkin (Entropy 2022, 24, 626),
Section 2.2, equations (5)--(7): https://doi.org/10.3390/e24050626.
The author's implementation is neither imported nor copied.

``fit_equations`` accepts already transformed Y and P, including signed P.
``fit`` and ``fit_kefrine`` use the same representation as the current
``sbercluster.attributed.kefrin_like`` on a canonical nonnegative graph.
This wrapper does not reproduce the paper's Section 4.1 preprocessing or
its published benchmarks. The objective is weighted squared error, exactly
the KMeans objective on [sqrt(rho)*Y, sqrt(xi)*P].
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
from scipy import sparse

ALGORITHM_REVISION = "euclidean_equations_5_7_max_sum_v1"


@dataclass
class PreparedData:
    features: np.ndarray
    rows: np.ndarray
    rho: float
    xi: float
    feature_origin: np.ndarray
    normalization: dict[str, Any]

    def augmented(self) -> np.ndarray:
        return np.hstack((np.sqrt(self.rho) * self.features,
                          np.sqrt(self.xi) * self.rows))


@dataclass
class FitResult:
    labels: np.ndarray
    objective: float
    feature_centers: np.ndarray
    network_centers: np.ndarray
    attribute_centers: np.ndarray
    diagnostics: dict[str, Any]


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _nonnegative(value: float, name: str) -> float:
    try:
        out = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative") from exc
    if not np.isfinite(out) or out < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return out


def _matrix(value: Any, name: str) -> np.ndarray:
    out = value.toarray() if sparse.issparse(value) else np.asarray(value)
    if np.iscomplexobj(out):
        raise ValueError(f"{name} must be real")
    out = np.array(out, dtype=np.float64, copy=True)
    if out.ndim != 2 or min(out.shape, default=0) < 1 or not np.isfinite(out).all():
        raise ValueError(f"{name} must be a nonempty finite 2D matrix")
    return out


def _squares(value: np.ndarray, name: str) -> float:
    with np.errstate(over="ignore", invalid="ignore"):
        out = float(np.einsum("ij,ij->", value, value))
    if not np.isfinite(out):
        raise ValueError(f"{name} must have finite squared norm")
    return out


def prepare_equal_scatter(X: np.ndarray, A: Any, graph_weight: float = 1.0) -> PreparedData:
    """Center X by columns and binary A by its grand mean, without dropping its diagonal.

    B[i,j] is one exactly when A[i,j] > 0. Sparse duplicates are summed and
    explicitly stored zeros are discarded before binarization. Negative raw
    weights are rejected; use ``fit_equations`` for a transformed signed P.
    At beta > 0 both feature TSS and grand-centered graph scatter must be
    positive. At beta = 0 a zero graph scatter is allowed and xi is zero.
    """
    x = _matrix(X, "X")
    beta = _nonnegative(graph_weight, "graph_weight")
    if sparse.issparse(A):
        if np.iscomplexobj(A.data):
            raise ValueError("A must be real")
        adjacency = sparse.csr_matrix(A, dtype=np.float64, copy=True)
        if not np.isfinite(adjacency.data).all() or np.any(adjacency.data < 0):
            raise ValueError("A must have finite nonnegative weights")
        adjacency.sum_duplicates()
        adjacency.eliminate_zeros()
        if not np.isfinite(adjacency.data).all():
            raise ValueError("Summed A weights must be finite")
        adjacency.data[:] = 1.0
        binary = adjacency.toarray()
    else:
        adjacency = _matrix(A, "A")
        if np.any(adjacency < 0):
            raise ValueError("A must have nonnegative weights")
        binary = (adjacency > 0).astype(np.float64)
    n = len(x)
    if binary.shape != (n, n):
        raise ValueError("A must have shape (len(X), len(X))")
    with np.errstate(over="ignore", invalid="ignore"):
        origin = x.mean(axis=0)
        features = x - origin
    if not np.isfinite(features).all():
        raise ValueError("Centered X must be finite")
    mean_graph = float(binary.mean())
    rows = binary - mean_graph
    feature_tss = _squares(features, "Centered X")
    graph_scatter = _squares(rows, "Grand-centered A")
    if beta > 0 and (feature_tss <= 0 or graph_scatter <= 0):
        raise ValueError("Positive graph_weight requires positive feature TSS and graph scatter")
    xi = beta * (feature_tss / graph_scatter) if beta > 0 else 0.0
    if not np.isfinite(xi):
        raise ValueError("Equal-scatter xi must be finite")
    return PreparedData(features, rows, 1.0, xi, origin, {
        "name": "binary_graph_grand_mean_equal_scatter",
        "feature_centering": "column_means",
        "graph_centering": "one_grand_mean_over_all_N_squared_entries_including_diagonal",
        "binarization": "positive_canonical_weights",
        "feature_origin": origin.tolist(),
        "graph_grand_mean": mean_graph,
        "feature_tss": feature_tss,
        "graph_grand_scatter": graph_scatter,
        "graph_column_centered_scatter": _squares(rows - rows.mean(axis=0), "Column-centered graph"),
        "graph_weight": beta,
        "rho": 1.0,
        "xi": xi,
        "positive_entries": int(np.count_nonzero(binary)),
        "diagonal_positive_entries": int(np.count_nonzero(np.diag(binary))),
        "binary_symmetric": bool(np.array_equal(binary, binary.T)),
    })


def _distances(Y: np.ndarray, P: np.ndarray, C: np.ndarray, L: np.ndarray,
               rho: float, xi: float) -> np.ndarray:
    # Direct residuals avoid cancellation in the norm expansion near a center.
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.zeros((len(Y), len(C)), dtype=np.float64)
        if rho > 0:
            delta = Y[:, None, :] - C[None, :, :]
            out += rho * np.einsum("ikv,ikv->ik", delta, delta)
        if xi > 0:
            delta = P[:, None, :] - L[None, :, :]
            out += xi * np.einsum("ikv,ikv->ik", delta, delta)
    if not np.isfinite(out).all():
        raise ValueError("Combined squared distances must be finite")
    return out


def _centers(Y: np.ndarray, P: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([Y[labels == c].mean(axis=0) for c in range(k)]),
            np.array([P[labels == c].mean(axis=0) for c in range(k)]))


def _labels(labels: np.ndarray, n: int, k: int) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.shape != (n,) or labels.dtype.kind not in "iu" or np.any(labels < 0) or np.any(labels >= k):
        raise ValueError("labels must be integers in 0..k-1, one for each row")
    if np.any(np.bincount(labels, minlength=k) == 0):
        raise ValueError("Every cluster must be nonempty")
    return labels.astype(np.int64, copy=True)


def objective_equations(Y: np.ndarray, P: np.ndarray, labels: np.ndarray,
                        rho: float = 1.0, xi: float = 1.0) -> dict[str, Any]:
    """Evaluate equation (5) at the means from (7), with no preprocessing."""
    y, p = _matrix(Y, "Y"), _matrix(P, "P")
    if p.shape != (len(y), len(y)):
        raise ValueError("P must have shape (len(Y), len(Y))")
    raw = np.asarray(labels)
    if raw.size == 0 or raw.dtype.kind not in "iu":
        raise ValueError("labels must be nonempty integers")
    k = int(raw.max()) + 1
    z = _labels(raw, len(y), k)
    rho, xi = _nonnegative(rho, "rho"), _nonnegative(xi, "xi")
    c, l = _centers(y, p, z, k)
    sse_y, sse_p = _squares(y - c[z], "Feature residual"), _squares(p - l[z], "Graph residual")
    value = rho * sse_y + xi * sse_p
    if not np.isfinite(value):
        raise ValueError("Objective must be finite")
    return {"objective": float(value), "SSE_Y": sse_y, "SSE_P": sse_p,
            "weighted_SSE_Y": rho * sse_y, "weighted_SSE_P": xi * sse_p,
            "feature_centers": c.tolist(), "network_centers": l.tolist()}


def seed_indices(Y: np.ndarray, P: np.ndarray, k: int, first_index: int,
                 rho: float = 1.0, xi: float = 1.0) -> np.ndarray:
    """Article seeding: maximize the SUM of distances to all existing seeds."""
    y, p = _matrix(Y, "Y"), _matrix(P, "P")
    k, first_index = _integer(k, "k"), _integer(first_index, "first_index", 0)
    if p.shape != (len(y), len(y)) or k > len(y) or first_index >= len(y):
        raise ValueError("Seeding requires matching square P, k <= N and a valid first index")
    rho, xi = _nonnegative(rho, "rho"), _nonnegative(xi, "xi")
    chosen = [first_index]
    scores = np.zeros(len(y))
    while len(chosen) < k:
        last = chosen[-1]
        with np.errstate(over="ignore", invalid="ignore"):
            scores += _distances(y, p, y[last:last + 1], p[last:last + 1], rho, xi)[:, 0]
        if not np.isfinite(scores).all():
            raise ValueError("Sum of seeding distances must be finite")
        remaining_scores = scores.copy()
        remaining_scores[chosen] = -np.inf
        chosen.append(int(np.argmax(remaining_scores)))
    return np.array(chosen, dtype=np.int64)


def _assign(distances: np.ndarray, previous: np.ndarray | None) -> np.ndarray:
    out = np.argmin(distances, axis=1)
    if previous is not None:
        # Keeping an exactly tied previous cluster avoids endless empty-cluster
        # repairs for duplicate rows while still obeying the minimum-distance rule.
        tied = distances[np.arange(len(out)), previous] == distances[np.arange(len(out)), out]
        out[tied] = previous[tied]
    return out


def _repair(labels: np.ndarray, distances: np.ndarray, k: int) -> list[dict[str, int]]:
    counts = np.bincount(labels, minlength=k)
    records = []
    errors = distances[np.arange(len(labels)), labels]
    for empty in np.flatnonzero(counts == 0):
        candidates = np.flatnonzero(counts[labels] > 1)
        point = int(candidates[np.argmax(errors[candidates])])
        old = int(labels[point])
        labels[point] = empty
        counts[old] -= 1
        counts[empty] += 1
        records.append({"point": point, "source": old, "destination": int(empty)})
    return records


def _run(Y: np.ndarray, P: np.ndarray, indices: np.ndarray,
         rho: float, xi: float, max_iter: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    k = len(indices)
    c, l = Y[indices].copy(), P[indices].copy()
    labels = None
    trace: list[dict[str, Any]] = []
    repairs_total = 0
    for iteration in range(1, max_iter + 1):
        distances = _distances(Y, P, c, l, rho, xi)
        assigned = _assign(distances, labels)
        repairs = _repair(assigned, distances, k)
        repairs_total += len(repairs)
        changes = len(Y) if labels is None else int(np.count_nonzero(assigned != labels))
        labels = assigned
        c, l = _centers(Y, P, labels, k)
        sse_y = _squares(Y - c[labels], "Feature residual")
        sse_p = _squares(P - l[labels], "Graph residual")
        value = rho * sse_y + xi * sse_p
        if not np.isfinite(value):
            raise ValueError("Objective must be finite")
        trace.append({"iteration": iteration, "objective": float(value),
                      "SSE_Y": sse_y, "SSE_P": sse_p,
                      "label_changes": changes, "empty_repairs": repairs})
        next_labels = _assign(_distances(Y, P, c, l, rho, xi), labels)
        if np.array_equal(next_labels, labels):
            break
    converged = bool(np.array_equal(next_labels, labels))
    monotone = all(b["objective"] <= a["objective"] + 1e-10 * max(1.0, abs(a["objective"]))
                   for a, b in zip(trace, trace[1:]))
    status = "labels_fixed_point" if converged else "iteration_cap_reached"
    info = {"objective": float(value), "SSE_Y": sse_y, "SSE_P": sse_p,
            "iterations": iteration, "converged": converged,
            "convergence_status": status, "cap_reached": not converged,
            "empty_repairs": repairs_total, "trace": trace,
            "objective_monotone": monotone,
            "initial_indices": indices.tolist(),
            "final_counts": np.bincount(labels, minlength=k).tolist()}
    return labels.copy(), c, l, info


def fit_equations(Y: np.ndarray, P: np.ndarray, k: int, seed: int,
                  n_init: int = 50, rho: float = 1.0, xi: float = 1.0,
                  max_iter: int = 300, initial_indices: Any = None) -> FitResult:
    """Alternate equations (6) and (7), retaining the smallest F across all starts.

    ``initial_indices`` is an explicit single-start diagnostic: n_init must be 1.
    Otherwise each restart uses its own local SeedSequence([seed, restart_id]);
    only the first seed is random. No global numpy RNG state is read or changed.
    Empty clusters are split from the greatest-error eligible observation,
    with index-order ties. The returned cap status refers to a final assignment
    check against the returned means, not merely a small objective change.
    """
    y, p = _matrix(Y, "Y"), _matrix(P, "P")
    k, seed = _integer(k, "k"), _integer(seed, "seed", 0)
    n_init, max_iter = _integer(n_init, "n_init"), _integer(max_iter, "max_iter")
    rho, xi = _nonnegative(rho, "rho"), _nonnegative(xi, "xi")
    if p.shape != (len(y), len(y)) or k > len(y):
        raise ValueError("Fit requires matching square P and k <= N")
    if rho == 0 and xi == 0:
        raise ValueError("At least one of rho and xi must be positive")
    _squares(y, "Y")
    _squares(p, "P")
    fixed = None
    if initial_indices is not None:
        fixed = np.asarray(initial_indices)
        if (n_init != 1 or fixed.shape != (k,) or fixed.dtype.kind not in "iu"
                or np.any(fixed < 0) or np.any(fixed >= len(y)) or len(np.unique(fixed)) != k):
            raise ValueError("initial_indices requires n_init=1 and k distinct valid integer indices")
        fixed = fixed.astype(np.int64)
    restarts = []
    best = None
    best_value = np.inf
    best_restart = -1
    for restart in range(n_init):
        if fixed is None:
            rng = np.random.default_rng(np.random.SeedSequence([seed, restart]))
            first = int(rng.integers(len(y)))
            indices = seed_indices(y, p, k, first, rho, xi)
        else:
            indices = fixed
        labels, c, l, info = _run(y, p, indices, rho, xi, max_iter)
        info["restart"] = restart
        restarts.append(info)
        if info["objective"] < best_value:
            best_value, best_restart = info["objective"], restart
            best = (labels, c, l)
    labels, c, l = best
    selected = restarts[best_restart]
    diagnostics = {
        "algorithm_revision": ALGORITHM_REVISION,
        "objective_definition": "rho*SSE_Y + xi*SSE_P; squared Euclidean",
        "preprocessing": "caller_supplied_Y_and_P",
        "rho": rho, "xi": xi, "seed": seed,
        "rng_rule": "local SeedSequence([seed,restart_id]); first point uniform",
        "seeding": "first_uniform_then_argmax_sum_to_all_previous_seeds",
        "assignment_ties": "retain_previous_exact_minimum_else_lowest_cluster_index",
        "seeding_ties": "lowest_remaining_observation_index",
        "empty_cluster_policy": "split_greatest_current_error_from_cluster_of_size_gt_one",
        "stop_rule": "labels_fixed_point_against_returned_means_or_iteration_cap",
        "max_iter": max_iter, "n_init_requested": n_init, "n_init_actual": len(restarts),
        "best_restart": best_restart, "iterations": selected["iterations"],
        "iterations_total": sum(r["iterations"] for r in restarts),
        "converged": selected["converged"], "convergence_status": selected["convergence_status"],
        "cap_reached": selected["cap_reached"],
        "capped_restarts": sum(r["cap_reached"] for r in restarts),
        "empty_repairs": selected["empty_repairs"],
        "empty_repairs_total": sum(r["empty_repairs"] for r in restarts),
        "objective": float(best_value), "SSE_Y": selected["SSE_Y"], "SSE_P": selected["SSE_P"],
        "weighted_SSE_Y": rho * selected["SSE_Y"], "weighted_SSE_P": xi * selected["SSE_P"],
        "objective_monotone_all_restarts": all(r["objective_monotone"] for r in restarts),
        "minimum_cluster_size": int(np.bincount(labels, minlength=k).min()),
        "restarts": restarts,
    }
    return FitResult(labels, float(best_value), c, l, c.copy(), diagnostics)


def fit(X: np.ndarray, A: Any, k: int, seed: int, n_init: int = 50,
        graph_weight: float = 1.0, max_iter: int = 300) -> FitResult:
    """Fit Euclidean equations using the declared equal-scatter comparison inputs."""
    data = prepare_equal_scatter(X, A, graph_weight)
    result = fit_equations(data.features, data.rows, k, seed, n_init,
                           rho=data.rho, xi=data.xi, max_iter=max_iter)
    result.attribute_centers = result.feature_centers + data.feature_origin
    result.diagnostics["preprocessing"] = data.normalization
    tss = data.normalization["feature_tss"]
    result.diagnostics["objective_normalized_by_feature_tss"] = result.objective / tss if tss > 0 else None
    result.diagnostics["degenerate_feature_tss"] = tss == 0
    return result


def fit_kefrine(x: np.ndarray, adjacency: Any, k: int = 4, beta: float = 1.0,
                seed: int = 1729, n_init: int = 50, max_iter: int = 300) -> tuple[np.ndarray, dict[str, Any]]:
    """Tuple interface for the synthetic driver; ``beta`` is graph_weight."""
    result = fit(x, adjacency, k, seed, n_init=n_init, graph_weight=beta, max_iter=max_iter)
    info = dict(result.diagnostics)
    info["attribute_centers"] = result.attribute_centers.tolist()
    info["feature_centers"] = result.feature_centers.tolist()
    info["network_centers"] = result.network_centers.tolist()
    return result.labels.copy(), info
