"""Small exact arithmetic checks, separate from the frozen scientific tests."""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"


def lower_priority():
    if sys.platform == "win32":
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        kernel.SetPriorityClass.restype = ctypes.c_int
        kernel.GetPriorityClass.argtypes = (ctypes.c_void_p,)
        kernel.GetPriorityClass.restype = ctypes.c_uint32
        handle = kernel.GetCurrentProcess()
        desired = 0x40 if os.environ.get("KEFRIN_CHECK_PRIORITY") == "Idle" else 0x4000
        if kernel.GetPriorityClass(handle) == 0x40:
            desired = 0x40
        if not kernel.SetPriorityClass(handle, desired):
            raise OSError("Could not lower process priority")
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        kernel.GetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                                  ctypes.POINTER(ctypes.c_size_t))
        kernel.SetProcessAffinityMask.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        if not kernel.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
            raise OSError("Could not read CPU affinity")
        one_cpu = process_mask.value & -process_mask.value
        if not kernel.SetProcessAffinityMask(handle, one_cpu):
            raise OSError("Could not set one CPU affinity")


def process_state():
    if sys.platform != "win32":
        return {"priority": "unchanged"}
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    handle = kernel.GetCurrentProcess()
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    kernel.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask))

    class MemoryCounters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                               "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = (ctypes.c_void_p, ctypes.POINTER(MemoryCounters), ctypes.c_uint32)
    if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
        raise OSError("Could not read own process working set")
    return {"priority": {0x40: "Idle", 0x4000: "BelowNormal"}.get(kernel.GetPriorityClass(handle), "other"),
            "affinity_mask": process_mask.value, "cpu_count": process_mask.value.bit_count(),
            "working_set_bytes": counters.WorkingSetSize, "peak_working_set_bytes": counters.PeakWorkingSetSize}


if __name__ == "__main__":
    lower_priority()

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_info, threadpool_limits

from scripts.competition_kefrin import (fit, fit_equations, fit_kefrine, objective_equations,
                                        prepare_equal_scatter, seed_indices)
from sbercluster.attributed import kefrin_like

X_EXACT = [[0], [1], [3], [4]]
A_EXACT = [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]]


def rational_center(matrix, indices):
    return [sum((F(matrix[i][j]) for i in indices), F(0)) / len(indices)
            for j in range(len(matrix[0]))]


def rational_sse(matrix, labels):
    centers = [rational_center(matrix, [i for i, z in enumerate(labels) if z == c])
               for c in range(max(labels) + 1)]
    score = sum(((F(value) - centers[labels[i]][j]) ** 2
                 for i, row in enumerate(matrix) for j, value in enumerate(row)), F(0))
    return score, centers


def rational_inputs():
    y_mean = rational_center(X_EXACT, list(range(4)))
    y = [[F(value) - y_mean[j] for j, value in enumerate(row)] for row in X_EXACT]
    graph_mean = sum((F(value) for row in A_EXACT for value in row), F(0)) / 16
    p = [[F(value) - graph_mean for value in row] for row in A_EXACT]
    feature_tss = sum((value ** 2 for row in y for value in row), F(0))
    graph_scatter = sum((value ** 2 for row in p for value in row), F(0))
    return y, p, feature_tss, graph_scatter


def asfloat(matrix):
    return np.array([[float(value) for value in row] for row in matrix])


def canonical(labels):
    translation = {}
    return tuple(translation.setdefault(int(label), len(translation)) for label in labels)


def rational_seed(y, p, k, first, rho, xi):
    chosen = [first]
    while len(chosen) < k:
        scores = {}
        for i in range(len(y)):
            if i not in chosen:
                scores[i] = sum((rho * sum(((a - b) ** 2 for a, b in zip(y[i], y[c])), F(0))
                                 + xi * sum(((a - b) ** 2 for a, b in zip(p[i], p[c])), F(0))
                                 for c in chosen), F(0))
        chosen.append(max(scores, key=lambda i: (scores[i], -i)))
    return chosen


def require_error(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError("Invalid input was accepted")


def verify():
    outcomes = []
    y, p, tss, scatter = rational_inputs()
    xi = tss / scatter
    data = prepare_equal_scatter(np.array(X_EXACT, float), csr_matrix(A_EXACT))
    np.testing.assert_allclose(data.features, asfloat(y), rtol=0, atol=0)
    np.testing.assert_allclose(data.rows, asfloat(p), rtol=0, atol=0)
    assert tss == 10 and scatter == F(15, 4) and xi == F(8, 3)
    assert data.normalization["graph_column_centered_scatter"] == 3.5
    assert data.normalization["graph_grand_scatter"] == 3.75
    augmented = data.augmented()
    exhaustive = []
    for tail in itertools.product((0, 1), repeat=3):
        z = (0,) + tail
        if max(z) == 0:
            continue
        sse_y, c = rational_sse(y, z)
        sse_p, l = rational_sse(p, z)
        expected = sse_y + xi * sse_p
        actual = objective_equations(data.features, data.rows, np.array(z), xi=float(xi))
        np.testing.assert_allclose(actual["feature_centers"], asfloat(c), rtol=0, atol=1e-14)
        np.testing.assert_allclose(actual["network_centers"], asfloat(l), rtol=0, atol=1e-14)
        np.testing.assert_allclose(actual["objective"], float(expected), rtol=1e-14, atol=1e-14)
        augmented_centers = np.array([augmented[np.array(z) == group].mean(axis=0) for group in range(2)])
        augmented_sse = float(np.square(augmented - augmented_centers[np.array(z)]).sum())
        np.testing.assert_allclose(augmented_sse, float(expected), rtol=1e-14, atol=1e-14)
        for i in range(4):
            distances = [sum(((y[i][j] - c[group][j]) ** 2 for j in range(1)), F(0))
                         + xi * sum(((p[i][j] - l[group][j]) ** 2 for j in range(4)), F(0))
                         for group in range(2)]
            aug_distances = np.square(augmented[i] - augmented_centers).sum(axis=1)
            np.testing.assert_allclose(aug_distances, [float(v) for v in distances], rtol=1e-14, atol=1e-14)
        exhaustive.append({"labels": list(z), "F_exact": str(expected),
                           "SSE_Y_exact": str(sse_y), "SSE_P_exact": str(sse_p)})
    assert len(exhaustive) == 7
    best_exact = min(F(r["F_exact"]) for r in exhaustive)
    assert best_exact == 9
    outcomes.append({"check": "all_7_bipartitions_rational_objectives_centers_assignments", "status": "PASS"})

    result = fit(np.array(X_EXACT, float), csr_matrix(A_EXACT), 2, 1729, n_init=50)
    np.testing.assert_allclose(result.objective, float(best_exact), rtol=1e-14, atol=1e-14)
    assert canonical(result.labels) == (0, 0, 1, 1)
    assert result.diagnostics["n_init_actual"] == 50 and len(result.diagnostics["restarts"]) == 50
    assert result.diagnostics["converged"] and result.diagnostics["objective_monotone_all_restarts"]
    np.testing.assert_allclose(result.diagnostics["objective_normalized_by_feature_tss"], .9)
    np.testing.assert_allclose(result.attribute_centers,
                               np.array([np.array(X_EXACT)[result.labels == c].mean(axis=0) for c in range(2)]))
    outcomes.append({"check": "50_starts_reach_tiny_enumerated_optimum_with_mean_centers", "status": "PASS"})

    seed_checks = []
    for first in range(4):
        expected = rational_seed(y, p, 3, first, F(1), xi)
        observed = seed_indices(data.features, data.rows, 3, first, xi=float(xi)).tolist()
        assert observed == expected
        seed_checks.append({"first": first, "indices": observed})
    triangle_y = [[F(0), F(0)], [F(10), F(0)], [F(6), F(8)], [F(0), F(1)]]
    zeros = [[F(0) for _ in range(4)] for _ in range(4)]
    observed = seed_indices(asfloat(triangle_y), asfloat(zeros), 3, 0, xi=0).tolist()
    assert observed == rational_seed(triangle_y, zeros, 3, 0, F(1), F(0)) == [0, 1, 2]
    assert np.argmax(np.square(asfloat(triangle_y)[[2, 3]] - asfloat(triangle_y)[1]).sum(axis=1)) == 1
    outcomes.append({"check": "max_sum_seeding_and_counterexample_to_last_center_rule", "status": "PASS"})

    shared_starts = []
    for first in range(4):
        indices = seed_indices(data.features, data.rows, 2, first, xi=data.xi)
        exact_fit = fit_equations(data.features, data.rows, 2, 0, n_init=1, xi=data.xi, initial_indices=indices)
        sklearn_fit = KMeans(2, init=augmented[indices], n_init=1, max_iter=300, tol=0,
                             algorithm="lloyd", random_state=0).fit(augmented)
        np.testing.assert_allclose(sklearn_fit.inertia_, exact_fit.objective, rtol=1e-14, atol=1e-14)
        assert adjusted_rand_score(sklearn_fit.labels_, exact_fit.labels) == 1
        np.testing.assert_allclose(sklearn_fit.cluster_centers_,
                                   np.hstack((exact_fit.feature_centers, np.sqrt(data.xi) * exact_fit.network_centers)),
                                   rtol=1e-14, atol=1e-14)
        shared_starts.append({"indices": indices.tolist(), "F": exact_fit.objective,
                              "ARI": 1.0, "sklearn_iterations": int(sklearn_fit.n_iter_),
                              "equation_iterations": exact_fit.diagnostics["iterations"]})
    outcomes.append({"check": "same_initial_centers_match_sklearn_lloyd_objective_labels_centers", "status": "PASS"})

    z20 = kefrin_like(np.array(X_EXACT, float), csr_matrix(A_EXACT), 2, seed=1729)
    sklearn50 = KMeans(2, n_init=50, max_iter=300, random_state=1729).fit(augmented)
    comparison = []
    for name, z, starts in (("current_like20", z20, 20), ("equal_scatter_like50", sklearn50.labels_, 50),
                            ("equations_max_sum50", result.labels, 50)):
        score = objective_equations(data.features, data.rows, z, xi=data.xi)
        comparison.append({"method": name, "n_init": starts, "labels": z.tolist(),
                           "F": score["objective"], "normalized_F": score["objective"] / 10,
                           "ARI_to_exact_tiny_partition": float(adjusted_rand_score([0, 0, 1, 1], z))})
    outcomes.append({"check": "current_like20_and_fair_like50_tiny_comparison", "status": "PASS"})

    signed_y = np.array([[0., 1.], [1., 0.], [4., 5.], [5., 4.]])
    signed_p = np.array([[1., -.5, -.25, -.25], [-.5, 1., -.25, -.25],
                         [-.25, -.25, 1., -.5], [-.25, -.25, -.5, 1.]])
    signed = fit_equations(signed_y, signed_p, 2, 91, n_init=3, rho=.7, xi=1.2)
    combined = np.hstack((np.sqrt(.7) * signed_y, np.sqrt(1.2) * signed_p))
    means = np.array([combined[signed.labels == c].mean(axis=0) for c in range(2)])
    np.testing.assert_allclose(signed.objective, np.square(combined - means[signed.labels]).sum(), rtol=1e-14)
    outcomes.append({"check": "signed_preprocessed_network_uses_squares_without_binarization", "status": "PASS"})

    shifted = fit(np.array(X_EXACT, float) + 100, csr_matrix(A_EXACT), 2, 1729, n_init=3)
    assert canonical(shifted.labels) == canonical(result.labels)
    np.testing.assert_allclose(shifted.objective, result.objective, rtol=1e-14)
    beta0 = fit(np.array(X_EXACT, float), csr_matrix((4, 4)), 2, 1729, n_init=3, graph_weight=0)
    assert beta0.diagnostics["xi"] == 0 and beta0.objective == 1
    require_error(lambda: fit(np.array(X_EXACT, float), csr_matrix((4, 4)), 2, 0))
    require_error(lambda: fit(np.zeros((4, 1)), csr_matrix(A_EXACT), 2, 0))
    sparse_zero = csr_matrix(([0., 2., 5.], ([0, 0, 1], [0, 1, 0])), shape=(4, 4))
    zeros_removed = prepare_equal_scatter(np.array(X_EXACT, float), sparse_zero)
    dense_support = prepare_equal_scatter(np.array(X_EXACT, float), sparse_zero.toarray())
    np.testing.assert_array_equal(zeros_removed.rows, dense_support.rows)
    assert zeros_removed.normalization["positive_entries"] == 2
    outcomes.append({"check": "translation_beta_zero_degenerate_scatter_and_sparse_stored_zeros", "status": "PASS"})

    duplicate = fit_equations(np.zeros((4, 1)), np.zeros((4, 4)), 3, 7, n_init=2)
    assert duplicate.objective == 0 and duplicate.diagnostics["minimum_cluster_size"] == 1
    assert duplicate.diagnostics["converged"] and duplicate.diagnostics["empty_repairs_total"] == 4
    outcomes.append({"check": "duplicate_rows_empty_repair_and_exact_minimum_ties", "status": "PASS"})

    capped = fit_equations(np.array([[0.], [1.], [2.], [8.], [9.], [10.]]), np.zeros((6, 6)),
                           2, 0, n_init=1, xi=0, max_iter=1, initial_indices=[0, 1])
    assert not capped.diagnostics["converged"] and capped.diagnostics["cap_reached"]
    assert capped.diagnostics["capped_restarts"] == 1
    assert capped.diagnostics["convergence_status"] == "iteration_cap_reached"
    np.testing.assert_allclose(capped.feature_centers,
                               np.array([np.array([[0.], [1.], [2.], [8.], [9.], [10.]])[capped.labels == c].mean(axis=0)
                                         for c in range(2)]))
    outcomes.append({"check": "iteration_cap_returns_means_and_explicit_unconverged_status", "status": "PASS"})

    np.random.seed(491)
    global_before = np.random.get_state()
    one = fit_kefrine(np.array(X_EXACT, float), csr_matrix(A_EXACT), k=2, seed=41, n_init=3)
    global_after = np.random.get_state()
    two = fit_kefrine(np.array(X_EXACT, float), csr_matrix(A_EXACT), k=2, seed=41, n_init=3)
    assert all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
               for a, b in zip(global_before, global_after))
    np.testing.assert_array_equal(one[0], two[0])
    assert one[1] == two[1] and one[1]["n_init_actual"] == 3
    json.dumps(one[1], allow_nan=False)
    outcomes.append({"check": "tuple_interface_json_diagnostics_local_rng_repeatability", "status": "PASS"})

    require_error(lambda: fit_equations([[np.nan]], [[0]], 1, 0))
    require_error(lambda: fit_equations([[1e200], [-1e200]], np.zeros((2, 2)), 2, 0))
    require_error(lambda: fit_equations([[1]], [[np.inf]], 1, 0))
    require_error(lambda: fit_equations([[1], [2]], [[0]], 1, 0))
    require_error(lambda: fit_equations([[1]], [[0]], 2, 0))
    require_error(lambda: fit_equations([[1]], [[0]], 1, -1))
    require_error(lambda: fit_equations([[1]], [[0]], 1, 0, n_init=0))
    require_error(lambda: fit_equations([[1]], [[0]], 1, 0, max_iter=0))
    require_error(lambda: fit_equations([[1]], [[0]], 1, 0, rho=0, xi=0))
    require_error(lambda: fit_equations([[1], [2]], np.zeros((2, 2)), 2, 0, n_init=1, initial_indices=[0, 0]))
    require_error(lambda: seed_indices(np.eye(4) * np.sqrt(3e307), np.zeros((4, 4)), 4, 0, xi=0))
    require_error(lambda: fit(np.array(X_EXACT), -csr_matrix(A_EXACT), 2, 0))
    require_error(lambda: fit(np.array(X_EXACT), csr_matrix(A_EXACT), 2, 0, graph_weight=np.inf))
    outcomes.append({"check": "invalid_inputs_and_overflow_rejected", "status": "PASS"})
    return {"status": "PASS", "checks": outcomes, "exhaustive_partitions": exhaustive,
            "exact_optimum": str(best_exact), "normalization": data.normalization,
            "seed_checks": seed_checks, "shared_start_checks": shared_starts,
            "comparison": comparison, "fit50_diagnostics": result.diagnostics,
            "signed_fit_diagnostics": signed.diagnostics, "cap_diagnostics": capped.diagnostics,
            "scope": "tiny mathematical verification only; no synthetic evaluation or real-data quality"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--monitor-handshake", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new result directory")
    args.output.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    lower_priority()
    begin = time.perf_counter()
    with threadpool_limits(limits=1):
        environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                       "packages": {name: importlib.metadata.version(name) for name in
                                    ("numpy", "scipy", "scikit-learn", "threadpoolctl")},
                       "threads": {name: os.environ[name] for name in
                                   ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                       "threadpool_info": threadpool_info(), "process_ready": process_state()}
        (args.output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
        if environment["process_ready"].get("working_set_bytes", 0) > 256 * 1024 ** 2:
            raise MemoryError("Import working set exceeds the authorized tiny-run limit of 256 MB")
        (args.output / "ready.json").write_text(json.dumps(environment["process_ready"]), encoding="utf-8")
        if args.monitor_handshake:
            deadline = time.monotonic() + 120
            while not (args.output / "permit.json").exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("Resource monitor did not permit the tiny run")
                time.sleep(.05)
        try:
            report = verify()
            after = process_state()
            environment["process_after"] = after
            baseline = environment["process_ready"].get("working_set_bytes", 0)
            if (after.get("working_set_bytes", 0) > 256 * 1024 ** 2
                    or after.get("working_set_bytes", 0) > baseline + 32 * 1024 ** 2):
                raise MemoryError("Working set exceeded the authorized tiny-run memory guard")
        except Exception as exc:
            (args.output / "failure.json").write_text(json.dumps({"status": "FAIL", "exception": repr(exc),
                                                                 "started_utc": started}, indent=2), encoding="utf-8")
            raise
    report["started_utc"] = started
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    report["wall_seconds"] = time.perf_counter() - begin
    report["command"] = [sys.executable, *sys.argv]
    report["source_sha256"] = {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in (ROOT / "scripts/competition_kefrin.py", Path(__file__),
                                            ROOT / "sbercluster/attributed.py")}
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (args.output / "checks.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": len(report["checks"]),
                      "wall_seconds": report["wall_seconds"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
