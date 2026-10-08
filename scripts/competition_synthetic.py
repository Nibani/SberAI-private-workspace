"""Preregistered synthetic tests of annual Potts clustering and frozen tracking.

The model sees features and adjacency only. Latent labels are generated separately
and are consumed by scorers, never by clustering or prototype assignment.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
import traceback

for _pool in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_pool] = "1"

ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA = "73d98737ef8dea1a5e40a77ab9a501a84c56005ced6e68827838e2399121d217"
DESIGN_SHA = "39132d0f10f00b74e4fc90c93343c5a2a503defcf8a09a8bba519cc730e6db27"
CORE = ("sbercluster/joint.py", "sbercluster/attributed.py", "sbercluster/panel.py",
        "sbercluster/tracking.py", "sbercluster/graph.py", "scripts/competition_kefrin.py")
M_ROWS = ((1, 1, 1, 0, 0, 1), (1, -1, -1, 0, 0, -1),
          (-1, 1, -1, 0, 0, -1), (-1, -1, 1, 0, 0, 1))
REGIMES = ("stable", "common_shift", "persistent_transition", "transition_and_shift")
STRESS = ("late_permanent_onset_34_censored", "broad_unbalanced_transition_0.6_onset_19")
EXTRA = ("weak_independent_permuted", "weak_derived_permuted")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not __import__("math").isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: clean(v) for k, v in row.items()})


def append_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    exists = path.exists()
    fields = list(rows[0])
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        if not exists:
            writer.writeheader()
        writer.writerows({key: clean(value) for key, value in row.items()} for row in rows)


def numeric():
    global np, csr_matrix, triu, connected_components, linear_sum_assignment
    global KMeans, SpectralClustering, adjusted_rand_score, threadpool_limits
    global P, T, A, knn_graph, fit_graph_regularized_kmeans, fit_kefrine
    import numpy as np
    from scipy.sparse import csr_matrix, triu
    from scipy.sparse.csgraph import connected_components
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import KMeans, SpectralClustering
    from sklearn.metrics import adjusted_rand_score
    from threadpoolctl import threadpool_limits
    from sbercluster import panel as P, tracking as T, attributed as A
    from sbercluster.graph import knn_graph
    from sbercluster.joint import fit_graph_regularized_kmeans
    from scripts.competition_kefrin import fit_kefrine


class DiskCounters:
    """Read physical disk idle percentages; bytes/s is not disk utilization."""
    def __init__(self):
        self.query = ctypes.c_void_p()
        self.counter = ctypes.c_void_p()
        self.error = None
        self.pdh = None
        try:
            if os.name != "nt":
                raise RuntimeError("Windows PDH physical disk counters unavailable")
            self.pdh = ctypes.WinDLL("pdh")
            self.pdh.PdhOpenQueryW.argtypes = (ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p))
            self.pdh.PdhAddEnglishCounterW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_void_p))
            self.pdh.PdhCollectQueryData.argtypes = (ctypes.c_void_p,)
            self.pdh.PdhGetFormattedCounterArrayW.argtypes = (ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p)
            self.pdh.PdhCloseQuery.argtypes = (ctypes.c_void_p,)
            if self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query)):
                raise RuntimeError("PdhOpenQueryW failed")
            if self.pdh.PdhAddEnglishCounterW(self.query, r"\PhysicalDisk(*)\% Idle Time", 0, ctypes.byref(self.counter)):
                raise RuntimeError("PhysicalDisk idle counter unavailable")
            self.pdh.PdhCollectQueryData(self.query)
        except Exception as exc:
            self.error = str(exc)

    def sample(self):
        if self.error:
            return {}
        class Value(ctypes.Structure):
            _fields_ = (("status", ctypes.c_uint32), ("value", ctypes.c_double))
        class Item(ctypes.Structure):
            _fields_ = (("name", ctypes.c_wchar_p), ("value", Value))
        self.pdh.PdhCollectQueryData(self.query)
        size, count = ctypes.c_uint32(), ctypes.c_uint32()
        status = self.pdh.PdhGetFormattedCounterArrayW(self.counter, 0x200, ctypes.byref(size), ctypes.byref(count), None)
        if status & 0xffffffff != 0x800007d2:
            return {}
        buf = ctypes.create_string_buffer(size.value)
        status = self.pdh.PdhGetFormattedCounterArrayW(self.counter, 0x200, ctypes.byref(size), ctypes.byref(count), buf)
        if status:
            return {}
        array = ctypes.cast(buf, ctypes.POINTER(Item))
        return {array[i].name: max(0., min(100., 100. - array[i].value.value))
                for i in range(count.value)
                if array[i].name != "_Total" and array[i].value.status in (0, 1)}

    def close(self):
        if self.pdh and self.query.value:
            self.pdh.PdhCloseQuery(self.query)


class ResourceMonitor:
    def __init__(self, output, heavy=False):
        import psutil
        self.psutil = psutil
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.blocked = threading.Event()
        self.io_busy = threading.Event()
        self.heavy = heavy
        self.samples = 0
        self.last_disk_count = 0
        self.disks = DiskCounters()
        self.gpu_command = shutil.which("nvidia-smi")
        self.errors = set()
        process = psutil.Process()
        self.process = process
        self.last_priority = None
        if os.name == "nt":
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            process.nice(10)
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(timeout=10):
            raise RuntimeError("Resource monitor did not produce a sample")
        self.gate()

    def loop(self):
        self.psutil.cpu_percent(interval=None)
        self.stop.wait(1)
        with self.output.open("a", encoding="utf-8") as log:
            while not self.stop.is_set():
                disk = self.disks.sample()
                self.last_disk_count = len(disk)
                gpu = {}
                if self.gpu_command:
                    try:
                        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                        raw = subprocess.check_output([self.gpu_command,
                            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], text=True, timeout=3,
                            creationflags=flags, stderr=subprocess.DEVNULL)
                        for line in raw.splitlines():
                            index, use, used, total = line.split(",")
                            gpu[index.strip()] = {"utilization": float(use), "memory_fraction": float(used) / max(float(total), 1)}
                    except Exception as exc:
                        self.errors.add("gpu:" + type(exc).__name__)
                cpu = self.psutil.cpu_percent(interval=None)
                memory = self.psutil.virtual_memory().percent
                disk_busy = any(v >= 95 for v in disk.values())
                compute_busy = cpu >= 95 or any(v["utilization"] >= 95 for v in gpu.values())
                busy = memory >= 95 or (self.heavy and compute_busy)
                if disk_busy:
                    self.io_busy.set()
                else:
                    self.io_busy.clear()
                priority = "Idle" if disk_busy or compute_busy else "BelowNormal"
                if priority != self.last_priority and os.name == "nt":
                    self.process.nice(self.psutil.IDLE_PRIORITY_CLASS if priority == "Idle" else self.psutil.BELOW_NORMAL_PRIORITY_CLASS)
                    try:
                        self.process.ionice(self.psutil.IOPRIO_VERYLOW if priority == "Idle" else self.psutil.IOPRIO_LOW)
                    except (AttributeError, OSError) as exc:
                        self.errors.add("io_priority:" + type(exc).__name__)
                    self.last_priority = priority
                if busy:
                    self.blocked.set()
                else:
                    self.blocked.clear()
                row = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                       "cpu_percent": cpu, "ram_percent": memory, "disk_percent": disk,
                       "gpu": gpu, "compute_paused": busy, "disk_backoff": disk_busy,
                       "priority_requested": priority, "priority_actual": self.process.nice(),
                       "heavy_compute": self.heavy, "rss": self.process.memory_info().rss,
                       "disk_error": self.disks.error, "gpu_available": bool(self.gpu_command)}
                log.write(json.dumps(row) + "\n")
                log.flush()
                self.samples += 1
                self.ready.set()
                self.stop.wait(5)

    def gate(self):
        while self.blocked.is_set():
            print("resource pause: RAM or active compute pressure >=95%", flush=True)
            self.stop.wait(10)
        if self.io_busy.is_set():
            self.stop.wait(1)
        if self.disks.error:
            raise RuntimeError("Individual disk utilization unverified: " + self.disks.error)
        if self.samples and not self.last_disk_count:
            raise RuntimeError("Physical disk counters produced no valid individual observations")

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        self.disks.close()


def rng(root, generator, purpose, suffix=0):
    return np.random.default_rng(np.random.SeedSequence([root, generator, purpose, suffix]))


def fit_seed(root, generator):
    return int(np.random.SeedSequence([root, generator, 5, 0]).generate_state(1)[0])


def latent(root, generator, n):
    labels = np.repeat(np.arange(4), n // 4)
    return rng(root, generator, 1).permutation(labels)


def event_truth(root, n, regime):
    base = latent(root, 2, n)
    states = np.repeat(base[:, None], 36, axis=1)
    changed = np.zeros(n, dtype=bool)
    onset = 34 if regime == STRESS[0] else 19
    if regime in ("persistent_transition", "transition_and_shift", *STRESS):
        amount = int(n / 4 * (0.6 if regime == STRESS[1] else 0.2))
        random = rng(root, 2, 4)
        for c in range(4):
            index = random.permutation(np.flatnonzero(base == c))[:amount]
            changed[index] = True
            destination = (0 if c != 0 else 1) if regime == STRESS[1] else (c + 1) % 4
            states[index, onset - 1:] = destination
    return base, states, changed, onset


def panel_data(root, strength, regime, n=160):
    base, states, changed, onset = event_truth(root, n, regime)
    random = rng(root, 2, 2)
    b = random.normal(0, .4, (n, 1, 5))
    eps = random.normal(0, .3, (n, 36, 5))
    blevel = random.normal(0, .15, (n, 1))
    elevel = random.normal(0, .08, (n, 36))
    delta = 1.0 if strength == "strong" else .25
    means = np.asarray(M_ROWS, dtype=float)[states]
    logits = np.array([-1.4, -2.0, -2.4, -.6, -1.6]) + delta * means[:, :, :5] + b + eps
    log_total = 10 + .3 * delta * means[:, :, 5] + blevel + elevel
    if regime in ("common_shift", "transition_and_shift"):
        h = np.maximum(np.arange(1, 37) - 12, 0) / 24
        logits += h[None, :, None] * np.array([0., .8, 0., 0., 0.])
        log_total += .25 * h[None]
    full = np.concatenate((logits, np.zeros((n, 36, 1))), axis=2)
    full -= full.max(axis=2, keepdims=True)
    exp = np.exp(full)
    shares = (exp / exp.sum(axis=2, keepdims=True))[:, :, :5]
    panel = P.Panel(ids=np.array([f"synthetic-{i:04}" for i in range(n)]),
                    names=np.array([f"synthetic-{i}" for i in range(n)]),
                    regions=np.repeat("synthetic", n),
                    periods=tuple(f"synthetic-month-{i:03}" for i in range(1, 37)),
                    totals=np.exp(log_total), shares=shares, sha256="generated")
    scaler = P.fit_scaler(panel, 12)
    monthly = P.monthly_attributes(panel, scaler)
    x = P.annual_profile(monthly, slice(0, 12))
    return x, monthly, {"base": base, "states": states, "changed": changed,
                        "onset": onset, "censored": regime == STRESS[0]}, panel


def gaussian_data(root, strength, n=160):
    labels = latent(root, 1, n)
    covariance = np.eye(6)
    covariance[0, 1] = covariance[1, 0] = .35
    covariance[2, 3] = covariance[3, 2] = .35
    eps = rng(root, 1, 2).standard_normal((n, 6)) @ np.linalg.cholesky(covariance).T
    x = (2.0 if strength == "strong" else .65) * np.asarray(M_ROWS)[labels] + eps
    q75, q25 = np.percentile(x, [75, 25], axis=0)
    scale = q75 - q25
    if (scale <= 1e-12).any():
        raise ValueError("Synthetic IQR is zero")
    return (x - np.median(x, axis=0)) / scale, labels


def make_graph(x, truth, root, generator, family, n_edges=1200):
    n = len(x)
    rows, cols = np.triu_indices(n, 1)
    source = family.removesuffix("_permuted")
    suffix = {"strong_noise": 1, "weak_independent": 2, "weak_derived": 3,
              "weak_noise": 4, "weak_conflict": 5, "strong_derived": 6}[source]
    random = rng(root, generator, 3, suffix)
    if "derived" in source:
        union = triu(knn_graph(x, k=min(15, n - 1))[0], k=1).tocoo()
        choice = random.choice(len(union.data), n_edges, replace=False)
        rr, cc = union.row[choice], union.col[choice]
    else:
        weights = np.ones(len(rows))
        if source == "weak_independent":
            weights[truth[rows] == truth[cols]] = 10
        elif source == "weak_conflict":
            nuisance = np.empty(n, dtype=int)
            for c in range(4):
                indexes = np.flatnonzero(truth == c)
                nuisance[indexes] = random.permutation(np.repeat(np.arange(4), len(indexes) // 4))
            weights[nuisance[rows] == nuisance[cols]] = 10
        scores = np.log(weights) + random.gumbel(size=len(weights))
        choice = np.argsort(-scores, kind="stable")[:n_edges]
        rr, cc = rows[choice], cols[choice]
    graph = csr_matrix((np.ones(len(rr) * 2),
                        (np.concatenate((rr, cc)), np.concatenate((cc, rr)))), shape=(n, n))
    if family.endswith("_permuted"):
        permutation = rng(root, generator, 6, suffix).permutation(n)
        graph = graph[permutation][:, permutation].tocsr()
    return graph


def feature_hartigan(x, initial, seed, max_sweeps=200):
    labels = initial.copy()
    k = int(labels.max()) + 1
    tss = np.square(x - x.mean(axis=0)).sum()
    random = np.random.default_rng(seed)
    trace, moves = [], []
    for sweep in range(max_sweeps):
        centers = A.centroids(x, labels)
        counts = np.bincount(labels, minlength=k)
        changed = 0
        for i in random.permutation(len(x)):
            old = labels[i]
            if counts[old] == 1:
                continue
            distances = np.square(x[i] - centers).sum(axis=1)
            removal = counts[old] / (counts[old] - 1) * distances[old]
            costs = (counts / (counts + 1) * distances - removal) / tss
            costs[old] = 0
            new = int(np.argmin(costs))
            if costs[new] < -1e-14:
                centers[old] -= (x[i] - centers[old]) / (counts[old] - 1)
                centers[new] += (x[i] - centers[new]) / (counts[new] + 1)
                labels[i] = new
                counts[old] -= 1
                counts[new] += 1
                changed += 1
        sse = float(np.square(x - A.centroids(x, labels)[labels]).sum())
        if trace and sse > trace[-1] + 1e-10 * max(1, trace[-1]):
            raise ArithmeticError("Feature-only Hartigan SSE increased")
        trace.append(sse)
        moves.append(changed)
        if not changed:
            break
    return labels, {"algorithm": "feature_only_exact_hartigan", "sse": trace[-1],
                    "tss": float(tss), "converged": moves[-1] == 0,
                    "sweeps": len(moves), "moves": moves, "sse_trace": trace,
                    "n_init_actual": 0, "supplied_baseline": True}


def fit_models(x, graph, seed, phase, selected, monitor, cache):
    """Deliberately has no truth argument."""
    n_init = 50
    key = hashlib.sha256(x.tobytes()).hexdigest()
    def call(name, operation):
        monitor.gate()
        begin = time.perf_counter()
        labels, details = operation()
        if len(np.unique(labels)) != 4:
            raise ValueError(f"{name}: fewer than four nonempty clusters")
        return labels, {**clean(details), "elapsed_seconds": time.perf_counter() - begin}
    if key not in cache:
        model = KMeans(4, n_init=n_init, max_iter=300, random_state=seed)
        z, info = call("kmeans", lambda: (model.fit_predict(x), {
            "n_init_actual": n_init, "max_iter": 300, "optimizer": "sklearn_kmeans++"}))
        cache[key] = {"km": (z, info), "hartigan": call("feature_hartigan",
                              lambda: feature_hartigan(x, z, seed))}
    result = {"kmeans50": cache[key]["km"], "feature_hartigan": cache[key]["hartigan"]}
    km = result["kmeans50"][0]
    result["joint_zero_reuse"] = call("joint_zero_reuse",
        lambda: fit_graph_regularized_kmeans(x, graph, km, alpha=0, seed=seed, max_sweeps=200))
    if not np.array_equal(result["joint_zero_reuse"][0], km):
        raise AssertionError("Production alpha=0 did not reuse KMeans")
    y = x - x.mean(axis=0)
    rows = A.binary(graph).toarray()
    rows -= rows.mean()
    weight = np.sqrt(np.square(y).sum() / np.square(rows).sum())
    like = KMeans(4, n_init=n_init, max_iter=300, random_state=seed)
    result["kefrin_like_equal_scatter50"] = call("kefrin_like",
        lambda: (like.fit_predict(np.hstack([y, weight * rows])),
                 {"algorithm": "pinned_kefrin_like_common_budget", "n_init_actual": n_init}))
    alphas = [.1, .25, .5, 1.] if phase == "tune" else sorted({.5, selected["joint_alpha"]})
    betas = [.25, .5, 1., 2.] if phase == "tune" else sorted({1., selected["kefrine_beta"]})
    for alpha in alphas:
        result[f"joint_{alpha:g}"] = call(f"joint_{alpha:g}", lambda alpha=alpha:
            fit_graph_regularized_kmeans(x, graph, km, alpha=alpha, seed=seed, max_sweeps=200))
    for beta in betas:
        result[f"kefrine_{beta:g}"] = call(f"kefrine_{beta:g}", lambda beta=beta:
            fit_kefrine(x, graph, k=4, beta=beta, seed=seed, n_init=n_init, max_iter=300))
    result["joint_alpha_0.5"] = result["joint_0.5"]
    result["kefrine_equations_equal_scatter50"] = result["kefrine_1"]
    if phase == "evaluation":
        result["joint_tuned"] = result[f"joint_{selected['joint_alpha']:g}"]
        result["kefrine_tuned"] = result[f"kefrine_{selected['kefrine_beta']:g}"]
        keep = ("kmeans50", "joint_zero_reuse", "joint_alpha_0.5", "feature_hartigan",
                "kefrin_like_equal_scatter50", "kefrine_equations_equal_scatter50",
                "joint_tuned", "kefrine_tuned")
        result = {name: result[name] for name in keep}
    return result


def mapping(labels, truth):
    table = np.zeros((4, 4), dtype=int)
    np.add.at(table, (labels, truth), 1)
    rows, cols = linear_sum_assignment(-table)
    result = np.empty(4, dtype=int)
    result[rows] = cols
    return result


def safe_fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def score_trajectory(labels, annual, truth, method, root, family, regime, variant):
    """Truth is used only after model output, with one fixed semantic mapping."""
    states, base, changed = truth["states"], truth["base"], truth["changed"]
    semantic = mapping(annual, base)
    mapped = semantic[labels]
    end = T.persistence(labels, annual, 6)
    pred = end["persistent_change_mask"]
    end_destination = semantic[np.maximum(end["persistent_new_type"], 0)]
    correct_end = pred & changed & (end_destination == states[:, -1])
    tp = int(correct_end.sum())
    fp = int((pred & ~correct_end).sum())
    eligible = changed & (not truth["censored"])
    onset = truth["onset"]
    first = {}
    events = []
    good_time = np.full(len(labels), -1, dtype=int)
    for month in range(18, labels.shape[1] + 1):
        flag = T.persistence(labels[:, :month], annual, 6)
        for i in np.flatnonzero(flag["persistent_change_mask"]):
            destination = int(semantic[flag["persistent_new_type"][i]])
            pair = (int(i), destination)
            if pair in first:
                continue
            first[pair] = month
            correct = bool(changed[i] and month >= onset and destination == states[i, month - 1])
            timely = correct and not truth["censored"] and month <= onset + 8
            if timely and good_time[i] < 0:
                good_time[i] = month
            events.append({"seed": root, "family": family, "regime": regime,
                           "method": method, "variant": variant, "object": int(i),
                           "destination": destination, "month": month,
                           "true_change": bool(changed[i]), "correct": correct,
                           "timely": timely, "right_censored": truth["censored"]})
    detected = eligible & (good_time >= 0)
    delay = np.where(detected, good_time - onset, 9)[eligible]
    unchanged = ~changed
    raw_switches = np.diff(labels[:, 12:], axis=1) != 0
    correct_states = mapped[:, 12:] == states[:, 12:]
    monthly_macro = []
    for column in range(correct_states.shape[1]):
        present = np.unique(states[:, column + 12])
        monthly_macro.append(np.mean([correct_states[states[:, column + 12] == group, column].mean()
                                      for group in present]))
    row = {"seed": root, "family": family, "regime": regime, "method": method,
           "variant": variant, "true_events": int(changed.sum()),
           "eligible_events": int(eligible.sum()), "censored_events": int((changed & truth["censored"]).sum()),
           "tail_tp": tp, "tail_fp": fp, "tail_fn": int(changed.sum()) - tp,
           "tail_precision": safe_fraction(tp, int(pred.sum())),
           "tail_recall": safe_fraction(tp, int(changed.sum())),
           "recall": safe_fraction(int(detected.sum()), int(eligible.sum())),
           "fp_fraction": safe_fraction(int((pred & unchanged).sum()), int(unchanged.sum())),
           "false_prefix_events": sum(not e["correct"] for e in events),
           "changed_destination_accuracy": float(np.mean(mapped[changed, 12:] == states[changed, 12:])) if changed.any() else None,
           "post_change_destination_accuracy": float(np.mean(mapped[changed, onset - 1:] == states[changed, onset - 1:])) if changed.any() else None,
           "state_accuracy": float(np.mean(correct_states)),
           "macro_state_accuracy": float(np.mean(monthly_macro)),
           "unchanged_state_accuracy": float(np.mean(correct_states[unchanged])) if unchanged.any() else None,
           "mean_monthly_ari": float(np.mean([adjusted_rand_score(states[:, t], labels[:, t]) for t in range(12, labels.shape[1])])),
           "stable_fraction_unchanged": float(np.mean(~raw_switches[unchanged].any(axis=1))) if unchanged.any() else None,
           "false_raw_switches": int(raw_switches[unchanged].sum()),
           "capped_delay_mean": float(delay.mean()) if len(delay) else None,
           "detected_delay_mean": float((good_time[detected] - onset).mean()) if detected.any() else None}
    return row, events


def static_row(x, graph, truth, method, fitted, generator, root, family, phase):
    labels, info = fitted
    centers = A.centroids(x, labels)
    projection = A.nearest(x, centers)
    score = __import__("sbercluster.joint", fromlist=["joint_objective"]).joint_objective(x, graph, labels, .5)
    return {"phase": phase, "generator": generator, "seed": root, "family": family,
            "method": method, "ari": float(adjusted_rand_score(truth, labels)),
            "projection_ari": float(adjusted_rand_score(truth, projection)),
            "projection_disagreement": float(np.mean(projection != labels)),
            "min_group_size": int(np.bincount(labels, minlength=4).min()),
            "group_sizes": json.dumps(np.bincount(labels, minlength=4).tolist()),
            "attribute_sse_over_tss": score["attribute_SSE_over_TSS"],
            "graph_cut_fraction": score["graph_cut_fraction"],
            "joint_alpha0.5_objective": score["objective"],
            "runtime_seconds": info["elapsed_seconds"],
            "converged": info.get("converged"), "sweeps": info.get("sweeps")}


def free_labels(monthly, seed, monitor):
    raw = []
    for month in range(monthly.shape[1]):
        monitor.gate()
        raw.append(KMeans(4, n_init=20, random_state=seed).fit_predict(monthly[:, month]))
    return np.stack(raw, axis=1)


def phase_run(cfg, phase, seeds, out, monitor, selected=None):
    static, temporal, event_rows, graphs = [], [], [], []
    out.mkdir(parents=True)
    names = [f["id"] for f in cfg["families"]]
    if phase == "evaluation":
        names += list(EXTRA)
    for root in seeds:
        cache, temporal_cache, free_cache = {}, {}, {}
        event_rows = []
        seed_dir = out / f"seed-{root}"
        seed_dir.mkdir()
        arrays = {}
        for generator in (1, 2):
            for family in names:
                strength = "strong" if family.startswith("strong") else "weak"
                if generator == 1:
                    x, truth = gaussian_data(root, strength)
                else:
                    x, monthly, truth_info, _ = panel_data(root, strength, "stable")
                    truth = truth_info["base"]
                graph = make_graph(x, truth, root, generator, family)
                np.savez_compressed(seed_dir / f"model-input-g{generator}-{family}.npz",
                                    X=x, adjacency=graph.toarray().astype("uint8"))
                np.savez_compressed(seed_dir / f"evaluator-truth-g{generator}-{family}.npz", labels=truth)
                degree = graph.getnnz(axis=1)
                graphs.append({"phase": phase, "generator": generator, "seed": root,
                               "family": family, "edges": graph.nnz // 2,
                               "degree_p10": float(np.quantile(degree, .1)),
                               "degree_median": float(np.median(degree)),
                               "degree_p90": float(np.quantile(degree, .9)),
                               "components": int(connected_components(graph)[0]),
                               "within_truth_edge_fraction": float(np.mean(truth[triu(graph, k=1).tocoo().row] == truth[triu(graph, k=1).tocoo().col]))})
                models = fit_models(x, graph, fit_seed(root, generator), phase, selected, monitor, cache)
                if family.startswith("weak") and not family.endswith("_permuted"):
                    monitor.gate()
                    begin = time.perf_counter()
                    z = SpectralClustering(4, affinity="precomputed", assign_labels="cluster_qr",
                                           random_state=fit_seed(root, generator)).fit_predict(graph)
                    static.append(static_row(x, graph, truth, "spectral_graph_diagnostic",
                        (z, {"elapsed_seconds": time.perf_counter() - begin}), generator, root, family, phase))
                for method, fitted in models.items():
                    static.append(static_row(x, graph, truth, method, fitted, generator, root, family, phase))
                write_json(seed_dir / f"fit-g{generator}-{family}.json",
                           {m: {"labels": z, "info": info} for m, (z, info) in models.items()})
                if phase != "evaluation" or generator != 2:
                    continue
                regimes = list(REGIMES)
                if family in ("weak_independent", "weak_derived"):
                    regimes += list(STRESS)
                models = {**models, "freeze_kmeans_negative_control": models["kmeans50"]}
                for regime in regimes:
                    tk = (strength, regime)
                    if tk not in temporal_cache:
                        xr, xm, actual, _ = panel_data(root, strength, regime)
                        np.testing.assert_array_equal(x, xr)
                        adjusted = P.remove_national_wave(xm, x)
                        temporal_cache[tk] = (xm, adjusted, actual)
                        free_cache[tk] = free_labels(adjusted, fit_seed(root, 2), monitor)
                    xm, adjusted, actual = temporal_cache[tk]
                    arrays[f"truth_{strength}_{regime}"] = actual["states"].astype("int8")
                    for method, (annual, _) in models.items():
                        centers = A.centroids(x, annual)
                        for variant, data in (("adjusted", adjusted), ("raw", xm)):
                            labels = (np.repeat(annual[:, None], 36, axis=1)
                                      if method.startswith("freeze_") else T.assign_monthly(data, centers))
                            row, events = score_trajectory(labels, annual, actual, method, root, family, regime, variant)
                            temporal.append(row)
                            event_rows += events
                            arrays[f"{family}_{regime}_{method}_{variant}"] = labels.astype("int8")
                        if not method.startswith("freeze_"):
                            aligned = np.stack([T.align_to(annual, free_cache[tk][:, t], 4) for t in range(36)], axis=1)
                            row, events = score_trajectory(aligned, annual, actual, method, root, family, regime, "free_reclustering_diagnostic")
                            temporal.append(row)
                            event_rows += events
                            arrays[f"{family}_{regime}_{method}_free"] = aligned.astype("int8")
        if arrays:
            np.savez_compressed(seed_dir / "temporal-labels-and-evaluator-truth.npz", **arrays)
        write_csv(out / "static-metrics.csv", static)
        write_csv(out / "temporal-metrics.csv", temporal)
        append_csv(out / "events.csv", event_rows)
        write_csv(out / "graph-diagnostics.csv", graphs)
        write_json(out / "checkpoint.json", {"completed_seeds": seeds[:seeds.index(root) + 1]})
        print(f"{phase}: completed seed {root} ({seeds.index(root)+1}/{len(seeds)})", flush=True)
    return static, temporal


def tune_selection(rows, cfg):
    choices = {}
    for prefix, values, field in (("joint", [.1, .25, .5, 1], "joint_alpha"),
                                  ("kefrine", [.25, .5, 1, 2], "kefrine_beta")):
        means = {}
        for value in values:
            selected = [r for r in rows if r["method"] == f"{prefix}_{value:g}"]
            if len(selected) != 96:
                raise ValueError("Incomplete tuning grid")
            means[value] = float(np.mean([r["ari"] for r in selected]))
        best = max(means.values())
        choices[field] = min(v for v, mean in means.items() if mean >= best - 1e-12)
        choices[prefix + "_candidate_means"] = means
    choices["tune_seeds"] = cfg["seeds"]["tune"]
    choices["rule"] = cfg["budget"]["meta_tune_metric"]
    return choices


def summarize(static, temporal, cfg, output):
    output.mkdir()
    contrasts = []
    roots = cfg["seeds"]["evaluation"]
    def sval(generator, root, family, method, key="ari"):
        found = [r[key] for r in static if r["generator"] == generator and r["seed"] == root and r["family"] == family and r["method"] == method]
        if len(found) != 1:
            raise ValueError((generator, root, family, method))
        return found[0]
    def tval(root, regime, method, key):
        found = [r[key] for r in temporal if r["seed"] == root and r["family"] == "weak_derived" and r["regime"] == regime and r["method"] == method and r["variant"] == "adjusted"]
        if len(found) != 1:
            raise ValueError((root, regime, method))
        return found[0]
    vectors = {}
    for root in roots:
        c1 = np.mean([(sval(g, root, "weak_independent", "joint_alpha_0.5") - sval(g, root, "weak_independent", "kmeans50")) -
                       (sval(g, root, "weak_derived", "joint_alpha_0.5") - sval(g, root, "weak_derived", "kmeans50")) for g in (1, 2)])
        c2 = np.mean([sval(g, root, f["id"], "kefrine_equations_equal_scatter50") - sval(g, root, f["id"], "kefrin_like_equal_scatter50") for g in (1, 2) for f in cfg["families"]])
        c3 = np.mean([tval(root, r, "joint_alpha_0.5", "recall") - tval(root, r, "kmeans50", "recall") for r in ("persistent_transition", "transition_and_shift")])
        c4 = tval(root, "common_shift", "joint_alpha_0.5", "fp_fraction") - tval(root, "common_shift", "kmeans50", "fp_fraction")
        for name, value in zip(("C1", "C2", "C3", "C4"), (c1, c2, c3, c4)):
            contrasts.append({"comparison": name, "seed": root, "difference": float(value)})
            vectors.setdefault(name, []).append(value)
    random = np.random.default_rng(cfg["seeds"]["bootstrap"])
    indexes = random.integers(0, len(roots), (10000, len(roots)))
    signs = random.choice((-1, 1), (10000, len(roots)))
    intervals = []
    for name, values in vectors.items():
        array = np.asarray(values)
        samples = array[indexes].mean(axis=1)
        mean = float(array.mean())
        p = (1 + np.sum(np.abs((signs * array).mean(axis=1)) >= abs(mean) - 1e-15)) / 10001
        intervals.append({"comparison": name, "mean": mean, "median": float(np.median(array)),
                          "ci95_low": float(np.quantile(samples, .025)), "ci95_high": float(np.quantile(samples, .975)), "signflip_p": float(p)})
    ordered = sorted(intervals, key=lambda r: r["signflip_p"])
    previous = 0
    for i, row in enumerate(ordered):
        previous = max(previous, min(1, (len(ordered) - i) * row["signflip_p"]))
        row["holm_adjusted_p"] = previous
    write_csv(output / "contrasts.csv", contrasts)
    write_csv(output / "confidence-intervals.csv", intervals)
    mechanisms = []
    for generator in (1, 2):
        for family in [f["id"] for f in cfg["families"]] + list(EXTRA):
            values = [sval(generator, root, family, "joint_alpha_0.5") -
                      sval(generator, root, family, "feature_hartigan") for root in roots]
            samples = np.asarray(values)[indexes].mean(axis=1)
            mechanisms.append({"generator": generator, "family": family,
                               "comparison": "joint_fixed_minus_feature_only_exact_hartigan",
                               "mean": float(np.mean(values)),
                               "ci95_low": float(np.quantile(samples, .025)),
                               "ci95_high": float(np.quantile(samples, .975)),
                               "inference": "preregistered_mechanistic_descriptive_comparison"})
    for generator in (1, 2):
        for family in ("weak_independent", "weak_derived"):
            values = [sval(generator, root, family, "joint_alpha_0.5") -
                      sval(generator, root, family + "_permuted", "joint_alpha_0.5") for root in roots]
            samples = np.asarray(values)[indexes].mean(axis=1)
            mechanisms.append({"generator": generator, "family": family,
                               "comparison": "joint_fixed_original_minus_same_W_vertex_permutation",
                               "mean": float(np.mean(values)),
                               "ci95_low": float(np.quantile(samples, .025)),
                               "ci95_high": float(np.quantile(samples, .975)),
                               "inference": "preregistered_mechanistic_descriptive_comparison"})
    write_csv(output / "mechanistic-comparisons.csv", mechanisms)
    aggregates = []
    for source, dimensions, metrics in ((static, ("generator", "family", "method"), ("ari", "projection_ari", "projection_disagreement", "min_group_size")),
        (temporal, ("family", "regime", "method", "variant"), ("mean_monthly_ari", "state_accuracy", "macro_state_accuracy", "changed_destination_accuracy", "unchanged_state_accuracy", "post_change_destination_accuracy", "recall", "fp_fraction", "capped_delay_mean"))):
        groups = {}
        for row in source:
            groups.setdefault(tuple(row[k] for k in dimensions), []).append(row)
        for keys, values in groups.items():
            for metric in metrics:
                nums = [row[metric] for row in values if row[metric] is not None]
                if not nums:
                    continue
                aggregates.append({**dict(zip(dimensions, keys)), "metric": metric,
                                   "mean": float(np.mean(nums)), "sd_between_seeds": float(np.std(nums, ddof=1)) if len(nums) > 1 else None,
                                   "min": float(min(nums)), "max": float(max(nums)), "n_seeds": len(nums)})
    write_csv(output / "summary.csv", aggregates)
    return intervals


def environment():
    packages = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "scikit-learn", "pandas", "psutil", "threadpoolctl")}
    expected = json.loads((ROOT / "reports/environment.json").read_text(encoding="utf-8"))
    for name, version in packages.items():
        if name in expected["packages"] and version != expected["packages"][name]:
            raise ValueError(f"{name}: {version} != {expected['packages'][name]}")
    return {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
            "packages": packages, "thread_environment": {key: os.environ[key] for key in
             ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}


def validate(cfg):
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    for strength in ("strong", "weak"):
        base_x, base_m, truth, panel = panel_data(44001, strength, "stable", n=16)
        check("positive compositional shares", (panel.shares > 0).all() and (panel.shares.sum(axis=2) < 1).all())
        check("external balanced truth", np.array_equal(np.bincount(truth["base"]), [4] * 4))
        for regime in (*REGIMES, *STRESS):
            x, monthly, actual, _ = panel_data(44001, strength, regime, n=16)
            check("same development X " + regime, np.array_equal(x, base_x))
            check("same development months " + regime, np.array_equal(monthly[:, :12], base_m[:, :12]))
        for family in [f["id"] for f in cfg["families"]] + list(EXTRA):
            graph = make_graph(base_x, truth["base"], 44001, 2, family, n_edges=40)
            check("symmetric fixed density " + family, (graph != graph.T).nnz == 0 and graph.nnz == 80 and not graph.diagonal().any())
            if family.endswith("_permuted"):
                original = make_graph(base_x, truth["base"], 44001, 2, family.removesuffix("_permuted"), n_edges=40)
                check("permutation degree multiset " + family, np.array_equal(np.sort(graph.getnnz(axis=1)), np.sort(original.getnnz(axis=1))))
        for regime in REGIMES:
            _, _, actual, _ = panel_data(44001, strength, regime, n=160)
            perfect = actual["states"]
            row, _ = score_trajectory(perfect, actual["base"], actual, "perfect", 44001, "fixture", regime, "fixture")
            if actual["changed"].any():
                check("perfect event recall and delay", row["recall"] == 1 and row["detected_delay_mean"] == 5)
                frozen = np.repeat(actual["base"][:, None], 36, axis=1)
                row, _ = score_trajectory(frozen, actual["base"], actual, "freeze", 44001, "fixture", regime, "fixture")
                check("freeze cannot win", row["recall"] == 0 and row["post_change_destination_accuracy"] == 0)
            else:
                check("unchanged fixture no false persistent", row["fp_fraction"] == 0)
                check("perfect unchanged state accuracy", row["unchanged_state_accuracy"] == 1 and row["macro_state_accuracy"] == 1)
    base, states, changed, onset = event_truth(44001, 160, STRESS[1])
    actual = {"base": base, "states": states, "changed": changed, "onset": onset, "censored": False}
    constant = np.zeros((160, 36), dtype=int)
    row, _ = score_trajectory(constant, actual["base"], actual, "constant", 44001, "fixture", STRESS[1], "fixture")
    check("broad stress macro rejects majority-only accuracy", row["macro_state_accuracy"] == .25 and np.isclose(row["state_accuracy"], .475))
    check("broad stress unchanged state accuracy", row["unchanged_state_accuracy"] == .25)
    x = np.array([0., 1., 1.1, 2.1])[:, None]
    annual = np.array([0, 1, 0, 1])
    projected = T.assign_monthly(np.repeat(x[:, None, :], 12, axis=1), A.centroids(x, annual))
    check("Pro exact projection counterexample", np.mean(projected[:, 0] != annual) == .5)
    check("Pro unchanged false-tail6", T.persistence(projected, annual, 6)["persistent_changes"] == 2)
    x = np.array([0., 3., 4., 7.])[:, None]
    labels, info = feature_hartigan(x, np.array([0, 0, 1, 1]), 71)
    check("SSE-only exact Hartigan", info["sse"] < 9 and info["converged"])
    check("seed independence", not set(cfg["seeds"]["tune"]) & set(cfg["seeds"]["evaluation"]))
    return {"status": "PASS_EXACT_FIXTURES", "checks": checks, "experiment_fits": 0,
            "notes": "Generated invariants, planted perfect/freeze scorers and hand examples only; no quality outcomes."}


def registration(args, cfg):
    if digest(args.plan) != PLAN_SHA or digest(args.design) != DESIGN_SHA:
        raise ValueError("Original plan bytes changed")
    pins = json.loads(args.pins.read_text(encoding="utf-8-sig"))
    for name, expected in pins.items():
        if digest(ROOT / name) != expected:
            raise ValueError("Scientific pin mismatch: " + name)
    files = {name: digest(ROOT / name) for name in (*CORE, "scripts/competition_synthetic.py")}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    shutil.copyfile(args.plan, args.output / "PLAN-original.md")
    shutil.copyfile(args.design, args.output / "design-original.json")
    shutil.copyfile(args.addendum, args.output / "PREREG-ADDENDUM.md")
    reg = {"plan_sha256": PLAN_SHA, "design_sha256": DESIGN_SHA,
           "addendum_sha256": digest(args.addendum), "files": files,
           "pins_sha256": digest(args.pins), "scientific_pins": pins, "environment": environment(),
           "base_commit": cfg["base_commit"], "git_head_at_registration": subprocess.check_output(
           ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
           "registered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "scientific_runs_before_registration": 0,
           "commands": {"run": "python -X utf8 -m scripts.competition_synthetic run --registration " +
                       str(args.output / "registration.json") + " --run-dir " + str(args.output / "run") +
                       " --diagnostics " + str(args.output / "diagnostics/diagnostics.json") +
                       " --slot-id LEAD_ASSIGNED_SLOT"}}
    write_json(args.output / "registration.json", reg)
    return reg


def run(args):
    regpath = args.registration.resolve()
    reg = json.loads(regpath.read_text(encoding="utf-8"))
    for name, expected in reg["files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Registered implementation changed: " + name)
    for name, expected in (("PLAN-original.md", reg["plan_sha256"]),
                           ("design-original.json", reg["design_sha256"]),
                           ("PREREG-ADDENDUM.md", reg["addendum_sha256"])):
        if digest(regpath.parent / name) != expected:
            raise ValueError("Registered plan changed: " + name)
    cfg = json.loads((regpath.parent / "design-original.json").read_text(encoding="utf-8"))
    diagnostic = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    if diagnostic["registration_sha256"] != digest(regpath) or diagnostic["status"] != "COMPLETED_MECHANISTIC_DIAGNOSTICS":
        raise ValueError("The registered real projection diagnostics must precede the large matrix")
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    args.run_dir.mkdir(parents=True)
    monitor = ResourceMonitor(args.run_dir / "resource-monitor.jsonl", heavy=True)
    try:
        monitor.start()
        numeric()
        from threadpoolctl import threadpool_info
        with threadpool_limits(limits=1):
            write_json(args.run_dir / "environment.json", {**environment(), "threadpools": threadpool_info(), "slot_id": args.slot_id})
            write_json(args.run_dir / "validation.json", validate(cfg))
            start = time.perf_counter()
            tune, _ = phase_run(cfg, "tune", cfg["seeds"]["tune"], args.run_dir / "tune", monitor)
            selected = tune_selection(tune, cfg)
            selected["plan_sha256"] = reg["plan_sha256"]
            selected["tune_metrics_sha256"] = digest(args.run_dir / "tune/static-metrics.csv")
            write_json(args.run_dir / "selected.json", selected)
            selected_sha = digest(args.run_dir / "selected.json")
            static, temporal = phase_run(cfg, "evaluation", cfg["seeds"]["evaluation"],
                                        args.run_dir / "evaluation", monitor, selected)
            intervals = summarize(static, temporal, cfg, args.run_dir / "report")
            if digest(args.run_dir / "selected.json") != selected_sha:
                raise AssertionError("Tuning changed during evaluation")
            for name, expected in reg["files"].items():
                if digest(ROOT / name) != expected:
                    raise AssertionError("Registered source changed during the run: " + name)
            result = {"status": "COMPLETED_SCIENTIFIC_RUN_PENDING_INDEPENDENT_REVIEW",
                      "duration_seconds": time.perf_counter() - start, "selected_sha256": selected_sha,
                      "registration_sha256": digest(regpath), "diagnostics_sha256": digest(args.diagnostics), "comparisons": intervals,
                      "all_evaluation_seeds": cfg["seeds"]["evaluation"],
                      "resource_samples": monitor.samples, "resource_errors": sorted(monitor.errors),
                      "no_2025_outcomes": True, "main_model_changed": False}
            write_json(args.run_dir / "result.json", result)
            print(json.dumps(clean(result), ensure_ascii=False, indent=2), flush=True)
    except Exception:
        write_json(args.run_dir / "failure.json", {"status": "FAILED", "traceback": traceback.format_exc(),
                                                "seed_failures_not_removed": True})
        raise
    finally:
        monitor.close()


def diagnostics(args):
    regpath = args.registration.resolve()
    reg = json.loads(regpath.read_text(encoding="utf-8"))
    for name, expected in reg["files"].items():
        if digest(ROOT / name) != expected:
            raise ValueError("Registered implementation changed: " + name)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    monitor = ResourceMonitor(args.output / "resource-monitor.jsonl")
    try:
        monitor.start()
        numeric()
        with threadpool_limits(limits=1):
            cfg = json.loads((regpath.parent / "design-original.json").read_text(encoding="utf-8"))
            write_json(args.output / "exact-fixtures.json", validate(cfg))
            panel_path = ROOT / "data/v12/panel.csv.gz"
            labels_path = ROOT / "reports/v1.2/labels.csv"
            model_path = ROOT / "reports/v1.2/model.npz"
            for path in (panel_path, labels_path, model_path):
                if digest(path) != reg["scientific_pins"][str(path.relative_to(ROOT)).replace("\\", "/")]:
                    raise ValueError("Pinned diagnostic input changed: " + str(path))
            panel = P.read_panel(panel_path)
            with labels_path.open(encoding="utf-8", newline="") as handle:
                source = {row["entity_id"]: int(row["type_2023"]) for row in csv.DictReader(handle)}
            if set(source) != set(panel.ids) or len(panel.ids) != 2016:
                raise ValueError("The real diagnostic cohort differs from pinned 2016 entities")
            annual = np.array([source[key] for key in panel.ids])
            monthly = P.monthly_attributes(panel, P.fit_scaler(panel, 12))
            x = P.annual_profile(monthly, slice(0, 12))
            with np.load(model_path, allow_pickle=False) as payload:
                centers = payload["centers"]
                ids = payload["ids"]
            np.testing.assert_array_equal(ids, panel.ids)
            np.testing.assert_allclose(centers, A.centroids(x, annual), atol=1e-12, rtol=1e-12)
            identical = np.repeat(x[:, None, :], 36, axis=1)
            fixed = T.assign_monthly(identical, centers)
            adjusted = T.assign_monthly(P.remove_national_wave(identical, x), centers)
            np.testing.assert_array_equal(fixed, adjusted)
            flags = T.persistence(fixed, annual, 6)
            squared = np.square(x[:, None, :] - centers[None, :, :]).sum(axis=2)
            nearest_two = np.sort(squared, axis=1)[:, :2]
            margins = nearest_two[:, 1] - nearest_two[:, 0]
            projected = fixed[:, 0]
            rows = [{"entity_id": key, "annual_type": int(annual[i]),
                     "projected_type": int(projected[i]),
                     "projection_disagreement": bool(annual[i] != projected[i]),
                     "persistent_disagreement_on_identical_profile": bool(flags["persistent_change_mask"][i]),
                     "squared_distance_margin": float(margins[i])}
                    for i, key in enumerate(panel.ids)]
            write_csv(args.output / "real-projection.csv", rows)
            report = {"status": "COMPLETED_MECHANISTIC_DIAGNOSTICS",
                      "registration_sha256": digest(regpath), "slot_id": args.slot_id,
                      "n": 2016, "k": 4,
                      "r0_projection_disagreement": float(np.mean(projected != annual)),
                      "r0_count": int(np.sum(projected != annual)),
                      "identical_profile_persistent_disagreement_count": int(flags["persistent_change_mask"].sum()),
                      "identical_profile_adjusted_and_raw_labels_equal": True,
                      "annual_centers_verified_against_pinned_model": True,
                      "by_annual_type": [{"type": c, "n": int((annual == c).sum()),
                                         "disagreement_count": int(np.sum((annual == c) & (projected != annual)))} for c in range(4)],
                      "distance_margin_quantiles": np.quantile(margins, [0, .01, .1, .5, 1]),
                      "input_sha256": {str(path.relative_to(ROOT)): digest(path) for path in (panel_path, labels_path, model_path)},
                      "environment": environment(), "experiment_fits": 0,
                      "limitations": ["Annual labels are a reference assignment, not external economic truth.",
                                      "The identical profile is a mechanistic control, not an observed economic transition.",
                                      "No temporal graph or interpolation is added."]}
            write_json(args.output / "diagnostics.json", report)
            print(json.dumps(clean(report), ensure_ascii=False, indent=2), flush=True)
    except Exception:
        write_json(args.output / "failure.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        monitor.close()



def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "register"):
        part = sub.add_parser(name)
        part.add_argument("--design", type=Path, required=True)
        part.add_argument("--output", type=Path, required=True)
        if name == "register":
            part.add_argument("--plan", type=Path, required=True)
            part.add_argument("--pins", type=Path, required=True)
            part.add_argument("--addendum", type=Path, required=True)
    part = sub.add_parser("run")
    part.add_argument("--registration", type=Path, required=True)
    part.add_argument("--run-dir", type=Path, required=True)
    part.add_argument("--diagnostics", type=Path, required=True)
    part.add_argument("--slot-id", required=True)
    part = sub.add_parser("diagnostics")
    part.add_argument("--registration", type=Path, required=True)
    part.add_argument("--output", type=Path, required=True)
    part.add_argument("--slot-id", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "diagnostics":
        diagnostics(args)
    else:
        cfg = json.loads(args.design.read_text(encoding="utf-8-sig"))
        if args.command == "validate":
            monitor = ResourceMonitor(args.output.with_suffix(".resources.jsonl"))
            try:
                monitor.start()
                numeric()
                monitor.gate()
                with threadpool_limits(limits=1):
                    report = validate(cfg)
                write_json(args.output, report)
                print(json.dumps(report, ensure_ascii=False, indent=2))
            finally:
                monitor.close()
        else:
            reg = registration(args, cfg)
            print(json.dumps({"registered_utc": reg["registered_utc"], "commands": reg["commands"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

