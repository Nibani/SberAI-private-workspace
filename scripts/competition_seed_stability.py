"""Explicit additional optimization stability run of the fixed production recipe.

python -B -m scripts.competition_seed_stability --execute

The first of 20 predeclared seeds is timed before the remaining fits. No seed
selection, model selection, outcomes, or temporal-label tracking is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

for _pool in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_pool] = "1"
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
import psutil
from scipy.optimize import linear_sum_assignment
from scipy.sparse import save_npz
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sbercluster import panel as P
from sbercluster.graph import knn_graph
from sbercluster.joint import fit_graph_regularized_kmeans, joint_objective

SEEDS = tuple(range(61001, 61021))
DEFAULT_OUTPUT = ROOT / "reports/competition-enhancement/seed-stability"
TYPE_KEYS = ("metro", "industrial", "periphery", "remote")
SOURCES = ("scripts/competition_seed_stability.py", "scripts/verify_seed_stability.py",
           "scripts/validate_v12_findings.py", "sbercluster/panel.py", "sbercluster/graph.py",
           "sbercluster/joint.py", "scripts/competition_synthetic.py", "scripts/science_portable_resources.py",
           "configs/v12.json", "data/v12/panel.csv.gz", "reports/v1.2/labels.csv")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def align(raw, reference):
    overlap = np.zeros((4, 4), dtype=int)
    np.add.at(overlap, (reference, raw), 1)
    rows, cols = linear_sum_assignment(-overlap)
    mapping = np.empty(4, dtype=int)
    mapping[cols] = rows
    return mapping[raw], mapping, overlap


def describe_agreement(values):
    return {"mean_agreement": float(np.mean(values)), "min_agreement": float(np.min(values)),
            "p10_agreement": float(np.quantile(values, .1)), "median_agreement": float(np.median(values)),
            "all_runs_agree_n": int(np.sum(values == 1)),
            "below_0_8_agreement_n": int(np.sum(values < .8))}


def save_results(output, labels, raw_matrix, aligned_matrix, seed_rows, fits, total_seconds, status):
    n = len(seed_rows)
    raw = pd.DataFrame({"entity_id": labels.entity_id, "production_type": labels.type_2023})
    aligned = raw.copy()
    for col, seed in enumerate(SEEDS[:n]):
        raw[f"seed_{seed}"] = raw_matrix[:, col]
        aligned[f"seed_{seed}"] = aligned_matrix[:, col]
    raw.to_csv(output / "assignments.csv", index=False)
    aligned.to_csv(output / "aligned_assignments.csv", index=False)
    reference = labels.type_2023.to_numpy(int)
    agree = (aligned_matrix == reference[:, None]).sum(axis=1)
    support = labels.copy()
    support["production_type_key"] = [TYPE_KEYS[c] for c in reference]
    support["agreement_count"] = agree
    support["completed_seed_count"] = n
    support["planned_seed_count"] = len(SEEDS)
    support["agreement_fraction_completed"] = agree / n
    # The requested denominator 20 is available only after all 20 fits completed.
    support["agreement_over_20"] = agree / len(SEEDS) if n == len(SEEDS) else np.nan
    for c, key in enumerate(TYPE_KEYS):
        support[f"aligned_{key}_count"] = (aligned_matrix == c).sum(axis=1)
    support.to_csv(output / "municipality_support.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(output / "seed_metrics.csv", index=False)
    dump(output / "fits.json", fits)
    values = agree / n
    groups = []
    for c, key in enumerate(TYPE_KEYS):
        mask = reference == c
        groups.append({"production_type": c, "production_type_key": key, "municipalities": int(mask.sum()),
                       **describe_agreement(values[mask])})
    pd.DataFrame(groups).to_csv(output / "group_summary.csv", index=False)
    summary = {"status": status, "planned_seed_count": len(SEEDS), "completed_seed_count": n,
               "seeds": list(SEEDS[:n]), "municipalities": len(labels), "groups": groups,
               "overall": describe_agreement(values),
               "seed_ari": {"min": min(r["ari_with_production"] for r in seed_rows),
                            "mean": float(np.mean([r["ari_with_production"] for r in seed_rows])),
                            "max": max(r["ari_with_production"] for r in seed_rows)},
               "converged_seed_count": sum(r["converged"] for r in seed_rows),
               "runtime_seconds": total_seconds,
               "interpretation": "Additional optimization stability on fixed 2023 X23 and fixed graph. Seed changes KMeans initialization and joint vertex order. Hungarian alignment maximizes overlap with production labels separately for each seed. Agreement is a run frequency, not a probability of economic truth, uncertainty from new data, bootstrap stability, or temporal persistence.",
               "threshold_note": "below_0_8 is a descriptive diagnostic count, not a calibrated confidence threshold; all municipalities are exported including low agreement."}
    dump(output / "summary.json", summary)
    manifest = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name not in ("artifact_sha256.json", "resource-monitor.jsonl")}
    dump(output / "artifact_sha256.json", manifest)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="explicitly run the optimizers; no fits without this flag")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.execute:
        parser.error("Use --execute for the explicit 20-fit run, or scripts.verify_seed_stability for saved-result acceptance")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("No overwrite/resume; choose a fresh --output: " + str(output))
    output.mkdir(parents=True)
    process = psutil.Process()
    process.cpu_affinity([process.cpu_affinity()[0]])
    if os.name == "nt":
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        from scripts.competition_synthetic import ResourceMonitor
        adapter = "original_windows_pdh_guard"
    else:
        from scripts.science_portable_resources import PortableResourceMonitor as ResourceMonitor
        adapter = "portable_psutil_partial_v1"
    registration = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "REGISTERED_BEFORE_ANY_FIT",
                    "seeds": list(SEEDS), "selection": "consecutive new seeds fixed before any results; no replacement or filtering",
                    "recipe": {"K": 4, "features": list(P.FEATURES), "X23": "median of the 12 standardized 2023 monthly vectors",
                               "graph": "production weighted attribute kNN, deterministic union", "graph_k": 15,
                               "alpha": .5, "kmeans_n_init": 50, "max_sweeps": 200,
                               "random_state": "each registered seed used for both KMeans and joint sweep ordering"},
                    "matching": "SciPy linear_sum_assignment on negative 4x4 production/raw overlap counts, separately for every seed",
                    "forecast_policy": "time first registered seed; if 20*first_fit_seconds + setup_seconds >600 stop before seed 2, honestly partial",
                    "inputs_and_sources_sha256": {name: sha(ROOT / name) for name in SOURCES},
                    "python": sys.version, "packages": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "psutil", "threadpoolctl")},
                    "resource_adapter": adapter, "affinity": process.cpu_affinity(), "priority_at_start": process.nice(),
                    "data_scope": "2023 only for features and graph; physical panel contains 2023-2024, cropped before transformation; no 2024 model selection/outcomes or 2025 inputs"}
    dump(output / "registration.json", registration)
    print(json.dumps({"phase": "registered", "seeds": list(SEEDS), "registration_sha256": sha(output / "registration.json")}), flush=True)
    started = time.perf_counter()
    monitor = ResourceMonitor(output / "resource-monitor.jsonl", heavy=True)
    try:
        monitor.start()
        with threadpool_limits(limits=1):
            panel = P.read_panel(ROOT / "data/v12/panel.csv.gz")
            panel = replace(panel, totals=panel.totals[:, :12], shares=panel.shares[:, :12], periods=panel.periods[:12])
            scaler = P.fit_scaler(panel, 12)
            x = P.annual_profile(P.monthly_attributes(panel, scaler), slice(0, 12))
            if x.shape != (2016, 6):
                raise ValueError("Expected the unchanged 2016x6 X23 production space")
            graph, graph_info = knn_graph(x, k=15)
            labels = pd.read_csv(ROOT / "reports/v1.2/labels.csv", usecols=["entity_id", "name", "region", "type_2023"])
            if labels.entity_id.tolist() != panel.ids.tolist() or sorted(labels.type_2023.unique()) != [0, 1, 2, 3]:
                raise ValueError("Reference labels/order mismatch")
            reference = labels.type_2023.to_numpy(int)
            np.save(output / "features_2023.npy", x, allow_pickle=False)
            save_npz(output / "production_graph.npz", graph)
            dump(output / "input_state.json", {"scaler": scaler.as_dict(), "graph": graph_info,
                                               "reference_objective": joint_objective(x, graph, reference, .5)})
            setup_seconds = time.perf_counter() - started
            raw_columns, aligned_columns, seed_rows, fits = [], [], [], []
            status = "COMPLETE_20_SEEDS"
            for seed in SEEDS:
                monitor.gate()
                fit_start = time.perf_counter()
                initial = KMeans(4, n_init=50, random_state=seed).fit_predict(x)
                raw, info = fit_graph_regularized_kmeans(x, graph, initial, alpha=.5, seed=seed, max_sweeps=200)
                seconds = time.perf_counter() - fit_start
                matched, mapping, overlap = align(raw, reference)
                raw_columns.append(raw)
                aligned_columns.append(matched)
                fits.append({"seed": seed, "mapping_raw_to_production": mapping.tolist(), "overlap": overlap.tolist(), "fit": info})
                seed_rows.append({"seed": seed, "seconds": seconds, "ari_with_production": float(adjusted_rand_score(reference, raw)),
                                  "agreement_count": int(np.sum(matched == reference)), "agreement_fraction": float(np.mean(matched == reference)),
                                  "objective": info["objective"], "converged": info["converged"], "sweeps": info["sweeps"],
                                  "convergence_status": info["convergence_status"]})
                print(json.dumps({"phase": "fit_completed", **seed_rows[-1]}), flush=True)
                if seed == SEEDS[0]:
                    forecast = setup_seconds + len(SEEDS) * seconds
                    timing = {"timing_seed": seed, "first_fit_seconds": seconds, "setup_seconds": setup_seconds,
                              "forecast_20_seeds_seconds": forecast, "maximum_forecast_seconds": 600,
                              "decision": "CONTINUE_ALL_20" if forecast <= 600 else "STOP_PARTIAL_BEFORE_SEED_2"}
                    dump(output / "timing.json", timing)
                    print(json.dumps({"phase": "forecast", **timing}), flush=True)
                    if forecast > 600:
                        status = "PARTIAL_RUNTIME_LIMIT_ONE_SEED"
                        break
            dump(output / "resources.json", {"samples": monitor.samples, "errors": sorted(monitor.errors), "adapter": adapter,
                                              "affinity": process.cpu_affinity(), "priority": process.nice()})
            summary = save_results(output, labels, np.column_stack(raw_columns), np.column_stack(aligned_columns), seed_rows, fits, time.perf_counter() - started, status)
    finally:
        monitor.close()
    print(json.dumps({"phase": "complete", "status": summary["status"], "completed": summary["completed_seed_count"], "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
