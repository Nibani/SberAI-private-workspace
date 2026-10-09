"""Read-only saved-label verification; never fits or writes results.

python -B -m scripts.verify_seed_stability
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
for _pool in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_pool] = "1"
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import load_npz, triu
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
REVIEWABLE_REVISIONS = frozenset(("scripts/science_portable_resources.py", "scripts/verify_seed_stability.py"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def metrics(values):
    return {"mean_agreement": float(np.mean(values)), "min_agreement": float(np.min(values)),
            "p10_agreement": float(np.quantile(values, .1)), "median_agreement": float(np.median(values)),
            "all_runs_agree_n": int(np.sum(values == 1)), "below_0_8_agreement_n": int(np.sum(values < .8))}


def check_source_pin(name, registered_sha, current_sha, revision=None, historical_sha=None):
    """A reviewed guard/verifier revision never changes a pre-outcome pin."""
    if revision is None:
        assert current_sha == registered_sha, "Registered input/source changed: " + name
        return
    assert name in REVIEWABLE_REVISIONS, "Fit/data source cannot use a post-run revision: " + name
    assert revision["historical_sha256"] == registered_sha == historical_sha, "Historical source bytes differ: " + name
    assert revision["current_sha256"] == current_sha, "Reviewed current source changed: " + name
    assert revision["scientific_fit_code_changed"] is False


def verify_registered_sources(folder, registration):
    lineage_path = folder / "provenance/post-run-source-lineage.json"
    lineage = read(lineage_path) if lineage_path.exists() else None
    revisions = lineage["source_revisions"] if lineage else {}
    if lineage:
        assert lineage["schema"] == "seed_stability_post_run_guard_revision_v1"
        assert lineage["registration_sha256"] == sha(folder / "registration.json")
        assert lineage["scientific_outputs_changed"] is False and lineage["optimizer_fits_rerun"] == 0
        assert set(revisions) == REVIEWABLE_REVISIONS
        assert lineage["actual_run_adapter"] == registration["resource_adapter"] == "original_windows_pdh_guard"
    for name, expected in registration["inputs_and_sources_sha256"].items():
        revision = revisions.get(name)
        historical = None
        if revision:
            snapshot = (folder / revision["historical_snapshot"]).resolve()
            assert snapshot.is_relative_to((folder / "provenance").resolve()), "Historical snapshot must stay in artifact provenance"
            historical = sha(snapshot)
        check_source_pin(name, expected, sha(ROOT / name), revision, historical)
    return lineage


def check_pin_handling():
    """Pure mechanical regression checks; no files, optimizers, or GPU calls."""
    old, new = "registered-old-bytes", "reviewed-new-bytes"
    revision = {"historical_sha256": old, "current_sha256": new, "scientific_fit_code_changed": False}
    check_source_pin("sbercluster/joint.py", old, old)
    check_source_pin("scripts/science_portable_resources.py", old, new, revision, old)
    check_source_pin("scripts/verify_seed_stability.py", old, new, revision, old)
    failures = (
        ("scripts/science_portable_resources.py", old, new, None, None),
        ("scripts/science_portable_resources.py", old, new, revision, "different-historical-bytes"),
        ("scripts/science_portable_resources.py", old, "unreviewed-current-bytes", revision, old),
        ("sbercluster/joint.py", old, new, revision, old),
        ("data/v12/panel.csv.gz", old, new, revision, old),
    )
    for args in failures:
        try:
            check_source_pin(*args)
        except AssertionError:
            continue
        raise AssertionError("Pin-handling regression accepted an invalid source revision")
    return {"status": "PASS_PIN_HANDLING_REGRESSION", "accepted_cases": 3, "rejected_cases": len(failures), "files_written": 0, "optimizer_fits": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ROOT / "reports/competition-enhancement/seed-stability")
    parser.add_argument("--check-pin-handling", action="store_true", help="also exercise pure pin-lineage regression cases")
    args = parser.parse_args()
    folder = args.artifact.resolve()
    registration, summary = read(folder / "registration.json"), read(folder / "summary.json")
    assert registration["seeds"] == list(range(61001, 61021))
    assert registration["status"] == "REGISTERED_BEFORE_ANY_FIT"
    recipe = registration["recipe"]
    assert (recipe["K"], recipe["graph_k"], recipe["alpha"], recipe["kmeans_n_init"], recipe["max_sweeps"]) == (4, 15, .5, 50, 200)
    lineage = verify_registered_sources(folder, registration)
    for name, expected in read(folder / "artifact_sha256.json").items():
        assert sha(folder / name) == expected, "Saved artifact changed: " + name
    reference = pd.read_csv(ROOT / "reports/v1.2/labels.csv", usecols=["entity_id", "name", "region", "type_2023"])
    raw, aligned = pd.read_csv(folder / "assignments.csv"), pd.read_csv(folder / "aligned_assignments.csv")
    support, rows = pd.read_csv(folder / "municipality_support.csv"), pd.read_csv(folder / "seed_metrics.csv")
    fits = read(folder / "fits.json")
    n = summary["completed_seed_count"]
    assert len(reference) == len(raw) == len(aligned) == len(support) == 2016
    assert rows.seed.tolist() == summary["seeds"] == registration["seeds"][:n]
    assert len(fits) == n and summary["planned_seed_count"] == 20
    if summary["status"] == "COMPLETE_20_SEEDS":
        assert n == 20
    else:
        assert summary["status"] == "PARTIAL_RUNTIME_LIMIT_ONE_SEED" and n == 1
        assert read(folder / "timing.json")["decision"] == "STOP_PARTIAL_BEFORE_SEED_2"
    z = reference.type_2023.to_numpy(int)
    for frame in (raw, aligned, support):
        assert frame.entity_id.tolist() == reference.entity_id.tolist()
    np.testing.assert_array_equal(raw.production_type, z)
    np.testing.assert_array_equal(aligned.production_type, z)
    np.testing.assert_array_equal(support.type_2023, z)
    assert support.name.tolist() == reference.name.tolist() and support.region.tolist() == reference.region.tolist()
    x, graph = np.load(folder / "features_2023.npy", allow_pickle=False), load_npz(folder / "production_graph.npz")
    assert x.shape == (2016, 6) and np.isfinite(x).all() and graph.shape == (2016, 2016)
    assert (graph != graph.T).nnz == 0 and not graph.diagonal().any() and np.isfinite(graph.data).all()
    edges, tss, mass = triu(graph, k=1).tocoo(), float(np.square(x - x.mean(axis=0)).sum()), float(graph.sum()) / 2
    agreement = []
    for pos, seed in enumerate(summary["seeds"]):
        a = raw[f"seed_{seed}"].to_numpy(int)
        assert sorted(np.unique(a)) == [0, 1, 2, 3]
        overlap = np.zeros((4, 4), dtype=int)
        np.add.at(overlap, (z, a), 1)
        ri, ci = linear_sum_assignment(-overlap)
        mapping = np.empty(4, dtype=int)
        mapping[ci] = ri
        score = int(overlap[ri, ci].sum())
        # Independent brute force check of the Hungarian maximum, including ties.
        assert score == max(sum(overlap[p[c], c] for c in range(4)) for p in itertools.permutations(range(4)))
        b = mapping[a]
        np.testing.assert_array_equal(b, aligned[f"seed_{seed}"])
        np.testing.assert_array_equal(mapping, fits[pos]["mapping_raw_to_production"])
        np.testing.assert_array_equal(overlap, fits[pos]["overlap"])
        close(adjusted_rand_score(z, a), rows.iloc[pos].ari_with_production)
        close(np.mean(b == z), rows.iloc[pos].agreement_fraction)
        assert int(np.sum(b == z)) == rows.iloc[pos].agreement_count
        centers = np.array([x[a == c].mean(axis=0) for c in range(4)])
        objective = float(np.square(x - centers[a]).sum()) / tss + .5 * float(edges.data[a[edges.row] != a[edges.col]].sum()) / mass
        close(objective, rows.iloc[pos].objective)
        info = fits[pos]["fit"]
        close(objective, info["objective"])
        assert (info["seed"], info["alpha"], info["max_sweeps"]) == (seed, .5, 200)
        assert bool(rows.iloc[pos].converged) == info["converged"]
        assert rows.iloc[pos].sweeps == info["sweeps"] == len(info["moves_per_sweep"])
        assert info["convergence_status"] == rows.iloc[pos].convergence_status
        trace = [item["objective"] for item in info["trace"]]
        assert len(trace) == info["sweeps"] + 1 and np.all(np.diff(trace) <= 1e-10)
        if info["converged"]:
            assert info["moves_per_sweep"][-1] == 0 and info["convergence_status"] == "coordinate_fixed_point"
        agreement.append(b == z)
    counts = np.column_stack(agreement).sum(axis=1)
    np.testing.assert_array_equal(counts, support.agreement_count)
    assert (support.completed_seed_count == n).all() and (support.planned_seed_count == 20).all()
    close(counts / n, support.agreement_fraction_completed)
    if n == 20:
        close(counts / 20, support.agreement_over_20)
    else:
        assert support.agreement_over_20.isna().all()
    keys = ("metro", "industrial", "periphery", "remote")
    assert support.production_type_key.tolist() == [keys[c] for c in z]
    mat = aligned[[f"seed_{seed}" for seed in summary["seeds"]]].to_numpy(int)
    for c, key in enumerate(keys):
        np.testing.assert_array_equal((mat == c).sum(axis=1), support[f"aligned_{key}_count"])
    for key, value in metrics(counts / n).items():
        close(value, summary["overall"][key])
    groups = pd.read_csv(folder / "group_summary.csv")
    for c, group in enumerate(summary["groups"]):
        assert group["production_type"] == c and group["production_type_key"] == keys[c]
        assert group["municipalities"] == int(np.sum(z == c))
        for key, value in metrics(counts[z == c] / n).items():
            close(value, group[key])
            close(value, groups.iloc[c][key])
    close(rows.ari_with_production.min(), summary["seed_ari"]["min"])
    close(rows.ari_with_production.mean(), summary["seed_ari"]["mean"])
    close(rows.ari_with_production.max(), summary["seed_ari"]["max"])
    assert summary["converged_seed_count"] == int(rows.converged.sum())
    timing = read(folder / "timing.json")
    close(timing["forecast_20_seeds_seconds"], timing["setup_seconds"] + 20 * timing["first_fit_seconds"])
    assert (timing["forecast_20_seeds_seconds"] <= 600) == (timing["decision"] == "CONTINUE_ALL_20")
    print(json.dumps({"status": "PASS_READ_ONLY", "completed_seed_count": n, "municipalities": 2016,
                      "checks": ["source and artifact hashes", "all saved labels", "Hungarian maximum by 24 permutations", "ARI", "objectives", "convergence traces", "per-MO support", "group and overall summaries"],
                      "optimizer_fits": 0, "files_written": 0,
                      "historical_source_lineage_verified": bool(lineage)}, ensure_ascii=False))
    if args.check_pin_handling:
        print(json.dumps(check_pin_handling()))


if __name__ == "__main__":
    main()
