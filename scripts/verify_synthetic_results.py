"""Check saved scientific invariants and reaggregate all synthetic results; no fits.

Portable module default: reports/competition-enhancement/synthetic/executed-20261009.
Only saved 2023/2024 diagnostics and planted synthetic rows are read.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

for pool in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[pool] = "1"
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION_SHA = "8a94e00e8bfe1fe8823c3411fcf48ce066de578da1142bceb579ced37d7f2415"
FAMILIES = ("strong_noise", "weak_independent", "weak_derived", "weak_noise", "weak_conflict", "strong_derived")


def digest(path):
    """Hash requested bytes; a missing logical CSV falls back to inflated gzip."""
    result = hashlib.sha256()
    stored = saved_path(path)
    opener = gzip.open if stored != path else Path.open
    with opener(stored, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def saved_path(path):
    if path.is_file():
        return path
    compressed = Path(str(path) + ".gz")
    if path.suffix != ".gz" and compressed.is_file():
        return compressed
    raise FileNotFoundError(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path):
    stored = saved_path(path)
    opener = gzip.open if stored.suffix == ".gz" else Path.open
    with opener(stored, mode="rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual, expected, label):
    if not math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-12):
        raise AssertionError(f"{label}: {actual} != {expected}")


def mean(values):
    return math.fsum(values) / len(values)


def quantile(values, p):
    ordered = sorted(float(x) for x in values)
    at = (len(ordered) - 1) * p
    lo = int(at)
    part = at - lo
    return ordered[lo] if part == 0 else ordered[lo] * (1 - part) + ordered[lo + 1] * part


def unique_index(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row[name] for name in fields)
        if key in result:
            raise AssertionError("Duplicate scientific row: " + str(key))
        result[key] = row
    return result


def verify(artifact, repo):
    manifest = read(artifact / "manifest.json")
    if manifest["status"] != "COMPLETE_ORIGINAL_REGISTERED_PLAN":
        raise AssertionError("The manifest does not declare a complete original plan")
    for name, expected in manifest["files_sha256"].items():
        relative = Path(name)
        path = (artifact / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(artifact.resolve()):
            raise AssertionError("Manifest path escapes artifact directory: " + name)
        if digest(path) != expected:
            raise AssertionError("Evidence hash mismatch: " + name)
    for name, expected in manifest.get("logical_files_sha256", {}).items():
        relative = Path(name)
        path = (artifact / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(artifact.resolve()):
            raise AssertionError("Logical manifest path escapes artifact directory: " + name)
        if digest(path) != expected:
            raise AssertionError("Restored evidence hash mismatch: " + name)
    regpath = artifact / "provenance/registration.json"
    if digest(regpath) != REGISTRATION_SHA:
        raise AssertionError("Original preregistration bytes changed")
    reg = read(regpath)
    cfg = read(artifact / "provenance/design-original.json")
    for name, expected in (("PLAN-original.md", reg["plan_sha256"]), ("design-original.json", reg["design_sha256"]), ("PREREG-ADDENDUM.md", reg["addendum_sha256"])):
        if digest(artifact / "provenance" / name) != expected:
            raise AssertionError("Original plan bytes changed: " + name)
    for name, expected in reg["files"].items():
        if digest(repo / name) != expected:
            raise AssertionError("Registered implementation changed: " + name)
    assert cfg["seeds"]["tune"] == list(range(11001, 11009))
    roots = cfg["seeds"]["evaluation"]
    assert roots == list(range(22001, 22033))
    assert tuple(f["id"] for f in cfg["families"]) == FAMILIES
    assert cfg["budget"]["n_init"] == 50 and cfg["budget"]["bootstrap_draws"] == 10000
    result = read(artifact / "full-run/result.json")
    assert result["status"] == "COMPLETED_SCIENTIFIC_RUN_PENDING_INDEPENDENT_REVIEW"
    assert result["all_evaluation_seeds"] == roots
    assert result["registration_sha256"] == digest(regpath)
    assert result["diagnostics_sha256"] == digest(artifact / "diagnostics/diagnostics.json")
    assert result["selected_sha256"] == digest(artifact / "full-run/selected.json")
    assert read(artifact / "full-run/tune/checkpoint.json")["completed_seeds"] == cfg["seeds"]["tune"]
    assert read(artifact / "full-run/evaluation/checkpoint.json")["completed_seeds"] == roots
    selected = read(artifact / "full-run/selected.json")
    assert selected["tune_metrics_sha256"] == digest(artifact / "full-run/tune/static-metrics.csv")
    assert selected["tune_seeds"] == cfg["seeds"]["tune"]
    decision = read(artifact / "provenance/execution-decision.json")
    assert decision["quality_outcomes_before_decision"] == 0
    assert decision["decision"] == "EXECUTE_ORIGINAL_FULL_PLAN_UNCHANGED"

    projection = csv_rows(artifact / "diagnostics/real-projection.csv")
    diagnostic = read(artifact / "diagnostics/diagnostics.json")
    assert len(projection) == diagnostic["n"] == 2016 and diagnostic["k"] == 4
    assert len({row["entity_id"] for row in projection}) == 2016
    disagreements = sum(row["projection_disagreement"] == "True" for row in projection)
    assert disagreements == diagnostic["r0_count"] == diagnostic["identical_profile_persistent_disagreement_count"]
    assert all((row["annual_type"] != row["projected_type"]) == (row["projection_disagreement"] == "True") for row in projection)
    assert all(row["persistent_disagreement_on_identical_profile"] == row["projection_disagreement"] for row in projection)
    close(disagreements / 2016, diagnostic["r0_projection_disagreement"], "real baseline projection rate")

    tune = csv_rows(artifact / "full-run/tune/static-metrics.csv")
    static = csv_rows(artifact / "full-run/evaluation/static-metrics.csv")
    temporal = csv_rows(artifact / "full-run/evaluation/temporal-metrics.csv")
    assert len(tune) == 1408 and len(static) == 4352 and len(temporal) == 29952
    sidx = unique_index(static, ("generator", "seed", "family", "method"))
    unique_index(tune, ("generator", "seed", "family", "method"))
    tidx = unique_index(temporal, ("seed", "family", "regime", "method", "variant"))
    assert {int(row["seed"]) for row in tune} == set(cfg["seeds"]["tune"])
    assert {int(row["seed"]) for row in static} == {int(row["seed"]) for row in temporal} == set(roots)
    for prefix, candidates, field in (("joint", (.1, .25, .5, 1.), "joint_alpha"), ("kefrine", (.25, .5, 1., 2.), "kefrine_beta")):
        means = {}
        for weight in candidates:
            values = [float(row["ari"]) for row in tune if row["method"] == f"{prefix}_{weight:g}"]
            assert len(values) == 96
            means[weight] = mean(values)
            close(means[weight], selected[prefix + "_candidate_means"][str(weight).removesuffix(".0")], "selected candidate mean")
        winner = min(weight for weight, score in means.items() if score >= max(means.values()) - 1e-12)
        assert selected[field] == winner

    def svalue(g, root, family, method):
        return float(sidx[(str(g), str(root), family, method)]["ari"])

    def tvalue(root, regime, method, metric):
        return float(tidx[(str(root), "weak_derived", regime, method, "adjusted")][metric])

    vectors = {name: [] for name in ("C1", "C2", "C3", "C4")}
    for root in roots:
        vectors["C1"].append(mean([(svalue(g, root, "weak_independent", "joint_alpha_0.5") - svalue(g, root, "weak_independent", "kmeans50")) - (svalue(g, root, "weak_derived", "joint_alpha_0.5") - svalue(g, root, "weak_derived", "kmeans50")) for g in (1, 2)]))
        vectors["C2"].append(mean([svalue(g, root, family, "kefrine_equations_equal_scatter50") - svalue(g, root, family, "kefrin_like_equal_scatter50") for g in (1, 2) for family in FAMILIES]))
        vectors["C3"].append(mean([tvalue(root, regime, "joint_alpha_0.5", "recall") - tvalue(root, regime, "kmeans50", "recall") for regime in ("persistent_transition", "transition_and_shift")]))
        vectors["C4"].append(tvalue(root, "common_shift", "joint_alpha_0.5", "fp_fraction") - tvalue(root, "common_shift", "kmeans50", "fp_fraction"))
    contrasts = unique_index(csv_rows(artifact / "full-run/report/contrasts.csv"), ("comparison", "seed"))
    assert len(contrasts) == 128
    for name, values in vectors.items():
        for root, value in zip(roots, values):
            close(value, contrasts[(name, str(root))]["difference"], "raw contrast")
    rng = np.random.default_rng(cfg["seeds"]["bootstrap"])
    indexes = rng.integers(0, len(roots), (10000, len(roots)))
    signs = rng.choice((-1, 1), (10000, len(roots)))
    intervals = unique_index(csv_rows(artifact / "full-run/report/confidence-intervals.csv"), ("comparison",))
    assert len(intervals) == 4
    pvalues = {}
    for name, values in vectors.items():
        observed = mean(values)
        samples = [mean([values[i] for i in draw]) for draw in indexes]
        extremes = sum(abs(mean([sign * value for sign, value in zip(draw, values)])) >= abs(observed) - 1e-15 for draw in signs)
        pvalues[name] = (1 + extremes) / 10001
        row = intervals[(name,)]
        for field, value in (("mean", observed), ("median", quantile(values, .5)), ("ci95_low", quantile(samples, .025)), ("ci95_high", quantile(samples, .975)), ("signflip_p", pvalues[name])):
            close(value, row[field], name + " " + field)
        reported = next(item for item in result["comparisons"] if item["comparison"] == name)
        for field in ("mean", "median", "ci95_low", "ci95_high", "signflip_p", "holm_adjusted_p"):
            close(row[field], reported[field], "result and interval CSV agree")
    previous = 0.
    for rank, name in enumerate(sorted(pvalues, key=pvalues.get)):
        previous = max(previous, min(1., (4 - rank) * pvalues[name]))
        close(previous, intervals[(name,)]["holm_adjusted_p"], "Holm correction")

    mechanisms = unique_index(csv_rows(artifact / "full-run/report/mechanistic-comparisons.csv"), ("generator", "family", "comparison"))
    assert len(mechanisms) == 20
    for key, row in mechanisms.items():
        generator, family, comparison = key
        if comparison == "joint_fixed_minus_feature_only_exact_hartigan":
            values = [svalue(generator, root, family, "joint_alpha_0.5") - svalue(generator, root, family, "feature_hartigan") for root in roots]
        elif comparison == "joint_fixed_original_minus_same_W_vertex_permutation":
            values = [svalue(generator, root, family, "joint_alpha_0.5") - svalue(generator, root, family + "_permuted", "joint_alpha_0.5") for root in roots]
        else:
            raise AssertionError("Unexpected mechanistic comparison: " + comparison)
        samples = [mean([values[i] for i in draw]) for draw in indexes]
        close(mean(values), row["mean"], "mechanistic mean")
        close(quantile(samples, .025), row["ci95_low"], "mechanistic lower CI")
        close(quantile(samples, .975), row["ci95_high"], "mechanistic upper CI")
        assert row["inference"] == "preregistered_mechanistic_descriptive_comparison"

    all_groups = defaultdict(list)
    static_metrics = ("ari", "projection_ari", "projection_disagreement", "min_group_size")
    temporal_metrics = ("mean_monthly_ari", "state_accuracy", "macro_state_accuracy", "changed_destination_accuracy", "unchanged_state_accuracy", "post_change_destination_accuracy", "recall", "fp_fraction", "capped_delay_mean")
    for source, fields in ((static, static_metrics), (temporal, temporal_metrics)):
        for row in source:
            for metric in fields:
                if row[metric] != "":
                    value = float(row[metric])
                    assert math.isfinite(value)
                    key = (row.get("generator", ""), row["family"], row.get("regime", ""), row["method"], row.get("variant", ""), metric)
                    all_groups[key].append(value)
    summaries = unique_index(csv_rows(artifact / "full-run/report/summary.csv"), ("generator", "family", "regime", "method", "variant", "metric"))
    assert set(summaries) == set(all_groups)
    for key, values in all_groups.items():
        row = summaries[key]
        assert len(values) == int(row["n_seeds"]) == 32
        average = mean(values)
        close(average, row["mean"], "summary mean")
        close(math.sqrt(math.fsum((x - average) ** 2 for x in values) / (len(values) - 1)), row["sd_between_seeds"], "summary SD")
        close(min(values), row["min"], "summary minimum")
        close(max(values), row["max"], "summary maximum")
    calibration = read(artifact / "timing-only-v1/result.json")
    assert calibration["quality_metrics_computed"] is False and calibration["quality_based_selection"] is False
    assert calibration["calibration_optimizer_calls"] == 420 and calibration["calibration_randomized_starts"] == 10400
    assert len(calibration["forecasts"]) == 16
    return {"status": "PASS_SYNTHETIC_SAVED_RESULTS", "optimizer_fits": 0,
            "verified_evidence_files": len(manifest["files_sha256"]), "tune_seeds": 8, "evaluation_seeds": 32,
            "verified_logical_files": len(manifest.get("logical_files_sha256", {})),
            "raw_static_rows": len(tune) + len(static), "raw_temporal_rows": len(temporal),
            "contrasts_checked": 128, "confirmatory_intervals_checked": 4, "mechanistic_intervals_checked": 20,
            "summary_rows_checked": len(summaries), "projection_disagreements_on_constant_profile": disagreements,
            "scope": "Hash and saved-row numerical verification; no fitting or 2025 outcome access."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check saved results; this is also the default")
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "reports/competition-enhancement/synthetic/executed-20261009")
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args()
    result = verify(args.artifact_dir.resolve(), args.repo.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
