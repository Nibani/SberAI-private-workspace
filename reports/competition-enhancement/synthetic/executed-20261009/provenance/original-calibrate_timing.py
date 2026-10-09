"""Timing-only calibration on the two preregistered seeds; no quality scores."""
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path("C:/Users/Andreyka/Documents/Codex/2026-10-08/goal-goal-c-users-andreyka-documents-2/work/repo")
FOLDER = Path(__file__).parent
sys.path.insert(0, str(REPO))
from scripts import competition_synthetic as R

SEEDS = (44001, 44002)
MONTHS = (1, 12, 19, 31, 36)
ALPHAS = (.1, .25, .5, 1.)
BETAS = (.25, .5, 1., 2.)
EXPECTED_RUNNER = "f549c10b7ff5571795dda451fadcaa3e3e8eca43a0ae95220efdc76115b6d619"


def verify(regpath):
    reg = json.loads(regpath.read_text(encoding="utf-8"))
    if reg["files"]["scripts/competition_synthetic.py"] != EXPECTED_RUNNER:
        raise ValueError("Unexpected registered runner")
    for name, sha in reg["files"].items():
        if R.digest(REPO / name) != sha:
            raise ValueError("Frozen source changed: " + name)
    for name, sha in (("PLAN-original.md", reg["plan_sha256"]),
                      ("design-original.json", reg["design_sha256"]),
                      ("PREREG-ADDENDUM.md", reg["addendum_sha256"])):
        if R.digest(regpath.parent / name) != sha:
            raise ValueError("Registered plan changed: " + name)
    return reg


def forecast(rows, families):
    def rate(stage, statistic, **criteria):
        values = [row["wall_seconds"] for row in rows if row["stage"] == stage
                  and all(row.get(key) == value for key, value in criteria.items())]
        if not values:
            raise ValueError("Missing timing stratum: " + str((stage, criteria)))
        return float(R.np.mean(values) if statistic == "mean" else max(values))
    forecasts = []
    for alpha in ALPHAS:
        for beta in BETAS:
            estimates = {}
            for statistic in ("mean", "observed_max"):
                totals = {}
                for generator in (1, 2):
                    for strength in ("strong", "weak"):
                        for stage in ("kmeans50", "feature_hartigan"):
                            totals[stage] = totals.get(stage, 0) + 40 * rate(stage, statistic, generator=generator, strength=strength)
                    for family in families:
                        roots = 32 if family.endswith("_permuted") else 40
                        for stage in ("generation_static", "kefrin_like50"):
                            totals[stage] = totals.get(stage, 0) + roots * rate(stage, statistic, generator=generator, family=family)
                        for stage, grid, fixed, selected in (("joint", ALPHAS, .5, alpha), ("kefrine", BETAS, 1., beta)):
                            if not family.endswith("_permuted"):
                                totals[stage] = totals.get(stage, 0) + 8 * sum(rate(stage, statistic, generator=generator, family=family, parameter=value) for value in grid)
                            totals[stage] = totals.get(stage, 0) + 32 * sum(rate(stage, statistic, generator=generator, family=family, parameter=value) for value in sorted({fixed, selected}))
                        if family.startswith("weak") and not family.endswith("_permuted"):
                            totals["spectral"] = totals.get("spectral", 0) + 40 * rate("spectral", statistic, generator=generator, family=family)
                for strength in ("strong", "weak"):
                    for regime in (*R.REGIMES, *R.STRESS) if strength == "weak" else R.REGIMES:
                        totals["generation_temporal"] = totals.get("generation_temporal", 0) + 32 * rate("generation_temporal", statistic, strength=strength, regime=regime)
                        totals["monthly_kmeans20"] = totals.get("monthly_kmeans20", 0) + 32 * 36 * rate("monthly_kmeans20", statistic, strength=strength, regime=regime)
                estimates[statistic] = {"components_seconds": totals, "total_seconds": sum(totals.values())}
            a, b = 1 + (alpha != .5), 1 + (beta != 1.)
            forecasts.append({"possible_joint_alpha": alpha, "possible_kefrine_beta": beta,
                              "randomized_starts": 313600 + 25600 * (b - 1),
                              "optimizer_calls": 14560 + 512 * (a - 1) + 512 * (b - 1),
                              "alpha0_noop_calls_not_timed": 608, "rate_forecasts": estimates})
    return forecasts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--diagnostics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--slot-id", required=True)
    args = parser.parse_args()
    regpath = args.registration.resolve()
    reg = verify(regpath)
    diagnostic = json.loads(args.diagnostics.read_text(encoding="utf-8"))
    if diagnostic["registration_sha256"] != R.digest(regpath) or diagnostic["status"] != "COMPLETED_MECHANISTIC_DIAGNOSTICS":
        raise ValueError("Registered actual r0 must precede calibration")
    cfg = json.loads((regpath.parent / "design-original.json").read_text(encoding="utf-8"))
    prereg = json.loads((FOLDER / "calibration-preregistration.json").read_text(encoding="utf-8"))
    if prereg["helper_sha256"] != R.digest(__file__) or prereg["registration_sha256"] != R.digest(regpath):
        raise ValueError("Calibration preregistration mismatch")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    monitor = R.ResourceMonitor(args.output / "resource-monitor.jsonl", heavy=True)
    monitor.process.cpu_affinity([monitor.process.cpu_affinity()[0]])
    rows = []

    def measure(stage, operation, **criteria):
        gate_start = time.perf_counter()
        monitor.gate()
        begin = time.perf_counter()
        cpu_before = monitor.process.cpu_times()
        value = operation()
        cpu_after = monitor.process.cpu_times()
        row = {"stage": stage, **criteria, "wall_seconds": time.perf_counter() - begin,
               "cpu_seconds": cpu_after.user + cpu_after.system - cpu_before.user - cpu_before.system,
               "resource_gate_wait_seconds": begin - gate_start,
               "rss_bytes_after": monitor.process.memory_info().rss}
        rows.append(row)
        with (args.output / "timings.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(row) + "\n")
        return value

    started = time.perf_counter()
    try:
        monitor.start()
        R.numeric()
        with R.threadpool_limits(limits=1):
            R.write_json(args.output / "environment.json", {**R.environment(), "slot_id": args.slot_id,
                         "affinity": monitor.process.cpu_affinity(), "quality_metrics_computed": False})
            families = [family["id"] for family in cfg["families"]] + list(R.EXTRA)
            for root in SEEDS:
                cache = {}
                for generator in (1, 2):
                    for family in families:
                        strength = "strong" if family.startswith("strong") else "weak"
                        criteria = {"seed": root, "generator": generator, "strength": strength, "family": family}
                        def inputs():
                            if generator == 1:
                                x, latent = R.gaussian_data(root, strength)
                            else:
                                x, _, generated, _ = R.panel_data(root, strength, "stable")
                                latent = generated["base"]
                            return x, R.make_graph(x, latent, root, generator, family)
                        x, graph = measure("generation_static", inputs, **criteria)
                        seed = R.fit_seed(root, generator)
                        key = (generator, strength)
                        if key not in cache:
                            baseline = measure("kmeans50", lambda: R.KMeans(4, n_init=50, max_iter=300, random_state=seed).fit_predict(x), **criteria)
                            measure("feature_hartigan", lambda: R.feature_hartigan(x, baseline, seed), **criteria)
                            cache[key] = baseline
                        baseline = cache[key]
                        y = x - x.mean(axis=0)
                        network = R.A.binary(graph).toarray()
                        network -= network.mean()
                        weight = R.np.sqrt(R.np.square(y).sum() / R.np.square(network).sum())
                        combined = R.np.hstack([y, weight * network])
                        measure("kefrin_like50", lambda: R.KMeans(4, n_init=50, max_iter=300, random_state=seed).fit_predict(combined), **criteria)
                        for alpha in ALPHAS:
                            measure("joint", lambda alpha=alpha: R.fit_graph_regularized_kmeans(x, graph, baseline, alpha=alpha, seed=seed, max_sweeps=200), parameter=alpha, **criteria)
                        for beta in BETAS:
                            measure("kefrine", lambda beta=beta: R.fit_kefrine(x, graph, k=4, beta=beta, seed=seed, n_init=50, max_iter=300), parameter=beta, **criteria)
                        if family.startswith("weak") and not family.endswith("_permuted"):
                            measure("spectral", lambda: R.SpectralClustering(4, affinity="precomputed", assign_labels="cluster_qr", random_state=seed).fit_predict(graph), **criteria)
                for strength in ("strong", "weak"):
                    regimes = (*R.REGIMES, *R.STRESS) if strength == "weak" else R.REGIMES
                    for regime in regimes:
                        criteria = {"seed": root, "strength": strength, "regime": regime}
                        def temporal_input():
                            x, monthly, _, _ = R.panel_data(root, strength, regime)
                            return R.P.remove_national_wave(monthly, x)
                        monthly = measure("generation_temporal", temporal_input, **criteria)
                        for month in MONTHS:
                            measure("monthly_kmeans20", lambda month=month: R.KMeans(4, n_init=20, max_iter=300, random_state=R.fit_seed(root, 2)).fit_predict(monthly[:, month - 1]), month=month, **criteria)
                print("timing-only completed seed " + str(root), flush=True)
            result = {"status": "COMPLETED_TIMING_ONLY_PENDING_REVIEW", "seeds": list(SEEDS), "months": list(MONTHS),
                      "helper_sha256": R.digest(__file__), "registration_sha256": R.digest(regpath),
                      "diagnostics_sha256": R.digest(args.diagnostics), "duration_seconds": time.perf_counter() - started,
                      "quality_metrics_computed": False, "quality_based_selection": False,
                      "counts": {stage: sum(row["stage"] == stage for row in rows) for stage in sorted({row["stage"] for row in rows})},
                      "calibration_optimizer_calls": sum(row["stage"] not in ("generation_static", "generation_temporal") for row in rows),
                      "calibration_randomized_starts": 10400, "forecasts": forecast(rows, families),
                      "unmeasured_overhead": ["quality scoring", "bootstrap and sign-flip summaries", "output compression and checkpoints", "resource pauses during full run"],
                      "limitations": "Mean and observed-max rates from two fixed calibration seeds are engineering estimates, not runtime guarantees; all forecast branches are retained and no method is selected."}
            expected_counts = {"generation_static": 32, "generation_temporal": 20, "kmeans50": 8,
                               "feature_hartigan": 8, "kefrin_like50": 32, "joint": 128,
                               "kefrine": 128, "spectral": 16, "monthly_kmeans20": 100}
            if result["counts"] != expected_counts or result["calibration_optimizer_calls"] != 420:
                raise AssertionError("Calibration budget differs from preregistration")
            verify(regpath)
            R.write_json(args.output / "result.json", result)
            print(json.dumps({"status": result["status"], "duration_seconds": result["duration_seconds"],
                              "calibration_optimizer_calls": 420, "quality_metrics_computed": False}, indent=2))
    except Exception:
        R.write_json(args.output / "failure.json", {"status": "FAILED_TIMING_ONLY", "traceback": traceback.format_exc(),
                                                   "completed_timing_rows": len(rows), "failed_seed_not_removed": True})
        raise
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
