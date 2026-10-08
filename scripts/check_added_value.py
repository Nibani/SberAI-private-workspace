"""Do the network groups add information about local economies beyond the spending level?

    python -m scripts.check_added_value                 # writes reports/v1.2-added-value
    python -m scripts.check_added_value --check         # recompute and compare with the saved tables

For every one of the 73 saved strict region folds (reports/v1.2/strict-region-validation)
the fold's training state is refitted with the recorded recipe and verified against the
saved scaler, level reference, centres, naming, graph and objective; the held-out region therefore never enters the scaler, the monthly
level reference, the graph, the groups or their naming. Held-out municipalities are
assigned to the training centres, and the saved held-out assignment is checked.

For each of the eight external indicators the same observed municipalities are predicted
out of region by eight least-squares models fitted on the other regions only:

    groups             intercept + group dummies (reference; equals the published strict check)
    level              intercept + spending level (6th v1.2 attribute, training scale)
    level_spline       restricted cubic spline of the level, knots at training quantiles
    level_groups       level + group dummies
    spline_groups      spline + group dummies
    level_shares       level + five continuous category coordinates
    spline_shares      spline + five continuous category coordinates
    spline_shares_groups  spline + coordinates + group dummies

Contrasts are paired: per municipality squared-error differences, summarised as the change
in mean squared error and in out-of-region R^2. Inference resamples whole regions: symmetric
bootstrap-t (main) and percentile bootstrap (sensitivity); a null simulation with the real
region sizes is saved as calibration.csv, and drop-one-region estimates show influence. Each contrast forms a family of eight indicators; bootstrap p-values are
adjusted with Holm's method inside the family (and, as a sensitivity check, over all tests)
and simultaneous intervals use Bonferroni. The order in which models and contrasts entered
the analysis is recorded in the generated README and in CONTRASTS.
The eight indicators were all inspected together, so a positive result is exploratory,
not a pre-registered confirmation. 2025 outcomes are not opened.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from dataclasses import replace
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_v12_findings import digest, fit_training, frozen_attributes, read_external, subset  # noqa: E402
from sbercluster import panel as P  # noqa: E402
from sbercluster.graph import knn_graph  # noqa: E402

FOLDS = ROOT / "reports/v1.2/strict-region-validation"
OUTPUT = ROOT / "reports/v1.2-added-value"
MODELS = ("groups", "level", "level_spline", "level_groups", "spline_groups", "level_shares", "spline_shares",
          "spline_shares_groups")
CONTRASTS = {
    # name: (candidate, baseline, role, status). Status records when the contrast entered the analysis.
    "groups_over_spline": ("spline_groups", "level_spline", "основное: группы сверх нелинейного уровня",
                           "primary, defined before the first full run"),
    "shares_over_spline": ("spline_shares", "level_spline", "непрерывные доли сверх нелинейного уровня",
                           "secondary, defined before the first full run"),
    "groups_vs_shares": ("spline_groups", "spline_shares", "группы против непрерывных долей (оба со сплайном)",
                         "secondary, defined before the first full run"),
    "groups_over_shares": ("spline_shares_groups", "spline_shares", "группы сверх сплайна и непрерывных долей",
                           "added after the first full run"),
    "groups_over_level": ("level_groups", "level", "справочно: группы сверх линейного уровня",
                          "reference, defined before the first full run"),
    "spline_over_level": ("level_spline", "level", "справочно: нелинейность уровня",
                          "reference, defined before the first full run"),
}
STATUS_RU = {"primary, defined before the first full run": "основное, задано до первого полного прогона",
             "secondary, defined before the first full run": "дополнительное, задано до первого полного прогона",
             "reference, defined before the first full run": "справочное, задано до первого полного прогона",
             "added after the first full run": "добавлено после первого полного прогона"}
LABELS = {
    "log_wage": "Логарифм зарплаты", "agri_share": "Доля сельского хозяйства",
    "manuf_share": "Доля обрабатывающей промышленности", "mining_share": "Доля добычи",
    "public_share": "Доля занятых в бюджетных отраслях", "log_pop": "Логарифм населения",
    "log_jobs_per_res": "Логарифм рабочих мест на жителя", "market_access": "Доступность рынков",
}
QUANTILES = (0.05, 0.35, 0.65, 0.95)


# ------------------------------------------------------------------ statistics
def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted (natural) cubic spline basis, Harrell's parameterisation, without intercept.

    Columns: x and K-2 nonlinear terms; the fit is linear beyond the outer knots.
    """
    x = np.asarray(x, dtype=float)
    k = np.asarray(knots, dtype=float)
    if len(k) < 3 or np.any(np.diff(k) <= 0):
        raise ValueError("At least three increasing knots are required")
    scale = (k[-1] - k[0]) ** 2
    cube = lambda v: np.maximum(v, 0.0) ** 3  # noqa: E731
    cols = [x]
    for j in range(len(k) - 2):
        term = (cube(x - k[j]) - cube(x - k[-2]) * (k[-1] - k[j]) / (k[-1] - k[-2])
                + cube(x - k[-1]) * (k[-2] - k[j]) / (k[-1] - k[-2]))
        cols.append(term / scale)
    return np.column_stack(cols)


def design(model: str, level: np.ndarray, shares: np.ndarray, groups: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Intercept plus the blocks named in the model: level|spline, shares, groups."""
    blocks = set(model.split("_"))
    parts = [np.ones((len(level), 1))]
    if "spline" in blocks:
        parts.append(rcs_basis(level, knots))
    elif "level" in blocks:
        parts.append(level[:, None])
    if "shares" in blocks:
        parts.append(shares)
    if "groups" in blocks:
        parts.append(np.eye(4)[groups][:, 1:])
    return np.hstack(parts)


def holm(pvalues) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def region_picks(n_regions: int, draws: int, seed: int) -> np.ndarray:
    """Cluster bootstrap draws: each row picks n_regions region indices with replacement."""
    return np.random.default_rng(seed).integers(0, n_regions, size=(draws, n_regions))


def resampled_sums(values: np.ndarray, codes: np.ndarray, n_regions: int, picks: np.ndarray) -> np.ndarray:
    """Sum of a per-municipality quantity over the regions in each draw (a region drawn twice counts twice)."""
    per_region = np.bincount(codes, weights=values, minlength=n_regions)
    return per_region[picks].sum(axis=1)


def studentized(d: np.ndarray, codes: np.ndarray, n_regions: int, picks: np.ndarray) -> dict:
    """Region-cluster bootstrap-t for the municipality-weighted mean of d.

    The standard error is the cluster-robust (linearised ratio) one,
    se = sqrt(R/(R-1) * sum_r (S_r - m n_r)^2) / N, with S_r and n_r the sum of d and the number
    of municipalities in region r. In every draw it is recomputed from the drawn regions and
    t* = (m* - m) / se*. The two-sided p is (1 + #{|t*| >= |t|}) / (B + 1); the matching
    symmetric interval is m +/- q * se with q the (1 - alpha) quantile of |t*|, so p < alpha
    exactly when the interval excludes zero, up to the +1 term and quantile interpolation.
    """
    k = n_regions
    s_r = np.bincount(codes, weights=d, minlength=k)
    n_r = np.bincount(codes, minlength=k).astype(float)
    total = n_r.sum()
    m = s_r.sum() / total
    se = float(np.sqrt(k / (k - 1) * np.sum((s_r - m * n_r) ** 2)) / total)
    n_b = n_r[picks].sum(axis=1)
    m_b = s_r[picks].sum(axis=1) / n_b
    q_b = (s_r ** 2)[picks].sum(axis=1) - 2 * m_b * (s_r * n_r)[picks].sum(axis=1) + m_b ** 2 * (n_r ** 2)[picks].sum(axis=1)
    se_b = np.sqrt(k / (k - 1) * np.maximum(q_b, 0)) / n_b
    with np.errstate(divide="ignore", invalid="ignore"):
        t_b = np.where(se_b > 0, (m_b - m) / se_b, 0.0)
        t = m / se if se > 0 else 0.0
    p = float(min(1.0, (1 + np.sum(np.abs(t_b) >= abs(t))) / (len(t_b) + 1)))
    return {"mean": float(m), "se": se, "t": float(t), "p": p, "abs_t": np.abs(t_b), "boot_mean": m_b}


def bootstrap_contrast(observed: np.ndarray, candidate: np.ndarray, baseline: np.ndarray,
                       regions: np.ndarray, picks: np.ndarray, families: int = 8) -> dict:
    """Paired change in MSE and in out-of-region R^2 with region-cluster bootstrap inference.

    Every municipality has weight one, so a resampled region enters with its number of
    municipalities and the statistic of a draw is the municipality-weighted mean over the drawn
    regions (the same estimator as the point estimate).

    Main inference: symmetric bootstrap-t (see `studentized`); the R^2 interval is the delta-MSE
    interval divided by -Var(y) of the evaluated municipalities, i.e. on the fixed scale of the
    reported R^2. Sensitivity: percentile p and percentile R^2 intervals in which Var(y) is
    recomputed in every draw.
    """
    _, codes = np.unique(regions, return_inverse=True)
    k = picks.shape[1]
    if codes.max() + 1 != k:
        raise ValueError("picks must index the regions present in this indicator")
    d = (candidate - observed) ** 2 - (baseline - observed) ** 2
    var = float(np.mean((observed - observed.mean()) ** 2))
    st = studentized(d, codes, k, picks)
    out = {"delta_mse": st["mean"], "delta_mse_se": st["se"], "t": st["t"], "delta_r2": -st["mean"] / var,
           "p_boot_t": st["p"]}
    for suffix, level in (("ci", 0.95), ("simultaneous", 1 - 0.05 / families)):
        q = float(np.quantile(st["abs_t"], level))
        out[f"delta_mse_{suffix}_low"], out[f"delta_mse_{suffix}_high"] = st["mean"] - q * st["se"], st["mean"] + q * st["se"]
        out[f"delta_r2_{suffix}_low"] = -(st["mean"] + q * st["se"]) / var
        out[f"delta_r2_{suffix}_high"] = -(st["mean"] - q * st["se"]) / var
    n = resampled_sums(np.ones_like(d), codes, k, picks)
    mean_y = resampled_sums(observed, codes, k, picks) / n
    var_b = resampled_sums(observed ** 2, codes, k, picks) / n - mean_y ** 2
    r2_b = -st["boot_mean"] / var_b
    out["p_percentile"] = bootstrap_p(st["boot_mean"])
    out["delta_r2_pct_ci_low"], out["delta_r2_pct_ci_high"] = percentile_interval(r2_b, 0.95)
    out["delta_r2_pct_simultaneous_low"], out["delta_r2_pct_simultaneous_high"] = percentile_interval(r2_b, 1 - 0.05 / families)
    return out


def region_influence(observed: np.ndarray, candidate: np.ndarray, baseline: np.ndarray, regions: np.ndarray) -> dict:
    """Delta R^2 recomputed without each region in turn (descriptive, not an inference)."""
    names, codes = np.unique(regions, return_inverse=True)
    k = len(names)
    d = (candidate - observed) ** 2 - (baseline - observed) ** 2
    n_r = np.bincount(codes, minlength=k).astype(float)
    s_d, s_y, s_y2 = (np.bincount(codes, weights=v, minlength=k) for v in (d, observed, observed ** 2))
    n = n_r.sum() - n_r
    mean_y = (s_y.sum() - s_y) / n
    var = (s_y2.sum() - s_y2) / n - mean_y ** 2
    drop = -((s_d.sum() - s_d) / n) / var
    full = -d.mean() / np.mean((observed - observed.mean()) ** 2)
    top = int(np.argmax(np.abs(drop - full)))
    return {"drop_one_region_min": float(drop.min()), "drop_one_region_max": float(drop.max()),
            "most_influential_region": str(names[top]), "delta_r2_without_it": float(drop[top]),
            "its_municipalities": int(n_r[top]), "its_share_of_delta_mse": float(s_d[top] / d.sum()) if d.sum() else np.nan}


def calibration(pred: pd.DataFrame, sims: int = 1000, draws: int = 2000, seed: int = 20261008) -> pd.DataFrame:
    """Null simulation with the real region sizes: rejection rates of both bootstrap p-values.

    d = u_region + e, E d = 0, with region effects and municipality noise normal, Student t(3)
    or centred exponential (skewed). Independent of the observed outcomes and predictions.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for col in ("log_wage", "mining_share"):
        sizes = pred[pred.indicator == col].groupby("region").size().to_numpy()
        k = len(sizes)
        codes = np.repeat(np.arange(k), sizes)
        for dist in ("normal", "t3", "skewed"):
            hits = np.zeros(4)
            for _ in range(sims):
                if dist == "normal":
                    u, e = rng.normal(size=k) * 0.5, rng.normal(size=len(codes))
                elif dist == "t3":
                    u, e = rng.normal(size=k) * 0.5, rng.standard_t(3, size=len(codes))
                else:
                    u, e = (rng.exponential(size=k) - 1) * 0.5, rng.exponential(size=len(codes)) - 1
                picks = rng.integers(0, k, size=(draws, k))
                st = studentized(u[codes] + e, codes, k, picks)
                pp = bootstrap_p(st["boot_mean"])
                hits += [st["p"] < 0.05, st["p"] < 0.05 / 8, pp < 0.05, pp < 0.05 / 8]
            rows.append({"region_sizes_of": col, "regions": k, "municipalities": len(codes), "noise": dist,
                         "simulations": sims, "draws": draws, "boot_t_rate_0.05": hits[0] / sims,
                         "boot_t_rate_0.00625": hits[1] / sims, "percentile_rate_0.05": hits[2] / sims,
                         "percentile_rate_0.00625": hits[3] / sims})
    return pd.DataFrame(rows)


def percentile_interval(samples: np.ndarray, level: float) -> tuple[float, float]:
    """Equal-tailed percentile interval (numpy linear interpolation between order statistics)."""
    tail = 100 * (1 - level) / 2
    return float(np.percentile(samples, tail)), float(np.percentile(samples, 100 - tail))


def bootstrap_p(samples: np.ndarray) -> float:
    """Two-sided percentile-inversion p-value for H0: statistic = 0.

    p = min(1, 2 * min((1 + #{T* <= 0}) / (B + 1), (1 + #{T* >= 0}) / (B + 1))).
    Draws exactly equal to zero count on both sides (conservative); the +1 terms keep p > 0,
    so the smallest attainable value is 2 / (B + 1). It is the smallest two-sided level at
    which the equal-tailed percentile interval excludes zero, up to the +1 and interpolation.
    """
    samples = np.asarray(samples, dtype=float)
    b = len(samples)
    below = (1 + np.sum(samples <= 0)) / (b + 1)
    above = (1 + np.sum(samples >= 0)) / (b + 1)
    return float(min(1.0, 2 * min(below, above)))


# ------------------------------------------------------------------ out-of-region predictions
RTOL, ATOL = 1e-10, 1e-12
# (name in the refitted state, path in fold_XXX.json)
NUMERIC_COMPONENTS = (("structure_center", ("scaler", "structure_center")), ("scale", ("scaler", "scale")),
                      ("level_center", ("scaler", "level_center")),
                      ("monthly_log_total_center", ("monthly_log_total_center",)), ("centers", ("centers",)),
                      ("graph_sigma", ("graph", "sigma")), ("graph_median_strength", ("graph", "median_strength")),
                      ("objective", ("fit", "objective")))
EXACT_COMPONENTS = ("naming", "graph_edges", "graph_components", "fit_sweeps", "train_n")


def state_values(state: dict) -> dict:
    sc = state["scaler"]
    return {"structure_center": sc.structure_center, "scale": sc.scale, "level_center": sc.level_center,
            "monthly_log_total_center": state["monthly_center"], "centers": state["centers"],
            "graph_sigma": state["graph_info"]["sigma"], "graph_median_strength": state["graph_info"]["median_strength"],
            "objective": state["fit_info"]["objective"], "naming": state["naming"],
            "graph_edges": state["graph_info"]["edges"], "graph_components": state["graph_info"]["components"],
            "fit_sweeps": state["fit_info"]["sweeps"], "train_n": len(state["labels"])}


def stored_values(stored: dict) -> dict:
    out = {}
    for name, path in NUMERIC_COMPONENTS:
        value = stored
        for key in path:
            value = value[key]
        out[name] = value
    out.update(naming=stored["naming"], graph_edges=stored["graph"]["edges"],
               graph_components=stored["graph"]["components"], fit_sweeps=stored["fit"]["sweeps"],
               train_n=stored["train_n"])
    return out


def verify_state(state: dict, stored: dict, fold: int, rtol: float = RTOL, atol: float = ATOL) -> dict:
    """Compare a refitted training state with the saved fold; return per-component deviations.

    Fold JSON stores floats with Python's shortest round-trip repr, so a stored float64 is read
    back exactly and a zero deviation means bitwise equality of that component. Arrays that are
    inside the saved state hash but not stored separately (training attributes, graph CSR,
    training labels) cannot be compared here; their own SHA-256 are reported by component_hashes.
    """
    actual, expected = state_values(state), stored_values(stored)
    diag = {}
    for name, _ in NUMERIC_COMPONENTS:
        a = np.asarray(actual[name], float)
        e = np.asarray(expected[name], float)
        if a.shape != e.shape:
            raise ValueError(f"Refitted {name} has shape {a.shape}, saved fold {fold} has {e.shape}")
        dev = np.abs(a - e)
        diag[f"{name}_max_abs"] = float(dev.max())
        diag[f"{name}_max_rel"] = float((dev / np.maximum(np.abs(e), np.finfo(float).tiny)).max())
        diag[f"{name}_bitwise_equal"] = bool(np.array_equal(a, e))
        if not np.allclose(a, e, rtol=rtol, atol=atol):
            raise ValueError(f"Refitted {name} differs from the saved fold {fold}")
    for name in EXACT_COMPONENTS:
        diag[f"{name}_equal"] = actual[name] == expected[name]
        if not diag[f"{name}_equal"]:
            raise ValueError(f"Refitted {name} differs from the saved fold {fold}")
    return diag


def component_hashes(state: dict) -> dict:
    """SHA-256 of each array that enters the saved state hash, in the same digest format.

    Lets another machine whose full state hash matches the saved one locate which component
    differs here.
    """
    w = knn_graph(state["x"], k=15)[0]
    sc = state["scaler"]
    parts = {"structure_center": (sc.structure_center,), "scale": (sc.scale,), "monthly_center": (state["monthly_center"],),
             "x": (state["x"],), "graph_csr": (w.indptr, w.indices, w.data), "labels": (state["labels"],),
             "centers": (state["centers"],)}
    full = digest(sc.structure_center, sc.scale, state["monthly_center"], state["x"], w.indptr, w.indices, w.data,
                  state["labels"], state["centers"])
    if full != state["hash"]:
        raise ValueError("Recomputed graph does not reproduce the refitted state hash")
    out = {f"sha256_{k}": digest(*v) for k, v in parts.items()}
    # Platform integer width is a common cause of unequal digests; record whether it explains the saved one.
    variants = set()
    for lab in ("int32", "int64"):
        for idx in ("int32", "int64"):
            variants.add(digest(sc.structure_center, sc.scale, state["monthly_center"], state["x"],
                                w.indptr.astype(idx), w.indices.astype(idx), w.data, state["labels"].astype(lab),
                                state["centers"]))
    out["dtypes"] = f"indptr={w.indptr.dtype},indices={w.indices.dtype},data={w.data.dtype},labels={state['labels'].dtype}"
    return out, variants


def predictions(source: Path, max_regions: int | None = None) -> tuple[pd.DataFrame, list[dict]]:
    cfg = json.loads((source / "configs/v12.json").read_text(encoding="utf-8"))
    panel = P.read_panel(source / cfg["inputs"]["panel"])
    panel = replace(panel, totals=panel.totals[:, :12], shares=panel.shares[:, :12], periods=panel.periods[:12])
    ext, _ = read_external(source, panel.ids, cfg)
    saved = pd.read_csv(source / "reports/v1.2/strict-region-validation/strict_assignments.csv").set_index("entity_id")
    rows, folds = [], []
    for f, region in enumerate(np.unique(panel.regions)[:max_regions]):
        stored = json.loads((source / f"reports/v1.2/strict-region-validation/fold_{f:03d}.json").read_text(encoding="utf-8"))
        if stored["region"] != region:
            raise ValueError(f"Fold {f} belongs to {stored['region']}, expected {region}")
        train, test = panel.regions != region, panel.regions == region
        if hashlib.sha256("\n".join(panel.ids[train]).encode()).hexdigest() != stored["train_ids_sha256"]:
            raise ValueError(f"Training membership differs from the saved fold {f}")
        state = fit_training(panel, train)
        diag = verify_state(state, stored, f)
        hashes, variants = component_hashes(state)
        x_train = state["x"]
        x_test = frozen_attributes(subset(panel, test), state["scaler"], state["monthly_center"])
        g_train = state["labels"]
        g_test = np.argmin(np.square(x_test[:, None, :] - state["centers"][None]).sum(axis=2), axis=1)
        heldout_equal = np.array_equal(g_test, saved.loc[panel.ids[test], "strict_type"].to_numpy(int))
        if not heldout_equal:
            raise ValueError(f"Held-out assignment differs from strict_assignments.csv in fold {f}")
        knots = np.quantile(x_train[:, 5], QUANTILES)
        test_pos = np.where(test)[0]
        for col in ext.columns:
            y = ext[col].to_numpy(float)
            ok_tr, ok_te = np.isfinite(y[train]), np.isfinite(y[test])
            if not ok_te.any():
                continue
            row_preds = {}
            for model in MODELS:
                d_tr = design(model, x_train[ok_tr, 5], x_train[ok_tr, :5], g_train[ok_tr], knots)
                d_te = design(model, x_test[ok_te, 5], x_test[ok_te, :5], g_test[ok_te], knots)
                beta = np.linalg.lstsq(d_tr, y[train][ok_tr], rcond=None)[0]
                row_preds[model] = d_te @ beta
            for j, pos in enumerate(test_pos[ok_te]):
                rows.append({"entity_id": str(panel.ids[pos]), "region": str(region), "indicator": col,
                             "observed": float(y[pos]), "group": int(g_test[ok_te][j]),
                             **{m: float(row_preds[m][j]) for m in MODELS}})
        folds.append({"fold": f, "region": str(region), "train_n": int(train.sum()), "test_n": int(test.sum()),
                      "state_verified": True, "heldout_assignments_equal": heldout_equal,
                      "state_sha256_saved": stored["state_sha256"], "state_sha256_here": state["hash"],
                      "state_sha256_bitwise_equal": state["hash"] == stored["state_sha256"],
                      "saved_hash_explained_by_int_width": stored["state_sha256"] in variants,
                      **hashes, **diag, "knots": [float(k) for k in knots]})
    return pd.DataFrame(rows), folds


# ------------------------------------------------------------------ summaries
def summarise(pred: pd.DataFrame, draws: int, seed: int, total_entities: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_rows, contrast_rows = [], []
    for col in LABELS:
        part = pred[pred.indicator == col]
        y = part.observed.to_numpy()
        var = float(np.mean((y - y.mean()) ** 2))
        regions = part.region.to_numpy()
        row = {"indicator": col, "label": LABELS[col], "n": len(part), "regions": int(part.region.nunique()),
               "coverage": len(part) / total_entities}
        for m in MODELS:
            e = part[m].to_numpy() - y
            row[f"r2_{m}"] = 1 - float(np.mean(e ** 2)) / var
            row[f"rmse_{m}"] = float(np.sqrt(np.mean(e ** 2)))
        model_rows.append(row)
        # One set of region draws per indicator, shared by all contrasts (common random numbers).
        picks = region_picks(row["regions"], draws, seed + sorted(LABELS).index(col))
        for name, (cand, base, role, status) in CONTRASTS.items():
            b = bootstrap_contrast(y, part[cand].to_numpy(), part[base].to_numpy(), regions, picks, len(LABELS))
            contrast_rows.append({
                "contrast": name, "status": status, "role": role, "candidate": cand, "baseline": base,
                "indicator": col, "label": LABELS[col], "n": len(part), "regions": row["regions"],
                "relative_delta_mse": b["delta_mse"] / float(np.mean((part[base].to_numpy() - y) ** 2)), **b,
                **region_influence(y, part[cand].to_numpy(), part[base].to_numpy(), regions)})
    contrasts = pd.DataFrame(contrast_rows)
    for p_col, prefix in (("p_boot_t", "p_holm"), ("p_percentile", "p_pct_holm")):
        contrasts[f"{prefix}_family"] = np.nan
        for name in CONTRASTS:
            mask = contrasts.contrast == name
            contrasts.loc[mask, f"{prefix}_family"] = holm(contrasts.loc[mask, p_col].to_numpy())
        contrasts[f"{prefix}_all"] = holm(contrasts[p_col].to_numpy())
    for prefix, tag in (("p_holm", ""), ("p_pct_holm", "pct_")):
        for scope in ("family", "all"):
            significant = contrasts[f"{prefix}_{scope}"] < 0.05
            contrasts[f"{tag}improves_holm_{scope}"] = significant & (contrasts.delta_r2 > 0)
            contrasts[f"{tag}worsens_holm_{scope}"] = significant & (contrasts.delta_r2 < 0)
    return pd.DataFrame(model_rows), contrasts


MODEL_RU = {"groups": "группы", "level": "уровень", "level_spline": "сплайн уровня", "level_groups": "уровень + группы",
            "spline_groups": "сплайн + группы", "level_shares": "уровень + доли", "spline_shares": "сплайн + доли",
            "spline_shares_groups": "сплайн + доли + группы"}


def _f(v: float, digits: int = 3) -> str:
    return f"{v:+.{digits}f}".replace("-", "−")


def _p(v: float) -> str:
    return f"{v:.4f}" if v >= 1e-4 else f"{v:.1e}"


def _sig(part: pd.DataFrame, scope: str, tag: str = "") -> str:
    up = part[part[f"{tag}improves_holm_{scope}"]].label.tolist()
    down = part[part[f"{tag}worsens_holm_{scope}"]].label.tolist()
    return (f"улучшение {len(up)} из 8" + (f" ({', '.join(up)})" if up else "")
            + f"; ухудшение {len(down)} из 8" + (f" ({', '.join(down)})" if down else ""))


def contrast_table(part: pd.DataFrame, digits: int = 3) -> list:
    lines = ["| Показатель | n | регионов | ΔR² | 95% интервал | одновременный интервал (Бонферрони, 8) "
             "| p | p Холма, семейство | p Холма, все 48 |",
             "|---|---:|---:|---:|---|---|---:|---:|---:|"]
    for r in part.itertuples():
        lines.append(f"| {r.label} | {r.n} | {r.regions} | {_f(r.delta_r2, digits)} | "
                     f"[{_f(r.delta_r2_ci_low, digits)}; {_f(r.delta_r2_ci_high, digits)}] | "
                     f"[{_f(r.delta_r2_simultaneous_low, digits)}; {_f(r.delta_r2_simultaneous_high, digits)}] | "
                     f"{_p(r.p_boot_t)} | {_p(r.p_holm_family)} | {_p(r.p_holm_all)} |")
    return lines


def render_readme(models: pd.DataFrame, contrasts: pd.DataFrame, folds: pd.DataFrame, draws: int, verification: dict,
                  calib: pd.DataFrame) -> str:
    total = len(contrasts)
    bitwise = int(folds.state_sha256_bitwise_equal.sum())
    lines = [
        "# Добавочная ценность сетевых групп сверх уровня расходов",
        "",
        "Вопрос: дают ли четыре группы v1.2 информацию о внешних показателях МО сверх уровня расходов "
        "и сверх пяти непрерывных долей категорий. Все прогнозы получены вне обучения, с исключением целого региона.",
        "",
        "## Воспроизведение",
        "",
        "```",
        "python -m scripts.check_added_value            # пересчитать этот каталог, около минуты, один поток",
        "python -m scripts.check_added_value --check    # пересчитать и сравнить с сохранёнными таблицами",
        "python -m unittest discover -s tests -p test_added_value.py",
        "```",
        "",
        "## Последовательность анализа",
        "",
        "1. До первого полного прогона заданы шесть моделей (уровень, сплайн уровня, уровень + группы, "
        "сплайн + группы, уровень + доли, сплайн + доли) и пять сравнений. Основным выбрано "
        "«сплайн + группы минус сплайн уровня». «Доли сверх сплайна» и «группы против долей» "
        "считались дополнительными, «группы сверх линейного уровня» и «сплайн против линейного уровня» — справочными. "
        "Предварительной регистрации не было: план зафиксирован только в коде до запуска.",
        "2. После первого полного прогона (все восемь показателей просмотрены) добавлены модель «только группы» "
        "(для сверки с опубликованным `strict_metrics.csv`), модель «сплайн + доли + группы» и сравнение "
        "«группы сверх сплайна и долей».",
        "3. После обсуждения результатов (коммит `2d0fdd3`, все исходы уже известны) изменены только способы "
        "вывода: основной p и интервалы — симметричный bootstrap-t по регионам вместо перцентильного (обоснование — "
        "имитация с нулевым эффектом ниже, не зависящая от исходов); число повторов 4 000 → "
        + f"{draws:,}".replace(",", " ") + "; в перцентильной проверке чувствительности Var(y) пересчитывается "
        f"в каждом повторе; добавлены общая поправка Холма по всем {total} проверкам и диагностика состояний "
        "складок. Модели, складки, прогнозы и точечные оценки не менялись.",
        "",
        "Статус каждого сравнения записан в столбце `status` файла `contrasts.csv`.",
        "",
        "## Как устроена проверка",
        "",
        f"- Складки из `reports/v1.2/strict-region-validation`, всего {len(folds)}. Для каждой складки обучающее "
        "состояние (скейлер, медианы уровня по месяцам, граф kNN15, совместная модель α = 0,5, центры, именование) "
        "заново строится только по остальным регионам. Проверяемые МО получают группу ближайшего обучающего центра.",
        "- Уровень — шестой признак v1.2 в масштабе обучающей части; доли — первые пять признаков. "
        "Сплайн — ограниченный кубический с узлами в квантилях 5/35/65/95% уровня обучающей части; "
        "нужен, чтобы группы не выигрывали только за счёт аппроксимации нелинейности уровня.",
        "- Внутри показателя все модели оцениваются на одном и том же составе МО (n ниже). "
        "Пропуск Росстата не заменяется нулём.",
        "- R² вне обучения: 1 − Σ(y − ŷ)² / Σ(y − ȳ)², где ȳ — среднее наблюдаемых значений тех же МО "
        "(как в `strict_metrics.csv`). ΔR² = R²(кандидат) − R²(база) = −ΔMSE / Var(y).",
        "",
        "## Статистика",
        "",
        f"- Бутстреп по регионам: в каждом из B = {draws} повторов из R регионов показателя с возвращением "
        "вытягиваются R регионов; регион, вытянутый дважды, входит дважды. Вес каждого МО равен единице, "
        "поэтому регион входит с числом своих МО, а статистика повтора — среднее по МО вытянутых регионов "
        "(тот же оценщик, что и точечная оценка). Для одного показателя все сравнения используют одни и те же "
        "повторы (зерно = 1729 + номер показателя по алфавиту).",
        "- Основной вывод — симметричный студентизированный бутстреп (bootstrap-t). Для средней разности "
        "квадратов ошибок m = Σ_r S_r / N кластерная стандартная ошибка se = √(R/(R−1)·Σ_r (S_r − m·n_r)²) / N, "
        "где S_r и n_r — сумма разностей и число МО региона r. В каждом повторе se* пересчитывается по "
        "вытянутым регионам, t* = (m* − m) / se*.",
        "- p = min(1, (1 + #{|t*| ≥ |t|}) / (B + 1)), t = m / se; повторы с |t*| = |t| учитываются как "
        f"отклонения (консервативно); минимально возможное p = 1/(B + 1) = {1 / (draws + 1):.1e}. "
        "Если se = 0 (одинаковые прогнозы), t = 0 и p = 1.",
        "- Интервал для ΔMSE: m ± q·se, q — квантиль |t*| уровня 0,95 (обычный 95% интервал) или 1 − 0,05/8 "
        "(одновременный интервал Бонферрони для восьми показателей). Интервал ΔR² — тот же интервал, делённый "
        "на −Var(y) оцениваемых МО, то есть в масштабе приведённого R². p < α тогда и только тогда, когда "
        "интервал уровня 1 − α не содержит ноль, с точностью до поправки +1 и интерполяции квантиля.",
        "- Почему bootstrap-t, а не перцентильный p из `2d0fdd3`: в имитации с нулевым эффектом при реальных "
        "размерах регионов перцентильный p отклоняет нулевую гипотезу чаще номинала, особенно при 48 регионах; "
        "bootstrap-t ближе к номиналу (таблица ниже, `calibration.csv`). Перцентильный p "
        "(p = min(1, 2·min((1 + #{T* ≤ 0}), (1 + #{T* ≥ 0})) / (B + 1)), нули в обоих хвостах) и перцентильные "
        "интервалы ΔR² с пересчётом Var(y) в каждом повторе сохранены как проверка чувствительности "
        "(`p_percentile`, `delta_r2_pct_*`).",
        "- Поправка Холма двух уровней. Основная — внутри семейства из восьми показателей для каждого сравнения "
        "(`p_holm_family`); для основного сравнения это итоговая проверка. Проверка чувствительности — одна "
        f"поправка Холма по всем {total} проверкам шести сравнений (`p_holm_all`). Одновременные интервалы "
        "соответствуют Бонферрони внутри семейства, а не Холму, поэтому значимый по Холму результат может "
        "иметь одновременный интервал, касающийся нуля.",
        "",
        "### Калибровка в имитации с нулевым эффектом",
        "",
        "Разность d = u_регион + e, E d = 0; размеры регионов взяты из данных. Доля отклонений нулевой гипотезы "
        "при номинальных уровнях 0,05 и 0,05/8 = 0,00625. Монте-Карло: стандартная ошибка доли около 0,007 при 0,05.",
        "",
        "| Размеры регионов | регионов | шум | bootstrap-t 0,05 | bootstrap-t 0,00625 | перцентильный 0,05 | перцентильный 0,00625 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for r in calib.itertuples(index=False):
        lines.append(f"| {LABELS[r[0]]} | {r[1]} | {r[3]} | {r[6]:.3f} | {r[7]:.3f} | {r[8]:.3f} | {r[9]:.3f} |")
    lines += [
        "",
        "## Сверка состояний складок",
        "",
        f"- Сравниваются компоненты, сохранённые в `fold_XXX.json`: {', '.join(n for n, _ in NUMERIC_COMPONENTS)} "
        f"(числовые, rtol = {RTOL:g}, atol = {ATOL:g}) и {', '.join(EXACT_COMPONENTS)} (точное равенство). "
        "Числа в JSON записаны кратчайшим обратимым представлением float64, поэтому нулевое отклонение означает "
        "побитовое совпадение.",
        f"- Результат по {len(folds)} складкам: наибольшее абсолютное отклонение по числовым компонентам "
        f"{verification['max_abs_over_components']:.1e}, побитово совпадают все числовые компоненты "
        f"в {verification['folds_all_numeric_bitwise']} складках из {len(folds)}, в остальных отличия порядка последнего "
        "бита float64; точные компоненты совпадают во всех. "
        f"Назначения проверяемых МО совпадают с `strict_assignments.csv` во всех {int(folds.heldout_assignments_equal.sum())} складках.",
        "- Модель «только группы» воспроизводит опубликованный `strict_metrics.csv` (столбец `types`) по всем "
        "восьми показателям.",
        f"- В прогоне {verification['environment']['platform'].split('-')[0]} полный SHA-256 состояния совпал с сохранённым в {bitwise} из {len(folds)} складок. Штатный "
        "`validate_v12_findings` в этом окружении тоже даёт хеш, отличный от сохранённого. Разная разрядность "
        f"целых (int32/int64 у графа и меток) расхождение не объясняет ни в одной складке. В хеш входят массивы, "
        "которые в `fold_XXX.json` отдельно не сохранены (признаки обучающей части, граф CSR, обучающие метки); "
        "их собственные SHA-256 записаны в `fold_state_diagnostics.csv` для побитового сравнения на другой машине. "
        "Причина расхождения не установлена.",
        *windows_lines(),
        "- Окружение, правила сериализации и список компонентов — в `summary.json` (`state_verification`).",
        "",
        "## R² вне обучения",
        "",
        "| Показатель | n | регионов | покрытие | " + " | ".join(MODEL_RU[m] for m in MODELS) + " |",
        "|---|---:|---:|---:|" + "---:|" * len(MODELS),
    ]
    for r in models.itertuples():
        lines.append(f"| {r.label} | {r.n} | {r.regions} | {r.coverage:.1%} | "
                     + " | ".join(f"{getattr(r, 'r2_' + m):.3f}".replace("-", "−") for m in MODELS) + " |")
    lines += ["", "RMSE по тем же моделям — в `models.csv`.", "", "## Сводка значимости", "",
              "| Сравнение | статус | Холм внутри 8 показателей | Холм по всем " + str(total) + " |", "|---|---|---|---|"]
    for name, (cand, base, role, status) in CONTRASTS.items():
        part = contrasts[contrasts.contrast == name]
        lines.append(f"| {MODEL_RU[cand]} − {MODEL_RU[base]} | {STATUS_RU[status]} | {_sig(part, 'family')} | {_sig(part, 'all')} |")
    lines += ["", "Проверка чувствительности: те же поправки для перцентильного p.", "",
              "| Сравнение | Холм внутри 8 показателей | Холм по всем " + str(total) + " |", "|---|---|---|"]
    for name, (cand, base, role, status) in CONTRASTS.items():
        part = contrasts[contrasts.contrast == name]
        lines.append(f"| {MODEL_RU[cand]} − {MODEL_RU[base]} | {_sig(part, 'family', 'pct_')} | {_sig(part, 'all', 'pct_')} |")
    lines.append("")
    for name, (cand, base, role, status) in CONTRASTS.items():
        part = contrasts[contrasts.contrast == name]
        lines += [f"## ΔR²: {MODEL_RU[cand]} минус {MODEL_RU[base]}", "", f"_{role}; {STATUS_RU[status]}_", ""]
        lines += contrast_table(part, digits=4 if name == "groups_over_shares" else 3)
        lines += ["", "Влияние отдельных регионов (описательно): ΔR² при исключении каждого региона по очереди.", "",
                  "| Показатель | ΔR² | диапазон без одного региона | самый влиятельный регион | его МО | ΔR² без него |",
                  "|---|---:|---|---|---:|---:|"]
        for r in part.itertuples():
            lines.append(f"| {r.label} | {_f(r.delta_r2, 3)} | [{_f(r.drop_one_region_min, 3)}; {_f(r.drop_one_region_max, 3)}] | "
                         f"{r.most_influential_region} | {r.its_municipalities} | {_f(r.delta_r2_without_it, 3)} |")
        lines += ["", f"После поправки внутри семейства: {_sig(part, 'family')}. "
                  f"После общей поправки по {total} проверкам: {_sig(part, 'all')}.", ""]
        if name == "groups_over_shares":
            lines += ["Перцентильные интервалы (Var(y) пересчитывается в каждом повторе), проверка чувствительности:", "",
                      "| Показатель | n | регионов | 95% интервал | одновременный интервал | p перцентильный |",
                      "|---|---:|---:|---|---|---:|"]
            for r in part.itertuples():
                lines.append(f"| {r.label} | {r.n} | {r.regions} | [{_f(r.delta_r2_pct_ci_low, 4)}; {_f(r.delta_r2_pct_ci_high, 4)}] | "
                             f"[{_f(r.delta_r2_pct_simultaneous_low, 4)}; {_f(r.delta_r2_pct_simultaneous_high, 4)}] | {_p(r.p_percentile)} |")
            lines += ["",
                "Вывод: добавочный выигрыш групп сверх непрерывных долей в этой проверке не подтверждён. "
                "Границы интервалов — это границы соответствующих интервалов, а не доказанные пределы эффекта. "
                "Обоснованного порога практической значимости для ΔR² у нас нет, поэтому формулировка "
                "«если выигрыш есть, он мал» не используется.",
                ""]
    lines += [
        "## Ограничения",
        "",
        "- Все восемь показателей просмотрены вместе; положительные исходы — разведочные, а не заранее "
        "зарегистрированная подтверждающая проверка. Часть сравнений добавлена после первого прогона "
        "(см. «Последовательность анализа»).",
        "- Исходы 2023 года; исходы 2025 года не открывались.",
        "- Отраслевые доли есть только у МО с опубликованными данными Росстата по отрасли "
        "(покрытие в таблице; у добычи — меньшинство МО и не все регионы). Выводы по отраслям "
        "относятся только к этому наблюдаемому составу.",
        "- Рецепт типологии (K = 4, α = 0,5, kNN15) выбран ранее по всей выборке и внутри складок "
        "не выбирается заново (не вложенный выбор); переобучаются только состояние и группы.",
        "- Модели линейны по долям: «сверх долей» означает сверх линейного вклада пяти координат; "
        "нелинейные эффекты долей не проверялись.",
        "- Незначимый результат не доказывает отсутствие эффекта.",
        "- В части сравнений оценку заметно сдвигает один регион, чаще всего города федерального значения "
        "с десятками внутригородских МО (Санкт-Петербург — 80, Москва — 144); см. таблицы влияния. Там "
        "bootstrap-t даёт широкие интервалы, а перцентильный вариант — узкие; выводы, расходящиеся между "
        "методами, не следует считать устойчивыми.",
        "- Бутстреп по регионам опирается на число регионов (48–73). Калибровка проверена только в "
        "имитации с тремя видами шума; при малом числе регионов (добыча) точность остаётся ниже.",
        "- Метод вывода (bootstrap-t вместо перцентильного) выбран после просмотра результатов `2d0fdd3`, "
        "по имитации, не зависящей от наблюдаемых исходов; оба варианта приведены.",
        "",
    ]
    return "\n".join(lines)


WINDOWS = OUTPUT / "windows-state-verification.json"
COMPONENT_RU = {"sha256_structure_center": "центр структуры", "sha256_scale": "масштаб", "sha256_monthly_center": "месячный центр уровня",
                "sha256_x": "признаки обучающей части", "sha256_graph_csr": "граф", "sha256_labels": "обучающие метки",
                "sha256_centers": "центры групп"}


def windows_lines(path: Path = WINDOWS) -> list:
    """Summarise the separate Windows protocol (written by the author's run), if it is present."""
    if not path.exists():
        return []
    w = json.loads(path.read_text(encoding="utf-8"))
    env = w["environment"]
    counts = w.get("component_hash_match_counts_vs_linux", {})
    match = ", ".join(f"{COMPONENT_RU.get(k, k)} {v}" for k, v in counts.items())
    return [
        f"- Отдельный запуск на {env['os']} (Python {env['python']}, numpy {env['packages']['numpy']}, scipy "
        f"{env['packages']['scipy']}, scikit-learn {env['packages']['scikit-learn']}) побитово воспроизвёл полный хеш "
        f"в {w['full_sha256_equal_folds']} из {w['completed_folds']} складок и назначения "
        f"{w['heldout_matching_municipalities']} исключённых МО: [`windows-state-verification.json`](windows-state-verification.json).",
        f"- Совпадение хешей компонентов между этим запуском и Linux-диагностикой (`fold_state_diagnostics.csv`), "
        f"число складок из {w['completed_folds']}: {match}. Расхождение Linux возникает уже в признаках обучающей части; "
        "граф строится по ним и наследует отличие, обучающие метки совпадают во всех складках. Какая операция даёт "
        "отличие в последнем бите, не установлено.",
    ]


def environment() -> dict:
    import platform
    from importlib.metadata import version
    from threadpoolctl import threadpool_info
    return {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "packages": {p: version(p) for p in ("numpy", "scipy", "pandas", "scikit-learn", "threadpoolctl")},
            "blas": [{k: i.get(k) for k in ("internal_api", "version", "architecture", "num_threads")}
                     for i in threadpool_info()]}


def _windows_localisation() -> dict | None:
    if not WINDOWS.exists():
        return None
    w = json.loads(WINDOWS.read_text(encoding="utf-8"))
    return {"windows_full_hash_equal_folds": w["full_sha256_equal_folds"],
            "component_hash_match_counts_vs_linux": w.get("component_hash_match_counts_vs_linux")}


def verification_summary(folds: pd.DataFrame) -> dict:
    numeric = [n for n, _ in NUMERIC_COMPONENTS]
    return {
        "compared_numeric": numeric, "compared_exact": list(EXACT_COMPONENTS), "rtol": RTOL, "atol": ATOL,
        "not_stored_in_fold_json": ["training attributes x", "graph CSR (indptr, indices, data)", "training labels"],
        "serialization": "fold JSON written by json.dumps: floats use the shortest repr that round-trips float64 "
                         "exactly; reading back gives the identical float64, so zero deviation means bitwise equality",
        "max_abs_by_component": {n: float(folds[f"{n}_max_abs"].max()) for n in numeric},
        "max_rel_by_component": {n: float(folds[f"{n}_max_rel"].max()) for n in numeric},
        "max_abs_over_components": float(max(folds[f"{n}_max_abs"].max() for n in numeric)),
        "folds_all_numeric_bitwise": int(folds[[f"{n}_bitwise_equal" for n in numeric]].all(axis=1).sum()),
        "exact_components_equal_all_folds": bool(folds[[f"{n}_equal" for n in EXACT_COMPONENTS]].all().all()),
        "heldout_assignments_equal_all_folds": bool(folds.heldout_assignments_equal.all()),
        "full_state_sha256_equal_folds": int(folds.state_sha256_bitwise_equal.sum()),
        "saved_hash_explained_by_int_width_folds": int(folds.saved_hash_explained_by_int_width.sum()),
        "cause_of_full_hash_difference": "not established",
        "localisation_from_windows_protocol": _windows_localisation(),
        "environment": environment(),
    }


def write_all(out: Path, pred: pd.DataFrame, folds: list, models: pd.DataFrame, contrasts: pd.DataFrame,
              calib: pd.DataFrame, draws: int, seed: int, source: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fold_table = pd.DataFrame(folds).drop(columns="knots")
    pred.to_csv(out / "predictions.csv", index=False, float_format="%.10g", lineterminator="\n")
    models.to_csv(out / "models.csv", index=False, float_format="%.10g", lineterminator="\n")
    contrasts.to_csv(out / "contrasts.csv", index=False, float_format="%.10g", lineterminator="\n")
    calib.to_csv(out / "calibration.csv", index=False, float_format="%.6g", lineterminator="\n")
    fold_table[["fold", "region", "train_n", "test_n", "state_verified", "heldout_assignments_equal",
                "state_sha256_bitwise_equal"]].to_csv(out / "folds.csv", index=False, lineterminator="\n")
    fold_table.to_csv(out / "fold_state_diagnostics.csv", index=False, float_format="%.6g", lineterminator="\n")
    inputs = ["data/v12/panel.csv.gz", "reports/external-national-2026-10-03/cohort-2023.csv",
              "reports/external-v4/municipality_bridge.csv",
              "reports/v1.2/strict-region-validation/strict_assignments.csv", "configs/v12.json"]
    verification = verification_summary(fold_table)
    summary = {
        "command": "python -m scripts.check_added_value", "bootstrap_draws": draws, "seed": seed,
        "folds": len(folds), "all_fold_states_verified": bool(fold_table.state_verified.all()),
        "spline_knot_quantiles": list(QUANTILES), "models": list(MODELS),
        "contrasts": {k: {"candidate": v[0], "baseline": v[1], "role": v[2], "status": v[3]} for k, v in CONTRASTS.items()},
        "bootstrap": {"unit": "region, drawn with replacement, R draws per replicate",
                      "weights": "each municipality weight 1; statistic = municipality-weighted mean over drawn regions",
                      "random_numbers": "one set of draws per indicator, shared by all contrasts",
                      "main": {"method": "symmetric region-cluster bootstrap-t",
                               "se": "sqrt(R/(R-1)*sum_r (S_r - m n_r)^2)/N, recomputed in every replicate",
                               "p_value": "min(1, (1+#{|t*|>=|t|})/(B+1)); se=0 gives t=0, p=1",
                               "min_p": 1 / (draws + 1),
                               "intervals": "m +/- q*se, q = quantile of |t*| at 0.95 or 1-0.05/8; R^2 interval = -interval/Var(y)"},
                      "sensitivity": {"method": "percentile",
                                      "p_value": "min(1, 2*min(1+#{T*<=0}, 1+#{T*>=0})/(B+1)), T* = delta MSE; zeros in both tails",
                                      "min_p": 2 / (draws + 1),
                                      "intervals": "equal-tailed percentile of delta R^2 with Var(y) recomputed per replicate"}},
        "multiplicity": {"family": "Holm within each contrast over 8 indicators (p_holm_family; percentile: p_pct_holm_family)",
                         "all": f"Holm over all {len(contrasts)} tests (p_holm_all; percentile: p_pct_holm_all), sensitivity check",
                         "simultaneous_intervals": "Bonferroni within the family, 95%/8"},
        "state_verification": verification,
        "inputs_sha256": {p: hashlib.sha256((source / p).read_bytes()).hexdigest() for p in inputs},
        "scope": ["Retrospective 2023 outcomes; 2025 outcomes are not opened.",
                  "All eight indicators inspected together; positive results are exploratory.",
                  "Sector shares cover only municipalities with published Rosstat counts for that sector; "
                  "conclusions are limited to that observed composition.",
                  "Groups come from consumption data; they are tested, not defined, by the indicators."],
    }
    (out / "README.md").write_text(render_readme(models, contrasts, fold_table, draws, verification, calib), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--max-regions", type=int, default=None, help="Smoke run on the first folds only")
    parser.add_argument("--check", action="store_true", help="Recompute into a temporary folder and compare tables")
    args = parser.parse_args()
    started = time.time()
    with threadpool_limits(limits=1):
        pred, folds = predictions(ROOT, args.max_regions)
    models, contrasts = summarise(pred, args.draws, args.seed, total_entities=2016)
    calib = calibration(pred)
    if args.check:
        saved_m = pd.read_csv(args.output / "models.csv")
        saved_c = pd.read_csv(args.output / "contrasts.csv")
        pd.testing.assert_frame_equal(calib, pd.read_csv(args.output / "calibration.csv"), check_exact=False, rtol=1e-5)
        pd.testing.assert_frame_equal(models.reset_index(drop=True), saved_m, check_exact=False, rtol=1e-5, atol=1e-8)
        pd.testing.assert_frame_equal(contrasts.reset_index(drop=True), saved_c, check_exact=False, rtol=1e-5, atol=1e-8)
        print(json.dumps({"check": "ok", "seconds": round(time.time() - started)}))
        return
    write_all(args.output, pred, folds, models, contrasts, calib, args.draws, args.seed, ROOT)
    print(json.dumps({"output": str(args.output), "folds": len(folds), "seconds": round(time.time() - started)}))


if __name__ == "__main__":
    main()
