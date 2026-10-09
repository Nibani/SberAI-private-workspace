"""Reproduce corrected v1.2 findings and fixed-recipe regional validation.

    python -m scripts.validate_v12_findings

Cached graph and model-selection grids are retained; no new grid is searched.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
from dataclasses import replace
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from sbercluster import panel as P, typology as TY, attributed as A, edges as E
from sbercluster.graph import knn_graph
from sbercluster.joint import fit_graph_regularized_kmeans
from scipy.sparse import triu

DESCRIPTIONS = {
    "metro": "Высокие расходы и высокая доля общепита",
    "industrial": "Промежуточные расходы среди оставшихся групп",
    "periphery": "Низкие расходы среди оставшихся групп",
    "remote": "Низкая доля маркетплейсов среди оставшихся групп",
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(*arrays):
    result = hashlib.sha256()
    for arr in arrays:
        arr = np.ascontiguousarray(arr)
        result.update(str(arr.shape).encode())
        result.update(str(arr.dtype).encode())
        result.update(arr.tobytes())
    return result.hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def subset(panel, mask):
    return replace(panel, ids=panel.ids[mask], names=panel.names[mask], regions=panel.regions[mask],
                   totals=panel.totals[mask, :12], shares=panel.shares[mask, :12], periods=panel.periods[:12])

def frozen_attributes(panel, scaler, monthly_level_center):
    np.testing.assert_array_equal(scaler.level_center, monthly_level_center)
    return P.annual_profile(P.monthly_attributes(panel, scaler), slice(0, 12))

def fit_training(panel, train):
    part = subset(panel, train)
    scaler = P.fit_scaler(part, 12)
    monthly_center = np.median(np.log(part.totals), axis=0)
    x = frozen_attributes(part, scaler, monthly_center)
    np.testing.assert_allclose(x, P.annual_profile(P.monthly_attributes(part, scaler), slice(0, 12)), rtol=0, atol=1e-14)
    w, graph_info = knn_graph(x, k=15)
    initial = KMeans(4, n_init=50, random_state=1729).fit_predict(x)
    raw, info = fit_graph_regularized_kmeans(x, w, initial, alpha=.5, seed=1729, max_sweeps=200)
    naming = TY.name_types(np.median(part.totals, axis=1), np.median(part.shares, axis=1), raw)
    labels = TY.canonical_labels(raw, naming)
    centers = np.array([x[labels == i].mean(axis=0) for i in range(4)])
    state_hash = digest(scaler.structure_center, scaler.scale, monthly_center, x, w.indptr, w.indices, w.data, labels, centers)
    return {"scaler": scaler, "monthly_center": monthly_center, "x": x, "labels": labels, "centers": centers,
            "fit_info": {k: info[k] for k in ("algorithm_revision", "converged", "convergence_status", "sweeps", "objective")},
            "graph_info": graph_info, "naming": {str(k): v for k, v in naming.items()}, "hash": state_hash}

def onehot(labels, k):
    return np.eye(k)[np.asarray(labels, dtype=int)]

def read_external(source, ids, cfg):
    cohort = pd.read_csv(source / cfg["inputs"]["rosstat_cohort"]).set_index("entity_id").reindex(ids)
    bridge = pd.read_csv(source / cfg["inputs"]["market_access"]).set_index("entity_id").reindex(ids)
    if cohort["territory_id"].isna().any():
        raise ValueError("Когорта Росстата не покрывает панель")
    pop = cohort["population_total"].astype(float)
    total = cohort["headcount_total"].astype(float)
    ext = pd.DataFrame({
        "log_wage": cohort["log_annual_monthly_wage_rubles"],
        "agri_share": cohort["agriculture_headcount_share"],
        "manuf_share": cohort["manufacturing_headcount_share"],
        "mining_share": cohort["headcount_B"] / total,
        "public_share": cohort["public_administration_education_health_headcount_share"],
        "log_pop": np.log(pop.where(pop > 0)),
        "log_jobs_per_res": cohort["log_covered_organization_jobs_per_resident"],
        "market_access": bridge["market_access"],
    }, index=ids).replace([np.inf, -np.inf], np.nan)
    return ext, cohort

def original_predictions(ext, types, admin, regions):
    result = []
    for col in ext.columns:
        y = ext[col].to_numpy(float)
        ok = np.isfinite(y)
        for region in np.unique(regions):
            train = ok & (regions != region)
            test = ok & (regions == region)
            if not test.any():
                continue
            dtrain = onehot(types[train], 4)
            dtest = onehot(types[test], 4)
            beta = np.linalg.lstsq(dtrain, y[train], rcond=None)[0]
            abeta = np.linalg.lstsq(admin[train], y[train], rcond=None)[0]
            for pos, pred, base in zip(np.where(test)[0], dtest @ beta, admin[test] @ abeta):
                result.append({"entity_id": str(ext.index[pos]), "region": str(region), "indicator": col,
                               "observed": float(y[pos]), "type_prediction": float(pred), "admin_prediction": float(base)})
    return pd.DataFrame(result)

def metrics(predictions):
    rows = []
    for col, part in predictions.groupby("indicator", sort=False):
        y = part.observed.to_numpy(float)
        tss = float(np.square(y - y.mean()).sum())
        row = {"indicator": col, "n": len(y), "regions": int(part.region.nunique())}
        for name, column in (("types", "type_prediction"), ("municipal_type", "admin_prediction")):
            residual = y - part[column].to_numpy(float)
            row[name] = float(1 - residual @ residual / tss)
            row[name + "_mae"] = float(np.mean(np.abs(residual)))
        if "combined_prediction" in part:
            residual = y - part["combined_prediction"].to_numpy(float)
            row["municipal_type_and_types"] = float(1 - residual @ residual / tss)
            row["municipal_type_and_types_mae"] = float(np.mean(np.abs(residual)))
        if "admin_zero_prediction" in part:
            residual = y - part["admin_zero_prediction"].to_numpy(float)
            row["municipal_type_original_zero"] = float(1 - residual @ residual / tss)
            row["unseen_admin_category_n"] = int(part["unseen_admin_category"].sum())
        rows.append(row)
    return pd.DataFrame(rows)

def strict_validate(args):
    started = time.perf_counter()
    source, output = Path(args.source), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((source / "configs/v12.json").read_text(encoding="utf-8"))
    if (cfg["seed"], cfg["methods"]["k"], cfg["network"]["k"]) != (1729, 4, 15):
        raise ValueError("Strict validation is defined for the recorded fixed recipe")
    panel_path = source / cfg["inputs"]["panel"]
    panel = P.read_panel(panel_path)
    panel = replace(panel, totals=panel.totals[:, :12], shares=panel.shares[:, :12], periods=panel.periods[:12])
    ext, cohort = read_external(source, panel.ids, cfg)
    original = pd.read_csv(getattr(args, "labels", source / "reports/v1.2/labels.csv")).set_index("entity_id").reindex(panel.ids).type_2023.to_numpy(int)
    admin_names, admin_codes = np.unique(cohort.municipal_district_type.fillna("нет данных").to_numpy(), return_inverse=True)
    admin = onehot(admin_codes, len(admin_names))
    orig_pred = original_predictions(ext, original, admin, panel.regions)
    orig_pred.to_csv(output / "original_regression_only_predictions.csv", index=False)
    orig_metrics = metrics(orig_pred)
    orig_metrics.to_csv(output / "original_regression_only_metrics.csv", index=False)
    regions = np.unique(panel.regions)
    if args.max_regions:
        regions = regions[:args.max_regions]
    predictions, assignments, fold_rows = [], [], []
    print(json.dumps({"entities": len(panel.ids), "regions": len(regions), "periods": panel.periods,
                      "admin_categories": admin_names.tolist(), "single_worker": True}, ensure_ascii=False), flush=True)
    invariant = None
    for f, region in enumerate(regions):
        fold_started = time.perf_counter()
        train, test = panel.regions != region, panel.regions == region
        state = fit_training(panel, train)
        x_test = frozen_attributes(subset(panel, test), state["scaler"], state["monthly_center"])
        heldout = np.argmin(np.square(x_test[:, None, :] - state["centers"][None, :, :]).sum(axis=2), axis=1)
        train_label = state["labels"]
        test_pos = np.where(test)[0]
        for pos, label in zip(test_pos, heldout):
            key = TY.TYPE_ORDER[int(label)]
            assignments.append({"entity_id": str(panel.ids[pos]), "region": str(region), "strict_type": int(label),
                                "type_key": key, "description": TY.TYPE_NAMES[key], "original_type": int(original[pos])})
        betas = {}
        for col in ext.columns:
            y = ext[col].to_numpy(float)
            valid_train, valid_test = np.isfinite(y[train]), np.isfinite(y[test])
            if not valid_test.any():
                continue
            tdesign = onehot(train_label[valid_train], 4)
            hdesign = onehot(heldout[valid_test], 4)
            beta = np.linalg.lstsq(tdesign, y[train][valid_train], rcond=None)[0]
            both_train = np.hstack([admin[train][valid_train], tdesign[:, 1:]])
            both_test = np.hstack([admin[test][valid_test], hdesign[:, 1:]])
            both_beta = np.linalg.lstsq(both_train, y[train][valid_train], rcond=None)[0]
            abeta = np.linalg.lstsq(admin[train][valid_train], y[train][valid_train], rcond=None)[0]
            unknown_admin = admin[test][valid_test][:, ~np.any(admin[train][valid_train] > 0, axis=0)].sum(axis=1) > 0
            admin_zero = admin[test][valid_test] @ abeta
            admin_pred = np.where(unknown_admin, y[train][valid_train].mean(), admin_zero)
            combined_pred = np.where(unknown_admin, hdesign @ beta, both_test @ both_beta)
            betas[col] = {"types": beta.tolist(), "municipal_type_and_types": both_beta.tolist(), "admin": abeta.tolist(), "train_n": int(valid_train.sum()),
                          "test_n": int(valid_test.sum()), "unseen_admin_category_n": int(unknown_admin.sum())}
            for pos, pred, base, combined, zero, unseen in zip(test_pos[valid_test], hdesign @ beta, admin_pred, combined_pred, admin_zero, unknown_admin):
                predictions.append({"entity_id": str(panel.ids[pos]), "region": str(region), "indicator": col,
                                    "observed": float(y[pos]), "type_prediction": float(pred), "admin_prediction": float(base),
                                    "combined_prediction": float(combined), "admin_zero_prediction": float(zero),
                                    "unseen_admin_category": bool(unseen)})
        if f == 0:
            disturbed_totals = panel.totals.copy()
            disturbed_shares = panel.shares.copy()
            disturbed_totals[test] *= 17.0
            disturbed_shares[test] = np.roll(disturbed_shares[test], 2, axis=2)
            disturbed = replace(panel, totals=disturbed_totals, shares=disturbed_shares)
            repeated = fit_training(disturbed, train)
            assert state["hash"] == repeated["hash"], "Данные heldout изменили train fit"
            disturbed_x = frozen_attributes(subset(disturbed, test), state["scaler"], state["monthly_center"])
            invariant = {"region": str(region), "heldout_rows": int(test.sum()), "totals_multiplier": 17,
                         "share_columns_rolled": 2, "train_state_sha256_before": state["hash"],
                         "train_state_sha256_after": repeated["hash"], "train_state_unchanged": True,
                         "heldout_attributes_changed": bool(not np.array_equal(x_test, disturbed_x)),
                         "checked": ["structure_center", "iqr_scale", "monthly_log_total_center", "train_attributes",
                                     "train_graph_csr", "train_type_labels", "train_centroids", "train_only_naming"]}
            dump(output / "heldout_invariance.json", invariant)
        model = {"region": str(region), "train_n": int(train.sum()), "test_n": int(test.sum()),
                 "train_ids_sha256": hashlib.sha256("\n".join(panel.ids[train]).encode()).hexdigest(),
                 "state_sha256": state["hash"], "scaler": state["scaler"].as_dict(),
                 "monthly_log_total_center": state["monthly_center"].tolist(), "centers": state["centers"].tolist(),
                 "naming": state["naming"], "fit": state["fit_info"], "graph": state["graph_info"], "betas": betas}
        dump(output / f"fold_{f:03d}.json", model)
        elapsed = time.perf_counter() - fold_started
        fold_rows.append({"region": str(region), "train_n": int(train.sum()), "test_n": int(test.sum()),
                          "seconds": elapsed, **state["fit_info"]})
        pd.DataFrame(predictions).to_csv(output / "strict_predictions.csv", index=False)
        pd.DataFrame(assignments).to_csv(output / "strict_assignments.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(output / "folds.csv", index=False)
        if f == 0 or (f + 1) % 10 == 0 or f + 1 == len(regions):
            print(json.dumps({"fold": f + 1, "of": len(regions), "elapsed_seconds": time.perf_counter() - started,
                              "last_fold_seconds": elapsed, "converged": state["fit_info"]["converged"]}, ensure_ascii=False), flush=True)
    predicted = pd.DataFrame(predictions)
    result = metrics(predicted)
    result.to_csv(output / "strict_metrics.csv", index=False)
    assign = pd.DataFrame(assignments)
    agreement = float(np.mean(assign.strict_type == assign.original_type))
    files = ["configs/v12.json", cfg["inputs"]["panel"], cfg["inputs"]["rosstat_cohort"], cfg["inputs"]["market_access"],
             "reports/v1.2/labels.csv", "sbercluster/panel.py", "sbercluster/joint.py", "sbercluster/graph.py", "sbercluster/typology.py", "sbercluster/attributed.py", "scripts/run_v12.py"]
    summary = {"historical_recipe_source_commit": "c7905a3", "input_sha256": {name: sha(source / name) for name in files},
               "script_sha256": sha(__file__), "versions": {k: version(k) for k in ("numpy", "scipy", "pandas", "scikit-learn")},
               "recipe": {"alpha": .5, "K": 4, "graph_k": 15, "seed": 1729, "kmeans_n_init": 50, "max_sweeps": 200,
                          "heldout_assignment": "nearest train centroid; no heldout graph", "months": "2023-01..2023-12",
                          "selection_status": "fixed historical globally selected recipe; not nested selection"},
               "cohort": {"entities": len(panel.ids), "regions_total": len(np.unique(panel.regions)), "regions_evaluated": len(regions),
                          "predicted_entities": len(assign)}, "strict_metrics": result.to_dict(orient="records"),
               "original_regression_only_metrics": orig_metrics.to_dict(orient="records"),
               "type_agreement_with_global_labels": agreement,
               "ari_with_global_labels": float(adjusted_rand_score(assign.original_type, assign.strict_type)),
               "converged_folds": int(sum(row["converged"] for row in fold_rows)), "invariance": invariant,
               "admin_baseline": "same finite-outcome cohort; OLS train fit; unseen category uses train outcome mean. Original zero prediction is retained separately. Combined model uses type-only prediction for unseen administrative category.",
               "agreement_interpretation": "Agreement of inductive nearest-centroid heldout assignment with full-panel graph labels; assignment rule also differs, so this is not pure retraining stability.",
               "naming_external_rosstat_used": False, "neutral_descriptions": DESCRIPTIONS,
               "runtime_seconds": time.perf_counter() - started,
               "limitations": ["Hyperparameters historically selected using global data, including 2024 criteria; no nested selection performed",
                               "External outcome complete-case cohorts differ by indicator; mining is sparse",
                               "One-worker deterministic fixed recipe; no uncertainty intervals recomputed",
                               "All 24 months are parsed by panel reader, then only 2023 enters fitting; no 2025 data read"]}
    dump(output / "summary.json", summary)
    result.to_csv(output.parent / "leave_one_region_out.csv", index=False)
    print(result.to_string(index=False), flush=True)
    print(json.dumps({"agreement": agreement, "ari": summary["ari_with_global_labels"], "seconds": summary["runtime_seconds"]}), flush=True)


def lag_source(panel, cfg):
    level = P.relative_level(panel)
    score23, lag23 = E.lagged_correlation(level[:, :12], cfg["network"]["max_lag_months"])
    score24, lag24 = E.lagged_correlation(level[:, 12:], cfg["network"]["max_lag_months"])
    g23 = E.knn_union(score23, cfg["network"]["k"])
    g24 = E.knn_union(score24, cfg["network"]["k"])
    both = triu(g23.multiply(g24), k=1).tocoo()
    l23, l24 = lag23[both.row, both.col], lag24[both.row, both.col]
    nonzero = (l23 != 0) & (l24 != 0)
    same = np.sign(l23[nonzero]) == np.sign(l24[nonzero])
    return {"edges_2023": g23.nnz // 2, "edges_2024": g24.nnz // 2,
            "edges_in_both_years": len(both.row), "pairs_with_nonzero_lag_both_years": int(nonzero.sum()),
            "same_direction_count": int(same.sum()), "same_direction_share": float(same.mean()),
            "interpretation": "Пересечение лаговых графов с ненулевым лагом в оба года; причинность не проверена."}


def selection_audit(panel, cfg, source, types, results=None):
    table = pd.read_csv((results or source / "reports/v1.2") / "methods.csv")
    eligible = table[table.eligible].copy()
    eligible["copeland_2023_recomputed"] = A.copeland(eligible, tuple((f"{c}_2023", s) for c, s in A.ICVI))
    simplicity = ["kmeans", "joint_0.1", "joint_0.25", "joint_0.5", "joint_1.0", "ward", "gmm", "kefrin", "spectral_graph", "leiden_graph"]
    eligible["simplicity"] = eligible.method.map({m: i for i, m in enumerate(simplicity)})
    chosen = eligible.sort_values(["copeland_2023_recomputed", "simplicity"], ascending=[False, True]).iloc[0].method
    scaler = P.fit_scaler(panel)
    x = P.annual_profile(P.monthly_attributes(panel, scaler), slice(0, 12))
    weights = knn_graph(x, k=cfg["network"]["k"])[0]
    initial = KMeans(4, n_init=50, random_state=cfg["seed"]).fit_predict(x)
    fitted = {}
    for method in (chosen, "joint_0.5"):
        raw = A.fit_method(method, x, weights, 4, cfg["seed"], kmeans_labels=initial)
        naming = TY.name_types(np.median(panel.totals[:, :12], axis=1), np.median(panel.shares[:, :12], axis=1), raw)
        fitted[method] = TY.canonical_labels(raw, naming)
    np.testing.assert_array_equal(fitted["joint_0.5"], types)
    return {"chosen_2023_only": chosen, "retained_method": "joint_0.5",
            "copeland_2023": eligible[["method", "copeland_2023_recomputed"]].to_dict(orient="records"),
            "ari_2023_only_vs_retained": float(adjusted_rand_score(fitted[chosen], types)),
            "changed_n": int(np.sum(fitted[chosen] != types)), "entities": len(types),
            "status": "Диагностика по сохранённой сетке; сеть и её исторический выбор не переоценивались."}


def refresh(source, output, skip_strict=False, cached_results=None):
    from scripts.run_v12 import practical_section, clean
    cfg = json.loads((source / "configs/v12.json").read_text(encoding="utf-8"))
    original = cached_results or source / "reports/v1.2"
    output.mkdir(parents=True, exist_ok=True)
    if output.resolve() != original.resolve():
        shutil.copytree(original, output, dirs_exist_ok=True)
    summary = json.loads((original / "summary.json").read_text(encoding="utf-8"))
    panel = P.read_panel(source / cfg["inputs"]["panel"])
    labels = pd.read_csv(original / "labels.csv", dtype={"monthly_types": str})
    if labels.entity_id.tolist() != panel.ids.tolist():
        raise ValueError("Stored labels and panel have different municipality order")
    types = labels.type_2023.to_numpy(int)
    ext, cohort = read_external(source, panel.ids, cfg)
    scaler = P.fit_scaler(panel)
    x23 = P.annual_profile(P.monthly_attributes(panel, scaler), slice(0, 12))
    ctx = {"panel": panel, "regions": panel.regions, "seed": cfg["seed"], "X23": x23,
           "rel_level": P.relative_level(panel), "cohort": cohort}
    track = {"types24": labels.type_2024.to_numpy(int), "types24_absolute": labels.type_2024_absolute.to_numpy(int),
             "persistence": {"persistent_change_mask": labels.persistent_change.to_numpy(bool),
                             "persistent_new_type": np.array([int(v[-1]) for v in labels.monthly_types.str.zfill(24)])}}
    practical = practical_section(ctx, output, cfg, types, track, cfg["practical"]["bootstrap_draws"], time.time())
    for key in ("analogues", "analogue_comparisons", "type_dynamics", "analogues_diagnostic", "analogue_comparisons_diagnostic"):
        summary["practical"][key] = practical[key].to_dict(orient="records")
    summary["practical"].update({"gap": practical["gap"], "marketplace_gap_pp": practical["marketplace_gap_pp"],
        "analogue_limitations": {"unit": "percentage_point", "growth": "100*(mean_total_2024/mean_total_2023-1)",
        "evaluation": "Ретроспективное сравнение: исходы аналогов 2024 уже доступны; это не прогноз будущего года.",
        "primary_common_entities": int(practical["analogues"].n.iloc[0]),
        "diagnostic_common_entities": int(practical["analogues_diagnostic"].n.iloc[0]),
        "resampling": "Парные различия ошибок, целые регионы с возвращением; среднее по МО, seed из config.",
        "random_control": "Одна фиксированная выборка соседей; интервал не охватывает выбор случайных аналогов.",
        "selection": "Типология и сеть исторически выбраны с использованием 2024; интервалы условны."}})
    summary["methods"]["selection_audit"] = selection_audit(panel, cfg, source, types, original)
    summary["validation"] = {"year_2024": "Валидация, использованная при выборе сети и модели; независимым test не является.",
                             "year_2025": "Значения не открывались и не оценивались."}
    summary["network"]["lead_lag_source"] = lag_source(panel, cfg)
    summary["scaler"] = scaler.as_dict()
    for name in ("types.csv", "type_dynamics.csv"):
        frame = pd.read_csv(output / name)
        frame["name"] = frame.key.map(TY.TYPE_NAMES)
        frame.to_csv(output / name, index=False)
    summary["types"]["names"] = {str(i): TY.TYPE_NAMES[k] for i, k in enumerate(TY.TYPE_ORDER)}
    summary["types"]["profiles"] = pd.read_csv(output / "types.csv").to_dict(orient="records")
    summary["types"]["missing_outcomes"] = "Внешние связи рассчитаны только на наблюдаемых исходах; верхняя граница для всей панели не установлена."
    if not skip_strict:
        strict_validate(SimpleNamespace(source=source, output=output / "strict-region-validation", max_regions=0,
                                       labels=original / "labels.csv"))
        strict = json.loads((output / "strict-region-validation/summary.json").read_text(encoding="utf-8"))
        summary["types"]["strict_leave_one_region_out"] = {"recipe": strict["recipe"], "metrics": strict["strict_metrics"],
            "limitations": strict["limitations"], "prediction_file": "strict-region-validation/strict_predictions.csv",
            "converged_folds": strict["converged_folds"], "ari_with_global_labels": strict["ari_with_global_labels"],
            "agreement_with_global_labels": strict["type_agreement_with_global_labels"],
            "agreement_interpretation": strict["agreement_interpretation"], "admin_baseline": strict["admin_baseline"]}
        summary["types"]["leave_one_region_out"] = strict["strict_metrics"]
    (output / "summary.json").write_text(json.dumps(clean(summary), ensure_ascii=False, indent=1, allow_nan=False) + "\n", encoding="utf-8")
    files = ["configs/v12.json", *cfg["inputs"].values(), "reports/v1.2/labels.csv", "reports/v1.2/methods.csv",
             "sbercluster/panel.py", "sbercluster/typology.py", "sbercluster/attributed.py", "sbercluster/edges.py",
             "sbercluster/graph.py", "sbercluster/joint.py", "scripts/run_v12.py", "scripts/validate_v12_findings.py"]
    provenance = {"command": "python -m scripts.validate_v12_findings", "config": cfg,
                  "sha256": {name: sha(source / name) for name in files},
                  "versions": {k: version(k) for k in ("numpy", "scipy", "pandas", "scikit-learn")},
                  "strict_recomputed": not skip_strict, "historical_grid_source_commit": "c7905a3",
                  "cached_tables": ["networks.csv", "network_effects.csv", "network_k_sensitivity.csv", "feature_spaces.csv", "k_selection.csv", "methods.csv"],
                  "limitations": "Сохранённая сетка использована для диагностического выбора; полный поиск не повторялся."}
    provenance["cached_table_sha256"] = {name: sha(original / name) for name in provenance["cached_tables"]}
    provenance["sha256"]["reports/v1.2/labels.csv"] = sha(original / "labels.csv")
    provenance["sha256"]["reports/v1.2/methods.csv"] = sha(original / "methods.csv")
    provenance["result_sha256"] = {path.name: sha(path) for path in output.glob("*.csv")}
    provenance["result_sha256"]["summary.json"] = sha(output / "summary.json")
    dump(output / "findings_provenance.json", provenance)
    dump(output / "provenance.json", {**json.loads((original / "provenance.json").read_text(encoding="utf-8")),
                                      "config": cfg, "correction_provenance": "findings_provenance.json"})


def check_results(source, expected, skip_strict):
    with tempfile.TemporaryDirectory(prefix="sber-v12-check-") as directory:
        regenerated = Path(directory)
        refresh(source, regenerated, skip_strict)
        names = ["analogues.csv", "analogue_comparisons.csv", "analogues_diagnostic.csv", "analogue_comparisons_diagnostic.csv",
                 "analogue_predictions.csv", "type_dynamics.csv", "persistent_changes.csv"]
        if not skip_strict:
            names += ["leave_one_region_out.csv", "strict-region-validation/strict_predictions.csv",
                      "strict-region-validation/strict_assignments.csv", "strict-region-validation/strict_metrics.csv"]
        for name in names:
            pd.testing.assert_frame_equal(pd.read_csv(expected / name), pd.read_csv(regenerated / name),
                                          check_exact=False, rtol=1e-10, atol=1e-12)
        a = json.loads((expected / "summary.json").read_text(encoding="utf-8"))
        b = json.loads((regenerated / "summary.json").read_text(encoding="utf-8"))
        for key in ("methods", "validation", "network", "practical", "types", "scaler"):
            compare_json(a[key], b[key], key)
        stored = json.loads((expected / "findings_provenance.json").read_text(encoding="utf-8"))
        current = json.loads((regenerated / "findings_provenance.json").read_text(encoding="utf-8"))
        if stored["sha256"] != current["sha256"] or stored["cached_table_sha256"] != current["cached_table_sha256"]:
            raise AssertionError("Findings provenance hashes are stale")
        print("Corrected findings, predictions and provenance reproduced", flush=True)


def compare_json(expected, actual, path=""):
    """Keep IDs and schemas exact while allowing floating-point roundoff."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or expected.keys() != actual.keys():
            raise AssertionError(f"JSON keys differ at {path}")
        for key in expected:
            compare_json(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            raise AssertionError(f"JSON list length differs at {path}")
        for i, (a, b) in enumerate(zip(expected, actual)):
            compare_json(a, b, f"{path}[{i}]")
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not np.isclose(expected, actual, rtol=1e-10, atol=1e-12):
            raise AssertionError(f"JSON number differs at {path}: {expected} vs {actual}")
    elif type(expected) is not type(actual) or expected != actual:
        raise AssertionError(f"JSON value differs at {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/v1.2")
    parser.add_argument("--check", action="store_true", help="reproduce in a temporary directory and compare")
    parser.add_argument("--skip-strict", action="store_true", help="only check the cheap diagnostics; not full acceptance")
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.check:
            check_results(ROOT, args.output, args.skip_strict)
        else:
            refresh(ROOT, args.output, args.skip_strict)


if __name__ == "__main__":
    main()

