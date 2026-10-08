"""Version 1.2: the complete analysis from repository files in one command.

    python -m scripts.run_v12                 # writes reports/v1.2
    python -m scripts.run_v12 --quick         # fewer bootstrap draws, for a smoke run

Steps: attributes -> nine edge rules compared on independent data -> choice of K
-> ten clustering methods on the attributed network -> tracking over 24 months
-> economic reading of the types -> practical checks. Every number in the v1.2
report is written to reports/v1.2 by this script; nothing is edited by hand.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from importlib import metadata
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, triu
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sbercluster import attributed as A  # noqa: E402
from sbercluster import edges as E  # noqa: E402
from sbercluster import panel as P  # noqa: E402
from sbercluster import tracking as T  # noqa: E402
from sbercluster import typology as TY  # noqa: E402
from sbercluster.centroids import centroids_from_map  # noqa: E402
from sbercluster.graph import knn_graph  # noqa: E402
from sbercluster.joint import fit_graph_regularized_kmeans  # noqa: E402
from sbercluster.metrics import network_quality_candidates  # noqa: E402

EXTERNAL_LABELS = {
    "log_wage": "Логарифм средней зарплаты (Росстат, 2023)",
    "agri_share": "Доля занятых в сельском хозяйстве",
    "manuf_share": "Доля занятых в обрабатывающей промышленности",
    "mining_share": "Доля занятых в добыче полезных ископаемых",
    "public_share": "Доля занятых в госуправлении, образовании, здравоохранении",
    "log_pop": "Логарифм численности населения",
    "log_jobs_per_res": "Логарифм числа рабочих мест на жителя",
    "market_access": "Доступность рынков (СберИндекс)",
}
RULE_LABELS = {
    "euclid_structure": "Евклидово расстояние: структура расходов (5 признаков)",
    "euclid_profile": "Евклидово расстояние: структура + уровень (6 признаков)",
    "cosine_spending": "Косинусное сходство векторов расходов в рублях",
    "corr_total": "Корреляция месячных рядов уровня расходов",
    "corr_multivariate": "Корреляция многомерных рядов (6 признаков)",
    "lagged_corr_total": "Лаговая корреляция (опережение до 2 мес.)",
    "dtw_total": "Динамическое выравнивание рядов (DTW)",
    "geographic": "Географическое расстояние между центроидами",
    "road": "Дорожное расстояние (СберИндекс)",
}
METHOD_LABELS = {
    "kmeans": "KMeans", "ward": "Ward", "gmm": "Гауссова смесь", "spectral_graph": "Спектральная (граф)",
    "leiden_graph": "Leiden (граф)", "kefrin": "KEFRiN-подобный K-means (признаки + строки смежности)",
    "joint_0.1": "Совместная модель α=0,1", "joint_0.25": "Совместная модель α=0,25",
    "joint_0.5": "Совместная модель α=0,5", "joint_1.0": "Совместная модель α=1",
}
METHOD_FAMILY = {"kmeans": "признаки", "ward": "признаки", "gmm": "признаки", "spectral_graph": "сеть",
                 "leiden_graph": "сеть", "kefrin": "признаки + сеть"}


# ------------------------------------------------------------------ helpers
def clean(value):
    """JSON-safe structure retaining float precision; NaN/inf become null."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def log(message: str, start: float) -> None:
    print(f"[{time.time() - start:7.1f} s] {message}", flush=True)


def load_external(cfg: dict, ids: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = pd.read_csv(ROOT / cfg["rosstat_cohort"]).set_index("entity_id").reindex(ids)
    bridge = pd.read_csv(ROOT / cfg["market_access"]).set_index("entity_id").reindex(ids)
    if cohort.index.isna().any() or cohort["territory_id"].isna().any():
        raise ValueError("Rosstat cohort does not cover every panel municipality")

    def positive_log(series):
        values = series.astype(float)
        return np.log(values.where(values > 0))

    total = cohort["headcount_total"].astype(float)
    external = pd.DataFrame({
        "log_wage": cohort["log_annual_monthly_wage_rubles"],
        "agri_share": cohort["agriculture_headcount_share"],
        "manuf_share": cohort["manufacturing_headcount_share"],
        "mining_share": cohort["headcount_B"] / total,
        "public_share": cohort["public_administration_education_health_headcount_share"],
        "log_pop": positive_log(cohort["population_total"]),
        "log_jobs_per_res": cohort["log_covered_organization_jobs_per_resident"],
        "market_access": bridge["market_access"],
    }, index=ids).replace([np.inf, -np.inf], np.nan)
    return external, cohort


def read_road_graph(cfg: dict, ids: np.ndarray) -> csr_matrix:
    stored = np.load(ROOT / cfg["road_graph"])
    graph = csr_matrix((stored["data"], stored["indices"], stored["indptr"]), shape=tuple(stored["shape"]))
    road_ids = json.loads((ROOT / cfg["road_graph_ids"]).read_text(encoding="utf-8"))["ids"]
    order = {entity: i for i, entity in enumerate(road_ids)}
    index = np.array([order[i] for i in ids])
    graph = graph[index][:, index]
    graph.data[:] = 1.0
    return graph


# ------------------------------------------------------------------ sections
def network_section(ctx, out: Path, cfg: dict, start: float) -> dict:
    k = cfg["network"]["k"]
    w23, w24 = slice(0, 12), slice(12, 24)
    rel = ctx["rel_level"]
    xrel = ctx["Xm"] - P.national_component(ctx["Xm"])[None]
    rub = ctx["panel"].shares * ctx["panel"].totals[:, :, None]
    rub23, rub24 = np.median(rub[:, w23], axis=1), np.median(rub[:, w24], axis=1)
    x23, x24 = ctx["X23"], ctx["X24_raw"]
    max_lag, band = cfg["network"]["max_lag_months"], cfg["network"]["dtw_band_months"]
    lag23 = E.lagged_correlation(rel[:, w23], max_lag)
    lag24 = E.lagged_correlation(rel[:, w24], max_lag)
    scores = {
        "euclid_structure": (-E.pairwise_euclidean(x23[:, :5]), -E.pairwise_euclidean(x24[:, :5])),
        "euclid_profile": (-E.pairwise_euclidean(x23), -E.pairwise_euclidean(x24)),
        "cosine_spending": (E.cosine_similarity(rub23), E.cosine_similarity(rub24)),
        "corr_total": (E.pearson_matrix(rel[:, w23]), E.pearson_matrix(rel[:, w24])),
        "corr_multivariate": (E.multivariate_correlation(xrel[:, w23]), E.multivariate_correlation(xrel[:, w24])),
        "lagged_corr_total": (lag23[0], lag24[0]),
        "dtw_total": (-E.dtw_distances(rel[:, w23], band), -E.dtw_distances(rel[:, w24], band)),
    }
    graphs = {name: (E.knn_union(a, k), E.knn_union(b, k)) for name, (a, b) in scores.items()}
    geo = -E.haversine_km(ctx["lat"], ctx["lon"])
    graphs["geographic"] = (E.knn_union(geo, k), None)
    graphs["road"] = (ctx["road"], None)
    log("nine edge rules built", start)

    reference = graphs["euclid_profile"][0]
    rows = []
    for name, (g23, g24) in graphs.items():
        row = {"rule": name, "label": RULE_LABELS[name], "static": g24 is None}
        row.update(E.evaluate_graph(g23, ctx["external"], ctx["regions"], reference=reference, later=g24))
        row["components"] = int(connected_components(g23)[0])
        rows.append(row)
    table = pd.DataFrame(rows)

    # Selection: drop geography-dominated rules, then Copeland over three criteria.
    gf = cfg["network"]["geography_filter"]
    table["geography_dominated"] = ((table["same_region_share"] > gf["max_same_region_share"]) &
                                    (table["assort_within_region_mean"] < gf["min_within_region_assortativity"]))
    eligible = table[~table["geography_dominated"] & ~table["static"]].copy()
    eligible["copeland"] = A.copeland(eligible, (("assort_mean", 1), ("assort_within_region_mean", 1),
                                                 ("stability_jaccard", 1))).values
    table = table.merge(eligible[["rule", "copeland"]], on="rule", how="left")
    chosen = eligible.sort_values(["copeland", "assort_within_region_mean"], ascending=False).iloc[0]["rule"]
    write_csv(out / "networks.csv", table)

    # Lead-lag consistency for edges present in both years.
    g23, g24 = graphs["lagged_corr_total"]
    both = triu(g23.multiply(g24), k=1).tocoo()
    l23, l24 = lag23[1][both.row, both.col], lag24[1][both.row, both.col]
    nonzero = (l23 != 0) & (l24 != 0)
    upper23 = triu(g23, k=1).tocoo()
    lead_lag = {"edges_in_both_years": int(len(both.row)),
                "share_nonzero_best_lag_2023": float(np.mean(lag23[1][upper23.row, upper23.col] != 0)),
                "same_direction_share_when_both_nonzero": float(np.mean(np.sign(l23[nonzero]) == np.sign(l24[nonzero])))
                if nonzero.any() else None,
                "pairs_with_nonzero_lag_both_years": int(nonzero.sum())}

    # How the rule changes the clustering result.
    effects = []
    km = ctx["kmeans4"]
    def leiden(adjacency):
        upper = triu(adjacency, k=1).tocoo()
        graph = ig.Graph(n=adjacency.shape[0], edges=list(zip(upper.row.tolist(), upper.col.tolist())))
        return np.array(la.find_partition(graph, la.ModularityVertexPartition, seed=ctx["seed"], n_iterations=-1).membership)

    for name, (g23, g24) in graphs.items():
        communities = leiden(g23)
        joint, fit = fit_graph_regularized_kmeans(ctx["X23"], g23, km, alpha=0.5, seed=ctx["seed"], max_sweeps=200)
        # Same attributes and start, only the graph of the other year: how much does the result depend on it?
        joint_other = None if g24 is None else fit_graph_regularized_kmeans(
            ctx["X23"], g24, km, alpha=0.5, seed=ctx["seed"], max_sweeps=200)[0]
        ev_comm = A.external_validity(communities, ctx["external"], ctx["regions"])
        ev_joint = A.external_validity(joint, ctx["external"], ctx["regions"])
        effects.append({
            "rule": name, "label": RULE_LABELS[name],
            "leiden_communities": int(communities.max() + 1),
            "leiden_modularity": network_quality_candidates(g23, communities)["NewmanQ"],
            "leiden_largest_share": float(np.bincount(communities).max() / len(communities)),
            "leiden_partial_r2_within_region_mean": ev_comm["partial_r2_within_region_mean"],
            "leiden_ari_with_kmeans4": adjusted_rand_score(km, communities),
            "joint_ari_with_kmeans4": adjusted_rand_score(km, joint),
            "joint_moved_share": float(np.mean(joint != km)),
            "joint_partial_r2_within_region_mean": ev_joint["partial_r2_within_region_mean"],
            "joint_cut_fraction": fit["graph_cut_fraction"],
            "leiden_ari_graph_2023_vs_2024": None if g24 is None else adjusted_rand_score(communities, leiden(g24)),
            "joint_ari_graph_2023_vs_2024": None if joint_other is None else adjusted_rand_score(joint, joint_other),
        })
    write_csv(out / "network_effects.csv", pd.DataFrame(effects))

    sens = []
    for kk in cfg["network"]["k_sensitivity"]:
        a23 = E.knn_union(scores["euclid_profile"][0], kk)
        a24 = E.knn_union(scores["euclid_profile"][1], kk)
        row = {"k": kk, **E.evaluate_graph(a23, ctx["external"], ctx["regions"], later=a24)}
        row["components"] = int(connected_components(a23)[0])
        sens.append(row)
    write_csv(out / "network_k_sensitivity.csv", pd.DataFrame(sens))
    log(f"network comparison done, chosen rule: {chosen}", start)
    return {"chosen_rule": chosen, "table": table, "lead_lag": lead_lag, "effects": pd.DataFrame(effects),
            "k_sensitivity": pd.DataFrame(sens)}


def feature_space_section(ctx, out: Path, cfg: dict, draws: int, start: float) -> pd.DataFrame:
    """Why six attributes and this scaling: K=4 KMeans in four candidate spaces.

    Internal indices are computed in each space and are not comparable across
    spaces; the decision rests on independent validity and stability.
    """
    xm = ctx["Xm"]
    annual_iqr = np.subtract(*np.percentile(ctx["X23"], [75, 25], axis=0))
    spaces = {
        "structure5_v11": ("5 долей категорий (пространство v1.1)", xm[:, :, :5]),
        "profile6": ("5 долей + уровень, размах месячных значений (выбрано)", xm),
        "profile6_annual_iqr": ("5 долей + уровень, размах годовых профилей", xm / annual_iqr),
        "profile6_relative": ("5 долей + уровень относительно медианы месяца", xm - P.national_component(xm)[None]),
    }
    reference = None
    rows = []
    for key, (label, monthly) in spaces.items():
        x = np.median(monthly[:, :12], axis=1)
        z = KMeans(4, n_init=50, random_state=ctx["seed"]).fit_predict(x)
        if key == "profile6":
            reference = z
        ev = A.external_validity(z, ctx["external"], ctx["regions"])
        w, _ = knn_graph(x, k=cfg["network"]["k"])
        row = {"space": key, "label": label, "attributes": monthly.shape[2], **A.icvi(x, z, w),
               "eta2_log_wage": ev["eta2_log_wage"], "partial_r2_within_region_log_wage": ev["partial_r2_within_region_log_wage"],
               "eta2_mean": ev["eta2_mean"], "partial_r2_within_region_mean": ev["partial_r2_within_region_mean"],
               "labels": z}
        row.update(A.stability("kmeans", monthly, z, 4, cfg["network"]["k"], draws, ctx["seed"] + 100))
        rows.append(row)
    for row in rows:
        row["ari_with_profile6"] = adjusted_rand_score(reference, row.pop("labels"))
    table = pd.DataFrame(rows)
    write_csv(out / "feature_spaces.csv", table)
    log("feature spaces compared", start)
    return table


def k_section(ctx, out: Path, cfg: dict, draws: int, start: float) -> dict:
    x, w = ctx["X23"], ctx["W23"]
    rows, labels = [], {}
    for k in cfg["k_selection"]["candidates"]:
        z = KMeans(k, n_init=50, random_state=ctx["seed"]).fit_predict(x)
        labels[k] = z
        row = {"k": k, **A.icvi(x, z, w)}
        ev = A.external_validity(z, ctx["external"], ctx["regions"])
        row.update({"eta2_mean": ev["eta2_mean"], "partial_r2_within_region_mean": ev["partial_r2_within_region_mean"]})
        row.update(A.stability("kmeans", ctx["Xm"], z, k, cfg["network"]["k"], draws, ctx["seed"] + k))
        rows.append(row)
    table = pd.DataFrame(rows)
    table["stability_score"] = (table["month_bootstrap_ari_mean"] + table["territory_subsample_ari_mean"]) / 2
    best = table.sort_values(["stability_score", "month_bootstrap_ari_p10"], ascending=False).iloc[0]
    write_csv(out / "k_selection.csv", table)
    nested = pd.crosstab(labels[4], labels[5]) if 4 in labels and 5 in labels else None
    log(f"K selection done, chosen K={int(best['k'])}", start)
    return {"chosen_k": int(best["k"]), "table": table,
            "k4_vs_k5": None if nested is None else nested.values.tolist()}


def methods_section(ctx, out: Path, cfg: dict, k: int, draws: int, start: float) -> dict:
    x23, x24 = ctx["X23"], ctx["X24"]
    w23, w24 = ctx["W23"], ctx["W24"]
    n = len(x23)
    km = KMeans(k, n_init=50, random_state=ctx["seed"]).fit_predict(x23)
    rows, fitted = [], {}
    for method in cfg["methods"]["list"]:
        z = np.unique(A.fit_method(method, x23, w23, k, ctx["seed"], kmeans_labels=km), return_inverse=True)[1]
        fitted[method] = z
        z24 = A.nearest(x24, A.centroids(x23, z))
        row = {"method": method, "label": METHOD_LABELS[method],
               "family": METHOD_FAMILY.get(method, "признаки + сеть"), "k": int(z.max() + 1)}
        row.update({f"{key}_2023": value for key, value in A.icvi(x23, z, w23).items()})
        row.update({f"{key}_2024": value for key, value in A.icvi(x24, z24, w24).items()})
        row["kept_type_2024"] = float(np.mean(z24 == z))
        row["ari_with_kmeans"] = adjusted_rand_score(km, z)
        ev = A.external_validity(z, ctx["external"], ctx["regions"])
        row.update({"eta2_mean": ev["eta2_mean"], "partial_r2_within_region_mean": ev["partial_r2_within_region_mean"],
                    "eta2_log_wage": ev["eta2_log_wage"]})
        row.update(A.stability(method, ctx["Xm"], z, k, cfg["network"]["k"], draws, ctx["seed"]))
        rows.append(row)
        log(f"method {method} evaluated", start)
    table = pd.DataFrame(rows)
    elig = cfg["methods"]["eligibility"]
    table["eligible"] = ((table["min_size_2023"] >= elig["min_type_share"] * n) &
                         (table["month_bootstrap_ari_p10"] >= elig["min_month_bootstrap_ari_p10"]))
    crit23 = tuple((f"{c}_2023", s) for c, s in A.ICVI)
    crit24 = tuple((f"{c}_2024", s) for c, s in A.ICVI)
    table["copeland_all_2023"] = A.copeland(table, crit23).values
    eligible = table[table["eligible"]].copy()
    eligible["copeland_2023"] = A.copeland(eligible, crit23).values
    eligible["copeland_2024"] = A.copeland(eligible, crit24).values
    eligible["copeland_total"] = eligible["copeland_2023"] + eligible["copeland_2024"]
    simplicity = {m: i for i, m in enumerate(["kmeans", "joint_0.1", "joint_0.25", "joint_0.5", "joint_1.0",
                                               "ward", "gmm", "kefrin", "spectral_graph", "leiden_graph"])}
    eligible["simplicity"] = eligible["method"].map(simplicity)
    chosen = eligible.sort_values(["copeland_total", "simplicity"], ascending=[False, True]).iloc[0]["method"]
    table = table.merge(eligible[["method", "copeland_2023", "copeland_2024", "copeland_total"]], on="method", how="left")
    write_csv(out / "methods.csv", table)
    labels = fitted[chosen]
    fit_info = None
    if chosen.startswith("joint_"):
        _, fit_info = fit_graph_regularized_kmeans(x23, w23, km, alpha=float(chosen.split("_")[1]),
                                                   seed=ctx["seed"], max_sweeps=200)
        fit_info = {key: fit_info[key] for key in ("alpha", "converged", "convergence_status", "sweeps",
                                                   "moves_per_sweep", "objective", "attribute_SSE_over_TSS",
                                                   "graph_cut_fraction")}
    log(f"methods compared, chosen: {chosen}", start)
    return {"chosen_method": chosen, "table": table, "labels": labels, "kmeans": km, "fit": fit_info}


def tracking_section(ctx, out: Path, cfg: dict, types: np.ndarray, centers: np.ndarray, start: float) -> dict:
    window = cfg["tracking"]["persistent_window_months"]
    adjusted = T.assign_monthly(ctx["Xadj"], centers)
    absolute = T.assign_monthly(ctx["Xm"], centers)
    pers = T.persistence(adjusted, types, window)
    pers_abs = T.persistence(absolute, types, window)
    chain = T.markov(adjusted, 4)
    types24 = A.nearest(ctx["X24"], centers)
    types24_abs = A.nearest(ctx["X24_raw"], centers)
    free = T.free_reclustering(ctx["Xadj"], types, cfg["tracking"]["survival_jaccard"], ctx["seed"])
    events = T.summarize_events(free, 4, ctx["panel"].periods)
    write_csv(out / "reclustering_events.csv", pd.DataFrame(events))
    # Free re-clustering of the whole 2024 profile: which groups would a fresh typology draw?
    free24 = T.align_to(types, KMeans(4, n_init=50, random_state=ctx["seed"]).fit_predict(ctx["X24"]), 4)
    level24 = np.median(ctx["panel"].totals[:, 12:], axis=1)
    shares24 = np.median(ctx["panel"].shares[:, 12:], axis=1)
    free24_profiles = []
    for c in range(4):
        m = free24 == c
        top = pd.Series(ctx["regions"][m]).value_counts().head(3)
        free24_profiles.append({"group": c, "n": int(m.sum()), "spending_rub_median": float(np.median(level24[m])),
                                **{f"{cat}_pct_median": 100 * float(np.median(shares24[m, j]))
                                   for j, cat in enumerate(P.CATEGORIES)},
                                "latitude_median": float(np.median(ctx["lat"][m])),
                                "wage_rub_median": float(np.exp(np.nanmedian(ctx["external"]["log_wage"].to_numpy(float)[m]))),
                                "top_regions": "; ".join(f"{r} ({v})" for r, v in top.items())})
    monthly_rows = []
    for t in range(24):
        w, info = knn_graph(ctx["Xm"][:, t], k=cfg["network"]["k"])
        nxt = knn_graph(ctx["Xm"][:, t + 1], k=cfg["network"]["k"])[0] if t < 23 else None
        m = A.icvi(ctx["Xadj"][:, t], adjusted[:, t], w)
        monthly_rows.append({"period": ctx["panel"].periods[t], "edges": info["edges"], "components": info["components"],
                             "edge_jaccard_next_month": None if nxt is None else E.edge_jaccard(w, nxt),
                             **{f"n_type_{c}": int(np.sum(adjusted[:, t] == c)) for c in range(4)},
                             **{f"n_type_{c}_absolute": int(np.sum(absolute[:, t] == c)) for c in range(4)},
                             **m, **{f"free_jaccard_type_{c}": free["months"][t]["jaccard_with_base"][c] for c in range(4)}})
    write_csv(out / "monthly_network.csv", pd.DataFrame(monthly_rows))
    log("tracking done", start)
    return {"adjusted": adjusted, "absolute": absolute, "persistence": pers, "persistence_absolute": pers_abs,
            "markov": chain, "types24": types24, "types24_absolute": types24_abs,
            "free_reclustering": {"events": events, "months": free["months"],
                                  "annual_2024_overlap": T.transition_counts(types, free24, 4).tolist(),
                                  "annual_2024_profiles": free24_profiles},
            "monthly": pd.DataFrame(monthly_rows)}


def typology_section(ctx, out: Path, types: np.ndarray, start: float) -> dict:
    panel, ext, cohort = ctx["panel"], ctx["external"], ctx["cohort"]
    regions = ctx["regions"]
    level23 = np.median(panel.totals[:, :12], axis=1)
    shares23 = np.median(panel.shares[:, :12], axis=1)
    population = cohort["population_total"].astype(float).where(lambda s: s > 0).to_numpy()
    mtype = cohort["municipal_district_type"].fillna("нет данных").to_numpy()
    reps = TY.representatives(ctx["X23"], types, panel.names, regions)
    rows = []
    for c, key in enumerate(TY.TYPE_ORDER):
        m = types == c
        pop = population[m]
        largest = np.where(m)[0][np.argsort(-np.nan_to_num(population[m], nan=-1))][:3]
        comp = pd.Series(mtype[m]).value_counts(normalize=True)
        row = {"type": c, "key": key, "name": TY.TYPE_NAMES[key], "n": int(m.sum()), "share": float(m.mean()),
               "population_share_observed": float(np.nansum(pop) / np.nansum(population)),
               "population_coverage": float(np.mean(np.isfinite(pop))),
               "spending_rub_median": float(np.median(level23[m]))}
        for j, cat in enumerate(P.CATEGORIES):
            row[f"{cat}_pct_median"] = 100 * float(np.median(shares23[m, j]))
        for col in ext.columns:
            vals = ext[col].to_numpy(float)[m]
            vals = vals[np.isfinite(vals)]
            row[f"{col}_median"] = float(np.median(vals)) if len(vals) else None
            row[f"{col}_coverage"] = float(len(vals) / m.sum())
        row["wage_rub_median"] = float(np.exp(row["log_wage_median"]))
        row["latitude_median"] = float(np.median(ctx["lat"][m]))
        for label in ["муниципальный район", "муниципальный округ", "городской округ",
                      "внутригородская территория города федерального значения"]:
            row[f"composition_{label}"] = float(comp.get(label, 0.0))
        row["moscow_petersburg_share"] = float(np.mean(np.isin(regions[m], ["Москва", "Санкт-Петербург"])))
        top = pd.Series(regions[m]).value_counts().head(3)
        row["top_regions"] = "; ".join(f"{r} ({v})" for r, v in top.items())
        row["representatives"] = "; ".join(reps[c])
        row["largest_by_population"] = "; ".join(f"{panel.names[i]} ({regions[i]})" for i in largest)
        rows.append(row)
    profiles = pd.DataFrame(rows)
    write_csv(out / "types.csv", profiles)

    ci_rows = []
    indicators = {"spending_rub": level23, "wage_rub": np.exp(ext["log_wage"].to_numpy(float)),
                  "marketplaces_pct": 100 * shares23[:, 1], "horeca_pct": 100 * shares23[:, 2],
                  "food_pct": 100 * shares23[:, 3],
                  "agri_share_pct": 100 * ext["agri_share"].to_numpy(float),
                  "mining_share_pct": 100 * ext["mining_share"].to_numpy(float),
                  "market_access": ext["market_access"].to_numpy(float)}
    for name, values in indicators.items():
        ci = TY.region_bootstrap(values, types, regions, 4, draws=1000, seed=ctx["seed"])
        for c in range(4):
            ci_rows.append({"indicator": name, "type": c, "median": ci[c, 0], "ci_low": ci[c, 1], "ci_high": ci[c, 2]})
    write_csv(out / "type_intervals.csv", pd.DataFrame(ci_rows))

    designs = {"municipal_type": A._dummies(mtype), "types": A._dummies(types),
               "municipal_type_and_types": np.hstack([A._dummies(mtype), A._dummies(types)[:, 1:]])}
    loro = A.leave_one_region_out_r2(ext, designs, regions)
    write_csv(out / "leave_one_region_out.csv", loro)
    validity = A.external_validity(types, ext, regions)
    log("typology done", start)
    return {"profiles": profiles, "intervals": pd.DataFrame(ci_rows), "loro": loro, "validity": validity}


def practical_section(ctx, out: Path, cfg: dict, types: np.ndarray, track: dict, draws: int, start: float) -> dict:
    panel, regions = ctx["panel"], ctx["regions"]
    n = len(types)
    rng = np.random.default_rng(ctx["seed"])
    mean23, mean24 = panel.totals[:, :12].mean(axis=1), panel.totals[:, 12:].mean(axis=1)
    log_growth = np.log(mean24 / mean23)
    growth = 100 * (mean24 / mean23 - 1)
    k = cfg["practical"]["analogues"]
    distance = E.pairwise_euclidean(ctx["X23"])
    np.fill_diagonal(distance, np.inf)
    nearest = np.argsort(distance, axis=1, kind="stable")[:, :k]
    others = np.arange(n)
    same_region = [others[(regions == regions[i]) & (others != i)] for i in range(n)]
    same_type = [others[(types == types[i]) & (others != i)] for i in range(n)]

    def random_median(pool):
        return float(np.median(growth[rng.choice(pool, min(k, len(pool)), replace=False)])) if len(pool) else np.nan

    def network_in_region(i):
        pool = same_region[i]
        return float(np.median(growth[pool[np.argsort(distance[i, pool], kind="stable")][:k]])) if len(pool) >= 3 else np.nan

    predictions = {
        "network_analogues": np.median(growth[nearest], axis=1),
        "network_analogues_same_region": np.array([network_in_region(i) for i in range(n)]),
        "random_same_region": np.array([random_median(p) for p in same_region]),
        "region_median_leave_one_out": np.array([np.median(growth[p]) if len(p) else np.nan for p in same_region]),
        "random_same_type": np.array([random_median(p) for p in same_type]),
        "national_median": np.full(n, np.median(growth)),
    }
    primary = ("network_analogues", "random_same_region", "region_median_leave_one_out", "national_median")
    common = np.all(np.isfinite(np.column_stack([predictions[name] for name in primary])), axis=1)
    diagnostic_common = np.all(np.isfinite(np.column_stack(list(predictions.values()))), axis=1)
    errors = {name: np.abs(pred - growth) for name, pred in predictions.items()}
    prediction_rows = []
    for name, pred in predictions.items():
        prediction_rows.extend({"entity_id": panel.ids[i], "region": regions[i], "predictor": name,
                                "observed_growth_pct": growth[i], "predicted_growth_pct": pred[i],
                                "absolute_error_pp": errors[name][i], "common_sample": bool(common[i]),
                                "diagnostic_common_sample": bool(diagnostic_common[i])}
                               for i in range(n))
    write_csv(out / "analogue_predictions.csv", pd.DataFrame(prediction_rows))
    rows = [{"predictor": name, "n": int(common.sum()), "unit": "percentage_point", "mae_pp": float(err[common].mean()),
             "median_ae_pp": float(np.median(errors[name][common]))} for name in primary
            for err in [errors[name]]]
    diagnostic_rows = [{"predictor": name, "n": int(diagnostic_common.sum()), "unit": "percentage_point",
                        "mae_pp": float(err[diagnostic_common].mean()),
                        "median_ae_pp": float(np.median(err[diagnostic_common]))} for name, err in errors.items()]
    pairs = [("network_analogues", "random_same_region"), ("network_analogues", "national_median"),
             ("network_analogues", "region_median_leave_one_out")]
    comparisons = []
    for a, b in pairs:
        diff = errors[a] - errors[b]
        mean, low, high = TY.paired_region_bootstrap(np.where(common, diff, np.nan), regions, draws, ctx["seed"])
        comparisons.append({"better": a, "baseline": b, "n": int(common.sum()), "unit": "percentage_point",
                            "mae_difference_pp": mean, "ci_low": low, "ci_high": high})
    write_csv(out / "analogues.csv", pd.DataFrame(rows))
    write_csv(out / "analogue_comparisons.csv", pd.DataFrame(comparisons))
    diagnostic_comparisons = []
    for a, b in [("network_analogues_same_region", "region_median_leave_one_out"),
                 ("network_analogues", "random_same_type")]:
        mean, low, high = TY.paired_region_bootstrap(
            np.where(diagnostic_common, errors[a] - errors[b], np.nan), regions, draws, ctx["seed"])
        diagnostic_comparisons.append({"better": a, "baseline": b, "n": int(diagnostic_common.sum()),
                                       "unit": "percentage_point", "mae_difference_pp": mean,
                                       "ci_low": low, "ci_high": high})
    write_csv(out / "analogues_diagnostic.csv", pd.DataFrame(diagnostic_rows))
    write_csv(out / "analogue_comparisons_diagnostic.csv", pd.DataFrame(diagnostic_comparisons))

    # Marketplace convergence and level divergence by 2023 type.
    mk23 = 100 * np.median(panel.shares[:, :12, 1], axis=1)
    mk24 = 100 * np.median(panel.shares[:, 12:, 1], axis=1)
    rel = ctx["rel_level"]
    d_rel = np.median(rel[:, 12:], axis=1) - np.median(rel[:, :12], axis=1)
    conv = []
    for c, key in enumerate(TY.TYPE_ORDER):
        m = types == c
        g_mean, g_low, g_high = TY.paired_region_bootstrap(np.where(m, log_growth, np.nan), regions, draws, ctx["seed"])
        r_mean, r_low, r_high = TY.paired_region_bootstrap(np.where(m, d_rel, np.nan), regions, draws, ctx["seed"])
        mk = TY.region_bootstrap(mk24 - mk23, m.astype(int), regions, 2, draws=draws, seed=ctx["seed"])[1]
        conv.append({"type": c, "key": key, "name": TY.TYPE_NAMES[key], "n": int(m.sum()),
                     "marketplaces_pct_2023": float(np.median(mk23[m])), "marketplaces_pct_2024": float(np.median(mk24[m])),
                     "marketplaces_ratio_2024_to_2023": float(np.median(mk24[m]) / np.median(mk23[m])),
                     "marketplaces_change_pp_median": mk[0], "marketplaces_change_ci_low": mk[1], "marketplaces_change_ci_high": mk[2],
                     "nominal_growth_mean_pct": 100 * (np.exp(g_mean) - 1), "nominal_growth_ci_low_pct": 100 * (np.exp(g_low) - 1),
                     "nominal_growth_ci_high_pct": 100 * (np.exp(g_high) - 1),
                     "relative_level_change_log": r_mean, "relative_level_change_ci_low": r_low,
                     "relative_level_change_ci_high": r_high,
                     "left_type_2024_adjusted_share": float(np.mean(track["types24"][m] != c)),
                     "left_type_2024_absolute_share": float(np.mean(track["types24_absolute"][m] != c))})
    convergence = pd.DataFrame(conv)
    write_csv(out / "type_dynamics.csv", convergence)

    # Gap between big-city services and the periphery: change of the relative level.
    metro, periphery = types == 0, types == 2
    unique = np.unique(regions)
    members = {r: np.where(regions == r)[0] for r in unique}
    gaps = []
    for _ in range(draws):
        chosen = np.concatenate([members[r] for r in rng.choice(unique, len(unique), replace=True)])
        dm, dp = d_rel[chosen][metro[chosen]], d_rel[chosen][periphery[chosen]]
        if len(dm) and len(dp):
            gaps.append(dm.mean() - dp.mean())
    gap = {"metro_minus_periphery_relative_level_change_log": float(d_rel[metro].mean() - d_rel[periphery].mean()),
           "ci_low": float(np.percentile(gaps, 2.5)), "ci_high": float(np.percentile(gaps, 97.5)),
           "level_ratio_2023": float(np.exp(np.median(np.median(rel[metro, :12], axis=1)) -
                                            np.median(np.median(rel[periphery, :12], axis=1)))),
           "level_ratio_2024": float(np.exp(np.median(np.median(rel[metro, 12:], axis=1)) -
                                            np.median(np.median(rel[periphery, 12:], axis=1))))}
    remote = types == 3
    marketplace_gap = {"periphery_median_pct_2023": float(np.median(mk23[periphery])),
                       "remote_median_pct_2023": float(np.median(mk23[remote])),
                       "periphery_median_pct_2024": float(np.median(mk24[periphery])),
                       "remote_median_pct_2024": float(np.median(mk24[remote]))}
    marketplace_gap["gap_2023_pp"] = marketplace_gap["periphery_median_pct_2023"] - marketplace_gap["remote_median_pct_2023"]
    marketplace_gap["gap_2024_pp"] = marketplace_gap["periphery_median_pct_2024"] - marketplace_gap["remote_median_pct_2024"]
    marketplace_gap["change_pp"] = marketplace_gap["gap_2024_pp"] - marketplace_gap["gap_2023_pp"]
    marketplace_gaps = []
    gap_rng = np.random.default_rng(ctx["seed"])
    for _ in range(draws):
        chosen = np.concatenate([members[r] for r in gap_rng.choice(unique, len(unique), replace=True)])
        p, r = chosen[periphery[chosen]], chosen[remote[chosen]]
        if len(p) and len(r):
            marketplace_gaps.append((np.median(mk24[p]) - np.median(mk24[r])) -
                                    (np.median(mk23[p]) - np.median(mk23[r])))
    marketplace_gap["conditional_region_ci_95_change_pp"] = np.percentile(marketplace_gaps, [2.5, 97.5]).tolist()
    marketplace_gap["bootstrap_valid_draws"] = len(marketplace_gaps)
    marketplace_gap["fixed_2023_types"] = True

    pers = track["persistence"]
    mask, new = pers["persistent_change_mask"], pers["persistent_new_type"]
    population = ctx["cohort"]["population_total"].astype(float).to_numpy()
    changes = pd.DataFrame({"entity_id": panel.ids[mask], "name": panel.names[mask], "region": regions[mask],
                            "type_2023": types[mask], "type_2024_h2": new[mask],
                            "population": population[mask]}).sort_values(["type_2023", "type_2024_h2", "population"],
                                                                         ascending=[True, True, False])
    write_csv(out / "persistent_changes.csv", changes)
    log("practical checks done", start)
    return {"analogues": pd.DataFrame(rows), "analogue_comparisons": pd.DataFrame(comparisons),
            "type_dynamics": convergence, "gap": gap, "persistent_changes": changes,
            "analogues_diagnostic": pd.DataFrame(diagnostic_rows),
            "analogue_comparisons_diagnostic": pd.DataFrame(diagnostic_comparisons),
            "marketplace_gap_pp": marketplace_gap}


# ------------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/v12.json")
    parser.add_argument("--output", type=Path, help="Override the output directory")
    parser.add_argument("--quick", action="store_true", help="Few bootstrap draws for a smoke run")
    args = parser.parse_args()
    start = time.time()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    out = args.output or ROOT / cfg["output"]
    out.mkdir(parents=True, exist_ok=True)
    draws_k = 5 if args.quick else cfg["k_selection"]["draws"]
    draws_m = 3 if args.quick else cfg["methods"]["stability_draws"]
    draws_b = 200 if args.quick else cfg["practical"]["bootstrap_draws"]

    inputs = cfg["inputs"]
    panel = P.read_panel(ROOT / inputs["panel"])
    scaler = P.fit_scaler(panel, cfg["features"]["development_months"])
    xm = P.monthly_attributes(panel, scaler)
    x23 = P.annual_profile(xm, slice(0, 12))
    xadj = P.remove_national_wave(xm, x23)
    external, cohort = load_external(inputs, panel.ids)
    coords = centroids_from_map(ROOT / inputs["map"], panel.ids,
                                cohort["municipal_district_center_lat"].to_numpy(float),
                                cohort["municipal_district_center_lon"].to_numpy(float))
    ctx = {"seed": cfg["seed"], "panel": panel, "Xm": xm, "X23": x23, "X24_raw": P.annual_profile(xm, slice(12, 24)),
           "Xadj": xadj, "X24": P.annual_profile(xadj, slice(12, 24)), "rel_level": P.relative_level(panel),
           "external": external, "cohort": cohort, "regions": panel.regions,
           "lat": coords["lat"], "lon": coords["lon"], "road": read_road_graph(inputs, panel.ids)}
    ctx["W23"] = knn_graph(x23, k=cfg["network"]["k"])[0]
    ctx["W24"] = knn_graph(ctx["X24"], k=cfg["network"]["k"])[0]
    ctx["kmeans4"] = KMeans(4, n_init=50, random_state=cfg["seed"]).fit_predict(x23)
    log("inputs loaded", start)

    network = network_section(ctx, out, cfg, start)
    if network["chosen_rule"] != "euclid_profile":
        print("WARNING: the selection rule picked", network["chosen_rule"], "- graph W still uses the six-attribute profile")
    spaces = feature_space_section(ctx, out, cfg, draws_k, start)
    ksel = k_section(ctx, out, cfg, draws_k, start)
    methods = methods_section(ctx, out, cfg, ksel["chosen_k"], draws_m, start)
    if ksel["chosen_k"] != 4:
        raise RuntimeError("Type naming and the report are defined for K=4; review the K selection")

    raw_labels = methods["labels"]
    naming = TY.name_types(np.median(panel.totals[:, :12], axis=1), np.median(panel.shares[:, :12], axis=1), raw_labels)
    types = TY.canonical_labels(raw_labels, naming)
    centers = np.array([x23[types == c].mean(axis=0) for c in range(4)])
    track = tracking_section(ctx, out, cfg, types, centers, start)
    typology = typology_section(ctx, out, types, start)
    practical = practical_section(ctx, out, cfg, types, track, draws_b, start)

    labels = pd.DataFrame({"entity_id": panel.ids, "name": panel.names, "region": panel.regions,
                           "type_2023": types, "type_2024": track["types24"],
                           "type_2024_absolute": track["types24_absolute"],
                           "kmeans_type": TY.canonical_labels(methods["kmeans"], TY.name_types(
                               np.median(panel.totals[:, :12], axis=1), np.median(panel.shares[:, :12], axis=1),
                               methods["kmeans"])),
                           "monthly_types": ["".join(map(str, row)) for row in track["adjusted"]],
                           "persistent_change": track["persistence"]["persistent_change_mask"],
                           "lat": ctx["lat"], "lon": ctx["lon"]})
    write_csv(out / "labels.csv", labels)
    np.savez_compressed(out / "model.npz", centers=centers, structure_center=scaler.structure_center,
                        scale=scaler.scale, level_center=scaler.level_center, ids=panel.ids)

    summary = {
        "version": cfg["version"],
        "inputs": {name: {"path": path, "sha256": P.file_sha256(ROOT / path)} for name, path in inputs.items()},
        "panel": {"entities": len(panel.ids), "months": len(panel.periods), "regions": int(len(np.unique(panel.regions)))},
        "scaler": scaler.as_dict(),
        "centroids": {key: value for key, value in coords.items() if key not in ("lat", "lon")},
        "feature_spaces": spaces.to_dict(orient="records"),
        "network": {"chosen_rule": network["chosen_rule"], "lead_lag": network["lead_lag"],
                    "rules": network["table"].to_dict(orient="records")},
        "k_selection": {"chosen_k": ksel["chosen_k"], "k4_vs_k5": ksel["k4_vs_k5"],
                        "table": ksel["table"].to_dict(orient="records")},
        "methods": {"chosen": methods["chosen_method"], "fit": methods["fit"],
                    "validation_status": "2024 участвовал в выборе модели; независимой проверки на этом годе нет.",
                    "ari_final_vs_kmeans": adjusted_rand_score(methods["kmeans"], raw_labels),
                    "moved_vs_kmeans_share": float(np.mean(labels["kmeans_type"].to_numpy() != types)),
                    "table": methods["table"].to_dict(orient="records")},
        "types": {"names": {str(i): TY.TYPE_NAMES[key] for i, key in enumerate(TY.TYPE_ORDER)},
                  "keys": list(TY.TYPE_ORDER), "sizes": np.bincount(types, minlength=4).tolist(),
                  "validity": typology["validity"], "profiles": typology["profiles"].to_dict(orient="records"),
                  "leave_one_region_out": typology["loro"].to_dict(orient="records")},
        "tracking": {
            "persistence": {k: v for k, v in track["persistence"].items() if not isinstance(v, np.ndarray)},
            "persistence_absolute": {k: v for k, v in track["persistence_absolute"].items() if not isinstance(v, np.ndarray)},
            "markov": track["markov"],
            "annual_transitions": T.transition_counts(types, track["types24"], 4).tolist(),
            "annual_transitions_absolute": T.transition_counts(types, track["types24_absolute"], 4).tolist(),
            "persistent_change_matrix": T.transition_counts(
                types[track["persistence"]["persistent_change_mask"]],
                track["persistence"]["persistent_new_type"][track["persistence"]["persistent_change_mask"]], 4).tolist(),
            "free_reclustering": track["free_reclustering"],
            "national_component": {
                "periods": list(panel.periods),
                "median_share_pct": (100 * np.median(panel.shares, axis=0)).tolist(),
                "median_spending_rub": np.median(panel.totals, axis=0).tolist()},
        },
        "practical": {"analogues": practical["analogues"].to_dict(orient="records"),
                      "analogue_comparisons": practical["analogue_comparisons"].to_dict(orient="records"),
                      "analogues_diagnostic": practical["analogues_diagnostic"].to_dict(orient="records"),
                      "analogue_comparisons_diagnostic": practical["analogue_comparisons_diagnostic"].to_dict(orient="records"),
                      "marketplace_gap_pp": practical["marketplace_gap_pp"],
                      "type_dynamics": practical["type_dynamics"].to_dict(orient="records"),
                      "gap": practical["gap"], "persistent_changes": int(len(practical["persistent_changes"]))},
    }
    dump(out / "summary.json", summary)
    versions = {}
    for package in ("numpy", "scipy", "scikit-learn", "pandas", "networkx", "igraph", "leidenalg"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    dump(out / "provenance.json", {"command": "python -m scripts.run_v12" + (" --quick" if args.quick else ""),
                                   "config": cfg, "python": platform.python_version(), "platform": platform.platform(),
                                   "packages": versions, "draws": {"k_selection": draws_k, "methods": draws_m,
                                                                   "bootstrap": draws_b}})
    if not args.quick:
        from scripts.validate_v12_findings import refresh
        refresh(ROOT, out, cached_results=out)
    log(f"done: {out}", start)


if __name__ == "__main__":
    main()
