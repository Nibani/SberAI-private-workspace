"""Is the choice among the eligible clustering methods driven by noise?

    python -m scripts.check_method_ties            # writes reports/v1.2-method-ties
    python -m scripts.check_method_ties --check    # recompute and compare with the saved tables

The v1.2 method comparison (reports/v1.2/methods.csv) keeps six eligible variants
(KMeans, KEFRiN-like, joint model with alpha 0.1/0.25/0.5/1) and picks the one with the
largest Copeland score over six ICVI on 2023 and 2024. This script refits the ten methods
exactly as scripts/run_v12.py does (checked against methods.csv), keeps every partition
fixed and re-evaluates the ICVI on random subsamples of 80% of the regions (whole regions,
without replacement; the kNN graph is restricted to the drawn municipalities). It reports

* how often each eligible method would win the same Copeland rule on a subsample;
* paired ICVI differences between the selected joint model (alpha = 0.5) and each method;
* how much the eligible partitions differ from each other on the full data.

Partitions are not refitted inside a draw: the question is whether the ICVI ranking of the
given partitions is stable to the composition of territories, not the stability of the
algorithms (that is reported in methods.csv). Subsample spread is a sensitivity measure,
not a confidence interval for the full-sample value.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sbercluster import attributed as A, panel as P  # noqa: E402
from sbercluster.graph import knn_graph  # noqa: E402

OUTPUT = ROOT / "reports/v1.2-method-ties"
SELECTED = "joint_0.5"
SIMPLICITY = ["kmeans", "joint_0.1", "joint_0.25", "joint_0.5", "joint_1.0",
              "ward", "gmm", "kefrin", "spectral_graph", "leiden_graph"]   # tie-break of run_v12
LABELS = {"kmeans": "KMeans", "ward": "Ward", "gmm": "Гауссова смесь", "spectral_graph": "Спектральная (граф)",
          "leiden_graph": "Leiden (граф)", "kefrin": "KEFRiN-подобный", "joint_0.1": "Совместная α=0,1",
          "joint_0.25": "Совместная α=0,25", "joint_0.5": "Совместная α=0,5", "joint_1.0": "Совместная α=1"}
ICVI_NAMES = [c for c, _ in A.ICVI]


def context(source: Path) -> dict:
    """Inputs exactly as in scripts/run_v12.main."""
    cfg = json.loads((source / "configs/v12.json").read_text(encoding="utf-8"))
    panel = P.read_panel(source / cfg["inputs"]["panel"])
    scaler = P.fit_scaler(panel, cfg["features"]["development_months"])
    xm = P.monthly_attributes(panel, scaler)
    x23 = P.annual_profile(xm, slice(0, 12))
    x24 = P.annual_profile(P.remove_national_wave(xm, x23), slice(12, 24))
    k = cfg["network"]["k"]
    return {"cfg": cfg, "panel": panel, "x23": x23, "x24": x24,
            "w23": knn_graph(x23, k=k)[0], "w24": knn_graph(x24, k=k)[0]}


def fit_partitions(ctx: dict) -> dict:
    cfg = ctx["cfg"]
    km = KMeans(cfg["methods"]["k"], n_init=50, random_state=cfg["seed"]).fit_predict(ctx["x23"])
    out = {}
    for method in cfg["methods"]["list"]:
        z = np.unique(A.fit_method(method, ctx["x23"], ctx["w23"], cfg["methods"]["k"], cfg["seed"], kmeans_labels=km),
                      return_inverse=True)[1]
        out[method] = (z, A.nearest(ctx["x24"], A.centroids(ctx["x23"], z)))
    return out


def scores(ctx: dict, parts: dict, mask: np.ndarray | None = None) -> pd.DataFrame:
    rows = []
    idx = np.arange(len(ctx["x23"])) if mask is None else np.where(mask)[0]
    w23 = ctx["w23"][idx][:, idx]
    w24 = ctx["w24"][idx][:, idx]
    for method, (z23, z24) in parts.items():
        a = A.icvi(ctx["x23"][idx], z23[idx], w23)
        b = A.icvi(ctx["x24"][idx], z24[idx], w24)
        rows.append({"method": method, **{f"{c}_2023": a[c] for c in ICVI_NAMES},
                     **{f"{c}_2024": b[c] for c in ICVI_NAMES}})
    return pd.DataFrame(rows)


def select(table: pd.DataFrame, eligible: list[str]) -> tuple[str, str]:
    """The run_v12 rule on 2023+2024, and the same rule on 2023 only."""
    t = table[table.method.isin(eligible)].reset_index(drop=True)
    c23 = A.copeland(t, tuple((f"{c}_2023", s) for c, s in A.ICVI)).to_numpy()
    c24 = A.copeland(t, tuple((f"{c}_2024", s) for c, s in A.ICVI)).to_numpy()
    rank = t.method.map(SIMPLICITY.index).to_numpy()
    both = t.method[np.lexsort((rank, -(c23 + c24)))[0]]
    only23 = t.method[np.lexsort((rank, -c23))[0]]
    return both, only23


def run(source: Path, draws: int, seed: int, fraction: float = 0.8):
    ctx = context(source)
    parts = fit_partitions(ctx)
    saved = pd.read_csv(source / "reports/v1.2/methods.csv").set_index("method")
    full = scores(ctx, parts).set_index("method")
    for col in [f"{c}_{y}" for c in ICVI_NAMES for y in (2023, 2024)]:
        np.testing.assert_allclose(full[col].to_numpy(float), saved.loc[full.index, col].to_numpy(float),
                                   rtol=1e-5, atol=1e-6, err_msg=f"{col} does not reproduce methods.csv (stored with 6 significant digits)")
    eligible = saved.index[saved.eligible.astype(bool)].tolist()
    chosen_full, chosen23_full = select(full.reset_index(), eligible)
    if chosen_full != SELECTED:
        raise ValueError(f"Full-data rule selects {chosen_full}, expected {SELECTED}")
    labels = pd.read_csv(source / "reports/v1.2/labels.csv").set_index("entity_id").loc[ctx["panel"].ids]
    if adjusted_rand_score(labels.type_2023, parts[SELECTED][0]) != 1.0:
        raise ValueError("Refitted joint model differs from labels.csv")

    regions = np.unique(ctx["panel"].regions)
    rng = np.random.default_rng(seed)
    winners, diffs = [], []
    for b in range(draws):
        drawn = rng.choice(regions, size=int(round(fraction * len(regions))), replace=False)
        mask = np.isin(ctx["panel"].regions, drawn)
        t = scores(ctx, parts, mask)
        both, only23 = select(t, eligible)
        winners.append({"draw": b, "municipalities": int(mask.sum()), "winner_2023_2024": both, "winner_2023": only23})
        ref = t.set_index("method").loc[SELECTED]
        for method in parts:
            if method != SELECTED:
                row = t.set_index("method").loc[method]
                for c, sign in A.ICVI:
                    for y in (2023, 2024):
                        diffs.append({"draw": b, "method": method, "icvi": c, "year": y,
                                      "selected_better": bool(sign * (ref[f"{c}_{y}"] - row[f"{c}_{y}"]) > 0),
                                      "difference": float(ref[f"{c}_{y}"] - row[f"{c}_{y}"])})
    winners = pd.DataFrame(winners)
    diffs = pd.DataFrame(diffs)

    freq = []
    for method in eligible:
        freq.append({"method": method, "label": LABELS[method],
                     "win_share_2023_2024": float((winners.winner_2023_2024 == method).mean()),
                     "win_share_2023": float((winners.winner_2023 == method).mean()),
                     "ari_with_selected": adjusted_rand_score(parts[SELECTED][0], parts[method][0]),
                     "municipalities_differing_from_selected": int(_differing(parts[SELECTED][0], parts[method][0]))})
    freq = pd.DataFrame(freq)

    full_diff = []
    for method in parts:
        if method == SELECTED:
            continue
        for c, sign in A.ICVI:
            for y in (2023, 2024):
                part = diffs[(diffs.method == method) & (diffs.icvi == c) & (diffs.year == y)]
                d = part.difference.to_numpy()
                full_diff.append({"method": method, "label": LABELS[method], "eligible": method in eligible,
                                  "icvi": c, "year": y, "direction": "больше лучше" if sign > 0 else "меньше лучше",
                                  "full_difference": float(full.loc[SELECTED, f"{c}_{y}"] - full.loc[method, f"{c}_{y}"]),
                                  "subsample_p2_5": float(np.nanpercentile(d, 2.5)),
                                  "subsample_p97_5": float(np.nanpercentile(d, 97.5)),
                                  "selected_better_share": float(part.selected_better.mean())})
    pairwise = pd.DataFrame(full_diff)
    pairwise["consistent_direction"] = (pairwise.selected_better_share >= 0.975) | (pairwise.selected_better_share <= 0.025)
    meta = {"chosen_full": chosen_full, "chosen_2023_only_full": chosen23_full, "eligible": eligible,
            "regions": len(regions), "fraction": fraction, "draws": draws, "seed": seed}
    return freq, pairwise, winners, meta


def _differing(a: np.ndarray, b: np.ndarray) -> int:
    """Municipalities whose label differs after the best one-to-one matching of groups."""
    from scipy.optimize import linear_sum_assignment
    table = pd.crosstab(a, b).to_numpy()
    r, c = linear_sum_assignment(-table)
    return int(len(a) - table[r, c].sum())


def render(freq: pd.DataFrame, pairwise: pd.DataFrame, meta: dict) -> str:
    def pct(v):
        return f"{100 * v:.1f}%"
    elig = pairwise[pairwise.eligible]
    other = pairwise[~pairwise.eligible]
    f = freq.set_index("method")
    joint = [m for m in f.index if m.startswith("joint_")]
    joint_both, joint_23 = f.loc[joint, "win_share_2023_2024"].sum(), f.loc[joint, "win_share_2023"].sum()
    sel_both, sel_23 = f.loc[SELECTED, "win_share_2023_2024"], f.loc[SELECTED, "win_share_2023"]
    others = [m for m in joint if m != SELECTED]
    alpha_min, alpha_max = f.loc[others, "municipalities_differing_from_selected"].min(), f.loc[others, "municipalities_differing_from_selected"].max()
    ari_min, ari_max = f.loc[others, "ari_with_selected"].min(), f.loc[others, "ari_with_selected"].max()
    km_diff = f.loc["kmeans", "municipalities_differing_from_selected"]
    lines = [
        "# Устойчивость выбора метода к составу территорий",
        "",
        "Вопрос: не определяется ли выбор совместной модели α = 0,5 среди допустимых методов случайными "
        "различиями ICVI в третьем знаке.",
        "",
        "## Воспроизведение",
        "",
        "```",
        "python -m scripts.check_method_ties            # около 10 минут, один поток",
        "python -m scripts.check_method_ties --check",
        "python -m unittest discover -s tests -p test_method_ties.py",
        "```",
        "",
        "## Как устроена проверка",
        "",
        "- Десять методов переобучаются тем же кодом, что в `scripts/run_v12.py`; ICVI на полных данных совпадают "
        "с `reports/v1.2/methods.csv`, разбиение α = 0,5 совпадает с `labels.csv` (ARI = 1).",
        f"- Разбиения фиксированы. В каждом из {meta['draws']} повторов случайно берутся "
        f"{round(meta['fraction'] * meta['regions'])} из {meta['regions']} регионов целиком (без возвращения), "
        "граф kNN ограничивается вытянутыми МО, и шесть ICVI пересчитываются за 2023 и 2024 годы.",
        "- К каждой подвыборке применяется то же правило выбора, что в v1.2: Копленд по шести ICVI за 2023 и "
        "2024 среди шести допустимых методов, при равенстве — более простой. Отдельно — правило только по 2023.",
        "- Разброс по подвыборкам показывает чувствительность ранжирования к составу территорий; это не "
        "доверительный интервал для значения на полных данных.",
        "",
        f"На полных данных правило выбирает `{meta['chosen_full']}`, правило только по 2023 — `{meta['chosen_2023_only_full']}`.",
        "",
        "## Вывод",
        "",
        f"- Семейство выбрано устойчиво: совместная модель с графовым штрафом (любое α) выигрывает правило выбора в "
        f"{pct(joint_both)} подвыборок (только по 2023 — в {pct(joint_23)}); KMeans и KEFRiN-подобный вариант — "
        f"в {pct(1 - joint_both)}.",
        f"- Конкретный вес α не определяется: α = 0,5 выигрывает в {pct(sel_both)} подвыборок "
        f"(только по 2023 — в {pct(sel_23)}). Выбор α внутри допустимых вариантов зависит от состава территорий.",
        f"- На разбиение это влияет мало: варианты α отличаются от α = 0,5 у {alpha_min}–{alpha_max} МО из 2016 "
        f"(ARI {ari_min:.3f}–{ari_max:.3f}); KMeans — у {km_diff} МО.",
        "- Относительно KMeans α = 0,5 даёт устойчивый размен, а не равенство: по SW и AVI 2023 лучше во всех "
        "подвыборках, по CH 2023 хуже во всех (таблица ниже). Поэтому корректная формулировка — «графовая "
        "регуляризация меняет баланс индексов предсказуемо; точное α выбрано по совокупному правилу и "
        "неустойчиво», а не «α = 0,5 лучше остальных».",
        "",
        "## Как часто выигрывает каждый допустимый метод",
        "",
        "| Метод | победы, 2023+2024 | победы, только 2023 | ARI с α = 0,5 | МО с другой группой |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in freq.itertuples():
        lines.append(f"| {r.label} | {pct(r.win_share_2023_2024)} | {pct(r.win_share_2023)} | "
                     f"{r.ari_with_selected:.3f} | {r.municipalities_differing_from_selected} |")
    lines += [
        "",
        "## Парные различия ICVI: α = 0,5 минус другой метод",
        "",
        "«Устойчиво» — α = 0,5 лучше (или хуже) другого метода по данному индексу не менее чем в 97,5% подвыборок.",
        "",
        f"- Допустимые методы: устойчивое направление в {int(elig.consistent_direction.sum())} из {len(elig)} "
        "пар «индекс × год × метод».",
        f"- Недопустимые методы (Ward, GMM, спектральный, Leiden): устойчивое направление в "
        f"{int(other.consistent_direction.sum())} из {len(other)} пар.",
        "",
        "| Метод | допустим | индекс | год | разность на полных данных | 2,5–97,5% подвыборок | доля, где α = 0,5 лучше |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for r in pairwise.itertuples():
        lines.append(f"| {r.label} | {'да' if r.eligible else 'нет'} | {r.icvi} | {r.year} | {r.full_difference:+.4g} | "
                     f"[{r.subsample_p2_5:+.4g}; {r.subsample_p97_5:+.4g}] | {pct(r.selected_better_share)} |")
    lines += [
        "",
        "## Ограничения",
        "",
        "- Разбиения не переобучаются внутри повторов; устойчивость самих алгоритмов — в `methods.csv` "
        "(бутстреп месяцев и подвыборки территорий).",
        "- Выбор α использовал 2024 год; эта проверка не делает его независимым тестом.",
        "- Подвыборка 80% регионов — мера чувствительности, а не оценка выборочной ошибки.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    started = time.time()
    with threadpool_limits(limits=1):
        freq, pairwise, winners, meta = run(ROOT, args.draws, args.seed)
    if args.check:
        pd.testing.assert_frame_equal(freq, pd.read_csv(args.output / "win_frequency.csv"), check_exact=False, rtol=1e-6)
        pd.testing.assert_frame_equal(pairwise, pd.read_csv(args.output / "pairwise_icvi.csv"), check_exact=False, rtol=1e-6)
        print(json.dumps({"check": "ok", "seconds": round(time.time() - started)}))
        return
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    freq.to_csv(out / "win_frequency.csv", index=False, float_format="%.10g", lineterminator="\n")
    pairwise.to_csv(out / "pairwise_icvi.csv", index=False, float_format="%.10g", lineterminator="\n")
    winners.to_csv(out / "draws.csv", index=False, lineterminator="\n")
    inputs = ["data/v12/panel.csv.gz", "configs/v12.json", "reports/v1.2/methods.csv", "reports/v1.2/labels.csv"]
    meta["inputs_sha256"] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in inputs}
    meta["command"] = "python -m scripts.check_method_ties"
    (out / "summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(render(freq, pairwise, meta), encoding="utf-8")
    print(json.dumps({"output": str(out), "seconds": round(time.time() - started)}))


if __name__ == "__main__":
    main()
