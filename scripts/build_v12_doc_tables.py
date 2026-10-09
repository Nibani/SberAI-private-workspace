"""Write the v1.2 result tables into the Markdown documents from reports/v1.2.

Tables live between markers such as ``<!-- v12:networks -->`` and ``<!-- /v12 -->``;
everything between them is regenerated, the surrounding text is left untouched.
This keeps every number in the documentation identical to the pipeline output.

    python -m scripts.build_v12_doc_tables            # rewrite tables in docs/*.md and README.md
    python -m scripts.build_v12_doc_tables --check    # fail if any table is stale
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sbercluster.typology import TYPE_NAMES, TYPE_ORDER, TYPE_SHORT  # noqa: E402

DOCUMENTS = ("README.md", "docs/REPORT_RU.md", "docs/METHODOLOGY_RU.md", "docs/NETWORKS_RU.md", "docs/TYPES_RU.md",
             "docs/DYNAMICS_RU.md")
MARKER = re.compile(r"(<!-- v12:(?P<name>[a-z0-9_]+) -->\n)(?P<body>.*?)(<!-- /v12 -->)", re.S)
FACT = re.compile(r"(<!-- f:(?P<name>[a-z0-9_]+) -->)(?P<body>.*?)(<!-- /f -->)")
SHORT = [TYPE_SHORT[k] for k in TYPE_ORDER]
# Regions of the Far Eastern federal district present in the panel naming of the SberIndex registry.
FAR_EAST = {"Республика Саха (Якутия)", "Сахалинская область", "Амурская область", "Хабаровский край",
            "Приморский край", "Камчатский край", "Еврейская автономная область", "Забайкальский край",
            "Магаданская область", "Чукотский автономный округ", "Республика Бурятия"}


def num(value, digits=3, percent=False, signed=False) -> str:
    """Russian number format: comma decimal separator, space thousands separator."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    value = float(value) * (100 if percent else 1)
    text = f"{value:+,.{digits}f}" if signed else f"{value:,.{digits}f}"
    text = text.replace(",", " ").replace(".", ",").replace("-", "−")
    return text + ("%" if percent else "")


def integer(value) -> str:
    return f"{int(round(float(value))):,}".replace(",", " ").replace("-", "−")


def table(header: list[str], rows: list[list[str]], align: str | None = None) -> str:
    align = align or "l" + "r" * (len(header) - 1)
    marks = {"l": "---", "r": "---:", "c": ":---:"}
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(marks[a] for a in align) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


class Tables:
    def __init__(self, directory: Path):
        self.d = directory
        self.summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))

    def csv(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.d / name)

    # ------------------------------------------------------------ network
    def networks(self) -> str:
        t = self.csv("networks.csv")
        chosen = self.summary["network"]["chosen_rule"]
        rows = []
        for _, r in t.iterrows():
            status = ("**выбрано**" if r["rule"] == chosen else
                      "география" if r["geography_dominated"] else "")
            rows.append([("**" + r["label"] + "**") if r["rule"] == chosen else r["label"],
                         num(r["same_region_share"], 2), "статично" if r["static"] else num(r["stability_jaccard"], 3),
                         num(r["assort_mean"], 3), num(r["assort_within_region_mean"], 3),
                         "—" if pd.isna(r["copeland"]) else integer(r["copeland"]), status])
        return table(["Правило рёбер (k = 15)", "Доля рёбер внутри региона", "Совпадение рёбер 2023→2024",
                      "Ассортативность", "Ассортативность внутри регионов", "Коупленд", ""], rows, "lrrrrrl")

    def network_assortativity(self) -> str:
        t = self.csv("networks.csv")
        labels = {"log_wage": "Зарплата", "agri_share": "Сельское х-во", "manuf_share": "Обрабатывающая",
                  "mining_share": "Добыча", "public_share": "Бюджетный сектор", "log_pop": "Население",
                  "log_jobs_per_res": "Рабочие места", "market_access": "Доступность рынков"}
        rows = [[r["label"]] + [num(r[f"assort_within_region_{k}"], 2) for k in labels] for _, r in t.iterrows()]
        return table(["Правило"] + list(labels.values()), rows)

    def network_effects(self) -> str:
        t = self.csv("network_effects.csv")
        def same(column, r):
            return "статично" if column not in r or pd.isna(r[column]) else num(r[column], 2)
        rows = [[r["label"], integer(r["leiden_communities"]), num(r["leiden_modularity"], 3),
                 num(r["leiden_partial_r2_within_region_mean"], 3), same("leiden_ari_graph_2023_vs_2024", r),
                 num(r["joint_moved_share"], 1, percent=True), num(r["joint_partial_r2_within_region_mean"], 3),
                 same("joint_ari_graph_2023_vs_2024", r)]
                for _, r in t.iterrows()]
        return table(["Правило", "Leiden: сообществ", "модульность", "частичный R²", "ARI графов 2023/2024",
                      "Совместная модель: сменили группу", "частичный R²", "ARI графов 2023/2024"], rows)

    def network_k(self) -> str:
        t = self.csv("network_k_sensitivity.csv")
        rows = [[integer(r["k"]), integer(r["edges"]), num(r["same_region_share"], 3), num(r["stability_jaccard"], 3),
                 num(r["assort_mean"], 3), num(r["assort_within_region_mean"], 3), integer(r["components"])]
                for _, r in t.iterrows()]
        return table(["k", "Рёбер", "Внутри региона", "Совпадение 2023→2024", "Ассортативность",
                      "Внутри регионов", "Компонент"], rows)

    def lead_lag(self) -> str:
        ll = self.summary["network"]["lead_lag"]
        rows = [["Лучший лаг не равен нулю (рёбра 2023 года)", num(ll["share_nonzero_best_lag_2023"], 1, percent=True)],
                ["Лаговых рёбер, сохранившихся в 2024 году", integer(ll["edges_in_both_years"])],
                ["Из них с ненулевым лагом в оба года", integer(ll["pairs_with_nonzero_lag_both_years"])],
                ["Направление опережения совпало", num(ll["same_direction_share_when_both_nonzero"], 1, percent=True)]]
        return table(["Показатель", "Значение"], rows)

    def feature_spaces(self) -> str:
        if not (self.d / "feature_spaces.csv").exists():
            return "_Таблица появится после запуска `python -m scripts.run_v12`._\n"
        t = self.csv("feature_spaces.csv")
        rows = [[r["label"], num(r["eta2_log_wage"], 3), num(r["partial_r2_within_region_log_wage"], 3),
                 num(r["partial_r2_within_region_mean"], 3), num(r["month_bootstrap_ari_mean"], 3),
                 num(r["month_bootstrap_ari_p10"], 3), num(r["territory_subsample_ari_mean"], 3),
                 num(r["ari_with_profile6"], 3)] for _, r in t.iterrows()]
        return table(["Пространство признаков (KMeans, K = 4)", "η² зарплаты", "Частичный R² зарплаты",
                      "Частичный R², среднее по 8", "ARI, бутстреп месяцев", "10-й перцентиль",
                      "ARI, подвыборки", "ARI с выбранным"], rows)

    # ------------------------------------------------------------ K and methods
    def k_selection(self) -> str:
        t = self.csv("k_selection.csv")
        chosen = self.summary["k_selection"]["chosen_k"]
        rows = [[("**" + integer(r["k"]) + "**") if r["k"] == chosen else integer(r["k"]), integer(r["min_size"]),
                 num(r["SW"], 3), integer(r["CH"]), num(r["S_Dbw"], 3), num(r["AVI"], 3), num(r["AVU"], 3),
                 num(r["MQ"], 3), num(r["partial_r2_within_region_mean"], 3), num(r["month_bootstrap_ari_mean"], 3),
                 num(r["territory_subsample_ari_mean"], 3), num(r["stability_score"], 3)] for _, r in t.iterrows()]
        return table(["K", "Мин. группа", "SW ↑", "CH ↑", "S_Dbw ↓", "AVI ↑", "AVU ↓", "MQ ↑",
                      "Частичный R²", "ARI месяцы", "ARI территории", "Устойчивость"], rows)

    def methods(self) -> str:
        t = self.csv("methods.csv")
        chosen = self.summary["methods"]["chosen"]
        rows = []
        for _, r in t.iterrows():
            name = ("**" + r["label"] + "**") if r["method"] == chosen else r["label"]
            rows.append([name, r["family"], integer(r["min_size_2023"]), num(r["SW_2023"], 3), integer(r["CH_2023"]),
                         num(r["S_Dbw_2023"], 3), num(r["AVI_2023"], 3), num(r["AVU_2023"], 3), num(r["MQ_2023"], 3),
                         num(r["month_bootstrap_ari_p10"], 2), "да" if r["eligible"] else "нет",
                         "—" if pd.isna(r["copeland_2023"]) else integer(r["copeland_2023"]),
                         "—" if pd.isna(r["copeland_2024"]) else integer(r["copeland_2024"])])
        return table(["Метод", "Семейство", "Мин. группа", "SW ↑", "CH ↑", "S_Dbw ↓", "AVI ↑", "AVU ↓", "MQ ↑",
                      "ARI p10", "Допустим", "Коупленд 2023", "Коупленд 2024"], rows, "llrrrrrrrrcrr")

    def methods_2024(self) -> str:
        t = self.csv("methods.csv")
        rows = [[r["label"], num(r["SW_2024"], 3), integer(r["CH_2024"]), num(r["S_Dbw_2024"], 3), num(r["AVI_2024"], 3),
                 num(r["AVU_2024"], 3), num(r["MQ_2024"], 3), num(r["kept_type_2024"], 1, percent=True),
                 num(r["partial_r2_within_region_mean"], 3)] for _, r in t.iterrows()]
        return table(["Метод", "SW ↑", "CH ↑", "S_Dbw ↓", "AVI ↑", "AVU ↓", "MQ ↑", "Тип сохранён в 2024",
                      "Частичный R²"], rows)

    # ------------------------------------------------------------ types
    def types(self) -> str:
        t = self.csv("types.csv").set_index("type")
        spec = [("МО", lambda r: integer(r["n"])), ("Доля МО", lambda r: num(r["share"], 1, percent=True)),
                ("Доля населения (где известно)", lambda r: num(r["population_share_observed"], 1, percent=True)),
                ("Расходы жителя, руб./мес.", lambda r: integer(r["spending_rub_median"])),
                ("Продовольствие, % от итога", lambda r: num(r["food_pct_median"], 1)),
                ("Общепит, %", lambda r: num(r["horeca_pct_median"], 1)),
                ("Маркетплейсы, %", lambda r: num(r["marketplaces_pct_median"], 1)),
                ("Транспорт, %", lambda r: num(r["transport_pct_median"], 1)),
                ("Зарплата (Росстат), руб. (доля МО с данными)", lambda r: f"{integer(r['wage_rub_median'])} ({num(r['log_wage_coverage'], 0, percent=True)})"),
                ("Занятые в сельском хозяйстве, % (доля МО с данными)", lambda r: f"{num(100 * r['agri_share_median'], 1)} ({num(r['agri_share_coverage'], 0, percent=True)})"),
                ("Занятые в добыче, % (доля МО с данными)",
                 lambda r: f"{num(100 * r['mining_share_median'], 1)} ({num(r['mining_share_coverage'], 0, percent=True)})"),
                ("Занятые в обрабатывающей пром., %", lambda r: num(100 * r["manuf_share_median"], 1)),
                ("Бюджетный сектор, %", lambda r: num(100 * r["public_share_median"], 1)),
                ("Доступность рынков", lambda r: num(r["market_access_median"], 0)),
                ("Муниципальные районы, %", lambda r: num(r["composition_муниципальный район"], 0, percent=True)),
                ("Городские округа, %", lambda r: num(r["composition_городской округ"], 0, percent=True)),
                ("Внутригородские территории, %",
                 lambda r: num(r["composition_внутригородская территория города федерального значения"], 0, percent=True)),
                ("Медианная широта", lambda r: num(r["latitude_median"], 1))]
        rows = [[label] + [fn(t.loc[c]) for c in range(4)] for label, fn in spec]
        return table(["Показатель (медиана)"] + SHORT, rows)

    def type_members(self) -> str:
        t = self.csv("types.csv")
        rows = [[f"**{TYPE_NAMES[r['key']]}**", r["top_regions"], r["representatives"], r["largest_by_population"]]
                for _, r in t.iterrows()]
        return table(["Тип", "Больше всего МО в регионах", "Типичные МО (ближе всего к центру)",
                      "Крупнейшие по населению"], rows, "llll")

    def type_intervals(self) -> str:
        t = self.csv("type_intervals.csv")
        labels = {"spending_rub": ("Расходы, руб.", 0), "wage_rub": ("Зарплата, руб.", 0),
                  "food_pct": ("Продовольствие, %", 1), "horeca_pct": ("Общепит, %", 1),
                  "marketplaces_pct": ("Маркетплейсы, %", 1), "agri_share_pct": ("Сельское хозяйство, %", 1),
                  "mining_share_pct": ("Добыча, %", 1), "market_access": ("Доступность рынков", 0)}
        rows = []
        for key, (label, digits) in labels.items():
            part = t[t["indicator"] == key].set_index("type")
            rows.append([label] + [f"{num(part.loc[c, 'median'], digits)} [{num(part.loc[c, 'ci_low'], digits)}; "
                                   f"{num(part.loc[c, 'ci_high'], digits)}]" for c in range(4)])
        return table(["Медиана [95% интервал, бутстреп регионов]"] + SHORT, rows)

    def loro(self) -> str:
        t = self.csv("leave_one_region_out.csv")
        labels = {"log_wage": "Логарифм зарплаты", "agri_share": "Доля сельского хозяйства",
                  "manuf_share": "Доля обрабатывающей промышленности", "mining_share": "Доля добычи",
                  "public_share": "Доля занятых в бюджетных отраслях", "log_pop": "Логарифм населения",
                  "log_jobs_per_res": "Логарифм рабочих мест на жителя", "market_access": "Доступность рынков"}
        rows = [[labels[r["indicator"]], integer(r["n"]), num(r["municipal_type"], 3), num(r["types"], 3),
                 num(r["municipal_type_and_types"], 3)] for _, r in t.iterrows()]
        return table(["Показатель", "МО с данными", "Тип МО (административный)", "Типы v1.2", "Оба"], rows)

    def validity(self) -> str:
        v = self.summary["types"]["validity"]
        labels = {"log_wage": "Логарифм зарплаты", "agri_share": "Доля сельского хозяйства",
                  "manuf_share": "Доля обрабатывающей промышленности", "mining_share": "Доля добычи",
                  "public_share": "Доля занятых в бюджетных отраслях", "log_pop": "Логарифм населения",
                  "log_jobs_per_res": "Логарифм рабочих мест на жителя", "market_access": "Доступность рынков"}
        rows = [[label, num(v[f"eta2_{k}"], 3), num(v[f"partial_r2_within_region_{k}"], 3)] for k, label in labels.items()]
        return table(["Показатель", "η² типов", "Частичный R² сверх региона"], rows)

    # ------------------------------------------------------------ dynamics
    def _matrix(self, matrix) -> str:
        rows = [[f"**{SHORT[i]}**"] + [integer(v) for v in row] + [integer(sum(row))] for i, row in enumerate(matrix)]
        return table(["2023 \\ 2024"] + SHORT + ["Всего"], rows)

    def transitions(self) -> str:
        return self._matrix(self.summary["tracking"]["annual_transitions"])

    def transitions_absolute(self) -> str:
        return self._matrix(self.summary["tracking"]["annual_transitions_absolute"])

    def persistent_matrix(self) -> str:
        return self._matrix(self.summary["tracking"]["persistent_change_matrix"])

    def markov(self) -> str:
        m = self.summary["tracking"]["markov"]
        rows = [[f"**{SHORT[i]}**"] + [num(v, 3) for v in row] + [num(m["expected_months_in_type"][i], 1),
                                                                   num(m["stationary"][i], 3)]
                for i, row in enumerate(m["matrix"])]
        return table(["Из \\ в"] + SHORT + ["Ожидаемо месяцев в типе", "Стационарная доля"], rows)

    def persistence(self) -> str:
        p, a = self.summary["tracking"]["persistence"], self.summary["tracking"]["persistence_absolute"]
        rows = [["МО без смены типа за 24 месяца", num(p["constant_share"], 1, percent=True), num(a["constant_share"], 1, percent=True)],
                ["Среднее число смен за 24 месяца", num(p["mean_switches"], 2), num(a["mean_switches"], 2)],
                ["Доля месяцев в годовом типе 2023", num(p["months_in_annual_type_share"], 1, percent=True),
                 num(a["months_in_annual_type_share"], 1, percent=True)],
                ["Устойчивые переходы (июль–декабрь 2024 в новом типе)", integer(p["persistent_changes"]),
                 integer(a["persistent_changes"])]]
        return table(["Показатель", "Без общей волны (основной)", "С общей волной"], rows)

    def reclustering(self) -> str:
        t = self.csv("reclustering_events.csv")
        rows = [[SHORT[int(r["type"])], str(int(r["year"])), integer(r["survival_months"]), integer(r["absorption_months"]),
                 integer(r["split_months"]), integer(r["disappearance_months"]), num(r["median_jaccard_with_base"], 2)]
                for _, r in t.iterrows()]
        return table(["Тип 2023", "Год", "Сохранение, мес.", "Поглощение", "Распад", "Исчезновение",
                      "Медиана Жаккара с типом 2023"], rows)

    def free2024(self) -> str:
        fr = self.summary["tracking"]["free_reclustering"]
        overlap = fr["annual_2024_overlap"]
        rows = []
        for g, prof in enumerate(fr["annual_2024_profiles"]):
            members = ", ".join(f"{SHORT[i]}: {integer(overlap[i][g])}" for i in range(4) if overlap[i][g])
            rows.append([f"Группа {g + 1}", integer(prof["n"]), members, integer(prof["spending_rub_median"]),
                         num(prof["marketplaces_pct_median"], 1), num(prof["latitude_median"], 1),
                         integer(prof["wage_rub_median"]), prof["top_regions"]])
        return table(["Свободная группа 2024", "МО", "Состав по типам 2023", "Расходы, руб.", "Маркетплейсы, %",
                      "Широта", "Зарплата, руб.", "Регионы"], rows, "lrlrrrrl")

    def type_dynamics(self) -> str:
        t = self.csv("type_dynamics.csv")
        rows = [[f"**{SHORT[int(r['type'])]}**", num(r["marketplaces_pct_2023"], 1), num(r["marketplaces_pct_2024"], 1),
                 "×" + num(r["marketplaces_ratio_2024_to_2023"], 2),
                 f"{num(r['nominal_growth_mean_pct'], 1)} [{num(r['nominal_growth_ci_low_pct'], 1)}; {num(r['nominal_growth_ci_high_pct'], 1)}]",
                 f"{num(r['relative_level_change_log'], 3, signed=True)} [{num(r['relative_level_change_ci_low'], 3, signed=True)}; "
                 f"{num(r['relative_level_change_ci_high'], 3, signed=True)}]",
                 num(r["left_type_2024_adjusted_share"], 1, percent=True)] for _, r in t.iterrows()]
        return table(["Тип 2023", "Маркетплейсы 2023, %", "2024, %", "Рост доли", "Номинальный рост расходов, % [95%]",
                      "Сдвиг уровня относительно страны, лог. [95%]", "Сменили тип к 2024"], rows)

    def gap(self) -> str:
        g = self.summary["practical"]["gap"]
        rows = [["Отношение медианных уровней расходов «крупные города / периферия», 2023", num(g["level_ratio_2023"], 3)],
                ["То же, 2024", num(g["level_ratio_2024"], 3)],
                ["Разница сдвигов уровня (города − периферия), лог.",
                 f"{num(g['metro_minus_periphery_relative_level_change_log'], 4, signed=True)} "
                 f"[{num(g['ci_low'], 4, signed=True)}; {num(g['ci_high'], 4, signed=True)}]"]]
        return table(["Показатель", "Значение"], rows)

    def monthly(self) -> str:
        t = self.csv("monthly_network.csv")
        rows = [[r["period"][:7], integer(r["edges"]), "—" if pd.isna(r["edge_jaccard_next_month"]) else num(r["edge_jaccard_next_month"], 3)]
                + [integer(r[f"n_type_{c}"]) for c in range(4)]
                + [num(r["SW"], 3), num(r["AVI"], 3), num(r["MQ"], 3)] for _, r in t.iterrows()]
        return table(["Месяц", "Рёбер", "Совпадение со след. месяцем"] + SHORT + ["SW", "AVI", "MQ"], rows)

    # ------------------------------------------------------------ practical
    def analogues(self) -> str:
        t = self.csv("analogues.csv")
        labels = {"network_analogues": "15 ближайших в сети (профиль 2023)",
                  "network_analogues_same_region": "15 ближайших в сети внутри своего региона",
                  "random_same_region": "15 случайных МО своего региона",
                  "region_median_leave_one_out": "Медиана своего региона",
                  "random_same_type": "15 случайных МО своего типа", "national_median": "Медиана страны"}
        rows = [[labels[r["predictor"]], num(r["mae_pp"], 3), num(r["median_ae_pp"], 3)] for _, r in t.iterrows()]
        return table(["Прогноз роста расходов 2024/2023 по", "Средняя ошибка, п.п.", "Медианная ошибка, п.п."], rows)

    def analogue_comparisons(self) -> str:
        t = self.csv("analogue_comparisons.csv")
        labels = {"network_analogues": "Сеть", "network_analogues_same_region": "Сеть внутри региона",
                  "random_same_region": "случайные МО региона", "national_median": "медиана страны",
                  "region_median_leave_one_out": "медиана региона", "random_same_type": "случайные МО типа"}
        rows = [[f"{labels[r['better']]} против: {labels[r['baseline']]}", num(r["mae_difference_pp"], 3, signed=True),
                 f"[{num(r['ci_low'], 3, signed=True)}; {num(r['ci_high'], 3, signed=True)}]"] for _, r in t.iterrows()]
        return table(["Сравнение", "Разность средних ошибок, п.п.", "95% интервал (бутстреп регионов)"], rows)

    def monitoring(self) -> str:
        t = self.csv("persistent_changes.csv")
        t = t.sort_values("population", ascending=False).head(12)
        rows = [[r["name"], r["region"], SHORT[int(r["type_2023"])], SHORT[int(r["type_2024_h2"])],
                 "—" if pd.isna(r["population"]) else integer(r["population"])] for _, r in t.iterrows()]
        return table(["МО", "Регион", "Тип 2023", "Тип во II полугодии 2024", "Население"], rows, "llllr")

    # ------------------------------------------------------------ inline facts
    def facts(self) -> dict[str, str]:
        s = self.summary
        nets = self.csv("networks.csv").set_index("rule")
        chosen_rule = s["network"]["chosen_rule"]
        dynamic = nets[~nets["static"]]
        ts = dynamic.loc[["corr_total", "corr_multivariate", "lagged_corr_total", "dtw_total"]]
        methods = self.csv("methods.csv").set_index("method")
        chosen = s["methods"]["chosen"]
        types = self.csv("types.csv").set_index("key")
        loro = self.csv("leave_one_region_out.csv").set_index("indicator")
        dyn = self.csv("type_dynamics.csv").set_index("key")
        an = self.csv("analogues.csv").set_index("predictor")
        cmp_ = pd.concat([self.csv("analogue_comparisons.csv"), self.csv("analogue_comparisons_diagnostic.csv")],
                         ignore_index=True).set_index(["better", "baseline"])
        events = self.csv("reclustering_events.csv")
        gap = s["practical"]["gap"]
        pers = s["tracking"]["persistence"]
        mk = s["tracking"]["markov"]
        f = {
            "entities": integer(s["panel"]["entities"]), "regions": integer(s["panel"]["regions"]),
            "rule_chosen": nets.loc[chosen_rule, "label"],
            "rule_stability": num(nets.loc[chosen_rule, "stability_jaccard"], 2),
            "rule_within": num(nets.loc[chosen_rule, "assort_within_region_mean"], 3),
            "rule_assort": num(nets.loc[chosen_rule, "assort_mean"], 3),
            "ts_stability_max": num(ts["stability_jaccard"].max(), 2), "ts_stability_min": num(ts["stability_jaccard"].min(), 2),
            "geo_within": num(nets.loc["geographic", "assort_within_region_mean"], 3),
            "road_within": num(nets.loc["road", "assort_within_region_mean"], 3),
            "geo_same_region": num(nets.loc["geographic", "same_region_share"], 0, percent=True),
            "road_same_region": num(nets.loc["road", "same_region_share"], 0, percent=True),
            "k": integer(s["k_selection"]["chosen_k"]),
            "method_chosen": methods.loc[chosen, "label"],
            "method_moved": num(s["methods"]["moved_vs_kmeans_share"], 1, percent=True),
            "method_ari": num(s["methods"]["ari_final_vs_kmeans"], 3),
            "method_cop_total": integer(methods.loc[chosen, "copeland_total"]),
            "kmeans_cop_total": integer(methods.loc["kmeans", "copeland_total"]),
            "constant_share": num(pers["constant_share"], 0, percent=True),
            "persistent_changes": integer(pers["persistent_changes"]),
            "remote_left": num(dyn.loc["remote", "left_type_2024_adjusted_share"], 0, percent=True),
            "mkt_remote_2023": num(dyn.loc["remote", "marketplaces_pct_2023"], 1),
            "mkt_remote_2024": num(dyn.loc["remote", "marketplaces_pct_2024"], 1),
            "mkt_remote_ratio": num(dyn.loc["remote", "marketplaces_ratio_2024_to_2023"], 1),
            "mkt_periphery_ratio": num(dyn.loc["periphery", "marketplaces_ratio_2024_to_2023"], 2),
            "mkt_metro_ratio": num(dyn.loc["metro", "marketplaces_ratio_2024_to_2023"], 2),
            "growth_metro": num(dyn.loc["metro", "nominal_growth_mean_pct"], 1),
            "growth_industrial": num(dyn.loc["industrial", "nominal_growth_mean_pct"], 1),
            "growth_periphery": num(dyn.loc["periphery", "nominal_growth_mean_pct"], 1),
            "growth_remote": num(dyn.loc["remote", "nominal_growth_mean_pct"], 1),
            "gap_2023": num(gap["level_ratio_2023"], 2), "gap_2024": num(gap["level_ratio_2024"], 2),
            "gap_change": num(gap["metro_minus_periphery_relative_level_change_log"], 3, signed=True),
            "gap_low": num(gap["ci_low"], 3, signed=True), "gap_high": num(gap["ci_high"], 3, signed=True),
            "mae_network": num(an.loc["network_analogues", "mae_pp"], 2),
            "mae_region": num(an.loc["random_same_region", "mae_pp"], 2),
            "mae_national": num(an.loc["national_median", "mae_pp"], 2),
            "mae_diff_region": num(cmp_.loc[("network_analogues", "random_same_region"), "mae_difference_pp"], 2, signed=True),
            "mae_diff_region_low": num(cmp_.loc[("network_analogues", "random_same_region"), "ci_low"], 2, signed=True),
            "mae_diff_region_high": num(cmp_.loc[("network_analogues", "random_same_region"), "ci_high"], 2, signed=True),
            "mae_diff_inregion": num(cmp_.loc[("network_analogues_same_region", "region_median_leave_one_out"), "mae_difference_pp"], 2, signed=True),
            "mae_diff_inregion_low": num(cmp_.loc[("network_analogues_same_region", "region_median_leave_one_out"), "ci_low"], 2, signed=True),
            "mae_diff_inregion_high": num(cmp_.loc[("network_analogues_same_region", "region_median_leave_one_out"), "ci_high"], 2, signed=True),
            "loro_wage_types": num(loro.loc["log_wage", "types"], 2), "loro_wage_mtype": num(loro.loc["log_wage", "municipal_type"], 2),
            "loro_pop_types": num(loro.loc["log_pop", "types"], 2), "loro_pop_mtype": num(loro.loc["log_pop", "municipal_type"], 2),
            "loro_ma_types": num(loro.loc["market_access", "types"], 2), "loro_ma_mtype": num(loro.loc["market_access", "municipal_type"], 2),
            "loro_agri_types": num(loro.loc["agri_share", "types"], 2), "loro_mining_types": num(loro.loc["mining_share", "types"], 2),
            "loro_manuf_types": num(loro.loc["manuf_share", "types"], 2),
            "expected_metro": num(mk["expected_months_in_type"][0], 0),
            "expected_remote": num(mk["expected_months_in_type"][3], 0),
        }
        for key in TYPE_ORDER:
            r = types.loc[key]
            f[f"n_{key}"] = integer(r["n"])
            f[f"share_{key}"] = num(r["share"], 0, percent=True)
            f[f"pop_{key}"] = num(r["population_share_observed"], 0, percent=True)
            f[f"spend_{key}"] = integer(r["spending_rub_median"])
            f[f"wage_{key}"] = integer(r["wage_rub_median"])
            f[f"food_{key}"] = num(r["food_pct_median"], 0)
            f[f"horeca_{key}"] = num(r["horeca_pct_median"], 1)
            f[f"mkt_{key}"] = num(r["marketplaces_pct_median"], 1)
            f[f"agri_{key}"] = num(100 * r["agri_share_median"], 1)
            f[f"mining_{key}"] = num(100 * r["mining_share_median"], 0)
            f[f"manuf_{key}"] = num(100 * r["manuf_share_median"], 0)
            f[f"access_{key}"] = num(r["market_access_median"], 0)
            f[f"rayon_{key}"] = num(r["composition_муниципальный район"], 0, percent=True)
            f[f"lat_{key}"] = num(r["latitude_median"], 1)
            f[f"wage_cov_{key}"] = num(r["log_wage_coverage"], 0, percent=True)
            f[f"agri_cov_{key}"] = num(r["agri_share_coverage"], 0, percent=True)
            f[f"mining_cov_{key}"] = num(r["mining_share_coverage"], 0, percent=True)
        labels = self.csv("labels.csv")
        remote = labels[labels["type_2023"] == TYPE_ORDER.index("remote")]
        f["remote_far_east_share"] = num(remote["region"].isin(FAR_EAST).mean(), 0, percent=True)
        for c, key in enumerate(TYPE_ORDER):
            part = events[events["type"] == c].set_index("year")
            f[f"free_jaccard_{key}_2023"] = num(part.loc[2023, "median_jaccard_with_base"], 2)
            f[f"free_jaccard_{key}_2024"] = num(part.loc[2024, "median_jaccard_with_base"], 2)
        effects = self.csv("network_effects.csv").set_index("rule")
        if "joint_ari_graph_2023_vs_2024" in effects:
            corr = effects.loc[["corr_total", "corr_multivariate", "lagged_corr_total", "dtw_total"], "joint_ari_graph_2023_vs_2024"]
            f["joint_ari_rule"] = num(effects.loc[chosen_rule, "joint_ari_graph_2023_vs_2024"], 2)
            f["effects_moved_rule"] = num(effects.loc[chosen_rule, "joint_moved_share"], 1, percent=True)
            f["joint_ari_ts_min"] = num(corr.min(), 2)
            f["joint_ari_ts_max"] = num(corr.max(), 2)
        if (self.d / "feature_spaces.csv").exists():
            fs = self.csv("feature_spaces.csv").set_index("space")
            f["fs_wage_v11"] = num(fs.loc["structure5_v11", "eta2_log_wage"], 2)
            f["fs_wage_v12"] = num(fs.loc["profile6", "eta2_log_wage"], 2)
            f["fs_pwage_v11"] = num(fs.loc["structure5_v11", "partial_r2_within_region_log_wage"], 3)
            f["fs_pwage_v12"] = num(fs.loc["profile6", "partial_r2_within_region_log_wage"], 3)
            f["fs_stab_v11"] = num(fs.loc["structure5_v11", "month_bootstrap_ari_mean"], 2)
            f["fs_stab_v12"] = num(fs.loc["profile6", "month_bootstrap_ari_mean"], 2)
            f["fs_stab_annual"] = num(fs.loc["profile6_annual_iqr", "month_bootstrap_ari_mean"], 2)
            f["fs_stab_relative"] = num(fs.loc["profile6_relative", "month_bootstrap_ari_mean"], 2)
            f["fs_ari_relative"] = num(fs.loc["profile6_relative", "ari_with_profile6"], 2)
        return f



def render(document: str, tables: Tables) -> str:
    def replace(match):
        builder = getattr(tables, match.group("name"), None)
        if builder is None:
            raise KeyError(f"Unknown table marker: {match.group('name')}")
        return match.group(1) + builder() + match.group(4)
    facts = tables.facts()

    def replace_fact(match):
        if match.group("name") not in facts:
            raise KeyError(f"Unknown fact marker: {match.group('name')}")
        return match.group(1) + facts[match.group("name")] + match.group(4)
    return FACT.sub(replace_fact, MARKER.sub(replace, document))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "reports/v1.2")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    tables = Tables(args.input)
    stale = []
    for name in DOCUMENTS:
        path = ROOT / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        new = render(text, tables)
        if new != text:
            stale.append(name)
            if not args.check:
                path.write_text(new, encoding="utf-8")
    if args.check and stale:
        raise SystemExit("Stale v1.2 tables: " + ", ".join(stale))
    print(json.dumps({"updated" if not args.check else "stale": stale}, ensure_ascii=False))


if __name__ == "__main__":
    main()
