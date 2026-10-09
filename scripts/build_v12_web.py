"""Static v1.2 sections of the research atlas, generated from the v1.2 result files.

    python -m scripts.build_v12_web                         # reads reports/v1.2, writes web/v12_sections.html
    python -m scripts.build_v12_web --input <dir> --output <file>

No fitting and no hand-typed numbers: every value is read from the files written
by ``python -m scripts.run_v12`` and only formatted here. Type names and colours
come from ``sbercluster/typology.py``. The output is plain HTML with inline SVG
(no scripts, fonts or external resources); the atlas template inserts it at
``<!-- V12_SECTIONS -->``.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sbercluster.typology import TYPE_COLORS, TYPE_NAMES, TYPE_ORDER, TYPE_SHORT  # noqa: E402

NNBSP = "\N{NARROW NO-BREAK SPACE}"  # thousands separator
NBSP = "\N{NO-BREAK SPACE}"
MINUS = "\N{MINUS SIGN}"
REPOSITORY = "https://github.com/Nibani/sber-public/blob/main/"
CSV_FILES = ("labels", "types", "type_intervals", "networks", "network_effects", "network_k_sensitivity",
             "feature_spaces", "k_selection", "methods", "monthly_network", "reclustering_events",
             "type_dynamics", "analogues", "analogue_comparisons", "persistent_changes")
OPTIONAL_FILES = ("network_effects", "feature_spaces")
TYPES = [{"id": i, "key": key, "name": TYPE_NAMES[key], "short": TYPE_SHORT[key], "color": TYPE_COLORS[key]}
         for i, key in enumerate(TYPE_ORDER)]
MONTHS_SHORT = ("янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек")
MONTHS_FULL = ("январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь",
               "октябрь", "ноябрь", "декабрь")
RULE_SHORT = {
    "euclid_structure": "Структура расходов", "euclid_profile": "Структура + уровень",
    "cosine_spending": "Косинус, рубли", "corr_total": "Корреляция уровня",
    "corr_multivariate": "Многомерная корреляция", "lagged_corr_total": "Лаговая корреляция",
    "dtw_total": "DTW", "geographic": "География", "road": "Дороги",
}
COMPOSITION = {"муниципальный район": "районы", "муниципальный округ": "округа",
               "городской округ": "городские округа",
               "внутригородская территория города федерального значения": "внутригородские территории"}
INTERVALS = {
    "spending_rub": ("Уровень расходов, тыс. руб. в месяц", 1e-3, 1),
    "wage_rub": ("Зарплата (Росстат), тыс. руб.", 1e-3, 1),
    "food_pct": ("Продовольствие, %", 1, 1),
    "horeca_pct": ("Общепит, %", 1, 1),
    "marketplaces_pct": ("Маркетплейсы, %", 1, 1),
    "agri_share_pct": ("Занятые в сельском хозяйстве, %", 1, 1),
    "mining_share_pct": ("Занятые в добыче, %", 1, 1),
    "market_access": ("Доступность рынков, индекс", 1, 0),
}
MO = ("МО", "МО", "МО")
MONTH_FORMS = ("месяц", "месяца", "месяцев")
DRAW_FORMS = ("повтор", "повтора", "повторов")
CHOSEN = ' class="v12-chosen"'
GROUP = ' class="v12-group"'
BADGE_SPACE = '<span class="v12-badge">выбрано</span>'
CHART_WIDTH = 640  # viewBox width of charts in a half-width column: text renders close to its nominal size


# ------------------------------------------------------------------ reading
def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value) -> float | None:
    """Finite float or None for blanks, NaN and infinities."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        result = float(text)
    return result if math.isfinite(result) else None


def flag(value) -> bool:
    return str(value).strip().lower() in {"true", "1"}


def load(directory: Path) -> dict:
    """Read and cross-check every v1.2 file used by the atlas."""
    directory = Path(directory)
    required = [name for name in CSV_FILES if name not in OPTIONAL_FILES]
    missing = [f"{name}.csv" for name in required if not (directory / f"{name}.csv").is_file()]
    missing += [name for name in ("summary.json",) if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"v1.2 results are incomplete in {directory}: missing {', '.join(missing)}")
    data = {name: read_rows(directory / f"{name}.csv") if (directory / f"{name}.csv").is_file() else []
            for name in CSV_FILES}
    data["summary"] = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    wage = directory / "wage_level_comparison.json"
    data["wage_level_comparison"] = json.loads(wage.read_text(encoding="utf-8")) if wage.is_file() else None
    provenance = directory / "provenance.json"
    data["provenance"] = json.loads(provenance.read_text(encoding="utf-8")) if provenance.is_file() else {}
    data["config"] = (data["provenance"].get("config")
                      or json.loads((ROOT / "configs/v12.json").read_text(encoding="utf-8")))
    for key, relative in (("added_value", "v1.2-added-value/contrasts.csv"),
                          ("method_ties", "v1.2-method-ties/win_frequency.csv")):
        source = directory.parent / relative
        data[key] = read_rows(source) if source.is_file() else []
    validate(data)
    return data


def transition_matrix(before: list[int], after: list[int], k: int = 4) -> list[list[int]]:
    matrix = [[0] * k for _ in range(k)]
    for a, b in zip(before, after):
        matrix[a][b] += 1
    return matrix


def validate(data: dict) -> None:
    labels = data["labels"]
    ids = [row["entity_id"] for row in labels]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("labels.csv must list every municipality exactly once")
    for column in ("type_2023", "type_2024", "type_2024_absolute"):
        if any(row[column] not in {"0", "1", "2", "3"} for row in labels):
            raise ValueError(f"labels.csv {column} must hold types 0..3")
    types = sorted(data["types"], key=lambda row: int(row["type"]))
    if [row["key"] for row in types] != list(TYPE_ORDER):
        raise ValueError("types.csv does not follow TYPE_ORDER")
    sizes = [sum(row["type_2023"] == str(c) for row in labels) for c in range(4)]
    if [int(row["n"]) for row in types] != sizes:
        raise ValueError("types.csv sizes differ from labels.csv")
    tracking = data["summary"]["tracking"]
    first = [int(row["type_2023"]) for row in labels]
    for key, column in (("annual_transitions", "type_2024"), ("annual_transitions_absolute", "type_2024_absolute")):
        if tracking[key] != transition_matrix(first, [int(row[column]) for row in labels]):
            raise ValueError(f"summary.json tracking.{key} differs from labels.csv")
    if any(sum(int(row[f"n_type_{c}"]) for c in range(4)) != len(ids) for row in data["monthly_network"]):
        raise ValueError("monthly_network.csv type counts do not cover every municipality")
    if data["summary"]["methods"]["chosen"] not in {row["method"] for row in data["methods"]}:
        raise ValueError("Chosen method is missing from methods.csv")
    if data["summary"]["network"]["chosen_rule"] not in {row["rule"] for row in data["networks"]}:
        raise ValueError("Chosen edge rule is missing from networks.csv")
    if len(data["persistent_changes"]) != tracking["persistence"]["persistent_changes"]:
        raise ValueError("persistent_changes.csv differs from summary.json")


# ------------------------------------------------------------------ formatting
def esc(value) -> str:
    return html.escape(str(value), quote=True)


def fnum(value, digits: int = 0, *, sign: bool = False) -> str:
    """Russian number: comma decimals, narrow no-break space thousands, true minus sign."""
    value = number(value)
    if value is None:
        return "нет"
    text = f"{abs(value):,.{digits}f}".replace(",", NNBSP).replace(".", ",")
    nonzero = any(ch not in "0," + NNBSP for ch in text)
    if value < 0 and nonzero:
        return MINUS + text
    return "+" + text if sign and value > 0 and nonzero else text


def fpct(fraction, digits: int = 1) -> str:
    value = number(fraction)
    return "нет" if value is None else fnum(100 * value, digits) + "%"


def finterval(low, high, digits: int, scale: float = 1.0) -> str:
    if number(low) is None or number(high) is None:
        return "нет"
    return f"[{fnum(scale * number(low), digits)}; {fnum(scale * number(high), digits)}]"


def plural(n: int, forms: tuple[str, str, str]) -> str:
    n = abs(int(n))
    if n % 10 == 1 and n % 100 != 11:
        return forms[0]
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return forms[1]
    return forms[2]


def count(n, forms: tuple[str, str, str]) -> str:
    return f"{fnum(n)}{NBSP}{plural(int(number(n)), forms)}"


def lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def short_name(name: str) -> str:
    """Same shortening as the ego network labels of the atlas template."""
    value = re.sub(r"^внутригородская территория города федерального значения\s+", "", name, flags=re.I)
    value = re.sub(r"^внутригородское муниципальное образование\s+(?:города\s+)?(?:Москвы|Санкт-Петербурга)\s+",
                   "", value, flags=re.I)
    value = re.sub(r"^(?:городской округ(?: город(?:-курорт)?)?|муниципальный округ|муниципальный район|"
                   r"город-курорт|город|пос[её]лок)\s+", "", value, flags=re.I)
    value = re.sub(r" муниципальный район$", " р-н", value, flags=re.I)
    return re.sub(r" муниципальный округ$", " окр.", value, flags=re.I)


def region_counts(text: str) -> str:
    """'Регион (24); Республика Саха (Якутия) (22)' -> 'Регион — 24, Республика Саха (Якутия) — 22'."""
    parts = []
    for item in (text or "").split(";"):
        match = re.fullmatch(r"\s*(.*\S)\s+\((\d+)\)\s*", item)
        if match:
            parts.append(f"{esc(match.group(1))} — {fnum(int(match.group(2)))}")
        elif item.strip():
            parts.append(esc(item.strip()))
    return ", ".join(parts)


def entity_button(entity: dict, *, with_region: bool = True) -> str:
    region = f' <span class="muted">({esc(entity["region"])})</span>' if with_region else ""
    return (f'<button type="button" class="table-link" data-select-entity="{esc(entity["entity_id"])}" '
            f'title="{esc(entity["name"])} · {esc(entity["region"])}">{esc(short_name(entity["name"]))}</button>{region}')


def type_label(t: dict) -> str:
    return (f'<span class="v12-type-label"><i class="v12-dot" style="--type-color:{t["color"]}" aria-hidden="true"></i>'
            f'{esc(t["short"])}</span>')


def quoted(t: dict) -> str:
    return f"«{esc(t['short'])}»"


def period_label(period: str) -> str:
    return f"{MONTHS_FULL[int(period[5:7]) - 1]} {period[:4]}"


def doc_link(filename: str, text: str) -> str:
    """Link to a v1.2 document of the repository, only when the document exists."""
    if not (ROOT / "docs" / filename.split("#", 1)[0]).is_file():
        return ""
    return f'<a href="{REPOSITORY}docs/{filename}" target="_blank" rel="noopener">{esc(text)}</a>'


def table_block(caption: str, table: str, label: str, *, wide: bool = False, cls: str = "") -> str:
    """Caption outside the scroll area, so it stays readable when the table scrolls sideways."""
    note = f'<p class="v12-caption">{caption}</p>' if caption else ""
    return (f'{note}<div class="table-scroll{" v12-wide" if wide else ""}" tabindex="0" role="region" '
            f'aria-label="{esc(label)}"><table class="v12-table{(" " + cls) if cls else ""}">{table}</table></div>')


# ------------------------------------------------------------------ SVG helpers
def text_width(text: str, size: float) -> float:
    """Generous width estimate for label placement (no font metrics offline)."""
    width = 0.0
    for ch in text:
        if ch.isspace() or ch in ".,:;·'|!":
            width += 0.3
        elif ch in "()[]-–−—/«»":
            width += 0.38
        elif ch.isdigit():
            width += 0.57
        elif ch.isupper():
            width += 0.7
        elif ch in "шщжюмфы":
            width += 0.78
        else:
            width += 0.58
    return width * size


def nice_step(span: float, target: int = 5) -> float:
    raw = span / max(target, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 2.5, 5, 10):
        if multiple * magnitude >= raw - 1e-12:
            return multiple * magnitude
    return 10 * magnitude


def ticks(low: float, high: float, target: int = 5) -> tuple[list[float], int]:
    step = nice_step(high - low, target)
    values, value = [], math.floor(low / step + 1e-9) * step
    while value <= high + step * 1e-6:
        values.append(round(value, 10))
        value += step
    digits = max(0, -math.floor(math.log10(step) + 1e-9))
    if abs(step / 10 ** -digits - round(step / 10 ** -digits)) > 1e-9:
        digits += 1
    return values, digits


def nice_ceiling(value: float, target: int = 5) -> float:
    values, _ = ticks(0, value, target)
    return values[-1] if values[-1] >= value else values[-1] + nice_step(value, target)


def f1(value: float) -> str:
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


def svg_text(x: float, y: float, content: str, *, anchor: str = "start", cls: str = "", extra: str = "") -> str:
    attrs = (f' text-anchor="{anchor}"' if anchor != "start" else "") + (f' class="{cls}"' if cls else "")
    return f'<text x="{f1(x)}" y="{f1(y)}"{attrs}{extra}>{content}</text>'


def svg_open(width: int, height: int, label: str) -> str:
    return f'<svg class="v12-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label)}">'


def chart(svg: str, label: str) -> str:
    return f'<div class="v12-chart" tabindex="0" role="region" aria-label="{esc(label)}">{svg}</div>'


def wrapped_svg_label(x: float, y: float, text: str, width: float, *, anchor: str = "start") -> str:
    lines = [""]
    for word in text.split():
        candidate = (lines[-1] + " " + word).strip()
        if lines[-1] and text_width(candidate, 13) > width:
            lines.append(word)
        else:
            lines[-1] = candidate
    start = y - 15 * (len(lines) - 1) / 2
    return "".join(svg_text(x, start + 15 * i, esc(line), anchor=anchor, cls="v12-strong")
                   for i, line in enumerate(lines))


def legend(items: list[tuple[str, str]], x: float, y: float, width: float, size: float = 13) -> tuple[str, float]:
    """Flowing legend of (colour, text); returns SVG and the y below it."""
    parts, cx, cy = [], x, y
    for color, text in items:
        item = 17 + text_width(text, size)
        if cx > x and cx + item > x + width:
            cx, cy = x, cy + 20
        parts.append(f'<rect x="{f1(cx)}" y="{f1(cy - 10)}" width="12" height="12" rx="2" fill="{color}"/>')
        parts.append(svg_text(cx + 17, cy, esc(text)))
        cx += item + 18
    return "".join(parts), cy + 16


def overlaps(a, b, margin: float = 2.0) -> bool:
    return not (a[2] + margin <= b[0] or b[2] + margin <= a[0] or a[3] + margin <= b[1] or b[3] + margin <= a[1])


def place_labels(points: list[dict], bounds: tuple[float, float, float, float], size: float = 13) -> list[dict]:
    """Greedy collision-free label placement around points (in priority order)."""
    obstacles = [(p["x"] - p["r"] - 1, p["y"] - p["r"] - 1, p["x"] + p["r"] + 1, p["y"] + p["r"] + 1) for p in points]
    boxes = []
    for index in sorted(range(len(points)), key=lambda i: points[i].get("priority", i)):
        p = points[index]
        width, gap, base = text_width(p["text"], size), p["r"] + 5, p["y"] + size * 0.36
        candidates = [("start", p["x"] + gap, base, False), ("end", p["x"] - gap, base, False),
                      ("middle", p["x"], p["y"] - gap - 2, False), ("middle", p["x"], p["y"] + gap + size - 2, False)]
        for shift in (15, 30, 45, 60):
            for direction in (-1, 1):
                candidates += [("start", p["x"] + gap + 6, base + direction * shift, True),
                               ("end", p["x"] - gap - 6, base + direction * shift, True)]
        chosen = None
        for anchor, x, y, leader in candidates:
            left = x if anchor == "start" else x - width if anchor == "end" else x - width / 2
            box = (left, y - size * 0.8, left + width, y + size * 0.25)
            if box[0] < bounds[0] or box[2] > bounds[2] or box[1] < bounds[1] or box[3] > bounds[3]:
                continue
            if any(overlaps(box, other) for other in boxes):
                continue
            if any(overlaps(box, obstacle, 1) for j, obstacle in enumerate(obstacles) if j != index):
                continue
            chosen = (anchor, x, y, leader, box)
            break
        if chosen is None:
            anchor, x, y, leader = candidates[0]
            chosen = (anchor, x, y, leader, (x, y - size * 0.8, x + width, y + size * 0.25))
        boxes.append(chosen[4])
        p["label"] = {"anchor": chosen[0], "x": chosen[1], "y": chosen[2], "leader": chosen[3]}
    return points


# ------------------------------------------------------------------ shared context
def context(data: dict) -> dict:
    labels = data["labels"]
    return {"n": len(labels), "by_key": {f"{row['name']} ({row['region']})": row for row in labels},
            "types": {int(row["type"]): row for row in data["types"]},
            "methods": {row["method"]: row for row in data["methods"]},
            "rules": {row["rule"]: row for row in data["networks"]},
            "chosen_method": data["summary"]["methods"]["chosen"],
            "chosen_rule": data["summary"]["network"]["chosen_rule"],
            "k_graph": data["config"]["network"]["k"], "chosen_k": data["summary"]["k_selection"]["chosen_k"],
            "categories": [a for a in data["config"]["features"]["attributes"] if a != "level"]}


def members(text: str, ctx: dict, limit: int = 3) -> str:
    rendered = []
    for item in [item.strip() for item in (text or "").split(";") if item.strip()][:limit]:
        entity = ctx["by_key"].get(item)
        rendered.append(entity_button(entity) if entity else esc(item))
    return "; ".join(rendered) if rendered else "нет данных"


def section_head(anchor: str, title: str, lead: str, tag: str) -> str:
    return (f'<div class="section-head"><div><h2 id="{anchor}">{esc(title)}</h2><p class="muted">{lead}</p></div>'
            f'<span class="tag">{esc(tag)}</span></div>')


def more(*links: str) -> str:
    present = [link for link in links if link]
    return f'<p class="note small v12-more">Подробнее: {" · ".join(present)}.</p>' if present else ""


# ------------------------------------------------------------------ section a: types
def type_card(t: dict, row: dict, ctx: dict) -> str:
    shares = [("Продовольствие", "food_pct_median"), ("Общепит", "horeca_pct_median"),
              ("Маркетплейсы", "marketplaces_pct_median")]
    scale = nice_ceiling(max(number(r[key]) for r in ctx["types"].values() for _, key in shares), 5)
    bars = "".join(
        f'<div class="v12-bar"><b>{esc(label)}</b><i style="--w:{100 * number(row[key]) / scale:.1f}%" '
        f'aria-hidden="true"></i><span>{fnum(row[key], 1)}%</span></div>' for label, key in shares)

    def rosstat(value_key: str, coverage_key: str, value: str) -> str:
        if number(row[value_key]) is None:
            return "нет данных"
        return f'{value} <span class="muted">· есть у {fpct(row[coverage_key], 0)} МО</span>'

    composition = sorted(((number(value), COMPOSITION.get(column[12:], column[12:]))
                          for column, value in row.items() if column.startswith("composition_") and number(value)),
                         key=lambda item: -item[0])
    other = 1 - sum(value for value, _ in composition)
    parts = [f"{esc(label)} {fpct(value, 0)}" for value, label in composition if round(100 * value) >= 1]
    if round(100 * other) >= 1:
        parts.append(f"прочие {fpct(other, 0)}")
    items = [
        ("Население", f'{fpct(row["population_share_observed"])} населения МО с известной численностью '
                      f'<span class="muted">· известна у {fpct(row["population_coverage"], 0)} МО типа</span>'),
        ("Уровень расходов", f'{fnum(1e-3 * number(row["spending_rub_median"]), 1)} тыс. руб. в месяц '
                             '<span class="muted">· медиана 2023</span>'),
        ("Зарплата (Росстат, 2023)", rosstat("wage_rub_median", "log_wage_coverage",
                                             f'{fnum(1e-3 * number(row["wage_rub_median"]), 1)} тыс. руб.')),
        ("Занятые в сельском хозяйстве", rosstat("agri_share_median", "agri_share_coverage", fpct(row["agri_share_median"]))),
        ("Занятые в добыче", rosstat("mining_share_median", "mining_share_coverage", fpct(row["mining_share_median"]))),
        ("Доступность рынков", rosstat("market_access_median", "market_access_coverage", fnum(row["market_access_median"], 1))),
        ("Типы МО", " · ".join(parts) or "нет данных"),
        ("Типичные, ближе к центру типа", members(row["representatives"], ctx)),
        ("Крупнейшие по населению", members(row["largest_by_population"], ctx)),
    ]
    details = "".join(f"<dt>{esc(name)}</dt><dd>{value}</dd>" for name, value in items)
    return (f'<article class="v12-type" style="--type-color:{t["color"]}" aria-labelledby="v12-type-{t["id"]}">'
            f'<p class="v12-kicker">{esc(t["short"])}</p><h3 id="v12-type-{t["id"]}">{esc(t["name"])}</h3>'
            f'<p class="v12-type-n"><b>{count(row["n"], MO)}</b> · {fpct(row["share"])} территорий</p>'
            '<div class="v12-bars" role="group" aria-label="Медианные отношения категорий к общему показателю '
            f'расходов, 2023">{bars}</div><dl>{details}</dl></article>')


def interval_table(data: dict) -> str:
    rows = data["type_intervals"]
    by = {(row["indicator"], int(row["type"])): row for row in rows}
    head = "".join(f'<th scope="col">{type_label(t)}</th>' for t in TYPES)
    body = []
    for indicator in dict.fromkeys(row["indicator"] for row in rows):
        label, scale, digits = INTERVALS.get(indicator, (indicator, 1, 2))
        cells = []
        for t in TYPES:
            row = by.get((indicator, t["id"]))
            if row is None or number(row["median"]) is None:
                cells.append('<td class="num">нет</td>')
            else:
                cells.append(f'<td class="num"><b>{fnum(scale * number(row["median"]), digits)}</b> '
                             f'<span class="muted">{finterval(row["ci_low"], row["ci_high"], digits, scale)}</span></td>')
        body.append(f'<tr><th scope="row">{esc(label)}</th>{"".join(cells)}</tr>')
    caption = ("Медианы и 95% интервалы при бутстрепе целых регионов. Проценты категорий — отношение к общему "
               "показателю расходов; доли занятых по Росстату считаются по МО, где есть данные (покрытие — в карточках).")
    return table_block(caption, f'<thead><tr><th scope="col">Показатель</th>{head}</tr></thead>'
                                f'<tbody>{"".join(body)}</tbody>', "Интервалы показателей типов")


def quick_warning(data: dict) -> str:
    command = str(data["provenance"].get("command", ""))
    if "--quick" not in command:
        return ""
    return ('<div class="callout"><strong>Предварительный расчёт.</strong> Числа ниже получены командой '
            f'<code>{esc(command)}</code> с уменьшенным числом повторов и могут измениться.</div>')


def section_types(data: dict, ctx: dict) -> str:
    method = ctx["methods"][ctx["chosen_method"]]
    rule = ctx["rules"][ctx["chosen_rule"]]
    lead = (f'Четыре типа построены по признакам 2023 года: {fnum(len(ctx["categories"]))} отношений категорий к общему '
            f'показателю расходов и уровень расходов относительно медианной территории месяца. Метод «{esc(method["label"])}» '
            f'выбран сравнением на атрибутированной сети (рёбра: «{esc(rule["label"])}»). Названия описывают расходы 2023 года. Внешние показатели приведены '
            'с покрытием доступных данных Росстата и СберИндекса. Эта модель открывается на карте по умолчанию '
            'и используется в основном рассказе. Прежние модели доступны в сравнении методов.')
    cards = "".join(type_card(t, ctx["types"][t["id"]], ctx) for t in TYPES)
    return (f'<section class="card v12-card" id="v12-types" aria-labelledby="v12-types-title">{quick_warning(data)}'
            f'{section_head("v12-types-title", "Профили структуры и уровня", lead, "6 признаков · " + count(ctx["n"], MO))}'
            f'<div class="v12-types">{cards}</div>'
            '<p class="note small">Проценты в полосах показывают медианное отношение категории к общему показателю '
            'расходов «Все категории» за 2023 год. Эти категории охватывают только часть бюджета. Уровень расходов — медиана месячного '
            'показателя «Все категории». Нажмите на название территории, чтобы открыть её на карте и в таблице аналогов.</p>'
            f'{interval_table(data)}{strict_transfer(data)}{more(doc_link("NETWORK_TYPOLOGY.md#что-группы-говорят-об-экономике", "типы и их проверка"))}</section>')


def added_value_note(data: dict) -> str:
    rows = [r for r in data.get("added_value", []) if r["contrast"] == "groups_over_spline"]
    if not rows:
        return ""
    family = sum(flag(r["improves_holm_family"]) for r in rows)
    all_tests = sum(flag(r["improves_holm_all"]) for r in rows)
    beyond = sum(flag(r["improves_holm_family"]) for r in data["added_value"] if r["contrast"] == "groups_over_shares")
    body = "".join(f'<tr><th scope="row">{esc(r["label"])}</th><td>{fnum(r["n"])}</td>'
                   f'<td>{fnum(r["delta_r2"], 3, sign=True)}</td>'
                   f'<td>{finterval(r["delta_r2_ci_low"], r["delta_r2_ci_high"], 3)}</td>'
                   f'<td>{fnum(r["p_holm_family"], 3)}</td><td>{fnum(r["p_holm_all"], 3)}</td></tr>' for r in rows)
    head = '<thead><tr><th>Показатель</th><th>МО</th><th>ΔR²</th><th>Обычный 95% интервал</th><th>p Холма, 8</th><th>p Холма, 48</th></tr></thead>'
    return (f'<h3 class="v12-h3">Что группы добавляют сверх уровня расходов</h3>'
            f'<p>При исключении региона группы улучшают сплайн уровня по {fnum(family)} из {fnum(len(rows))} показателей '
            f'после поправки Холма внутри семейства, по {fnum(all_tests)} после общей поправки на 48 проверок. '
            f'Сверх непрерывных отношений категорий подтверждён прирост по {fnum(beyond)} из {fnum(len(rows))} показателей. '
            'Отсутствие значимости не доказывает отсутствие эффекта. Группы помогают кратко описать профиль; '
            'их отдельное экономическое преимущество требует проверки.</p>'
            f'{table_block("Группы и сплайн уровня против одного сплайна. Интервалы не одновременные; поправки показаны в отдельных столбцах.", head + "<tbody>" + body + "</tbody>", "Добавочная ценность групп")}'
            f'<p class="note small">Анализ разведочный. Часть моделей и способ статистического вывода добавлены после просмотра исходов. '
            f'Для добычи доступно 14,4% панели. <a href="{REPOSITORY}reports/v1.2-added-value/README.md">Полный протокол, одновременные интервалы и ограничения</a>.</p>')


def strict_transfer(data: dict) -> str:
    result = data["summary"].get("types", {}).get("strict_leave_one_region_out")
    if not result:
        return '<p class="note small">Строгий перенос с исключением региона из всех стадий обучения пока не включён.</p>'
    names = {"log_wage": "Логарифм зарплаты", "log_population": "Логарифм населения", "log_pop": "Логарифм населения",
             "log_jobs_per_res": "Логарифм рабочих мест на жителя",
             "market_access": "Доступность рынков", "agri_share": "Занятые в сельском хозяйстве",
             "mining_share": "Занятые в добыче", "manuf_share": "Занятые в обрабатывающих производствах",
             "public_share": "Доля занятых в бюджетных отраслях"}
    rows = result["metrics"]
    body = "".join(f'<tr><th scope="row">{esc(names.get(row["indicator"], row["indicator"]))}</th>'
                   f'<td class="num">{fnum(row["n"])}</td><td class="num">{fnum(row["regions"])}</td>'
                   f'<td class="num">{fnum(row["types"], 3)}</td>'
                   f'<td class="num">{fnum(row["municipal_type"], 3)}</td></tr>' for row in rows)
    head = ('<thead><tr><th scope="col">Показатель</th><th scope="col">МО с данными</th>'
            '<th scope="col">Регионов</th><th scope="col">R²: расходные профили</th>'
            '<th scope="col">R²: тип муниципалитета</th></tr></thead>')
    lead = (f'Регион исключён из шкалирования, месячного центра уровня, графа, кластеризации, '
            f'правил именования и экономической модели. Сошлись {fnum(result["converged_folds"])} разбиения. '
            'Проверяется фиксированное исторически выбранное правило; параметры ранее выбраны по всей панели, '
            'включая 2024 год. Вложенный выбор параметров не выполнен. '
            'Каждый показатель оценивается на собственной наблюдаемой выборке; неизвестные исходы в таблицу не входят. '
            'Если административный тип исключённого МО не встречался в обучении, контроль использует среднее обучающих исходов.')
    agreement = (f'Назначения по ближайшему обучающему центру совпали с исходными графовыми группами у '
                 f'{fpct(result["agreement_with_global_labels"])} МО, ARI {fnum(result["ari_with_global_labels"], 3)}. '
                 'Здесь одновременно меняются обучающая выборка и правило назначения; '
                 'это сравнение само по себе не измеряет устойчивость переобучения. '
                 'Наблюдаемые связи и устойчивость не устанавливают производственную специализацию.')
    wage = data.get("wage_level_comparison")
    wage_note = (f'<p><b>Для зарплаты уровень расходов точнее групп.</b> На тех же {fnum(wage["n"])} МО '
                 f'регрессия по медиане логарифмов месячных расходов 2023 года даёт R² {fnum(wage["level_r2"], 3)}, '
                 f'четыре группы дают {fnum(wage["types_r2"], 3)}. Каждый регион исключён из обучения регрессии. '
                 'Группы служат компактным описанием потребления. Добавочная ценность проверена отдельно ниже.</p>') if wage else ''
    return (f'<h3 class="v12-h3">Перенос в исключённый регион</h3><p>{lead}</p>{wage_note}'
            f'{table_block("Все рассчитанные исходы, включая слабые результаты. R² может быть отрицательным.", head + "<tbody>" + body + "</tbody>", "Строгий перенос по регионам")}'
            f'<p class="note small">{agreement}</p>{added_value_note(data)}')


# ------------------------------------------------------------------ section b: network
def network_scatter(data: dict, ctx: dict) -> str:
    rows = data["networks"]
    dynamic = [row for row in rows if not flag(row["static"])]
    static = [row for row in rows if flag(row["static"])]
    threshold = number(data["config"]["network"]["geography_filter"]["min_within_region_assortativity"])
    width, height = CHART_WIDTH, 430
    x0, x1, y0, y1 = 56, 420, 24, 370
    sx0, sx1 = 448, 632
    xmax = nice_ceiling(max(number(row["stability_jaccard"]) for row in dynamic) * 1.05, 4)
    ymax = nice_ceiling(max(number(row["assort_within_region_mean"]) for row in rows) * 1.05, 5)

    def X(v):
        return x0 + (x1 - x0) * v / xmax

    def Y(v):
        return y1 - (y1 - y0) * v / ymax

    parts = [svg_open(width, height, "Правила рёбер: совпадение рёбер между годами и ассортативность внутри регионов")]
    xt, xd = ticks(0, xmax, 4)
    yt, yd = ticks(0, ymax, 5)
    parts.append(f'<rect x="{sx0}" y="{y0}" width="{sx1 - sx0}" height="{y1 - y0}" class="v12-panel"/>')
    for v in yt:
        parts.append(f'<line class="v12-grid" x1="{x0}" x2="{x1}" y1="{f1(Y(v))}" y2="{f1(Y(v))}"/>')
        parts.append(f'<line class="v12-grid" x1="{sx0}" x2="{sx1}" y1="{f1(Y(v))}" y2="{f1(Y(v))}"/>')
        parts.append(svg_text(x0 - 8, Y(v) + 4, fnum(v, yd), anchor="end", cls="v12-muted"))
    for v in xt:
        parts.append(f'<line class="v12-grid" x1="{f1(X(v))}" x2="{f1(X(v))}" y1="{y0}" y2="{y1}"/>')
        parts.append(svg_text(X(v), y1 + 18, fnum(v, xd), anchor="middle", cls="v12-muted"))
    parts.append(f'<line class="v12-axis" x1="{x0}" x2="{x1}" y1="{y1}" y2="{y1}"/>')
    parts.append(f'<line class="v12-axis" x1="{x0}" x2="{x0}" y1="{y0}" y2="{y1}"/>')
    if threshold is not None and threshold <= ymax:
        parts.append(f'<line class="v12-threshold" x1="{x0}" x2="{sx1}" y1="{f1(Y(threshold))}" y2="{f1(Y(threshold))}"/>')
        parts.append(svg_text(x1 - 4, Y(threshold) - 6, f"порог фильтра географии: {fnum(threshold, 2)}",
                              anchor="end", cls="v12-muted v12-small"))
    parts.append(svg_text((x0 + x1) / 2, height - 10, "Совпадение рёбер 2023 → 2024 (Жаккар)", anchor="middle",
                          cls="v12-muted"))
    parts.append(svg_text(14, (y0 + y1) / 2, "Ассортативность внутри регионов", anchor="middle", cls="v12-muted",
                          extra=f' transform="rotate(-90 14 {f1((y0 + y1) / 2)})"'))
    parts.append(svg_text((sx0 + sx1) / 2, y0 + 20, "Статичные графы", anchor="middle", cls="v12-strong"))
    parts.append(svg_text((sx0 + sx1) / 2, y0 + 37, "совпадение = 1", anchor="middle", cls="v12-muted v12-small"))
    parts.append(svg_text((sx0 + sx1) / 2, y0 + 52, "по построению", anchor="middle", cls="v12-muted v12-small"))
    order = sorted(dynamic, key=lambda r: (r["rule"] != ctx["chosen_rule"], -number(r["stability_jaccard"])))
    points = [{"row": row, "x": X(number(row["stability_jaccard"])), "y": Y(number(row["assort_within_region_mean"])),
               "r": 8 if row["rule"] == ctx["chosen_rule"] else 6, "priority": i,
               "text": RULE_SHORT.get(row["rule"], row["label"])} for i, row in enumerate(order)]
    place_labels(points, (x0 + 4, y0 + 2, x1 - 2, y1 - 4))
    static_points = [{"row": row, "x": sx0 + 26, "y": Y(number(row["assort_within_region_mean"])), "r": 6,
                      "priority": i, "text": RULE_SHORT.get(row["rule"], row["label"])} for i, row in enumerate(static)]
    place_labels(static_points, (sx0 + 4, y0 + 60, sx1 - 4, y1 - 4))
    for p in points + static_points:
        row, label = p["row"], p["label"]
        chosen = row["rule"] == ctx["chosen_rule"]
        cls = "v12-pt-chosen" if chosen else "v12-pt-geo" if flag(row["geography_dominated"]) else "v12-pt"
        stability = (fnum(row["stability_jaccard"], 3) if number(row["stability_jaccard"]) is not None
                     else "1 по построению")
        title = (f'{esc(row["label"])}: совпадение рёбер {stability}, ассортативность внутри регионов '
                 f'{fnum(row["assort_within_region_mean"], 3)}')
        if label["leader"]:
            parts.append(f'<line class="v12-leader" x1="{f1(p["x"])}" y1="{f1(p["y"])}" '
                         f'x2="{f1(label["x"])}" y2="{f1(label["y"] - 4)}"/>')
        parts.append(f'<circle class="{cls}" cx="{f1(p["x"])}" cy="{f1(p["y"])}" r="{p["r"]}"><title>{title}</title></circle>')
        parts.append(svg_text(label["x"], label["y"], esc(p["text"]), anchor=label["anchor"],
                              cls="v12-strong" if chosen else ""))
    parts.append("</svg>")
    return chart("".join(parts), "Диаграмма правил построения рёбер")


def network_table(data: dict, ctx: dict) -> str:
    body = []
    for index, row in enumerate(data["networks"], 1):
        chosen = row["rule"] == ctx["chosen_rule"]
        if chosen:
            status = '<span class="v12-badge">выбрано</span>'
        elif flag(row["geography_dominated"]):
            status = "отброшено: повторяет географию"
        elif flag(row["static"]):
            status = "статичное, вне сравнения"
        else:
            status = "участвует"
        stability = (fnum(row["stability_jaccard"], 3) if number(row["stability_jaccard"]) is not None
                     else '<span class="muted">1 по построению</span>')
        copeland = fnum(row["copeland"], 0) if number(row.get("copeland")) is not None else "—"
        body.append(f'<tr{CHOSEN if chosen else ""}><td class="num">{index}</td><th scope="row">{esc(row["label"])}</th>'
                    f'<td class="num">{fnum(row["edges"])}</td><td class="num">{fpct(row["same_region_share"])}</td>'
                    f'<td class="num">{stability}</td><td class="num">{fnum(row["assort_mean"], 3)}</td>'
                    f'<td class="num">{fnum(row["assort_within_region_mean"], 3)}</td><td class="num">{copeland}</td>'
                    f'<td class="v12-nowrap">{status}</td></tr>')
    up = NBSP + "↑"
    head = (f'<tr><th scope="col">№</th><th scope="col">Правило рёбер</th><th scope="col">Рёбер</th>'
            f'<th scope="col">Внутри региона</th><th scope="col">Совпадение 2023 → 2024{up}</th>'
            f'<th scope="col">Ассортативность{up}</th><th scope="col">Внутри регионов{up}</th>'
            f'<th scope="col">Копленд{up}</th><th scope="col">Статус</th></tr>')
    caption = ("Ассортативность — корреляция значений внешних показателей на концах рёбер, среднее по показателям; "
               "«внутри регионов» — после вычета региональных средних. Баллы Коупленда считаются только среди правил, "
               "участвующих в выборе.")
    return table_block(caption, f"<thead>{head}</thead><tbody>{''.join(body)}</tbody>", "Сравнение правил построения рёбер",
                       cls="v12-rules")


def k_sensitivity(data: dict, ctx: dict) -> str:
    rows = sorted(data["network_k_sensitivity"], key=lambda r: int(r["k"]))
    reference = ctx["rules"].get("euclid_profile", {}).get("label", "структура + уровень")
    first, last = rows[0], rows[-1]
    components = sorted({int(row["components"]) for row in rows})
    connected = ("граф остаётся связным при всех k" if components == [1]
                 else "число компонент связности: " + ", ".join(fnum(c) for c in components))
    text = (f'Чувствительность к числу соседей посчитана для правила «{esc(reference)}», по которому строится граф '
            f'для методов ниже. При росте k от {fnum(first["k"])} до {fnum(last["k"])} ассортативность внутри регионов '
            f'меняется с {fnum(first["assort_within_region_mean"], 3)} до {fnum(last["assort_within_region_mean"], 3)}, '
            f'совпадение рёбер 2023 → 2024 — с {fnum(first["stability_jaccard"], 3)} до {fnum(last["stability_jaccard"], 3)}, '
            f'доля рёбер внутри региона — с {fpct(first["same_region_share"])} до {fpct(last["same_region_share"])}; '
            f'{connected}. В основном расчёте k = {fnum(ctx["k_graph"])}.')
    if ctx["chosen_rule"] != "euclid_profile":
        text += " Выбранное правило отличается от правила графа методов; это нужно учитывать при чтении сравнения."
    body = "".join(
        f'<tr{CHOSEN if int(row["k"]) == int(ctx["k_graph"]) else ""}><th scope="row">{fnum(row["k"])}</th>'
        f'<td class="num">{fnum(row["edges"])}</td><td class="num">{fpct(row["same_region_share"])}</td>'
        f'<td class="num">{fnum(row["stability_jaccard"], 3)}</td><td class="num">{fnum(row["assort_mean"], 3)}</td>'
        f'<td class="num">{fnum(row["assort_within_region_mean"], 3)}</td><td class="num">{fnum(row["components"])}</td></tr>'
        for row in rows)
    head = ('<thead><tr><th scope="col">k</th><th scope="col">Рёбер</th><th scope="col">Внутри региона</th>'
            '<th scope="col">Совпадение 2023 → 2024</th><th scope="col">Ассортативность</th>'
            '<th scope="col">Внутри регионов</th><th scope="col">Компонент связности</th></tr></thead>')
    return (f'<h3 class="v12-h3">Сколько соседей</h3><p>{text}</p>'
            + table_block("", f"{head}<tbody>{body}</tbody>", "Чувствительность к числу соседей", cls="v12-auto"))


def network_effects(data: dict, ctx: dict) -> str:
    rows = data["network_effects"]
    if not rows or "joint_ari_graph_2023_vs_2024" not in rows[0]:
        return "", ""
    dynamic = [row for row in rows if number(row["joint_ari_graph_2023_vs_2024"]) is not None]
    chosen = next((row for row in dynamic if row["rule"] == ctx["chosen_rule"]), None)
    others = [number(row["joint_ari_graph_2023_vs_2024"]) for row in dynamic if row["rule"] != ctx["chosen_rule"]]
    leiden = [number(row["leiden_ari_graph_2023_vs_2024"]) for row in dynamic
              if number(row["leiden_ari_graph_2023_vs_2024"]) is not None]
    text = ""
    if chosen and others and leiden:
        joint_values = [number(row["joint_ari_graph_2023_vs_2024"]) for row in dynamic]
        text = (f'<p>Правило рёбер меняет и результат кластеризации. Если при тех же признаках заменить граф 2023 года '
                f'графом 2024 года, совместная модель признаков и графа на выбранном графе сохраняет разбиение с ARI '
                f'{fnum(chosen["joint_ari_graph_2023_vs_2024"], 3)}, на остальных динамических правилах — от '
                f'{fnum(min(others), 3)} до {fnum(max(others), 3)}. Сообщества Leiden, построенные только по графу, '
                f'{"воспроизводятся хуже" if max(leiden) < min(joint_values) else "воспроизводятся иначе"}: ARI от '
                f'{fnum(min(leiden), 3)} до {fnum(max(leiden), 3)}.</p>')
    body = []
    for row in rows:
        static = number(row["joint_ari_graph_2023_vs_2024"]) is None

        def ari(key: str) -> str:
            return '<span class="muted">статичный граф</span>' if static else fnum(row[key], 3)

        body.append(f'<tr{CHOSEN if row["rule"] == ctx["chosen_rule"] else ""}><th scope="row">{esc(row["label"])}</th>'
                    f'<td class="num">{fnum(row["leiden_communities"])}</td><td class="num">{fnum(row["leiden_modularity"], 3)}</td>'
                    f'<td class="num">{ari("leiden_ari_graph_2023_vs_2024")}</td>'
                    f'<td class="num">{fnum(row["joint_ari_with_kmeans4"], 3)}</td>'
                    f'<td class="num">{ari("joint_ari_graph_2023_vs_2024")}</td>'
                    f'<td class="num">{fnum(row["joint_partial_r2_within_region_mean"], 3)}</td></tr>')
    head = ('<thead><tr><th scope="col">Правило рёбер</th><th scope="col">Сообществ Leiden</th>'
            '<th scope="col">Модулярность</th><th scope="col">Leiden: ARI графов 2023 и 2024</th>'
            '<th scope="col">Совместная модель: ARI с KMeans</th><th scope="col">Совместная модель: ARI графов 2023 и 2024</th>'
            '<th scope="col">Совместная модель: partial R² внутри регионов</th></tr></thead>')
    caption = ("ARI сравнивает разбиения без учёта номеров групп: 1 — одинаковые. Partial R² — средняя доля дисперсии "
               "внешних показателей, которую группы объясняют сверх региона.")
    details = ('<details class="comparison-details"><summary>Как правило рёбер меняет кластеризацию</summary>'
               f'<div class="details-body">{table_block(caption, head + "<tbody>" + "".join(body) + "</tbody>", "Влияние правила рёбер на кластеризацию", cls="v12-rules")}'
               '</div></details>')
    return text, details


def section_network(data: dict, ctx: dict) -> str:
    rows = data["networks"]
    rule = ctx["rules"][ctx["chosen_rule"]]
    indicators = [c for c in rows[0] if c.startswith("assort_within_region_") and c != "assort_within_region_mean"]
    dropped = [row for row in rows if flag(row["geography_dominated"])]
    rules = count(len(rows), ("правило", "правила", "правил"))
    lead = (f'Вершины сети — {count(ctx["n"], MO)}. Каждая территория выбирает до {fnum(ctx["k_graph"])} ближайших соседей '
            f'с доступными данными; после симметризации связей может стать больше. Я сравнил {rules} построения рёбер на {count(len(indicators), ("показателе", "показателях", "показателях"))}, '
            'которые в признаки не входят: зарплата, занятость, население, доступность рынков.')
    verdict = (f'Выбрано правило «{esc(rule["label"])}»'
               + (f' с суммой баллов Коупленда {fnum(rule["copeland"], 0)}' if number(rule.get("copeland")) is not None else "")
               + "." + (" Как повторяющие административную географию отброшены: "
                        + ", ".join(f'«{esc(row["label"])}»' for row in dropped) + "." if dropped else ""))
    verdict += (' Стабильность рёбер предпочтительна для годовых аналогов, но может снижать оценку корреляций и DTW по 12 месячным точкам. '
                'Внешние показатели использованы и для выбора рёбер, и для оценки групп; независимой проверкой выбора сети эта оценка не служит.')
    effects_text, effects_details = network_effects(data, ctx)
    return (f'<section class="card v12-card" id="v12-network" aria-labelledby="v12-network-title">'
            f'{section_head("v12-network-title", "Как выбрана сеть", lead, rules + " построения рёбер")}'
            f'<div class="v12-split"><div>{network_scatter(data, ctx)}'
            '<p class="note small">Каждая точка — правило построения рёбер. Статичные графы не меняются между годами, '
            'поэтому их совпадение рёбер равно 1 по построению; они показаны отдельно по той же вертикальной шкале. '
            'Наведите курсор на точку, чтобы увидеть значения.</p></div>'
            f'<div class="v12-text"><h3 class="v12-h3">Правило выбора</h3><p>{esc(data["config"]["network"]["selection_rule"].replace("независимым показателям", "внешним показателям"))}</p>'
            f'<p>{verdict}</p>{effects_text}</div></div>{effects_details}'
            f'{network_table(data, ctx)}{k_sensitivity(data, ctx)}'
            f'{more(doc_link("NETWORK_TYPOLOGY.md#девять-правил-рёбер", "сравнение девяти правил рёбер"))}</section>')


# ------------------------------------------------------------------ section c: methods
ICVI = (("SW", "↑"), ("CH", "↑"), ("S_Dbw", "↓"), ("AVI", "↑"), ("AVU", "↓"), ("MQ", "↑"))
ICVI_DIGITS = {"SW": 3, "CH": 1, "S_Dbw": 3, "AVI": 3, "AVU": 3, "MQ": 3}


def methods_table(data: dict, ctx: dict) -> str:
    rules = data["config"]["methods"]["eligibility"]
    n, min_share, min_ari = ctx["n"], number(rules["min_type_share"]), number(rules["min_month_bootstrap_ari_p10"])
    icvi_head = "".join(f'<th scope="col"{GROUP if i == 0 else ""}>{name}{NBSP}{arrow}</th>'
                        for i, (name, arrow) in enumerate(ICVI))
    head = ('<thead><tr><th scope="col" rowspan="2" class="v12-sticky">Метод</th><th scope="col" rowspan="2">Семейство</th>'
            f'<th scope="col" rowspan="2">Малая группа, МО</th><th scope="col" rowspan="2">ARI p10{NBSP}↑</th>'
            '<th scope="col" rowspan="2">Допустим</th>'
            f'<th scope="colgroup" colspan="4" class="v12-group">Баллы Коупленда{NBSP}↑</th>'
            '<th scope="colgroup" colspan="6" class="v12-group">ICVI 2023 · обучение</th>'
            '<th scope="colgroup" colspan="6" class="v12-group">ICVI 2024 · ретроспектива</th></tr>'
            '<tr><th scope="col" class="v12-group">2023, все</th><th scope="col">2023</th><th scope="col">2024</th>'
            f'<th scope="col">Сумма</th>{icvi_head}{icvi_head}</tr></thead>')
    body = []
    for row in data["methods"]:
        chosen = row["method"] == ctx["chosen_method"]
        reasons = []
        if number(row["min_size_2023"]) < min_share * n:
            reasons.append("малая группа")
        if number(row["month_bootstrap_ari_p10"]) is None or number(row["month_bootstrap_ari_p10"]) < min_ari:
            reasons.append("неустойчив")
        eligible = "да" if flag(row["eligible"]) else "нет: " + ", ".join(reasons or ["критерий"])
        badge = '<span class="v12-badge">выбран</span>' if chosen else ""
        cells = [f'<th scope="row" class="v12-sticky">{esc(row["label"])}{badge}</th>', f'<td>{esc(row["family"])}</td>',
                 f'<td class="num">{fnum(row["min_size_2023"])}</td>',
                 f'<td class="num">{fnum(row["month_bootstrap_ari_p10"], 3)}</td>', f'<td class="v12-nowrap">{eligible}</td>']
        for i, key in enumerate(("copeland_all_2023", "copeland_2023", "copeland_2024", "copeland_total")):
            value = fnum(row[key], 0) if number(row.get(key)) is not None else "—"
            cells.append(f'<td class="num{" v12-group" if i == 0 else ""}">{"<b>" + value + "</b>" if key == "copeland_total" else value}</td>')
        for year in ("2023", "2024"):
            cells += [f'<td class="num{" v12-group" if j == 0 else ""}">{fnum(row[f"{name}_{year}"], ICVI_DIGITS[name])}</td>'
                      for j, (name, _) in enumerate(ICVI)]
        body.append(f'<tr{CHOSEN if chosen else ""}>{"".join(cells)}</tr>')
    draws = data["provenance"].get("draws", {}).get("methods")
    caption = ("ARI p10 — 10-й перцентиль согласия с исходным разбиением при бутстрепе месяцев 2023 года"
               + (f" ({count(draws, DRAW_FORMS)})" if draws else "")
               + ". «2023, все» — баллы Коупленда по шести ICVI среди всех методов; остальные баллы — только среди "
               "допустимых. Стрелка показывает лучшее направление индекса. ICVI 2024: профили 2024 года после вычета "
               "общей волны, отнесённые к центрам 2023 года. Таблица прокручивается вбок.")
    return table_block(caption, head + f'<tbody>{"".join(body)}</tbody>', "Сравнение методов кластеризации", wide=True)


def k_details(data: dict, ctx: dict) -> str:
    rows = sorted(data["k_selection"], key=lambda r: int(r["k"]))
    chosen = ctx["chosen_k"]
    best = next(row for row in rows if int(row["k"]) == chosen)
    scores = [number(row["stability_score"]) for row in rows if int(row["k"]) != chosen]
    body = "".join(
        f'<tr{CHOSEN if int(row["k"]) == chosen else ""}><th scope="row">{fnum(row["k"])}</th>'
        f'<td class="num">{fnum(row["min_size"])}</td><td class="num">{fnum(row["SW"], 3)}</td>'
        f'<td class="num">{fnum(row["MQ"], 3)}</td><td class="num">{fnum(row["eta2_mean"], 3)}</td>'
        f'<td class="num">{fnum(row["month_bootstrap_ari_mean"], 3)}</td>'
        f'<td class="num">{fnum(row["territory_subsample_ari_mean"], 3)}</td>'
        f'<td class="num"><b>{fnum(row["stability_score"], 3)}</b></td></tr>' for row in rows)
    head = ('<thead><tr><th scope="col">K</th><th scope="col">Малая группа</th><th scope="col">SW</th><th scope="col">MQ</th>'
            '<th scope="col">Внешняя η², среднее</th><th scope="col">ARI, бутстреп месяцев</th>'
            '<th scope="col">ARI, подвыборки 80%</th><th scope="col">Устойчивость</th></tr></thead>')
    text = (f'<h3 class="v12-h3">Число типов</h3><p>K выбрано до сравнения методов, на KMeans: K = {fnum(chosen)} даёт '
            f'устойчивость {fnum(best["stability_score"], 3)}, остальные K — от {fnum(min(scores), 3)} до '
            f'{fnum(max(scores), 3)}. {esc(data["config"]["k_selection"]["selection_rule"])}</p>')
    details = ('<details class="comparison-details"><summary>Выбор числа типов K</summary><div class="details-body">'
               f'{table_block("", head + "<tbody>" + body + "</tbody>", "Выбор числа типов", cls="v12-auto")}</div></details>')
    return text, details


def feature_spaces(data: dict, ctx: dict) -> str:
    rows = data["feature_spaces"]
    if not rows:
        return ""
    body = []
    for row in rows:
        chosen = "(выбрано)" in row["label"]
        label = esc(row["label"].replace("(выбрано)", "").strip())
        body.append(f'<tr{CHOSEN if chosen else ""}><th scope="row">{label}{BADGE_SPACE if chosen else ""}</th>'
                    f'<td class="num">{fnum(row["attributes"])}</td><td class="num">{fnum(row["min_size"])}</td>'
                    f'<td class="num">{fnum(row["eta2_mean"], 3)}</td><td class="num">{fnum(row["partial_r2_within_region_mean"], 3)}</td>'
                    f'<td class="num">{fnum(row["month_bootstrap_ari_p10"], 3)}</td>'
                    f'<td class="num">{fnum(row["territory_subsample_ari_p10"], 3)}</td>'
                    f'<td class="num">{fnum(row["ari_with_profile6"], 3)}</td></tr>')
    head = ('<thead><tr><th scope="col">Пространство признаков</th><th scope="col">Признаков</th>'
            '<th scope="col">Малая группа</th><th scope="col">Внешняя η², среднее</th>'
            '<th scope="col">Partial R² внутри регионов</th><th scope="col">ARI p10, бутстреп месяцев</th>'
            '<th scope="col">ARI p10, подвыборки 80%</th><th scope="col">ARI с выбранным</th></tr></thead>')
    caption = (f'KMeans при K = {fnum(ctx["chosen_k"])} в каждом пространстве. Внутренние индексы в разных пространствах '
               'несравнимы, поэтому показаны внешняя валидность и устойчивость.')
    return ('<details class="comparison-details"><summary>Пространство признаков</summary><div class="details-body">'
            f'{table_block(caption, head + "<tbody>" + "".join(body) + "</tbody>", "Сравнение пространств признаков", cls="v12-rules")}'
            '</div></details>')



def section_methods(data: dict, ctx: dict) -> str:
    rows = data["methods"]
    chosen = ctx["methods"][ctx["chosen_method"]]
    eligible = [row for row in rows if flag(row["eligible"])]
    families = list(dict.fromkeys(row["family"] for row in rows))
    summary = data["summary"]["methods"]
    lead = (f'{count(len(rows), ("вариант", "варианта", "вариантов"))} из {count(len(families), ("семейства", "семейств", "семейств"))}: '
            f'{", ".join(esc(f) for f in families)}. Индексы качества посчитаны на 2023 году, где методы обучались, '
            'и на 2024 году. Данные 2024 участвовали в выборе метода и сети; независимым тестом они не служат. Проверка на новом периоде ещё предстоит.')
    fit = summary.get("fit") or {}
    k_text, k_table = k_details(data, ctx)
    text = (f'<h3 class="v12-h3">Правило выбора</h3><p>{esc(data["config"]["methods"]["selection_rule"])}</p>'
            f'<p>Допустимы {fnum(len(eligible))} из {fnum(len(rows))}. Выбран «{esc(chosen["label"])}» '
            f'(сумма баллов Коупленда {fnum(chosen["copeland_total"], 0)}). От KMeans он отличается у '
            f'{fpct(summary["moved_vs_kmeans_share"])} МО, ARI с KMeans {fnum(summary["ari_final_vs_kmeans"], 3)}.'
            + (f' Модель сошлась за {count(fit["sweeps"], ("проход", "прохода", "проходов"))}; доля рёбер между типами '
               f'{fpct(fit["graph_cut_fraction"])}.' if fit.get("converged") else "") + '</p>')
    text += ('<p>Граф kNN построен в том же пространстве шести расходных признаков. Совместная модель '
             'регуляризует KMeans: штрафует разные назначения у похожих территорий. AVI, AVU и MQ оцениваются '
             'на этом графе и не дают независимого подтверждения групп. Уменьшение целевой функции не гарантирует '
             'улучшения каждого индекса. Независимый от расходов дорожный граф проверен в архиве; '
             'преимущество сверх регионального контроля там не подтверждено.</p>'
             f'<p>Силуэт SW = {fnum(chosen["SW_2023"], 3)}: профили групп слабо обособлены. Границы типов '
             'не следует воспринимать как разрывы между изолированными экономиками. K выбрано по устойчивости '
             'на отдельном этапе; более высокий ICVI сам по себе не определяет число типов.</p>')
    audit = summary.get("selection_audit")
    if audit:
        alternative = ctx["methods"][audit["chosen_2023_only"]]["label"]
        text += (f'<p>При выборе только по 2023 году побеждает «{esc(alternative)}»: '
                 f'меняются {fnum(audit["changed_n"])} из {fnum(audit["entities"])} назначений '
                 f'({fpct(audit["changed_n"] / audit["entities"])}). Это показывает близость двух разбиений; '
                 'устойчивость экономических выводов к смене веса отдельно не проверена.</p>')
    ties = data.get("method_ties", [])
    if ties:
        joint = [r for r in ties if r["method"].startswith("joint_")]
        selected = next(r for r in ties if r["method"] == ctx["chosen_method"])
        differences = [int(r["municipalities_differing_from_selected"]) for r in joint if r is not selected]
        text += (f'<p>Среди {fnum(len(ties))} допустимых вариантов семейство совместной модели выигрывает в '
                 f'{fpct(sum(float(r["win_share_2023_2024"]) for r in joint))} региональных подвыборок, '
                 f'точный выбранный вес в {fpct(selected["win_share_2023_2024"])}. Другие α меняют '
                 f'{fnum(min(differences))}–{fnum(max(differences))} назначений. Это 500 подвыборок по 58 из 73 регионов '
                 'с фиксированными разбиениями, без повторного обучения; проверяется чувствительность ранжирования. '
                 f'<a href="{REPOSITORY}reports/v1.2-method-ties/README.md">Размен между индексами и условия сравнения</a>.</p>')
    return (f'<section class="card v12-card" id="v12-methods" aria-labelledby="v12-methods-title">'
            f'{section_head("v12-methods-title", "Сравнение методов на атрибутированной сети", lead, "шесть ICVI · 2023 и 2024")}'
            f'<div class="v12-split"><div class="v12-text">{text}</div><div class="v12-text">{k_text}</div></div>'
            f'{k_table}{feature_spaces(data, ctx)}{methods_table(data, ctx)}'
            '<p class="note small">SW и CH описывают компактность профилей, S_Dbw — разброс и плотность, AVI, AVU и MQ — '
            'связи на графе. Ни один индекс по отдельности не подтверждает экономический смысл типов. Индексы 2024 года '
            'считаются по фиксированным центрам 2023 года и проверяют перенос разбиения на следующий год.</p>'
            f'{more(doc_link("NETWORK_TYPOLOGY.md#почему-четыре-группы", "методология и формулы"))}</section>')


# ------------------------------------------------------------------ section d: dynamics
def alluvial(matrix: list[list[int]], title_after: str) -> str:
    width, height = CHART_WIDTH, 470
    left_x, right_x, bar = 182, 442, 16
    top, bottom, gap = 44, 426, 14
    scale = (bottom - top - gap * 3) / sum(map(sum, matrix))
    rows = [sum(r) for r in matrix]
    cols = [sum(matrix[i][j] for i in range(4)) for j in range(4)]
    left, right, y = [], [], top
    for v in rows:
        left.append(y)
        y += v * scale + gap
    y = top
    for v in cols:
        right.append(y)
        y += v * scale + gap
    parts = [svg_open(width, height, f"Переходы между типами: 2023 → {title_after}")]
    parts.append(svg_text(left_x + bar / 2, 24, "2023", anchor="middle", cls="v12-strong"))
    parts.append(svg_text(right_x + bar / 2, 24, title_after, anchor="middle", cls="v12-strong"))
    lo, ro = left[:], right[:]
    xa, xb = left_x + bar, right_x
    mid = (xa + xb) / 2
    for i in range(4):
        for j in range(4):
            value = matrix[i][j]
            if not value:
                continue
            h = max(value * scale, .8)
            a, b = lo[i], ro[j]
            path = (f"M{f1(xa)} {f1(a)} C{f1(mid)} {f1(a)} {f1(mid)} {f1(b)} {f1(xb)} {f1(b)} L{f1(xb)} {f1(b + h)} "
                    f"C{f1(mid)} {f1(b + h)} {f1(mid)} {f1(a + h)} {f1(xa)} {f1(a + h)} Z")
            parts.append(f'<path d="{path}" fill="{TYPES[i]["color"]}" class="{"v12-stay" if i == j else "v12-move"}">'
                         f'<title>{esc(TYPES[i]["name"])} → {esc(TYPES[j]["name"])}: {count(value, MO)}</title></path>')
            lo[i] += value * scale
            ro[j] += value * scale
    for i, t in enumerate(TYPES):
        lh, rh = max(rows[i] * scale, 2), max(cols[i] * scale, 2)
        parts.append(f'<rect x="{left_x}" y="{f1(left[i])}" width="{bar}" height="{f1(lh)}" rx="3" fill="{t["color"]}">'
                     f'<title>{esc(t["name"])}: {count(rows[i], MO)} в 2023 году</title></rect>')
        parts.append(f'<rect x="{right_x}" y="{f1(right[i])}" width="{bar}" height="{f1(rh)}" rx="3" fill="{t["color"]}">'
                     f'<title>{esc(t["name"])}: {count(cols[i], MO)} в 2024 году</title></rect>')
        cy, ry, delta = left[i] + lh / 2, right[i] + rh / 2, cols[i] - rows[i]
        parts.append(wrapped_svg_label(left_x - 10, cy - 5, t["short"], left_x - 20, anchor="end"))
        parts.append(svg_text(left_x - 10, cy + 27, count(rows[i], MO), anchor="end", cls="v12-muted"))
        parts.append(wrapped_svg_label(right_x + bar + 10, ry - 5, t["short"], width - right_x - bar - 20))
        parts.append(svg_text(right_x + bar + 10, ry + 27,
                              f"{count(cols[i], MO)} ({fnum(delta, 0, sign=True) if delta else '±0'})", cls="v12-muted"))
    parts.append("</svg>")
    return chart("".join(parts), "Диаграмма переходов между типами")


def transition_table(matrix: list[list[int]]) -> str:
    head = "".join(f'<th scope="col">{type_label(t)}</th>' for t in TYPES)
    body = "".join(
        f'<tr><th scope="row">{type_label(TYPES[i])}</th>'
        + "".join(f'<td class="num">{"<b>" + fnum(v) + "</b>" if i == j else fnum(v)}</td>' for j, v in enumerate(row))
        + "</tr>" for i, row in enumerate(matrix))
    table = f'<thead><tr><th scope="col">2023 \\ 2024</th>{head}</tr></thead><tbody>{body}</tbody>'
    return ('<details class="comparison-details"><summary>Таблица переходов 2023 → 2024</summary>'
            f'<div class="details-body">{table_block("", table, "Таблица переходов")}</div></details>')


def monthly_chart(data: dict) -> str:
    rows = data["monthly_network"]
    width = CHART_WIDTH
    keys, top = legend([(t["color"], t["short"]) for t in TYPES], 10, 20, width - 20)
    x0, x1, y0 = 52, width - 8, top + 10
    y1 = y0 + 210
    height = y1 + 44
    ymax = max(sum(int(row[f"n_type_{c}"]) for c in range(4)) for row in rows)
    yt, yd = ticks(0, ymax, 5)
    band = (x1 - x0) / len(rows)
    barw = band * 0.72

    def Y(v):
        return y1 - (y1 - y0) * v / ymax

    parts = [svg_open(width, height, "Число МО каждого типа по месяцам"), keys]
    for v in yt:
        if v <= ymax:
            parts.append(f'<line class="v12-grid" x1="{x0}" x2="{x1}" y1="{f1(Y(v))}" y2="{f1(Y(v))}"/>')
            parts.append(svg_text(x0 - 6, Y(v) + 4, fnum(v, yd), anchor="end", cls="v12-muted"))
    for t, row in enumerate(rows):
        x, stack = x0 + t * band + (band - barw) / 2, y1
        counts = [int(row[f"n_type_{c}"]) for c in range(4)]
        group = [f'<g><title>{esc(period_label(row["period"]))}: '
                 + "; ".join(f"{esc(TYPES[c]['short'])} — {fnum(counts[c])}" for c in range(4)) + "</title>"]
        for c in reversed(range(4)):
            h = (y1 - y0) * counts[c] / ymax
            group.append(f'<rect x="{f1(x)}" y="{f1(stack - h)}" width="{f1(barw)}" height="{f1(h)}" fill="{TYPES[c]["color"]}"/>')
            stack -= h
        parts.append("".join(group) + "</g>")
        month = int(row["period"][5:7])
        if month in (1, 4, 7, 10):
            parts.append(svg_text(x + barw / 2, y1 + 17, MONTHS_SHORT[month - 1], anchor="middle", cls="v12-muted"))
        if month == 1:
            parts.append(svg_text(x + barw / 2, y1 + 34, row["period"][:4], anchor="middle", cls="v12-strong"))
            if t:
                sep = x0 + t * band
                parts.append(f'<line class="v12-year" x1="{f1(sep)}" x2="{f1(sep)}" y1="{f1(y0 - 4)}" y2="{f1(y1 + 38)}"/>')
    parts.append(f'<line class="v12-axis" x1="{x0}" x2="{x1}" y1="{f1(y1)}" y2="{f1(y1)}"/></svg>')
    return chart("".join(parts), "Число МО каждого типа по месяцам")


def events_table(data: dict) -> str:
    body = []
    for row in sorted(data["reclustering_events"], key=lambda r: (int(r["type"]), r["year"])):
        body.append(f'<tr><th scope="row">{type_label(TYPES[int(row["type"])])}</th><td>{esc(row["year"])}</td>'
                    f'<td class="num">{fnum(row["survival_months"])}</td><td class="num">{fnum(row["absorption_months"])}</td>'
                    f'<td class="num">{fnum(row["split_months"])}</td><td class="num">{fnum(row["disappearance_months"])}</td>'
                    f'<td class="num">{fnum(row["median_jaccard_with_base"], 2)}</td></tr>')
    head = ('<thead><tr><th scope="col">Тип 2023</th><th scope="col">Год</th><th scope="col">Сохраняется, мес.</th>'
            '<th scope="col">Поглощён, мес.</th><th scope="col">Распался, мес.</th><th scope="col">Исчез, мес.</th>'
            '<th scope="col">Медиана Жаккара</th></tr></thead>')
    note = ('Каждый месяц кластеризуется заново и сопоставляется с типами 2023 года по наибольшему пересечению '
            '(схема MONIC). «Сохраняется»: совпадение не меньше половины состава и по Жаккару; «поглощён»: большинство '
            'членов внутри более крупной группы; «распался»: члены разошлись по двум и более группам; иначе «исчез».')
    return ('<details class="comparison-details"><summary>Свободная перекластеризация по месяцам</summary>'
            f'<div class="details-body">{table_block(note, head + "<tbody>" + "".join(body) + "</tbody>", "События свободной перекластеризации")}'
            '</div></details>')


def reclustering_text(data: dict, matrix: list[list[int]]) -> str:
    remote = TYPE_ORDER.index("remote")
    t = TYPES[remote]
    events = {(int(r["type"]), r["year"]): r for r in data["reclustering_events"]}
    years = sorted({year for _, year in events})
    if len(years) < 2 or (remote, years[0]) not in events:
        return ""
    first, second = events[(remote, years[0])], events[(remote, years[1])]

    def months(row):
        return sum(int(row[key]) for key in ("survival_months", "absorption_months", "split_months", "disappearance_months"))

    s1, s2, a2 = int(first["survival_months"]), int(second["survival_months"]), int(second["absorption_months"])
    text = (f'<b>Свободная перекластеризация.</b> Если заново кластеризовать каждый месяц и сопоставить группы с типами '
            f'2023 года, тип {quoted(t)} сохраняется в {fnum(s1)} из {fnum(months(first))} месяцев {years[0]} года и в '
            f'{fnum(s2)} из {fnum(months(second))} месяцев {years[1]} года')
    if a2:
        scope = f"во всех {fnum(a2)}" if a2 == months(second) else f"в {fnum(a2)}"
        text += f'; {scope} {plural(a2, ("месяце", "месяцах", "месяцах"))} {years[1]} года его МО входят в более крупную группу.'
    else:
        text += "."
    free = data["summary"]["tracking"]["free_reclustering"]
    overlap, profiles = free.get("annual_2024_overlap"), free.get("annual_2024_profiles")
    if overlap and profiles:
        target = max(range(4), key=lambda j: overlap[remote][j])
        group = next(p for p in profiles if int(p["group"]) == target)
        latitudes = sorted(number(row["lat"]) for row in data["labels"] if number(row["lat"]) is not None)
        median_lat = (latitudes[len(latitudes) // 2] + latitudes[(len(latitudes) - 1) // 2]) / 2
        northern = number(group["latitude_median"]) > median_lat
        inflow = "".join(f", {fnum(overlap[i][target])} — из типа {quoted(TYPES[i])}"
                         for i in range(4) if i != remote and overlap[i][target])
        text += (f' Свободная кластеризация годового профиля {years[1]} года относит большинство его МО к '
                 f'{"более северной " if northern else ""}группе из {count(group["n"], MO)} (медианная широта '
                 f'{fnum(group["latitude_median"], 1)}° против {fnum(median_lat, 1)}° у всех МО): {fnum(overlap[remote][target])} '
                 f'из {fnum(sum(overlap[remote]))} МО типа{inflow}. Чаще всего в ней встречаются: '
                 f'{region_counts(group["top_regions"])}.')
    dynamics = {int(r["type"]): r for r in data["type_dynamics"]}.get(remote)
    if dynamics:
        text += (f' Медианное отношение маркетплейсов к общему показателю в этом типе выросло с '
                 f'{fnum(dynamics["marketplaces_pct_2023"], 1)}% до {fnum(dynamics["marketplaces_pct_2024"], 1)}%.')
    kept = sum(matrix[i][remote] for i in range(4))
    text += (f' Типы 2024 года на карте получены отнесением к фиксированным центрам 2023 года; в 2024 году '
             f'к этому типу отнесены {count(kept, MO)}.')
    return f'<p>{text}</p>'


def section_dynamics(data: dict, ctx: dict) -> str:
    current = ROOT / 'reports/temporal-v4/real-correction.json'
    if current.is_file():
        from scripts.current_evidence_web import render_temporal
        return render_temporal(current)
    tracking = data["summary"]["tracking"]
    matrix, absolute = tracking["annual_transitions"], tracking["annual_transitions_absolute"]
    n = sum(map(sum, matrix))
    changed = n - sum(matrix[i][i] for i in range(4))
    changed_abs = n - sum(absolute[i][i] for i in range(4))
    persistence = tracking["persistence"]
    months_total = len(data["monthly_network"])
    window = persistence["persistent_change_window_months"]
    lead = ('Центры типов зафиксированы по 2023 году. Каждый месяц и годовой профиль 2024 года относятся к ближайшему '
            'центру после вычета общей помесячной волны — сдвига медианной территории, общего для всех МО '
            '(например, роста маркетплейсов).')
    remote = next(row for row in data["type_dynamics"] if row["key"] == "remote")
    interpretation = (
        f'<p>У группы с низкой долей маркетплейсов медианное отношение расходов на маркетплейсах к общему итогу '
        f'выросло с {fnum(remote["marketplaces_pct_2023"], 1)}% до {fnum(remote["marketplaces_pct_2024"], 1)}% '
        f'({fnum(remote["marketplaces_ratio_2024_to_2023"], 2)} раза). Без поправки на общую волну '
        f'{fpct(remote["left_type_2024_absolute_share"])} её МО получили другое назначение. '
        'Это изменение расходного профиля. Сокращение разрыва с другими группами в отдельной проверке не подтверждено.</p>'
        '<p>Расширение доставки в удалённых территориях могло бы объяснить этот сдвиг, но это гипотеза. '
        'Доли расходов также меняются при перераспределении покупок и изменении охвата данных. '
        'Проверка механизма требует сведений о доставке и торговой инфраструктуре; сама смена типа его не доказывает.</p>')
    facts = [
        (fpct(persistence["constant_share"]), f"МО все {count(months_total, MONTH_FORMS)} остаются в одном типе"),
        (fpct(persistence["months_in_annual_type_share"]), "месяцев МО проводят в своём типе 2023 года"),
        (fnum(persistence["mean_switches"], 2), f"смены типа в среднем за {count(months_total, MONTH_FORMS)}"),
        (fnum(persistence["persistent_changes"]), f"МО держат новый тип все последние {count(window, MONTH_FORMS)}"),
    ]
    tiles = "".join(f'<div class="metric"><b>{value}</b><small>{esc(text)}</small></div>' for value, text in facts)
    markov = "; ".join(f"{quoted(TYPES[c])} — {fnum(v, 1)}" for c, v in enumerate(tracking["markov"]["expected_months_in_type"]))
    side = (f'<p>За год тип сменили {count(changed, MO)} из {fnum(n)} ({fpct(changed / n)}); без поправки на общую волну — '
            f'{count(changed_abs, MO)}. Наведите курсор на ленту, чтобы увидеть число МО.</p>'
            f'<p>По помесячным переходам ожидаемое число месяцев подряд в типе: {markov}. Это свойство однородной '
            'марковской цепи по всей панели. Для прогноза отдельной территории эта оценка не проверена.</p>')
    return (f'<section class="card v12-card" id="v12-dynamics" aria-labelledby="v12-dynamics-title">'
            f'{section_head("v12-dynamics-title", "Как меняются профили", lead, count(months_total, MONTH_FORMS) + " · 2023–2024")}'
            f'<div class="v12-text">{interpretation}</div><div class="metrics v12-metrics">{tiles}</div>'
            f'<div class="v12-split"><div><h3 class="v12-h3">Типы 2023 и 2024 года</h3>'
            f'{alluvial(matrix, "2024 · с поправкой")}{transition_table(matrix)}</div>'
            f'<div class="v12-text"><h3 class="v12-h3">По месяцам</h3>{monthly_chart(data)}'
            '<p class="note small">Помесячное отнесение шумнее годового: отдельный месяц может временно сдвинуть '
            f'территорию к соседнему типу. Наведите курсор на столбец, чтобы увидеть числа.</p>{side}</div></div>'
            f'<div class="v12-text">{reclustering_text(data, matrix)}</div>{events_table(data)}'
            f'{more(doc_link("NETWORK_TYPOLOGY.md#как-группы-меняются", "метод отслеживания и проверки"))}</section>')


# ------------------------------------------------------------------ section e: practical
def predictor_label(name: str, k: int) -> str:
    labels = {"network_analogues": f"{k} аналогов по профилю",
              "network_analogues_same_region": f"{k} аналогов по профилю в своём регионе",
              "random_same_region": f"{k} случайных МО своего региона",
              "region_median_leave_one_out": "Медиана своего региона",
              "random_same_type": f"{k} случайных МО своего типа", "national_median": "Медиана всех МО"}
    return labels.get(name, name)


def mae_chart(data: dict, k: int) -> str:
    rows = sorted(data["analogues"], key=lambda r: number(r["mae_pp"]))
    width, row_h, top = CHART_WIDTH, 34, 10
    bottom = top + row_h * len(rows)
    height = bottom + 42
    x0, x1 = 290, 588
    xmax = nice_ceiling(max(number(r["mae_pp"]) for r in rows) * 1.05, 4)
    xt, xd = ticks(0, xmax, 4)

    def X(v):
        return x0 + (x1 - x0) * v / xmax

    parts = [svg_open(width, height, "Средняя абсолютная ошибка оценки роста расходов")]
    for v in xt:
        parts.append(f'<line class="v12-grid" x1="{f1(X(v))}" x2="{f1(X(v))}" y1="{top}" y2="{bottom}"/>')
        parts.append(svg_text(X(v), bottom + 18, fnum(v, xd), anchor="middle", cls="v12-muted"))
    for i, row in enumerate(rows):
        y, network = top + i * row_h, row["predictor"].startswith("network")
        name = predictor_label(row["predictor"], k)
        parts.append(svg_text(x0 - 10, y + row_h / 2 + 4, esc(name), anchor="end", cls="v12-strong" if network else ""))
        w = X(number(row["mae_pp"])) - x0
        parts.append(f'<rect x="{x0}" y="{f1(y + 7)}" width="{f1(w)}" height="{row_h - 14}" rx="3" '
                     f'class="{"v12-bar-net" if network else "v12-bar-base"}"><title>{esc(name)}: {fnum(row["mae_pp"], 2)} п.п., '
                     f'медиана ошибки {fnum(row["median_ae_pp"], 2)} п.п.</title></rect>')
        parts.append(svg_text(x0 + w + 6, y + row_h / 2 + 4, fnum(row["mae_pp"], 2)))
    parts.append(svg_text((x0 + x1) / 2, height - 6, "Средняя абсолютная ошибка, п.п.", anchor="middle", cls="v12-muted"))
    parts.append("</svg>")
    return chart("".join(parts), "Ошибка оценки роста расходов по аналогам")


def forest_chart(data: dict, k: int) -> str:
    rows = data["analogue_comparisons"]
    width, row_h, top = CHART_WIDTH, 50, 12
    bottom = top + row_h * len(rows)
    height = bottom + 44
    x0, x1 = 300, 626
    low = min(min(number(r["ci_low"]), number(r["mae_difference_pp"])) for r in rows)
    high = max(max(number(r["ci_high"]), number(r["mae_difference_pp"])) for r in rows)
    pad = (max(high, 0) - min(low, 0)) * 0.08
    xt, xd = ticks(min(low, 0) - pad, max(high, 0) + pad, 4)
    lo_v, hi_v = min(xt[0], min(low, 0) - pad), max(xt[-1], max(high, 0) + pad)

    def X(v):
        return x0 + (x1 - x0) * (v - lo_v) / (hi_v - lo_v)

    parts = [svg_open(width, height, "Парные разности ошибок с 95% интервалами")]
    for v in xt:
        if lo_v - 1e-12 <= v <= hi_v + 1e-12:
            parts.append(f'<line class="v12-grid" x1="{f1(X(v))}" x2="{f1(X(v))}" y1="{top}" y2="{bottom}"/>')
            parts.append(svg_text(X(v), bottom + 18, fnum(v, xd), anchor="middle", cls="v12-muted"))
    parts.append(f'<line class="v12-zero" x1="{f1(X(0))}" x2="{f1(X(0))}" y1="{top - 4}" y2="{bottom}"/>')
    for i, row in enumerate(rows):
        y = top + i * row_h + row_h / 2
        better, baseline = predictor_label(row["better"], k), lower_first(predictor_label(row["baseline"], k))
        parts.append(svg_text(x0 - 12, y - 4, esc(better), anchor="end", cls="v12-strong"))
        parts.append(svg_text(x0 - 12, y + 13, "против: " + esc(baseline), anchor="end", cls="v12-muted"))
        a, b, m = X(number(row["ci_low"])), X(number(row["ci_high"])), X(number(row["mae_difference_pp"]))
        title = (f'{esc(better)} минус {esc(baseline)}: {fnum(row["mae_difference_pp"], 2)} п.п., 95% интервал '
                 f'{finterval(row["ci_low"], row["ci_high"], 2)}')
        parts.append(f'<g><title>{title}</title><line class="v12-ci" x1="{f1(a)}" x2="{f1(b)}" y1="{f1(y)}" y2="{f1(y)}"/>'
                     f'<line class="v12-ci" x1="{f1(a)}" x2="{f1(a)}" y1="{f1(y - 6)}" y2="{f1(y + 6)}"/>'
                     f'<line class="v12-ci" x1="{f1(b)}" x2="{f1(b)}" y1="{f1(y - 6)}" y2="{f1(y + 6)}"/>'
                     f'<circle class="v12-pt-chosen" cx="{f1(m)}" cy="{f1(y)}" r="6"/></g>')
    parts.append(svg_text(X(lo_v) + 2, height - 6, "← ошибка аналогов меньше", cls="v12-muted"))
    parts.append(svg_text(x1, height - 6, "больше →", anchor="end", cls="v12-muted"))
    parts.append("</svg>")
    return chart("".join(parts), "Парные разности ошибок")


def dumbbell(data: dict) -> str:
    rows = sorted(data["type_dynamics"], key=lambda r: int(r["type"]))
    width, row_h, top = CHART_WIDTH, 64, 34
    bottom = top + row_h * len(rows)
    height = bottom + 42
    x0, x1 = 196, 576
    xmax = nice_ceiling(max(number(r["marketplaces_pct_2024"]) for r in rows) * 1.05, 4)
    xt, xd = ticks(0, xmax, 4)

    def X(v):
        return x0 + (x1 - x0) * v / xmax

    parts = [svg_open(width, height, "Медианное отношение маркетплейсов к общему показателю в 2023 и 2024 годах по типам")]
    parts.append(f'<circle cx="{x0 + 6}" cy="15" r="5" class="v12-hollow"/>{svg_text(x0 + 16, 19, "2023", cls="v12-muted")}')
    parts.append(f'<circle cx="{x0 + 72}" cy="15" r="5" class="v12-filled"/>{svg_text(x0 + 82, 19, "2024", cls="v12-muted")}')
    for v in xt:
        parts.append(f'<line class="v12-grid" x1="{f1(X(v))}" x2="{f1(X(v))}" y1="{top}" y2="{bottom}"/>')
        parts.append(svg_text(X(v), bottom + 18, fnum(v, xd) + "%", anchor="middle", cls="v12-muted"))
    for i, row in enumerate(rows):
        t = TYPES[int(row["type"])]
        y = top + i * row_h + row_h / 2
        a, b = X(number(row["marketplaces_pct_2023"])), X(number(row["marketplaces_pct_2024"]))
        ratio = fnum(row["marketplaces_ratio_2024_to_2023"], 2)
        parts.append(wrapped_svg_label(x0 - 12, y + 4, t["short"], x0 - 20, anchor="end"))
        parts.append(f'<g><title>{esc(t["name"])}: {fnum(row["marketplaces_pct_2023"], 1)}% → '
                     f'{fnum(row["marketplaces_pct_2024"], 1)}%, ×{ratio}</title>'
                     f'<line x1="{f1(a)}" x2="{f1(b)}" y1="{f1(y)}" y2="{f1(y)}" stroke="{t["color"]}" stroke-width="4" '
                     f'stroke-linecap="round"/><circle cx="{f1(a)}" cy="{f1(y)}" r="6" class="v12-hollow-type" '
                     f'stroke="{t["color"]}"/><circle cx="{f1(b)}" cy="{f1(y)}" r="6.5" fill="{t["color"]}"/></g>')
        parts.append(svg_text(b + 12, y + 4, f"×{ratio}", cls="v12-strong"))
    parts.append(svg_text((x0 + x1) / 2, height - 6, "Медианное отношение маркетплейсов к общему показателю, %",
                          anchor="middle", cls="v12-muted"))
    parts.append("</svg>")
    return chart("".join(parts), "Маркетплейсы по типам")


def dynamics_table(data: dict) -> str:
    body = []
    for row in sorted(data["type_dynamics"], key=lambda r: int(r["type"])):
        body.append(
            f'<tr><th scope="row">{type_label(TYPES[int(row["type"])])}</th>'
            f'<td class="num">{fnum(row["marketplaces_pct_2023"], 1)} → {fnum(row["marketplaces_pct_2024"], 1)}</td>'
            f'<td class="num">{fnum(row["marketplaces_change_pp_median"], 2, sign=True)} '
            f'<span class="muted">{finterval(row["marketplaces_change_ci_low"], row["marketplaces_change_ci_high"], 2)}</span></td>'
            f'<td class="num">{fnum(row["nominal_growth_mean_pct"], 1, sign=True)} '
            f'<span class="muted">{finterval(row["nominal_growth_ci_low_pct"], row["nominal_growth_ci_high_pct"], 1)}</span></td>'
            f'<td class="num">{fnum(row["relative_level_change_log"], 3, sign=True)} '
            f'<span class="muted">{finterval(row["relative_level_change_ci_low"], row["relative_level_change_ci_high"], 3)}</span></td>'
            f'<td class="num">{fpct(row["left_type_2024_adjusted_share"])} / {fpct(row["left_type_2024_absolute_share"])}</td></tr>')
    head = ('<thead><tr><th scope="col">Тип 2023</th><th scope="col">Маркетплейсы: медиана типа, % (2023 → 2024)</th>'
            '<th scope="col">Медиана изменения по МО, п.п.</th><th scope="col">Средний номинальный рост расходов, %</th>'
            '<th scope="col">Среднее изменение относительного уровня, лог. ед.</th>'
            '<th scope="col">Сменили тип в 2024: с поправкой / без</th></tr></thead>')
    caption = ("Группы — типы 2023 года; 95% интервалы — бутстреп целых регионов. Маркетплейсы — отношение к общему "
               "показателю расходов; медиана изменений по МО не равна разности медиан типа. Относительный уровень — "
               "логарифм расходов минус медиана всех МО того же месяца.")
    return table_block(caption, head + f'<tbody>{"".join(body)}</tbody>', "Изменения по типам")


def convergence_text(data: dict) -> str:
    rows = sorted(data["type_dynamics"], key=lambda r: int(r["type"]))
    gap = data["summary"]["practical"]["gap"]
    mk23 = [number(r["marketplaces_pct_2023"]) for r in rows]
    mk24 = [number(r["marketplaces_pct_2024"]) for r in rows]
    ratio = [number(r["marketplaces_ratio_2024_to_2023"]) for r in rows]
    fastest = max(range(len(rows)), key=lambda i: ratio[i])
    lowest = min(range(len(rows)), key=lambda i: mk23[i])
    text = ("Медианное отношение маркетплейсов к общему показателю выросло во всех типах"
            if all(b > a for a, b in zip(mk23, mk24)) else "Отношение маркетплейсов к общему показателю менялось по-разному")
    text += (f'; сильнее всего относительно — в типе {quoted(TYPES[int(rows[fastest]["type"])])} (×{fnum(ratio[fastest], 2)})'
             + (", где оно было самым низким" if fastest == lowest else "") + ". ")
    text += (f'Отношение наибольшего значения среди типов к наименьшему: {fnum(max(mk23) / min(mk23), 2)} в 2023 году и '
             f'{fnum(max(mk24) / min(mk24), 2)} в 2024; разница в процентных пунктах: {fnum(max(mk23) - min(mk23), 1)} '
             f'и {fnum(max(mk24) - min(mk24), 1)}.')
    metro, periphery = TYPES[TYPE_ORDER.index("metro")], TYPES[TYPE_ORDER.index("periphery")]
    diff, low, high = (number(gap["metro_minus_periphery_relative_level_change_log"]), number(gap["ci_low"]),
                       number(gap["ci_high"]))
    if low > 0:
        verdict = "разрыв в уровне расходов немного вырос"
    elif high < 0:
        verdict = "разрыв в уровне расходов немного сократился"
    else:
        verdict = "изменение разрыва в уровне расходов не отличается от нуля"
    text2 = (f'Уровень расходов: медианный уровень типа {quoted(metro)} выше, чем у типа {quoted(periphery)}, в '
             f'{fnum(gap["level_ratio_2023"], 2)} раза в 2023 году и в {fnum(gap["level_ratio_2024"], 2)} раза в 2024. '
             f'Относительный уровень первого изменился на {fnum(diff, 3, sign=True)} лог. ед. больше, чем у второго, '
             f'95% интервал {finterval(low, high, 3)}: {verdict}.')
    tested = data["summary"]["practical"].get("marketplace_gap_pp")
    gap_text = ""
    if tested:
        low, high = tested["conditional_region_ci_95_change_pp"]
        a, b = TYPES[TYPE_ORDER.index("periphery")], TYPES[TYPE_ORDER.index("remote")]
        gap_text = (f'<p>Разрыв {quoted(a)} минус {quoted(b)} при фиксированном составе 2023 года: '
                    f'{fnum(tested["gap_2023_pp"], 2)} → {fnum(tested["gap_2024_pp"], 2)} п.п.; '
                    f'изменение {fnum(tested["change_pp"], 2, sign=True)} п.п., '
                    f'95% региональный интервал {finterval(low, high, 2)}. '
                    + ('Сокращение разрыва этим интервалом не подтверждено.' if low <= 0 <= high else
                       'Интервал условен на фиксированных группах и наблюдаемой панели.') + '</p>')
    return f'<p>{text}</p><p>{text2}</p>{gap_text}'


def monitoring(data: dict) -> str:
    current = ROOT / 'reports/temporal-v4/real-correction.json'
    if current.is_file():
        from scripts.current_evidence_web import render_monitoring
        return render_monitoring(current)
    rows = data["persistent_changes"]
    window = data["summary"]["tracking"]["persistence"]["persistent_change_window_months"]
    entities = {row["entity_id"]: row for row in data["labels"]}
    directions = {}
    for row in rows:
        key = (int(row["type_2023"]), int(row["type_2024_h2"]))
        directions[key] = directions.get(key, 0) + 1
    direction_rows = "".join(
        f'<tr><td>{type_label(TYPES[a])}</td><td>{type_label(TYPES[b])}</td><td class="num">{fnum(v)}</td></tr>'
        for (a, b), v in sorted(directions.items(), key=lambda item: (-item[1], item[0])))
    populous = sorted(rows, key=lambda r: (number(r["population"]) is None, -(number(r["population"]) or 0), r["entity_id"]))[:10]
    examples = []
    for row in populous:
        entity = entities.get(row["entity_id"], {"entity_id": row["entity_id"], "name": row["name"], "region": row["region"]})
        population = fnum(row["population"]) if number(row["population"]) is not None else "нет данных"
        examples.append(f'<tr><th scope="row">{entity_button(entity, with_region=False)}</th><td>{esc(row["region"])}</td>'
                        f'<td class="v12-nowrap">{type_label(TYPES[int(row["type_2023"])])} → '
                        f'{type_label(TYPES[int(row["type_2024_h2"])])}</td><td class="num">{population}</td></tr>')
    lead = (f'У {count(len(rows), MO)} новый тип держится все последние {count(window, MONTH_FORMS)} 2024 года. '
            'Эти территории стоит изучить подробнее. Чтобы проверить структурный сдвиг, нужно сверить границы, население '
            'и охват безналичных платежей. Нажмите на название, чтобы открыть территорию.')
    directions_table = table_block("", '<thead><tr><th scope="col">Тип 2023</th><th scope="col">Новый тип</th>'
                                       f'<th scope="col">МО</th></tr></thead><tbody>{direction_rows}</tbody>',
                                   "Устойчивые смены по направлениям")
    examples_table = table_block(f"{fnum(len(examples))} крупнейших по населению",
                                 '<thead><tr><th scope="col">Территория</th><th scope="col">Регион</th>'
                                 '<th scope="col">Смена типа</th><th scope="col">Население</th></tr></thead>'
                                 f'<tbody>{"".join(examples)}</tbody>', "Крупнейшие территории с устойчивой сменой типа")
    return (f'<h3 class="v12-h3">Кого наблюдать</h3><div class="v12-split"><div class="v12-text"><p>{lead}</p></div>'
            f'<div>{directions_table}</div></div>{examples_table}')


def section_practice(data: dict, ctx: dict) -> str:
    k = int(data["config"]["practical"]["analogues"])
    analogues = {row["predictor"]: row for row in data["analogues"]}
    comparisons = data["analogue_comparisons"]
    net, nat = analogues.get("network_analogues"), analogues.get("national_median")
    text = (f'Для каждой территории рост расходов 2024 года к 2023 году (средние за год, 100 × (2024 / 2023 − 1)) оценивается медианой '
            f'роста её {fnum(k)} аналогов. Аналоги выбираются по профилю 2023 года; рост 2024 года в выбор не входит. '
            f'Сравнение на {count(data["analogues"][0]["n"], MO)}, где определены все ориентиры.')
    if net and nat:
        text += (f' Средняя абсолютная ошибка {fnum(k)} аналогов по расходному профилю составляет {fnum(net["mae_pp"], 2)} п.п., '
                 f'у медианы всех МО {fnum(nat["mae_pp"], 2)} п.п.')
    text += (" Во всех парных сравнениях 95% интервал разности ниже нуля: аналоги по профилю точнее этих ориентиров."
             if all(number(r["ci_high"]) < 0 for r in comparisons) else
             " Не все 95% интервалы разности ниже нуля; преимущество аналогов подтверждается не везде.")
    draws = data["provenance"].get("draws", {}).get("bootstrap")
    method = ("Интервалы — бутстреп целых регионов" + (f" ({count(draws, DRAW_FORMS)})" if draws else "")
              + ". Проверка использует уже наблюдённые годы. Точность прогноза будущего года ещё предстоит оценить.")
    limitations = data["summary"]["practical"].get("analogue_limitations", {})
    method += " " + " ".join(esc(limitations[key]) for key in ("random_control", "selection") if key in limitations)
    command = esc(data["provenance"].get("command", "python -m scripts.run_v12"))
    source = (f'Числа разделов собраны из <code>reports/v1.2</code> (команда <code>{command}</code>) '
              'скриптом <code>scripts/build_v12_web.py</code>.')
    report = doc_link("NETWORK_TYPOLOGY.md", "методика сетевых групп и проверок")
    diagnostic = data["summary"]["practical"].get("analogues_diagnostic", [])
    diagnostic_html = ""
    if diagnostic:
        body = "".join(f'<tr><th scope="row">{esc(predictor_label(row["predictor"], k))}</th>'
                       f'<td class="num">{fnum(row["n"])}</td><td class="num">{fnum(row["mae_pp"], 2)}</td></tr>'
                       for row in diagnostic)
        head = '<thead><tr><th scope="col">Ориентир</th><th scope="col">МО</th><th scope="col">MAE, п.п.</th></tr></thead>'
        diagnostic_html = ('<details class="comparison-details"><summary>Дополнительные контроли на общей выборке</summary>'
                           '<div class="details-body">' + table_block(
                               'Локальная сеть требует не менее трёх других МО в регионе. Все ориентиры здесь пересчитаны на общей допустимой выборке.',
                               head + '<tbody>' + body + '</tbody>', 'Диагностика аналогов') + '</div></details>')
    return (f'<section class="card v12-card" id="v12-practice" aria-labelledby="v12-practice-title">'
            + section_head("v12-practice-title", "Практические выводы",
                           "Три проверки для работы с типами: подбор аналогов, сближение структуры расходов и список "
                           "территорий для наблюдения. Все проверки ретроспективные.", "аналоги · маркетплейсы · мониторинг")
            + f'<h3 class="v12-h3">Точность аналогов</h3><div class="v12-text"><p>{text}</p></div>'
            f'<div class="v12-split"><div>{mae_chart(data, k)}</div><div>{forest_chart(data, k)}</div></div>'
            f'<p class="note small">{method}</p>{diagnostic_html}<h3 class="v12-h3">Маркетплейсы и уровень расходов</h3>'
            f'<div class="v12-split"><div>{dumbbell(data)}</div><div class="v12-text">{convergence_text(data)}</div></div>'
            f'{dynamics_table(data)}{monitoring(data)}'
            f'<p class="note small v12-source">{source}{" Методика и ограничения — " + report + "." if report else ""}</p>'
            '</section>')


# ------------------------------------------------------------------ assembly
def render_sections(directory: Path) -> str:
    data = load(directory)
    ctx = context(data)
    sections = [section_types(data, ctx), section_network(data, ctx), section_methods(data, ctx),
                section_dynamics(data, ctx), section_practice(data, ctx)]
    current = ROOT / 'reports/conditional-v4/atlas.json'
    from scripts.current_evidence_web import render_economics
    current_sections = render_economics(current) if current.is_file() else ''
    return (current_sections + '<!-- Generated by scripts/build_v12_web.py from reports/v1.2; do not edit by hand. -->\n'
            '<div class="v12-sections" id="v12-sections">\n' + "\n".join(sections) + "\n</div>\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "reports/v1.2")
    parser.add_argument("--output", type=Path, default=ROOT / "web/v12_sections.html")
    args = parser.parse_args()
    content = render_sections(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8", newline="\n")
    print(f"Created {args.output} ({len(content.encode('utf-8')):,} bytes)")


if __name__ == "__main__":
    main()
