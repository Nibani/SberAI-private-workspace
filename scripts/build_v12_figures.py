"""Figures of version 1.2, read from the files of ``scripts.run_v12``.

    python -m scripts.build_v12_figures
    python -m scripts.build_v12_figures --input reports/v1.2 --output-dir docs/images/v12

Nothing is fitted here. Every number on a figure is read from the result files
(CSV and JSON) or from the published municipal map, so the figures change only
when the analysis changes.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch, Patch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import Formatter, MaxNLocator  # noqa: E402
from matplotlib.transforms import Bbox, ScaledTranslation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sbercluster.attributed import ICVI  # noqa: E402
from sbercluster.panel import CATEGORIES, CATEGORY_LABELS_RU  # noqa: E402
from sbercluster.typology import TYPE_COLORS, TYPE_NAMES, TYPE_ORDER, TYPE_SHORT  # noqa: E402

# ------------------------------------------------------------------ style
INK = "#1f2328"          # primary text
INK2 = "#4f5862"         # secondary text, axis labels
INK3 = "#7d8590"         # notes, muted text
GRID = "#e5e8eb"         # hairline grid
SPINE = "#c3c9d0"
MUTED = "#a3aab3"        # context marks
MUTED_LIGHT = "#d0d5db"
EMPHASIS = "#2b313a"     # the one mark a chart is about (chosen rule, method, predictor)
BAND = "#eef0f3"         # highlight band behind a chosen row or K
PANEL = "#f5f6f8"        # light panel background
TYPE_HEX = [TYPE_COLORS[key] for key in TYPE_ORDER]
DPI = 180
PNG_WIDTH = 10.0         # inches: 1 800 px, shown at ~900 px on GitHub
PAGE = (13.333, 7.5)     # 16:9 slide, inches
FONT = "DejaVu Sans"

RC = {
    "font.family": FONT,
    "font.size": 10,
    "text.color": INK,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "axes.labelsize": 10,
    "axes.labelcolor": INK2,
    "axes.edgecolor": SPINE,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "axes.unicode_minus": True,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.major.pad": 5,
    "ytick.major.pad": 5,
    "legend.frameon": False,
    "legend.fontsize": 9.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
}

NBSP = " "
MINUS = "−"
MONTHS_RU = ("янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек")

RULE_SHORT = {
    "euclid_structure": "Евклид: только структура",
    "euclid_profile": "Евклид: структура + уровень",
    "cosine_spending": "Косинус: расходы в рублях",
    "corr_total": "Корреляция рядов уровня",
    "corr_multivariate": "Корреляция шести рядов",
    "lagged_corr_total": "Лаговая корреляция",
    "dtw_total": "DTW рядов уровня",
    "geographic": "География",
    "road": "Дороги",
}
SERIES_RULES = ("corr_total", "corr_multivariate", "lagged_corr_total", "dtw_total")
METHOD_SHORT = {"kefrin": "KEFRiN-подобный K-means"}
CATEGORY_SHORT = {"health": "здоровье", "marketplaces": "маркетплейсы", "horeca": "общепит", "food": "продовольствие",
                  "transport": "транспорт"}
FAMILY_LABEL = {"признаки": "Только признаки", "сеть": "Только сеть", "признаки + сеть": "Признаки + сеть"}
INDICATOR_LABELS = {
    "log_wage": "Зарплата",
    "agri_share": "Занятые в сельском хозяйстве",
    "manuf_share": "Занятые в обработке",
    "mining_share": "Занятые в добыче",
    "public_share": "Занятые в бюджетном секторе",
    "log_pop": "Население",
    "log_jobs_per_res": "Рабочие места на жителя",
    "market_access": "Доступность рынков",
}
INDICATOR_GENITIVE = {
    "log_wage": "зарплаты", "agri_share": "доли сельского хозяйства", "manuf_share": "доли обработки",
    "mining_share": "доли добычи", "public_share": "доли бюджетного сектора", "log_pop": "населения",
    "log_jobs_per_res": "рабочих мест на жителя", "market_access": "доступности рынков",
}
PREDICTOR_LABELS = {
    "network_analogues": "Сетевые аналоги",
    "network_analogues_same_region": "Сетевые аналоги в своём регионе",
    "random_same_region": "Случайные МО своего региона",
    "region_median_leave_one_out": "Медиана своего региона",
    "random_same_type": "Случайные МО своего типа",
    "national_median": "Медиана по России",
}
PREDICTOR_GENITIVE = {
    "network_analogues": "сетевых аналогов",
    "network_analogues_same_region": "сетевых аналогов в своём регионе",
    "random_same_region": "случайных МО своего региона",
    "region_median_leave_one_out": "медианы своего региона",
    "random_same_type": "случайных МО своего типа",
    "national_median": "медианы по России",
}
PROFILE_PANELS = (
    # indicator in type_intervals.csv, title, scale, digits, coverage column in types.csv
    ("spending_rub", "Расходы на жителя,\nтыс. ₽ в месяц", 1e-3, 1, None),
    ("food_pct", "Продовольствие,\n% расходов", 1.0, 1, None),
    ("horeca_pct", "Общепит,\n% расходов", 1.0, 1, None),
    ("marketplaces_pct", "Маркетплейсы,\n% расходов", 1.0, 1, None),
    ("wage_rub", "Зарплата (Росстат),\nтыс. ₽ в месяц", 1e-3, 1, "log_wage_coverage"),
    ("agri_share_pct", "Сельское хозяйство,\n% занятых", 1.0, 1, "agri_share_coverage"),
    ("mining_share_pct", "Добыча полезных\nископаемых, % занятых", 1.0, 1, "mining_share_coverage"),
    ("market_access", "Доступность рынков,\nиндекс СберИндекса", 1.0, 0, "market_access_coverage"),
)
REGION_SHORT = {
    "Республика Саха (Якутия)": "Якутия", "Удмуртская Республика": "Удмуртия", "Чувашская Республика": "Чувашия",
    "Чеченская Республика": "Чечня", "Кабардино-Балкарская Республика": "Кабардино-Балкария",
    "Карачаево-Черкесская Республика": "Карачаево-Черкесия", "Ямало-Ненецкий автономный округ": "ЯНАО",
}
INSET_REGIONS = ("Москва", "Санкт-Петербург")
COVERAGE_NOTE = 0.9      # coverage below this share is printed next to a Rosstat median


# ------------------------------------------------------------------ formatting
def fmt_num(value, digits: int = 1, signed: bool = False) -> str:
    """Russian number format: decimal comma, no-break space between thousands, true minus sign."""
    if value is None or not np.isfinite(float(value)):
        return "н/д"
    value = float(value)
    text = f"{abs(value):,.{digits}f}".replace(",", NBSP).replace(".", ",")
    if not text.strip("0," + NBSP):  # rounds to zero: no sign
        return text
    if value < 0:
        return MINUS + text
    return "+" + text if signed else text


def fmt_int(value) -> str:
    return fmt_num(value, 0)


def fmt_pct(share, digits: int = 0, signed: bool = False) -> str:
    """A fraction (0.48) as a percentage (48%)."""
    return fmt_num(100 * float(share), digits, signed) + "%"


def fmt_pp(value, digits: int = 1, signed: bool = True) -> str:
    return fmt_num(value, digits, signed) + NBSP + "п.п."


def fmt_ci(low, high, digits: int = 1) -> str:
    return f"[{fmt_num(low, digits)}; {fmt_num(high, digits)}]"


def eq(name, value) -> str:
    """«K = 4» that never breaks across lines."""
    return f"{name}{NBSP}={NBSP}{value}"


def plural(n, forms: tuple[str, str, str]) -> str:
    """Russian plural: (1 регион, 2 региона, 5 регионов)."""
    n = abs(int(n)) % 100
    if 11 <= n <= 14:
        return forms[2]
    return forms[0] if n % 10 == 1 else forms[1] if 2 <= n % 10 <= 4 else forms[2]


def count_word(n, forms) -> str:
    return f"{fmt_int(n)}{NBSP}{plural(n, forms)}"


def sentence(text: str) -> str:
    """End a sentence with exactly one period."""
    text = text.rstrip()
    return text if text.endswith((".", "!", "?")) else text + "."


def lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def upper_first(text: str) -> str:
    return text[:1].upper() + text[1:]


def mid_sentence(name: str) -> str:
    """A name inside a sentence: «Крупные города» -> «крупные города», but a name that opens
    with a proper noun phrase («Дальний Восток и Север») or a Latin name (Leiden) keeps its capitals."""
    words = name.split()
    if (len(words) > 1 and words[1][:1].isupper()) or name[:1].isascii():
        return name
    return lower_first(name)


def month_label(period: str, with_year: bool = False) -> str:
    year, month = period[:4], int(period[5:7])
    return f"{MONTHS_RU[month - 1]}\n{year}" if with_year else MONTHS_RU[month - 1]


def short_region(region: str) -> str:
    if region in REGION_SHORT:
        return REGION_SHORT[region]
    region = region.replace(" область", " обл.").replace(" автономный округ", " АО")
    return region.replace("Республика ", "").replace(" Республика", "")


def type_short(index: int) -> str:
    return TYPE_SHORT[TYPE_ORDER[int(index)]]


def type_mid(index: int) -> str:
    return mid_sentence(type_short(index))


def flow(i: int, j: int) -> str:
    return f"{type_mid(i)} → {type_mid(j)}"


def tint(color, amount: float):
    """Blend a colour with white; amount 0 keeps it, 1 gives white."""
    r, g, b = to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


class CommaFormatter(Formatter):
    """Tick labels with a decimal comma and the fewest digits that show every tick exactly."""

    def __init__(self, scale: float = 1.0, suffix: str = "", digits: int | None = None):
        self.scale, self.suffix, self.digits = scale, suffix, digits
        self.locs = []

    def __call__(self, x, pos=None):
        digits = self.digits
        if digits is None:
            values = [v * self.scale for v in (self.locs if len(self.locs) else [x])]
            digits = next((d for d in range(5) if all(abs(round(v, d) - v) < 1e-9 * max(1.0, abs(v))
                                                      for v in values)), 4)
        return fmt_num(x * self.scale, digits) + self.suffix


def comma_axis(ax, axis: str = "y", **kwargs) -> None:
    for name in (("x", "y") if axis == "both" else (axis,)):
        getattr(ax, f"{name}axis").set_major_formatter(CommaFormatter(**kwargs))


def grid(ax, axis: str = "y") -> None:
    ax.grid(axis=axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ SVG map geometry
_ALLOWED = set("MLZ0123456789.,- \t\r\n")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def parse_svg_path(d: str) -> list[np.ndarray]:
    """Closed rings (m, 2) of an SVG path made only of absolute M, L and Z commands.

    Every ring ends at its first point; SVG coordinates are returned unchanged
    (y grows downward).
    """
    unsupported = set(d) - _ALLOWED
    if unsupported:
        raise ValueError(f"Only absolute M/L/Z commands are supported, found {sorted(unsupported)}")
    rings = []
    for part in re.split(r"(?=M)", d.strip()):
        values = np.array(_NUMBER.findall(part), dtype=float)
        if values.size % 2:
            raise ValueError("Odd number of coordinates in a subpath")
        points = values.reshape(-1, 2)
        if len(points) < 3:
            continue
        if not np.array_equal(points[0], points[-1]):
            points = np.vstack([points, points[:1]])
        rings.append(points)
    return rings


def ring_area(ring: np.ndarray) -> float:
    """Signed shoelace area of a closed ring (positive = counter-clockwise with y up)."""
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


def outline_path(rings: list[np.ndarray], height: float) -> MPath:
    """One matplotlib path per municipality with y pointing up.

    Rings nested inside an odd number of other rings are holes; they are wound
    against the outer rings so that any fill rule leaves them empty.
    """
    flipped = [np.column_stack([r[:, 0], height - r[:, 1]]) for r in rings]
    shapes = [MPath(r) for r in flipped]
    vertices, codes = [], []
    for i, ring in enumerate(flipped):
        probe = ring[: min(len(ring) - 1, 5)]
        depth = sum(shapes[j].contains_points(probe).mean() > 0.5 for j in range(len(flipped)) if j != i)
        if (ring_area(ring) > 0) != (depth % 2 == 0):
            ring = ring[::-1]
        code = np.full(len(ring), MPath.LINETO, dtype=np.uint8)
        code[0], code[-1] = MPath.MOVETO, MPath.CLOSEPOLY
        vertices.append(ring)
        codes.append(code)
    return MPath(np.concatenate(vertices), np.concatenate(codes))


def load_outlines(map_path: Path, ids) -> dict[str, MPath]:
    geometry = json.loads(Path(map_path).read_text(encoding="utf-8"))
    height = float(geometry["viewBox"][3])
    paths = {p["id"]: p["d"] for p in geometry["paths"]}
    missing = [i for i in ids if i not in paths]
    if missing:
        raise ValueError(f"{len(missing)} municipalities have no outline in {map_path}, e.g. {missing[:3]}")
    return {i: outline_path(parse_svg_path(paths[i]), height) for i in ids}


# ------------------------------------------------------------------ inputs
@dataclass
class Results:
    labels: pd.DataFrame
    types: pd.DataFrame
    intervals: pd.DataFrame
    networks: pd.DataFrame
    k_sensitivity: pd.DataFrame
    k_selection: pd.DataFrame
    methods: pd.DataFrame
    monthly: pd.DataFrame
    events: pd.DataFrame
    dynamics: pd.DataFrame
    analogues: pd.DataFrame
    comparisons: pd.DataFrame
    persistent: pd.DataFrame
    loro: pd.DataFrame
    summary: dict
    config: dict

    @property
    def n(self) -> int:
        return len(self.labels)

    @property
    def months(self) -> int:
        return len(self.monthly)

    @property
    def years(self) -> tuple[str, str]:
        periods = self.monthly["period"].astype(str)
        return periods.iloc[0][:4], periods.iloc[-1][:4]

    @property
    def regions(self) -> int:
        return int(self.summary["panel"]["regions"])

    @property
    def k(self) -> int:
        return int(self.summary["k_selection"]["chosen_k"])

    @property
    def n_indicators(self) -> int:
        return len(self.loro)

    @property
    def version(self) -> str:
        return str(self.summary.get("version", self.config.get("version", "")))

    @property
    def rosstat_year(self) -> str:
        """Vintage of the Rosstat cohort, read from its file name (cohort-2023.csv)."""
        path = self.summary.get("inputs", {}).get("rosstat_cohort", {}).get("path", "")
        match = re.search(r"(\d{4})(?=\.csv)", path)
        return match.group(1) if match else self.years[0]

    def months_in(self, year: str) -> int:
        return int(self.monthly["period"].astype(str).str.startswith(year).sum())

    def interval(self, indicator: str) -> pd.DataFrame:
        return self.intervals[self.intervals["indicator"] == indicator].set_index("type").sort_index()

    def type_row(self, c: int):
        return self.types.loc[self.types["type"] == c].iloc[0]

    def chosen_rule(self):
        return self.networks.loc[self.networks["rule"] == self.summary["network"]["chosen_rule"]].iloc[0]

    def chosen_method(self):
        return self.methods.loc[self.methods["method"] == self.summary["methods"]["chosen"]].iloc[0]

    def mae(self, predictor: str) -> float:
        return float(self.analogues.loc[self.analogues["predictor"] == predictor, "mae_pp"].iloc[0])

    def comparison(self, better: str, baseline: str):
        rows = self.comparisons[(self.comparisons["better"] == better) & (self.comparisons["baseline"] == baseline)]
        return rows.iloc[0] if len(rows) else None


def load_results(directory: Path) -> Results:
    directory = Path(directory)

    def read(name, **kwargs):
        path = directory / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run `python -m scripts.run_v12` first")
        return pd.read_csv(path, **kwargs) if name.endswith(".csv") else json.loads(path.read_text(encoding="utf-8"))

    provenance = directory / "provenance.json"
    if provenance.exists():
        config = json.loads(provenance.read_text(encoding="utf-8"))["config"]
    else:
        print(f"WARNING: {provenance} is missing (unfinished run?); using configs/v12.json", file=sys.stderr)
        config = json.loads((ROOT / "configs/v12.json").read_text(encoding="utf-8"))
    return Results(
        labels=read("labels.csv", dtype={"monthly_types": str}), types=read("types.csv"),
        intervals=read("type_intervals.csv"), networks=read("networks.csv"),
        k_sensitivity=read("network_k_sensitivity.csv"), k_selection=read("k_selection.csv"),
        methods=read("methods.csv"), monthly=read("monthly_network.csv"), events=read("reclustering_events.csv"),
        dynamics=read("type_dynamics.csv"), analogues=read("analogues.csv"),
        comparisons=read("analogue_comparisons.csv"), persistent=read("persistent_changes.csv"),
        loro=read("leave_one_region_out.csv"), summary=read("summary.json"), config=config)


# ------------------------------------------------------------------ layout helpers
def text_width_in(fig, text: str, size: float, weight: str = "normal", family: str = FONT) -> float:
    """Rendered width of the widest line of ``text`` in inches."""
    renderer = fig.canvas.get_renderer()
    prop = FontProperties(family=family, size=size, weight=weight)
    widths = [renderer.get_text_width_height_descent(line, prop, ismath=False)[0] for line in str(text).split("\n")]
    return max(widths) / fig.dpi


def labels_width_in(fig, labels, size: float, weight: str = "normal") -> float:
    return max(text_width_in(fig, label, size, weight) for label in labels)


def wrap(fig, text: str, width_in: float, size: float, weight: str = "normal") -> str:
    """Greedy word wrap measured with the real font; no-break spaces never split."""
    lines, current = [], ""
    for word in text.split(" "):
        trial = f"{current} {word}" if current else word
        if current and text_width_in(fig, trial, size, weight) > 0.98 * width_in:
            lines.append(current)
            current = word
        else:
            current = trial
    lines.append(current)
    return "\n".join(lines)


def spec_inches(fig, spec) -> tuple[float, float, float, float]:
    """Left, bottom, width, height of a SubplotSpec in inches."""
    box = spec.get_position(fig)
    fw, fh = fig.get_size_inches()
    return box.x0 * fw, box.y0 * fh, box.width * fw, box.height * fh


def add_axes_in(fig, left: float, bottom: float, width: float, height: float):
    fw, fh = fig.get_size_inches()
    return fig.add_axes([left / fw, bottom / fh, width / fw, height / fh])


def place_labels(ax, xy, texts, size=9.5, weights=None, radius_pt=6.5, margin_pt=2.0, avoid=()):
    """Put each label next to its point without overlapping labels, other points or the frame.

    Near slots come first; a far slot gets a thin leader line. A label never sits
    closer to another point than to its own, so it cannot be read as the neighbour's.
    """
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    px = fig.dpi / 72
    points = ax.transData.transform(np.asarray(xy, dtype=float))
    obstacles = [Bbox.from_extents(x - radius_pt * px, y - radius_pt * px, x + radius_pt * px, y + radius_pt * px)
                 for x, y in points]
    frame = ax.get_window_extent(renderer)
    near = [(9, 0, "left", "center"), (-9, 0, "right", "center"), (0, 9, "center", "bottom"),
            (0, -9, "center", "top"), (7, 7, "left", "bottom"), (-7, 7, "right", "bottom"),
            (7, -7, "left", "top"), (-7, -7, "right", "top")]
    far = [(16, 18, "left", "bottom"), (16, -18, "left", "top"), (-16, 18, "right", "bottom"),
           (-16, -18, "right", "top"), (0, 26, "center", "bottom"), (0, -26, "center", "top"),
           (30, 0, "left", "center"), (-30, 0, "right", "center"), (24, 32, "left", "bottom"),
           (24, -32, "left", "top")]
    placed = list(avoid)

    def distance(box, point):
        dx = max(box.x0 - point[0], 0, point[0] - box.x1)
        dy = max(box.y0 - point[1], 0, point[1] - box.y1)
        return np.hypot(dx, dy)

    def crosses_point(start, end, k):
        """Does the leader from point k to its label pass near another point?"""
        seg = end - start
        for i, p in enumerate(points):
            if i == k:
                continue
            t = np.clip(np.dot(p - start, seg) / max(np.dot(seg, seg), 1e-9), 0, 1)
            if np.hypot(*(start + t * seg - p)) < radius_pt * px:
                return True
        return False

    relpos = {"left": 0.0, "center": 0.5, "right": 1.0, "bottom": 0.0, "top": 1.0}
    for k, ((x, y), text) in enumerate(zip(xy, texts)):
        style = {"fontsize": size, "color": INK, "weight": (weights or ["normal"] * len(texts))[k]}
        done = False
        for leader, (dx, dy, ha, va) in [(False, o) for o in near] + [(True, o) for o in far]:
            if leader and crosses_point(points[k], points[k] + np.array([dx, dy]) * px, k):
                continue
            arrow = {"arrowstyle": "-", "color": INK3, "linewidth": 0.7, "shrinkA": 1, "shrinkB": 4,
                     "relpos": (relpos[ha], relpos[va])} if leader else None
            note = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                               arrowprops=arrow, **style)
            note.update_positions(renderer)
            box = Text.get_window_extent(note, renderer)  # the text only, without its leader line
            box = Bbox.from_extents(box.x0 - margin_pt * px, box.y0 - margin_pt * px,
                                    box.x1 + margin_pt * px, box.y1 + margin_pt * px)
            inside = frame.x0 <= box.x0 and box.x1 <= frame.x1 and frame.y0 <= box.y0 and box.y1 <= frame.y1
            clash = any(box.overlaps(o) for i, o in enumerate(obstacles) if i != k) or \
                any(box.overlaps(p) for p in placed)
            own = distance(box, points[k])
            closer = any(distance(box, p) < own for i, p in enumerate(points) if i != k)
            if inside and not clash and not (closer and not leader):
                placed.append(box)
                done = True
                break
            note.remove()
        if not done:  # no free slot: right side with a leader
            note = ax.annotate(text, (x, y), xytext=(12, 0), textcoords="offset points", ha="left", va="center",
                               arrowprops={"arrowstyle": "-", "color": INK3, "linewidth": 0.7}, **style)
            note.update_positions(renderer)
            placed.append(Text.get_window_extent(note, renderer))


def png_figure(height: float, title: str, subtitle: str | None = None, note: str | None = None,
               width: float = PNG_WIDTH):
    """Figure with a title block and a footnote; returns (figure, top, bottom) margins in inches."""
    fig = plt.figure(figsize=(width, height))

    def anchored(x_in, y_in, from_top):
        """Text position in inches from the top-left or bottom-left corner; survives a resize."""
        return fig.transFigure + ScaledTranslation(x_in, -y_in if from_top else y_in, fig.dpi_scale_trans)

    fig.text(0, 1, title, fontsize=15, weight="bold", va="top", transform=anchored(0.25, 0.2, True))
    top = 0.62
    if subtitle:
        wrapped = wrap(fig, subtitle, width - 0.5, 10.5)
        fig.text(0, 1, wrapped, fontsize=10.5, color=INK2, va="top", linespacing=1.3,
                 transform=anchored(0.25, 0.58, True))
        top = 0.58 + 0.2 * (wrapped.count("\n") + 1) + 0.2
    bottom = 0.15
    if note:
        wrapped = wrap(fig, note, width - 0.5, 9)
        fig.text(0, 0, wrapped, fontsize=9, color=INK3, va="bottom", linespacing=1.3,
                 transform=anchored(0.25, 0.12, False))
        bottom = 0.12 + 0.17 * (wrapped.count("\n") + 1) + 0.15
    return fig, top, bottom


def region_spec(fig, left: float, right: float, top: float, bottom: float):
    """A single SubplotSpec bounded by margins in inches."""
    w, h = fig.get_size_inches()
    return fig.add_gridspec(1, 1, left=left / w, right=1 - right / w, top=1 - top / h, bottom=bottom / h)[0]


def save_png(fig, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.15, metadata={"Software": None})
    plt.close(fig)


# ------------------------------------------------------------------ chart: map of types
MAP_COLUMN = 0.24  # share of the width taken by the Moscow / St Petersburg insets
MAP_GAP = 0.15


def map_frame(paths) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Per-path extents and the padded frame (x0, y0, x1, y1) of the whole map."""
    extents = np.array([p.get_extents().extents for p in paths])
    pad = 0.005 * (extents[:, 2].max() - extents[:, 0].min())
    return extents, (extents[:, 0].min() - pad, extents[:, 1].min() - pad,
                     extents[:, 2].max() + pad, extents[:, 3].max() + pad)


def legend_height(size: float) -> float:
    return 2 * size * 1.75 / 72 + 0.15


def map_height(width: float, aspect: float, insets: bool = True, legend: bool = True, size: float = 10) -> float:
    """Height in inches that a map of this width needs, legend included."""
    column = MAP_COLUMN * width if insets else 0.0
    return (width - column - (MAP_GAP if insets else 0.0)) * aspect + (legend_height(size) if legend else 0.0)


def draw_types_map(fig, spec, res: Results, outlines: dict, insets: bool = True, legend: bool = True,
                   size: float = 10, rasterized: bool = False):
    """Municipal polygons coloured by type, with Moscow and St Petersburg enlarged on the left."""
    x0_in, y0_in, w_in, h_in = spec_inches(fig, spec)
    ids = res.labels["entity_id"].tolist()
    types = res.labels["type_2023"].to_numpy(int)
    regions = res.labels["region"].to_numpy()
    paths = [outlines[i] for i in ids]
    colors = [TYPE_HEX[t] for t in types]
    extents, (mx0, my0, mx1, my1) = map_frame(paths)
    aspect = (my1 - my0) / (mx1 - mx0)

    boxes = []
    if insets:
        for region in INSET_REGIONS:
            member = np.flatnonzero(regions == region)
            if len(member):
                b = extents[member]
                bx0, by0, bx1, by1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
                margin = 0.1 * max(bx1 - bx0, by1 - by0)
                boxes.append((region, bx0 - margin, by0 - margin, bx1 + margin, by1 + margin))
    counts = np.bincount(types, minlength=len(TYPE_ORDER))
    labels = [f"{TYPE_NAMES[key]} · {fmt_int(counts[c])} МО" for c, key in enumerate(TYPE_ORDER)]
    legend_h = legend_height(size) if legend else 0.0
    column = MAP_COLUMN * w_in if boxes else 0.0
    gap = MAP_GAP if boxes else 0.0
    map_w = w_in - column - gap
    map_h = map_w * aspect
    if map_h > h_in - legend_h:
        map_h = h_in - legend_h
        map_w = map_h / aspect
    left = x0_in + (w_in - (column + gap + map_w)) / 2
    map_bottom = y0_in + legend_h + (h_in - legend_h - map_h) / 2
    ax = add_axes_in(fig, left + column + gap, map_bottom, map_w, map_h)
    collection = PathCollection(paths, facecolors=colors, edgecolors="white", linewidths=0.12)
    collection.set_rasterized(rasterized)
    ax.add_collection(collection)
    ax.set_xlim(mx0, mx1)
    ax.set_ylim(my0, my1)
    ax.axis("off")

    if boxes:
        title_h = size * 1.6 / 72
        natural = [column * (by1 - by0) / (bx1 - bx0) for _, bx0, by0, bx1, by1 in boxes]
        room = map_h - len(boxes) * title_h - 0.15 * (len(boxes) - 1)
        scale = min(1.0, room / sum(natural))
        top = map_bottom + map_h
        for (region, bx0, by0, bx1, by1), natural_h in zip(boxes, natural):
            ih, iw = natural_h * scale, column * scale
            top -= title_h
            inset = add_axes_in(fig, left + column - iw, top - ih, iw, ih)
            near = np.flatnonzero((extents[:, 2] >= bx0) & (extents[:, 0] <= bx1) &
                                  (extents[:, 3] >= by0) & (extents[:, 1] <= by1))
            part = PathCollection([paths[i] for i in near], facecolors=[colors[i] for i in near],
                                  edgecolors="white", linewidths=0.3)
            part.set_rasterized(rasterized)
            inset.add_collection(part)
            inset.set_xlim(bx0, bx1)
            inset.set_ylim(by0, by1)
            inset.set_xticks([])
            inset.set_yticks([])
            for spine in inset.spines.values():
                spine.set_visible(True)
                spine.set_color(SPINE)
                spine.set_linewidth(0.8)
            inset.set_title(region, fontsize=size - 0.5, color=INK2, weight="normal", pad=3, loc="left")
            ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, edgecolor=INK, linewidth=0.8))
            fig.add_artist(ConnectionPatch(xyA=(1, 0.5), coordsA=inset.transAxes, xyB=(bx0, (by0 + by1) / 2),
                                           coordsB=ax.transData, color=INK3, linewidth=0.6))
            top -= ih + 0.15
    if legend:
        fw, fh = fig.get_size_inches()
        handles = [Patch(facecolor=TYPE_HEX[c], edgecolor="none", label=labels[c]) for c in range(len(TYPE_ORDER))]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=((x0_in + w_in / 2) / fw, (map_bottom - 0.1) / fh),
                  bbox_transform=fig.transFigure, ncol=2, fontsize=size, handlelength=1.0, handleheight=1.0,
                  columnspacing=1.8, borderaxespad=0.0, labelspacing=0.55)
    return ax


# ------------------------------------------------------------------ chart: choice of the edge rule
def draw_network_choice(fig, spec, res: Results, size: float = 10):
    nets = res.networks
    chosen = res.summary["network"]["chosen_rule"]
    dynamic = nets[~nets["static"].astype(bool)].reset_index(drop=True)
    static = nets[nets["static"].astype(bool)].sort_values("assort_within_region_mean", ascending=False)
    gs = spec.subgridspec(1, 2, width_ratios=[4.2, 1.35], wspace=0.05)
    ax = fig.add_subplot(gs[0])
    strip = fig.add_subplot(gs[1], sharey=ax)
    y_all = nets["assort_within_region_mean"].astype(float)
    ax.set_ylim(min(0.0, float(y_all.min()) * 1.2), float(y_all.max()) * 1.2)
    x = dynamic["stability_jaccard"].astype(float)
    ax.set_xlim(0, float(x.max()) * 1.3)
    grid(ax, "both")
    threshold = res.config.get("network", {}).get("geography_filter", {}).get("min_within_region_assortativity")
    if threshold is not None:
        for target in (ax, strip):
            target.axhline(threshold, color=INK3, linewidth=0.9, zorder=1)
        ax.text(0.985, threshold, f"порог исключения {fmt_num(threshold, 2)}", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=size - 1, color=INK3)
    is_chosen = (dynamic["rule"] == chosen).to_numpy()
    yv = dynamic["assort_within_region_mean"].astype(float)
    ax.scatter(x[~is_chosen], yv[~is_chosen], s=60, color=MUTED, edgecolors="white", linewidths=1.5, zorder=3)
    ax.scatter(x[is_chosen], yv[is_chosen], s=125, color=EMPHASIS, edgecolors="white", linewidths=2, zorder=4)
    order = np.argsort(~is_chosen, kind="stable")  # the chosen rule gets the first pick of slots
    texts = [RULE_SHORT.get(dynamic.loc[i, "rule"], dynamic.loc[i, "label"]) for i in order]
    weights = ["bold" if is_chosen[i] else "normal" for i in order]
    place_labels(ax, np.column_stack([x, yv])[order], texts, size=size, weights=weights)
    ax.set_xlabel(f"Устойчивость рёбер {res.years[0]} → {res.years[1]} (доля общих рёбер, Jaccard)")
    ax.set_ylabel(f"Ассортативность внутри регионов\n(среднее по {res.n_indicators} независимым показателям)")
    comma_axis(ax, "both")

    strip.set_xlim(0, 1)
    grid(strip, "y")
    strip.tick_params(axis="y", labelleft=False)
    strip.set_xticks([])
    strip.spines["left"].set_visible(False)
    strip.set_facecolor(PANEL)
    # Labels sit in the empty upper part of the strip, with leaders down to the points.
    xs = np.linspace(0.3, 0.7, len(static)) if len(static) > 1 else np.array([0.5])
    y_top = ax.get_ylim()[1]
    for k, (xi, (_, row)) in enumerate(zip(xs, static.iterrows())):
        value = float(row["assort_within_region_mean"])
        strip.scatter([xi], [value], s=60, color="white", edgecolors=INK2, linewidths=1.4, zorder=3)
        text = (f"{RULE_SHORT.get(row['rule'], row['label'])}:\n{fmt_pct(row['same_region_share'])} рёбер\n"
                "внутри региона")
        strip.annotate(text, (xi, value), xytext=(xi, y_top * (0.9 - 0.3 * k)), textcoords="data", ha="center",
                       va="top", fontsize=size - 1.5, color=INK, linespacing=1.15,
                       arrowprops={"arrowstyle": "-", "color": INK3, "linewidth": 0.7, "shrinkA": 2, "shrinkB": 5})
    strip.set_title("Статичные правила", fontsize=size, color=INK2, loc="center")
    strip.set_xlabel("не зависят от расходов:\nустойчивость = 1", fontsize=size - 1.5, color=INK3)
    return ax, strip


# ------------------------------------------------------------------ chart: choice of K
def draw_k_choice(fig, spec, res: Results, size: float = 10):
    table = res.k_selection.sort_values("k").reset_index(drop=True)
    chosen = res.k
    ks = table["k"].to_numpy(int)
    gs = spec.subgridspec(1, 2, width_ratios=[1.15, 1.6], wspace=0.16)
    left = fig.add_subplot(gs[0])
    small = gs[1].subgridspec(2, 3, hspace=0.6, wspace=0.55)

    def mark(ax):
        ax.axvspan(chosen - 0.4, chosen + 0.4, color=BAND, zorder=0)

    mark(left)
    grid(left, "y")
    lines = [("territory_subsample_ari_p10", "10-й перцентиль ARI, подвыборки МО", MUTED_LIGHT, 1.6),
             ("month_bootstrap_ari_p10", "10-й перцентиль ARI, бутстреп месяцев", MUTED, 1.6),
             ("stability_score", "Устойчивость (среднее ARI)", EMPHASIS, 2.4)]
    for column, label, color, width in lines:
        left.plot(ks, table[column].astype(float), color=color, linewidth=width, marker="o",
                  markersize=4.5 if width < 2 else 7, markeredgecolor="white", markeredgewidth=1.2, zorder=3,
                  label=label)
    best = float(table.loc[table["k"] == chosen, "stability_score"].iloc[0])
    left.annotate(f"{eq('K', chosen)}: {fmt_num(best, 2)}", (chosen, best), xytext=(-10, 10),
                  textcoords="offset points", ha="right", fontsize=size, weight="bold", color=INK, zorder=5,
                  bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.9})
    low = float(np.nanmin(table[[c for c, *_ in lines]].to_numpy(float)))
    left.set_ylim(max(0.0, low - 0.2), 1.03)
    left.set_xlim(ks.min() - 0.5, ks.max() + 0.5)
    left.set_xticks(ks)
    left.set_xlabel("Число типов K")
    left.set_title("Устойчивость разбиения", fontsize=size + 0.5)
    comma_axis(left, "y")
    handles, names = left.get_legend_handles_labels()
    left.legend(handles[::-1], names[::-1], loc="lower right", fontsize=size - 1, handlelength=1.6,
                borderaxespad=0.2)

    for idx, (name, sign) in enumerate(ICVI):
        ax = fig.add_subplot(small[idx // 3, idx % 3])
        mark(ax)
        grid(ax, "y")
        values = table[name].to_numpy(float)
        ok = np.isfinite(values)
        ax.plot(ks[ok], values[ok], color=MUTED, linewidth=1.8, zorder=2)
        ax.scatter(ks[ok], values[ok], s=18, color=MUTED, zorder=3)
        best_k = ks[ok][np.argmax(sign * values[ok])]
        ax.scatter([best_k], [values[ks == best_k][0]], s=44, color=EMPHASIS, edgecolors="white", linewidths=1.2,
                   zorder=4)
        ax.set_title(f"{name} {'↑' if sign > 0 else '↓'}", fontsize=size, loc="left")
        if not ok.all():
            ax.text(0.98, 0.97, "н/д: K = " + ", ".join(str(k) for k in ks[~ok]), transform=ax.transAxes,
                    ha="right", va="top", fontsize=size - 2, color=INK3)
        ax.set_xticks(ks)
        ax.set_xlim(ks.min() - 0.5, ks.max() + 0.5)
        ax.tick_params(labelsize=size - 1.5)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        comma_axis(ax, "y")
    return left


# ------------------------------------------------------------------ chart: methods heat-table
def method_label(row) -> str:
    return METHOD_SHORT.get(row["method"], row["label"])


def draw_methods(fig, spec, res: Results, size: float = 10):
    """Heat-table of ICVI ranks (1 = best) for 2023 and 2024, admission checks and Copeland scores."""
    table = res.methods.reset_index(drop=True)
    chosen = res.summary["methods"]["chosen"]
    x0, y0, width, height = spec_inches(fig, spec)
    ax = add_axes_in(fig, x0, y0, width, height)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    names = [name for name, _ in ICVI]
    ranks = {}
    for year in ("2023", "2024"):
        for name, sign in ICVI:
            values = table[f"{name}_{year}"].astype(float) * sign
            ranks[(year, name)] = values.fillna(-np.inf).rank(ascending=False, method="min").astype(int).to_numpy()
    cmap = LinearSegmentedColormap.from_list("rank", ["#2e3a48", "#eef1f4"])
    elig = res.config.get("methods", {}).get("eligibility", {})
    min_n = int(np.ceil(float(elig["min_type_share"]) * res.n)) if "min_type_share" in elig else None
    min_ari = float(elig["min_month_bootstrap_ari_p10"]) if "min_month_bootstrap_ari_p10" in elig else None

    rows = []
    for family in dict.fromkeys(table["family"]):
        rows.append(("group", family))
        rows.extend(("method", i) for i in table.index[table["family"] == family])
    label_w = max(labels_width_in(fig, [method_label(r) for _, r in table.iterrows()], size, "bold"),
                  labels_width_in(fig, [FAMILY_LABEL.get(f, f) for f in table["family"]], size - 1)) + 0.25
    gap = 0.22
    unit = (width - label_w - 3 * gap) / (12 + 2 * 1.5 + 3 * 1.15)
    cell, adm_w, cop_w = unit, 1.5 * unit, 1.15 * unit
    xs_2023 = label_w + np.arange(6) * cell
    xs_2024 = xs_2023[-1] + cell + gap + np.arange(6) * cell
    xs_adm = xs_2024[-1] + cell + gap + np.arange(2) * adm_w
    xs_cop = xs_adm[-1] + adm_w + gap + np.arange(3) * cop_w
    right = xs_cop[-1] + cop_w
    header_h = (size * 1.5 + 2 * (size - 2) * 1.35) / 72 + 0.12
    row_h = min(0.36, (height - header_h) / len(rows))
    small = size - 1.5

    y = header_h
    for kind, item in rows:
        if kind == "group":
            ax.text(0.04, y + row_h * 0.62, FAMILY_LABEL.get(item, item), ha="left", va="center", fontsize=size - 1,
                    color=INK3, weight="bold")
            ax.plot([0.04, right], [y + row_h * 0.98, y + row_h * 0.98], color=GRID, linewidth=0.8)
            y += row_h
            continue
        row = table.loc[item]
        mid = y + row_h / 2
        is_chosen = row["method"] == chosen
        if is_chosen:
            ax.add_patch(Rectangle((0, y), right, row_h, color=BAND, zorder=0, linewidth=0))
        ax.text(0.14, mid, method_label(row), ha="left", va="center", fontsize=size,
                weight="bold" if is_chosen else "normal")
        for year, xs in (("2023", xs_2023), ("2024", xs_2024)):
            for j, name in enumerate(names):
                rank = ranks[(year, name)][item]
                value = row[f"{name}_{year}"]
                shade = cmap((rank - 1) / max(len(table) - 1, 1))
                ax.add_patch(Rectangle((xs[j] + 0.02, y + 0.02), cell - 0.04, row_h - 0.04, color=shade, zorder=1,
                                       linewidth=0))
                luminance = 0.2126 * shade[0] + 0.7152 * shade[1] + 0.0722 * shade[2]
                ax.text(xs[j] + cell / 2, mid, "н/д" if pd.isna(value) else str(rank), ha="center", va="center",
                        fontsize=size - 1 if pd.notna(value) else small, zorder=2,
                        color="white" if luminance < 0.5 else INK)
        size_ok = min_n is None or row["min_size_2023"] >= min_n
        ari_ok = min_ari is None or row["month_bootstrap_ari_p10"] >= min_ari
        for x, ok, text in ((xs_adm[0], size_ok, fmt_int(row["min_size_2023"])),
                            (xs_adm[1], ari_ok, fmt_num(row["month_bootstrap_ari_p10"], 2))):
            ax.text(x + adm_w / 2, mid, text if ok else f"✕ {text}", ha="center", va="center", fontsize=size - 0.5,
                    color=INK if ok else INK2, weight="normal" if ok else "bold")
        for x, column in zip(xs_cop, ("copeland_2023", "copeland_2024", "copeland_total")):
            value = row[column]
            text = "—" if pd.isna(value) else fmt_num(value, 0, signed=True)
            ax.text(x + cop_w / 2, mid, text, ha="center", va="center", fontsize=size - 0.5,
                    weight="bold" if column == "copeland_total" and is_chosen else "normal",
                    color=INK if pd.notna(value) else INK3)
        y += row_h

    top_line = size * 0.75 / 72
    first, last = res.years
    groups = ((xs_2023[0], xs_2023[-1] + cell, f"Ранг, {first} (обучение)"),
              (xs_2024[0], xs_2024[-1] + cell, f"Ранг, {last} (валидация)"),
              (xs_adm[0], xs_adm[-1] + adm_w, "Допуск"),
              (xs_cop[0], xs_cop[-1] + cop_w, "Коупленд"))
    for a, b, text in groups:
        ax.text((a + b) / 2, top_line, text, ha="center", va="center", fontsize=size - 0.5, weight="bold",
                color=INK2)
        ax.plot([a + 0.04, b - 0.04], [top_line * 2 + 0.03] * 2, color=SPINE, linewidth=0.8)
    sub = (header_h + top_line * 2) / 2 + 0.02
    for xs in (xs_2023, xs_2024):
        for j, (name, sign) in enumerate(ICVI):
            ax.text(xs[j] + cell / 2, sub, f"{name}\n{'↑' if sign > 0 else '↓'}", ha="center", va="center",
                    fontsize=small, color=INK2, linespacing=1.15)
    adm_titles = (f"мин. тип\n≥{NBSP}{min_n}" if min_n else "мин. тип",
                  f"ARI p10\n≥{NBSP}{fmt_num(min_ari, 1)}" if min_ari is not None else "ARI p10")
    for x, text in zip(xs_adm, adm_titles):
        ax.text(x + adm_w / 2, sub, text, ha="center", va="center", fontsize=small, color=INK2, linespacing=1.15)
    for x, text in zip(xs_cop, (first, last, "сумма")):
        ax.text(x + cop_w / 2, sub, text, ha="center", va="center", fontsize=small, color=INK2)
    ax.text(0.04, sub, "Метод", ha="left", va="center", fontsize=size - 0.5, color=INK2, weight="bold")
    return ax


# ------------------------------------------------------------------ chart: type profiles
def draw_type_profiles(fig, spec, res: Results, size: float = 10):
    gs = spec.subgridspec(2, 4, hspace=0.75, wspace=0.14)
    rows = np.arange(len(TYPE_ORDER))
    axes = []
    for idx, (indicator, title, scale, digits, coverage_col) in enumerate(PROFILE_PANELS):
        ax = fig.add_subplot(gs[idx // 4, idx % 4])
        axes.append(ax)
        data = res.interval(indicator)
        med, low, high = (data[c].to_numpy(float) * scale for c in ("median", "ci_low", "ci_high"))
        grid(ax, "x")
        for c in rows:
            ax.plot([low[c], high[c]], [c, c], color=TYPE_HEX[c], linewidth=2.6, solid_capstyle="round", zorder=2)
            ax.scatter([med[c]], [c], s=50, color=TYPE_HEX[c], edgecolors="white", linewidths=1.4, zorder=3)
        xmax = float(np.nanmax(high)) * 1.55
        ax.set_xlim(0, xmax)
        for c in rows:
            ax.text(high[c] + 0.03 * xmax, c, fmt_num(med[c], digits), va="center", ha="left", fontsize=size - 0.5)
            if coverage_col:
                share = float(res.type_row(c)[coverage_col])
                if share < COVERAGE_NOTE:
                    ax.text(high[c] + 0.03 * xmax, c + 0.4, f"охват {fmt_pct(share)}", va="center", ha="left",
                            fontsize=size - 1.5, color=INK3)
        ax.set_ylim(len(rows) - 0.35, -0.6)
        ax.set_yticks(rows)
        if idx % 4 == 0:
            ax.set_yticklabels([type_short(c) for c in rows], fontsize=size, color=INK)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.spines["left"].set_visible(False)
        panel_w = ax.get_position().width * fig.get_figwidth() * 1.1  # a title may reach into the gap
        if text_width_in(fig, title, size - 0.5, "bold") > panel_w:
            title = wrap(fig, title.replace("\n", " "), panel_w, size - 0.5, "bold")
        ax.set_title(title, fontsize=size - 0.5, loc="left", weight="bold", color=INK, linespacing=1.2)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.tick_params(axis="x", labelsize=size - 1.5)
        comma_axis(ax, "x")
    return axes


# ------------------------------------------------------------------ chart: out-of-region R²
def loro_labels(res: Results) -> list[str]:
    loro = res.loro.sort_values("types", ascending=True)
    return [f"{INDICATOR_LABELS.get(r['indicator'], r['indicator'])}\n{fmt_int(r['n'])} МО с данными"
            for _, r in loro.iterrows()]


def draw_out_of_region(fig, spec, res: Results, size: float = 10):
    loro = res.loro.sort_values("types", ascending=True).reset_index(drop=True)
    ax = fig.add_subplot(spec)
    series = (("municipal_type", "Статус МО (район, округ, городской округ…)", MUTED_LIGHT),
              ("municipal_type_and_types", f"Статус МО + типы v{res.version}", MUTED),
              ("types", f"Типы v{res.version}", EMPHASIS))
    height = 0.25
    y = np.arange(len(loro))
    for k, (column, label, color) in enumerate(series):
        ax.barh(y + (k - 1) * (height + 0.03), loro[column].astype(float), height=height, color=color, label=label,
                zorder=2)
    for yi, value in zip(y, loro["types"].astype(float)):
        ax.text(max(value, 0) + 0.006, yi + height + 0.03, fmt_num(value, 2), va="center", ha="left",
                fontsize=size - 0.5, color=INK, weight="bold")
    ax.axvline(0, color=INK2, linewidth=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(loro_labels(res), fontsize=size - 0.5, linespacing=1.2, color=INK)
    grid(ax, "x")
    ax.spines["left"].set_visible(False)
    values = loro[[s[0] for s in series]].to_numpy(float)
    ax.set_xlim(min(0.0, float(np.nanmin(values))) - 0.02, float(np.nanmax(values)) + 0.06)
    ax.set_ylim(-0.65, len(loro) - 0.35)
    ax.set_xlabel("R² вне региона (больше — лучше)")
    comma_axis(ax, "x")
    handles = [Patch(color=c, label=l) for _, l, c in reversed(series)]
    ax.legend(handles=handles, loc="lower right", fontsize=size - 0.5, handlelength=1.2, borderaxespad=0.2)
    return ax


# ------------------------------------------------------------------ chart: transitions
def top_flows(counts, count: int = 3) -> list[tuple[int, int, int]]:
    counts = np.asarray(counts)
    flows = [(int(counts[i, j]), i, j) for i in range(len(counts)) for j in range(len(counts)) if i != j]
    flows = [f for f in sorted(flows, key=lambda f: (-f[0], f[1], f[2])) if f[0] > 0]
    return [(i, j, n) for n, i, j in flows[:count]]


def draw_sankey(fig, ax, counts, res: Results, size: float = 10):
    """Alluvial diagram of annual type transitions drawn with Bézier patches."""
    counts = np.asarray(counts, dtype=float)
    k = len(counts)
    total = counts.sum()
    gap = 0.04 * total
    height = total + gap * (k - 1)
    out_n, in_n = counts.sum(axis=1), counts.sum(axis=0)
    left_labels = [f"{type_short(i)}\n{fmt_int(out_n[i])}" for i in range(k)]
    right_labels = [fmt_int(in_n[i]) for i in range(k)]  # same order and colours as on the left
    ax_w = ax.get_position().width * fig.get_figwidth()
    lab_l = labels_width_in(fig, left_labels, size) + 0.12
    lab_r = labels_width_in(fig, right_labels, size) + 0.12
    unit = max(ax_w - lab_l - lab_r, 0.5)  # inches per data unit between the two columns
    block = 0.12 / unit
    ax.set_xlim(-lab_l / unit, 1 + lab_r / unit)
    left_tops = height - np.concatenate([[0], np.cumsum(out_n + gap)[:-1]])
    right_tops = height - np.concatenate([[0], np.cumsum(in_n + gap)[:-1]])
    source, target = left_tops.copy(), right_tops.copy()
    for i in range(k):
        for j in range(k):
            n = counts[i, j]
            if n <= 0:
                continue
            s, t = source[i], target[j]
            source[i] -= n
            target[j] -= n
            a, b = block, 1 - block
            mid = (a + b) / 2
            vertices = [(a, s), (mid, s), (mid, t), (b, t), (b, t - n), (mid, t - n), (mid, s - n), (a, s - n), (a, s)]
            codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO,
                     MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
            ax.add_patch(PathPatch(MPath(vertices, codes), facecolor=TYPE_HEX[i], edgecolor="none",
                                   alpha=0.2 if i == j else 0.65))
    for i in range(k):
        ax.add_patch(Rectangle((0, left_tops[i] - out_n[i]), block, out_n[i], color=TYPE_HEX[i], linewidth=0))
        ax.add_patch(Rectangle((1 - block, right_tops[i] - in_n[i]), block, in_n[i], color=TYPE_HEX[i], linewidth=0))
        ax.text(-0.1 / unit, left_tops[i] - out_n[i] / 2, left_labels[i], ha="right", va="center", fontsize=size,
                linespacing=1.15)
        ax.text(1 + 0.1 / unit, right_tops[i] - in_n[i] / 2, right_labels[i], ha="left", va="center", fontsize=size,
                linespacing=1.15)
    first, last = res.years
    for xpos, text in ((block / 2, first), (1 - block / 2, last)):
        ax.text(xpos, height + 0.025 * total, text, ha="center", va="bottom", fontsize=size, weight="bold")
    ax.set_ylim(-0.04 * total, height + 0.08 * total)
    ax.axis("off")


def draw_monthly_counts(ax, res: Results, size: float = 10, label_width: float | None = None):
    monthly = res.monthly
    periods = monthly["period"].astype(str).tolist()
    t = np.arange(len(periods))
    grid(ax, "y")
    for c in range(len(TYPE_ORDER)):
        values = monthly[f"n_type_{c}"].to_numpy(float)
        ax.plot(t, values, color=TYPE_HEX[c], linewidth=2, solid_joinstyle="round", zorder=3)
        ax.scatter([t[-1]], [values[-1]], s=24, color=TYPE_HEX[c], edgecolors="white", linewidths=1, zorder=4)
        label = f"{type_short(c)} · {fmt_int(values[-1])}"
        if label_width:
            label = wrap(ax.figure, label, label_width, size - 1)
        ax.text(t[-1] + 0.7, values[-1], label, va="center", ha="left", fontsize=size - 1, color=INK,
                linespacing=1.1)
    for i, p in enumerate(periods):
        if p[5:7] == "01" and i > 0:
            ax.axvline(i - 0.5, color=SPINE, linewidth=0.8, zorder=1)
    ticks = [i for i, p in enumerate(periods) if p[5:7] in ("01", "07")]
    ax.set_xticks(ticks)
    ax.set_xticklabels([month_label(periods[i], with_year=periods[i][5:7] == "01") for i in ticks],
                       fontsize=size - 1.5)
    ax.set_xlim(-0.5, len(periods) - 0.5)
    ax.set_ylim(0, float(monthly[[f"n_type_{c}" for c in range(len(TYPE_ORDER))]].to_numpy().max()) * 1.12)
    ax.set_ylabel("Число МО в типе")
    ax.tick_params(axis="y", labelsize=size - 1.5)
    comma_axis(ax, "y")


def draw_transitions(fig, spec, res: Results, size: float = 10, flows_note: bool = True,
                     max_label: float | None = None):
    x0, y0, width, height = spec_inches(fig, spec)
    label_w = labels_width_in(fig, [f"{type_short(c)} · {fmt_int(9999)}" for c in range(len(TYPE_ORDER))],
                              size - 1)
    if max_label:
        label_w = min(label_w, max_label)
    label_w += 0.25
    flows = top_flows(np.array(res.summary["tracking"]["annual_transitions"]), 4)
    note_h = 0.32 + 0.19 * len(flows) if flows_note else 0.0
    sankey_w = 0.53 * width
    ax = add_axes_in(fig, x0, y0 + note_h, sankey_w, height - note_h - 0.3)
    counts = np.array(res.summary["tracking"]["annual_transitions"])
    draw_sankey(fig, ax, counts, res, size=size)
    ax.set_title(f"Тип {res.years[0]} → тип {res.years[1]}", fontsize=size + 0.5, loc="center", pad=16)
    if flows_note:
        fw, fh = fig.get_size_inches()
        fig.text((x0 + 0.1) / fw, (y0 + note_h - 0.04) / fh, "Главные переходы", fontsize=size - 1, color=INK2,
                 weight="bold", va="top")
        fig.text((x0 + 0.1) / fw, (y0 + note_h - 0.26) / fh,
                 "\n".join(f"{upper_first(flow(i, j))}: {fmt_int(n)}" for i, j, n in flows),
                 fontsize=size - 1, color=INK2, va="top", linespacing=1.35)
    left = x0 + sankey_w + 0.75
    ax2 = add_axes_in(fig, left, y0 + 0.45, x0 + width - left - label_w, height - 0.45 - 0.3)
    draw_monthly_counts(ax2, res, size=size, label_width=max_label)
    ax2.set_title("Число МО по месяцам", fontsize=size + 0.5)
    return ax, ax2


# ------------------------------------------------------------------ chart: dynamics
def draw_dynamics(fig, spec, res: Results, size: float = 10):
    dyn = res.dynamics.sort_values("type").reset_index(drop=True)
    gs = spec.subgridspec(1, 2, width_ratios=[1.45, 1], wspace=0.14)
    rows = np.arange(len(dyn))
    first, last = res.years

    ax = fig.add_subplot(gs[0])
    grid(ax, "x")
    v0, v1 = dyn["marketplaces_pct_2023"].to_numpy(float), dyn["marketplaces_pct_2024"].to_numpy(float)
    xmax = float(max(v0.max(), v1.max())) * 1.85
    for c in rows:
        color = TYPE_HEX[int(dyn.loc[c, "type"])]
        ax.plot([v0[c], v1[c]], [c, c], color=tint(color, 0.45), linewidth=3, solid_capstyle="round", zorder=2)
        ax.scatter([v0[c]], [c], s=50, color=tint(color, 0.55), edgecolors="white", linewidths=1.3, zorder=3)
        ax.scatter([v1[c]], [c], s=60, color=color, edgecolors="white", linewidths=1.3, zorder=4)
        ax.text(v0[c] - 0.015 * xmax, c, fmt_num(v0[c], 1), ha="right", va="center", fontsize=size - 1, color=INK2)
        ax.text(v1[c] + 0.018 * xmax, c, fmt_num(v1[c], 1), ha="left", va="center", fontsize=size - 0.5)
        change = (f"{fmt_pp(dyn.loc[c, 'marketplaces_change_pp_median'])}\n"
                  f"{fmt_ci(dyn.loc[c, 'marketplaces_change_ci_low'], dyn.loc[c, 'marketplaces_change_ci_high'])}")
        ax.text(0.99 * xmax, c, change, ha="right", va="center", fontsize=size - 1, color=INK2, linespacing=1.15)
    ax.text(v0[0], -0.6, first, ha="center", va="bottom", fontsize=size - 1, color=INK2)
    ax.text(v1[0], -0.6, last, ha="center", va="bottom", fontsize=size - 1, color=INK, weight="bold")
    ax.text(0.99 * xmax, -0.6, "медиана изменений\nпо МО [95% ДИ]", ha="right", va="bottom", fontsize=size - 1.5,
            color=INK3, linespacing=1.1)
    ax.set_xlim(0, xmax)
    ax.set_ylim(len(rows) - 0.45, -1.1)
    ax.set_yticks(rows)
    ax.set_yticklabels([type_short(t) for t in dyn["type"]], fontsize=size, color=INK)
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
    ax.set_xlabel("Доля маркетплейсов, % (медиана по МО)")
    ax.set_title("Маркетплейсы", fontsize=size + 0.5)
    comma_axis(ax, "x")

    ax2 = fig.add_subplot(gs[1], sharey=ax)
    grid(ax2, "x")
    g, lo, hi = (dyn[c].to_numpy(float) for c in
                 ("nominal_growth_mean_pct", "nominal_growth_ci_low_pct", "nominal_growth_ci_high_pct"))
    xmax2 = float(hi.max()) * 1.32
    for c in rows:
        color = TYPE_HEX[int(dyn.loc[c, "type"])]
        ax2.plot([lo[c], hi[c]], [c, c], color=color, linewidth=2.6, solid_capstyle="round", zorder=2)
        ax2.scatter([g[c]], [c], s=55, color=color, edgecolors="white", linewidths=1.3, zorder=3)
        ax2.text(hi[c] + 0.025 * xmax2, c, fmt_num(g[c], 1) + "%", ha="left", va="center", fontsize=size - 0.5)
    ax2.set_xlim(0, xmax2)
    ax2.tick_params(axis="y", labelleft=False)
    ax2.spines["left"].set_visible(False)
    ax2.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax2.set_xlabel(f"Рост расходов, {last} к {first}, %")
    ax2.set_title("Номинальный рост расходов", fontsize=size + 0.5)
    comma_axis(ax2, "x")
    return ax, ax2


# ------------------------------------------------------------------ chart: analogues
def analogue_labels(res: Results) -> tuple[list[str], list[str]]:
    an = res.analogues.sort_values("mae_pp", ascending=False)
    top = [PREDICTOR_LABELS.get(p, p) for p in an["predictor"]]
    bottom = []
    for _, r in res.comparisons.iterrows():
        better = "в своём регионе: " if r["better"].endswith("same_region") else ""
        bottom.append(upper_first(f"{better}против {PREDICTOR_GENITIVE.get(r['baseline'], r['baseline'])}"))
    return top, bottom


def draw_analogues(fig, spec, res: Results, size: float = 10):
    an = res.analogues.sort_values("mae_pp", ascending=False).reset_index(drop=True)
    comp = res.comparisons.reset_index(drop=True)
    top_labels, bottom_labels = analogue_labels(res)
    gs = spec.subgridspec(2, 1, height_ratios=[len(an), len(comp) + 0.4], hspace=0.62)
    ax = fig.add_subplot(gs[0])
    y = np.arange(len(an))
    network = an["predictor"].str.startswith("network").to_numpy()
    ax.barh(y, an["mae_pp"].astype(float), height=0.58, color=[EMPHASIS if n else MUTED_LIGHT for n in network],
            zorder=2)
    for yi, value, net in zip(y, an["mae_pp"].astype(float), network):
        ax.text(value + 0.03, yi, fmt_num(value, 2), va="center", ha="left", fontsize=size - 0.5,
                weight="bold" if net else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(top_labels, fontsize=size - 0.5, color=INK)
    grid(ax, "x")
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, float(an["mae_pp"].max()) * 1.15)
    ax.set_xlabel("Средняя абсолютная ошибка прогноза роста, п.п. (меньше — лучше)")
    ax.set_title("Ошибка прогноза", fontsize=size + 0.5)
    comma_axis(ax, "x")

    ax2 = fig.add_subplot(gs[1])
    y2 = np.arange(len(comp))[::-1]
    grid(ax2, "x")
    ax2.axvline(0, color=INK2, linewidth=0.9, zorder=1)
    for yi, (_, row) in zip(y2, comp.iterrows()):
        ax2.plot([row["ci_low"], row["ci_high"]], [yi, yi], color=EMPHASIS, linewidth=2.6, solid_capstyle="round",
                 zorder=2)
        ax2.scatter([row["mae_difference_pp"]], [yi], s=55, color=EMPHASIS, edgecolors="white", linewidths=1.3,
                    zorder=3)
        ax2.text(row["ci_low"] - 0.012, yi, fmt_num(row["mae_difference_pp"], 2), va="center", ha="right",
                 fontsize=size - 0.5)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(bottom_labels, fontsize=size - 0.5, color=INK)
    lo = float(min(comp["ci_low"].min(), 0))
    hi = float(max(comp["ci_high"].max(), 0))
    span = hi - lo
    ax2.set_xlim(lo - 0.3 * span, hi + 0.15 * span)
    ax2.set_ylim(-0.6, len(comp) - 0.4)
    ax2.spines["left"].set_visible(False)
    ax2.xaxis.set_major_locator(MaxNLocator(5))
    ax2.set_xlabel("Разница ошибок сетевых аналогов и правила, п.п. (< 0 — аналоги точнее)")
    ax2.set_title("Парные разницы с 95% интервалом", fontsize=size + 0.5)
    comma_axis(ax2, "x")
    return ax, ax2


# ------------------------------------------------------------------ derived facts used in texts
def top_regions(row, count: int = 3) -> list[str]:
    parts = [p.strip() for p in str(row["top_regions"]).split(";") if p.strip()]
    return [short_region(re.sub(r"\s*\(\d+\)$", "", p)) for p in parts[:count]]


def stayed_share(counts) -> float:
    counts = np.asarray(counts, dtype=float)
    return float(np.trace(counts) / counts.sum())


def subsample_share(res: Results) -> str:
    """Territory subsample size as stated in the configured selection rule ("80%"), if any."""
    match = re.search(r"(\d+)\s*%\s*территор", res.config.get("k_selection", {}).get("selection_rule", ""))
    return f"{match.group(1)}%" if match else ""


# ------------------------------------------------------------------ README / report figures
def build_pngs(res: Results, outlines: dict, out: Path, only: set | None = None) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    first, last = res.years
    probe = plt.figure(figsize=(PNG_WIDTH, 2))  # for measuring label widths

    def margin(labels, size=10.0):
        return 0.25 + labels_width_in(probe, labels, size) + 0.12

    def want(name):
        return only is None or name in only

    def finish(fig, name):
        written.append(out / f"{name}.png")
        save_png(fig, written[-1])

    if want("types-map"):
        fig, top, bottom = png_figure(
            6.6, f"Типы локальных экономик, {first}",
            f"{count_word(res.n, ('муниципалитет', 'муниципалитета', 'муниципалитетов'))} панели СберИндекса; "
            f"тип назначен по годовому профилю расходов {first} г. и сети сходства.",
            "Белые области — территории вне панели. Москва и Санкт-Петербург увеличены слева. Границы: СберИндекс "
            "и OpenStreetMap, CC BY-SA 4.0; равновеликая проекция Альберса.")
        _, (x0, y0, x1, y1) = map_frame(list(outlines.values()))
        needed = top + map_height(PNG_WIDTH - 0.5, (y1 - y0) / (x1 - x0)) + bottom + 0.1
        fig.set_size_inches(PNG_WIDTH, needed)
        draw_types_map(fig, region_spec(fig, 0.25, 0.25, top, bottom), res, outlines)
        finish(fig, "types-map")

    if want("network-choice"):
        rule = res.chosen_rule()
        fig, top, bottom = png_figure(
            6.2, "Выбор правила рёбер сети",
            f"{count_word(len(res.networks), ('правило', 'правила', 'правил'))} kNN ({eq('k', res.config['network']['k'])}) "
            f"на одних и тех же МО. Выбрано «{RULE_SHORT.get(rule['rule'], rule['label'])}»: правее и выше — лучше.",
            "Ассортативность — корреляция показателя на двух концах ребра после вычета средних по региону; "
            "показатели Росстата и доступность рынков в построении сети не участвуют. Правило выбрано по "
            "Коупленду из трёх критериев: ассортативность, она же внутри регионов, устойчивость рёбер.")
        draw_network_choice(fig, region_spec(fig, 1.2, 0.25, top + 0.1, bottom + 0.62), res)
        finish(fig, "network-choice")

    if want("k-choice"):
        share = subsample_share(res)
        fig, top, bottom = png_figure(
            5.9, "Выбор числа типов K",
            f"KMeans при K = {res.k_selection['k'].min()}…{res.k_selection['k'].max()}. ICVI указывают на разные K, "
            "поэтому решает устойчивость разбиения.",
            f"Устойчивость — среднее ARI при бутстрепе месяцев и на подвыборках {share + ' ' if share else ''}МО. "
            "Тёмная точка — лучшее K по индексу; ↑ — больше лучше, ↓ — меньше лучше; серая полоса — выбранное K.")
        draw_k_choice(fig, region_spec(fig, 0.75, 0.2, top + 0.15, bottom + 0.55), res)
        finish(fig, "k-choice")

    if want("methods"):
        method = res.chosen_method()
        fig, top, bottom = png_figure(
            7.0, "Сравнение методов по шести ICVI",
            f"{count_word(len(res.methods), ('метод', 'метода', 'методов'))} при {eq('K', res.k)}; ранг 1 — лучший. "
            f"Выбран метод «{method['label']}».",
            f"{last}: профили отнесены к центрам {first} г.; этот год участвовал в выборе модели. Допуск: наименьший тип и 10-й "
            "перцентиль ARI при бутстрепе месяцев не ниже порогов. Коупленд считается только среди допустимых "
            "методов; н/д — индекс не определён и проигрывает все сравнения.", width=10.5)
        draw_methods(fig, region_spec(fig, 0.2, 0.2, top, bottom), res)
        finish(fig, "methods")

    if want("type-profiles"):
        fig, top, bottom = png_figure(
            6.9, "Профили типов: расходы и независимые данные",
            f"Медиана по МО и 95% интервал бутстрепа по регионам. Верхний ряд — признаки СберИндекса {first} г., "
            "нижний — показатели, которые в построении типов не участвуют.",
            f"Зарплата и занятость — Росстат {res.rosstat_year}, медианы по МО с данными; охват показан, если он "
            f"меньше {fmt_pct(COVERAGE_NOTE)}. "
            "Доли занятости рассчитаны по наблюдаемым значениям. Значения в МО с пропусками неизвестны; "
            "верхняя граница для всей панели не установлена.")
        left = margin([type_short(c) for c in range(len(TYPE_ORDER))])
        draw_type_profiles(fig, region_spec(fig, left, 0.25, top + 0.5, bottom + 0.35), res)
        finish(fig, "type-profiles")

    if want("out-of-region"):
        fig, top, bottom = png_figure(
            7.0, "Что типы предсказывают за пределами своего региона",
            "Регион исключается целиком: средние по группам считаются на остальных регионах и переносятся на него.",
            "Статус МО — муниципальный район, муниципальный округ, городской округ, внутригородская территория. "
            f"Показатели: Росстат {res.rosstat_year} (зарплата, население и рабочие места — в логарифмах) и "
            "доступность рынков СберИндекса.")
        draw_out_of_region(fig, region_spec(fig, margin(loro_labels(res), 9.5), 0.3, top + 0.05, bottom + 0.55), res)
        finish(fig, "out-of-region")

    if want("transitions"):
        counts = np.array(res.summary["tracking"]["annual_transitions"])
        fig, top, bottom = png_figure(
            6.0, f"Переходы между типами, {first} → {last}",
            f"{fmt_pct(stayed_share(counts), 1)} МО сохранили тип. Годовой профиль {last} г. отнесён к ближайшему "
            f"центру {first} г. после вычета общенациональной волны; справа — то же для каждого месяца.",
            "Ширина потока пропорциональна числу МО, цвет — тип в начале периода.")
        draw_transitions(fig, region_spec(fig, 0.25, 0.25, top, bottom), res)
        finish(fig, "transitions")

    if want("dynamics"):
        gap = res.summary["practical"]["gap"]
        faster = gap["ci_low"] > 0
        fig, top, bottom = png_figure(
            5.4, f"Динамика типов, {first} → {last}",
            "Доля маркетплейсов выросла во всех типах. Отношение медианных уровней расходов крупных городов и "
            f"периферии: {fmt_num(gap['level_ratio_2023'], 2)} → {fmt_num(gap['level_ratio_2024'], 2)}"
            + ("; разрыв растёт." if faster else "."),
            f"Тип — по {first} г. Изменение доли — медиана изменений по МО (не равна разности медиан), 95% интервал "
            "бутстрепа по регионам. Рост — среднее по МО логарифма отношения средних месячных расходов; номинальные "
            "рубли без поправки на цены.")
        left = margin([type_short(c) for c in range(len(TYPE_ORDER))])
        draw_dynamics(fig, region_spec(fig, left, 0.25, top + 0.3, bottom + 0.6), res)
        finish(fig, "dynamics")

    if want("analogues"):
        k = res.config.get("practical", {}).get("analogues")
        fig, top, bottom = png_figure(
            7.2, "Сетевые аналоги точнее предсказывают рост расходов",
            f"Прогноз роста расходов МО в {last} г. — медиана роста у {k} ближайших по профилю {first} г. МО; "
            "сам МО в прогноз не входит.",
            f"Ошибка — в процентных пунктах логарифма роста, n = {fmt_int(res.analogues['n'].iloc[0])} МО. "
            "Интервалы — бутстреп по регионам для парной разницы ошибок.")
        draw_analogues(fig, region_spec(fig, margin(sum(analogue_labels(res), [])), 0.3, top + 0.15,
                                        bottom + 0.6), res)
        finish(fig, "analogues")
    plt.close(probe)
    return written


# ------------------------------------------------------------------ entry point
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "reports/v1.2", help="Output directory of run_v12")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/images/v12", help="Where PNG figures go")
    parser.add_argument("--map", type=Path, help="Municipal SVG map (default: the map recorded in summary.json)")
    parser.add_argument("--only", help="Comma-separated subset of figure names (for development)")
    args = parser.parse_args(argv)
    res = load_results(args.input)
    map_path = args.map or ROOT / res.summary.get("inputs", {}).get("map", {}).get(
        "path", "reports/contest-v3/municipal_map.json")
    only = set(args.only.split(",")) if args.only else None
    with plt.rc_context(RC):
        outlines = load_outlines(map_path, res.labels["entity_id"].tolist())
        for path in build_pngs(res, outlines, args.output_dir, only):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
