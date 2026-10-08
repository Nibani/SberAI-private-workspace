"""Version 1.2 panel and attribute space.

Every municipality i and month t is described by six attributes:

* five structure coordinates  s_itc = (log(E_itc / E_it) - a_c) / q_c, the
  established v1.1 convention (a_c median, q_c IQR of all 2023 monthly values);
* the spending level  l_it = (log E_it - median_j log E_jt) / q_l, i.e. the
  distance from the median municipality of the same month, so that inflation and
  national seasonality do not move every territory at once.

The annual profile is the median of the twelve monthly vectors. Monthly tracking
uses the same attributes after removing the national monthly component
(the cross-sectional median of each coordinate), see ``remove_national_wave``.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CATEGORIES = ("health", "marketplaces", "horeca", "food", "transport")
CATEGORY_LABELS_RU = ("Здоровье", "Маркетплейсы", "Общественное питание", "Продовольствие", "Транспорт")
FEATURES = (*CATEGORIES, "level")
FEATURE_LABELS_RU = (*CATEGORY_LABELS_RU, "Уровень расходов")
PERIODS = tuple(f"{year}-{month:02d}-01" for year in (2023, 2024) for month in range(1, 13))


@dataclass(frozen=True)
class Panel:
    ids: np.ndarray        # (N,) canonical ``tid_<territory_id>``, sorted
    names: np.ndarray      # (N,)
    regions: np.ndarray    # (N,)
    periods: tuple
    totals: np.ndarray     # (N, T) rubles
    shares: np.ndarray     # (N, T, 5) fractions of the total
    sha256: str


@dataclass(frozen=True)
class Scaler:
    structure_center: np.ndarray   # (5,)
    scale: np.ndarray              # (6,) IQRs: five structure coordinates and the level
    development_months: int
    level_center: np.ndarray | None = None  # training-panel monthly log-total medians

    def as_dict(self) -> dict:
        return {"structure_center": self.structure_center.tolist(), "scale": self.scale.tolist(),
                "development_months": self.development_months,
                "features": list(FEATURES),
                "level_center": None if self.level_center is None else self.level_center.tolist()}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_panel(path: Path) -> Panel:
    """Read the tidy panel written by ``scripts/export_panel.py`` and validate it."""
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_id: dict[str, dict] = {}
    for row in rows:
        entity = by_id.setdefault(row["entity_id"], {"name": row["name"], "region": row["region"], "rows": {}})
        if row["entity_id"] != f"tid_{row['territory_id']}":
            raise ValueError(f"Canonical ID mismatch: {row['entity_id']}")
        if entity["name"] != row["name"] or entity["region"] != row["region"]:
            raise ValueError(f"Metadata changes across months: {row['entity_id']}")
        if row["period"] in entity["rows"]:
            raise ValueError(f"Duplicate month: {row['entity_id']} {row['period']}")
        values = [float(row["total_rub"])] + [float(row[f"{c}_pct"]) / 100 for c in CATEGORIES]
        if not all(np.isfinite(values)) or min(values) <= 0:
            raise ValueError(f"Non-positive or non-finite value: {row['entity_id']} {row['period']}")
        entity["rows"][row["period"]] = values
    ids = np.array(sorted(by_id))
    if any(set(by_id[i]["rows"]) != set(PERIODS) for i in ids):
        raise ValueError("Every municipality must have all 24 months of 2023-2024")
    values = np.array([[by_id[i]["rows"][p] for p in PERIODS] for i in ids], dtype=float)
    return Panel(ids=ids, names=np.array([by_id[i]["name"] for i in ids]),
                 regions=np.array([by_id[i]["region"] for i in ids]), periods=PERIODS,
                 totals=values[:, :, 0], shares=values[:, :, 1:], sha256=file_sha256(path))


def _iqr(values: np.ndarray, axis=None) -> np.ndarray:
    q75, q25 = np.percentile(values, [75, 25], axis=axis)
    return q75 - q25


def relative_level(panel: Panel) -> np.ndarray:
    """log total minus the median log total of the same month (N, T)."""
    log_total = np.log(panel.totals)
    return log_total - np.median(log_total, axis=0, keepdims=True)


def fit_scaler(panel: Panel, development_months: int = 12) -> Scaler:
    """Medians and IQRs from the pooled monthly values of the development year."""
    log_share = np.log(panel.shares[:, :development_months]).reshape(-1, len(CATEGORIES))
    level = relative_level(panel)[:, :development_months].ravel()
    scale = np.append(_iqr(log_share, axis=0), _iqr(level))
    if np.any(scale <= 0):
        raise ValueError("Zero interquartile range: attribute cannot be scaled")
    return Scaler(structure_center=np.median(log_share, axis=0), scale=scale,
                  development_months=development_months,
                  level_center=np.median(np.log(panel.totals), axis=0))


def monthly_attributes(panel: Panel, scaler: Scaler) -> np.ndarray:
    """Six standardized attributes for every municipality and month (N, T, 6)."""
    structure = (np.log(panel.shares) - scaler.structure_center) / scaler.scale[:5]
    # A held-out panel must use the training reference, even when its composition changes.
    if scaler.level_center is None:
        raise ValueError("Scaler has no training level reference; refit it before transformation")
    if np.shape(scaler.level_center) != (panel.totals.shape[1],):
        raise ValueError("Training level reference does not match the panel months")
    level = (np.log(panel.totals) - scaler.level_center) / scaler.scale[5]
    return np.concatenate([structure, level[:, :, None]], axis=2)


def annual_profile(monthly: np.ndarray, months: slice) -> np.ndarray:
    return np.median(monthly[:, months], axis=1)


def national_component(monthly: np.ndarray) -> np.ndarray:
    """Cross-sectional median of every attribute in every month (T, 6)."""
    return np.median(monthly, axis=0)


def remove_national_wave(monthly: np.ndarray, reference_profile: np.ndarray) -> np.ndarray:
    """Shift each month so that its national median equals the reference-year median.

    Distances between municipalities within a month are unchanged; only the
    common seasonal and national movement (for example the marketplace boom)
    is removed before municipalities are compared with frozen type centers.
    """
    shift = national_component(monthly) - np.median(reference_profile, axis=0)
    return monthly - shift[None]
