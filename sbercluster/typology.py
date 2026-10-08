"""Economic reading of the types: names, profiles and uncertainty.

Names are assigned by explicit rules on the 2023 type medians, not by cluster
index, so a refit with permuted labels receives the same names:

* the highest spending level and food-service share -> high spending and services;
* the lowest marketplace share -> low marketplace share;
* the lowest spending level among the rest -> low spending and everyday purchases;
* the remaining type -> medium spending.

Industry and geography describe the observed composition, not the naming rule.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TYPE_NAMES = {
    "metro": "Высокие расходы и услуги",
    "remote": "Низкая доля маркетплейсов",
    "periphery": "Низкие расходы и повседневные покупки",
    "industrial": "Средние расходы",
}
TYPE_ORDER = ("metro", "industrial", "periphery", "remote")
TYPE_SHORT = TYPE_NAMES.copy()
# One palette for the atlas, figures and the PDF deck: Okabe-Ito colours, chosen to stay
# distinguishable for the common colour-vision deficiencies (Okabe & Ito, 2008).
TYPE_COLORS = {"metro": "#CC79A7", "industrial": "#0072B2", "periphery": "#009E73", "remote": "#D55E00"}


def name_types(level: np.ndarray, shares: np.ndarray, labels: np.ndarray) -> dict[int, str]:
    """Map cluster ids to type keys with the rules in the module docstring.

    ``level`` is the 2023 median monthly spending (rubles) and ``shares`` the
    2023 median shares (fractions) in the order health, marketplaces, horeca,
    food, transport.
    """
    clusters = list(np.unique(labels))
    if len(clusters) != 4:
        raise ValueError("Type naming is defined for four types")
    med_level = {c: float(np.median(level[labels == c])) for c in clusters}
    med_share = {c: np.median(shares[labels == c], axis=0) for c in clusters}
    metro = max(clusters, key=lambda c: (med_level[c], med_share[c][2]))
    if metro != max(clusters, key=lambda c: med_share[c][2]):
        raise ValueError("The richest type is not also the food-service leader; naming rule does not apply")
    rest = [c for c in clusters if c != metro]
    remote = min(rest, key=lambda c: med_share[c][1])
    rest = [c for c in rest if c != remote]
    periphery = min(rest, key=lambda c: med_level[c])
    industrial = next(c for c in rest if c != periphery)
    return {int(metro): "metro", int(remote): "remote", int(periphery): "periphery", int(industrial): "industrial"}


def canonical_labels(labels: np.ndarray, naming: dict[int, str]) -> np.ndarray:
    """Relabel clusters so that 0..3 follow TYPE_ORDER."""
    position = {key: i for i, key in enumerate(TYPE_ORDER)}
    return np.array([position[naming[int(c)]] for c in labels])


def region_bootstrap(values: np.ndarray, labels: np.ndarray, regions: np.ndarray, k: int,
                     statistic=np.median, draws: int = 1000, seed: int = 1729) -> np.ndarray:
    """Percentile 95% intervals of a per-type statistic, resampling whole regions.

    Returns an array (k, 3): estimate, lower, upper. Missing values are ignored.
    """
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    estimate = np.array([statistic(values[ok & (labels == c)]) if np.any(ok & (labels == c)) else np.nan
                         for c in range(k)])
    unique = np.unique(regions)
    members = {r: np.where(regions == r)[0] for r in unique}
    rng = np.random.default_rng(seed)
    samples = np.full((draws, k), np.nan)
    for b in range(draws):
        chosen = np.concatenate([members[r] for r in rng.choice(unique, len(unique), replace=True)])
        v, z = values[chosen], labels[chosen]
        keep = np.isfinite(v)
        for c in range(k):
            sel = keep & (z == c)
            if sel.any():
                samples[b, c] = statistic(v[sel])
    lower, upper = np.nanpercentile(samples, [2.5, 97.5], axis=0)
    return np.column_stack([estimate, lower, upper])


def paired_region_bootstrap(diff: np.ndarray, regions: np.ndarray, draws: int = 2000,
                            seed: int = 1729) -> tuple[float, float, float]:
    """Mean of a paired difference with a region-cluster percentile interval."""
    diff = np.asarray(diff, dtype=float)
    ok = np.isfinite(diff)
    diff, regions = diff[ok], np.asarray(regions)[ok]
    unique = np.unique(regions)
    members = {r: np.where(regions == r)[0] for r in unique}
    rng = np.random.default_rng(seed)
    means = [diff[np.concatenate([members[r] for r in rng.choice(unique, len(unique), replace=True)])].mean()
             for _ in range(draws)]
    lower, upper = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lower), float(upper)


def representatives(x: np.ndarray, labels: np.ndarray, names: np.ndarray, regions: np.ndarray,
                    count: int = 3) -> dict[int, list[str]]:
    """Municipalities closest to each type center (typical members)."""
    out = {}
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        center = x[members].mean(axis=0)
        order = members[np.argsort(np.square(x[members] - center).sum(axis=1))][:count]
        out[int(c)] = [f"{names[i]} ({regions[i]})" for i in order]
    return out


def composition(labels: np.ndarray, categories: np.ndarray, k: int) -> pd.DataFrame:
    """Share of each category value inside each type."""
    table = pd.crosstab(pd.Series(labels, name="type"), pd.Series(categories, name="category"))
    return table.reindex(range(k), fill_value=0)
