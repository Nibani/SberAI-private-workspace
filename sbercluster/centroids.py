"""Geographic centroids of municipalities from the published SVG map.

The map stores outlines in a spherical Albers equal-area projection
(parallels 50/70N, central meridian 100E) scaled to SVG units. The linear SVG
scale is calibrated on municipalities whose administrative center coordinates
are known, then polygon centroids are projected back to latitude/longitude.
Only the geographic edge rule uses these coordinates.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

PHI1, PHI2, LAMBDA0 = np.radians(50.0), np.radians(70.0), np.radians(100.0)
N_CONE = (np.sin(PHI1) + np.sin(PHI2)) / 2
C_CONE = np.cos(PHI1) ** 2 + 2 * N_CONE * np.sin(PHI1)
_POINT = re.compile(r"(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)")


def polygon_centroid(path: str) -> tuple[float, float]:
    """Area-weighted centroid of an SVG path made of M/L/Z subpaths."""
    area = cx = cy = 0.0
    for part in re.split(r"(?=M)", path):
        points = np.array(_POINT.findall(part), dtype=float)
        if len(points) < 3:
            continue
        nxt = np.roll(points, -1, axis=0)
        cross = points[:, 0] * nxt[:, 1] - nxt[:, 0] * points[:, 1]
        area += cross.sum() / 2
        cx += ((points[:, 0] + nxt[:, 0]) * cross).sum() / 6
        cy += ((points[:, 1] + nxt[:, 1]) * cross).sum() / 6
    if area == 0:
        raise ValueError("Degenerate outline")
    return cx / area, cy / area


def _forward(lat, lon):
    rho = np.sqrt(C_CONE - 2 * N_CONE * np.sin(np.radians(lat))) / N_CONE
    theta = N_CONE * (np.radians(lon) - LAMBDA0)
    return rho * np.sin(theta), -rho * np.cos(theta)


def centroids_from_map(map_path: Path, ids: np.ndarray, known_lat: np.ndarray, known_lon: np.ndarray) -> dict:
    """Latitude/longitude of outline centroids for ``ids`` and the calibration error."""
    geometry = json.loads(Path(map_path).read_text(encoding="utf-8"))
    paths = {p["id"]: p["d"] for p in geometry["paths"]}
    svg = np.array([polygon_centroid(paths[i]) for i in ids])
    ok = np.isfinite(known_lat) & np.isfinite(known_lon)
    x, y = _forward(known_lat[ok], known_lon[ok])
    ax, bx = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), svg[ok, 0], rcond=None)[0]
    ay, by = np.linalg.lstsq(np.column_stack([y, np.ones_like(y)]), svg[ok, 1], rcond=None)[0]
    xs, ys = (svg[:, 0] - bx) / ax, (svg[:, 1] - by) / ay
    rho, theta = np.hypot(xs, ys), np.arctan2(xs, -ys)
    lon = np.degrees(LAMBDA0 + theta / N_CONE)
    lat = np.degrees(np.arcsin(np.clip((C_CONE - (rho * N_CONE) ** 2) / (2 * N_CONE), -1, 1)))
    lat1, lon1, lat2, lon2 = map(np.radians, (known_lat[ok], known_lon[ok], lat[ok], lon[ok]))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    error = 2 * 6371.0 * np.arcsin(np.sqrt(h))
    return {"lat": lat, "lon": lon, "calibration_points": int(ok.sum()),
            "median_km_to_admin_center": float(np.median(error)),
            "p90_km_to_admin_center": float(np.percentile(error, 90))}
