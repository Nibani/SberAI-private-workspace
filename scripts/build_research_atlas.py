"""Build the self-contained research atlas from existing, read-only results.

This command does no fitting and does not modify any source artifact.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import csv
import json
import math
import os
import sys
import tempfile
from collections import Counter
from statistics import median
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CATEGORIES = ("Здоровье", "Маркетплейсы", "Общественное питание", "Продовольствие", "Транспорт")
TOTAL = "Все категории"
PILOT_PERIOD = "2023-12-01"
VALIDATION_FILES = (
    "frozen_prototypes.json", "reference_assignments.csv", "monthly_assignments.csv",
    "cluster_profiles.csv", "consensus_neighbors.csv", "peer_validation.json",
    "external_market_access.json", "confirmation_metrics.json",
)


def _finite(value: str, context: str, *, positive: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid number in {context}: {value!r}") from exc
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"Expected {'positive ' if positive else ''}finite number in {context}")
    return number


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def read_panel(path: Path, scaler_path: Path) -> dict:
    report = json.loads(scaler_path.read_text(encoding="utf-8"))
    scaler = report.get("scaler", {})
    center = scaler.get("ratio_center")
    iqr = scaler.get("ratio_iqr")
    if not isinstance(center, list) or not isinstance(iqr, list) or len(center) != 5 or len(iqr) != 5:
        raise ValueError("Graph report has no five-dimensional ratio scaler")
    center = [_finite(v, "ratio_center") for v in center]
    iqr = [_finite(v, "ratio_iqr", positive=True) for v in iqr]

    required = {"entity_id", "territory_id", "period", "mo", "region_name", TOTAL, *CATEGORIES}
    rows = _read_csv(path)
    entities: dict[str, dict] = {}
    periods: set[str] = set()
    seen: set[tuple[str, str]] = set()
    for line, row in enumerate(rows, 2):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Panel missing columns: {', '.join(sorted(missing))}")
        entity_id, period = row["entity_id"].strip(), row["period"].strip()
        key = (entity_id, period)
        if not entity_id or key in seen:
            raise ValueError(f"Blank or duplicate panel key at line {line}: {key}")
        if entity_id != f"tid_{row['territory_id'].strip()}":
            raise ValueError(f"Canonical ID mismatch at line {line}")
        seen.add(key); periods.add(period)
        total = _finite(row[TOTAL], f"{key}/{TOTAL}", positive=True)
        values = [_finite(row[name], f"{key}/{name}", positive=True) for name in CATEGORIES]
        ratios = [value / total for value in values]
        features = [(math.log(ratio) - center[i]) / iqr[i] / math.sqrt(5) for i, ratio in enumerate(ratios)]
        entity = entities.setdefault(entity_id, {
            "id": entity_id, "tid": row["territory_id"].strip(), "name": row["mo"].strip(),
            "region": row["region_name"].strip(), "rows": {},
        })
        if entity["name"] != row["mo"].strip() or entity["region"] != row["region_name"].strip():
            raise ValueError(f"Entity metadata changes across panel: {entity_id}")
        entity["rows"][period] = {"total": total, "ratios": ratios, "features": features}
    periods_sorted = sorted(periods)
    if len(periods_sorted) != 24 or periods_sorted[0] != "2023-01-01" or periods_sorted[-1] != "2024-12-01":
        raise ValueError("Expected the complete January 2023–December 2024 panel")
    if len(entities) != 2016:
        raise ValueError(f"Expected 2016 territories, found {len(entities)}")
    for entity_id, entity in entities.items():
        if set(entity["rows"]) != set(periods_sorted):
            raise ValueError(f"Incomplete period join for {entity_id}")
    output = []
    for entity in sorted(entities.values(), key=lambda e: (e["name"].casefold(), e["region"].casefold(), e["id"])):
        ordered = [entity["rows"][period] for period in periods_sorted]
        reference = entity["rows"][PILOT_PERIOD]
        development = [entity["rows"][period] for period in periods_sorted if period < "2024-01-01"]
        output.append({
            "id": entity["id"], "tid": entity["tid"], "name": entity["name"], "region": entity["region"],
            "totals": [round(row["total"], 6) for row in ordered],
            "ratios": [[round(100 * value, 6) for value in row["ratios"]] for row in ordered],
            "annual_ratios": [round(100 * median(row["ratios"][i] for row in development), 6) for i in range(5)],
            "features": [round(value, 8) for value in reference["features"]],
            "annual_features": [round(median(row["features"][i] for row in development), 8) for i in range(5)],
        })
    return {"periods": periods_sorted, "entities": output, "scaler": scaler}


def read_pilot(partitions_path: Path, metrics_path: Path, entity_ids: set[str]) -> dict:
    chosen: dict[str, int] = {}
    for line, row in enumerate(_read_csv(partitions_path), 2):
        if row.get("method") == "kmeans" and row.get("period") == PILOT_PERIOD:
            entity_id = row.get("entity_id", "").strip()
            if entity_id in chosen:
                raise ValueError(f"Duplicate pilot assignment at line {line}: {entity_id}")
            chosen[entity_id] = int(row["cluster"])
    if set(chosen) != entity_ids:
        raise ValueError("Pilot assignments do not join exactly to the panel")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    matching = [row for row in metrics if row.get("method") == "kmeans" and row.get("period") == PILOT_PERIOD]
    if len(matching) != 1:
        raise ValueError("Expected one December 2023 KMeans metric record")
    metric = matching[0]
    for name in ("SW", "CH", "AVI", "AVU"):
        _finite(metric[name], f"pilot metric {name}")
    return {
        "period": PILOT_PERIOD, "method": "KMeans", "labels": chosen,
        "metric": {key: metric.get(key) for key in ("n", "k", "min_cluster_size", "SW", "CH", "AVI", "AVU", "S_Dbw", "S_Dbw_status", "MQ", "MQ_status")},
        "cluster_sizes": dict(sorted(Counter(chosen.values()).items())),
    }


def _require_columns(rows: list[dict], columns: set[str], name: str) -> None:
    if not rows or not columns.issubset(rows[0]):
        raise ValueError(f"{name} missing columns: {', '.join(sorted(columns.difference(rows[0] if rows else {})))}")


def read_validation(directory: Path, entity_ids: set[str], periods: set[str]) -> dict:
    missing = [name for name in VALIDATION_FILES if not (directory / name).is_file()]
    if missing:
        raise ValueError(f"Validation run is incomplete; missing: {', '.join(missing)}")
    frozen = json.loads((directory / "frozen_prototypes.json").read_text("utf-8"))
    ids = frozen.get("ids")
    if not isinstance(ids, list) or len(ids) != len(set(ids)) or set(ids) != entity_ids:
        raise ValueError("Frozen prototype IDs do not join exactly to the panel")
    labels = frozen.get("reference_labels")
    if not isinstance(labels, list) or len(labels) != len(ids):
        raise ValueError("Frozen reference labels are not aligned to IDs")
    centers = frozen.get("centers")
    if not centers or any(len(row) != 5 for row in centers):
        raise ValueError("Frozen prototype centers must be five-dimensional")
    for row in centers:
        for value in row: _finite(value, "prototype center")

    reference = _read_csv(directory / "reference_assignments.csv")
    _require_columns(reference, {"entity_id", "cluster"}, "reference_assignments.csv")
    ref_ids = [row["entity_id"] for row in reference]
    if len(ref_ids) != len(set(ref_ids)) or set(ref_ids) != entity_ids:
        raise ValueError("Reference assignments do not join exactly to panel IDs")
    reference_map = {row["entity_id"]: int(row["cluster"]) for row in reference}

    monthly = _read_csv(directory / "monthly_assignments.csv")
    _require_columns(monthly, {"entity_id", "period", "cluster", "prototype_margin"}, "monthly_assignments.csv")
    monthly_seen: set[tuple[str, str]] = set()
    monthly_map: dict[str, list[int]] = {entity_id: [] for entity_id in sorted(entity_ids)}
    by_key = {}
    for row in monthly:
        key = (row["entity_id"], row["period"])
        if key in monthly_seen or row["entity_id"] not in entity_ids or row["period"] not in periods:
            raise ValueError(f"Invalid monthly assignment key: {key}")
        monthly_seen.add(key); by_key[key] = int(row["cluster"])
        _finite(row["prototype_margin"], f"prototype margin {key}")
    expected = {(entity, period) for entity in entity_ids for period in periods}
    if monthly_seen != expected:
        raise ValueError("Monthly assignments are not a complete panel join")
    for entity_id in monthly_map:
        monthly_map[entity_id] = [by_key[(entity_id, period)] for period in sorted(periods)]

    neighbor_rows = _read_csv(directory / "consensus_neighbors.csv")
    _require_columns(neighbor_rows, {"entity_id", "neighbor_id", "rank", "month_edge_frequency", "profile_distance_2023", "distance_km"}, "consensus_neighbors.csv")
    neighbors: dict[str, list[dict]] = {entity_id: [] for entity_id in sorted(entity_ids)}
    neighbor_keys = set()
    for row in neighbor_rows:
        source, target = row["entity_id"], row["neighbor_id"]
        key = (source, int(row["rank"]))
        if source not in entity_ids or target not in entity_ids or source == target or key in neighbor_keys:
            raise ValueError(f"Invalid consensus neighbor row: {source}/{target}")
        neighbor_keys.add(key)
        neighbors[source].append({"id": target, "rank": int(row["rank"]),
            "frequency": _finite(row["month_edge_frequency"], "month edge frequency"),
            "profile_distance": _finite(row["profile_distance_2023"], "profile distance"),
            "distance_km": _finite(row["distance_km"], "distance km")})

    profiles = _read_csv(directory / "cluster_profiles.csv")
    _require_columns(profiles, {"cluster", "n", "category", "median_standardized_log_ratio", "q25", "q75"}, "cluster_profiles.csv")
    peer = json.loads((directory / "peer_validation.json").read_text("utf-8"))
    validate_neighbor_coverage(neighbors, peer)
    if reference_map != dict(zip(ids, labels)):
        raise ValueError("Reference assignments differ from frozen labels")
    market = json.loads((directory / "external_market_access.json").read_text("utf-8"))
    confirmation = json.loads((directory / "confirmation_metrics.json").read_text("utf-8"))
    if confirmation.get("candidate") != frozen.get("candidate"):
        raise ValueError("Confirmation candidate differs from frozen model")
    return {"confirmation": confirmation, "status": "validated", "candidate": frozen.get("candidate"), "assignment_rule": frozen.get("assignment_rule"),
            "prototype_fidelity": frozen.get("prototype_fidelity"), "labels": reference_map,
            "monthly": monthly_map, "neighbors": neighbors, "cluster_profiles": profiles,
            "peer_validation": peer, "external_market_access": market}


def validate_neighbor_coverage(neighbors, peer):
    active = {key for key, values in neighbors.items() if values}
    if len(active) != peer.get('n_common') or len(neighbors)-len(active) != peer.get('n_excluded'):
        raise ValueError('Peer coverage differs from validation cohort')
    for source, values in neighbors.items():
        values.sort(key=lambda row: row['rank'])
        if not values:
            continue
        targets = [row['id'] for row in values]
        if len(values)!=15 or [row['rank'] for row in values]!=list(range(1,16)) or len(set(targets))!=15:
            raise ValueError(f'Expected 15 distinct ranked consensus neighbors for {source}')
        if source in targets or not set(targets).issubset(active):
            raise ValueError('Consensus neighbor is outside the common cohort')


def read_reference(directory: Path, entity_ids: set[str]) -> dict:
    membership_path = directory / "membership_stability.json"
    labels_path = directory / "labels/kmeans_k4__reference.npy"
    published_labels = directory / 'reference_assignments.csv'
    if not membership_path.is_file() or not (labels_path.is_file() or published_labels.is_file()):
        raise ValueError("Reference run lacks membership IDs or frozen kmeans_k4 labels")
    membership = json.loads(membership_path.read_text("utf-8"))
    ids = membership.get("ids")
    if not isinstance(ids, list) or len(ids) != len(set(ids)) or set(ids) != entity_ids:
        raise ValueError("Reference IDs do not join exactly to the panel")
    if labels_path.is_file():
        labels = np.load(labels_path, allow_pickle=False)
    else:
        rows = _read_csv(published_labels)
        mapping = {row['entity_id']: int(row['cluster']) for row in rows}
        if len(rows) != len(mapping) or set(mapping) != entity_ids:
            raise ValueError('Published reference assignments do not join exactly')
        labels = np.asarray([mapping[key] for key in ids], dtype=int)
    if labels.ndim != 1 or len(labels) != len(ids) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Reference labels are not an aligned integer vector")
    unique = sorted(int(value) for value in np.unique(labels))
    if unique != list(range(4)):
        raise ValueError(f"Expected contiguous kmeans_k4 labels 0..3, found {unique}")
    metrics_path = directory / "reference_metrics.json"
    if not metrics_path.is_file():
        raise ValueError("Reference run lacks reference_metrics.json")
    metrics = json.loads(metrics_path.read_text("utf-8"))
    selected = [row for row in metrics if row.get("candidate") == "kmeans_k4"]
    if len(selected) != 1:
        raise ValueError("Expected one kmeans_k4 reference metric record")
    metric = selected[0]
    for name in ("SW", "CH", "AVI", "AVU"):
        _finite(metric[name], f"reference metric {name}")
    return {"candidate": "kmeans_k4", "groups": 4,
            "profile": "median monthly standardized log-ratio profile for 2023",
            "metric": {key: metric.get(key) for key in ("n", "k", "min_cluster_size", "SW", "CH", "AVI", "AVU", "S_Dbw", "MQ", "MQ_status", "negative_silhouette_fraction")},
            "labels": {entity_id: int(label) for entity_id, label in zip(ids, labels)}}

def read_stability(path: Path | None) -> list[dict] | None:
    if path is None:
        return None
    target = path / "stability.json" if path.is_dir() else path
    if not target.is_file():
        raise ValueError(f"Stability result not found: {target}")
    value = json.loads(target.read_text("utf-8"))
    if not isinstance(value, list) or not value:
        raise ValueError("Stability result must be a non-empty JSON array")
    for index, row in enumerate(value):
        if not isinstance(row, dict) or not {"candidate", "kind", "parameter", "ARI", "k"}.issubset(row):
            raise ValueError(f"Invalid stability row {index}")
        ari = _finite(row["ARI"], f"stability ARI row {index}")
        if not -1 <= ari <= 1:
            raise ValueError(f"Stability ARI outside [-1, 1] at row {index}")
    return value


def encode_payload(payload: dict, *, canonical: bool = True) -> str:
    """Serialize data safely for a classic HTML script element."""
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False, sort_keys=canonical)
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return encoded.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def package_payload(payload: dict, *, compressed: bool = True) -> str:
    """Deterministic offline transport; decompression does not change scientific data."""
    raw = encode_payload(payload)
    if not compressed:
        return raw
    packed = gzip.compress(raw.encode("utf-8"), compresslevel=6, mtime=0)
    # gzip's OS byte varies across Python/zlib platforms; fix it for byte reproducibility.
    packed = packed[:9] + b"\xff" + packed[10:]
    # Keep the public transport framing stable for existing offline readers.
    return encode_payload({"encoding": "gzip-base64", "data": base64.b64encode(packed).decode("ascii")}, canonical=False)


def read_contest(path: Path, entity_ids: set[str], map_path: Path | None = None) -> dict:
    target = path / 'atlas_extension.json' if path.is_dir() else path
    value = json.loads(target.read_text('utf-8'))
    for mode in ('raw','relative'):
        labels = value['assignments'][mode]
        if set(labels) != entity_ids or any(type(z) is not int or not 0 <= z < 4 for z in labels.values()):
            raise ValueError('Contest memberships do not join exactly')
        matrix = value['dynamics'][mode]['transition_matrix']
        if len(matrix) != 4 or any(len(row)!=4 for row in matrix):
            raise ValueError('Expected 4 by 4 transition matrix')
        if any(type(x) is not int or x<0 for row in matrix for x in row) or sum(map(sum,matrix))!=len(entity_ids):
            raise ValueError('Transition counts do not conserve the cohort')
    for candidate, labels in value.get('network',{}).get('labels',{}).items():
        if set(labels) != entity_ids or any(type(z) is not int or z<0 for z in labels.values()):
            raise ValueError('Network memberships do not join exactly: '+candidate)
    for name, model in value.get('map_models', {}).items():
        labels = model['labels']
        groups = {profile['id'] for profile in model['profiles']}
        if set(labels) != entity_ids or any(type(z) is not int or z not in groups for z in labels.values()):
            raise ValueError('Map model memberships do not join exactly: ' + name)
        if sum(profile['n'] for profile in model['profiles']) != len(entity_ids):
            raise ValueError('Map model sizes do not conserve the cohort: ' + name)
    if map_path:
        geometry = json.loads(map_path.read_text('utf-8'))
        keys = [p['id'] for p in geometry['paths']]
        if len(keys)!=len(set(keys)) or not set(keys).issubset(entity_ids):
            raise ValueError('Map has duplicate or unknown territories')
        if geometry['matched'] != len(keys):
            raise ValueError('Map coverage differs from actual paths')
        value['map'] = geometry
    encode_payload(value)  # Reject nonfinite values before touching the published HTML.
    return value

def render_atlas(template: str, payload: dict, *, compressed: bool = True,
                 split_assets: bool = False) -> tuple[str, dict[str, bytes]]:
    """Render a standalone document or an offline bundle from validated data."""
    return _render_packaged_atlas(template, payload, package_payload(payload, compressed=compressed),
                                  split_assets=split_assets)


def _render_packaged_atlas(template: str, payload: dict, encoded: str, *,
                           split_assets: bool = False) -> tuple[str, dict[str, bytes]]:
    """Render either a standalone document or an offline classic-script bundle."""
    marker = "/*__ATLAS_DATA__*/null"
    if template.count(marker) != 1:
        raise ValueError("Template must contain exactly one atlas data marker")
    assets = {}
    if split_assets:
        script_start = template.rfind("<script", 0, template.index(marker))
        if script_start < 0:
            raise ValueError("Atlas data marker must be inside a classic script element")
        script = ('"use strict";\nwindow.__SBERAI_ATLAS__={payload:' + encoded + '};\n').encode("utf-8")
        filename = "payload-" + hashlib.sha256(script).hexdigest() + ".js"
        assets[filename] = script
        # A synchronous classic script also works when opened directly with file://.
        template = (template[:script_start] + f'<script src="assets/{filename}"></script>\n'
                    + template[script_start:])
        encoded = ('(window.__SBERAI_ATLAS__?.payload??(()=>{throw new Error('
                   '"Не найден файл данных атласа. Откройте index.html вместе с каталогом assets.");})())')
    rendered = template.replace(marker, encoded)
    if "<!-- V12_SECTIONS -->" in rendered:
        sections = ""
        if payload.get("contest", {}).get("v12"):
            # The generated markup shares the lossless data package, keeping the page compact.
            sections = '<div id="v12-content"></div>'
        rendered = rendered.replace("<!-- V12_SECTIONS -->", sections)
    for placeholder, source in (
        ("/*__ATLAS_FONTS__*/", ROOT / "web/atlas_fonts.css"),
        ("/*__ATLAS_THEME__*/", ROOT / "web/atlas_theme.css"),
        ("/*__ATLAS_STORY__*/", ROOT / "web/atlas_story.js"),
        ("<!--__ACCEPTED_RESULTS__-->", ROOT / "web/accepted_results.html"),
        ("<!--__ARCHIVE_RESULTS__-->", ROOT / "web/archive_results.html"),
        ("<!-- PRACTICAL_CASES -->", ROOT / "web/practical_cases.html"),
        ("<!-- ICVI_RESULTS -->", ROOT / "web/icvi_results.html"),
        ("<!-- CLUSTER_STABILITY -->", ROOT / "web/cluster_stability.html"),
        ("<!--__FONT_LICENSES__-->", ROOT / "web/font_licenses.html"),
    ):
        if placeholder in rendered:
            content = source.read_text(encoding="utf-8")
            if placeholder == "/*__ATLAS_THEME__*/":
                content += "\n" + (ROOT / "web/atlas_map.css").read_text(encoding="utf-8")
                # Keep source formatting and comments; trim only line-edge whitespace in the bundle.
                content = "\n".join(line.strip() for line in content.splitlines()
                                    if line.strip())
            rendered = rendered.replace(placeholder, content)
    return rendered, assets


def _atomic_write(output: Path, content: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("wb", dir=output.parent,
                                         prefix=".atlas-", suffix=".tmp", delete=False) as stream:
            temporary = stream.name
            stream.write(content)
        os.replace(temporary, output)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def write_atlas(output: Path, rendered: str, assets: dict[str, bytes]) -> None:
    """Publish immutable assets first and atomically switch the HTML last."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if assets:
        asset_directory = output.parent / "assets"
        asset_directory.mkdir(exist_ok=True)
        for filename, content in assets.items():
            destination = asset_directory / filename
            if not destination.exists() or destination.read_bytes() != content:
                _atomic_write(destination, content)
    _atomic_write(output, rendered.encode("utf-8"))


def build(args: argparse.Namespace) -> None:
    standalone_output = getattr(args, "standalone_output", None)
    if standalone_output:
        if not getattr(args, "split_assets", False):
            raise ValueError("--standalone-output requires --split-assets")
        if standalone_output.resolve() == args.output.resolve():
            raise ValueError("--standalone-output must differ from --output")
    if args.validation_run and not args.reference_run:
        raise ValueError("--validation-run requires --reference-run for matching development metrics")
    panel = read_panel(args.panel, args.graph_report)
    ids = {entity["id"] for entity in panel["entities"]}
    pilot = read_pilot(args.partitions, args.metrics, ids)
    reference = read_reference(args.reference_run, ids) if args.reference_run else None
    validation = read_validation(args.validation_run, ids, set(panel["periods"])) if args.validation_run else None
    for entity in panel["entities"]:
        entity["pilot_cluster"] = pilot["labels"][entity["id"]]
        if reference:
            entity["annual_cluster"] = reference["labels"][entity["id"]]
        if reference or validation:
            entity["features"] = entity["annual_features"]
        if validation:
            entity["reference_cluster"] = validation["labels"][entity["id"]]
            entity["monthly_clusters"] = validation["monthly"][entity["id"]]
    status = "validation" if validation else "reference" if reference else "pilot"
    groups = len(set(validation["labels"].values())) if validation else reference["groups"] if reference else pilot["metric"]["k"]
    payload = {
        "meta": {"entities": len(panel["entities"]), "months": len(panel["periods"]),
                 "status": status, "groups": groups, "ratio_iqr": panel["scaler"]["ratio_iqr"], "reference_period": PILOT_PERIOD,
                 "profile_label": "годовой медианный профиль 2023" if reference or validation else "декабрь 2023"},
        "categories": list(CATEGORIES), "periods": panel["periods"], "entities": panel["entities"],
        "pilot": {key: value for key, value in pilot.items() if key != "labels"},
        "reference": {key: value for key, value in reference.items() if key != "labels"} if reference else None,
        "validation": validation, "stability": read_stability(args.stability_run or args.reference_run),
    }
    if getattr(args,'contest_run',None):
        if not validation:
            raise ValueError('--contest-run requires the frozen validation contract')
        payload['contest'] = read_contest(args.contest_run,ids,getattr(args,'map_data',None))
    if getattr(args, "v12", None):
        from scripts.extend_atlas_v12 import attach_v12
        attach_v12(payload, args.v12)
    template = args.template.read_text(encoding="utf-8")
    encoded = package_payload(payload, compressed=not getattr(args, "uncompressed", False))
    rendered, assets = _render_packaged_atlas(template, payload, encoded,
                                             split_assets=getattr(args, "split_assets", False))
    write_atlas(args.output, rendered, assets)
    if standalone_output:
        standalone, _ = _render_packaged_atlas(template, payload, encoded)
        write_atlas(standalone_output, standalone, {})
        print(f"Created {standalone_output} (standalone copy of the same data)")
    print(f"Created {args.output} ({len(panel['entities'])} territories, {len(panel['periods'])} months, {payload['meta']['status']})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=ROOT / "data/processed/panel.csv")
    parser.add_argument("--graph-report", type=Path, default=ROOT / "reports/graph_preparation.json")
    parser.add_argument("--partitions", type=Path, default=ROOT / "reports/experiments/2026-09-23/partitions.csv")
    parser.add_argument("--metrics", type=Path, default=ROOT / "reports/experiments/2026-09-23/metrics.json")
    parser.add_argument("--validation-run", type=Path, help="Directory containing the complete frozen validation contract")
    parser.add_argument("--reference-run", type=Path, help="Run directory with frozen annual kmeans_k4 labels and IDs")
    parser.add_argument("--stability-run", type=Path, help="Optional stability.json file or containing directory")
    parser.add_argument('--contest-run',type=Path,help='Validated contest extension directory or JSON file')
    parser.add_argument('--map-data',type=Path,help='Derived municipal SVG geometry JSON')
    parser.add_argument('--v12', type=Path, help='Optional validated six-attribute results directory')
    parser.add_argument("--template", type=Path, default=ROOT / "web/research_template.html")
    parser.add_argument("--uncompressed", action="store_true", help="Embed plain JSON instead of the default lossless gzip package")
    parser.add_argument("--split-assets", action="store_true", help="Write a small HTML with an adjacent offline assets directory")
    parser.add_argument("--standalone-output", type=Path,
                        help="Also write a standalone HTML from the same data; requires --split-assets")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/atlas-preview/index.html")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
