"""Attach the v1.2 typology to the published offline atlas without the private panel.

    python -m scripts.build_v12_web        # 1. static sections: reports/v1.2 -> web/v12_sections.html
    python -m scripts.extend_atlas_v12     # 2. v1.2 data + re-render: docs/index.html, docs/assets/

The current payload is read back from the offline bundle (``scripts/atlas_payload.py``).
``attach_v12`` adds the v1.2 map layers, per-territory v1.2 fields (type, 2024 type,
monthly types, six annual attributes and the 15 network neighbours) and type metadata,
replacing any previous v1.2 entries without touching v1.1 fields. The page is then
re-rendered with ``web/research_template.html``. Packaging is deterministic, so a second
run gives identical bytes; previous immutable payload files remain available.
``attach_v12`` is also used by ``scripts/build_research_atlas.py --v12 <dir>``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from sbercluster import edges as E  # noqa: E402
from sbercluster import panel as P  # noqa: E402
from sbercluster.typology import TYPE_COLORS, TYPE_NAMES, TYPE_ORDER, TYPE_SHORT  # noqa: E402

LAYER_PREFIX = "v12_"
DEFAULT_LAYER = "v12_types"
DEFAULT_TYPE = "remote"  # the page opens on the most populous municipality of this type
ENTITY_FIELDS = ("v12_type", "v12_type_2024", "v12_monthly", "v12_features", "v12_neighbors")
HASHED_FILES = ("labels.csv", "methods.csv", "networks.csv", "summary.json", "types.csv")
PAYLOAD_ASSET = re.compile(r"payload-[0-9a-f]{64}\.js")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _label(rows: list[dict[str, str]], key: str, value: str) -> str:
    matches = [row["label"] for row in rows if row[key] == value]
    if len(matches) != 1:
        raise ValueError(f"Expected one row with {key}={value!r}")
    return matches[0]


def _transitions(before: dict[str, int], after: dict[str, int]) -> list[list[int]]:
    matrix = [[0] * len(TYPE_ORDER) for _ in TYPE_ORDER]
    for entity_id, start in before.items():
        matrix[start][after[entity_id]] += 1
    return matrix


def _relative(path: Path) -> str | None:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return None


def _config(directory: Path) -> dict:
    provenance = directory / "provenance.json"
    if provenance.is_file():
        config = json.loads(provenance.read_text(encoding="utf-8")).get("config")
        if config:
            return config
    return json.loads((ROOT / "configs/v12.json").read_text(encoding="utf-8"))


def network_neighbors(summary: dict, config: dict) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Six v1.2 attributes of the 2023 annual profile and the k nearest municipalities.

    The same computation as the analogue check of ``scripts/run_v12.py``: exact Euclidean
    distances, the territory itself excluded, ties resolved by a stable sort in panel order.
    """
    path = ROOT / config["inputs"]["panel"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != summary["inputs"]["panel"]["sha256"]:
        raise ValueError(f"{path} differs from the panel used for the v1.2 results")
    panel = P.read_panel(path)
    scaler = P.fit_scaler(panel, config["features"]["development_months"])
    profile = P.annual_profile(P.monthly_attributes(panel, scaler), slice(0, 12))  # 2023, as in run_v12
    distance = E.pairwise_euclidean(profile)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argsort(distance, axis=1, kind="stable")[:, :int(config["practical"]["analogues"])]
    return [str(i) for i in panel.ids], profile, nearest


def attach_v12(payload: dict, input_dir: Path | str) -> dict:
    """Add the v1.2 layers and per-territory fields in place (previous v1.2 entries are replaced)."""
    directory = Path(input_dir)
    contest = payload.get("contest")
    if not isinstance(contest, dict):
        raise ValueError("Atlas payload has no contest section to extend")
    ids = [entity["id"] for entity in payload["entities"]]
    rows = _rows(directory / "labels.csv")
    by_id = {row["entity_id"]: row for row in rows}
    if len(by_id) != len(rows) or set(by_id) != set(ids):
        raise ValueError("v1.2 labels do not join exactly to the atlas territories")
    allowed = {str(i) for i in range(len(TYPE_ORDER))}

    def column(name: str) -> dict[str, int]:
        if any(by_id[entity_id][name] not in allowed for entity_id in ids):
            raise ValueError(f"labels.csv {name} must hold types 0..{len(TYPE_ORDER) - 1}")
        return {entity_id: int(by_id[entity_id][name]) for entity_id in ids}

    first, adjusted, absolute = column("type_2023"), column("type_2024"), column("type_2024_absolute")
    monthly = {entity_id: by_id[entity_id]["monthly_types"] for entity_id in ids}
    if any(len(value) != len(payload["periods"]) or set(value) - allowed for value in monthly.values()):
        raise ValueError("labels.csv monthly_types must hold one type digit per atlas month")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    tracking = summary["tracking"]
    for key, values in (("annual_transitions", adjusted), ("annual_transitions_absolute", absolute)):
        if _transitions(first, values) != tracking[key]:
            raise ValueError(f"summary.json tracking.{key} differs from labels.csv")
    config = _config(directory)
    method = _label(_rows(directory / "methods.csv"), "method", summary["methods"]["chosen"])
    rule = _label(_rows(directory / "networks.csv"), "rule", summary["network"]["chosen_rule"])
    colors = [TYPE_COLORS[key] for key in TYPE_ORDER]

    panel_ids, profile, nearest = network_neighbors(summary, config)
    if set(panel_ids) != set(ids):
        raise ValueError("The v1.2 panel does not join exactly to the atlas territories")
    row_of = {entity_id: row for row, entity_id in enumerate(panel_ids)}
    for entity in payload["entities"]:
        entity_id, row = entity["id"], row_of[entity["id"]]
        entity["v12_type"] = first[entity_id]
        entity["v12_type_2024"] = adjusted[entity_id]
        entity["v12_monthly"] = [int(value) for value in monthly[entity_id]]
        # Browser coordinates ignore libm's platform-dependent final bits. Neighbour IDs
        # above still come from the full-precision scientific calculation.
        entity["v12_features"] = [round(float(value), 10) for value in profile[row]]
        entity["v12_neighbors"] = [panel_ids[j] for j in nearest[row]]

    def profiles(values: dict[str, int]) -> list[dict]:
        sizes = [0] * len(TYPE_ORDER)
        for value in values.values():
            sizes[value] += 1
        return [{"id": i, "name": TYPE_NAMES[key], "n": sizes[i]} for i, key in enumerate(TYPE_ORDER)]

    def changed(values: dict[str, int]) -> int:
        return sum(values[entity_id] != first[entity_id] for entity_id in ids)

    layers = {
        "v12_types": {
            "name": "Структура и уровень расходов · 6 признаков", "labels": first, "profiles": profiles(first), "colors": colors,
            "labels_2024": {"raw": absolute, "relative": adjusted},
            "dynamics": {"raw": {"changed": changed(absolute)}, "relative": {"changed": changed(adjusted)}},
            "mode_labels": {"raw": "Без поправки", "relative": "С поправкой на общую волну"},
            "note": (f"«{method}» на атрибутированной сети (структура расходов по категориям и их уровень, "
                     f"медианы 2023 года; рёбра — «{rule}»). Год 2024: годовой профиль 2024 года отнесён к ближайшему "
                     "центру типов 2023 года без поправки или после вычета общей помесячной волны. Цвета закреплены "
                     "за профилями. 2024 год участвовал в выборе модели; это ретроспективная проверка."),
        },
        "v12_types_2024": {
            "name": "Структура и уровень · 2024 с поправкой", "labels": adjusted, "profiles": profiles(adjusted),
            "colors": colors,
            "note": ("Годовой профиль 2024 года после вычета общей помесячной волны отнесён к ближайшему центру типов "
                     "2023 года. Переходы показаны в разделе «Как меняются профили». "
                     "2024 год участвовал в выборе модели."),
        },
    }
    others = {key: value for key, value in contest.get("map_models", {}).items() if not key.startswith(LAYER_PREFIX)}
    contest["map_models"] = {**others, **layers}

    types = {int(row["type"]): row for row in _rows(directory / "types.csv")}
    largest = types[TYPE_ORDER.index(DEFAULT_TYPE)]["largest_by_population"].split(";")[0].strip()
    keys = {f"{row['name']} ({row['region']})": row["entity_id"] for row in rows}
    if largest not in keys:
        raise ValueError(f"Default territory {largest!r} is not in labels.csv")
    contest["v12"] = {
        "types": [{"id": i, "key": key, "name": TYPE_NAMES[key], "short": TYPE_SHORT[key], "color": TYPE_COLORS[key]}
                  for i, key in enumerate(TYPE_ORDER)],
        "attributes": list(config["features"]["attributes"]),
        "neighbors": int(config["practical"]["analogues"]),
        "default_entity": keys[largest],
    }
    from scripts.build_v12_web import render_sections
    contest["v12"]["ui_fragment"] = render_sections(directory)
    contest["default_map_model"] = DEFAULT_LAYER
    relative = _relative(directory)
    if relative is not None:
        hashes = contest.setdefault("source_hashes", {})
        for name in HASHED_FILES:
            hashes[f"{relative}/{name}"] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        wage = directory / "wage_level_comparison.json"
        if wage.is_file():
            hashes[f"{relative}/{wage.name}"] = hashlib.sha256(wage.read_bytes()).hexdigest()
        hashes[config["inputs"]["panel"]] = summary["inputs"]["panel"]["sha256"]
        contest["source_hashes"] = dict(sorted(hashes.items()))
    return payload


def remove_stale_assets(directory: Path, keep: set[str]) -> list[str]:
    """Delete hash-named payload scripts that the rendered page no longer references."""
    removed = []
    if directory.is_dir():
        for path in sorted(directory.iterdir()):
            if PAYLOAD_ASSET.fullmatch(path.name) and path.name not in keep:
                path.unlink()
                removed.append(path.name)
    return removed


def build(atlas: Path, input_dir: Path, template: Path, output: Path) -> dict:
    from scripts.atlas_payload import read_atlas_payload
    from scripts.build_research_atlas import render_atlas, write_atlas

    payload = attach_v12(read_atlas_payload(atlas), input_dir)
    rendered, assets = render_atlas(template.read_text(encoding="utf-8"), payload, split_assets=True)
    write_atlas(output, rendered, assets)
    for path in [output, *(output.parent / "assets" / name for name in assets)]:
        os.chmod(path, 0o644)  # the atomic writer creates private temporary files
    return {"output": str(output), "assets": sorted(assets)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "reports/v1.2", help="v1.2 results directory")
    parser.add_argument("--atlas", type=Path, default=ROOT / "docs/index.html", help="offline atlas to read the payload from")
    parser.add_argument("--output", type=Path, help="page to write (default: --atlas)")
    parser.add_argument("--template", type=Path, default=ROOT / "web/research_template.html")
    args = parser.parse_args()
    from scripts.build_v12_web import render_sections

    sections = ROOT / "web/v12_sections.html"
    if not sections.is_file() or sections.read_text(encoding="utf-8") != render_sections(args.input):
        raise SystemExit(f"{sections.relative_to(ROOT)} is missing or out of date for {args.input}; "
                         f"run: python -m scripts.build_v12_web --input {args.input}")
    print(json.dumps(build(args.atlas, args.input, args.template, args.output or args.atlas), ensure_ascii=False))


if __name__ == "__main__":
    main()
