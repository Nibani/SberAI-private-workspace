"""Export the monthly panel stored in the published atlas as a tidy, hash-pinned CSV.

The atlas bundle already redistributes every value used by version 1.2: monthly
totals in rubles and the five category ratios (percent of the total, six decimals)
for 2 016 municipalities and 24 months. The CSV makes this input explicit, so the
whole v1.2 analysis runs from the repository without downloads. Values are copied
verbatim; nothing is refitted or rounded again.

    python -m scripts.export_panel --output data/v12/panel.csv.gz
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path

from scripts.atlas_payload import read_atlas_payload

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = ("entity_id", "territory_id", "name", "region", "period", "total_rub",
           "health_pct", "marketplaces_pct", "horeca_pct", "food_pct", "transport_pct")
ATLAS_CATEGORIES = ["Здоровье", "Маркетплейсы", "Общественное питание", "Продовольствие", "Транспорт"]


def panel_rows(payload: dict) -> list[list[str]]:
    if payload.get("categories") != ATLAS_CATEGORIES:
        raise ValueError("Unexpected atlas category order")
    periods = payload["periods"]
    if len(periods) != 24 or periods[0] != "2023-01-01" or periods[-1] != "2024-12-01":
        raise ValueError("Expected the complete January 2023 - December 2024 panel")
    rows = []
    for entity in sorted(payload["entities"], key=lambda e: e["id"]):
        if entity["id"] != f"tid_{entity['tid']}":
            raise ValueError(f"Canonical ID mismatch: {entity['id']}")
        if len(entity["totals"]) != 24 or len(entity["ratios"]) != 24:
            raise ValueError(f"Incomplete series: {entity['id']}")
        for period, total, ratios in zip(periods, entity["totals"], entity["ratios"]):
            if total <= 0 or len(ratios) != 5 or min(ratios) <= 0:
                raise ValueError(f"Non-positive value: {entity['id']} {period}")
            rows.append([entity["id"], entity["tid"], entity["name"], entity["region"], period,
                         repr(float(total)), *[repr(float(r)) for r in ratios]])
    if len(rows) != 2016 * 24:
        raise ValueError(f"Expected 48 384 rows, found {len(rows)}")
    return rows


def write_deterministic_gzip(rows: list[list[str]], output: Path) -> str:
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(COLUMNS)
    writer.writerows(rows)
    raw = text.getvalue().encode("utf-8")
    packed = gzip.compress(raw, compresslevel=9, mtime=0)
    packed = packed[:9] + b"\xff" + packed[10:]          # fixed OS byte for byte reproducibility
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(packed)
    return hashlib.sha256(packed).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--atlas", type=Path, default=ROOT / "docs/index.html")
    parser.add_argument("--output", type=Path, default=ROOT / "data/v12/panel.csv.gz")
    args = parser.parse_args()
    payload = read_atlas_payload(args.atlas)
    digest = write_deterministic_gzip(panel_rows(payload), args.output)
    manifest = {
        "file": args.output.name, "sha256": digest, "rows": 2016 * 24, "entities": 2016, "months": 24,
        "columns": list(COLUMNS),
        "source": "СберИндекс, потребительские безналичные расходы по МО (CC BY-SA 4.0); значения скопированы "
                  "из опубликованного пакета атласа без изменений",
        "derivation": "python -m scripts.export_panel; совпадает для атласа v1.1 и v1.2, поскольку ряды МО не менялись",
        "units": {"total_rub": "модельная оценка средних месячных безналичных расходов жителя МО, руб.",
                  "*_pct": "отношение категории к общему показателю, %; категории не обязаны давать 100%"},
    }
    (args.output.parent / "panel.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": digest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
