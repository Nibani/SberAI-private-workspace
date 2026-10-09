"""Render the current atlas from its saved data without rebuilding any model.

    python -m scripts.render_saved_atlas
    python -m scripts.render_saved_atlas --output preview/index.html --standalone
"""
from __future__ import annotations

import argparse
import hashlib
import json
import gzip
from pathlib import Path

from scripts.atlas_payload import read_atlas_payload
from scripts.build_research_atlas import ROOT, render_atlas, write_atlas
from scripts.build_v12_web import render_sections
from scripts.logical_storage import LogicalStorage

BASE_INPUT = "data/atlas-base.json.gz"
BASE_SHA256 = "a9cdfe5ab3d9044446cd93d3a30c354244c90b7cf3cc2ea8c40e3fbd3bdf8dd9"

PINNED_REPORTS = ("labels.csv", "methods.csv", "networks.csv", "summary.json",
                  "types.csv", "wage_level_comparison.json")


def check_report_sources(contest: dict) -> None:
    hashes = contest.get("source_hashes", {})
    storage = LogicalStorage(ROOT)
    for name in PINNED_REPORTS:
        relative = "reports/v1.2/" + name
        expected = hashes.get(relative)
        if not expected or not storage.exists(relative) or hashlib.sha256(storage.read_bytes(relative)).hexdigest() != expected:
            raise ValueError(f"Saved atlas report source is missing or has changed: {relative}")


def build(atlas: Path, output: Path, *, standalone: bool = False) -> dict:
    base = ROOT / BASE_INPUT
    if atlas.resolve() == (ROOT / "docs/index.html").resolve() and base.is_file():
        raw = base.read_bytes()
        if hashlib.sha256(raw).hexdigest() != BASE_SHA256:
            raise ValueError("Frozen base atlas input changed")
        payload = json.loads(gzip.decompress(raw))
    else:
        payload = read_atlas_payload(atlas)
    contest = payload.get("contest", {})
    if contest.get("v12"):
        check_report_sources(contest)
        contest["default_map_model"] = "v12_types"
        contest["v12"]["ui_fragment"] = render_sections(ROOT / "reports/v1.2")
    adapter = ROOT / "scripts/attach_current_findings.py"
    if adapter.is_file():
        from scripts.attach_current_findings import attach
        payload = attach(payload)
    html, assets = render_atlas((ROOT / "web/research_template.html").read_text(encoding="utf-8"),
                                payload, split_assets=not standalone)
    write_atlas(output, html, assets)
    return {"output": str(output), "assets": sorted(assets), "standalone": standalone}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--standalone", action="store_true")
    args = parser.parse_args()
    atlas = ROOT / "docs/index.html"
    print(json.dumps(build(atlas, args.output or atlas, standalone=args.standalone),
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
