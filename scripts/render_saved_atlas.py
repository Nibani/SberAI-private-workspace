"""Render the current atlas from its saved data without rebuilding any model.

    python -m scripts.render_saved_atlas
    python -m scripts.render_saved_atlas --output preview/index.html --standalone
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.atlas_payload import read_atlas_payload
from scripts.build_research_atlas import ROOT, render_atlas, write_atlas
from scripts.build_v12_web import render_sections

PINNED_REPORTS = ("labels.csv", "methods.csv", "networks.csv", "summary.json",
                  "types.csv", "wage_level_comparison.json")


def check_report_sources(contest: dict) -> None:
    hashes = contest.get("source_hashes", {})
    for name in PINNED_REPORTS:
        relative = "reports/v1.2/" + name
        expected = hashes.get(relative)
        path = ROOT / relative
        if not expected or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Saved atlas report source is missing or has changed: {relative}")


def build(atlas: Path, output: Path, *, standalone: bool = False) -> dict:
    payload = read_atlas_payload(atlas)
    contest = payload.get("contest", {})
    if contest.get("v12"):
        check_report_sources(contest)
        contest["default_map_model"] = "v12_types"
        contest["v12"]["ui_fragment"] = render_sections(ROOT / "reports/v1.2")
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
