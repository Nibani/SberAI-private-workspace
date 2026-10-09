"""Verify saved method-choice arithmetic; does not refit partitions or ICVI."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from scripts.logical_storage import LogicalStorage

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    store = LogicalStorage(root)
    base = "reports/v1.2-method-ties/"
    summary = store.read_json(base + "summary.json")
    for name, digest in summary["inputs_sha256"].items():
        if hashlib.sha256(store.read_bytes(name)).hexdigest() != digest:
            raise ValueError(f"Method-ties input fingerprint differs: {name}")
    import io
    draws = pd.read_csv(io.BytesIO(store.read_bytes(base + "draws.csv")))
    frequency = pd.read_csv(io.BytesIO(store.read_bytes(base + "win_frequency.csv")))
    pairwise = pd.read_csv(io.BytesIO(store.read_bytes(base + "pairwise_icvi.csv")))
    if len(draws) != summary["draws"] or draws.draw.tolist() != list(range(summary["draws"])):
        raise ValueError("Incomplete/duplicate method-tie draws")
    eligible = set(summary["eligible"])
    if set(frequency.method) != eligible or frequency.method.duplicated().any():
        raise ValueError("Frequency methods differ from registered saved methods")
    for suffix in ("2023_2024", "2023"):
        winners = draws["winner_" + suffix]
        if not set(winners).issubset(eligible):
            raise ValueError("Ineligible winner in method-tie draws")
        expected = [(winners == method).mean() for method in frequency.method]
        np.testing.assert_allclose(frequency["win_share_" + suffix], expected, rtol=0, atol=1e-10)
    if len(pairwise) != 9 * 6 * 2 or pairwise.duplicated(["method", "icvi", "year"]).any():
        raise ValueError("Incomplete/duplicate pairwise ICVI table")
    if not (pairwise.subsample_p2_5 <= pairwise.subsample_p97_5).all():
        raise ValueError("Reversed ICVI sensitivity interval")
    selected = frequency.set_index("method").loc[summary["chosen_full"]]
    if selected.municipalities_differing_from_selected != 0:
        raise ValueError("Selected partition differs from itself")
    return {"status": "PASS", "scope": "All saved draw winner counts, input fingerprints and pairwise table contracts; no model/ICVI refit",
            "draws": len(draws), "chosen_full": summary["chosen_full"],
            "selected_win_share_2023_2024": float(selected.win_share_2023_2024),
            "selected_win_share_2023": float(selected.win_share_2023)}


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False))
