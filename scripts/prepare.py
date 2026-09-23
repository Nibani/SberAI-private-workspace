"""Deterministic data preparation. Does not import or call model fitting."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sbercluster.canonical import prepare
from sbercluster.data import audit_and_prepare

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    inputs = list((root / "data/raw").glob("*.csv"))
    if len(inputs) > 1:
        raise SystemExit("Expected zero or one CSV reference")
    if inputs:
        audit_and_prepare(inputs[0], root)
    result = prepare(root)
    print(f"Prepared {result['complete_panel_territories']} territories x {result['n_months']} months. No fitting performed.")
