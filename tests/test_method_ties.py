"""Checks for the robustness of the method choice to the composition of territories."""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import check_method_ties as MT

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports/v1.2-method-ties"


class SelectionRuleTests(unittest.TestCase):
    def table(self, values):
        rows = []
        for method, v in values.items():
            rows.append({"method": method, **{f"{c}_{y}": v * s for c, s in MT.A.ICVI for y in (2023, 2024)}})
        return pd.DataFrame(rows)

    def test_rule_picks_dominant_method_and_breaks_ties_by_simplicity(self):
        eligible = ["kmeans", "joint_0.5", "joint_0.1"]
        self.assertEqual(MT.select(self.table({"kmeans": 1.0, "joint_0.5": 2.0, "joint_0.1": 1.5}), eligible),
                         ("joint_0.5", "joint_0.5"))
        tie = self.table({"kmeans": 1.0, "joint_0.5": 1.0, "joint_0.1": 1.0})
        self.assertEqual(MT.select(tie, eligible), ("kmeans", "kmeans"))

    def test_ineligible_methods_do_not_enter_the_rule(self):
        t = self.table({"kmeans": 1.0, "joint_0.5": 2.0, "leiden_graph": 9.0})
        self.assertEqual(MT.select(t, ["kmeans", "joint_0.5"])[0], "joint_0.5")

    def test_differing_counts_after_matching_groups(self):
        a = np.array([0, 0, 1, 1, 2, 2])
        self.assertEqual(MT._differing(a, np.array([2, 2, 0, 0, 1, 1])), 0)
        self.assertEqual(MT._differing(a, np.array([2, 2, 0, 1, 1, 1])), 1)


@unittest.skipUnless((RESULTS / "win_frequency.csv").exists(), "method-tie tables are not generated")
class SavedResultTests(unittest.TestCase):
    def test_saved_tables_are_consistent(self):
        freq = pd.read_csv(RESULTS / "win_frequency.csv")
        draws = pd.read_csv(RESULTS / "draws.csv")
        summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["chosen_full"], MT.SELECTED)
        self.assertEqual(len(draws), summary["draws"])
        self.assertAlmostEqual(freq.win_share_2023_2024.sum(), 1.0)
        self.assertAlmostEqual(freq.win_share_2023.sum(), 1.0)
        for r in freq.itertuples():
            self.assertAlmostEqual(r.win_share_2023_2024, (draws.winner_2023_2024 == r.method).mean())
        self.assertEqual(freq.set_index("method").loc[MT.SELECTED, "municipalities_differing_from_selected"], 0)
        pairwise = pd.read_csv(RESULTS / "pairwise_icvi.csv")
        self.assertEqual(len(pairwise), 9 * 6 * 2)
        self.assertTrue((pairwise.subsample_p2_5 <= pairwise.subsample_p97_5).all())
        # The README states a stable trade-off against KMeans; keep that sentence tied to the table.
        km = pairwise[(pairwise.method == "kmeans") & (pairwise.year == 2023)].set_index("icvi").selected_better_share
        self.assertEqual((km["SW"], km["AVI"], km["CH"]), (1.0, 1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
