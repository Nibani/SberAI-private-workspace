"""Independent arithmetic and leakage checks for the corrected findings."""
import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from sbercluster import panel as P
from sbercluster import typology as TY

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports/v1.2"


class FrozenLevelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = P.read_panel(ROOT / "data/v12/panel.csv.gz")

    def test_heldout_transform_keeps_training_reference_when_cohort_changes(self):
        train = replace(self.panel, totals=self.panel.totals[10:, :12], shares=self.panel.shares[10:, :12])
        scaler = P.fit_scaler(train)
        heldout = replace(self.panel, totals=self.panel.totals[:2, :12], shares=self.panel.shares[:2, :12])
        before = P.monthly_attributes(heldout, scaler)
        shifted = heldout.totals.copy()
        shifted[1] *= 1000
        after = P.monthly_attributes(replace(heldout, totals=shifted), scaler)
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_allclose(after[1, :, 5] - before[1, :, 5], np.log(1000) / scaler.scale[5])
        np.testing.assert_allclose(before[:, :, 5],
                                   (np.log(heldout.totals) - np.median(np.log(train.totals), axis=0)) / scaler.scale[5])

    def test_legacy_scaler_without_training_reference_is_rejected(self):
        fitted = P.fit_scaler(self.panel)
        with self.assertRaisesRegex(ValueError, "training level reference"):
            P.monthly_attributes(self.panel, replace(fitted, level_center=None))


class FindingArithmeticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = P.read_panel(ROOT / "data/v12/panel.csv.gz")
        cls.summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))

    def test_percentage_point_errors_are_arithmetic_growth(self):
        frame = pd.read_csv(RESULTS / "analogue_predictions.csv")
        expected = 100 * (self.panel.totals[:, 12:].mean(axis=1) / self.panel.totals[:, :12].mean(axis=1) - 1)
        by_id = dict(zip(self.panel.ids, expected))
        np.testing.assert_allclose(frame.observed_growth_pct, frame.entity_id.map(by_id), rtol=1e-5)
        np.testing.assert_allclose(frame.absolute_error_pp,
                                   np.abs(frame.predicted_growth_pct - frame.entity_id.map(by_id)), atol=1e-4)
        table = pd.read_csv(RESULTS / "analogues.csv")
        self.assertEqual(set(table.unit), {"percentage_point"})
        for row in table.itertuples():
            selected = frame[(frame.predictor == row.predictor) & frame.common_sample]
            self.assertEqual(row.n, len(selected))
            self.assertAlmostEqual(row.mae_pp, selected.absolute_error_pp.mean(), places=4)

    def test_marketplace_gap_uses_fixed_type_medians(self):
        labels = pd.read_csv(RESULTS / "labels.csv").type_2023.to_numpy()
        g = self.summary["practical"]["marketplace_gap_pp"]
        for year, months in ((2023, slice(0, 12)), (2024, slice(12, 24))):
            shares = 100 * np.median(self.panel.shares[:, months, 1], axis=1)
            expected = np.median(shares[labels == 2]) - np.median(shares[labels == 3])
            self.assertAlmostEqual(g[f"gap_{year}_pp"], expected)
        self.assertAlmostEqual(g["change_pp"], g["gap_2024_pp"] - g["gap_2023_pp"])

    def test_strict_metrics_reproduce_from_saved_predictions(self):
        predictions = pd.read_csv(RESULTS / "strict-region-validation/strict_predictions.csv")
        table = pd.read_csv(RESULTS / "leave_one_region_out.csv")
        for row in table.itertuples():
            part = predictions[predictions.indicator == row.indicator]
            self.assertEqual(row.n, len(part))
            self.assertEqual(row.regions, part.region.nunique())
            tss = np.square(part.observed - part.observed.mean()).sum()
            for metric, column in (("types", "type_prediction"), ("municipal_type", "admin_prediction"),
                                   ("municipal_type_original_zero", "admin_zero_prediction")):
                expected = 1 - np.square(part.observed - part[column]).sum() / tss
                self.assertAlmostEqual(getattr(row, metric), expected, places=12)

    def test_unseen_administrative_categories_use_training_mean(self):
        from scripts.validate_v12_findings import read_external
        cfg = json.loads((ROOT / "configs/v12.json").read_text(encoding="utf-8"))
        external, _ = read_external(ROOT, self.panel.ids, cfg)
        predictions = pd.read_csv(RESULTS / "strict-region-validation/strict_predictions.csv")
        unseen = predictions[predictions.unseen_admin_category]
        self.assertGreater(len(unseen), 0)
        for row in unseen.itertuples():
            values = external[row.indicator].to_numpy(float)
            training = values[(self.panel.regions != row.region) & np.isfinite(values)]
            self.assertAlmostEqual(row.admin_prediction, training.mean(), places=12)

    def test_names_and_validation_are_consistent(self):
        profiles = pd.read_csv(RESULTS / "types.csv")
        for row in profiles.itertuples():
            self.assertEqual(row.name, TY.TYPE_NAMES[row.key])
        self.assertEqual(self.summary["methods"]["chosen"], "joint_0.5")
        self.assertIn("выборе", self.summary["validation"]["year_2024"])
        self.assertIn("не является", self.summary["validation"]["year_2024"])


if __name__ == "__main__":
    unittest.main()
