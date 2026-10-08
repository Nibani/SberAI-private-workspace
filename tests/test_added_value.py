"""Checks for the added-value comparison of network groups against the spending level."""
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import check_added_value as AV

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports/v1.2-added-value"
STRICT = ROOT / "reports/v1.2/strict-region-validation"


class ModelTests(unittest.TestCase):
    def test_restricted_spline_is_linear_outside_outer_knots(self):
        knots = np.array([-1.0, -0.2, 0.3, 1.2])
        x = np.linspace(-4, 5, 4001)
        basis = AV.rcs_basis(x, knots)
        self.assertEqual(basis.shape, (len(x), 3))
        for side in (x < knots[0], x > knots[-1]):
            np.testing.assert_allclose(np.diff(basis[side], n=2, axis=0), 0, atol=1e-9)
        with self.assertRaises(ValueError):
            AV.rcs_basis(x, np.array([0.0, 0.0, 1.0]))

    def test_design_blocks(self):
        rng = np.random.default_rng(3)
        level, shares, groups = rng.normal(size=20), rng.normal(size=(20, 5)), np.arange(20) % 4
        knots = np.quantile(level, AV.QUANTILES)
        widths = {m: AV.design(m, level, shares, groups, knots).shape[1] for m in AV.MODELS}
        self.assertEqual(widths, {"groups": 4, "level": 2, "level_spline": 4, "level_groups": 5,
                                  "spline_groups": 7, "level_shares": 7, "spline_shares": 9,
                                  "spline_shares_groups": 12})


class InferenceTests(unittest.TestCase):
    def test_holm_matches_step_down_definition(self):
        np.testing.assert_allclose(AV.holm([0.01, 0.04, 0.03, 0.5]), [0.04, 0.09, 0.09, 0.5])
        np.testing.assert_allclose(AV.holm([0.6, 0.6]), [1.0, 1.0])

    def test_resampled_sums_count_repeated_regions(self):
        codes = np.array([0, 0, 1, 2, 2, 2])
        values = np.array([1.0, 2.0, 10.0, 0.5, 0.5, 0.5])
        picks = np.array([[0, 0, 2], [1, 1, 1], [0, 1, 2]])
        np.testing.assert_allclose(AV.resampled_sums(values, codes, 3, picks), [7.5, 30.0, 14.5])
        np.testing.assert_allclose(AV.resampled_sums(np.ones(6), codes, 3, picks), [7, 3, 6])

    def test_percentile_p_formula_and_bounds(self):
        b = 999
        self.assertAlmostEqual(AV.bootstrap_p(np.full(b, 0.3)), 2 / (b + 1))
        self.assertAlmostEqual(AV.bootstrap_p(np.full(b, -0.3)), 2 / (b + 1))
        self.assertEqual(AV.bootstrap_p(np.zeros(b)), 1.0)          # zeros count in both tails
        samples = np.r_[np.full(30, -1.0), np.full(969, 1.0)]
        self.assertAlmostEqual(AV.bootstrap_p(samples), 2 * 31 / 1000)

    def test_studentized_se_formula_and_degenerate_case(self):
        rng = np.random.default_rng(5)
        codes = np.repeat(np.arange(6), [3, 5, 2, 4, 6, 1])
        d = rng.normal(size=len(codes))
        picks = rng.integers(0, 6, size=(500, 6))
        st = AV.studentized(d, codes, 6, picks)
        s_r = np.bincount(codes, weights=d)
        n_r = np.bincount(codes).astype(float)
        m = s_r.sum() / n_r.sum()
        se = np.sqrt(6 / 5 * np.sum((s_r - m * n_r) ** 2)) / n_r.sum()
        self.assertAlmostEqual(st["mean"], d.mean())
        self.assertAlmostEqual(st["se"], se)
        # per-draw se agrees with a direct computation on the drawn regions
        draw = picks[0]
        n_b, s_b = n_r[draw], s_r[draw]
        m_b = s_b.sum() / n_b.sum()
        se_b = np.sqrt(6 / 5 * np.sum((s_b - m_b * n_b) ** 2)) / n_b.sum()
        self.assertAlmostEqual(st["abs_t"][0], abs((m_b - m) / se_b))
        same = AV.studentized(np.zeros(len(codes)), codes, 6, picks)
        self.assertEqual((same["t"], same["p"]), (0.0, 1.0))

    def test_identity_draw_reproduces_point_estimate(self):
        rng = np.random.default_rng(8)
        regions = np.repeat(np.array(list("abcdefgh")), 7)
        y = rng.normal(size=len(regions))
        cand, base = y + rng.normal(scale=.5, size=len(y)), y + rng.normal(scale=.7, size=len(y))
        out = AV.bootstrap_contrast(y, cand, base, regions, np.tile(np.arange(8), (5, 1)))
        self.assertAlmostEqual(out["delta_r2_pct_ci_low"], out["delta_r2"])
        self.assertAlmostEqual(out["delta_r2_pct_ci_high"], out["delta_r2"])
        self.assertAlmostEqual(out["delta_r2"], (1 - np.mean((cand - y) ** 2) / y.var()) - (1 - np.mean((base - y) ** 2) / y.var()))

    def test_p_and_interval_agree_on_controlled_examples(self):
        rng = np.random.default_rng(11)
        regions = np.repeat(np.arange(40), 12)
        y = rng.normal(size=len(regions))
        for shift in (0.0, 0.05, 0.1, 0.2, 0.4):
            cand = y + rng.normal(scale=.6, size=len(y))
            base = y + rng.normal(scale=np.sqrt(.36 + shift), size=len(y))
            out = AV.bootstrap_contrast(y, cand, base, regions, AV.region_picks(40, 4000, 3))
            excludes = out["delta_r2_ci_low"] > 0 or out["delta_r2_ci_high"] < 0
            if abs(out["p_boot_t"] - 0.05) > 0.003:
                self.assertEqual(out["p_boot_t"] < 0.05, excludes, shift)
            simultaneous = out["delta_r2_simultaneous_low"] > 0 or out["delta_r2_simultaneous_high"] < 0
            if abs(out["p_boot_t"] - 0.05 / 8) > 0.001:
                self.assertEqual(out["p_boot_t"] < 0.05 / 8, simultaneous, shift)

    def test_region_influence_drops_whole_regions(self):
        rng = np.random.default_rng(2)
        regions = np.repeat(np.array(list("abcde")), [3, 2, 4, 5, 3])
        y = rng.normal(size=len(regions))
        cand, base = y + rng.normal(size=len(y)), y + rng.normal(size=len(y))
        out = AV.region_influence(y, cand, base, regions)
        full = -np.mean((cand - y) ** 2 - (base - y) ** 2) / y.var()
        drops = {r: -np.mean((cand - y)[regions != r] ** 2 - (base - y)[regions != r] ** 2) / y[regions != r].var()
                 for r in "abcde"}
        top = max(drops, key=lambda r: abs(drops[r] - full))
        self.assertEqual(out["most_influential_region"], top)
        self.assertAlmostEqual(out["delta_r2_without_it"], drops[top])
        self.assertAlmostEqual(out["drop_one_region_min"], min(drops.values()))
        self.assertAlmostEqual(out["drop_one_region_max"], max(drops.values()))


@unittest.skipUnless((RESULTS / "contrasts.csv").exists(), "added-value tables are not generated")
class SavedResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = pd.read_csv(RESULTS / "models.csv")
        cls.contrasts = pd.read_csv(RESULTS / "contrasts.csv")
        cls.pred = pd.read_csv(RESULTS / "predictions.csv")
        cls.folds = pd.read_csv(RESULTS / "folds.csv")
        cls.diag = pd.read_csv(RESULTS / "fold_state_diagnostics.csv")
        cls.summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))

    def test_all_folds_verified_and_held_out_once(self):
        self.assertEqual(len(self.folds), 73)
        self.assertTrue(self.folds.state_verified.all() and self.folds.heldout_assignments_equal.all())
        self.assertEqual(int(self.folds.test_n.sum()), 2016)
        self.assertFalse(self.pred.duplicated(["entity_id", "indicator"]).any())
        saved = pd.read_csv(STRICT / "strict_assignments.csv").set_index("entity_id").strict_type
        np.testing.assert_array_equal(self.pred.group.to_numpy(), saved.loc[self.pred.entity_id].to_numpy())

    def test_state_diagnostics_within_recorded_tolerance(self):
        v = self.summary["state_verification"]
        for name, _ in AV.NUMERIC_COMPONENTS:
            stored = self.diag[f"{name}_max_abs"]
            self.assertTrue((stored <= v["atol"] + v["rtol"] * 1e3).all(), name)
            self.assertAlmostEqual(v["max_abs_by_component"][name], stored.max(), places=20)
        for name in AV.EXACT_COMPONENTS:
            self.assertTrue(self.diag[f"{name}_equal"].all(), name)
        self.assertEqual(v["cause_of_full_hash_difference"], "not established")
        self.assertTrue(self.diag.sha256_x.str.fullmatch("[0-9a-f]{64}").all())

    def test_group_only_model_reproduces_published_strict_check(self):
        strict = pd.read_csv(STRICT / "strict_metrics.csv").set_index("indicator")
        for row in self.models.itertuples():
            if row.indicator in strict.index:
                self.assertEqual(row.n, strict.loc[row.indicator, "n"])
                self.assertAlmostEqual(row.r2_groups, strict.loc[row.indicator, "types"], places=10)

    def test_contrasts_are_paired_on_identical_rows(self):
        m = self.models.set_index("indicator")
        for row in self.contrasts.itertuples():
            self.assertEqual(row.n, m.loc[row.indicator, "n"])
            expected = m.loc[row.indicator, f"r2_{row.candidate}"] - m.loc[row.indicator, f"r2_{row.baseline}"]
            self.assertAlmostEqual(row.delta_r2, expected, places=8)
            self.assertLessEqual(row.delta_r2_simultaneous_low, row.delta_r2_ci_low)
            self.assertGreaterEqual(row.delta_r2_simultaneous_high, row.delta_r2_ci_high)

    def test_p_values_intervals_and_holm_are_consistent(self):
        c = self.contrasts
        self.assertEqual(len(c), 8 * len(AV.CONTRASTS))
        for p_col, prefix in (("p_boot_t", "p_holm"), ("p_percentile", "p_pct_holm")):
            for _, part in c.groupby("contrast"):
                self.assertEqual(sorted(part.indicator), sorted(AV.LABELS))
                np.testing.assert_allclose(part[f"{prefix}_family"], AV.holm(part[p_col]), atol=1e-9)
            np.testing.assert_allclose(c[f"{prefix}_all"], AV.holm(c[p_col]), atol=1e-9)
            self.assertTrue((c[f"{prefix}_all"] >= c[f"{prefix}_family"] - 1e-12).all())
        draws = self.summary["bootstrap_draws"]
        self.assertTrue((c.p_boot_t >= (1 - 1e-9) / (draws + 1)).all())
        self.assertTrue((c.p_percentile >= 2 * (1 - 1e-9) / (draws + 1)).all())
        clear = (c.p_boot_t - 0.05).abs() > 2 / draws
        excludes = (c.delta_r2_ci_low > 0) | (c.delta_r2_ci_high < 0)
        self.assertTrue(((c.p_boot_t < 0.05) == excludes)[clear].all())
        pct_excludes = (c.delta_r2_pct_ci_low > 0) | (c.delta_r2_pct_ci_high < 0)
        clear = (c.p_percentile - 0.05).abs() > 2 / draws
        self.assertTrue(((c.p_percentile < 0.05) == pct_excludes)[clear].all())

    def test_analysis_order_and_scope_recorded(self):
        statuses = {k: v["status"] for k, v in self.summary["contrasts"].items()}
        self.assertEqual(statuses["groups_over_spline"], "primary, defined before the first full run")
        self.assertEqual(statuses["groups_over_shares"], "added after the first full run")
        readme = (RESULTS / "README.md").read_text(encoding="utf-8")
        self.assertIn("Последовательность анализа", readme)
        self.assertIn("добавочный выигрыш групп сверх непрерывных долей в этой проверке не подтверждён", readme)
        self.assertNotIn("если выигрыш есть, он мал»", readme.replace("«если выигрыш есть, он мал» не используется", ""))
        self.assertTrue(any("2025" in s for s in self.summary["scope"]))

    def test_calibration_recorded(self):
        calib = pd.read_csv(RESULTS / "calibration.csv")
        self.assertEqual(len(calib), 6)
        self.assertTrue(calib["boot_t_rate_0.05"].between(0.02, 0.09).all())

    def test_first_fold_refits_to_saved_predictions(self):
        pred, folds = AV.predictions(ROOT, max_regions=1)
        saved = self.pred[self.pred.region == folds[0]["region"]].reset_index(drop=True)
        pd.testing.assert_frame_equal(pred.reset_index(drop=True), saved, check_exact=False, rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
