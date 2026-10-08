import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics import adjusted_rand_score

from sbercluster import attributed as A
from sbercluster import panel as P
from sbercluster import tracking as T
from sbercluster import typology as TY


def synthetic_panel(n=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = ["entity_id,territory_id,name,region,period,total_rub,health_pct,marketplaces_pct,horeca_pct,food_pct,transport_pct"]
    for i in range(n):
        for period in P.PERIODS:
            shares = rng.uniform(1, 30, size=5)
            rows.append(",".join([f"tid_{i}", str(i), f"MO {i}", f"R{i % 3}", period,
                                  f"{rng.uniform(1e4, 5e4):.1f}", *[f"{s:.6f}" for s in shares]]))
    return "\n".join(rows) + "\n"


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "panel.csv.gz"
        self.path.write_bytes(gzip.compress(synthetic_panel().encode("utf-8")))
        self.panel = P.read_panel(self.path)

    def tearDown(self):
        self.directory.cleanup()

    def test_read_panel_shapes_and_order(self):
        self.assertEqual(self.panel.totals.shape, (12, 24))
        self.assertEqual(self.panel.shares.shape, (12, 24, 5))
        self.assertEqual(list(self.panel.ids), sorted(self.panel.ids))

    def test_missing_month_is_rejected(self):
        text = synthetic_panel().splitlines()
        broken = Path(self.directory.name) / "broken.csv"
        broken.write_text("\n".join(text[:-1]) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            P.read_panel(broken)

    def test_level_is_centred_on_monthly_median(self):
        x = P.monthly_attributes(self.panel, P.fit_scaler(self.panel))
        np.testing.assert_allclose(np.median(x[:, :, 5], axis=0), 0, atol=1e-12)

    def test_structure_uses_pooled_development_median_and_iqr(self):
        scaler = P.fit_scaler(self.panel)
        x = P.monthly_attributes(self.panel, scaler)
        pooled = x[:, :12, :5].reshape(-1, 5)
        np.testing.assert_allclose(np.median(pooled, axis=0), 0, atol=1e-12)
        q75, q25 = np.percentile(pooled, [75, 25], axis=0)
        np.testing.assert_allclose(q75 - q25, 1, atol=1e-12)

    def test_removing_national_wave_keeps_distances(self):
        x = P.monthly_attributes(self.panel, P.fit_scaler(self.panel))
        reference = P.annual_profile(x, slice(0, 12))
        adjusted = P.remove_national_wave(x, reference)
        np.testing.assert_allclose(P.national_component(adjusted), np.tile(np.median(reference, axis=0), (24, 1)), atol=1e-12)
        d = lambda m: np.linalg.norm(m[:, None] - m[None], axis=2)
        np.testing.assert_allclose(d(adjusted[:, 5]), d(x[:, 5]), atol=1e-12)


class AttributedTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        centers = np.array([[0, 0], [8, 0], [0, 8], [8, 8.0]])
        self.truth = np.repeat(np.arange(4), 25)
        self.x = centers[self.truth] + rng.normal(scale=0.5, size=(100, 2))
        same = (self.truth[:, None] == self.truth[None]).astype(float)
        np.fill_diagonal(same, 0)
        self.w = csr_matrix(same)

    def test_methods_recover_separated_types(self):
        for method in ("kmeans", "ward", "gmm", "kefrin", "joint_0.5", "spectral_graph", "leiden_graph"):
            labels = A.fit_method(method, self.x, self.w, 4, seed=3)
            self.assertAlmostEqual(adjusted_rand_score(self.truth, labels), 1.0, msg=method)

    def test_copeland_prefers_dominant_method_and_penalises_missing(self):
        table = pd.DataFrame({"SW": [0.5, 0.4, 0.3], "CH": [10, 9, 8], "S_Dbw": [0.5, 0.6, np.nan],
                              "AVI": [0.9, 0.8, 0.7], "AVU": [0.4, 0.5, 0.6], "MQ": [0.6, 0.5, 0.4]})
        scores = A.copeland(table)
        self.assertEqual(list(scores), [2, 0, -2])

    def test_external_validity_and_out_of_region_r2(self):
        regions = np.tile(["a", "b", "c", "d", "e"], 20)
        external = pd.DataFrame({"y": self.truth * 2.0 + 1})
        ev = A.external_validity(self.truth, external, regions)
        self.assertAlmostEqual(ev["eta2_y"], 1.0)
        self.assertAlmostEqual(ev["partial_r2_within_region_y"], 1.0)
        aligned = A.external_validity(self.truth, external, np.repeat(["a", "b", "c", "d"], 25))
        self.assertTrue(np.isnan(aligned["partial_r2_within_region_y"]))
        mixed = np.tile(np.repeat(["a", "b"], 2), 25)
        loro = A.leave_one_region_out_r2(external, {"types": A._dummies(self.truth)}, mixed)
        self.assertAlmostEqual(loro.loc[0, "types"], 1.0)

    def test_nearest_and_centroids(self):
        centers = A.centroids(self.x, self.truth)
        np.testing.assert_array_equal(A.nearest(self.x, centers), self.truth)


class TrackingTests(unittest.TestCase):
    def test_markov_rows_and_stationary_distribution(self):
        labels = np.array([[0, 0, 1, 1], [1, 1, 1, 0], [0, 1, 0, 0]])
        result = T.markov(labels, 2)
        matrix = np.array(result["matrix"])
        np.testing.assert_allclose(matrix.sum(axis=1), 1)
        stationary = np.array(result["stationary"])
        np.testing.assert_allclose(stationary @ matrix, stationary, atol=1e-12)
        self.assertAlmostEqual(stationary.sum(), 1)

    def test_persistence_counts_late_stable_changes(self):
        base = np.array([0, 0, 1])
        labels = np.array([[0] * 12, [0] * 4 + [1] * 8, [1, 0] * 6])
        result = T.persistence(labels, base, window=6)
        self.assertEqual(result["persistent_changes"], 1)
        self.assertTrue(result["persistent_change_mask"][1])
        self.assertAlmostEqual(result["constant_share"], 1 / 3)

    def test_monic_events(self):
        base = np.array([0] * 10 + [1] * 10 + [2] * 10 + [3] * 10)
        survive = base.copy()
        self.assertEqual(T.monic_events(base, survive, 4, 0.5), ["survival"] * 4)
        absorbed = np.array([0] * 10 + [1] * 10 + [2] * 10 + [1] * 8 + [3] * 2)
        self.assertEqual(T.monic_events(base, absorbed, 4, 0.5)[3], "absorption")
        split = np.array([0] * 10 + [1] * 10 + [2] * 10 + [0] * 4 + [2] * 4 + [3] * 2)
        self.assertEqual(T.monic_events(base, split, 4, 0.5)[3], "split")

    def test_align_to_recovers_permutation(self):
        base = np.repeat(np.arange(3), 5)
        permuted = np.array([2, 0, 1])[base]
        np.testing.assert_array_equal(T.align_to(base, permuted, 3), base)

    def test_assign_monthly_is_nearest_center(self):
        centers = np.array([[0.0, 0.0], [10.0, 10.0]])
        monthly = np.array([[[1.0, 1.0], [9.0, 9.0]], [[11.0, 9.0], [0.0, -1.0]]])
        np.testing.assert_array_equal(T.assign_monthly(monthly, centers), [[0, 1], [1, 0]])


class TypologyTests(unittest.TestCase):
    def test_naming_rules_are_label_invariant(self):
        level = np.repeat([45000.0, 26000.0, 20000.0, 33000.0], 5)
        shares = np.zeros((20, 5))
        shares[:, 1] = np.repeat([0.10, 0.10, 0.12, 0.05], 5)
        shares[:, 2] = np.repeat([0.07, 0.04, 0.02, 0.03], 5)
        labels = np.repeat(np.arange(4), 5)
        naming = TY.name_types(level, shares, labels)
        self.assertEqual(naming, {0: "metro", 1: "industrial", 2: "periphery", 3: "remote"})
        permuted = np.array([3, 1, 0, 2])[labels]
        np.testing.assert_array_equal(TY.canonical_labels(permuted, TY.name_types(level, shares, permuted)),
                                      TY.canonical_labels(labels, naming))

    def test_region_bootstrap_interval_contains_estimate(self):
        rng = np.random.default_rng(2)
        values = rng.normal(size=200)
        labels = np.repeat([0, 1], 100)
        regions = np.tile(np.arange(20), 10)
        ci = TY.region_bootstrap(values, labels, regions, 2, draws=200, seed=1)
        self.assertTrue(np.all((ci[:, 1] <= ci[:, 0]) & (ci[:, 0] <= ci[:, 2])))


if __name__ == "__main__":
    unittest.main()
