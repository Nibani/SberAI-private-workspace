import unittest

import numpy as np
from scipy.sparse import csr_matrix

from sbercluster import edges as E


class EdgeRuleTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_knn_union_is_binary_symmetric_and_loop_free(self):
        x = self.rng.normal(size=(40, 3))
        a = E.knn_union(-E.pairwise_euclidean(x), 5)
        self.assertEqual((a != a.T).nnz, 0)
        self.assertTrue(np.all(a.diagonal() == 0))
        self.assertTrue(np.all(a.data == 1))
        self.assertTrue(np.all(np.asarray(a.sum(axis=1)).ravel() >= 5))

    def test_knn_union_keeps_exact_nearest_neighbours(self):
        x = np.array([[0.0], [1.0], [10.0], [11.0]])
        a = E.knn_union(-E.pairwise_euclidean(x), 1).toarray()
        np.testing.assert_array_equal(a, [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])

    def test_pairwise_euclidean_matches_direct_formula(self):
        x = self.rng.normal(size=(15, 4))
        direct = np.sqrt(((x[:, None] - x[None]) ** 2).sum(axis=2))
        np.testing.assert_allclose(E.pairwise_euclidean(x), direct, atol=1e-10)

    def test_pearson_matches_numpy_and_constant_series_is_zero(self):
        x = self.rng.normal(size=(6, 12))
        np.testing.assert_allclose(E.pearson_matrix(x), np.corrcoef(x), atol=1e-12)
        x[0] = 3.0
        self.assertTrue(np.all(E.pearson_matrix(x)[0] == 0))

    def test_multivariate_correlation_is_mean_of_attribute_correlations(self):
        x = self.rng.normal(size=(5, 10, 3))
        expected = np.mean([np.corrcoef(x[:, :, p]) for p in range(3)], axis=0)
        np.testing.assert_allclose(E.multivariate_correlation(x), expected, atol=1e-12)

    def test_lagged_correlation_detects_lead(self):
        base = self.rng.normal(size=14)
        series = np.vstack([base[1:13], base[0:12]])   # series 1 is series 0 delayed by one month
        best, lag = E.lagged_correlation(series, 2)
        self.assertAlmostEqual(best[0, 1], 1.0, places=10)
        self.assertEqual(lag[0, 1], 1)
        self.assertEqual(lag[1, 0], -1)

    def test_dtw_properties(self):
        t = np.arange(12)
        a = np.sin(t / 2)
        shifted = np.sin((t - 1) / 2)
        noise = self.rng.normal(size=12)
        d = E.dtw_distances(np.vstack([a, a, shifted, noise]), band=2)
        self.assertAlmostEqual(d[0, 1], 0.0, places=12)
        np.testing.assert_allclose(d, d.T, atol=1e-12)
        z = lambda v: (v - v.mean()) / v.std()
        self.assertLess(d[0, 2], np.linalg.norm(z(a) - z(shifted)))
        self.assertLess(d[0, 2], d[0, 3])

    def test_dtw_with_zero_band_is_euclidean(self):
        x = self.rng.normal(size=(5, 9))
        z = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
        expected = np.sqrt(((z[:, None] - z[None]) ** 2).sum(axis=2))
        np.testing.assert_allclose(E.dtw_distances(x, band=0, block=2), expected, atol=1e-10)

    def test_haversine_known_distance(self):
        d = E.haversine_km(np.array([55.7558, 59.9343]), np.array([37.6173, 30.3351]))
        self.assertAlmostEqual(d[0, 1], 634, delta=5)

    def test_assortativity_extremes(self):
        clique = np.ones((3, 3)) - np.eye(3)
        two = np.block([[clique, np.zeros((3, 3))], [np.zeros((3, 3)), clique]])
        values = np.array([0, 0, 0, 1, 1, 1.0])
        self.assertAlmostEqual(E.assortativity(csr_matrix(two), values), 1.0)
        bipartite = np.block([[np.zeros((3, 3)), np.ones((3, 3))], [np.ones((3, 3)), np.zeros((3, 3))]])
        self.assertAlmostEqual(E.assortativity(csr_matrix(bipartite), values), -1.0)

    def test_within_group_residual_and_shares(self):
        values = np.array([1.0, 3.0, 10.0, 14.0, np.nan])
        groups = np.array(["a", "a", "b", "b", "b"])
        np.testing.assert_allclose(E.within_group_residual(values, groups)[:4], [-1, 1, -2, 2])
        a = csr_matrix(np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]]))
        self.assertAlmostEqual(E.same_group_share(a, np.array(["x", "x", "y"])), 0.5)
        self.assertEqual(E.edge_jaccard(a, a), 1.0)
        b = csr_matrix(np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]))
        self.assertEqual(E.edge_jaccard(a, b), 0.0)


if __name__ == "__main__":
    unittest.main()
