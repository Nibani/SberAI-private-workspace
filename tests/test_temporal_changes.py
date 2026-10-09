import unittest
import csv
from pathlib import Path
import numpy as np
from sbercluster.temporal_changes import (projected_tracking, detect_mean_changes,
    fit_calendar, temporal_channels, train_variance, match_boundaries)


class TemporalChanges(unittest.TestCase):
    def test_legacy_saved_model_projection_equivalence(self):
        """Missing legacy level_center must reproduce every saved monthly label."""
        from sbercluster import panel as P
        root=Path(__file__).resolve().parents[1]
        panel=P.read_panel(root/'data/v12/panel.csv.gz')
        model=np.load(root/'reports/v1.2/model.npz')
        level=(model['level_center'] if 'level_center' in model.files
               else np.median(np.log(panel.totals),axis=0))
        scaler=P.Scaler(model['structure_center'],model['scale'],12,level)
        xm=P.monthly_attributes(panel,scaler); ref=P.annual_profile(xm,slice(0,12))
        adjusted=P.remove_national_wave(xm,ref)
        with (root/'reports/v1.2/labels.csv').open(encoding='utf-8',newline='') as stream:
            saved={r['entity_id']:r for r in csv.DictReader(stream)}
        graph=np.array([int(saved[i]['type_2023']) for i in panel.ids])
        tracking=projected_tracking(ref,adjusted,model['centers'],graph)
        expected=np.array([[int(c) for c in saved[i]['monthly_types']] for i in panel.ids])
        np.testing.assert_array_equal(tracking['projected_labels'],expected)
        constant=projected_tracking(ref,np.repeat(ref[:,None],24,axis=1),model['centers'],graph)
        self.assertEqual(constant['persistent_changes'],0)
        self.assertEqual(np.count_nonzero(constant['projected_baseline']!=graph),48)

    def test_tail_is_not_any_six_month_run(self):
        x = np.zeros((1,24,1)); x[:,6:12] = 5
        r = projected_tracking([[0]], x, [[0],[5]], [1])
        self.assertEqual(r['persistent_changes'], 0)
        self.assertEqual(r['switch_counts'][0], 2)
        self.assertTrue(r['projection_disagreement'][0])

    def test_ties_are_deterministic(self):
        r = projected_tracking([[0]], np.zeros((1,24,1)), [[-1],[1]], [1])
        self.assertEqual(r['projected_baseline'][0], 0)
        self.assertEqual(r['distance_margin'][0,0], 0)

    def test_calendar_and_common_shock(self):
        rng = np.random.default_rng(123)
        means = rng.normal(size=(9,2))
        calendar = np.column_stack([np.sin(np.arange(12)*2*np.pi/12), np.cos(np.arange(12)*2*np.pi/12)])
        raw = means[:,None]+np.tile(calendar,(2,1))[None]
        fitted = fit_calendar(raw)
        channels = temporal_channels(raw, fitted)
        np.testing.assert_allclose(channels['absolute'],np.repeat(means[:,None],24,axis=1),atol=1e-14)
        shock = raw.copy(); shock[:,12:] += [3,-2]
        changed = temporal_channels(shock, fitted)
        np.testing.assert_allclose(changed['relative'],channels['relative'],atol=1e-14)
        np.testing.assert_allclose(changed['common_drift'][12:]-channels['common_drift'][12:],np.tile([3,-2],(12,1)),atol=1e-14)
        self.assertTrue(detect_mean_changes(changed['absolute'][0],1,3)['persistent_boundaries'])
        self.assertFalse(detect_mean_changes(changed['relative'][0],1,3)['boundaries'])

    def test_evaluation_cannot_change_calibration(self):
        x = np.random.default_rng(14).normal(size=(32,24,5)); y = x.copy(); y[:,12:] *= 100
        np.testing.assert_array_equal(fit_calendar(x),fit_calendar(y))
        self.assertEqual(train_variance(x),train_variance(y))

    def test_return_and_censoring(self):
        x = np.zeros((24,1)); x[12:18] = 10
        r = detect_mean_changes(x,1,3)
        self.assertEqual(r['boundaries'],[12,18])
        self.assertEqual(r['persistent_boundaries'],[12,18])
        x[:] = 0; x[21:] = 10
        r = detect_mean_changes(x,1,3)
        self.assertEqual(r['boundaries'],[21])
        self.assertEqual(r['persistent_boundaries'],[])
        self.assertTrue(r['events'][0]['censored'])

    def test_matching_maximum_cardinality(self):
        self.assertEqual(len(match_boundaries([11,12],[12,13])),2)
        self.assertEqual(len(match_boundaries([12],[11,13])),1)

    def test_explicit_invalid(self):
        for penalty in [0,-1,float('nan'),float('inf'),'invalid',True,1j]:
            with self.assertRaises(ValueError): detect_mean_changes(np.zeros((24,2)),penalty)
        for x in [np.zeros((0,2)),np.zeros(24),np.ones((2,2))]:
            with self.assertRaises(ValueError): detect_mean_changes(x,1)
        with self.assertRaises(ValueError): temporal_channels(np.zeros((2,23,2)),np.zeros((12,2)))
        with self.assertRaises(ValueError): projected_tracking([[0]],np.zeros((1,24,1)),[[0]],[2])


if __name__ == '__main__': unittest.main()
