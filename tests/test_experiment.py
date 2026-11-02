"""Tests for conformal quantiles, temporal causality, and settlement accounting."""
import unittest
import numpy as np
import pandas as pd
from experiment import (conformal_radius, daily_intervals, daily_calibrated_bounds,
                        nonconformity, make_demo, prepare, trading_signals)


class ExperimentTests(unittest.TestCase):
    def test_cqr_scores_and_finite_sample_expansion(self):
        lower, upper = np.array([0., 0., 0.]), np.array([2., 2., 2.])
        scores = nonconformity(lower, upper, np.array([-2., 1., 5.]))
        np.testing.assert_array_equal(scores, [2., 0., 3.])
        q = conformal_radius(scores, .25)
        self.assertEqual(q, 3.)
        np.testing.assert_array_equal(lower - q, [-3., -3., -3.])
        np.testing.assert_array_equal(upper + q, [5., 5., 5.])

    def test_cqr_updates_use_base_not_expanded_interval(self):
        times = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
        lower, upper, actual = np.zeros(48), np.ones(48), np.full(48, 5.)
        lo, hi = daily_calibrated_bounds(lower, upper, actual, times,
                                        np.full(24, 2.), .1, 24)
        np.testing.assert_array_equal(hi[:24], np.full(24, 3.))
        np.testing.assert_array_equal(hi[24:], np.full(24, 5.))
        np.testing.assert_array_equal(lo[24:], np.full(24, -4.))

    def test_finite_sample_quantile(self):
        self.assertEqual(conformal_radius(np.arange(1, 10), .2), 8)
        self.assertTrue(np.isinf(conformal_radius([1, 2], .1)))
        with self.assertRaises(ValueError):
            conformal_radius([], .1)

    def test_day_labels_do_not_change_same_day_intervals(self):
        times = pd.date_range("2025-01-01", periods=48, freq="h", tz="UTC")
        pred, y = np.zeros(48), np.ones(48)
        original = daily_intervals(pred, y, times, np.ones(24), .1, 24)
        y[:24] = 100
        changed = daily_intervals(pred, y, times, np.ones(24), .1, 24)
        np.testing.assert_array_equal(original[1][:24], changed[1][:24])
        self.assertTrue(np.all(changed[1][24:] == 100))

    def test_historical_features_and_gaps(self):
        raw = make_demo(30)
        prepared = prepare(raw)
        self.assertEqual(prepared.da_lag_48.iloc[0], raw.price_da.iloc[0])
        with self.assertRaises(ValueError):
            prepare(raw.drop(index=100))

    def test_direction_cost_and_threshold(self):
        positions = trading_signals(np.array([3, -5, -1, 2]), np.array([5, -3, 1, 4]), 2)
        np.testing.assert_array_equal(positions, [1, -1, 0, 0])
        spread = np.array([10, -8, 100, 100])
        pnl = positions * spread - 2 * np.abs(positions)
        np.testing.assert_array_equal(pnl, [8, 6, 0, 0])


if __name__ == "__main__":
    unittest.main()
