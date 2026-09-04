from __future__ import annotations

import unittest

import numpy as np

from dptme_reference import direct_temporal_matrix, dptme_t10, dptme_t2_five_way, run_trials


class DptmeReferenceTest(unittest.TestCase):
    def test_t10_mapping_matches_direct(self):
        x = np.arange(10 * 3 * 32, dtype=np.int16).reshape(10, 3, 32) - 100
        weight = np.arange(100, dtype=np.int16).reshape(10, 10) - 50
        bias = np.arange(10, dtype=np.int32)
        np.testing.assert_array_equal(dptme_t10(x, weight, bias), direct_temporal_matrix(x, weight, bias))

    def test_t2_five_way_handles_full_and_tail_groups(self):
        x = np.arange(2 * 81 * 32, dtype=np.int16).reshape(2, 81, 32) - 80
        weight = np.asarray([[3, -2], [5, 7]], dtype=np.int16)
        bias = np.asarray([11, -13], dtype=np.int32)
        np.testing.assert_array_equal(
            dptme_t2_five_way(x, weight, bias), direct_temporal_matrix(x, weight, bias)
        )

    def test_random_regression_and_utilization(self):
        result = run_trials(trials=3)
        self.assertEqual(result["T10"]["hidden_mismatches"], 0)
        self.assertEqual(result["T2"]["event_mismatches"], 0)
        self.assertEqual(result["T2"]["cycles_per_81_position_head_tile"], 34)
        self.assertAlmostEqual(result["T2"]["slot_utilization"], 162 / 170)


if __name__ == "__main__":
    unittest.main()
