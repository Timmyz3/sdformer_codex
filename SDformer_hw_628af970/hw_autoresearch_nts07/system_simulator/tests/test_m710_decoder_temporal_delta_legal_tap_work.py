#!/usr/bin/env python3
import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "analyze_m710_decoder_temporal_delta_legal_tap_work.py"
SPEC = importlib.util.spec_from_file_location("m710", str(MODULE_PATH))
M710 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M710)


class M710Tests(unittest.TestCase):
    def test_legal_tap_weights(self):
        got = M710.legal_tap_weights(2, 3)
        np.testing.assert_array_equal(got, np.array([[4, 6, 6], [6, 9, 9]], dtype=np.uint8))

    def test_full_and_delta_counts(self):
        mask = np.zeros((10, 1, 1, 2, 2), dtype=np.uint8)
        mask[0, 0, 0, 0, 0] = 1   # legal tap multiplicity 4
        mask[1, 0, 0, 0, 0] = 1   # unchanged; no delta
        mask[1, 0, 0, 1, 1] = 1   # transition on; multiplicity 9
        mask[2, 0, 0, 0, 0] = 0   # transition off; multiplicity 4
        got = M710.count_mask(mask, M710.legal_tap_weights(2, 2))
        self.assertEqual(got["full_active_sources"], 3)
        self.assertEqual(got["delta_initial_active_sources"], 1)
        self.assertEqual(got["delta_transition_sources"], 3)
        self.assertEqual(got["delta_sources"], 4)
        self.assertEqual(got["full_active_legal_tap_events"], 17)
        self.assertEqual(got["delta_initial_plus_xor_legal_tap_events"], 26)

    def test_constant_one_is_cheaper_as_delta(self):
        mask = np.ones((10, 1, 1, 1, 1), dtype=np.uint8)
        got = M710.count_mask(mask, M710.legal_tap_weights(1, 1))
        self.assertEqual(got["full_active_legal_tap_events"], 40)
        self.assertEqual(got["delta_initial_plus_xor_legal_tap_events"], 4)

    def test_alternating_is_more_expensive_as_delta(self):
        mask = np.zeros((10, 1, 1, 1, 1), dtype=np.uint8)
        mask[::2] = 1
        got = M710.count_mask(mask, M710.legal_tap_weights(1, 1))
        self.assertEqual(got["full_active_legal_tap_events"], 20)
        self.assertEqual(got["delta_initial_plus_xor_legal_tap_events"], 40)

    def test_aggregate_is_ratio_of_sums(self):
        rows = [
            {"module": "D0", "full_product_work": 10, "delta_product_work": 20,
             "full_active_legal_tap_events": 1, "delta_initial_plus_xor_legal_tap_events": 2},
            {"module": "D0", "full_product_work": 90, "delta_product_work": 90,
             "full_active_legal_tap_events": 9, "delta_initial_plus_xor_legal_tap_events": 9},
        ]
        got = M710.aggregate(rows, ["module"])[0]
        self.assertAlmostEqual(got["delta_over_full_product_work"], 1.1)

    def test_bad_shape_rejected(self):
        with self.assertRaises(ValueError):
            M710.count_mask(np.zeros((9, 1, 1, 1, 1), dtype=np.uint8), np.ones((1, 1), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
