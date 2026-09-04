from __future__ import annotations

import unittest

import numpy as np

from class_gate_multicast_projection_reference import (
    class_gate_multicast_projection,
    dense_selector_projection,
    run_trials,
)


class ClassGateMulticastProjectionTest(unittest.TestCase):
    def test_same_class_channel_product_is_multicast_exactly(self):
        k_event = np.asarray([[1, 0], [1, 1], [0, 1]], dtype=bool)
        score_class = np.asarray([0, 0, 1], dtype=np.int16)
        gate_by_class = np.asarray([3, 5], dtype=np.int16)
        weight = np.asarray([[2, -1], [4, 7]], dtype=np.int16)
        expected = dense_selector_projection(k_event, score_class, gate_by_class, weight)
        actual, counters = class_gate_multicast_projection(
            k_event, score_class, gate_by_class, weight
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(counters["baseline_active_lanes"], 4)
        self.assertEqual(counters["class_channel_terms"], 3)
        self.assertEqual(counters["max_token_fanout"], 2)

    def test_empty_k_has_no_product_or_fanout(self):
        k_event = np.zeros((3, 2), dtype=bool)
        score_class = np.asarray([0, 0, 1], dtype=np.int16)
        gate_by_class = np.asarray([3, 5], dtype=np.int16)
        weight = np.ones((2, 4), dtype=np.int16)
        actual, counters = class_gate_multicast_projection(
            k_event, score_class, gate_by_class, weight
        )
        self.assertFalse(actual.any())
        self.assertEqual(counters["class_channel_terms"], 0)
        self.assertEqual(counters["max_token_fanout"], 0)

    def test_random_integer_equivalence(self):
        result = run_trials(trials=10)
        self.assertEqual(result["mismatches"], 0)
        self.assertGreater(result["compared_outputs"], 0)

    def test_all_gate_int8_products_fit_signed17(self):
        gates = np.arange(257, dtype=np.int64)[:, None]
        weights = np.arange(-128, 128, dtype=np.int64)[None, :]
        products = gates * weights
        self.assertGreaterEqual(int(products.min()), -(1 << 16))
        self.assertLessEqual(int(products.max()), (1 << 16) - 1)

        encoded = products & ((1 << 17) - 1)
        decoded = np.where(encoded & (1 << 16), encoded - (1 << 17), encoded)
        np.testing.assert_array_equal(decoded, products)


if __name__ == "__main__":
    unittest.main()
