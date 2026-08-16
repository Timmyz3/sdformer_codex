from __future__ import annotations

import unittest

import numpy as np

from temporal_signature_projection_reference import (
    dense_projection,
    run_trials,
    temporal_signature_projection,
)


class TemporalSignatureProjectionTest(unittest.TestCase):
    def test_common_lanes_are_read_once_and_outputs_match(self):
        k0 = np.asarray([[1, 1, 0, 0]], dtype=bool)
        k1 = np.asarray([[0, 1, 1, 0]], dtype=bool)
        weight = np.arange(12, dtype=np.int16).reshape(4, 3) - 5
        gate0 = np.asarray([3], dtype=np.int16)
        gate1 = np.asarray([7], dtype=np.int16)
        expected = dense_projection(k0, k1, weight, gate0, gate1)
        got0, got1, traffic = temporal_signature_projection(k0, k1, weight, gate0, gate1)
        np.testing.assert_array_equal(got0, expected[0])
        np.testing.assert_array_equal(got1, expected[1])
        self.assertEqual(traffic["baseline_weight_row_reads"], 4)
        self.assertEqual(traffic["union_weight_row_reads"], 3)
        self.assertEqual(traffic["intersection_reused_reads"], 1)

    def test_zero_and_identical_patterns(self):
        weight = np.eye(4, dtype=np.int16)
        gate = np.asarray([2], dtype=np.int16)
        zero = np.zeros((1, 4), dtype=bool)
        _, _, empty = temporal_signature_projection(zero, zero, weight, gate, gate)
        self.assertEqual(empty["baseline_weight_row_reads"], 0)
        pattern = np.asarray([[1, 0, 1, 0]], dtype=bool)
        _, _, same = temporal_signature_projection(pattern, pattern, weight, gate, gate)
        self.assertEqual(same["baseline_weight_row_reads"], 4)
        self.assertEqual(same["union_weight_row_reads"], 2)

    def test_random_integer_equivalence(self):
        result = run_trials(trials=20)
        self.assertEqual(result["mismatches"], 0)
        self.assertGreater(result["compared_outputs"], 0)


if __name__ == "__main__":
    unittest.main()
