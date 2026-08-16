#!/usr/bin/env python3

import unittest

import numpy as np

from analyze_projection_accumulator_range import (
    analyze_arrays,
    required_signed_bits,
)


class ProjectionAccumulatorRangeTest(unittest.TestCase):
    def test_exact_and_bounds(self) -> None:
        activations = np.asarray([[2, 0], [3, 4]], dtype=np.int64)
        weight = np.asarray([[5, -2], [-3, 7]], dtype=np.int64)
        bias = np.asarray([1, -1], dtype=np.int64)
        expected = activations @ weight.T + bias[None, :]
        row = analyze_arrays(activations, weight, bias, expected)
        self.assertEqual(row["mismatches"], 0)
        self.assertEqual(row["max_abs_final"], 18)
        self.assertEqual(row["actual_order_independent_bound"], 38)
        self.assertGreaterEqual(
            row["universal_int8_bound"],
            row["gate511_weight_exact_bound"],
        )

    def test_mismatch_and_bits(self) -> None:
        activations = np.asarray([[1]], dtype=np.int64)
        weight = np.asarray([[-8]], dtype=np.int64)
        bias = np.asarray([0], dtype=np.int64)
        row = analyze_arrays(
            activations, weight, bias, np.asarray([[0]], dtype=np.int64)
        )
        self.assertEqual(row["mismatches"], 1)
        self.assertEqual(required_signed_bits(0), 1)
        self.assertEqual(required_signed_bits(8), 5)


if __name__ == "__main__":
    unittest.main()
