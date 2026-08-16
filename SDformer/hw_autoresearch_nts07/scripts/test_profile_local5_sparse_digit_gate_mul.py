#!/usr/bin/env python3
"""Unit tests for the Local5 sparse-digit gate multiplier screen."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profile_local5_sparse_digit_gate_mul import (  # noqa: E402
    analyze_descriptor_chunk,
    bitmap_popcount,
)


class SparseDigitGateMulTest(unittest.TestCase):
    def test_bitmap_popcount(self) -> None:
        values = np.asarray([0, 1, 0xF0, 0xFFFFFFFF], dtype=np.uint64)
        self.assertEqual(bitmap_popcount(values).tolist(), [0, 1, 4, 32])

    def test_unique_gate_and_digit_cycles(self) -> None:
        gates = np.asarray(
            [
                [16, 16, 15, 0, 0],
                [32, 31, 0, 0, 0],
            ],
            dtype=np.uint16,
        )
        valid = np.asarray([0b00111, 0b00011], dtype=np.uint8)
        k_bitmap = np.asarray([0b1011, 0b11], dtype=np.uint64)
        # row0: unique gates 16 and 15, each with 3 K lanes -> 6 terms,
        # cycles 3*1 + 3*4 = 15. row1: 32 and 31, each with 2 lanes.
        expected = np.asarray([6, 4], dtype=np.uint32)
        result = analyze_descriptor_chunk(gates, valid, k_bitmap, expected)
        self.assertEqual(result["baseline_terms"].tolist(), [6, 4])
        self.assertEqual(result["digit_serial_cycles"].tolist(), [15, 12])
        self.assertEqual(int(result["popcount_hist"][1]), 5)
        self.assertEqual(int(result["popcount_hist"][4]), 3)
        self.assertEqual(int(result["popcount_hist"][5]), 2)

    def test_expected_term_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(AssertionError, "producer"):
            analyze_descriptor_chunk(
                np.asarray([[16, 0, 0, 0, 0]], dtype=np.uint16),
                np.asarray([1], dtype=np.uint8),
                np.asarray([3], dtype=np.uint64),
                np.asarray([1], dtype=np.uint32),
            )


if __name__ == "__main__":
    unittest.main()
