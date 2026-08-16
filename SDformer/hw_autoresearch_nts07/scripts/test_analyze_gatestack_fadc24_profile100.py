#!/usr/bin/env python3
"""FADC24 profile100容量上下界测试。"""

from __future__ import annotations

import unittest

from analyze_gatestack_fadc24_profile100 import classify, destination_byte_bounds


class Fadc24ProfileBoundsTest(unittest.TestCase):
    def test_zero(self) -> None:
        self.assertEqual(destination_byte_bounds(0, 0, 0), (0, 0))

    def test_dense_single_term_uses_bitmap_bound(self) -> None:
        self.assertEqual(destination_byte_bounds(1, 52, 52), (21, 21))

    def test_distribution_range(self) -> None:
        lower, upper = destination_byte_bounds(4, 30, 20)
        self.assertEqual(lower, 30)
        self.assertEqual(upper, 30)

    def test_known_stage3_overflow_is_ambiguous_without_distribution(self) -> None:
        result = classify(61, 814, 52)
        self.assertFalse(result["fadc24_guaranteed_fit"])
        self.assertFalse(result["fadc24_impossible_fit"])
        self.assertTrue(result["fadc24_ambiguous"])


if __name__ == "__main__":
    unittest.main()
