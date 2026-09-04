#!/usr/bin/env python3
"""CICC BWAC/FAED DSE 的纯函数单元测试。"""

import unittest

from scripts.analyze_cicc_bwac_faed import encode_groups, signed_bits


class TestCiccBwacFaed(unittest.TestCase):
    def test_signed_bits(self) -> None:
        self.assertEqual(signed_bits(0), 1)
        self.assertEqual(signed_bits(3), 3)
        self.assertEqual(signed_bits(-4), 3)
        self.assertEqual(signed_bits(127), 8)
        self.assertEqual(signed_bits(-128), 8)

    def test_dense_small_values_prefer_minbw(self) -> None:
        row = encode_groups([1, -1, 1, -1] * 4, 16)
        self.assertEqual(row["adaptive_mode_groups"]["minbw"], 1)
        self.assertLess(row["ratios_vs_fixed8"]["adaptive"], 1.0)

    def test_sparse_values_can_prefer_bitmap(self) -> None:
        row = encode_groups([0] * 15 + [1], 16)
        self.assertEqual(row["adaptive_mode_groups"]["bwac_bitmap"], 1)


if __name__ == "__main__":
    unittest.main()
