from __future__ import annotations

import unittest

from scripts.summarize_h67_rqtb_physical_flow import (
    align_occupancy_histogram,
    speedup_distribution,
)


class SummarizeH67RqtbPhysicalFlowTest(unittest.TestCase):
    def test_speedup_is_dimensionless_float_ratio(self) -> None:
        result = speedup_distribution([150, 240], [100, 120])
        self.assertAlmostEqual(result["mean"], 1.75)
        self.assertAlmostEqual(result["p50"], 1.75)
        self.assertEqual(result["max"], 2.0)

    def test_invalid_cycle_lists_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "长度一致"):
            speedup_distribution([100], [50, 60])
        with self.assertRaisesRegex(ValueError, "正数"):
            speedup_distribution([100], [0])

    def test_only_one_zero_boundary_sample_may_be_trimmed(self) -> None:
        aligned, trimmed = align_occupancy_histogram([3, 2], 4, 1)
        self.assertEqual(aligned, [2, 2])
        self.assertEqual(trimmed, -1)
        aligned, added = align_occupancy_histogram([2, 2], 5, 1)
        self.assertEqual(aligned, [3, 2])
        self.assertEqual(added, 1)
        with self.assertRaisesRegex(ValueError, "不一致"):
            align_occupancy_histogram([2, 2], 6, 1)


if __name__ == "__main__":
    unittest.main()
