#!/usr/bin/env python3
"""SFCW有限credit事件模型测试。"""

import unittest

from scripts.model_stage_fissioned_context_wavefront import (
    bounded_three_stage_cycles,
)


class TestStageFissionedContextWavefront(unittest.TestCase):
    def test_balanced_pipeline_overlaps(self) -> None:
        result = bounded_three_stage_cycles([2] * 4, [3] * 4, [4] * 4, 1)
        self.assertLess(result["cycles"], 4 * (2 + 3 + 4))
        self.assertEqual(result["starts"], [4, 4, 4])

    def test_depth_and_lengths_are_checked(self) -> None:
        with self.assertRaises(ValueError):
            bounded_three_stage_cycles([1], [1], [1], 0)
        with self.assertRaises(ValueError):
            bounded_three_stage_cycles([1, 2], [1], [1], 1)

    def test_deeper_fifo_never_hurts(self) -> None:
        depth1 = bounded_three_stage_cycles([1] * 8, [8] * 8, [2] * 8, 1)
        depth4 = bounded_three_stage_cycles([1] * 8, [8] * 8, [2] * 8, 4)
        self.assertLessEqual(depth4["cycles"], depth1["cycles"])


if __name__ == "__main__":
    unittest.main()
