#!/usr/bin/env python3
"""Motion ECGB有限流水与存储公式测试。"""

import unittest

from scripts.model_motion_ecgb_ordered_profile import (
    finalization_cycles,
    group_payload_bits,
    pingpong_cycles,
)


class TestMotionEcgbOrderedProfile(unittest.TestCase):
    def test_pingpong_overlap(self) -> None:
        self.assertEqual(pingpong_cycles([4, 4], [10, 10]), 24)
        self.assertLess(pingpong_cycles([4, 4], [10, 10]), 28)

    def test_payload_increases_with_group_span(self) -> None:
        g1 = group_payload_bits(
            terms=8, active_lanes=32, dim=96, tokens=162, windows=1
        )
        g8 = group_payload_bits(
            terms=8, active_lanes=32, dim=96, tokens=162, windows=8
        )
        self.assertGreater(g8, g1)

    def test_ibf_keeps_token_drain_but_halves_two_bank_tail(self) -> None:
        self.assertEqual(finalization_cycles(162, 1, "commit_rmw"), 164)
        self.assertEqual(finalization_cycles(162, 1, "ibf_pipelined"), 84)
        self.assertEqual(finalization_cycles(162, 2, "ibf_pipelined"), 168)
        self.assertGreater(
            finalization_cycles(162, 1, "ibf_pipelined"),
            finalization_cycles(162, 1, "bias_free_lower_bound"),
        )


if __name__ == "__main__":
    unittest.main()
