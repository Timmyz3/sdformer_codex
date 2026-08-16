from __future__ import annotations

import unittest

from scripts.local5_erep_schedule_reference import (
    TaskTiming,
    evaluate_window,
    simulate_two_slot_pipeline,
)


class Local5ErepScheduleReferenceTest(unittest.TestCase):
    def test_two_slots_apply_backpressure_before_third_fill(self) -> None:
        result = simulate_two_slot_pipeline(
            [TaskTiming(1, 10), TaskTiming(1, 10), TaskTiming(1, 10)]
        )
        self.assertEqual(result.cycles, 31)
        self.assertEqual(result.producer_busy_cycles, 3)
        self.assertEqual(result.consumer_busy_cycles, 30)
        self.assertEqual(result.producer_slot_stall_cycles, 9)
        self.assertEqual(result.consumer_wait_cycles, 1)

    def test_equal_boundary_four_way_ablation(self) -> None:
        result = evaluate_window(
            [10, 10], [20, 20], output_tiles=2, stripe_width=2
        )
        self.assertEqual(result.direct_serial_cycles, 120)
        self.assertEqual(result.reuse_only_cycles, 100)
        self.assertEqual(result.overlap_only_cycles, 90)
        self.assertEqual(result.erep_cycles, 90)

    def test_remainder_stripe_and_common_drain(self) -> None:
        result = evaluate_window(
            [8, 8, 8],
            [5, 5, 5],
            output_tiles=3,
            stripe_width=2,
            drain_cycles_per_tile=7,
        )
        self.assertEqual(result.direct_serial_cycles, 138)
        self.assertEqual(result.reuse_only_cycles, 114)
        self.assertEqual(result.overlap_only_cycles, 98)
        self.assertEqual(result.erep_cycles, 88)

    def test_invalid_inputs_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_window([], [], output_tiles=1)
        with self.assertRaises(ValueError):
            evaluate_window([1], [1, 2], output_tiles=1)
        with self.assertRaises(ValueError):
            evaluate_window([1], [1], output_tiles=0)
        with self.assertRaises(ValueError):
            evaluate_window([1], [1], output_tiles=1, stripe_width=0)


if __name__ == "__main__":
    unittest.main()
