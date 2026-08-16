from __future__ import annotations

import unittest

from scripts.local5_erep_schedule_reference_v2 import (
    HeadWork,
    WindowWork,
    evaluate_window,
)


def fixture() -> WindowWork:
    return WindowWork(
        identity="sample0/stage0/block0/window0",
        heads=(
            HeadWork(3, 10, (30, 31, 32), (20, 21, 22)),
            HeadWork(4, 12, (34, 35, 36), (23, 24, 25)),
            HeadWork(2, 8, (27, 28, 29), (17, 18, 19)),
        ),
        prepare_cycles_by_tile=(1, 2, 3),
        drain_cycles_by_tile=(5, 6, 7),
    )


class Local5ErepScheduleReferenceV2Test(unittest.TestCase):
    def test_four_candidates_have_explicit_resources_and_ledger(self) -> None:
        rows = evaluate_window(fixture())
        self.assertEqual(set(rows), {
            "c0_direct_serial",
            "c1_reuse_only_s2",
            "c2_overlap_only",
            "c3_erep_s2",
        })
        self.assertEqual(rows["c0_direct_serial"].acc_contexts, 1)
        self.assertEqual(rows["c0_direct_serial"].epoch_slots, 0)
        self.assertEqual(rows["c1_reuse_only_s2"].acc_contexts, 2)
        self.assertEqual(rows["c1_reuse_only_s2"].epoch_slots, 1)
        self.assertEqual(rows["c2_overlap_only"].acc_contexts, 1)
        self.assertEqual(rows["c2_overlap_only"].epoch_slots, 2)
        self.assertEqual(rows["c3_erep_s2"].acc_contexts, 2)
        self.assertEqual(rows["c3_erep_s2"].epoch_slots, 2)
        self.assertTrue(all(row.events for row in rows.values()))

    def test_tile_specific_execute_cycles_are_not_width_multiplied_aliases(self) -> None:
        rows = evaluate_window(fixture())
        erep = rows["c3_erep_s2"]
        execute = [
            event for event in erep.events
            if event.kind == "epoch_read_builder_execute"
        ]
        durations = [event.end - event.start for event in execute[:6]]
        self.assertEqual(durations, [20, 21, 23, 24, 17, 18])

    def test_record_transactions_include_remainder_stripe(self) -> None:
        rows = evaluate_window(fixture())
        records_per_stripe = 3 + 4 + 2
        self.assertEqual(
            rows["c3_erep_s2"].epoch_record_writes,
            records_per_stripe * 2,
        )
        self.assertEqual(
            rows["c3_erep_s2"].epoch_record_reads,
            records_per_stripe * 3,
        )

    def test_half_open_slot_intervals_do_not_overlap(self) -> None:
        rows = evaluate_window(fixture())
        erep = rows["c3_erep_s2"]
        for slot in (0, 1):
            slot_rows = sorted(
                (
                    event for event in erep.events
                    if event.resource == f"epoch_slot_{slot}"
                ),
                key=lambda event: (event.start, event.end),
            )
            for left, right in zip(slot_rows, slot_rows[1:]):
                self.assertLessEqual(left.end, right.start)

    def test_invalid_tile_shape_fails_closed(self) -> None:
        with self.assertRaises(ValueError):
            WindowWork(
                identity="bad",
                heads=(HeadWork(0, 1, (2,), (1,)),),
                prepare_cycles_by_tile=(0, 0),
                drain_cycles_by_tile=(0,),
            )


if __name__ == "__main__":
    unittest.main()
