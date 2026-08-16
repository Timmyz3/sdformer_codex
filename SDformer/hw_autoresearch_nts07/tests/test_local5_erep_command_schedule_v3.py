from __future__ import annotations

import unittest

from scripts.local5_erep_command_schedule_v3 import (
    HeadCommandWork,
    PhaseTrace,
    RelativeCommand,
    WindowCommandWork,
    evaluate_window,
)


def cmd(cycle: int, resource: str, kind: str, identity: str) -> RelativeCommand:
    return RelativeCommand(cycle, resource, kind, identity)


def fill(records: int, offset: int = 0) -> PhaseTrace:
    commands = []
    for record in range(records):
        identity = f"r{record}"
        commands.append(cmd(offset + record, "relation_workspace_1rw", "relation_read", identity))
        commands.append(cmd(offset + records + record, "epoch_slot_1rw", "epoch_record_write", identity))
    return PhaseTrace(offset + 2 * records + 1, tuple(commands))


def execute(records: int, duration_extra: int, bank: int) -> PhaseTrace:
    commands = []
    for record in range(records):
        identity = f"r{record}"
        commands.append(cmd(record, "epoch_slot_1rw", "epoch_record_read", identity))
        commands.append(cmd(record, "fifo2_enq", "fifo_enqueue", identity))
        commands.append(cmd(record, "fifo2_deq", "fifo_dequeue", identity))
        commands.append(cmd(records + record, f"acc_bank_{bank}_1rw", "acc_write", identity))
    return PhaseTrace(2 * records + duration_extra, tuple(commands))


def direct(records: int, extra: int, bank: int) -> PhaseTrace:
    commands = tuple(
        cmd(record, f"acc_bank_{bank}_1rw", "acc_write", f"r{record}")
        for record in range(records)
    )
    return PhaseTrace(records + extra, commands)


def fixture() -> WindowCommandWork:
    heads = []
    for head, records in enumerate((2, 1, 2)):
        heads.append(
            HeadCommandWork(
                epoch_records=records,
                fill=fill(records, offset=head),
                direct_by_tile=tuple(
                    direct(records, 5 + tile, head % 5) for tile in range(3)
                ),
                execute_by_tile=tuple(
                    execute(records, 3 + tile, head % 5) for tile in range(3)
                ),
            )
        )
    prepare = tuple(PhaseTrace(1 + tile, ()) for tile in range(3))
    drain = tuple(
        PhaseTrace(2 + tile, (cmd(0, "drain_read_1rw", "drain_read", f"t{tile}"),))
        for tile in range(3)
    )
    return WindowCommandWork("w0", tuple(heads), prepare, drain)


class Local5ErepCommandScheduleV3Test(unittest.TestCase):
    def test_all_candidate_ledgers_are_resource_legal(self) -> None:
        rows = evaluate_window(fixture())
        self.assertEqual(set(rows), {
            "c0_direct_serial", "c1_reuse_only_s2",
            "c2_overlap_only", "c3_erep_s2",
        })
        self.assertTrue(all(row.cycles > 0 for row in rows.values()))
        self.assertEqual(rows["c3_erep_s2"].acc_contexts, 2)
        self.assertEqual(rows["c3_erep_s2"].epoch_slots, 2)

    def test_slot_ownership_has_no_sealed_gap(self) -> None:
        result = evaluate_window(fixture())["c3_erep_s2"]
        for slot in (0, 1):
            task_ids = {
                event.identity for event in result.events
                if event.resource == f"slot_{slot}_owner"
            }
            for task_id in task_ids:
                rows = sorted(
                    (event for event in result.events if event.resource == f"slot_{slot}_owner" and event.identity == task_id),
                    key=lambda event: event.start,
                )
                self.assertEqual(rows[0].kind, "FILL")
                self.assertEqual(rows[-1].kind, "CONSUME")
                for left, right in zip(rows, rows[1:]):
                    self.assertEqual(left.end, right.start)

    def test_context_ownership_covers_prepare_through_drain(self) -> None:
        result = evaluate_window(fixture())["c3_erep_s2"]
        owners = [event for event in result.events if event.kind == "OWNED"]
        prepares = [event for event in result.events if event.kind == "PREPARE"]
        drains = [event for event in result.events if event.kind == "DRAIN"]
        self.assertEqual(len(owners), 3)
        self.assertEqual(len(prepares), 3)
        self.assertEqual(len(drains), 3)
        for owner in owners:
            tile = int(owner.identity.split("_")[1])
            prepare_event = next(event for event in prepares if event.identity == f"tile_{tile}_prepare")
            drain_event = next(event for event in drains if event.identity == f"tile_{tile}_drain")
            self.assertEqual(owner.start, prepare_event.start)
            self.assertEqual(owner.end, drain_event.end)

    def test_record_level_counts_include_remainder(self) -> None:
        result = evaluate_window(fixture())["c3_erep_s2"]
        records = 2 + 1 + 2
        self.assertEqual(result.epoch_record_writes, records * 2)
        self.assertEqual(result.epoch_record_reads, records * 3)

    def test_fifo_overflow_and_1rw_collision_fail_closed(self) -> None:
        bypass = PhaseTrace(
            1,
            (
                cmd(0, "fifo2_enq", "fifo_enqueue", "a"),
                cmd(0, "fifo2_deq", "fifo_dequeue", "a"),
            ),
        )
        self.assertEqual(bypass.count_kind("fifo_enqueue"), 1)
        self.assertEqual(bypass.count_kind("fifo_dequeue"), 1)
        with self.assertRaises(ValueError):
            PhaseTrace(
                1,
                (cmd(0, "fifo2_deq", "fifo_dequeue", "a"),),
            )
        with self.assertRaises(ValueError):
            PhaseTrace(
                3,
                (
                    cmd(0, "fifo2_enq", "fifo_enqueue", "a"),
                    cmd(1, "fifo2_enq", "fifo_enqueue", "b"),
                    cmd(2, "fifo2_enq", "fifo_enqueue", "c"),
                ),
            )
        with self.assertRaises(ValueError):
            PhaseTrace(
                1,
                (
                    cmd(0, "epoch_slot_1rw", "epoch_record_read", "a"),
                    cmd(0, "epoch_slot_1rw", "epoch_record_read", "b"),
                ),
            )

    def test_every_output_tile_preserves_strict_input_head_order(self) -> None:
        result = evaluate_window(fixture())["c3_erep_s2"]
        executes = [event for event in result.events if event.kind == "EXECUTE"]
        for tile in range(3):
            tile_rows = sorted(
                (event for event in executes if f"_tile_{tile}_" in event.identity),
                key=lambda event: event.start,
            )
            observed_heads = [
                int(event.identity.split("_head_")[1].split("_")[0])
                for event in tile_rows
            ]
            self.assertEqual(observed_heads, [0, 1, 2])

    def test_zero_duration_prepare_or_drain_is_forbidden(self) -> None:
        with self.assertRaises(ValueError):
            PhaseTrace(0, ())


if __name__ == "__main__":
    unittest.main()
