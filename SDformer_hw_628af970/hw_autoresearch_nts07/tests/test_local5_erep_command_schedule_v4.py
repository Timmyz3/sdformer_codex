from __future__ import annotations

from dataclasses import replace
import unittest

from scripts.local5_erep_command_schedule_v4 import (
    CommandKind,
    CommandResource,
    Event,
    HeadCommandWork,
    LEGAL_COMMAND_PAIRS,
    PhaseTrace,
    RelativeCommand,
    ScheduleResult,
    WindowCommandWork,
    evaluate_window,
)


def cmd(cycle: int, resource: str, kind: str, identity: str) -> RelativeCommand:
    return RelativeCommand(cycle, resource, kind, identity)


def fill(records: int, offset: int = 0) -> PhaseTrace:
    commands = []
    for record in range(records):
        identity = f"r{record}"
        commands.append(
            cmd(
                offset + record,
                "relation_workspace_1rw",
                "relation_read",
                identity,
            )
        )
        commands.append(
            cmd(
                offset + records + record,
                "epoch_slot_1rw",
                "epoch_record_write",
                identity,
            )
        )
    return PhaseTrace(offset + 2 * records + 1, tuple(commands))


def execute(records: int, duration_extra: int, bank: int) -> PhaseTrace:
    commands = []
    for record in range(records):
        identity = f"r{record}"
        commands.append(
            cmd(record, "epoch_slot_1rw", "epoch_record_read", identity)
        )
        commands.append(cmd(record, "fifo2_enq", "fifo_enqueue", identity))
        commands.append(cmd(record, "fifo2_deq", "fifo_dequeue", identity))
        commands.append(
            cmd(
                records + record,
                f"acc_bank_{bank}_1rw",
                "acc_write",
                identity,
            )
        )
    return PhaseTrace(2 * records + duration_extra, tuple(commands))


def direct(records: int, extra: int, bank: int) -> PhaseTrace:
    commands = tuple(
        cmd(record, f"acc_bank_{bank}_1rw", "acc_write", f"r{record}")
        for record in range(records)
    )
    return PhaseTrace(records + extra, commands)


def fixture() -> WindowCommandWork:
    output_tiles = (0, 1, 2)
    heads = []
    for head, records in enumerate((2, 1, 2)):
        heads.append(
            HeadCommandWork(
                input_head=head,
                epoch_records=records,
                fill=fill(records, offset=head),
                direct_by_tile=tuple(
                    direct(records, 5 + tile, head % 5) for tile in output_tiles
                ),
                execute_by_tile=tuple(
                    execute(records, 3 + tile, head % 5) for tile in output_tiles
                ),
            )
        )
    prepare = tuple(
        PhaseTrace(
            1 + tile,
            (
                cmd(
                    0,
                    "context_prepare_1rw",
                    "context_prepare",
                    f"t{tile}",
                ),
            ),
        )
        for tile in output_tiles
    )
    drain = tuple(
        PhaseTrace(
            2 + tile,
            (cmd(0, "drain_read_1rw", "drain_read", f"t{tile}"),),
        )
        for tile in output_tiles
    )
    return WindowCommandWork(
        identity="w0",
        heads=tuple(heads),
        output_tiles=output_tiles,
        prepare_by_tile=prepare,
        drain_by_tile=drain,
    )


class Local5ErepCommandScheduleV4Test(unittest.TestCase):
    def test_frozen_resource_kind_pairs_are_exact(self) -> None:
        expected = {
            ("relation_workspace_1rw", "relation_read"),
            ("epoch_slot_1rw", "epoch_record_write"),
            ("epoch_slot_1rw", "epoch_record_read"),
            ("fifo2_enq", "fifo_enqueue"),
            ("fifo2_deq", "fifo_dequeue"),
            *((f"acc_bank_{bank}_1rw", "acc_write") for bank in range(5)),
            ("context_prepare_1rw", "context_prepare"),
            ("drain_read_1rw", "drain_read"),
        }
        observed = {(resource.value, kind.value) for resource, kind in LEGAL_COMMAND_PAIRS}
        self.assertEqual(observed, expected)
        for resource, kind in expected:
            command = cmd(0, resource, kind, "valid")
            self.assertIsInstance(command.resource, CommandResource)
            self.assertIsInstance(command.kind, CommandKind)

    def test_fictional_resource_and_kind_bypasses_fail_closed(self) -> None:
        invalid = (
            ("reviewer_private_1rw", "relation_read"),
            ("epoch_slot_99_1rw", "epoch_record_read"),
            ("acc_bank_5_1rw", "acc_write"),
            ("relation_workspace_1rw", "reviewer_private_read"),
        )
        for resource, kind in invalid:
            with self.subTest(resource=resource, kind=kind):
                with self.assertRaises(ValueError):
                    cmd(0, resource, kind, "forged")

    def test_wrong_kind_resource_pairs_and_record_resources_fail(self) -> None:
        invalid_pairs = (
            ("relation_workspace_1rw", "epoch_record_write"),
            ("acc_bank_0_1rw", "epoch_record_read"),
            ("epoch_slot_1rw", "relation_read"),
            ("fifo2_enq", "fifo_dequeue"),
            ("fifo2_deq", "fifo_enqueue"),
            ("context_prepare_1rw", "drain_read"),
            ("drain_read_1rw", "context_prepare"),
        )
        for resource, kind in invalid_pairs:
            with self.subTest(resource=resource, kind=kind):
                with self.assertRaises(ValueError):
                    cmd(0, resource, kind, "mismatch")
        with self.assertRaises(ValueError):
            PhaseTrace(1, ()).count_kind("reviewer_private_kind")

    def test_cycle_duration_and_record_count_reject_bool_float_and_range(self) -> None:
        for cycle in (True, False, 0.0, 1.5, -1):
            with self.subTest(field="cycle", value=cycle):
                with self.assertRaises(ValueError):
                    cmd(cycle, "relation_workspace_1rw", "relation_read", "r0")
        for duration in (True, False, 0.0, 1.5, 0, -1):
            with self.subTest(field="duration", value=duration):
                with self.assertRaises(ValueError):
                    PhaseTrace(duration, ())
        with self.assertRaises(ValueError):
            PhaseTrace(
                1,
                (cmd(1, "relation_workspace_1rw", "relation_read", "r0"),),
            )

        head = fixture().heads[0]
        for records in (True, False, 1.0, -1, 451):
            with self.subTest(field="epoch_records", value=records):
                with self.assertRaises(ValueError):
                    replace(head, epoch_records=records)

    def test_schedule_counters_and_event_cycles_are_strict_integers(self) -> None:
        for start, end in ((True, 1), (0.0, 1), (0, True), (0, 1.0), (1, 1)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    Event("r", "k", start, end, "i")

        result = evaluate_window(fixture())["c3_erep_s2"]
        invalid_updates = (
            {"cycles": True},
            {"epoch_record_writes": 1.0},
            {"epoch_record_reads": False},
            {"acc_contexts": 0},
            {"acc_contexts": 3},
            {"epoch_slots": -1},
            {"epoch_slots": 3},
        )
        for update in invalid_updates:
            with self.subTest(update=update):
                with self.assertRaises(ValueError):
                    replace(result, **update)

        with self.assertRaises(ValueError):
            ScheduleResult("c", 0, (), 0, 0, True, 0)

    def test_input_head_identity_and_order_are_explicit_and_strict(self) -> None:
        window = fixture()
        for input_head in (True, 0.0, -1):
            with self.subTest(input_head=input_head):
                with self.assertRaises(ValueError):
                    replace(window.heads[0], input_head=input_head)

        duplicate = replace(window.heads[1], input_head=0)
        out_of_range = replace(window.heads[2], input_head=3)
        invalid_heads = (
            (window.heads[1], window.heads[0], window.heads[2]),
            (window.heads[0], duplicate, window.heads[2]),
            (window.heads[0], window.heads[1], out_of_range),
        )
        for heads in invalid_heads:
            with self.subTest(heads=tuple(head.input_head for head in heads)):
                with self.assertRaises(ValueError):
                    replace(window, heads=heads)

    def test_output_tile_identity_and_order_are_explicit_and_strict(self) -> None:
        window = fixture()
        invalid_tiles = (
            (0, 2, 1),
            (0, 1, 1),
            (0, 1, 3),
            (0, True, 2),
            (0, 1, 2.0),
        )
        for output_tiles in invalid_tiles:
            with self.subTest(output_tiles=output_tiles):
                with self.assertRaises(ValueError):
                    replace(window, output_tiles=output_tiles)

    def test_all_candidate_ledgers_are_resource_legal(self) -> None:
        rows = evaluate_window(fixture())
        self.assertEqual(
            set(rows),
            {
                "c0_direct_serial",
                "c1_reuse_only_s2",
                "c2_overlap_only",
                "c3_erep_s2",
            },
        )
        self.assertTrue(all(row.cycles > 0 for row in rows.values()))
        self.assertTrue(all(name == row.candidate for name, row in rows.items()))
        self.assertEqual(rows["c3_erep_s2"].acc_contexts, 2)
        self.assertEqual(rows["c3_erep_s2"].epoch_slots, 2)

    def test_slot_ownership_has_no_sealed_gap(self) -> None:
        result = evaluate_window(fixture())["c3_erep_s2"]
        for slot in (0, 1):
            task_ids = {
                event.identity
                for event in result.events
                if event.resource == f"slot_{slot}_owner"
            }
            for task_id in task_ids:
                rows = sorted(
                    (
                        event
                        for event in result.events
                        if event.resource == f"slot_{slot}_owner"
                        and event.identity == task_id
                    ),
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
            prepare_event = next(
                event
                for event in prepares
                if event.identity == f"tile_{tile}_prepare"
            )
            drain_event = next(
                event
                for event in drains
                if event.identity == f"tile_{tile}_drain"
            )
            self.assertEqual(owner.start, prepare_event.start)
            self.assertEqual(owner.end, drain_event.end)

    def test_s2_remainder_rebuilds_final_width_one_stripe(self) -> None:
        records = 2 + 1 + 2
        for candidate in ("c1_reuse_only_s2", "c3_erep_s2"):
            with self.subTest(candidate=candidate):
                result = evaluate_window(fixture())[candidate]
                self.assertEqual(result.epoch_record_writes, records * 2)
                self.assertEqual(result.epoch_record_reads, records * 3)

                final_executes = sorted(
                    (
                        event
                        for event in result.events
                        if event.kind == "EXECUTE" and "_tile_2_" in event.identity
                    ),
                    key=lambda event: event.start,
                )
                self.assertEqual(len(final_executes), 3)
                self.assertEqual(
                    [
                        int(event.identity.split("_head_")[1].split("_")[0])
                        for event in final_executes
                    ],
                    [0, 1, 2],
                )
                final_owner = next(
                    event
                    for event in result.events
                    if event.kind == "OWNED" and event.identity == "tile_2_context"
                )
                self.assertEqual(final_owner.resource, "context_0_owner")
                self.assertFalse(any("tile_3" in event.identity for event in result.events))

    def test_fifo_bypass_overflow_underflow_and_1rw_collision(self) -> None:
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
            PhaseTrace(1, (cmd(0, "fifo2_deq", "fifo_dequeue", "a"),))
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
        rows = evaluate_window(fixture())
        for candidate in ("c1_reuse_only_s2", "c2_overlap_only", "c3_erep_s2"):
            executes = [event for event in rows[candidate].events if event.kind == "EXECUTE"]
            for output_tile in range(3):
                tile_rows = sorted(
                    (
                        event
                        for event in executes
                        if f"_tile_{output_tile}_" in event.identity
                    ),
                    key=lambda event: event.start,
                )
                observed_heads = [
                    int(event.identity.split("_head_")[1].split("_")[0])
                    for event in tile_rows
                ]
                self.assertEqual(observed_heads, [0, 1, 2])

        directs = [event for event in rows["c0_direct_serial"].events if event.kind == "DIRECT"]
        for output_tile in range(3):
            observed_heads = [
                int(event.identity.split("_head_")[1].split("_")[0])
                for event in directs
                if event.identity.startswith(f"tile_{output_tile}_")
            ]
            self.assertEqual(observed_heads, [0, 1, 2])

    def test_epoch_record_identity_and_pair_counts_are_exact(self) -> None:
        head = fixture().heads[0]
        wrong_relation_identity = PhaseTrace(
            head.fill.duration,
            tuple(
                replace(command, identity="wrong")
                if command.kind is CommandKind.RELATION_READ
                else command
                for command in head.fill.commands
            ),
        )
        with self.assertRaisesRegex(ValueError, "relation-read/epoch-write"):
            replace(head, fill=wrong_relation_identity)

        wrong_read_identity = PhaseTrace(
            5,
            (
                cmd(0, "epoch_slot_1rw", "epoch_record_read", "wrong0"),
                cmd(0, "fifo2_enq", "fifo_enqueue", "wrong0"),
                cmd(0, "fifo2_deq", "fifo_dequeue", "wrong0"),
                cmd(1, "epoch_slot_1rw", "epoch_record_read", "wrong1"),
                cmd(1, "fifo2_enq", "fifo_enqueue", "wrong1"),
                cmd(1, "fifo2_deq", "fifo_dequeue", "wrong1"),
            ),
        )
        with self.assertRaises(ValueError):
            replace(
                head,
                execute_by_tile=(wrong_read_identity,) + head.execute_by_tile[1:],
            )

        original = head.execute_by_tile[0]
        for kind, message in (
            (CommandKind.FIFO_ENQUEUE, "epoch-read/FIFO-enqueue"),
            (CommandKind.FIFO_DEQUEUE, "epoch-read/FIFO-dequeue"),
        ):
            wrong_fifo_identity = PhaseTrace(
                original.duration,
                tuple(
                    replace(command, identity="wrong")
                    if command.kind is kind
                    else command
                    for command in original.commands
                ),
            )
            with self.subTest(kind=kind.value):
                with self.assertRaisesRegex(ValueError, message):
                    replace(
                        head,
                        execute_by_tile=(wrong_fifo_identity,)
                        + head.execute_by_tile[1:],
                    )


if __name__ == "__main__":
    unittest.main()
