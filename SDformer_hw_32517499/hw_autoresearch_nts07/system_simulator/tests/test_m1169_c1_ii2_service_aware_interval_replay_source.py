#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Bounded tests for the M1169 source-only II=2 interval recurrence."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import random
import sys
import unittest

SOURCE = (Path(__file__).resolve().parents[1] / "scripts" /
          "build_m1169_c1_ii2_service_aware_interval_replay_source.py")
SPEC = importlib.util.spec_from_file_location("m1169_source_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class M1169IntervalReplayTest(unittest.TestCase):
    def test_floor_quota_conservation_and_distribution(self) -> None:
        counts = {}
        total = 0
        previous = 0
        for task in range(M.TASKS):
            begin, end = M.floor_quota(task, M.TASKS, M.EVENTS_PER_AXIS)
            self.assertEqual(begin, previous)
            previous = end
            count = end - begin
            counts[count] = counts.get(count, 0) + 1
            total += count
        self.assertEqual(previous, M.EVENTS_PER_AXIS)
        self.assertEqual(total, M.EVENTS_PER_AXIS)
        self.assertEqual(counts, {87: 616_896, 88: 195_264})

    def test_closed_form_matches_explicit_random_gaps_and_overlaps(self) -> None:
        rng = random.Random(0x1169)
        for _ in range(2_000):
            intervals = []
            first = rng.randrange(0, 30)
            for _task in range(rng.randrange(1, 10)):
                first += rng.randrange(0, 16)
                intervals.append((first, rng.randrange(1, 12)))
            explicit = M.explicit_beat_simulation(intervals)
            previous = None
            offset = 0
            for requested, beats in intervals:
                result = M.advance_zero_stall_ii2(previous, requested, beats)
                self.assertEqual(result.first_completed_cycle, explicit[offset])
                self.assertEqual(result.last_completed_cycle,
                                 explicit[offset + beats - 1])
                previous = result.last_completed_cycle
                offset += beats

    def test_stalls_never_improve_fixed_model_and_are_not_silently_admitted(self) -> None:
        intervals = [(0, 5), (0, 4), (40, 3)]
        zero = M.explicit_beat_simulation(intervals)
        for index in range(len(zero)):
            request = [0] * len(zero)
            response = [0] * len(zero)
            request[index] = index + 1
            response[index] = index + 2
            request_result = M.explicit_beat_simulation(intervals, request, ())
            response_result = M.explicit_beat_simulation(intervals, (), response)
            self.assertTrue(all(a >= b for a, b in zip(request_result, zero)))
            self.assertTrue(all(a >= b for a, b in zip(response_result, zero)))
            self.assertNotEqual(request_result, zero)
            self.assertNotEqual(response_result, zero)

    def test_task_major_axis_minor_and_component_ratio(self) -> None:
        replay = M.IntervalReplay(4, 18, M.AXES)
        starts = {
            "candidate": (0, 5, 10, 15),
            "strongest_zero": (0, 12, 24, 36),
            "same_coordinate_bit": (0, 9, 18, 27),
        }
        for task in range(4):
            for axis in M.AXES:
                replay.consume_interval(axis, task, starts[axis][task])
        value = replay.finalize()
        self.assertEqual(value["status"],
                         "PASS_EXACT_INTERVAL_RECURRENCE__COMPONENT_SCHEDULE_ONLY")
        self.assertGreater(value["component_schedule_ratios"]
                           ["strongest_zero_over_candidate"]["ratio_decimal"], 1.0)
        self.assertFalse(value["claim_boundary"]["rtl_cycles"])
        self.assertFalse(value["claim_boundary"]["system_speedup"])
        self.assertEqual(value["expanded_beats"], 0)

    def test_order_drop_duplicate_and_bool_rejected(self) -> None:
        for action in (
            lambda: M.IntervalReplay(2, 4, M.AXES).consume_interval(
                "strongest_zero", 0, 0),
            lambda: M.IntervalReplay(2, 4, M.AXES).consume_interval(
                "candidate", 1, 0),
            lambda: M.floor_quota(True, 2, 4),
            lambda: M.advance_zero_stall_ii2(None, True, 1),
            lambda: M.advance_zero_stall_ii2(None, 0, True),
        ):
            with self.assertRaises(M.Failure):
                action()

    def test_exact_schedule_schema_and_provenance(self) -> None:
        source = hashlib.sha256(b"m1169-test-task").hexdigest()
        mapping = {
            "axis": "candidate", "chunk": 0, "operator": 0,
            "partition": 0, "requested_cycle_first": 7, "sample": 0,
            "source_task_provenance_sha256": source,
            "task_sequence_ordinal": 0,
        }
        mapping["schedule_record_provenance_sha256"] = M.record_provenance(
            "candidate", 0, 0, 0, 0, 0, 7, source)
        record = M.ScheduleRecord.from_mapping(mapping)
        self.assertEqual(record.requested_cycle_first, 7)
        for mutation in ("extra", "provenance", "coordinate", "bool"):
            attacked = dict(mapping)
            if mutation == "extra":
                attacked["extra"] = 0
            elif mutation == "provenance":
                attacked["schedule_record_provenance_sha256"] = "0" * 64
            elif mutation == "coordinate":
                attacked["task_sequence_ordinal"] = 1
            else:
                attacked["requested_cycle_first"] = True
            with self.assertRaises(M.Failure):
                M.ScheduleRecord.from_mapping(attacked)

    def test_bounded_oracle_is_source_only(self) -> None:
        value = M.bounded_source_oracle()
        self.assertEqual(value["production_schedule_opened"], False)
        self.assertEqual(value["production_records_consumed"], 0)
        self.assertEqual(value["production_beats_expanded"], 0)
        self.assertEqual(value["m1161_result_consumed"], False)
        self.assertGreaterEqual(value["closed_form_explicit_cases"], 1_000)
        with self.assertRaisesRegex(M.Failure, "sealed M1161"):
            M.ProductionReplay()


if __name__ == "__main__":
    unittest.main()
