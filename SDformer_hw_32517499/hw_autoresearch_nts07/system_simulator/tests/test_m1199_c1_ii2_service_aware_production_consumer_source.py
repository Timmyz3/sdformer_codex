#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Bounded source tests for M1199; never invoke the production entry."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

SOURCE = (Path(__file__).resolve().parents[1] / "scripts" /
          "run_m1199_c1_ii2_service_aware_production_consumer_one_shot_source.py")
SPEC = importlib.util.spec_from_file_location("m1199_source_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class M1199ProductionConsumerSourceTest(unittest.TestCase):
    def test_bounded_oracle_does_not_mutate_production_namespace(self) -> None:
        before = M._namespace_paths()
        self.assertEqual(before, ())
        value = M.bounded_source_self_test()
        self.assertEqual(value["status"],
                         "PASS_M1199_BOUNDED_CONSUMER_ORACLE__PRODUCTION_STOP")
        self.assertEqual(value["production_schedule_opened"], False)
        self.assertEqual(value["production_records_consumed"], 0)
        self.assertEqual(value["production_namespace_mutated"], False)
        self.assertEqual(value["terminal"]["expanded_beats"], 0)
        self.assertGreater(value["terminal"]["component_schedule_ratios"]
                           ["strongest_zero_over_candidate"]["ratio_decimal"], 1.0)
        self.assertEqual(before, M._namespace_paths())

    def test_frozen_chain_preflight_is_metadata_only(self) -> None:
        value = M.source_preflight(True)
        self.assertEqual(value["m1161_result_outer_seal_file_sha256"],
                         M.M1161_OUTER_FILE_SHA)
        self.assertEqual(value["m1196_outer_seal_file_sha256"],
                         M.M1196_OUTER_FILE_SHA)
        self.assertEqual(value["m1169_source_sha256"], M.M1169_SOURCE_SHA)
        self.assertEqual(value["m1170_outer_seal_file_sha256"],
                         M.M1170_OUTER_FILE_SHA)
        self.assertEqual(value["m1141_records"], M.EXPECTED_RECORDS)
        self.assertEqual(value["production_schedule_opened"], False)
        self.assertEqual(value["production_records_consumed"], 0)
        self.assertEqual(value["m1196_score"], 99)

    def test_strict_json_rejects_duplicate_and_nonfinite(self) -> None:
        with self.assertRaises(M.Failure):
            M.strict_json_bytes(b'{"axis":1,"axis":2}')
        with self.assertRaises(M.Failure):
            M.strict_json_bytes(b'{"axis":NaN}')

    def test_resource_guard_and_same_uid_guard_are_live(self) -> None:
        value = M.resource_preflight()
        self.assertGreaterEqual(value["cpus"], M.MIN_CPUS)
        self.assertGreaterEqual(value["mem_available_bytes"], M.MIN_MEM_AVAILABLE)
        self.assertGreaterEqual(value["commit_headroom_bytes"],
                                M.MIN_COMMIT_HEADROOM)
        self.assertGreaterEqual(value["disk_free_bytes"], M.MIN_DISK_FREE)
        self.assertEqual(value["same_uid_conflicts"], 0)

    def test_one_shot_order_and_claim_boundary_are_explicit(self) -> None:
        text = SOURCE.read_text(encoding="utf-8")
        marker = text.index('phase = "CONSUME_PERSISTENT_ATTEMPT_BEFORE_SCHEDULE_OPEN"')
        schedule = text.index('phase = "STREAM_M1141_TASK_INTERVALS_TO_M1169_RECURRENCE"')
        self.assertLess(marker, schedule)
        self.assertIn("M1199_SINGLE_ATTEMPT_CONSUMED__NO_AUTOMATIC_RETRY", text)
        self.assertIn('"per_event_output_written": False', text)
        self.assertIn('"component_weight_service_schedule_only": True', text)
        self.assertIn('"rtl_cycles_or_system_speedup": False', text)

    def test_nonzero_argument_is_rejected_before_production_main(self) -> None:
        original = list(sys.argv)
        original_main = M.production_main
        called = []
        try:
            sys.argv[:] = [str(SOURCE), "unexpected"]
            M.production_main = lambda: called.append(True)
            with self.assertRaisesRegex(M.Failure, "zero arguments"):
                M.main()
            self.assertEqual(called, [])
        finally:
            sys.argv[:] = original
            M.production_main = original_main

    def test_production_entry_not_called_by_tests(self) -> None:
        self.assertFalse(M.RESULT.exists())
        self.assertFalse(M.ATTEMPT.exists())
        self.assertFalse(M.LOCK.exists())
        self.assertEqual(tuple(M.RESULTS.glob(M.FAILURE_PREFIX + "*")), ())


if __name__ == "__main__":
    unittest.main()
