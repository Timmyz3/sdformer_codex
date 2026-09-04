#!/usr/bin/env python3
"""Small, source-only tests for M1007.  Never stream the frozen M410 ledger."""
import importlib.util
import inspect
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "scripts/m1007_c1_matched_common_charge_address_replay_source.py"
SPEC = importlib.util.spec_from_file_location("m1007_source_tested", SOURCE)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class M1007Test(unittest.TestCase):
    def test_small_oracle_matches_frozen_m505(self):
        value = M.small_oracle()
        self.assertEqual(value["status"], "PASS_M1007_SMALL_ORACLE__NO_FULL_REPLAY")
        self.assertFalse(value["full_51840000_replayed"])

    def test_parent_trace_is_one_operation_per_cycle(self):
        events = list(M.parent_cycle_trace([1, 2, 3, 7, 15, 31, 0]))
        self.assertTrue(events)
        for event in events:
            self.assertIn(event["op"], ("IDLE", "READ", "WRITE"))
            self.assertLessEqual(int(event["op"] in ("READ", "WRITE")), 1)
        self.assertEqual([e["cycle"] for e in events], list(range(len(events))))

    def test_parent_trace_matches_frozen_fields(self):
        masks = np.asarray([1, 3, 5], dtype=np.uint16)
        ours = M.parent_summary(list(M.parent_cycle_trace(masks)))
        frozen = M.M505.simulate_liveness_task(masks, False)
        self.assertEqual(ours["cycles"], frozen["liveness_cycles"])
        self.assertEqual(ours["macro_reads"], frozen["macro_reads"])
        self.assertEqual(ours["macro_writes"], frozen["macro_writes"])
        self.assertEqual(ours["forwarded_reads"], frozen["forwarded_reads"])

    def test_common_charge_accepts_only_timestamp_shift(self):
        common = [{"resource": resource, "op": "READ", "bank": 0,
                   "address": i, "bytes": i + 1, "transaction": resource,
                   "cycle": i}
                  for i, resource in enumerate(M.COMMON_RESOURCES)]
        designs = {name: [dict(event, cycle=event["cycle"] + 100 * index)
                          for event in common]
                   for index, name in enumerate(M.DESIGNS)}
        policy = {resource: {"mode": "include_both", "capacity_bytes": 64,
                             "ports": "1RW", "latency_cycles": 1}
                  for resource in M.COMMON_RESOURCES}
        value = M.verify_matched_common_charge(designs, policy)
        self.assertEqual(value["status"], "PASS_M1007_MATCHED_COMMON_CHARGE")
        self.assertTrue(value["cycle_merge_pending"])

    def test_common_charge_rejects_asymmetry(self):
        common = [{"resource": resource, "op": "READ", "bank": 0,
                   "address": 0, "bytes": 1, "transaction": resource,
                   "cycle": 0} for resource in M.COMMON_RESOURCES]
        designs = {name: list(common) for name in M.DESIGNS}
        designs["candidate"] = designs["candidate"][:-1]
        policy = {resource: {"mode": "include_both", "capacity_bytes": 1,
                             "ports": "1RW", "latency_cycles": 1}
                  for resource in M.COMMON_RESOURCES}
        with self.assertRaisesRegex(RuntimeError, "asymmetric common charge"):
            M.verify_matched_common_charge(designs, policy)

    def test_paired_psum_conflict_blocks_capacity_admission(self):
        psum = [{"cycle": 2, "op": "READ", "bank": 0, "address": 1},
                {"cycle": 2, "op": "WRITE", "bank": 1, "address": 65}]
        value = M.packing_summary(psum, [], coverage_complete=True)
        self.assertEqual(value["psum_depth_packed_pair"]["conflict_cycles"], 1)
        self.assertFalse(value["capacity_only_214912B_admitted"])

    def test_weight_half_slot_overlap_blocks_capacity_admission(self):
        weight = [{"cycle": 4, "op": "READ", "bank": 0, "address": 1},
                  {"cycle": 4, "op": "READ", "bank": 1, "address": 2}]
        value = M.packing_summary([], weight, coverage_complete=True)
        self.assertEqual(value["weight_half_slot_overlap_cycles"], 1)
        self.assertFalse(value["capacity_only_214912B_admitted"])

    def test_incomplete_coverage_blocks_even_conflict_free_case(self):
        value = M.packing_summary([], [], coverage_complete=False)
        self.assertTrue(value["psum_depth_packed_pair"]["one_rw_legal"])
        self.assertFalse(value["capacity_only_214912B_admitted"])

    def test_full_stream_is_generator_and_not_called(self):
        self.assertTrue(inspect.isgeneratorfunction(M.stream_parent_memh))
        stream = M.stream_parent_memh()
        self.assertTrue(inspect.isgenerator(stream))
        stream.close()


if __name__ == "__main__":
    unittest.main()
