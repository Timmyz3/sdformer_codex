#!/usr/bin/env python3
"""Directed and adversarial tests for the source-only M1056 model."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


HW = Path(__file__).resolve().parents[2]
SOURCE = HW / "system_simulator/scripts/run_m1056_c1_exact_1rw_arbitration_replay_source.py"
SPEC = importlib.util.spec_from_file_location("m1056_test_source", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load M1056 source")
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class M1056Tests(unittest.TestCase):
    def event(self, name, order, ready, address, dependencies=()):
        return M.PortEvent(name, 0, order, 0, 0, address,
                           "WRITE" if dependencies else "READ", ready,
                           dependencies)

    def test_small_oracle(self):
        self.assertEqual(
            M.small_oracle()["status"],
            "PASS_M1056_SMALL_ORACLE__NO_FULL_REPLAY_NO_EDA",
        )

    def test_multiplicity_two_different_addresses_share_port(self):
        result = M.arbitrate_group([
            self.event("a", 0, 3, 1), self.event("b", 1, 3, 65)
        ], 0)
        self.assertEqual([result.grants[key].cycle for key in ("a", "b")], [3, 4])
        self.assertEqual(result.queue_peak, 2)

    def test_multiplicity_three_is_program_order_deterministic(self):
        events = [self.event("c", 2, 1, 2), self.event("a", 0, 1, 0),
                  self.event("b", 1, 1, 1)]
        result = M.arbitrate_group(events, 0)
        self.assertEqual(result.grant_order, ["a", "b", "c"])
        self.assertEqual([result.grants[key].cycle for key in result.grant_order],
                         [1, 2, 3])

    def test_cross_task_input_order_is_not_append_order(self):
        result = M.arbitrate_group([
            M.PortEvent("t0", 0, 0, 0, 0, 1, "READ", 9),
            M.PortEvent("t1", 1, 32, 0, 0, 2, "READ", 2),
        ], 0)
        self.assertEqual(result.grant_order, ["t1", "t0"])

    def test_same_address_raw_dependency(self):
        result = M.arbitrate_group([
            M.PortEvent("r", 1, 2, 0, 0, 5, "READ", 0,
                        (M.Dependency("w", 1),)),
            M.PortEvent("w", 0, 1, 0, 0, 5, "WRITE", 4),
        ], 0)
        self.assertEqual(result.grants["w"].cycle, 4)
        self.assertEqual(result.grants["r"].cycle, 5)

    def test_dependency_cycle_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "deadlock"):
            M.arbitrate_group([
                M.PortEvent("a", 0, 0, 0, 0, 1, "WRITE", 0,
                            (M.Dependency("b", 0),)),
                M.PortEvent("b", 0, 1, 0, 0, 2, "WRITE", 0,
                            (M.Dependency("a", 0),)),
            ], 0)

    def test_duplicate_program_order_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "identity drift"):
            M.arbitrate_group([
                self.event("a", 0, 0, 1), self.event("b", 0, 0, 2)
            ], 0)

    def test_delay_cascades_and_is_not_naive_plus_conflicts(self):
        plans = [M.TaskPlan(0, 0, 8, 3), M.TaskPlan(1, 0, 8, 3)]
        nominal = M.nominal_m1016_sequence_cycles(plans)
        result = M.replay_task_sequence(plans)
        self.assertEqual(nominal, 20)
        self.assertEqual(result.sample_cycles_after_commit, 22)
        self.assertEqual(result.total_nominal_excess_accesses, 16)
        self.assertNotEqual(result.sample_cycles_after_commit - nominal,
                            result.total_nominal_excess_accesses)
        self.assertEqual([task.work_start for task in result.tasks], [0, 11])

    def test_no_conflict_is_m1016_pipeline_identity(self):
        plans = [M.TaskPlan(0, 3, 16, 0), M.TaskPlan(1, 5, 16, 1)]
        result = M.replay_task_sequence(plans, commit_cycles=19)
        self.assertEqual(result.sample_cycles_after_commit,
                         M.nominal_m1016_sequence_cycles(plans, 19))
        self.assertEqual(result.total_nominal_excess_accesses, 0)

    def test_capacity_bytes_and_port_feasibility_are_separate(self):
        plan = [M.TaskPlan(0, 0, 8, 0)]
        fits = M.replay_task_sequence(plan, capacity_bytes=214_912)
        too_large = M.replay_task_sequence(plan, capacity_bytes=245_761)
        self.assertTrue(fits.capacity_bytes_pass)
        self.assertTrue(fits.port_feasibility_pass)
        self.assertFalse(too_large.capacity_bytes_pass)
        self.assertTrue(too_large.port_feasibility_pass)

    def test_three_design_asymmetry_rejected(self):
        receipt = {"task": 0, "counts": {
            "psum": 16, "weight": 1, "source": 2, "dma": 0, "commit": 0,
        }}
        receipts = {name: [receipt] for name in M.DESIGNS}
        configs = {name: M.ArbiterConfig() for name in M.DESIGNS}
        self.assertEqual(
            M.validate_three_design_common_coordinate(receipts, configs)["status"],
            "PASS_M1056_THREE_DESIGN_COMMON_COORDINATE",
        )
        receipts["candidate"] = [{"task": 0, "counts": {
            "psum": 15, "weight": 1, "source": 2, "dma": 0, "commit": 0,
        }}]
        with self.assertRaisesRegex(RuntimeError, "asymmetric"):
            M.validate_three_design_common_coordinate(receipts, configs)

    def test_three_design_replay_uses_same_arbiter_coordinate(self):
        receipt = {"task": 0, "counts": {
            "psum": 16, "weight": 1, "source": 2, "dma": 0, "commit": 0,
        }}
        result = M.replay_three_design_sequences(
            {
                "candidate": [M.TaskPlan(0, 0, 8, 0)],
                "strongest_zero": [M.TaskPlan(0, 0, 16, 0)],
                "same_coordinate_bit": [M.TaskPlan(0, 0, 16, 0)],
            },
            {name: [receipt] for name in M.DESIGNS},
        )
        self.assertEqual(result["status"],
                         "PASS_M1056_THREE_DESIGN_EXACT_1RW_REPLAY")
        self.assertTrue(result["capacity_bytes_pass"])
        self.assertTrue(result["port_feasibility_pass"])

    def test_claim_boundary_remains_closed(self):
        boundary = M.small_oracle()["claim_boundary"]
        self.assertTrue(boundary["source_only"])
        for key in ("full_51840000_replay", "capacity_only_214912B_admitted",
                    "matched_cycles_admitted", "speedup_admitted", "rtl_cycles",
                    "paper_ppa_ready"):
            self.assertFalse(boundary[key])


if __name__ == "__main__":
    unittest.main()
