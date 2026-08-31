#!/usr/bin/env python3
"""Bounded source tests for M1162; no simulator or EDA invocation."""
import importlib.util
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE.parents[1] / "verif_m1162_c1_common_charge_protocol/static_check_m1162_common_charge_protocol_source.py"
SPEC = importlib.util.spec_from_file_location("m1162_source_checker_tested", CHECKER)
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class M1162ProtocolSourceTest(unittest.TestCase):
    def test_frozen_identity_and_wrapper_structure(self):
        self.assertEqual(M.check_frozen_identity()["docs359"], M.DOC359_SHA)
        value = M.check_wrapper()
        self.assertTrue(value["request_valid_ready_independent"])
        self.assertEqual(value["total_added_state_bits"], 40)

    def test_weight_then_psum_partial_accept_no_duplicate(self):
        m = M.ProtocolModel()
        out = m.step(issue=True, first=True, weight_ready=True)
        self.assertTrue(out["weight_valid"] and out["psum_valid"])
        for _ in range(4):
            out = m.step(issue=True, first=True, weight_ready=True)
            self.assertFalse(out["weight_valid"])
            self.assertTrue(out["psum_valid"])
        m.step(issue=True, first=True, weight_ready=True, psum_ready=True)
        self.assertEqual((m.weight_fires, m.psum_fires), (1, 1))

    def test_psum_then_weight_partial_accept_no_duplicate(self):
        m = M.ProtocolModel()
        m.step(issue=True, first=True, psum_ready=True)
        for _ in range(4):
            out = m.step(issue=True, first=True, psum_ready=True)
            self.assertTrue(out["weight_valid"])
            self.assertFalse(out["psum_valid"])
        m.step(issue=True, first=True, weight_ready=True, psum_ready=True)
        self.assertEqual((m.weight_fires, m.psum_fires), (1, 1))

    def test_skewed_responses_and_backpressure(self):
        m = M.ProtocolModel()
        m.step(issue=True, first=True, weight_ready=True, psum_ready=True)
        out = m.step(issue=True, first=True, weight_response=True,
                     core_ready=False)
        self.assertFalse(out["core_valid"])
        out = m.step(issue=True, first=True, weight_response=True,
                     psum_response=True, core_ready=False)
        self.assertTrue(out["core_valid"])
        self.assertFalse(out["weight_ready_out"] or out["psum_ready_out"])
        out = m.step(issue=True, first=True, weight_response=True,
                     psum_response=True, core_ready=True)
        self.assertTrue(out["response_accept"])

    def test_nonfirst_never_requests_or_accepts_psum(self):
        m = M.ProtocolModel()
        out = m.step(issue=True, first=False, weight_ready=True,
                     psum_ready=True)
        self.assertFalse(out["psum_valid"])
        self.assertEqual(m.psum_fires, 0)
        m.step(issue=True, first=False, weight_response=True)
        self.assertTrue(m.active is False)

    def test_early_spurious_and_nonfirst_psum_response_sticky(self):
        m = M.ProtocolModel()
        m.step(issue=True, first=True, weight_ready=True,
               weight_response=True)
        self.assertTrue(m.fault)
        m.reset()
        m.step(issue=False, weight_response=True)
        self.assertTrue(m.fault)
        m.reset()
        m.step(issue=True, first=False, weight_ready=True)
        m.step(issue=True, first=False, psum_response=True)
        self.assertTrue(m.fault)

    def test_cancellation_mutation_and_reset(self):
        m = M.ProtocolModel()
        m.step(issue=True, first=True, weight_ready=True)
        m.step(issue=False, first=True)
        self.assertTrue(m.fault)
        m.reset()
        self.assertFalse(m.active or m.fault)
        m.step(issue=True, first=True, mutate=True)
        self.assertTrue(m.fault)

    def test_tb_plan_and_contract(self):
        self.assertEqual(M.check_tb_and_plan()["directed_cover_classes"], 12)
        self.assertTrue(M.check_contract()["no_performance_inheritance"])


if __name__ == "__main__":
    unittest.main()
