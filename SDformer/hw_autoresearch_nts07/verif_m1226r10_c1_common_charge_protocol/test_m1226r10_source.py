#!/usr/bin/env python3
"""Positive and fail-closed mutation tests for M1226/R10 source."""

import importlib.util
import unittest
from pathlib import Path


CHECKER_PATH = Path(__file__).with_name("check_m1226r10_source.py")
SPEC = importlib.util.spec_from_file_location("check_m1226r10_source",
                                              CHECKER_PATH)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class M1226SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = (checker.Path(__file__).with_name(
            "tb_m1226r10_m1162_common_charge_protocol_unit_delay_r10.sv")
            .read_text())

    def assert_rejected(self, mutant: str, reason: str) -> None:
        self.assertTrue(checker.audit_text(mutant), reason)

    def replace_after(self, anchor: str, old: str, new: str) -> str:
        start = self.text.index(anchor)
        pos = self.text.index(old, start)
        return self.text[:pos] + new + self.text[pos + len(old):]

    def test_canonical_structure_passes(self) -> None:
        self.assertEqual(checker.audit_text(self.text), [])

    def test_request_ready_retirement_removal_is_rejected(self) -> None:
        mutant = self.replace_after(
            "task automatic serve_normal_beat(",
            "weight_req_ready = 1'b0;\n"
            "            psum_req_ready = 1'b0;",
            "weight_req_ready = 1'b1;\n"
            "            psum_req_ready = 1'b1;")
        self.assert_rejected(mutant, "ready retirement removal was accepted")

    def test_request_overshoot_gate_removal_is_rejected(self) -> None:
        mutant = self.replace_after(
            "task automatic serve_normal_beat(",
            "weight_fire_count > w0 + 1",
            "weight_fire_count > w0 + 2")
        self.assert_rejected(mutant, "one-fire overshoot weakening accepted")

    def test_extra_response_posedge_restoration_is_rejected(self) -> None:
        anchor = "// Exact-accept retirement: no extra posedge is permitted here.\n"
        mutant = self.text.replace(anchor + "            @(negedge clk_core);",
            anchor + "            @(posedge clk_core); #1ps;\n"
            "            @(negedge clk_core);", 1)
        self.assert_rejected(mutant, "extra response posedge was accepted")

    def test_response_stability_gate_removal_is_rejected(self) -> None:
        mutant = self.text.replace("weight_data !== '0",
                                   "weight_data === '0", 1)
        self.assert_rejected(mutant, "unstable response service accepted")

    def test_beat_two_retirement_gate_removal_is_rejected(self) -> None:
        mutant = self.text.replace(
            "|| dut.weight_request_accepted_q\n"
            "                        || dut.psum_request_accepted_q",
            "|| dut.psum_request_accepted_q", 1)
        self.assert_rejected(mutant, "wrapper retirement removal accepted")

    def test_normal_request_dump_removal_is_rejected(self) -> None:
        mutant = self.text.replace('"normal_issue_request"',
                                   '"normal_issue"', 1)
        self.assert_rejected(mutant, "normal request dump removal accepted")

    def test_normal_response_dump_removal_is_rejected(self) -> None:
        mutant = self.text.replace('"normal_response_accept"',
                                   '"normal_response"', 1)
        self.assert_rejected(mutant, "normal response dump removal accepted")

    def test_zero_sva_gate_removal_is_rejected(self) -> None:
        mutant = self.text.replace("zero_sva_failures_required=true",
                                   "zero_sva_failures_required=false", 1)
        self.assert_rejected(mutant, "zero-SVA-failure gate removal accepted")

    def test_workload_mutation_is_rejected(self) -> None:
        mutant = self.text.replace("16'h0003", "16'h0001", 1)
        self.assert_rejected(mutant, "two-source normal row mutation accepted")

    def test_claim_mutation_is_rejected(self) -> None:
        mutant = self.text.replace("timing_verified=false",
                                   "timing_verified=true", 1)
        self.assert_rejected(mutant, "claim mutation accepted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
