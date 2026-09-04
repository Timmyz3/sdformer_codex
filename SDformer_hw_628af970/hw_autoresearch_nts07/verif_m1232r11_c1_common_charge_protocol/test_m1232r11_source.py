#!/usr/bin/env python3
"""Positive and fail-closed mutation tests for M1232/R11 source."""

import importlib.util
import unittest
from pathlib import Path


CHECKER_PATH = Path(__file__).with_name("check_m1232r11_source.py")
SPEC = importlib.util.spec_from_file_location("check_m1232r11_source",
                                              CHECKER_PATH)
assert SPEC is not None and SPEC.loader is not None
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class M1232SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        here = Path(__file__).resolve().parent
        cls.text = (here /
            "tb_m1232r11_m1162_common_charge_protocol_unit_delay_r11.sv"
            ).read_text()
        cls.r10 = (here.parent /
            "verif_m1226r10_c1_common_charge_protocol" /
            "tb_m1226r10_m1162_common_charge_protocol_unit_delay_r10.sv"
            ).read_text()

    def assert_rejected(self, mutant: str, reason: str) -> None:
        self.assertTrue(checker.audit_text(mutant, self.r10), reason)

    def replace_in_task(self, task_name: str, old: str, new: str) -> str:
        original = checker.extract_task(self.text, task_name)
        self.assertIsNotNone(original)
        assert original is not None
        self.assertIn(old, original)
        mutated = original.replace(old, new, 1)
        return self.text.replace(original, mutated, 1)

    def test_canonical_structure_passes(self) -> None:
        self.assertEqual(checker.audit_text(self.text, self.r10), [])

    def test_random_ready_retirement_removal_is_rejected(self) -> None:
        mutant = self.replace_in_task(
            "random_legal_transaction",
            "weight_req_ready = 1'b0;\n"
            "            psum_req_ready = 1'b0;\n"
            "            random_request_window_active = 1'b0;",
            "weight_req_ready = 1'b1;\n"
            "            psum_req_ready = 1'b1;\n"
            "            random_request_window_active = 1'b0;")
        self.assert_rejected(mutant,
                             "random ready retirement removal accepted")

    def test_random_request_window_disable_is_rejected(self) -> None:
        mutant = self.replace_in_task(
            "random_legal_transaction",
            "random_request_window_active = 1'b1;",
            "random_request_window_active = 1'b0;")
        self.assert_rejected(mutant,
                             "disabled random exact-one counter window accepted")

    def test_random_request_window_immediate_override_is_rejected(self) -> None:
        mutant = self.replace_in_task(
            "random_legal_transaction",
            "random_request_window_active = 1'b1;",
            "random_request_window_active = 1'b1;\n"
            "            random_request_window_active = 1'b0;")
        self.assert_rejected(mutant,
                             "immediately disabled request window accepted")

    def test_random_request_window_comment_decoy_is_rejected(self) -> None:
        mutant = self.replace_in_task(
            "random_legal_transaction",
            "random_request_window_active = 1'b1;",
            "// random_request_window_active = 1'b1;\n"
            "            random_request_window_active = 1'b0;")
        self.assert_rejected(mutant,
                             "comment-only request-window enable accepted")

    def test_random_request_window_string_decoy_is_rejected(self) -> None:
        mutant = self.replace_in_task(
            "random_legal_transaction",
            "random_request_window_active = 1'b1;",
            '$display("random_request_window_active = 1\'b1;");\n'
            "            random_request_window_active = 1'b0;")
        self.assert_rejected(mutant,
                             "string-only request-window enable accepted")

    def test_random_hold_loop_zero_trip_is_rejected(self) -> None:
        mutant = self.replace_in_task(
            "random_legal_transaction",
            "repeat (hold_cycles) begin",
            "repeat (0) begin")
        self.assert_rejected(mutant,
                             "zero-trip random response hold body accepted")

    def test_random_hold_assignment_immediate_override_is_rejected(self) -> None:
        mutant = self.replace_in_task(
            "random_legal_transaction",
            "hold_cycles = 1 + prng_q[9:7];",
            "hold_cycles = 1 + prng_q[9:7];\n"
            "            hold_cycles = 0;")
        self.assert_rejected(mutant,
                             "immediately zeroed hold-cycle budget accepted")

    def test_random_hold_loop_comment_decoy_is_rejected(self) -> None:
        mutant = self.replace_in_task(
            "random_legal_transaction",
            "repeat (hold_cycles) begin",
            "// repeat (hold_cycles) begin\n"
            "            repeat (0) begin")
        self.assert_rejected(mutant,
                             "comment-only positive hold loop accepted")

    def test_random_hold_loop_string_decoy_is_rejected(self) -> None:
        mutant = self.replace_in_task(
            "random_legal_transaction",
            "repeat (hold_cycles) begin",
            '$display("repeat (hold_cycles) begin");\n'
            "            repeat (0) begin")
        self.assert_rejected(mutant,
                             "string-only positive hold loop accepted")

    def test_random_post_response_oracle_removal_is_rejected(self) -> None:
        mutant = self.replace_in_task(
            "random_legal_transaction",
            "|| response_accept_count != response0 + 1\n"
            "                    || weight_req_valid",
            "|| weight_req_valid")
        self.assert_rejected(mutant,
                             "post-retirement response-count removal accepted")

    def test_random_extra_response_posedge_is_rejected(self) -> None:
        anchor = "// Exact response retirement: no extra response posedge is allowed.\n"
        mutant = self.replace_in_task(
            "random_legal_transaction",
            anchor + "            @(negedge clk_core);",
            anchor + "            @(posedge clk_core); #1ps;\n"
            "            @(negedge clk_core);")
        self.assert_rejected(mutant,
                             "random extra response posedge accepted")

    def test_random_core_ready_posedge_race_is_rejected(self) -> None:
        mutant = self.replace_in_task(
            "random_legal_transaction",
            "@(negedge clk_core);\n"
            "            force dut.core_issue_data_ready = 1'b1;",
            "@(posedge clk_core); #1ps;\n"
            "            force dut.core_issue_data_ready = 1'b1;")
        self.assert_rejected(mutant, "random core-ready posedge race accepted")

    def test_random_tuple_retirement_removal_is_rejected(self) -> None:
        mutant = self.replace_in_task(
            "random_legal_transaction",
            "retire_random_forced_issue_tuple();",
            "// tuple retirement removed")
        self.assert_rejected(mutant, "random tuple retirement removal accepted")

    def test_random_response_stability_removal_is_rejected(self) -> None:
        mutant = self.replace_in_task(
            "random_legal_transaction",
            "weight_data !== '0",
            "weight_data === '0")
        self.assert_rejected(mutant,
                             "random response stability weakening accepted")

    def test_random_post_retirement_edge_removal_is_rejected(self) -> None:
        mutant = self.replace_in_task(
            "random_legal_transaction",
            "@(posedge clk_core); #1ps;\n"
            "            if (weight_fire_count != w0 + 1",
            "#1ps;\n"
            "            if (weight_fire_count != w0 + 1")
        self.assert_rejected(mutant,
                             "random post-retirement edge removal accepted")

    def test_random_sva_mask_injection_is_rejected(self) -> None:
        mutant = self.replace_in_task(
            "random_legal_transaction",
            "random_request_window_active = 1'b1;",
            "random_request_window_active = 1'b1;\n"
            "            request_hold_attack_mode = 1'b1;")
        self.assert_rejected(mutant, "random SVA mask injection accepted")

    def test_tuple_helper_early_core_ready_release_is_rejected(self) -> None:
        helper = checker.extract_task(
            self.text, "retire_random_forced_issue_tuple")
        self.assertIsNotNone(helper)
        assert helper is not None
        mutant_helper = helper.replace(
            "release dut.issue_request_parent_id;",
            "release dut.issue_request_parent_id;\n"
            "            release dut.core_issue_data_ready;", 1)
        mutant = self.text.replace(helper, mutant_helper, 1)
        self.assert_rejected(mutant,
                             "early core-ready release in tuple helper accepted")

    def test_random_state_dump_removal_is_rejected(self) -> None:
        mutant = self.replace_in_task(
            "random_legal_transaction",
            'dump_r9_liveness_state("r11_random_request_retire"',
            'dump_r9_liveness_state("random_retire"')
        self.assert_rejected(mutant, "random retirement state dump removal accepted")

    def test_r10_normal_mutation_is_rejected(self) -> None:
        mutant = self.replace_in_task(
            "serve_normal_beat",
            "normal_retired_beats = normal_retired_beats + 1;",
            "normal_retired_beats = normal_retired_beats + 2;")
        self.assert_rejected(mutant, "R10 normal task mutation accepted")

    def test_workload_count_mutation_is_rejected(self) -> None:
        mutant = self.text.replace("test_index < 24", "test_index < 23", 1)
        self.assert_rejected(mutant, "24-random workload mutation accepted")

    def test_normal_row_mutation_is_rejected(self) -> None:
        mutant = self.text.replace(
            "prep_mask = (row == 0) ? 16'h0003 : 16'h0000",
            "prep_mask = (row == 0) ? 16'h0001 : 16'h0000", 1)
        self.assert_rejected(mutant, "normal two-source row mutation accepted")

    def test_zero_sva_gate_removal_is_rejected(self) -> None:
        mutant = self.text.replace("zero_sva_failures_required=true",
                                   "zero_sva_failures_required=false", 1)
        self.assert_rejected(mutant, "zero-SVA gate removal accepted")

    def test_claim_inflation_is_rejected(self) -> None:
        mutant = self.text.replace("functional_vcs_only=false",
                                   "functional_vcs_only=true", 1)
        self.assert_rejected(mutant, "source claim inflation accepted")


if __name__ == "__main__":
    unittest.main(verbosity=2)
