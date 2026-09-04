#!/opt/anaconda3/envs/pytorch310/bin/python3.10
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE / "check_m1337r15_source.py"
spec = importlib.util.spec_from_file_location("m1337r15_check", CHECKER)
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)


class Tests(unittest.TestCase):
    def test_01_good_registered_stage_trace(self):
        result = M.runtime_model(M.good_trace())
        self.assertTrue(result["pass"])
        self.assertEqual(result["stage"], 7)

    def test_02_missing_each_cycle_fails(self):
        trace = M.good_trace()
        for index in range(len(trace)):
            self.assertFalse(M.runtime_model(trace[:index] + trace[index + 1:])["pass"])

    def test_03_second_request_same_edge_as_first_accept_fails(self):
        trace = M.good_trace()
        merged = dict(trace[1])
        merged.update({"weight": True, "request_valid": True,
                       "source_valid": True, "source": 1,
                       "first": False, "last": True})
        mutant = [trace[0], merged] + trace[3:]
        self.assertTrue(M.runtime_model(mutant)["fault"])

    def test_04_second_accept_commit_row_task_collapse_fails(self):
        trace = M.good_trace()
        merged = dict(trace[3])
        merged.update({"commit": True, "address": 0, "row": True,
                       "row_id": 0, "task": True, "epoch": 0x9001})
        mutant = trace[:3] + [merged]
        self.assertTrue(M.runtime_model(mutant)["fault"])

    def test_05_each_late_milestone_requires_prior_registered_stage(self):
        trace = M.good_trace()
        for pair in ((3, 4), (4, 5), (5, 6)):
            mutant = [dict(row) for row in trace]
            merged = dict(mutant[pair[0]])
            merged.update(mutant[pair[1]])
            mutant = mutant[:pair[0]] + [merged] + mutant[pair[1] + 1:]
            self.assertTrue(M.runtime_model(mutant)["fault"])

    def test_06_unknown_source_address_row_epoch_fail(self):
        for index, key in ((0, "source"), (2, "source"), (4, "address"),
                           (5, "row_id"), (6, "epoch")):
            trace = [dict(row) for row in M.good_trace()]
            trace[index][key] = None
            self.assertTrue(M.runtime_model(trace)["fault"], (index, key))

    def test_07_unknown_event_control_fails(self):
        for index, key in ((0, "weight"), (1, "response"), (3, "core"),
                           (4, "commit"), (5, "row"), (6, "task")):
            trace = [dict(row) for row in M.good_trace()]
            trace[index][key] = None
            self.assertTrue(M.runtime_model(trace)["fault"], (index, key))

    def test_08_wrong_tuple_and_child_identity_fail(self):
        for index, key, value in ((0, "source", 1), (0, "first", False),
                                  (2, "source", 0), (2, "last", False),
                                  (4, "address", 1), (5, "row_id", 1),
                                  (6, "epoch", 0x9002)):
            trace = [dict(row) for row in M.good_trace()]
            trace[index][key] = value
            self.assertTrue(M.runtime_model(trace)["fault"], (index, key))

    def test_09_attack_mask_and_design_fault_fail(self):
        for key in ("attack", "fault"):
            trace = [dict(row) for row in M.good_trace()]
            trace[1][key] = True
            self.assertTrue(M.runtime_model(trace)["fault"])

    def test_10_extra_or_reordered_event_fails(self):
        trace = M.good_trace()
        for mutant in ([trace[4]] + trace, trace + [trace[6]],
                       trace[:4] + [trace[5], trace[4]] + trace[6:]):
            self.assertTrue(M.runtime_model(mutant)["fault"])

    def test_11_witness_structure_passes(self):
        M.check_witness_text(M.WITNESS.read_text())

    def test_12_each_active_child_output_tie_is_rejected(self):
        source = M.WITNESS.read_text()
        for port in ("response_accept", "core_accept", "psum_commit_fire",
                     "psum_commit_address", "row_complete_fire",
                     "row_complete_id", "task_done_fire", "task_done_epoch"):
            expression = M.EXPECTED_BIND[port]
            mutant = source.replace(".%s(%s)" % (port, expression.replace("&&", " && ")),
                                    ".%s(1'b0)" % port, 1)
            if mutant == source:
                mutant = source.replace(".%s(%s)" % (port, expression),
                                        ".%s(1'b0)" % port, 1)
            self.assertNotEqual(mutant, source, port)
            with self.assertRaises(AssertionError, msg=port):
                M.check_witness_text(mutant)

    def test_13_each_attack_and_fault_tie_is_rejected(self):
        source = M.WITNESS.read_text()
        for port in ("request_hold_attack_mode", "weight_service_attack_mode",
                     "psum_service_attack_mode", "protocol_error", "boundary_fault",
                     "core_fault", "m935_fault", "weight_service_fault",
                     "psum_service_fault"):
            expression = M.EXPECTED_BIND[port]
            mutant = source.replace(".%s(%s)" % (port, expression),
                                    ".%s(1'b0)" % port, 1)
            self.assertNotEqual(mutant, source, port)
            with self.assertRaises(AssertionError, msg=port):
                M.check_witness_text(mutant)

    def test_14_comments_cannot_rescue_tied_bind(self):
        source = M.WITNESS.read_text()
        original = ".response_accept(dut.response_accept_w)"
        mutant = source.replace(original,
                                ".response_accept(1'b1) // " + original, 1)
        with self.assertRaises(AssertionError):
            M.check_witness_text(mutant)

    def test_15_pass_before_or_outside_success_branch_is_rejected(self):
        source = M.WITNESS.read_text()
        pass_line = next(line for line in source.splitlines()
                         if "$display(\"PASS_M1337R15" in line)
        mutant = source.replace(pass_line + "\n", "", 1)
        mutant = mutant.replace("        if (pass === 1'b1) begin\n",
                                pass_line + "\n        if (pass === 1'b1) begin\n", 1)
        with self.assertRaises(AssertionError):
            M.check_witness_text(mutant)

    def test_16_same_edge_frontier_source_mutation_is_rejected(self):
        source = M.WITNESS.read_text()
        mutant = source.replace("case (stage_q)",
                                "integer core_after;\ncore_after = core_accepts_q + core_accept;\ncase (stage_q)", 1)
        with self.assertRaises(AssertionError):
            M.check_witness_text(mutant)

    def test_17_exact_seven_member_filelist(self):
        expected = [str(path) for path in
                    (M.FOUNDRY, M.M528, M.M935, M.M1162,
                     M.SVA, M.R13_TB, M.WITNESS)]
        self.assertEqual(M.FILELIST.read_text().splitlines(), expected)

    def test_18_r14_failed_seal_and_six_fn_bound(self):
        self.assertIn("review.json", M.verify_dir(M.R14_FAILED, M.R14_FAILED_SEAL))
        review = json.loads((M.R14_FAILED / "review.json").read_text())
        self.assertEqual(review["false_negative_count"], 6)

    def test_19_contract_ledger_exact_and_mutation_one_rejected(self):
        contract = json.loads(M.CONTRACT.read_text())
        M.check_contract_dict(contract)
        for key in ("represented_ledger_bytes", "physically_integrated_parent_bytes",
                    "external_common_charge_bytes"):
            mutant = copy.deepcopy(contract)
            mutant["frozen_design"][key] = 1
            with self.assertRaises(AssertionError, msg=key):
                M.check_contract_dict(mutant)

    def test_20_no_release_or_execution_authority(self):
        self.assertFalse(list(M.HW.glob("contracts/m1337*c1*r15*release*.json")))
        text = M.WITNESS.read_text() + M.FILELIST.read_text() + CHECKER.read_text()
        self.assertNotIn("dc_shell", text)
        self.assertNotIn("pt_shell", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
