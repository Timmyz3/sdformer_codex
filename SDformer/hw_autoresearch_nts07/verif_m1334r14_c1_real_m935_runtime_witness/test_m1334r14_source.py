#!/opt/anaconda3/envs/pytorch310/bin/python3.10
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE / "check_m1334r14_source.py"
spec = importlib.util.spec_from_file_location("m1334r14_check", CHECKER)
M = importlib.util.module_from_spec(spec); sys.modules[spec.name] = M
spec.loader.exec_module(M)


class Tests(unittest.TestCase):
    def test_01_good_monotonic_trace(self):
        result = M.runtime_model(M.good_trace())
        self.assertTrue(result["pass"])
        self.assertEqual(result["stage"], 7)
        self.assertEqual(result["counts"], {"weight": 2, "psum": 1,
                         "response": 2, "core": 2, "commit": 1,
                         "row": 1, "task": 1})

    def test_02_missing_each_milestone_never_passes(self):
        trace = M.good_trace()
        for index in range(len(trace)):
            self.assertFalse(M.runtime_model(trace[:index] + trace[index + 1:])["pass"])

    def test_03_reordered_child_outputs_fail(self):
        trace = M.good_trace()
        for mutant in (
                [trace[5]] + trace[:5] + trace[6:],
                trace[:5] + [trace[6], trace[5]] + trace[7:],
                trace[:6] + [trace[7], trace[6]]):
            self.assertTrue(M.runtime_model(mutant)["fault"])

    def test_03b_second_request_before_first_core_accept_fails(self):
        trace = M.good_trace()
        mutant = trace[:2] + [trace[3], trace[2]] + trace[4:]
        self.assertTrue(M.runtime_model(mutant)["fault"])

    def test_04_first_nonfirst_tuple_attacks_fail(self):
        for index, key, value in ((0, "source", 1), (0, "first", False),
                                  (3, "source", 0), (3, "last", False)):
            trace = [dict(row) for row in M.good_trace()]
            trace[index][key] = value
            self.assertTrue(M.runtime_model(trace)["fault"])

    def test_05_extra_request_or_completion_fails(self):
        for extra in ({"kind": "weight", "source": 2, "first": False, "last": False},
                      {"kind": "psum", "source": 0, "first": True},
                      {"kind": "commit", "address": 0},
                      {"kind": "row", "row_id": 0},
                      {"kind": "task", "epoch": 0x9001}):
            self.assertTrue(M.runtime_model(M.good_trace() + [extra])["fault"])

    def test_06_wrong_child_identity_fails(self):
        for index, key, value in ((5, "address", 1), (6, "row_id", 1),
                                  (7, "epoch", 0x9002)):
            trace = [dict(row) for row in M.good_trace()]
            trace[index][key] = value
            self.assertTrue(M.runtime_model(trace)["fault"])

    def test_07_attack_mask_or_design_fault_fails(self):
        for key in ("attack", "fault"):
            trace = [dict(row) for row in M.good_trace()]
            trace[2][key] = True
            self.assertTrue(M.runtime_model(trace)["fault"])

    def test_08_witness_structure_exact(self):
        M.check_witness_text(M.WITNESS.read_text())

    def test_09_guard_or_force_mutation_rejected(self):
        source = M.WITNESS.read_text()
        mutants = (
            source.replace("final begin : witness_final_oracle",
                           "initial begin\nforce issue_request_valid = 1'b1;\nend\nfinal begin : witness_final_oracle", 1),
            source.replace("bind tb_m1270r13_m1162_real_m935_protocol_unit_delay_r13", "", 1),
            source.replace("dut.response_accept_w", "1'b1", 1),
            source.replace("row_complete_valid && row_complete_ready", "1'b1", 1),
            source.replace("PASS_M1334R14_REAL_M935_RUNTIME_WITNESS", "PASS_REMOVED", 1),
        )
        for mutant in mutants:
            with self.assertRaises(AssertionError):
                M.check_witness_text(mutant)

    def test_10_filelist_is_frozen_design_plus_witness(self):
        expected = [str(path) for path in
                    (M.FOUNDRY, M.M528, M.M935, M.M1162, M.SVA, M.R13_TB, M.WITNESS)]
        self.assertEqual(M.FILELIST.read_text().splitlines(), expected)

    def test_11_readiness_and_r13_no_go_seals(self):
        self.assertIn("review.json", M.verify_dir(M.READINESS, M.READINESS_SEAL))
        self.assertIn("review.json", M.verify_dir(M.R13_REVIEW, M.R13_REVIEW_SEAL))

    def test_12_no_release_or_execution_authority(self):
        self.assertFalse(list(M.HW.glob("contracts/m1334*c1*r14*release*.json")))
        text = M.WITNESS.read_text() + M.FILELIST.read_text() + CHECKER.read_text()
        self.assertNotIn("/opt/synopsys/vcs", text)
        self.assertNotIn("dc_shell", text)
        self.assertNotIn("pt_shell", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
