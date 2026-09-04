#!/usr/bin/env python3
"""Mutation suite for the additive M1261 hardened R12 source checker."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TB = HERE / "tb_m1258r12_m1162_common_charge_protocol_unit_delay_r12.sv"
CHECKER = HERE / "check_m1261r12_source.py"


def check(text):
    with tempfile.NamedTemporaryFile("w", suffix=".sv", delete=False) as handle:
        handle.write(text)
        path = Path(handle.name)
    try:
        result = subprocess.run(
            ["python3", str(CHECKER), "--candidate", str(path)],
            cwd=str(ROOT), universal_newlines=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return result, json.loads(result.stdout)
    finally:
        path.unlink()


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = TB.read_text()

    def reject(self, mutant):
        self.assertNotEqual(mutant, self.source)
        result, payload = check(mutant)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(payload["errors"])

    def test_01_canonical(self):
        result, payload = check(self.source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(payload["errors"])

    def test_02_child_prefix_shadow(self):
        self.reject(self.source.replace(
            "force dut.u_frozen_m935.issue_request_valid = 1'b1;",
            "force dut.u_frozen_m935.issue_request_valid_shadow = 1'b1;", 1))

    def test_03_boundary_claim_comment_decoy(self):
        self.reject(self.source.replace(
            "boundary_only=true integrated_random=false",
            "boundary_only=false integrated_random=false", 1).replace(
            "// M1258/R12 additive",
            "// boundary_only=true\n// M1258/R12 additive", 1))

    def test_04_integrated_normal_claim_comment_decoy(self):
        self.reject(self.source.replace(
            "integrated_normal_m935_evidence=true",
            "integrated_normal_m935_evidence=false", 1).replace(
            "// M1258/R12 additive",
            "// integrated_normal_m935_evidence=true\n// M1258/R12 additive", 1))

    def test_05_normal_call_commented(self):
        self.reject(self.source.replace(
            "        normal_m935_completion();",
            "        // normal_m935_completion();", 1))

    def test_06_parent_request_force(self):
        self.reject(self.source.replace(
            "force dut.u_frozen_m935.issue_request_valid = 1'b1;",
            "force dut.issue_request_valid = 1'b1;", 1))

    def test_07_parent_force_comment_is_inert(self):
        result, payload = check(self.source.replace(
            "// M1258/R12 additive",
            "// force dut.issue_request_valid = 1'b1;\n// M1258/R12 additive", 1))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(payload["errors"])

    def test_08_child_comment_parent_actual(self):
        self.reject(self.source.replace(
            "force dut.u_frozen_m935.issue_request_parent_id = 6'b0;",
            "// force dut.u_frozen_m935.issue_request_parent_id = 6'b0;\n"
            "            force dut.issue_request_parent_id = 6'b0;", 1))

    def test_09_integrated_random_inflation(self):
        self.reject(self.source.replace(
            "integrated_random=false", "integrated_random=true", 1))

    def test_10_integrated_m935_inflation(self):
        self.reject(self.source.replace(
            "integrated_m935_claim=false", "integrated_m935_claim=true", 1))

    def test_11_normal_load_drift(self):
        self.reject(self.source.replace(
            "prep_mask = (row == 0) ? 16'h0003 : 16'h0000;",
            "prep_mask = (row == 0) ? 16'h0007 : 16'h0000;", 1))

    def test_12_normal_serve_drift(self):
        self.reject(self.source.replace(
            "        input integer beat_index\n    );",
            "        input integer beat_index\n    );\n        // drift", 1))

    def test_13_duplicate_pass_display(self):
        anchor = "        $finish;"
        pass_line = next(line for line in self.source.splitlines()
                         if "$display(\"PASS_M1258R12" in line)
        self.reject(self.source.replace(anchor, pass_line + "\n" + anchor, 1))

    def test_14_phase_comment_decoy(self):
        self.reject(self.source.replace(
            "PHASE_M1258R12_BOUNDARY_ONLY_DIRECTED_ENTER",
            "PHASE_M1258R12_BOUNDARY_DIRECTED_ENTER", 1).replace(
            "// M1258/R12 additive",
            "// PHASE_M1258R12_BOUNDARY_ONLY_DIRECTED_ENTER\n"
            "// M1258/R12 additive", 1))

    def test_15_extra_child_prefix_internal(self):
        self.reject(self.source.replace(
            "// M1258/R12 additive",
            "force dut.u_frozen_m935.issue_request_valid.internal = 1'b1;\n"
            "// M1258/R12 additive", 1))

    def test_16_duplicate_normal_call(self):
        self.reject(self.source.replace(
            "        normal_m935_completion();",
            "        normal_m935_completion();\n"
            "        normal_m935_completion();", 1))

    def test_17_headline_claim_inflation(self):
        self.reject(self.source.replace(
            "system_speedup=false headline=false",
            "system_speedup=false headline=true", 1))

    def test_18_exact_child_ready_shadow(self):
        self.reject(self.source.replace(
            "force dut.u_frozen_m935.issue_data_ready = 1'b1;",
            "force dut.u_frozen_m935.issue_data_ready_shadow = 1'b1;", 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
