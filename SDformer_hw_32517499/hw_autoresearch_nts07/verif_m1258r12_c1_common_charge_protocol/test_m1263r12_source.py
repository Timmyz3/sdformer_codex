#!/usr/bin/env python3
"""Combined 18-test regression plus final M1263 near-neighbour attacks."""

import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TB = HERE / "tb_m1258r12_m1162_common_charge_protocol_unit_delay_r12.sv"
CHECKER = HERE / "check_m1263r12_source.py"
PRIOR_TESTS = HERE / "test_m1261r12_source.py"

spec = importlib.util.spec_from_file_location("m1261_tests", str(PRIOR_TESTS))
prior_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior_tests)


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


class FinalHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = TB.read_text()
        cls.pass_line = next(line for line in cls.source.splitlines()
                             if '$display("PASS_M1258R12' in line)

    def reject(self, mutant):
        self.assertNotEqual(mutant, self.source)
        result, payload = check(mutant)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(payload["errors"])

    def test_19_canonical(self):
        result, payload = check(self.source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(payload["errors"])

    def test_20_phase_ordinary_string_cannot_replace_display(self):
        token = "PHASE_M1258R12_BOUNDARY_ONLY_DIRECTED_ENTER"
        self.reject(self.source.replace(
            '$display("' + token + '");',
            '$display("DECOY_PHASE");\n'
            '        string phase_shadow = "' + token + '";', 1))

    def test_21_fatal_cannot_replace_pass_display(self):
        self.reject(self.source.replace(
            '$display("PASS_M1258R12_', '$fatal(1, "PASS_M1258R12_', 1))

    def test_22_phase_shadow_suffix(self):
        token = "PHASE_M1258R12_BOUNDARY_ONLY_DIRECTED_ENTER"
        self.reject(self.source.replace(token, token + "_SHADOW", 1))

    def test_23_pass_shadow_suffix(self):
        self.reject(self.source.replace(
            "PASS_M1258R12_M1162_BOUNDARY_ONLY_SOURCE_CANDIDATE ",
            "PASS_M1258R12_M1162_BOUNDARY_ONLY_SOURCE_CANDIDATE_SHADOW ", 1))

    def test_24_duplicate_allowed_force_in_helper(self):
        line = "            force dut.u_frozen_m935.issue_request_valid = 1'b1;"
        self.reject(self.source.replace(line, line + "\n" + line, 1))

    def test_25_extra_allowed_force_outside_helper(self):
        line = "            force dut.u_frozen_m935.issue_request_valid = 1'b1;"
        self.reject(self.source.replace("        $finish;", line + "\n        $finish;", 1))

    def test_26_claim_false_with_ordinary_string_decoy(self):
        mutant = self.source.replace(
            "integrated_normal_m935_evidence=true",
            "integrated_normal_m935_evidence=false", 1).replace(
            "// M1258/R12 additive",
            'string claim_shadow = "integrated_normal_m935_evidence=true";\n'
            "// M1258/R12 additive", 1)
        self.reject(mutant)

    def test_27_normal_call_comment_and_string_decoy(self):
        self.reject(self.source.replace(
            "        normal_m935_completion();",
            '        // normal_m935_completion();\n'
            '        string normal_shadow = "normal_m935_completion();";', 1))

    def test_28_extra_release_in_unauthorized_task(self):
        self.reject(self.source.replace(
            "    task automatic clear_public_drivers;",
            "    task automatic clear_public_drivers;\n"
            "        release dut.u_frozen_m935.issue_request_valid;", 1))

    def test_29_force_rhs_drift(self):
        self.reject(self.source.replace(
            "force dut.u_frozen_m935.issue_request_valid = 1'b1;",
            "force dut.u_frozen_m935.issue_request_valid = 1'b0;", 1))

    def test_30_comment_and_string_force_remain_inert(self):
        mutant = self.source.replace(
            "// M1258/R12 additive",
            '// force dut.issue_request_valid = 1\'b1;\n'
            'string force_shadow = "force dut.issue_request_valid = 1\'b1;";\n'
            "// M1258/R12 additive", 1)
        result, payload = check(mutant)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse(payload["errors"])


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(prior_tests.Tests))
    suite.addTests(loader.loadTestsFromTestCase(FinalHardeningTests))
    return suite


if __name__ == "__main__":
    unittest.main(verbosity=2)
