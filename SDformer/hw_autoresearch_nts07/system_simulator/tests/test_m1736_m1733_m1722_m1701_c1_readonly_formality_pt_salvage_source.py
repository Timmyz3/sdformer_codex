#!/usr/bin/env python3
from __future__ import print_function

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


HW = Path(__file__).resolve().parents[2]
RUNNER = HW / "dc_handoff/scripts/run_m1736_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_one_shot.py"
CONTRACT = HW / "contracts/m1736_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_source_contract_r1_20260901.json"
SPEC = importlib.util.spec_from_file_location("m1736_runner", str(RUNNER))
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


class M1736Tests(unittest.TestCase):
    def test_01_m1733_attempt_failure_are_sealed_and_exhausted(self):
        M.verify_seal(M.M1733_ATTEMPT, M.FIXED_SHA["m1733_attempt_manifest"],
                      M.FIXED_SHA["m1733_attempt_outer"])
        M.verify_seal(M.M1733_FAILURE, M.FIXED_SHA["m1733_failure_manifest"],
                      M.FIXED_SHA["m1733_failure_outer"])
        M.verify_pt_evidence()

    def test_02_transitive_m1734_m1735_authority_is_exact(self):
        M.verify_predecessor_authority()
        self.assertEqual(M.sha(M.M1733_RUNNER), M.FIXED_SHA["m1733_runner"])

    def test_03_m1722_formality_is_reused_without_tool(self):
        proof = M.load_m1733().verify_m1722_formality_reuse()
        self.assertEqual(proof["passing_compare_points"], 16549)
        self.assertEqual(proof["macro_instances_per_side"], 9)

    def test_04_pt_machine_and_scope_are_exact(self):
        machine = M.parse_machine()
        self.assertEqual(machine["setup_wns_ns"], "0.027871")
        self.assertEqual(machine["hold_wns_ns"], "0.001827")
        self.assertEqual(machine["macro_count"], "9")

    def test_05_only_two_startup_errors_and_main_completed(self):
        lines = (M.PTSTA / "pt.raw.log").read_text(errors="replace").splitlines()
        errors = [line for line in lines if line.startswith("Error:")]
        self.assertEqual(errors, [
            "Error: Library Compiler executable path is not set. (PT-063)",
            'Error: can\'t read "::env(HOME)": no such variable'])
        main = "set design_name m935_m912_three_stage_exact_parent_match_product_capture_island"
        self.assertLess(lines.index(errors[1]), lines.index(main))
        self.assertLess(lines.index(main), lines.index("quit"))
        self.assertLess(lines.index("quit"), lines.index(
            "Diagnostics summary: 2 errors, 5 warnings, 30 informationals"))

    def test_06_coverage_is_disclosed_not_misrepresented(self):
        coverage = (M.PTSTA / "reports/analysis_coverage.rpt").read_text()
        for token in ("setup                 13860     13851 (100%)         0 (  0%)         9 (  0%)",
                      "hold                  13860     13851 (100%)         0 (  0%)         9 (  0%)",
                      "min_pulse_width       78506     50526 ( 64%)         0 (  0%)     27980 ( 36%)"):
            self.assertIn(token, coverage)
        self.assertFalse(M.RESULT_CLAIMS["paper_ppa_ready"])

    def test_07_no_tool_license_network_or_subprocess_path(self):
        text = RUNNER.read_text()
        for token in ("import subprocess", "import socket", "fm_shell", "dc_shell",
                      "lmutil", "socket.", "requests.", "urllib",
                      "SNPSLMD_LICENSE_FILE", "subprocess.run", "subprocess.Popen"):
            self.assertNotIn(token, text)
        self.assertIn('"eda_runs": 0', text)
        self.assertIn('"license_queries": 0', text)
        self.assertIn('"network_calls": 0', text)

    def test_08_authority_precedes_attempt_and_evidence_copy(self):
        main = RUNNER.read_text()[RUNNER.read_text().index("def main()") :]
        ordered = ("verify_authority()", "verify_predecessor_authority()",
                   "verify_m1722_formality_reuse()", "verify_pt_evidence()",
                   "namespaces_fresh()", "ATTEMPT.mkdir()", "shutil.copytree(PTSTA",
                   "publish_no_replace(STAGE, RESULT)")
        cursor = 0
        for token in ordered:
            position = main.find(token, cursor)
            self.assertGreaterEqual(position, 0, token)
            cursor = position + len(token)

    def test_09_fresh_authority_and_namespaces_are_absent(self):
        for path in (M.M1737, M.M1738, Path(str(M.M1738) + ".sha256"),
                     Path(str(M.M1738) + ".sha256.seal.sha256"),
                     M.ATTEMPT, M.RESULT, M.FAILURE):
            self.assertFalse(os.path.lexists(path), str(path))
        with self.assertRaisesRegex(M.Failure, "authority absent"):
            M.verify_authority()

    def test_10_contract_claims_and_source_inventory(self):
        contract = M.strict_json(CONTRACT)
        M.verify_contract_sources(contract)
        self.assertEqual(contract["claim_boundary"], M.SOURCE_CLAIMS)
        self.assertTrue(all(value is False for value in M.SOURCE_CLAIMS.values()))

    def test_11_machine_mutations_fail(self):
        original = (M.PTSTA / "reports/timing_summary_machine.txt").read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "machine.txt"
            for changed in (original.replace("0.027871", "-0.000001", 1),
                            original.replace("macro_count=9", "macro_count=8", 1),
                            original + "setup_wns_ns=9\n"):
                path.write_text(changed)
                values = {}
                failed = False
                try:
                    for row in path.read_text().splitlines():
                        key, value = row.split("=", 1)
                        if key in values:
                            raise M.Failure("duplicate")
                        values[key] = value
                    expected = M.parse_machine()
                    if values != expected:
                        raise M.Failure("drift")
                except M.Failure:
                    failed = True
                self.assertTrue(failed)

    def test_12_source_only_author_executed_nothing(self):
        execution = M.strict_json(CONTRACT)["author_execution"]
        self.assertTrue(execution["source_only"])
        for key in ("runner_runs", "eda_runs", "license_queries", "network_calls",
                    "attempts_created", "results_created"):
            self.assertEqual(execution[key], 0)
        self.assertFalse(execution["release_created"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
