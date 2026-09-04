#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Source-only mutations for M1250; no VCS, simv, EDA, GPU, or remote."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE / "static_check_m1250_c1_r11_vcs_release_source.py"
SPEC = importlib.util.spec_from_file_location("m1250_release_checker_test", CHECKER)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class Tests(unittest.TestCase):
    def test_01_canonical_runner_and_filelist(self) -> None:
        self.assertEqual(M.audit_runner(M.RUNNER.read_text()), [])
        self.assertEqual(M.audit_filelist(M.FILELIST.read_text()), [])

    def test_02_old_tb_filelist_mutation_rejected(self) -> None:
        changed = M.FILELIST.read_text().replace("m1232r11", "m1226r10")
        self.assertTrue(M.audit_filelist(changed))

    def test_03_second_compile_rejected(self) -> None:
        changed = M.RUNNER.read_text() + '\n"${VCS_BIN}" -full64\n'
        self.assertIn("compile count", M.audit_runner(changed))

    def test_04_second_sim_rejected(self) -> None:
        changed = M.RUNNER.read_text() + "\n./simv -no_save\n"
        self.assertIn("sim count", M.audit_runner(changed))

    def test_05_shared_timeout_rejected(self) -> None:
        changed = M.RUNNER.read_text().replace(
            "/usr/bin/timeout --signal=TERM --kill-after=30s", "timeout_removed", 1)
        self.assertIn("separate timeout count", M.audit_runner(changed))

    def test_06_error_assertion_fatal_gate_removed_rejected(self) -> None:
        changed = M.RUNNER.read_text().replace(
            "if rg -qi '(^|[^[:alnum:]_])(Error|Fatal|Assertion|\\$error|\\$fatal)([^[:alnum:]_]|$)' compile.log sim.log; then exit 35; fi",
            "if rg -qi '(Warning)' compile.log sim.log; then exit 35; fi", 1)
        self.assertTrue(any("error/assertion" in row for row in M.audit_runner(changed)))

    def test_07_phase_pair_removed_rejected(self) -> None:
        changed = M.RUNNER.read_text().replace(
            " RANDOM NORMAL_M935 CLEAN_RESET_PREP", " RANDOM CLEAN_RESET_PREP", 1)
        self.assertIn("phase population", M.audit_runner(changed))

    def test_08_random_24_gate_removed_rejected(self) -> None:
        changed = M.RUNNER.read_text().replace("for index in $(seq 0 23)",
                                               "for index in $(seq 0 22)", 1)
        self.assertTrue(any("24 random" in row for row in M.audit_runner(changed)))

    def test_09_normal_cover_removed_rejected(self) -> None:
        changed = M.RUNNER.read_text().replace("normal_m935_rows=1", "normal_m935_rows=0", 1)
        self.assertTrue(any("normal_m935_rows" in row for row in M.audit_runner(changed)))

    def test_10_failure_quarantine_removed_rejected(self) -> None:
        changed = M.RUNNER.read_text().replace("failed_or_incomplete.$$.quarantine",
                                               "no_quarantine", 1)
        self.assertTrue(any("failed_or_incomplete" in row for row in M.audit_runner(changed)))

    def test_11_environment_fail_closed(self) -> None:
        names = ("M1250_EXPECTED_RELEASE_SHA256",
                 "M1250_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256",
                 "M1250_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256",
                 "M1250_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256")
        good = {name: "a" * 64 for name in names}; self.assertTrue(M.env_gate(good))
        for name in names:
            changed = dict(good); changed.pop(name); self.assertFalse(M.env_gate(changed))

    def test_12_retry_or_cleanup_rejected(self) -> None:
        for addition in ("\nautomatic_retry=true\n", "\nrm -rf payload\n"):
            self.assertIn("destructive/retry behavior",
                          M.audit_runner(M.RUNNER.read_text() + addition))


if __name__ == "__main__":
    unittest.main(verbosity=2)
