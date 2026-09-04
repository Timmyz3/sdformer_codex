#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Local source mutations for M1221; no VCS or EDA."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE / "static_check_m1221_c1_r9_vcs_release_source.py"
SPEC = importlib.util.spec_from_file_location("m1221_release_checker_test", CHECKER)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = M; SPEC.loader.exec_module(M)


class Tests(unittest.TestCase):
    def test_01_canonical_runner_and_filelist(self) -> None:
        self.assertEqual(M.audit_runner(M.RUNNER.read_text()), [])
        self.assertEqual(M.audit_filelist(M.FILELIST.read_text()), [])

    def test_02_r8_tb_filelist_mutation_rejected(self) -> None:
        changed = M.FILELIST.read_text().replace("m1219r9", "m1210r8")
        self.assertTrue(M.audit_filelist(changed))

    def test_03_second_compile_rejected(self) -> None:
        text = M.RUNNER.read_text(); changed = text + '\n"${VCS_BIN}" -full64\n'
        self.assertIn("compile count", M.audit_runner(changed))

    def test_04_second_sim_rejected(self) -> None:
        text = M.RUNNER.read_text(); changed = text + "\n./simv -no_save\n"
        self.assertIn("sim count", M.audit_runner(changed))

    def test_05_timeout_dump_removed_rejected(self) -> None:
        changed = M.RUNNER.read_text().replace("phase_watchdog_timeout_dump.txt", "phase_dump_removed.txt")
        self.assertTrue(any("phase_watchdog" in row for row in M.audit_runner(changed)))

    def test_06_phase_pair_removed_rejected(self) -> None:
        changed = M.RUNNER.read_text().replace(" RANDOM NORMAL_M935 CLEAN_RESET_PREP", " RANDOM CLEAN_RESET_PREP")
        self.assertIn("phase population", M.audit_runner(changed))

    def test_07_quarantine_removed_rejected(self) -> None:
        changed = M.RUNNER.read_text().replace("failed_or_incomplete.$$.quarantine", "no_quarantine")
        self.assertTrue(any("failed_or_incomplete" in row for row in M.audit_runner(changed)))

    def test_08_environment_fail_closed(self) -> None:
        names = ("M1221_EXPECTED_RELEASE_SHA256", "M1221_EXPECTED_RELEASE_HAMMER_REVIEW_SHA256",
                 "M1221_EXPECTED_RELEASE_HAMMER_MANIFEST_SHA256",
                 "M1221_EXPECTED_RELEASE_HAMMER_OUTER_SEAL_FILE_SHA256")
        good = {name: "a" * 64 for name in names}; self.assertTrue(M.env_gate(good))
        for name in names:
            changed = dict(good); changed.pop(name); self.assertFalse(M.env_gate(changed))

    def test_09_no_eda_execution_in_tests(self) -> None:
        self.assertFalse(M.RUNNER.name.endswith(".executed"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
