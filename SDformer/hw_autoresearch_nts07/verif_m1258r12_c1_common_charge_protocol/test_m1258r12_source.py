#!/usr/bin/env python3
"""Mutation tests for the source-only M1258/R12 checker."""

import subprocess
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TB = HERE / "tb_m1258r12_m1162_common_charge_protocol_unit_delay_r12.sv"
CHECKER = HERE / "check_m1258r12_source.py"


def check(text: str) -> subprocess.CompletedProcess:
    with tempfile.NamedTemporaryFile("w", suffix=".sv", delete=False) as handle:
        handle.write(text)
        path = Path(handle.name)
    try:
        return subprocess.run(
            ["python3", str(CHECKER), "--candidate", str(path)],
            cwd=str(ROOT), universal_newlines=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    finally:
        path.unlink()


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = TB.read_text()

    def assert_rejected(self, mutant: str) -> None:
        result = check(mutant)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_01_canonical(self) -> None:
        result = check(self.source)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_02_parent_request_force_rejected(self) -> None:
        self.assert_rejected(self.source.replace(
            "force dut.u_frozen_m935.issue_request_valid = 1'b1;",
            "force dut.issue_request_valid = 1'b1;", 1))

    def test_03_parent_ready_force_rejected(self) -> None:
        self.assert_rejected(self.source.replace(
            "force dut.u_frozen_m935.issue_data_ready = 1'b1;",
            "force dut.core_issue_data_ready = 1'b1;", 1))

    def test_04_boundary_label_removal_rejected(self) -> None:
        self.assert_rejected(self.source.replace(
            "PHASE_M1258R12_BOUNDARY_ONLY_RANDOM_ENTER",
            "PHASE_M1258R12_RANDOM_ENTER", 1))

    def test_05_integrated_random_inflation_rejected(self) -> None:
        self.assert_rejected(self.source.replace(
            "integrated_random=false", "integrated_random=true", 1))

    def test_06_integrated_claim_inflation_rejected(self) -> None:
        self.assert_rejected(self.source.replace(
            "integrated_m935_claim=false", "integrated_m935_claim=true", 1))

    def test_07_normal_real_task_drift_rejected(self) -> None:
        self.assert_rejected(self.source.replace(
            "normal_m935_completion;\n        integer",
            "normal_m935_completion;\n        // drift\n        integer", 1))

    def test_08_random_reset_removal_rejected(self) -> None:
        anchor = ("task automatic random_boundary_transaction(input integer index);"
                  "\n        integer")
        start = self.source.index(anchor)
        tail = self.source[start:]
        tail = tail.replace("            reset_dut();\n", "", 1)
        self.assert_rejected(self.source[:start] + tail)

    def test_09_child_seam_force_removal_rejected(self) -> None:
        self.assert_rejected(self.source.replace(
            "force dut.u_frozen_m935.issue_request_parent_id = 6'b0;",
            "", 1))

    def test_10_integrated_normal_token_removal_rejected(self) -> None:
        self.assert_rejected(self.source.replace(
            "PHASE_M1258R12_INTEGRATED_NORMAL_M935_COMPLETE",
            "PHASE_M1258R12_NORMAL_M935_COMPLETE", 1))

    def test_11_random_legal_rename_rejected(self) -> None:
        self.assert_rejected(self.source.replace(
            "random_boundary_transaction", "random_legal_transaction"))

    def test_12_parent_force_comment_decoy_does_not_fail(self) -> None:
        result = check(self.source.replace(
            "// M1258/R12 additive", 
            "// force dut.issue_request_valid = 1'b1;\n// M1258/R12 additive",
            1))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
