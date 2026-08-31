#!/opt/anaconda3/envs/pytorch310/bin/python3.10
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_m1021_m1018_m1019_m1022_c2_saif_environment_repair_source.py"
SPEC = importlib.util.spec_from_file_location("m1021_checker", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


class TestM1021Source(unittest.TestCase):
    def setUp(self):
        self.text = MOD.RUNNER.read_text()

    def test_canonical_source(self):
        self.assertEqual(MOD.main()["status"],
                         "PASS_M1021_M1018_M1019_M1022_ENVIRONMENT_REPAIR_SOURCE")

    def assert_rejected(self, text):
        with self.assertRaises(RuntimeError):
            MOD.audit_runner_text(text)

    def test_missing_vcs_home_export_rejected(self):
        self.assert_rejected(self.text.replace('export VCS_HOME="${expected_vcs_home}"', ""))

    def test_wrong_support_script_identity_rejected(self):
        self.assert_rejected(self.text.replace(
            "b34e06a92b05856532f868d32c0c81f1708506096856ad9a97bd27e2bd60215b",
            "0" * 64))

    def test_missing_independent_hammer_pin_rejected(self):
        self.assert_rejected(self.text.replace("M1022_EXPECTED_M1020_OUTER_SHA256", "REMOVED"))

    def test_stale_result_namespace_rejected(self):
        self.assert_rejected(self.text.replace(
            'result="${hw_root}/results/m1022_m1001_c2_three_axis_mapped_gate_saif_r3_20260829"',
            'result="${hw_root}/results/m1013_bad"'))

    def test_axis_or_case_loss_rejected(self):
        self.assert_rejected(self.text.replace("for axis in k1 k8 k1x8; do",
                                               "for axis in k1 k8; do"))
        self.assert_rejected(self.text.replace("for case_id in 0 1 2 3 4; do",
                                               "for case_id in 0 1 2 3; do"))


if __name__ == "__main__":
    unittest.main()
