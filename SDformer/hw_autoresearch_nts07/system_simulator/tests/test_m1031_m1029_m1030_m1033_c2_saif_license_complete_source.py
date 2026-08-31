#!/opt/anaconda3/envs/pytorch310/bin/python3.10
from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_m1031_m1029_m1030_m1033_c2_saif_license_complete_source.py"
SPEC = importlib.util.spec_from_file_location("m1031_checker", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


class TestM1031LicenseCompleteSource(unittest.TestCase):
    def setUp(self):
        self.text = MOD.RUNNER.read_text(encoding="utf-8")

    def assert_text_rejected(self, text):
        with self.assertRaises(RuntimeError):
            MOD.audit_runner_text(text)

    def test_canonical_source(self):
        self.assertEqual(MOD.main()["status"],
                         "PASS_M1031_M1029_M1030_M1033_LICENSE_COMPLETE_SOURCE")

    def test_missing_license_rejected_without_value(self):
        with self.assertRaisesRegex(RuntimeError, "nonempty license route"):
            MOD.audit_license_environment({})
        secret = "SECRET_SHOULD_NEVER_LEAVE_CHECKER"
        result = MOD.audit_license_environment({"SNPSLMD_LICENSE_FILE": secret})
        self.assertNotIn(secret, repr(result))
        self.assertEqual(result, {"license_route_present": True,
                                  "license_value_recorded": False})

    def test_missing_license_guard_source_rejected(self):
        self.assert_text_rejected(self.text.replace(
            'if [[ -n "${LM_LICENSE_FILE:-}" || -n "${SNPSLMD_LICENSE_FILE:-}" ]]; then',
            'if true; then'))

    def test_wrong_tiny_sha_rejected(self):
        self.assert_text_rejected(self.text.replace(MOD.EXPECTED["tiny"], "0" * 64))

    def test_missing_preflight_call_rejected(self):
        self.assert_text_rejected(self.text.replace(
            "phase=LICENSE_CHECKOUT_PREFLIGHT\nrun_license_preflight",
            "phase=LICENSE_CHECKOUT_PREFLIGHT\n# injected missing preflight"))

    def test_occupied_namespace_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "collision"):
            MOD.audit_namespace({"attempt": True, "result": False})

    def test_active_collision_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "VCS/DC/FM/PT collision"):
            MOD.audit_collision(["python", "vcs1"])

    def test_axis_case_or_dut_scope_loss_rejected(self):
        self.assert_text_rejected(self.text.replace("for axis in k1 k8 k1x8; do",
                                                   "for axis in k1 k8; do"))
        self.assert_text_rejected(self.text.replace("for case_id in 0 1 2 3 4; do",
                                                   "for case_id in 0 1 2 3; do"))


if __name__ == "__main__":
    unittest.main()
