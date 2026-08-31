from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_m1044_m1043_m1046_c2_saif_ucli_power_source.py"
SPEC = importlib.util.spec_from_file_location("m1044_checker", SCRIPT)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MOD)


class TestM1044UcliPowerCompleteSource(unittest.TestCase):
    def setUp(self):
        self.text = MOD.RUNNER.read_text(encoding="utf-8")

    def assert_text_rejected(self, text: str) -> None:
        with self.assertRaises(RuntimeError):
            MOD.audit_runner_text(text)

    def test_canonical_source(self):
        self.assertEqual(MOD.main()["status"],
                         "PASS_M1044_M1043_M1046_UCLI_POWER_COMPLETE_SOURCE")

    def test_missing_debug_access_rejected(self):
        self.assert_text_rejected(self.text.replace("-debug_access+r", "-debug_access+none"))

    def test_missing_lca_rejected(self):
        self.assert_text_rejected(self.text.replace(" -lca", ""))

    def test_ucli_execution_failure_does_not_consume_attempt(self):
        with self.assertRaisesRegex(RuntimeError, "UCLI power preflight execution"):
            MOD.audit_preflight_outcome(sim_rc=1)
        self.assertFalse(MOD.M1046_ATTEMPT.exists())

    def test_missing_or_empty_saif_rejected_before_attempt(self):
        with self.assertRaisesRegex(RuntimeError, "SAIF missing or empty"):
            MOD.audit_preflight_outcome(saif_exists=False, saif_bytes=0)
        with self.assertRaisesRegex(RuntimeError, "SAIF missing or empty"):
            MOD.audit_preflight_outcome(saif_bytes=0)

    def test_wrong_saif_hierarchy_or_duration_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "hierarchy/duration"):
            MOD.audit_preflight_outcome(dut_hierarchy=False)
        with self.assertRaisesRegex(RuntimeError, "hierarchy/duration"):
            MOD.audit_preflight_outcome(duration_ns=0)

    def test_good_preflight_is_pre_attempt(self):
        self.assertEqual(MOD.audit_preflight_outcome(),
                         {"preflight_passed": True, "attempt_consumed": False})

    def test_missing_preflight_call_rejected(self):
        self.assert_text_rejected(self.text.replace(
            "phase=UCLI_POWER_SAIF_PREFLIGHT\nrun_ucli_power_preflight",
            "phase=UCLI_POWER_SAIF_PREFLIGHT\n# injected missing preflight"))

    def test_axis_or_case_loss_rejected(self):
        self.assert_text_rejected(self.text.replace("for axis in k1 k8 k1x8; do",
                                                   "for axis in k1 k8; do"))
        self.assert_text_rejected(self.text.replace("for case_id in 0 1 2 3 4; do",
                                                   "for case_id in 0 1 2 3; do"))

    def test_namespace_collision_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "namespace collision"):
            MOD.audit_namespace({"attempt": True, "result": False})

    def test_license_value_not_returned(self):
        secret = "M1044_SECRET_MUST_NOT_LEAVE_CHECKER"
        result = MOD.audit_license_environment({"LM_LICENSE_FILE": secret})
        self.assertNotIn(secret, repr(result))
        self.assertEqual(result, {"license_route_present": True,
                                  "license_value_recorded": False})


if __name__ == "__main__":
    unittest.main()
