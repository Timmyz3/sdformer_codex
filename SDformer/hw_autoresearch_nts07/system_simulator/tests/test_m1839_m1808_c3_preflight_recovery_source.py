#!/usr/bin/env python3
from __future__ import print_function

import importlib.util
import json
import os
from pathlib import Path
import unittest


HW = Path(__file__).resolve().parents[2]
CHECKER = HW / "system_simulator/scripts/check_m1839_m1808_c3_preflight_recovery_source.py"
SPEC = importlib.util.spec_from_file_location("m1839_checker", str(CHECKER))
C = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(C)


class M1839RecoverySourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = C.CONTRACT.read_text()
        cls.value = json.loads(cls.text)

    def reject(self, mutate):
        value = json.loads(self.text); mutate(value)
        with self.assertRaises(C.CheckFailure):
            C.validate_sources(json.dumps(value, sort_keys=True))

    def test_01_actual_source_passes(self):
        result = C.validate_sources()
        self.assertEqual(result["status"], "PASS_M1839_SCHEMA_CLOSED_RECOVERY_SOURCE")
        self.assertTrue(result["m1838_six_escapes_closed"])

    def test_02_m1838_fail_is_exact_and_sealed(self):
        C.verify_frozen_evidence()
        self.assertEqual(C.sha(C.M1838 / "review.json"), C.FIXED_SHA["m1838_review"])

    def test_03_attempt_result_private_absent(self):
        for path in (C.ATTEMPT, C.RESULT, C.PRIVATE):
            self.assertFalse(os.path.lexists(str(path)))

    # Exact replay of all six M1838 escapes.
    def test_04_escape_diagnosis_eda_true(self):
        self.reject(lambda v: v["diagnosis"].update(license_or_eda_reached=True))

    def test_05_escape_diagnosis_manifest_zero(self):
        self.reject(lambda v: v["diagnosis"].update(correct_m1815_manifest_sha256="0" * 64))

    def test_06_escape_diagnosis_attempt_true(self):
        self.reject(lambda v: v["diagnosis"].update(attempt_consumed=True))

    def test_07_escape_milestone_m9999(self):
        self.reject(lambda v: v.update(milestone="M9999"))

    def test_08_escape_purpose_immediate_launch(self):
        self.reject(lambda v: v.update(purpose="authorize immediate launch"))

    def test_09_escape_unknown_top_level_launch(self):
        self.reject(lambda v: v.update(launch_authorized_now=True))

    # Top-level closure and scalar type attacks.
    def test_10_reject_top_level_missing_diagnosis(self):
        self.reject(lambda v: v.pop("diagnosis"))

    def test_11_reject_top_level_unknown(self):
        self.reject(lambda v: v.update(unknown_governance=False))

    def test_12_reject_milestone_type(self):
        self.reject(lambda v: v.update(milestone=["M1839"]))

    def test_13_reject_purpose_type(self):
        self.reject(lambda v: v.update(purpose={"text": C.PURPOSE}))

    # Nested diagnosis unknown/missing/type.
    def test_14_reject_diagnosis_unknown_key(self):
        self.reject(lambda v: v["diagnosis"].update(hidden_launch=False))

    def test_15_reject_diagnosis_missing_key(self):
        self.reject(lambda v: v["diagnosis"].pop("diagnosis_is_not_a_tool_result"))

    def test_16_reject_diagnosis_bool_as_integer(self):
        self.reject(lambda v: v["diagnosis"].update(attempt_consumed=0))

    # Supersession closure.
    def test_17_reject_supersession_unknown(self):
        self.reject(lambda v: v["supersession"].update(release=True))

    def test_18_reject_supersession_missing(self):
        self.reject(lambda v: v["supersession"].pop("m1838_formal_fail_bound"))

    def test_19_reject_supersession_type(self):
        self.reject(lambda v: v["supersession"].update(m1838_formal_fail_bound=1))

    # M1838 review closure.
    def test_20_reject_m1838_unknown(self):
        self.reject(lambda v: v["m1838_failed_review"].update(pass_override=False))

    def test_21_reject_m1838_missing_escape(self):
        self.reject(lambda v: v["m1838_failed_review"]["reproduced_escapes"].pop())

    def test_22_reject_m1838_escape_count_bool(self):
        self.reject(lambda v: v["m1838_failed_review"].update(escape_count=True))

    # Preflight evidence closure.
    def test_23_reject_preflight_unknown(self):
        self.reject(lambda v: v["preflight_rejection_evidence"].update(hidden=False))

    def test_24_reject_preflight_missing_failure_hash(self):
        self.reject(lambda v: v["preflight_rejection_evidence"].pop("failure_json_sha256"))

    def test_25_reject_preflight_zero_as_false(self):
        self.reject(lambda v: v["preflight_rejection_evidence"].update(vcs_compiles=False))

    # Frozen identity closure.
    def test_26_reject_identity_unknown(self):
        self.reject(lambda v: v["frozen_identity"].update(alias="old"))

    def test_27_reject_identity_missing_runner(self):
        self.reject(lambda v: v["frozen_identity"].pop("runner_sha256"))

    def test_28_reject_identity_hash_type(self):
        self.reject(lambda v: v["frozen_identity"].update(runner_sha256=[C.FIXED_SHA["runner"]]))

    # Recovery policy closure.
    def test_29_reject_policy_unknown(self):
        self.reject(lambda v: v["manual_recovery_policy"].update(retry_budget=1))

    def test_30_reject_policy_missing_m1841(self):
        self.reject(lambda v: v["manual_recovery_policy"].pop("m1841_double_sealed_release_required"))

    def test_31_reject_policy_one_as_true(self):
        self.reject(lambda v: v["manual_recovery_policy"].update(proposed_relaunches=True))

    def test_32_reject_policy_two_relaunches(self):
        self.reject(lambda v: v["manual_recovery_policy"].update(proposed_relaunches=2))

    # Future result hammer closure.
    def test_33_reject_hammer_unknown(self):
        self.reject(lambda v: v["future_independent_result_hammer"].update(skip_preflight=False))

    def test_34_reject_hammer_missing_attempt(self):
        self.reject(lambda v: v["future_independent_result_hammer"].pop("must_audit_unique_consumed_attempt"))

    def test_35_reject_hammer_bool_as_integer(self):
        self.reject(lambda v: v["future_independent_result_hammer"].update(required=1))

    # Claim/execution closure.
    def test_36_reject_claim_unknown(self):
        self.reject(lambda v: v["claim_boundary"].update(launch=True))

    def test_37_reject_claim_missing(self):
        self.reject(lambda v: v["claim_boundary"].pop("headline"))

    def test_38_reject_claim_false_as_zero(self):
        self.reject(lambda v: v["claim_boundary"].update(headline=0))

    def test_39_reject_execution_unknown(self):
        self.reject(lambda v: v["author_execution"].update(tool_runs=0))

    def test_40_reject_execution_missing(self):
        self.reject(lambda v: v["author_execution"].pop("license_queries"))

    def test_41_reject_execution_zero_as_false(self):
        self.reject(lambda v: v["author_execution"].update(ptpx_runs=False))

    # Source inventory closure.
    def test_42_reject_source_row_unknown(self):
        self.reject(lambda v: v["source_files"][0].update(role="checker"))

    def test_43_reject_source_row_missing_hash(self):
        self.reject(lambda v: v["source_files"][0].pop("sha256"))

    def test_44_reject_source_hash_type(self):
        self.reject(lambda v: v["source_files"][0].update(sha256=["0" * 64]))

    def test_45_reject_duplicate_json_key(self):
        attack = self.text.replace('"milestone": "M1839",',
                                   '"milestone": "M1839",\n  "milestone": "M9999",', 1)
        with self.assertRaises(C.CheckFailure): C.validate_sources(attack)

    def test_46_checker_has_no_launch_or_license_api(self):
        text = CHECKER.read_text()
        for token in ("import subprocess", "subprocess.", "lmutil", "pt_shell",
                      "vcs -full64", "ATTEMPT.mkdir"):
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
