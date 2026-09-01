#!/usr/bin/env python3
from __future__ import print_function

import importlib.util
import json
import os
from pathlib import Path
import unittest


HW = Path(__file__).resolve().parents[2]
CHECKER = HW / "system_simulator/scripts/check_m1837_m1808_c3_preflight_recovery_source.py"
SPEC = importlib.util.spec_from_file_location("m1837_checker", str(CHECKER))
C = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(C)


class M1837RecoverySourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract_text = C.CONTRACT.read_text()
        cls.contract = json.loads(cls.contract_text)

    def mutated_contract(self, mutate):
        value = json.loads(self.contract_text)
        mutate(value)
        return json.dumps(value, sort_keys=True)

    def assert_contract_rejected(self, mutate):
        with self.assertRaises(C.CheckFailure):
            C.validate_sources(self.mutated_contract(mutate))

    def test_01_actual_source_and_governance_state_pass(self):
        result = C.validate_sources()
        self.assertEqual(result["status"], "PASS_M1837_ONE_MANUAL_RECOVERY_SOURCE")
        self.assertFalse(result["attempt_consumed"])
        self.assertFalse(result["launch_authorized_now"])

    def test_02_preflight_failure_is_preserved_and_fully_sealed(self):
        members = C.verify_sealed_directory(
            C.PREFLIGHT_QUARANTINE, C.FIXED_SHA["preflight_manifest"],
            C.FIXED_SHA["preflight_outer"])
        self.assertEqual(members, {"failure.json": C.FIXED_SHA["preflight_failure"]})
        self.assertFalse(os.path.lexists(str(C.ORIGINAL_FAILURE)))

    def test_03_attempt_result_private_are_all_absent(self):
        for path in (C.ATTEMPT, C.RESULT, C.PRIVATE):
            self.assertFalse(os.path.lexists(str(path)), str(path))

    def test_04_exact_preflight_semantics_pass(self):
        C.validate_failure_value(dict(C.EXPECTED_FAILURE))

    def test_05_reject_attempt_consumed_true(self):
        value = dict(C.EXPECTED_FAILURE); value["attempt_consumed"] = True
        with self.assertRaises(C.CheckFailure): C.validate_failure_value(value)

    def test_06_reject_nonzero_vcs(self):
        value = json.loads(json.dumps(C.EXPECTED_FAILURE)); value["counts"]["vcs_compiles"] = 1
        with self.assertRaises(C.CheckFailure): C.validate_failure_value(value)

    def test_07_reject_nonzero_sim(self):
        value = json.loads(json.dumps(C.EXPECTED_FAILURE)); value["counts"]["simv_runs"] = 1
        with self.assertRaises(C.CheckFailure): C.validate_failure_value(value)

    def test_08_reject_nonzero_saif(self):
        value = json.loads(json.dumps(C.EXPECTED_FAILURE)); value["counts"]["saif_files"] = 1
        with self.assertRaises(C.CheckFailure): C.validate_failure_value(value)

    def test_09_reject_nonzero_ptpx(self):
        value = json.loads(json.dumps(C.EXPECTED_FAILURE)); value["counts"]["ptpx_runs"] = 1
        with self.assertRaises(C.CheckFailure): C.validate_failure_value(value)

    def test_10_reject_non_source_chain_phase(self):
        value = dict(C.EXPECTED_FAILURE); value["phase"] = "LICENSE_PREFLIGHT"
        with self.assertRaises(C.CheckFailure): C.validate_failure_value(value)

    def test_11_correct_m1815_manifest_is_exact_64_hex(self):
        digest = C.FIXED_SHA["m1815_manifest"]
        self.assertEqual(len(digest), 64)
        self.assertEqual(digest, "5b124a54f4bfe9b64369990958a053175358d97783f080aff08b99c923233099")

    def test_12_original_runner_and_source_identity_are_exact(self):
        C.verify_original_authority()
        self.assertEqual(C.sha(C.RUNNER), C.FIXED_SHA["runner"])
        self.assertEqual(C.sha(C.M1808_CONTRACT), C.FIXED_SHA["m1808_contract"])

    def test_13_reject_wrong_failure_hash(self):
        self.assert_contract_rejected(lambda v: v["preflight_rejection_evidence"].update(
            failure_json_sha256="0" * 64))

    def test_14_reject_wrong_quarantine_name(self):
        self.assert_contract_rejected(lambda v: v["preflight_rejection_evidence"].update(
            quarantine="results/old.failure"))

    def test_15_reject_failure_delete_claim(self):
        self.assert_contract_rejected(lambda v: v["preflight_rejection_evidence"].update(
            preserved_not_deleted=False))

    def test_16_reject_wrong_m1815_manifest_pin(self):
        self.assert_contract_rejected(lambda v: v["frozen_identity"].update(
            m1815_correct_manifest_sha256=C.FIXED_SHA["m1815_manifest"] + "0"))

    def test_17_reject_wrong_m1816_release(self):
        self.assert_contract_rejected(lambda v: v["frozen_identity"].update(
            m1816_release_sha256="0" * 64))

    def test_18_reject_different_runner(self):
        self.assert_contract_rejected(lambda v: v["manual_recovery_policy"].update(
            same_runner_sha256="0" * 64))

    def test_19_reject_two_relaunches(self):
        self.assert_contract_rejected(lambda v: v["manual_recovery_policy"].update(
            proposed_relaunches=2))

    def test_20_reject_automatic_retry(self):
        self.assert_contract_rejected(lambda v: v["manual_recovery_policy"].update(
            automatic_retry=True))

    def test_21_reject_original_release_alone_sufficient(self):
        self.assert_contract_rejected(lambda v: v["manual_recovery_policy"].update(
            original_m1816_alone_no_longer_sufficient=False))

    def test_22_reject_missing_independent_source_review(self):
        self.assert_contract_rejected(lambda v: v["manual_recovery_policy"].update(
            independent_m1837_source_review_required=False))

    def test_23_reject_missing_final_recovery_release(self):
        self.assert_contract_rejected(lambda v: v["manual_recovery_policy"].update(
            separately_double_sealed_final_recovery_release_required=False))

    def test_24_reject_final_release_created_now(self):
        self.assert_contract_rejected(lambda v: v["manual_recovery_policy"].update(
            final_recovery_release_created_now=True))

    def test_25_reject_second_relaunch_after_failure(self):
        self.assert_contract_rejected(lambda v: v["manual_recovery_policy"].update(
            second_relaunch_forbidden_even_if_recovery_fails=False))

    def test_26_reject_result_hammer_omitting_preflight_failure(self):
        self.assert_contract_rejected(lambda v: v["future_independent_result_hammer"].update(
            must_audit_preflight_failure_sha256=""))

    def test_27_reject_result_hammer_omitting_consumed_attempt(self):
        self.assert_contract_rejected(lambda v: v["future_independent_result_hammer"].update(
            must_audit_unique_consumed_attempt=""))

    def test_28_reject_result_hammer_that_can_hide_preflight(self):
        self.assert_contract_rejected(lambda v: v["future_independent_result_hammer"].update(
            may_not_hide_or_replace_preflight_failure=False))

    def test_29_reject_any_claim_promotion(self):
        self.assert_contract_rejected(lambda v: v["claim_boundary"].update(
            component_energy=True))

    def test_30_reject_author_eda_execution(self):
        self.assert_contract_rejected(lambda v: v["author_execution"].update(
            ptpx_runs=1))

    def test_31_reject_author_result_creation(self):
        self.assert_contract_rejected(lambda v: v["author_execution"].update(
            results_created=1))

    def test_32_reject_duplicate_json_key(self):
        attack = self.contract_text.replace(
            '"milestone": "M1837",',
            '"milestone": "M1837",\n  "milestone": "M1836",', 1)
        with self.assertRaises(C.CheckFailure): C.validate_sources(attack)

    def test_33_checker_contains_no_launch_or_license_api(self):
        text = CHECKER.read_text()
        for token in ("import subprocess", "subprocess.", "lmutil", "fm_shell",
                      "pt_shell", "vcs -full64", "ATTEMPT.mkdir"):
            self.assertNotIn(token, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
