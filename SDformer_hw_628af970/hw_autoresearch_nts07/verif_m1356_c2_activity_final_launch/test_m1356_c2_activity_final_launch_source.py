#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""No-EDA tests for the M1356 final-launch authority source."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest import mock


HERE = Path(__file__).resolve().parent
CHECKER = HERE / "static_check_m1356_c2_activity_final_launch_source.py"
SPEC = importlib.util.spec_from_file_location("m1356_final_launch_source", CHECKER)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


class Tests(unittest.TestCase):
    def runner(self):
        return M.RUNNER.read_text(encoding="utf-8")

    def test_01_m1350_family_revalidates(self):
        self.assertTrue(M.M.validate_common(skip_author=False)["strict_json"])

    def test_02_m1353_exact_authority_passes(self):
        review = M.verify_m1353()
        self.assertEqual(review["fresh_hammer"]["false_negatives"], 0)

    def test_03_canonical_receipts_are_exact(self):
        audit = M.audit_receipts(self.runner())
        self.assertEqual(audit["failure_fields"], 13)
        self.assertEqual(audit["attempt_fields"], 9)
        self.assertEqual(audit["success_claims"], 9)

    def test_04_failure_identity_deletion_rejected(self):
        mutant = self.runner().replace(r"runner_sha256=%s\n", "", 1)
        with self.assertRaises(AssertionError): M.audit_receipts(mutant)

    def test_05_attempt_identity_deletion_rejected(self):
        marker = "printf 'status=M1344_ATTEMPT_CONSUMED"
        start = self.runner().index(marker)
        mutant = self.runner()[:start] + self.runner()[start:].replace(
            r"source_contract_sha256=%s\n", "", 1)
        with self.assertRaises(AssertionError): M.audit_receipts(mutant)

    def test_06_success_identity_deletion_rejected(self):
        mutant = self.runner().replace("'runner_sha256':sha(runner),", "", 1)
        with self.assertRaises(AssertionError): M.audit_receipts(mutant)

    def test_07_success_claim_lift_rejected(self):
        mutant = self.runner().replace("'performance':False", "'performance':True", 1)
        with self.assertRaisesRegex(AssertionError, "claims"):
            M.audit_receipts(mutant)

    def test_08_success_extra_claim_rejected(self):
        mutant = self.runner().replace("'headline':False}",
                                       "'headline':False,'launch_authorized':False}", 1)
        with self.assertRaisesRegex(AssertionError, "claims"):
            M.audit_receipts(mutant)

    def test_09_wrong_active_sha_expression_rejected(self):
        mutant = self.runner().replace("'source_contract_sha256':sha(contract)",
                                       "'source_contract_sha256':sha(runner)", 1)
        with self.assertRaises(AssertionError): M.audit_receipts(mutant)

    def test_10_duplicate_success_identity_rejected(self):
        token = "'runner_sha256':sha(runner),"
        mutant = self.runner().replace(token, token + token, 1)
        with self.assertRaises(AssertionError): M.audit_receipts(mutant)

    def test_11_collision_and_namespaces_pass(self):
        audit = M.audit_collision_and_namespaces(self.runner())
        self.assertTrue(audit["collision_before_attempt"])
        self.assertTrue(audit["attempt_fresh"])

    def test_12_attempt_collision_rejected(self):
        with mock.patch.object(M.os.path, "lexists", return_value=True):
            with self.assertRaisesRegex(AssertionError, "consumed/resident"):
                M.audit_collision_and_namespaces(self.runner())

    def test_13_collision_gate_reorder_rejected(self):
        mutant = self.runner().replace(
            'phase="RESOURCE_PREFLIGHT"\ncollision_gate\nresource_gate',
            'phase="RESOURCE_PREFLIGHT"\nresource_gate\ncollision_gate', 1)
        with self.assertRaisesRegex(AssertionError, "dominate"):
            M.audit_collision_and_namespaces(mutant)

    def test_14_attempt_publish_duplicate_rejected(self):
        token = 'publish_no_replace "${ATTEMPT_STAGE}" "${ATTEMPT}"'
        mutant = self.runner().replace(token, token + "\n" + token, 1)
        with self.assertRaisesRegex(AssertionError, "cardinality"):
            M.audit_collision_and_namespaces(mutant)

    def test_15_source_contract_launch_is_false(self):
        contract = M.validate_contract(skip_author=True)
        self.assertIs(contract["authorization"]["launch_authorized"], False)
        self.assertEqual(contract["claim_boundary"], M.EXACT_CLAIMS)

    def test_16_contract_launch_lift_rejected(self):
        contract = M.strict_json(M.CONTRACT)
        contract["authorization"]["launch_authorized"] = True
        with mock.patch.object(M, "strict_json", return_value=contract):
            with self.assertRaisesRegex(AssertionError, "authorization"):
                M.validate_contract(skip_author=True)

    def test_17_contract_extra_claim_rejected(self):
        contract = M.strict_json(M.CONTRACT)
        contract["claim_boundary"]["launch_authorized"] = False
        with mock.patch.object(M, "strict_json", return_value=contract):
            with self.assertRaisesRegex(AssertionError, "claims"):
                M.validate_contract(skip_author=True)

    def test_18_duplicate_json_key_rejected(self):
        text = M.CONTRACT.read_text(encoding="utf-8").replace(
            '"status":', '"status":"DUPLICATE","status":', 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "contract.json"; path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "duplicate JSON key"):
                M.strict_json(path)

    def test_19_future_blind_absent_and_residue_rejected(self):
        self.assertTrue(M.validate_future("source_absent")["future_blind_absent"])
        original = M.FUTURE_BLIND
        with tempfile.TemporaryDirectory() as tmp:
            M.FUTURE_BLIND = Path(tmp) / "residue"; M.FUTURE_BLIND.mkdir()
            try:
                with self.assertRaisesRegex(AssertionError, "residue"):
                    M.validate_future("source_absent")
            finally:
                M.FUTURE_BLIND = original

    def test_20_docs_ucli_and_no_execution_boundary(self):
        self.assertEqual(M.sha(M.DOCS359), M.DOCS359_SHA256)
        self.assertEqual(M.sha(M.UCLI), M.UCLI_SHA256)
        text = CHECKER.read_text(encoding="utf-8")
        self.assertNotIn("lmstat -a", text)
        self.assertNotIn("subprocess.run([str(M.RUNNER)", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
