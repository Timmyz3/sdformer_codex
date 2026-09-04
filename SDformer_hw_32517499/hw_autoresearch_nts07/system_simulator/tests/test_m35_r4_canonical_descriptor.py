#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = (Path(__file__).resolve().parents[1] /
          "scripts/analyze_m35_r4_canonical_descriptor.py")
SPEC = importlib.util.spec_from_file_location("m35_r4_canonical", str(SCRIPT))
M35_R4 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M35_R4)


class M35R4CanonicalDescriptorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract, cls.rows, cls.anchors = M35_R4.validate_contract()
        cls.report = M35_R4.build_report()

    def test_frozen_identity_and_one_to_one_ids(self):
        identity = self.report["identity"]
        self.assertEqual(identity["contract_sha256"],
                         M35_R4.EXPECTED_CONTRACT_SHA256)
        self.assertEqual(identity["descriptor_rows_sha256"],
                         M35_R4.EXPECTED_DESCRIPTOR_ROWS_SHA256)
        self.assertEqual(identity["rtl_fingerprint64"],
                         M35_R4.EXPECTED_FINGERPRINT64)
        self.assertEqual([row["descriptor_id"] for row in self.rows],
                         list(range(10)))
        self.assertEqual([row["delta"] for row in self.rows],
                         M35_R4.EXPECTED_DELTAS)
        serialized = M35_R4.canonical_descriptor_bytes(self.rows)
        self.assertEqual(M35_R4.hashlib.sha256(serialized).hexdigest(),
                         M35_R4.EXPECTED_DESCRIPTOR_ROWS_SHA256)

    def test_all_exact_rows_admit_and_all_other_ids_reject(self):
        for row in self.rows:
            self.assertEqual(
                M35_R4.raw_tuple_to_descriptor_id(row["terms"], self.rows),
                row["descriptor_id"],
            )
        self.assertEqual(
            self.report["descriptor_boundary"]
            ["invalid_descriptor_ids_rejected"],
            list(range(10, 16)),
        )

    def test_exhaustive_legacy_noncanonical_space_rejected(self):
        audit = self.report["descriptor_boundary"]
        self.assertEqual(audit["legacy_r3_frozen_delta_tuples"], 3620)
        self.assertEqual(
            audit["legacy_review_noncanonical_tuples"], 3577)
        self.assertEqual(
            audit["r4_review_noncanonical_tuples_rejected"], 3577)
        self.assertEqual(
            audit["r4_additional_order_or_hole_variants_rejected"], 33)
        self.assertEqual(audit["r4_exact_rows_accepted"], 10)
        self.assertEqual(
            audit["r4_total_legacy_frozen_tuples_rejected"], 3610)
        witness = audit["first_rejected_witness"]
        with self.assertRaisesRegex(ValueError, "not one frozen canonical"):
            M35_R4.raw_tuple_to_descriptor_id(
                [M35_R4.numeric_slot(value)
                 for value in witness["numeric_slots"]], self.rows)

    def test_invalid_slot_metadata_and_type_coercions_reject(self):
        bad_negative = [dict(term) for term in self.rows[0]["terms"]]
        bad_negative[1]["negative"] = True
        with self.assertRaisesRegex(ValueError, "invalid-slot metadata"):
            M35_R4.raw_tuple_to_descriptor_id(bad_negative, self.rows)
        bad_shift = [dict(term) for term in self.rows[0]["terms"]]
        bad_shift[2]["shift"] = 1
        with self.assertRaisesRegex(ValueError, "invalid-slot metadata"):
            M35_R4.raw_tuple_to_descriptor_id(bad_shift, self.rows)
        bad_bool = [dict(term) for term in self.rows[0]["terms"]]
        bad_bool[0]["valid"] = 1
        with self.assertRaisesRegex(ValueError, "exact bool"):
            M35_R4.raw_tuple_to_descriptor_id(bad_bool, self.rows)
        bad_shift_type = [dict(term) for term in self.rows[0]["terms"]]
        bad_shift_type[0]["shift"] = 1.0
        with self.assertRaisesRegex(ValueError, "exact integer"):
            M35_R4.raw_tuple_to_descriptor_id(bad_shift_type, self.rows)

    def test_signed56_identity_and_static_rtl_boundary(self):
        product = self.report["product_identity"]
        self.assertEqual(product["total_products"], 100070)
        self.assertEqual(product["mismatches"], 0)
        self.assertTrue(product["all_products_fit_signed56"])
        rtl = self.report["rtl_candidate"]
        self.assertTrue(rtl["descriptor_rom_matches_contract"])
        self.assertTrue(rtl["runtime_raw_descriptor_payload_absent"])
        self.assertTrue(rtl["invalid_id_default_reject_present"])
        self.assertEqual(rtl["integer_multiplication_operators_lexical"], 0)
        self.assertEqual(
            rtl["verification_status"],
            "STATIC_CANDIDATE_ONLY_NOT_VCS_OR_SYNTHESIS",
        )

    def test_duplicate_keys_and_contract_path_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            duplicate = Path(directory) / "duplicate.json"
            duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                M35_R4.read_json(duplicate)
            copy = Path(directory) / "contract.json"
            copy.write_text(json.dumps(self.contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen canonical contract path"):
                M35_R4.validate_contract(copy)

    def test_claim_boundary_is_model_and_static_only(self):
        admission = self.report["admission"]
        self.assertTrue(
            admission["ten_frozen_descriptor_id_membership_model_admitted"])
        self.assertTrue(
            admission["legacy_3577_noncanonical_tuple_rejection_model_admitted"])
        self.assertTrue(admission["signed56_integer_identity_model_admitted"])
        self.assertTrue(admission["rtl_static_rom_identity_admitted"])
        for key, value in admission.items():
            if key not in (
                "ten_frozen_descriptor_id_membership_model_admitted",
                "legacy_3577_noncanonical_tuple_rejection_model_admitted",
                "signed56_integer_identity_model_admitted",
                "rtl_static_rom_identity_admitted",
            ):
                self.assertFalse(value, key)
        self.assertIn("different checkpoint",
                      self.report["generality_tradeoff"]["cost"])


if __name__ == "__main__":
    unittest.main()
