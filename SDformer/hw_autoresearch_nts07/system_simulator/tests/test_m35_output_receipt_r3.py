#!/usr/bin/env python3
"""Identity and exact-comparison checks for the M35 output receipt r3 overlay."""

import hashlib
import json
import unittest
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW_ROOT = ROOT / "hw_autoresearch_nts07"
RECEIPT_PATH = HW_ROOT / "contracts/m35_output_receipt_r3_20260822.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class M35ReceiptR3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))

    def test_recursive_receipt_identity(self):
        r3 = self.receipt
        self.assertEqual(r3["schema"], "m35_output_receipt_v3")
        r2_path = ROOT / r3["supersedes"]["path"]
        self.assertEqual(digest(r2_path), r3["supersedes"]["sha256"])
        r2 = json.loads(r2_path.read_text(encoding="utf-8"))
        self.assertEqual(r2["schema"], r3["m35_r2_recursive_anchor"]["schema"])
        self.assertEqual(r2["status"], r3["m35_r2_recursive_anchor"]["status"])
        m33 = r3["m33_final_recursive_anchor"]
        m33_path = ROOT / m33["path"]
        self.assertEqual(digest(m33_path), m33["sha256"])
        m33_receipt = json.loads(m33_path.read_text(encoding="utf-8"))
        self.assertEqual(m33_receipt["schema"], m33["schema"])
        self.assertEqual(m33_receipt["status"], m33["status"])

    def test_formality_and_area_anchors(self):
        m35 = self.receipt["m35_r2_recursive_anchor"]
        m33 = self.receipt["m33_final_recursive_anchor"]
        self.assertEqual([m35["formality_r7"][key] for key in
                          ("passing_compare_points", "failing_compare_points",
                           "unmatched_compare_points")], [2333, 0, 0])
        self.assertEqual([m33["formality_flat_r2"][key] for key in
                          ("passing_compare_points", "failing_compare_points",
                           "unmatched_compare_points")], [655, 0, 0])
        self.assertEqual(m35["dc_sta_r7"]["cell_area_um2"], 19633.571938)
        self.assertEqual(m33["dc_sta_flat_r2"]["cell_area_um2"], 12997.403898)

    def test_exact_density_math_and_claim_boundary(self):
        row = self.receipt["strict_fair_flat_standalone_comparison"]
        area33 = Fraction(12997403898, 1000000)
        area35 = Fraction(19633571938, 1000000)
        self.assertEqual(Fraction(**row["m35_over_m33_area_exact"]), area35 / area33)
        self.assertEqual(Fraction(**row["m35_over_m33_peak_result_rate_exact"]),
                         Fraction(2, 1))
        self.assertEqual(Fraction(**row["m35_over_m33_result_rate_per_area_exact"]),
                         2 * area33 / area35)
        self.assertEqual(Fraction(**row["m35_area_per_result_reduction_exact"]),
                         1 - area35 / (2 * area33))
        self.assertTrue(row["strict_fair_density_admitted"])
        self.assertFalse(row["integrated_density_admitted"])
        self.assertFalse(self.receipt["paper_ppa_ready"])
        self.assertFalse(self.receipt["headline_admitted"])


if __name__ == "__main__":
    unittest.main()
