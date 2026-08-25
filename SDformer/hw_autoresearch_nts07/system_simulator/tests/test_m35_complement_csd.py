#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_m35_complement_csd.py"
SPEC = importlib.util.spec_from_file_location("m35_csd", str(SCRIPT))
M35 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M35)


class M35ComplementCSDTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = M35.build_report(M35.DEFAULT_CONTRACT)

    def test_checkpoint_deltas_and_term_counts(self):
        rows = self.report["thresholds"]
        self.assertEqual([row["delta"] for row in rows],
                         [2, 15, 1, 21, 110, 18, 121, 144, 97, 588])
        self.assertEqual([row["csd_nonzero_terms"] for row in rows],
                         [1, 2, 1, 3, 3, 2, 3, 2, 3, 4])
        self.assertEqual([row["minimum_signed_power_terms"] for row in rows],
                         [1, 2, 1, 3, 3, 2, 3, 2, 3, 4])
        self.assertTrue(all(row["minimum_term_count_exhaustively_proven"]
                            for row in rows))
        self.assertTrue(all(row["maximum_shift"] <= 9 for row in rows))
        self.assertEqual(
            self.report["architecture_bound"]["minimum_term_search_maximum_shift"],
            12,
        )
        self.assertIn(
            "H>=13",
            self.report["architecture_bound"]["minimum_term_search_sufficiency_lemma"],
        )

    def test_signed_edges_are_exact(self):
        for row in self.report["thresholds"]:
            for accumulator in (-(1 << 31), -1, 0, 1, (1 << 31) - 1):
                self.assertEqual(
                    M35.complement_product(
                        accumulator, row["delta"], row["csd_terms"]
                    ),
                    accumulator * row["threshold_uq0p24_raw"],
                )

    def test_regression_and_architecture_bound(self):
        regression = self.report["regression"]
        self.assertEqual(regression["total_cases"], 10150)
        self.assertEqual(regression["mismatches"], 0)
        architecture = self.report["architecture_bound"]
        self.assertEqual(architecture["runtime_integer_multiplier_products_per_output"], 0)
        self.assertEqual(architecture["signed_shift_terms_per_output_upper_bound"], 4)
        self.assertTrue(architecture["correction_fits_signed42"])
        self.assertEqual(architecture["correction_minimum"],
                         -(1 << 31) * 588)
        self.assertEqual(architecture["correction_maximum"],
                         ((1 << 31) - 1) * 588)
        self.assertLessEqual(architecture["signed42_minimum"],
                             architecture["correction_minimum"])
        self.assertGreaterEqual(architecture["signed42_maximum"],
                                architecture["correction_maximum"])
        self.assertTrue(
            architecture["design_target_has_no_math_resource_or_timing_admission"]
        )

    def test_claim_boundary(self):
        admission = self.report["admission"]
        self.assertTrue(admission["checkpoint_complement_bound_admitted"])
        self.assertTrue(admission["integer_csd_identity_admitted"])
        for name, value in admission.items():
            if name not in (
                "checkpoint_complement_bound_admitted",
                "integer_csd_identity_admitted",
            ):
                self.assertFalse(value, name)

    def test_type_and_hash_drift_fail_closed(self):
        with self.assertRaises(TypeError):
            M35.canonical_signed_digit(1.5)
        contract = json.loads(M35.DEFAULT_CONTRACT.read_text(encoding="utf-8"))
        contract["inputs"]["m33_uq_cross_product"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash drift"):
                M35.build_report(path)


if __name__ == "__main__":
    unittest.main()
