#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/analyze_m32_threshold_carry_late_scale_r2.py"
)
SPEC = importlib.util.spec_from_file_location("m32_r2", str(SCRIPT))
M32 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M32)


class M32ThresholdCarryLateScaleR2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = M32.build_report(M32.DEFAULT_CONTRACT)

    def test_candidate_language_and_threshold_manifest(self):
        census = self.report["candidate_census"]
        self.assertEqual(census["candidate_factorable_operators"], 10)
        self.assertEqual(census["candidate_factorable_cycles"], 105888197)
        self.assertEqual(census["candidate_factorable_outputs_per_sample"], 30456000)
        self.assertEqual(census["continuous_preserved_operators"], 2)
        self.assertTrue(all(
            not row["semantic_admission"] for row in census["candidates"]
        ))
        thresholds = self.report["checkpoint_threshold_audit"]["producer_thresholds"]
        self.assertEqual(len(thresholds), 10)
        self.assertTrue(all(row["shape"] == [] for row in thresholds))
        self.assertFalse(self.report["semantic_admission"])
        self.assertFalse(self.report["headline_admitted"])

    def test_balanced_radix_product_extremes_and_oracle(self):
        oracle = self.report["signed_product_oracle"]
        self.assertEqual(oracle["signed_int8_products_per_output"], 20)
        self.assertEqual(oracle["cases"], 4152)
        self.assertEqual(oracle["mismatches"], 0)
        for accumulator in (-(1 << 31), -1, 0, 1, (1 << 31) - 1):
            for threshold in (-(1 << 23), -1, 0, 1, (1 << 23) - 1):
                product, acc_digits, threshold_digits = (
                    M32.balanced_product_acc32_q24(accumulator, threshold)
                )
                self.assertEqual(product, accumulator * threshold)
                self.assertEqual(len(acc_digits) * len(threshold_digits), 20)
                self.assertTrue(all(-128 <= digit <= 127 for digit in acc_digits))
                self.assertTrue(all(-128 <= digit <= 127 for digit in threshold_digits))

    def test_control_charge_changes_the_gate(self):
        rows = {
            (row["line"], row["variant"]): row
            for row in self.report["control_charged_cycle_sensitivity"]["rows"]
        }
        local20 = rows[("local", "balanced_radix20_exact_product")]
        self.assertEqual(local20["products_per_output"], 20)
        self.assertEqual(local20["outputs_per_cycle_floor"], 4)
        self.assertEqual(local20["proportional_frontend_control_cycles"], 1974013)
        self.assertEqual(local20["control_charged_proposal_cycles_sensitivity"], 226409234)
        self.assertFalse(local20["crosses_2p75x_sensitivity"])

        motion20 = rows[("motion", "balanced_radix20_exact_product")]
        self.assertEqual(motion20["proportional_frontend_control_cycles"], 2026532)
        self.assertEqual(motion20["control_charged_proposal_cycles_sensitivity"], 224198314)
        self.assertTrue(motion20["crosses_2p75x_sensitivity"])
        self.assertFalse(motion20["crosses_3x_sensitivity"])

    def test_dual_trace_is_not_a_decorative_input(self):
        audit = self.report["dual_trace_crosscheck"]
        self.assertEqual(audit["operators"], 12)
        self.assertEqual(audit["records"], 120)
        self.assertEqual(audit["samples"], 10)

    def test_supplemental_hash_drift_fails_closed(self):
        contract = json.loads(M32.DEFAULT_CONTRACT.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tempdir:
            path = Path(tempdir) / "contract.json"
            contract["inputs"]["threshold_manifest"]["sha256"] = "0" * 64
            path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash drift"):
                M32.build_report(path)

    def test_threshold_value_drift_fails_closed(self):
        candidates = self.report["candidate_census"]["candidates"]
        manifest_path = M32.resolve_path(
            json.loads(M32.DEFAULT_CONTRACT.read_text(encoding="utf-8"))
            ["inputs"]["threshold_manifest"]["path"]
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["producers"][0]["value_float32"] = 0.5
        with self.assertRaisesRegex(ValueError, "raw-byte drift"):
            M32.verify_threshold_manifest(
                manifest,
                candidates,
                self.report["identity"]["verified_supplemental_sha256"]
                ["threshold_extractor"],
            )


if __name__ == "__main__":
    unittest.main()
