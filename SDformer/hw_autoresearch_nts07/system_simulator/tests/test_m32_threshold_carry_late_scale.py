#!/usr/bin/env python3

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/analyze_m32_threshold_carry_late_scale.py"
)
SPEC = importlib.util.spec_from_file_location("m32_threshold_carry", str(SCRIPT))
M32 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M32)


class M32ThresholdCarryLateScaleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = M32.build_report(M32.DEFAULT_CONTRACT)

    def test_frozen_census_and_continuous_paths(self):
        census = self.report["census"]
        self.assertEqual(census["factorable_bypass_operators"], 10)
        self.assertEqual(census["factorable_bypass_cycles"], 105888197)
        self.assertEqual(census["factorable_outputs_per_sample"], 30456000)
        self.assertEqual(census["continuous_bypass_operators"], 2)
        self.assertEqual(census["continuous_bypass_cycles"], 27099543)
        continuous = set(row["name"] for row in census["continuous_preserved"])
        self.assertEqual(continuous, {
            "sttmultires_unet.encoders.swin3d.patch_embed.head.conv.0",
            "sttmultires_unet.encoders.swin3d.patch_embed.proj.conv_res",
        })
        self.assertTrue(all(row["calls"] == 10 for row in census["factorable"]))

    def test_resource_explicit_sensitivity_is_not_a_claim(self):
        rows = {
            (row["line"], row["variant"]): row
            for row in self.report["cycle_sensitivity"]["rows"]
        }
        local12 = rows[("local", "byte12_arithmetic_lower_bound")]
        self.assertEqual(local12["late_scale_outputs_per_cycle"], 8)
        self.assertEqual(local12["late_scale_cycles"], 3807000)
        self.assertEqual(local12["proposal_compute_cycles_sensitivity"], 220628221)
        self.assertAlmostEqual(local12["speedup_vs_fixed_sensitivity"], 2.8140925952)
        self.assertTrue(local12["crosses_2p75x"])
        self.assertFalse(local12["crosses_3x"])

        motion48 = rows[("motion", "stress48")]
        self.assertEqual(motion48["late_scale_outputs_per_cycle"], 2)
        self.assertEqual(motion48["late_scale_cycles"], 15228000)
        self.assertFalse(motion48["crosses_3x"])
        self.assertFalse(self.report["headline_admitted"])
        self.assertIn("not an executable", self.report["cycle_sensitivity"]["interpretation"])

    def _mutated_contract(self, mutate_key, mutate_rows=None, mutate_text=None):
        original = json.loads(M32.DEFAULT_CONTRACT.read_text(encoding="utf-8"))
        source_path = M32.resolve_path(original["inputs"][mutate_key]["path"])
        tempdir = tempfile.TemporaryDirectory()
        temp_root = Path(tempdir.name)
        target = temp_root / source_path.name
        if mutate_rows is not None:
            with source_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
                fields = list(rows[0].keys())
            mutate_rows(rows)
            with target.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        else:
            text = source_path.read_text(encoding="utf-8")
            target.write_text(mutate_text(text), encoding="utf-8")
        original["inputs"][mutate_key]["path"] = str(target)
        original["inputs"][mutate_key]["sha256"] = M32.sha256(target)
        contract_path = temp_root / "contract.json"
        contract_path.write_text(json.dumps(original), encoding="utf-8")
        return tempdir, contract_path

    def test_nonbinary_predecessor_fails_closed(self):
        target = self.report["census"]["factorable"][0]["producer"]

        def mutate(rows):
            for row in rows:
                if row["name"] == target:
                    row["output_mode"] = "ternary"

        tempdir, contract = self._mutated_contract("atlif_activity", mutate_rows=mutate)
        try:
            with self.assertRaisesRegex(ValueError, "non-binary ATLIF"):
                M32.build_report(contract)
        finally:
            tempdir.cleanup()

    def test_shape_mismatch_fails_closed(self):
        target = self.report["census"]["factorable"][0]["name"]

        def mutate(rows):
            for row in rows:
                if row["kind"] == "operator" and row["name"] == target:
                    row["input_shape"] = "[1]"
                    break

        tempdir, contract = self._mutated_contract("execution_trace", mutate_rows=mutate)
        try:
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                M32.build_report(contract)
        finally:
            tempdir.cleanup()

    def test_source_semantic_drift_fails_closed(self):
        def mutate(text):
            return text.replace(
                "return out * thre, thre_updates",
                "return out, thre_updates",
                1,
            )

        tempdir, contract = self._mutated_contract("atlif_source", mutate_text=mutate)
        try:
            with self.assertRaisesRegex(ValueError, "out \* thre"):
                M32.build_report(contract)
        finally:
            tempdir.cleanup()

    def test_scalar_factor_algebra_and_bias_order(self):
        weights = [3, -5, 7, 2]
        bits = [1, 0, 1, 1]
        theta = 0.875
        bias = -1.25
        direct = sum(w * (theta * b) for w, b in zip(weights, bits)) + bias
        carried = theta * sum(w * b for w, b in zip(weights, bits)) + bias
        incorrectly_scaled_bias = theta * (sum(w * b for w, b in zip(weights, bits)) + bias)
        self.assertEqual(direct, carried)
        self.assertNotEqual(direct, incorrectly_scaled_bias)


if __name__ == "__main__":
    unittest.main()
