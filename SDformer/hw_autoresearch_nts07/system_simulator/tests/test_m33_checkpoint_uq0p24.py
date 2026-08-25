#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/analyze_m33_checkpoint_uq0p24.py"
)
SPEC = importlib.util.spec_from_file_location("m33_uq", str(SCRIPT))
M33 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M33)


class M33CheckpointUQ0P24Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = M33.build_report(M33.DEFAULT_CONTRACT)

    def test_all_ten_thresholds_are_exact(self):
        rows = self.report["thresholds"]
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(row["exact_roundtrip"] for row in rows))
        self.assertEqual(
            [row["uq0p24_raw_hex"] for row in rows],
            [
                "fffffe", "fffff1", "ffffff", "ffffeb", "ffff92",
                "ffffee", "ffff87", "ffff70", "ffff9f", "fffdb4",
            ],
        )

    def test_radix_and_signed56_contract(self):
        schedule = self.report["radix_schedule"]
        self.assertEqual(schedule["signed_int8_products_per_output"], 20)
        self.assertEqual(schedule["outputs_per_96_lane_cycle_floor"], 4)
        self.assertEqual(schedule["active_multiplier_lanes_full_packet"], 80)
        self.assertEqual(schedule["spare_multiplier_lanes_full_packet"], 16)
        self.assertTrue(self.report["signed56_range_proof"]["fits"])
        self.assertEqual(
            self.report["signed56_range_proof"]["minimum_product"],
            -(1 << 55) + (1 << 31),
        )

    def test_balanced_radix_boundaries(self):
        for value in (0, 1, 63, 64, 127, 128, (1 << 24) - 1):
            digits = M33.balanced_radix128_unsigned24(value)
            self.assertEqual(
                sum(digit * (128 ** index)
                    for index, digit in enumerate(digits)),
                value,
            )
            self.assertTrue(all(-64 <= digit <= 63 for digit in digits[:3]))
            self.assertTrue(0 <= digits[3] <= 8)

    def test_claim_boundary(self):
        admission = self.report["admission"]
        self.assertTrue(admission["threshold_representation_admitted"])
        for field in (
            "full_fixed_point_pipeline_admitted", "rne_saturation_bias_admitted",
            "rtl_admitted", "cycle_performance_admitted", "headline_admitted",
        ):
            self.assertFalse(admission[field])

    def test_contract_hash_drift_fails_closed(self):
        contract = json.loads(
            M33.DEFAULT_CONTRACT.read_text(encoding="utf-8")
        )
        contract["inputs"]["threshold_manifest"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash drift"):
                M33.build_report(path)


if __name__ == "__main__":
    unittest.main()
