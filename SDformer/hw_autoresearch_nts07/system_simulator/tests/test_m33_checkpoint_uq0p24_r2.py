#!/usr/bin/env python3

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_m33_checkpoint_uq0p24_r2.py"
SPEC = importlib.util.spec_from_file_location("m33_uq_r2", str(SCRIPT))
M33 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M33)


class M33CheckpointUQ0P24R2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = M33.build_report(M33.DEFAULT_CONTRACT)

    def test_raw_decode_and_checkpoint_exactness(self):
        rows = self.report["thresholds"]
        self.assertEqual(len(rows), 10)
        self.assertEqual(
            [row["uq0p24_raw_hex"] for row in rows],
            ["fffffe", "fffff1", "ffffff", "ffffeb", "ffff92",
             "ffffee", "ffff87", "ffff70", "ffff9f", "fffdb4"],
        )
        for row in rows:
            word, uq_raw = M33.float32_raw_to_exact_uq0p24(
                bytes.fromhex(row["float32_raw_le_hex"])
            )
            self.assertEqual("{:08x}".format(word), row["float32_raw_word_hex"])
            self.assertEqual(uq_raw, row["uq0p24_raw"])

    def test_constructive_cross_product_edges(self):
        for accumulator in (-(1 << 31), -129, -1, 0, 1, 127,
                            (1 << 31) - 1):
            for threshold in (0, 1, 63, 64, 127, 128,
                              (1 << 24) - 2, (1 << 24) - 1):
                self.assertEqual(
                    M33.cross_product(accumulator, threshold),
                    accumulator * threshold,
                )

    def test_regression_is_frozen_and_zero_mismatch(self):
        regression = self.report["cross_product_regression"]
        self.assertEqual(regression["total_cases"], 10150)
        self.assertEqual(regression["mismatches"], 0)
        self.assertEqual(regression["seed_hex"], "0x4d333202")
        self.assertEqual(len(regression["vector_and_result_sha256"]), 64)

    def test_type_and_range_fail_closed(self):
        with self.assertRaises(TypeError):
            M33.balanced_radix128(1.5, 4, 0, (1 << 24) - 1)
        with self.assertRaises(ValueError):
            M33.uq24_digits(1 << 24)
        with self.assertRaises(ValueError):
            M33.float32_raw_to_exact_uq0p24(bytes.fromhex("0000803f"))

    def test_signed56_formula_and_claim_boundary(self):
        proof = self.report["signed56_range_proof"]
        self.assertEqual(proof["minimum_headroom"], 1 << 31)
        self.assertEqual(proof["maximum_headroom"], 2164260862)
        self.assertTrue(proof["fits"])
        admission = self.report["admission"]
        self.assertTrue(admission["threshold_representation_admitted"])
        self.assertTrue(admission["integer_cross_product_identity_admitted"])
        for name, value in admission.items():
            if name not in (
                "threshold_representation_admitted",
                "integer_cross_product_identity_admitted",
            ):
                self.assertFalse(value, name)

    def test_contract_hash_drift_fails_closed(self):
        contract = json.loads(M33.DEFAULT_CONTRACT.read_text(encoding="utf-8"))
        contract["inputs"]["threshold_manifest"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contract.json"
            path.write_text(json.dumps(contract), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash drift"):
                M33.build_report(path)


if __name__ == "__main__":
    unittest.main()
