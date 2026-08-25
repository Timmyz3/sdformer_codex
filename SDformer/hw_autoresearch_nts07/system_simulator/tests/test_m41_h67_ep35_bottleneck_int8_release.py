#!/usr/bin/env python3

import copy
import importlib.util
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / (
    "scripts/validate_m41_h67_ep35_bottleneck_int8_release.py")
SPEC = importlib.util.spec_from_file_location("m41_release", str(SCRIPT))
M41 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M41)


class M41H67BottleneckInt8ReleaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = M41.validate()
        cls.pin = M41.strict_json(M41.DEFAULT_PIN)
        cls.result_path = next(
            M41.resolve_under_root(row["path"]) for row in cls.pin["files"]
            if row["role"] == "result")
        cls.result = M41.strict_json(cls.result_path)

    def test_recursive_release_validation(self):
        self.assertEqual(
            self.summary["status"],
            "PASS_M41_RELEASE_PIN_RECURSIVE_PAYLOAD_AND_CLAIM_BOUNDARY_VALIDATION")
        self.assertEqual(self.summary["files_validated"], 23)
        self.assertEqual(self.summary["weights_validated"], 21233664)
        self.assertEqual(self.summary["raw_conv_output_values_anchored"], 92160000)

    def test_quantization_population_and_error(self):
        self.assertEqual(len(self.result["layers"]), 4)
        for layer in self.result["layers"]:
            self.assertEqual(layer["preclip_violation_count"], 0)
            self.assertEqual(layer["reserved_negative_128_count"], 0)
            self.assertLess(layer["weight_error"]["normalized_l2_error"], 0.02)
            self.assertLessEqual(layer["weight_error"]["max_error_div_stored_scale"], 0.5)

    def test_local_raw_conv_gate_is_not_full_network_accuracy(self):
        audit = self.result["local_raw_convolution_audit"]
        self.assertIsNone(audit["full_network_accuracy"])
        self.assertIsNone(audit["valid825_accuracy"])
        for row in audit["aggregate_by_layer"].values():
            self.assertTrue(row["gate_pass"])
            self.assertLessEqual(row["normalized_l2_error"], 0.03)
            self.assertGreaterEqual(row["cosine_similarity"], 0.999)

    def test_accumulator_bounds_are_19_tight_and_21_dense(self):
        bridge = self.result["m40_schedule_bridge"]
        self.assertEqual(bridge["checkpoint_tight_accumulator_signed_bits"], 19)
        self.assertEqual(bridge["dense_envelope_accumulator_signed_bits"], 21)
        self.assertFalse(bridge["full_96_output_tile_fits_current_residency"])

    def test_value_mask_is_not_misclaimed_as_cycle_speedup(self):
        admission = self.result["admission"]
        self.assertTrue(admission[
            "weight_value_mask_fetch_potential_statistics_admitted"])
        self.assertFalse(admission[
            "multicast_accumulator_cycle_reduction_admitted"])
        for layer in self.result["layers"]:
            for size in ("16", "24", "48", "96"):
                row = layer["value_mask_multicast"]["blocked_value_mask"][size]
                self.assertGreater(row["destination_updates_per_command"], 1.0)
                self.assertLess(row["dense_int8_bytes_div_value_mask_bytes"], 1.0)

    def test_late_scale_rne_counts_are_conserved_and_scoped(self):
        audit = self.result["late_scale_elision_audit"]
        self.assertFalse(audit["m35_rounding_rtl_admitted"])
        self.assertFalse(audit["late_scale_elision_rtl_admitted"])
        self.assertFalse(audit["cycle_reduction_admitted"])
        for row in audit["aggregate_by_layer"].values():
            self.assertEqual(row["rne_exact_bypass"] + row["rne_changed"],
                             row["values"])
            self.assertGreater(row["rne_exact_bypass_fraction_all"], 0.9999)

    def test_forbidden_claims_remain_closed(self):
        admission = self.result["admission"]
        forbidden = (
            "dynamic_no_running_batchnorm_admitted",
            "full_network_accuracy_admitted",
            "real_m40_address_cycle_schedule_admitted",
            "system_speedup_admitted",
            "integrated_rtl_or_synopsys_admitted",
            "ppa_power_energy_admitted",
            "date_best_paper_readiness_admitted",
        )
        for key in forbidden:
            self.assertFalse(admission[key], key)

    def test_duplicate_keys_and_nonstandard_numbers_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                M41.strict_json(duplicate)
            for index, token in enumerate(("NaN", "Infinity", "-Infinity")):
                path = root / "bad_{}.json".format(index)
                path.write_text('{"a":' + token + '}', encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "non-standard JSON"):
                    M41.strict_json(path)

    def test_bool_integer_and_csv_nan_rejected(self):
        with self.assertRaisesRegex(ValueError, "exact integer"):
            M41.exact_int(True, "attack")
        with self.assertRaisesRegex(ValueError, "NaN/Infinity"):
            M41.parse_csv_float({"x": "NaN"}, "x")

    def test_path_escape_rejected(self):
        for attack in ("../escape", "/tmp/escape"):
            with self.assertRaisesRegex(ValueError, "path"):
                M41.resolve_under_root(attack)

    def test_reserved_negative_128_rejected(self):
        layer = copy.deepcopy(self.result["layers"][0])
        payload = bytearray(768 * 768 * 3 * 3)
        payload[19] = 0x80
        with self.assertRaisesRegex(ValueError, "reserved -128"):
            M41.validate_weight_payload(bytes(payload), layer)

    def test_nan_scale_payload_rejected(self):
        layer = copy.deepcopy(self.result["layers"][0])
        scales = struct.pack("<f", float("nan")) + b"\x00" * (767 * 4)
        fixed = b"\x00" * (768 * 4)
        with self.assertRaisesRegex(ValueError, "NaN/Infinity"):
            M41.validate_scale_payloads(scales, fixed, layer)

    def test_pin_sha_and_byte_mutations_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for field, value in (("sha256", "0" * 64), ("bytes", 1)):
                forged = copy.deepcopy(self.pin)
                forged["files"][0][field] = value
                path = root / "pin_{}.json".format(field)
                path.write_text(json.dumps(forged), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "pinned"):
                    M41.validate(path)

    def test_exporter_rejects_existing_output_directory(self):
        exporter_path = next(
            M41.resolve_under_root(row["path"]) for row in self.pin["files"]
            if row["role"] == "exporter")
        source = exporter_path.read_text(encoding="utf-8")
        self.assertIn('require(not output_dir.exists(), "refusing to overwrite output directory")',
                      source)
        self.assertIn('require(not path.exists(), "refusing to overwrite:', source)
        process = subprocess.run(
            ["/opt/anaconda3/envs/pytorch310/bin/python", str(exporter_path),
             "--output-dir", str(self.result_path.parent)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, cwd=str(M41.ROOT))
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("refusing to overwrite output directory", process.stdout)


if __name__ == "__main__":
    unittest.main()
