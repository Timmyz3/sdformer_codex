#!/usr/bin/env python3
"""Adversarial tests for the canonical M41-r2 integer-oracle release."""

import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zlib


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / (
    "hw_autoresearch_nts07/system_simulator/scripts/"
    "validate_m41_r2_integer_oracle.py")
SPEC = importlib.util.spec_from_file_location("m41_r2_validator", str(SCRIPT))
M41 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M41)


class M41R2IntegerOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = M41.strict_json(M41.CONTRACT_PATH)
        cls.result = M41.strict_json(M41.RESULT_PATH)
        manifest_path = M41.resolve_under_root(
            cls.contract["identity"]["frozen_inputs"][
                "m40_source_manifest"]["path"])
        manifest = M41.strict_json(manifest_path)
        cls.manifest_records = sorted(
            manifest["records"],
            key=lambda row: (row["operator_index"], row["sample_id"]))
        cls.raw_bitmap = zlib.decompress(M41.BITMAP_PATH.read_bytes())

    def test_01_full_canonical_recursive_validation(self):
        summary = M41.canonical_validate()
        self.assertEqual(summary["status"],
                         "PASS_M41_R2_CANONICAL_INTEGER_ORACLE_AND_EXCEPTION_TRACE")
        self.assertEqual(summary["accumulators_anchored"], 92160000)
        self.assertEqual(summary["observed_exception_counts_by_operator"],
                         [0, 0, 0, 1348])

    def test_02_contract_does_not_predeclare_exception_conclusion(self):
        oracle = self.contract["integer_oracle"]
        self.assertIsNone(oracle["predeclared_exception_counts"])
        self.assertEqual(oracle["required_population"], 92160000)

    def test_03_backend_identity_is_complete_and_explicit(self):
        tool = self.result["toolchain"]
        self.assertEqual(tool["authoritative_operator"], "aten::_int_mm")
        self.assertEqual(tool["authoritative_input_dtypes"],
                         ["torch.int8", "torch.int8"])
        self.assertEqual(tool["authoritative_output_dtype"], "torch.int32")
        self.assertTrue(tool["deterministic_algorithms"])
        self.assertTrue(tool["cudnn_deterministic"])
        self.assertFalse(tool["cudnn_benchmark"])
        self.assertIn("cuda_matmul_allow_tf32", tool)
        self.assertIn("cudnn_allow_tf32", tool)
        self.assertTrue(tool["cuda_runtime_compiled_version"])
        self.assertGreater(tool["cudnn_runtime_version"], 0)

    def test_04_all_quantized_payloads_and_checkpoint_are_directly_pinned(self):
        frozen = self.contract["identity"]["frozen_inputs"]
        self.assertIn("checkpoint", frozen)
        for operator in range(4):
            for suffix in ("weight_s8", "scale_f32", "scale_uq31", "acc_init"):
                receipt = frozen["o{}_{}".format(operator, suffix)]
                self.assertEqual(len(receipt["sha256"]), 64)
                self.assertEqual(M41.sha256(M41.resolve_under_root(receipt["path"])),
                                 receipt["sha256"])

    def test_05_all_weight_payloads_reject_reserved_negative_128(self):
        frozen = self.contract["identity"]["frozen_inputs"]
        for operator in range(4):
            path = M41.resolve_under_root(
                frozen["o{}_weight_s8".format(operator)]["path"])
            self.assertNotIn(b"\x80", path.read_bytes())

    def test_06_duplicate_json_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"a":1,"a":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                M41.strict_json(path)

    def test_07_nonfinite_json_number_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nan.json"
            path.write_text('{"a":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite JSON"):
                M41.strict_json(path)

    def test_08_bool_is_not_accepted_as_an_integer(self):
        with self.assertRaisesRegex(ValueError, "exact integer"):
            M41.exact_int(True, "attack")

    def test_09_absolute_and_dotdot_paths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "relative"):
            M41.resolve_under_root("/tmp/escape")
        with self.assertRaisesRegex(ValueError, "dot-dot"):
            M41.resolve_under_root("hw_autoresearch_nts07/../escape")

    def test_10_canonical_bitmap_sha_mutation_is_rejected(self):
        payload = bytearray(M41.BITMAP_PATH.read_bytes())
        payload[len(payload) // 2] ^= 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutated.zlib"
            path.write_bytes(bytes(payload))
            with self.assertRaisesRegex(ValueError, "canonical bitmap SHA"):
                M41.canonical_validate(bitmap_path=path)

    def test_11_nonzero_oracle_mismatch_is_rejected_before_claims(self):
        mutated = copy.deepcopy(self.result)
        mutated["population"]["integer_float_accumulator_mismatches"] = 1
        with self.assertRaisesRegex(ValueError, "result population"):
            M41.validate_result_semantics(
                mutated, self.contract, self.manifest_records, self.raw_bitmap)

    def test_12_backend_policy_mutation_is_rejected(self):
        mutated = copy.deepcopy(self.result)
        mutated["toolchain"]["cudnn_deterministic"] = False
        with self.assertRaisesRegex(ValueError, "toolchain arithmetic"):
            M41.validate_result_semantics(
                mutated, self.contract, self.manifest_records, self.raw_bitmap)

    def test_13_stream_histogram_and_fifo_math_are_recomputed(self):
        payload = bytes(bytearray([0x03] + [0] * 11 + [0x01] + [0] * 11))
        stream = M41.bitmap_stream(payload)
        self.assertEqual(stream["vectors"], 2)
        self.assertEqual(stream["exceptions"], 3)
        self.assertEqual(stream["arrival_histogram"], {"1": 1, "2": 1})
        self.assertEqual(stream["peak_exceptions_in_one_vector"], 2)
        self.assertEqual(stream["longest_consecutive_scalar_exception_run"], 2)
        self.assertEqual(
            stream["fixed_scalar_service_fifo_requirements"]["1"][
                "minimum_fifo_entries_arrivals_before_service"], 2)

    def test_14_exception_sidecar_claims_remain_closed(self):
        sidecar = self.result["exception_sidecar_trace_thresholds"]
        self.assertFalse(sidecar["rtl_admitted"])
        self.assertFalse(sidecar["cycle_reduction_admitted"])
        self.assertFalse(sidecar["ppa_or_power_admitted"])
        self.assertFalse(sidecar["system_speedup_admitted"])
        admission = self.result["admission"]
        self.assertFalse(admission["exception_sidecar_rtl_admitted"])
        self.assertFalse(admission["integrated_cycle_reduction_admitted"])
        self.assertFalse(admission["synopsys_ppa_power_energy_admitted"])


if __name__ == "__main__":
    unittest.main()
