#!/usr/bin/env python3
"""Directed tests for the compact M1544 capture validator."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zlib


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "hw_autoresearch_nts07/system_handoff/scripts/"
    "validate_m1544_ep34_sparse_capture_handoff.py")
M1458 = ROOT / (
    "hw_autoresearch_nts07/results/"
    "m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831/manifest.json")
SPEC = importlib.util.spec_from_file_location("m1544_validator", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def dump_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dump_zlib_jsonl(path, rows):
    payload = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows)
    path.write_bytes(zlib.compress(payload, 9))


def signed_codes_hex(codes, width_bits=8):
    width = width_bits // 8
    return b"".join(int(code).to_bytes(width, "little", signed=True) for code in codes).hex()


def sample_order_value():
    source = json.loads(M1458.read_text(encoding="utf-8"))
    rows = []
    for item in source["cohort"]["samples"]:
        rows.append({key: item[key] for key in (
            "global_sample_id", "sequence", "sequence_sample_id", "sample_key", "sha256")})
    return {
        "schema": M.SAMPLE_SCHEMA,
        "identity": {
            "checkpoint_sha256": M.CHECKPOINT_SHA256,
            "m1458_manifest_sha256": M.M1458_MANIFEST_SHA256,
            "m1458_inner_manifest_sha256": M.M1458_INNER_SHA256,
            "m1458_outer_file_sha256": M.M1458_OUTER_SHA256,
        },
        "samples": rows,
    }


def codebook(authority):
    return {
        "width_bits": 8, "signed": True, "zero_point": 0, "unit_code": 1,
        "scale_numerator": 1, "scale_denominator": 1, "rounding": "nearest_even",
        "saturation": "signed_clamp", "authority": authority,
        "diagnostic_capture_only": True, "hardware_quant_authority": False,
    }


def make_layer(layer_id, target, module, operator_order, s1=False):
    blocks = []
    base = layer_id * 4096
    for output_tile in range(1):
        for source_group in range(1):
            address = base + (output_tile + source_group) * 16
            blocks.append({
                "source_group_id": source_group, "output_tile_id": output_tile,
                "address": address, "bank_key": (address // 16) % 4,
                "row_buffer_key": "%d:%d:%d" % (layer_id, output_tile, source_group),
            })
    return {
        "layer_id": layer_id, "target": target, "module_name": module,
        "operator_order": operator_order, "input_channels": 4, "output_channels": 4,
        "group_width": 4, "output_tile_width": 4,
        "codebook": codebook(
            "diagnostic_fixed_point_codeword" if s1 else "captured_binary_codeword"),
        "weight_layout": {
            "base_address": base, "bank_count": 4, "row_bytes": 16,
            "address_formula":
            "base_address+(output_tile_id*source_group_count+source_group_id)*row_bytes",
            "bank_formula": "(address//row_bytes)%bank_count",
            "row_buffer_baseline": "ordinary_same_capacity_LRU_weight_row_buffer",
            "blocks": blocks,
        },
        "s1_eligible": s1,
        "s1_magnitude_bin_edges_abs_code": [0, 1, 2, 4] if s1 else [],
    }


def gates():
    return {
        "S1": {
            "metadata_plus_beta_over_saved_weight_bytes_veto": 0.25,
            "beta_port_cycle_regression_veto": 0.05,
            "mean_delta_aee_max": 0.02,
            "per_sequence_delta_aee_max": 0.03,
        },
        "S2": {
            "total_metadata_over_weight_bytes_max": 0.02,
            "metadata_reduction_vs_g11_min": 8.0,
            "dynamic_same_block_keep_drop_witness_required": True,
        },
        "TSBG": {
            "aggregate_fc1_fc2_cycle_speedup_min": 1.15,
            "every_sequence_cycle_speedup_min": 1.05,
            "energy_branch_cycle_regression_max": 0.05,
            "energy_branch_weight_byte_reduction_min": 0.30,
            "energy_branch_memory_energy_reduction_min": 0.20,
        },
    }


def build_fixture(root):
    samples = sample_order_value()
    dump_json(root / "sample_order.json", samples)
    layer_rows = [
        make_layer(0, "FC1", "fixture.mlp.fc1", 10),
        make_layer(1, "FC2", "fixture.mlp.fc2", 11),
        make_layer(2, "PATCH", "fixture.patch", 12, True),
    ]
    dump_json(root / "layers.json", {
        "schema": M.LAYER_SCHEMA,
        "status": "STATIC_WEIGHT_LAYOUT_COMPLETE__NO_CYCLE_OR_ENERGY_CLAIM",
        "layers": layer_rows,
    })
    token_rows = []
    global_order = 0
    for sample in samples["samples"]:
        for layer in layer_rows:
            token_rows.append({
                "schema": M.TOKEN_SCHEMA, "global_order": global_order,
                "sample_global_id": sample["global_sample_id"],
                "sequence": sample["sequence"],
                "sequence_sample_id": sample["sequence_sample_id"],
                "sample_key": sample["sample_key"], "operator_order": layer["operator_order"],
                "layer_id": layer["layer_id"], "token_order": 0, "window_order": None,
                "spatial_y": 0, "spatial_x": 0,
                "groups": [{
                    "source_group_id": 0, "valid_channels": 4,
                    "support_hex": "05", "sign_hex": "04", "nonunit_hex": "04",
                    "nonzero_codes_le_hex": signed_codes_hex([1, -2]),
                }],
            })
            global_order += 1
    dump_zlib_jsonl(root / "token_source_groups.jsonl.zlib", token_rows)
    s1_rows = []
    for sample in samples["samples"]:
        s1_rows.append({
            "schema": M.S1_SCHEMA, "sample_global_id": sample["global_sample_id"],
            "layer_id": 2, "output_tile_id": 0,
            "count_by_magnitude_bin": [0, 1, 1],
            "beta_abs_code_debt_by_magnitude_bin": [0, 3, 8],
            "nonzero_source_count": 2, "beta_rounding": "ceil_upper_bound",
        })
    dump_zlib_jsonl(root / "s1_histogram_debt.jsonl.zlib", s1_rows)
    dump_json(root / "capture_manifest.json", {
        "schema": M.MANIFEST_SCHEMA, "status": M.MANIFEST_STATUS,
        "identity": {
            "checkpoint_sha256": M.CHECKPOINT_SHA256,
            "m1458_manifest_sha256": M.M1458_MANIFEST_SHA256,
            "m1458_inner_manifest_sha256": M.M1458_INNER_SHA256,
            "m1458_outer_file_sha256": M.M1458_OUTER_SHA256,
            "m1458_sample_order_sha256": M.M1458_ORDER_SHA256,
            "m1540_review_sha256": M.M1540_REVIEW_SHA256,
            "m1541_review_sha256": M.M1541_REVIEW_SHA256,
        },
        "population": {"samples": 40, "layers": 3, "token_records": 120,
                       "s1_histogram_rows": 40},
        "files": {"sample_order": "sample_order.json", "layers": "layers.json",
                  "tokens": "token_source_groups.jsonl.zlib",
                  "s1": "s1_histogram_debt.jsonl.zlib"},
        "encoding": {
            "token_container": "canonical_jsonl_zlib_level9",
            "zero_groups": "omitted_from_groups_but_token_record_retained",
            "support_sign_nonunit": "little_endian_channel_bitsets",
            "codes": "signed_little_endian_nonzero_only", "full_fp_tensor_saved": False,
            "static_weight_mapping_repeated_per_token": False,
        },
        "coverage": {"all_40_samples": True, "all_layers_each_sample": True,
                     "targets": ["FC1", "FC2", "PATCH"], "token_records": 120},
        "admission_gates": gates(),
        "claim_boundary": {
            "capture_only": True, "static_opportunity": False, "cycles": False,
            "speedup": False, "traffic": False, "energy": False, "aee": False,
            "rtl": False, "paper_headline": False,
            "hardware_quantization_authority": False, "model_bit_exact": False,
            "tsbg_exact_scope": "captured_codeword_and_contributor_only",
            "formal_int8_bridge_required": True,
        },
    })
    (root / "RUN_COMPLETE.txt").write_text(
        "M1544_EP34_SPARSE_CAPTURE_COMPLETE__NO_HARDWARE_CLAIM\n", encoding="ascii")
    reseal(root)


def reseal(root):
    names = sorted(M.CAPTURE_MEMBERS - {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    lines = []
    for name in names:
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        lines.append(digest + "  " + name)
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="ascii")
    outer = hashlib.sha256((root / "SHA256SUMS").read_bytes()).hexdigest()
    (root / "SHA256SUMS.seal.sha256").write_text(
        outer + "  SHA256SUMS\n", encoding="ascii")


class M1544ValidatorTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "capture"
        self.root.mkdir()
        build_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def assert_rejected(self, mutation):
        mutation(self.root)
        reseal(self.root)
        with self.assertRaises(M.M1544Error):
            M.validate_capture(self.root)

    def read_rows(self, name):
        payload = zlib.decompress((self.root / name).read_bytes())
        return [json.loads(line.decode("utf-8")) for line in payload.splitlines()]

    def test_valid_fixture(self):
        result = M.validate_capture(self.root)
        self.assertEqual(result["samples"], 40)
        self.assertEqual(result["layers"], 3)
        self.assertEqual(result["token_records"], 120)
        self.assertFalse(result["cycles_admitted"])

    def test_checkpoint_substitution_rejected(self):
        def mutate(root):
            value = json.loads((root / "capture_manifest.json").read_text())
            value["identity"]["checkpoint_sha256"] = "0" * 64
            dump_json(root / "capture_manifest.json", value)
        self.assert_rejected(mutate)

    def test_sample_order_swap_rejected(self):
        def mutate(root):
            value = json.loads((root / "sample_order.json").read_text())
            value["samples"][0]["sample_key"] = "wrong.npy"
            dump_json(root / "sample_order.json", value)
        self.assert_rejected(mutate)

    def test_sign_outside_support_rejected(self):
        def mutate(root):
            rows = self.read_rows("token_source_groups.jsonl.zlib")
            rows[0]["groups"][0]["sign_hex"] = "06"
            dump_zlib_jsonl(root / "token_source_groups.jsonl.zlib", rows)
        self.assert_rejected(mutate)

    def test_code_population_rejected(self):
        def mutate(root):
            rows = self.read_rows("token_source_groups.jsonl.zlib")
            rows[0]["groups"][0]["nonzero_codes_le_hex"] = signed_codes_hex([1])
            dump_zlib_jsonl(root / "token_source_groups.jsonl.zlib", rows)
        self.assert_rejected(mutate)

    def test_zero_group_rejected(self):
        def mutate(root):
            rows = self.read_rows("token_source_groups.jsonl.zlib")
            rows[0]["groups"][0].update({
                "support_hex": "00", "sign_hex": "00", "nonunit_hex": "00",
                "nonzero_codes_le_hex": "",
            })
            dump_zlib_jsonl(root / "token_source_groups.jsonl.zlib", rows)
        self.assert_rejected(mutate)

    def test_weight_address_rejected(self):
        def mutate(root):
            value = json.loads((root / "layers.json").read_text())
            value["layers"][0]["weight_layout"]["blocks"][0]["address"] += 16
            dump_json(root / "layers.json", value)
        self.assert_rejected(mutate)

    def test_s1_debt_population_rejected(self):
        def mutate(root):
            rows = self.read_rows("s1_histogram_debt.jsonl.zlib")
            rows[0]["nonzero_source_count"] += 1
            dump_zlib_jsonl(root / "s1_histogram_debt.jsonl.zlib", rows)
        self.assert_rejected(mutate)

    def test_m1541_s2_metadata_cap_rejected(self):
        def mutate(root):
            value = json.loads((root / "capture_manifest.json").read_text())
            value["admission_gates"]["S2"]["total_metadata_over_weight_bytes_max"] = 0.03
            dump_json(root / "capture_manifest.json", value)
        self.assert_rejected(mutate)

    def test_m1541_tsbg_sequence_floor_rejected(self):
        def mutate(root):
            value = json.loads((root / "capture_manifest.json").read_text())
            value["admission_gates"]["TSBG"]["every_sequence_cycle_speedup_min"] = 1.0
            dump_json(root / "capture_manifest.json", value)
        self.assert_rejected(mutate)

    def test_hardware_quant_authority_injection_rejected(self):
        def mutate(root):
            value = json.loads((root / "layers.json").read_text())
            value["layers"][0]["codebook"]["hardware_quant_authority"] = True
            value["layers"][0]["codebook"]["diagnostic_capture_only"] = False
            dump_json(root / "layers.json", value)
        self.assert_rejected(mutate)

    def test_extra_full_tensor_rejected(self):
        (self.root / "full_tensor.fp32").write_bytes(b"forbidden")
        with self.assertRaises(M.M1544Error):
            M.validate_capture(self.root)


if __name__ == "__main__":
    unittest.main()
