#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Local source/small-model tests; never reads the real M1458 result."""
from __future__ import annotations

import copy
import importlib.util
import numpy as np
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/hammer_m1501_m1458_motion_ep34_live93_capture_result_safe_audit_source.py"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


M = load("test_m1501_source", SOURCE)


class Payload(dict):
    @property
    def files(self):
        return list(self)


def projection_payload():
    row = {"windows_captured": 1, "heads": 2, "spatial_tokens": 1,
           "temporal_tokens": 2, "lanes": 2}
    exponent = np.asarray([-4, -3, -2, -1], dtype=np.int16)
    code = np.asarray([
        [100, -64, 1, 0], [100, -50, 0, 2],
        [100, 0, -20, 7], [100, -1, 4, 3]], dtype=np.int8)
    scale = np.exp2(exponent.astype(np.float32))
    weight = code.astype(np.float32) * scale[:, None]
    bias = np.asarray([0.5, -0.25, 1.0, -2.0], dtype=np.float32)
    bias_acc = np.rint(bias / (scale / np.float32(128.0))).astype(np.int64)
    data = Payload(
        q_shape=np.asarray([2, 1, 2, 1, 2], dtype=np.int32),
        k_shape=np.asarray([2, 1, 2, 1, 2], dtype=np.int32),
        q_bits_packed=np.asarray([0x55], dtype=np.uint8),
        k_bits_packed=np.asarray([0xAA], dtype=np.uint8),
        gate_q17=np.zeros((1, 2, 2), dtype=np.uint16),
        projection_weight_float32=weight,
        projection_weight_int8=code,
        projection_weight_scale_exp2=exponent,
        projection_bias_float32=bias,
        projection_bias_acc_int64=bias_acc,
    )
    return data, row


def manifest():
    checkpoint = "/root/private_data/work/checkpoint_epoch34.pth"
    return {
        "identity": {
            "selection": {"selected": {
                "checkpoint": {"absolute_path": checkpoint}}},
            "checkpoint_load_audit": {
                "checkpoint": checkpoint,
                "checkpoint_overlay_keys": 210,
                "model_overlay_keys": 210,
                "missing_count": 0,
                "unexpected_count": 0,
                "overlay_missing_count": 0,
                "overlay_unexpected_count": 0,
                "missing_sample": [],
                "unexpected_sample": [],
                "remap": "v1",
            },
        },
        "sentinel": {"preserved": True},
    }


class Tests(unittest.TestCase):
    def reject(self, value):
        with self.assertRaises(M.M1501Error):
            M.validate_checkpoint_load_audit(value)

    def test_01_exact_safe_superset_passes(self):
        M.validate_checkpoint_load_audit(manifest())

    def test_02_every_nonzero_mismatch_rejected(self):
        for key in M.ZERO_FIELDS:
            value = manifest()
            value["identity"]["checkpoint_load_audit"][key] = 1
            with self.subTest(key=key):
                self.reject(value)

    def test_03_every_missing_field_rejected(self):
        for key in M.AUDIT_KEYS:
            value = manifest()
            value["identity"]["checkpoint_load_audit"].pop(key)
            with self.subTest(key=key):
                self.reject(value)

    def test_04_unknown_extra_field_rejected(self):
        value = manifest()
        value["identity"]["checkpoint_load_audit"]["unknown"] = 0
        self.reject(value)

    def test_05_each_overlay_key_count_mismatch_rejected(self):
        for key in M.OVERLAY_FIELDS:
            value = manifest()
            value["identity"]["checkpoint_load_audit"][key] = 209
            with self.subTest(key=key):
                self.reject(value)

    def test_06_nonempty_samples_rejected(self):
        for key in ("missing_sample", "unexpected_sample"):
            value = manifest()
            value["identity"]["checkpoint_load_audit"][key] = ["bad.key"]
            with self.subTest(key=key):
                self.reject(value)

    def test_07_bool_counts_rejected(self):
        for key in M.ZERO_FIELDS + M.OVERLAY_FIELDS:
            value = manifest()
            value["identity"]["checkpoint_load_audit"][key] = (
                False if key in M.ZERO_FIELDS else True)
            with self.subTest(key=key):
                self.reject(value)

    def test_08_path_and_remap_drift_rejected(self):
        value = manifest()
        value["identity"]["checkpoint_load_audit"]["checkpoint"] += ".wrong"
        self.reject(value)
        value = manifest()
        value["identity"]["checkpoint_load_audit"]["remap"] = "v2"
        self.reject(value)

    def test_09_delegate_receives_only_two_field_normalization(self):
        value = manifest()
        with mock.patch.object(M, "FROZEN_VALIDATE_MANIFEST") as frozen:
            M.validate_manifest(value)
        frozen.assert_called_once()
        delegated = frozen.call_args.args[0]
        self.assertEqual(delegated["identity"]["checkpoint_load_audit"],
                         {"missing_count": 0, "unexpected_count": 0})
        self.assertEqual(delegated["sentinel"], value["sentinel"])
        self.assertEqual(value, manifest())

    def test_10_validate_result_restores_frozen_delegate(self):
        original = M.M1455.validate_manifest
        original_attention = (
            M.M1455.M1401.M1338.validate_attention_exact_archive)
        def fake_result(root):
            self.assertIs(M.M1455.validate_manifest, M.validate_manifest)
            self.assertIs(
                M.M1455.M1401.M1338.validate_attention_exact_archive,
                M.validate_attention_exact_archive)
            return {"status": "PASS_M1455_M1434_EP34_LIVE93_CAPTURE_RESULT",
                    "claim_boundary": {"cycles": False, "speedup": False}}
        with mock.patch.object(M.M1455, "validate_result", side_effect=fake_result):
            output = M.validate_result(Path("/not/read"))
        self.assertIs(M.M1455.validate_manifest, original)
        self.assertIs(
            M.M1455.M1401.M1338.validate_attention_exact_archive,
            original_attention)
        self.assertEqual(output["status"],
                         "PASS_M1501_M1458_EP34_LIVE93_CAPTURE_RESULT")
        self.assertTrue(output["audit_adapter"][
            "all_other_validation_delegated_to_exact_m1455"])

    def test_11_source_has_no_remote_capture_gpu_or_eda_action(self):
        text = SOURCE.read_text()
        for token in ("subprocess", "paramiko", "torch.cuda", "os.kill",
                      "ssh ", "vcs", "dc_shell", "pt_shell"):
            self.assertNotIn(token, text)
        self.assertNotIn('add_argument("--run"', text)

    def test_12_exact_predecessor_and_authority_source_bindings(self):
        self.assertEqual(M.sha256(M.M1455_SOURCE), M.M1455_SHA256)
        self.assertEqual(M.sha256(M.M1458_SOURCE), M.M1458_SHA256)
        self.assertEqual(M.sha256(M.M1489_SOURCE), M.M1489_SHA256)

    def test_13_exact_enriched_projection_payload_passes(self):
        data, row = projection_payload()
        M.validate_projection_arrays(data, row)

    def test_14_projection_unknown_and_missing_keys_rejected(self):
        for attack in ("unknown", "missing"):
            data, row = projection_payload()
            if attack == "unknown":
                data["unknown_projection"] = np.asarray([0], dtype=np.int8)
            else:
                data.pop("projection_bias_acc_int64")
            with self.subTest(attack=attack), self.assertRaises(M.M1501Error):
                M.validate_projection_arrays(data, row)

    def test_15_projection_wrong_dtype_rejected(self):
        for key, dtype in (
            ("projection_weight_float32", np.float64),
            ("projection_weight_int8", np.int16),
            ("projection_weight_scale_exp2", np.int32),
            ("projection_bias_float32", np.float64),
            ("projection_bias_acc_int64", np.int32),
        ):
            data, row = projection_payload()
            data[key] = data[key].astype(dtype)
            with self.subTest(key=key), self.assertRaises(M.M1501Error):
                M.validate_projection_arrays(data, row)

    def test_16_projection_nonfinite_rejected(self):
        for key in ("projection_weight_float32",
                    "projection_bias_float32"):
            data, row = projection_payload()
            data[key] = data[key].copy()
            data[key].flat[0] = np.nan
            with self.subTest(key=key), self.assertRaises(M.M1501Error):
                M.validate_projection_arrays(data, row)

    def test_17_projection_quantization_relation_rejected(self):
        data, row = projection_payload()
        data["projection_weight_int8"] = (
            data["projection_weight_int8"].copy())
        data["projection_weight_int8"][0, 0] -= np.int8(1)
        with self.assertRaises(M.M1501Error):
            M.validate_projection_arrays(data, row)
        data, row = projection_payload()
        data["projection_bias_acc_int64"] = (
            data["projection_bias_acc_int64"].copy())
        data["projection_bias_acc_int64"][0] += np.int64(1)
        with self.assertRaises(M.M1501Error):
            M.validate_projection_arrays(data, row)


if __name__ == "__main__":
    unittest.main()
