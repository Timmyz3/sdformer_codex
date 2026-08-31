#!/opt/anaconda3/envs/pytorch310/bin/python3.10
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zlib

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/hammer_m1338_m1327_final_ep34_capture_result_source.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


M = load("test_m1338_source", SOURCE)


class RawFixture:
    def __init__(self, values: np.ndarray | None = None):
        self.temp = tempfile.TemporaryDirectory(prefix="m1338_raw_")
        self.root = Path(self.temp.name) / "result"
        (self.root / "payloads").mkdir(parents=True)
        self.values = (np.array([0.0, 1.0, -2.0, np.nan, np.inf, -np.inf],
                                dtype="<f4") if values is None else
                       np.asarray(values, dtype="<f4"))
        self.raw = self.values.tobytes(order="C")
        self.compressed_rel = "payloads/scalar.fp32.zlib"
        self.support_rel = "payloads/scalar.support_sign.le.bitpack"
        (self.root / self.compressed_rel).write_bytes(zlib.compress(self.raw))
        positive = np.packbits(self.values > 0, bitorder="little").tobytes()
        negative = np.packbits(self.values < 0, bitorder="little").tobytes()
        (self.root / self.support_rel).write_bytes(positive + negative)
        elements = int(self.values.size)
        self.row = {
            "input": {"shape": [elements], "stride": [1], "dtype": "torch.float32",
                      "elements": elements, "bytes": elements * 4,
                      "active": int(np.count_nonzero(self.values != 0)),
                      "positive": int(np.count_nonzero(self.values > 0)),
                      "negative": int(np.count_nonzero(self.values < 0)),
                      "nonfinite": int(np.count_nonzero(~np.isfinite(self.values)))},
            "payload": {"retained": True,
                        "raw_fp32_sha256": hashlib.sha256(self.raw).hexdigest(),
                        "compressed_fp32": self.compressed_rel,
                        "compressed_sha256": M.sha256(self.root / self.compressed_rel),
                        "support_sign": self.support_rel,
                        "support_sign_sha256": M.sha256(self.root / self.support_rel),
                        "positive_plane_bytes": (elements + 7) // 8,
                        "negative_plane_bytes": (elements + 7) // 8}}
        self.refresh_seal()

    def refresh_seal(self):
        payload = self.row["payload"]
        payload["compressed_sha256"] = M.sha256(self.root / self.compressed_rel)
        payload["support_sign_sha256"] = M.sha256(self.root / self.support_rel)
        self.seal_rows = {self.compressed_rel: payload["compressed_sha256"],
                          self.support_rel: payload["support_sign_sha256"]}

    def validate(self):
        M.validate_one_retained_payload(self.root, self.seal_rows, self.row)

    def close(self):
        self.temp.cleanup()


class AttentionFixture:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m1338_attention_")
        self.path = Path(self.temp.name) / "one.npz"
        self.row = {"windows_captured": 1, "heads": 1, "spatial_tokens": 1,
                    "temporal_tokens": 2, "lanes": 7}
        self.values = {
            "q_shape": np.array([2, 1, 1, 1, 7], dtype=np.int32),
            "k_shape": np.array([2, 1, 1, 1, 7], dtype=np.int32),
            "q_bits_packed": np.array([0x55, 0x15], dtype=np.uint8),
            "k_bits_packed": np.array([0x2A, 0x2A], dtype=np.uint8),
            "gate_q17": np.array([[[0, 256]]], dtype=np.uint16),
        }
        self.write()

    def write(self):
        np.savez_compressed(self.path, **self.values)

    def validate(self):
        M.validate_attention_npz(self.path, self.row)

    def close(self):
        self.temp.cleanup()


class Tests(unittest.TestCase):
    def test_01_positive_raw_content_derived_contract(self):
        fixture = RawFixture()
        try:
            fixture.validate()
        finally:
            fixture.close()

    def test_02_positive_attention_exact_archive_and_zero_tail(self):
        fixture = AttentionFixture()
        try:
            fixture.validate()
        finally:
            fixture.close()

    def test_03_m1336_failure_and_source_policy_are_exact(self):
        review = M.verify_failed_predecessor()
        self.assertEqual(review["accepted_attack_count"], 9)
        policy = M.validate_source_policy()
        self.assertFalse(policy["production_authorized"])

    def test_04_raw_length_not_equal_input_bytes_rejected(self):
        fixture = RawFixture(np.array([0.0], dtype="<f4"))
        try:
            raw = b"abc"
            (fixture.root / fixture.compressed_rel).write_bytes(zlib.compress(raw))
            fixture.row["payload"]["raw_fp32_sha256"] = hashlib.sha256(raw).hexdigest()
            fixture.refresh_seal()
            with self.assertRaisesRegex(M.M1338Error, "length|extent"):
                fixture.validate()
        finally:
            fixture.close()

    def test_05_plane_extent_not_derived_rejected(self):
        fixture = RawFixture(np.array([0.0], dtype="<f4"))
        try:
            fixture.row["payload"]["positive_plane_bytes"] = 2
            fixture.row["payload"]["negative_plane_bytes"] = 2
            (fixture.root / fixture.support_rel).write_bytes(b"\0\0\0\0")
            fixture.refresh_seal()
            with self.assertRaisesRegex(M.M1338Error, "derived"):
                fixture.validate()
        finally:
            fixture.close()

    def test_06_zlib_trailing_garbage_rejected(self):
        fixture = RawFixture(np.array([0.0], dtype="<f4"))
        try:
            (fixture.root / fixture.compressed_rel).write_bytes(
                zlib.compress(fixture.raw) + b"TRAILING")
            fixture.refresh_seal()
            with self.assertRaisesRegex(M.M1338Error, "trailing"):
                fixture.validate()
        finally:
            fixture.close()

    def test_07_raw_statistics_disagreement_rejected(self):
        fixture = RawFixture(np.array([1.0], dtype="<f4"))
        try:
            fixture.row["input"].update({"active": 0, "positive": 0,
                                         "negative": 0, "nonfinite": 0})
            with self.assertRaisesRegex(M.M1338Error, "statistics"):
                fixture.validate()
        finally:
            fixture.close()

    def test_08_support_sign_disagreement_rejected(self):
        fixture = RawFixture(np.array([1.0], dtype="<f4"))
        try:
            (fixture.root / fixture.support_rel).write_bytes(b"\x00\x01")
            fixture.refresh_seal()
            with self.assertRaisesRegex(M.M1338Error, "support signs"):
                fixture.validate()
        finally:
            fixture.close()

    def test_09_support_padding_rejected(self):
        fixture = RawFixture(np.array([0.0], dtype="<f4"))
        try:
            (fixture.root / fixture.support_rel).write_bytes(b"\xfe\x00")
            fixture.refresh_seal()
            with self.assertRaisesRegex(M.M1338Error, "padding"):
                fixture.validate()
        finally:
            fixture.close()

    def test_10_fp32_payload_with_float16_label_rejected(self):
        fixture = RawFixture(np.array([0.0], dtype="<f4"))
        try:
            fixture.row["input"]["dtype"] = "torch.float16"
            with self.assertRaisesRegex(M.M1338Error, "dtype"):
                fixture.validate()
        finally:
            fixture.close()

    def test_11_attention_invented_member_rejected(self):
        fixture = AttentionFixture()
        try:
            fixture.values["invented_payload"] = np.array([1], dtype=np.uint8)
            fixture.write()
            with self.assertRaisesRegex(M.M1338Error, "member set"):
                fixture.validate()
        finally:
            fixture.close()

    def test_12_attention_nonzero_packbits_tail_rejected(self):
        fixture = AttentionFixture()
        try:
            fixture.values["q_bits_packed"][-1] |= np.uint8(0xC0)
            fixture.write()
            with self.assertRaisesRegex(M.M1338Error, "padding"):
                fixture.validate()
        finally:
            fixture.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
