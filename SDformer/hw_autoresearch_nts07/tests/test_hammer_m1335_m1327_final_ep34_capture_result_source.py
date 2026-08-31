#!/opt/anaconda3/envs/pytorch310/bin/python3.10
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import zlib

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/hammer_m1335_m1327_final_ep34_capture_result_source.py"
OLD_TEST = ROOT / "hw_autoresearch_nts07/tests/test_hammer_m1333_m1327_final_ep34_capture_result_source.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


M = load("test_m1335_source", SOURCE)
B = load("test_m1335_old_fixture", OLD_TEST)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class StrongFixture:
    def __init__(self):
        self.old = B.BaseFixture()
        self.root = self.old.root

        ordered_path = self.root / "unified_ordered_records.jsonl"
        ordered = [json.loads(line) for line in ordered_path.read_text().splitlines()]
        for row in ordered:
            payload = row["payload"]
            if payload.get("retained") is not True:
                continue
            raw = ("m1335-raw-{}-{}".format(row["global_sample_id"],
                                             row["global_order"])).encode()
            compressed = zlib.compress(raw)
            compressed_path = self.root / payload["compressed_fp32"]
            support_path = self.root / payload["support_sign"]
            compressed_path.write_bytes(compressed)
            support = bytes([row["global_order"] & 255,
                             (row["global_order"] >> 8) & 255])
            support_path.write_bytes(support)
            payload["raw_fp32_sha256"] = hashlib.sha256(raw).hexdigest()
            payload["compressed_sha256"] = M.sha256(compressed_path)
            payload["support_sign_sha256"] = M.sha256(support_path)
            payload["positive_plane_bytes"] = 1
            payload["negative_plane_bytes"] = 1
        ordered_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                        for row in ordered), encoding="utf-8")

        operators = [{"name": row["name"], "operator": row["operator"],
                      "scope": row["scope"], "calls": 40}
                     for row in M.frozen_operator_rows()]
        write_json(self.root / "operator_runtime.json", operators)
        atlif = [{"name": row["name"], "output_mode": row["output_mode"],
                  "threshold_mode": row["threshold_mode"], "calls": 40}
                 for row in M.frozen_atlif_rows()]
        write_json(self.root / "atlif_activity.json", atlif)

        attention_path = self.root / "attention_qk/manifest.json"
        attention = json.loads(attention_path.read_text())
        for ordinal, row in enumerate(attention["records"]):
            payload = self.root / "attention_qk" / Path(row["file"]).name
            q = np.array([ordinal & 255, (ordinal * 3 + 1) & 255], dtype=np.uint8)
            k = np.array([(ordinal * 5 + 2) & 255, (ordinal * 7 + 3) & 255],
                         dtype=np.uint8)
            gate = np.array([[[ordinal % 257, (ordinal + 1) % 257]]],
                            dtype=np.uint16)
            np.savez_compressed(payload,
                                q_shape=np.array([2, 1, 1, 1, 8], dtype=np.int32),
                                k_shape=np.array([2, 1, 1, 1, 8], dtype=np.int32),
                                q_bits_packed=q, k_bits_packed=k, gate_q17=gate)
            row.update({"windows_captured": 1, "heads": 1,
                        "spatial_tokens": 1, "temporal_tokens": 2, "lanes": 8,
                        "q_active_bits": int(np.unpackbits(q, bitorder="little").sum()),
                        "k_active_bits": int(np.unpackbits(k, bitorder="little").sum()),
                        "gate_nonzero": int(np.count_nonzero(gate)),
                        "gate_min": int(gate.min()), "gate_max": int(gate.max()),
                        "sha256": M.sha256(payload)})
        write_json(attention_path, attention)
        B.seal(self.root)

    def close(self):
        self.old.close()


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = StrongFixture()

    @classmethod
    def tearDownClass(cls):
        cls.base.close()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m1335_case_")
        self.root = Path(self.temp.name) / "result"
        shutil.copytree(self.base.root, self.root)

    def tearDown(self):
        self.temp.cleanup()

    def reject(self):
        B.seal(self.root)
        with self.assertRaises(M.M1335Error):
            M.validate_result(self.root)

    def ordered(self):
        path = self.root / "unified_ordered_records.jsonl"
        return path, [json.loads(line) for line in path.read_text().splitlines()]

    def write_ordered(self, path, rows):
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                        encoding="utf-8")

    def attention(self):
        path = self.root / "attention_qk/manifest.json"
        return path, json.loads(path.read_text())

    def rewrite_npz(self, record, **changes):
        payload = self.root / "attention_qk" / Path(record["file"]).name
        with np.load(payload, allow_pickle=False) as data:
            values = {name: data[name] for name in data.files}
        values.update(changes)
        np.savez_compressed(payload, **values)
        record["sha256"] = M.sha256(payload)

    def test_01_positive_full_fixture(self):
        result = M.validate_result(self.root)
        self.assertEqual(result["population"]["retained"], 320)
        self.assertEqual(result["population"]["attention"], 480)

    def test_02_predecessor_failure_authority_is_exact(self):
        review = M.verify_failed_predecessor()
        self.assertEqual(review["false_negative_count"], 5)

    def test_03_runtime_identity_is_explicit(self):
        M.validate_runtime_identity()
        self.assertEqual(Path(sys.executable).resolve(), M.PYTHON.resolve())
        self.assertEqual(np.__version__, M.NUMPY_VERSION)

    def test_04_broken_canonical_symlink_is_residue(self):
        broken = Path(self.temp.name) / "broken-canonical"
        os.symlink(Path(self.temp.name) / "absent-target", broken)
        self.assertTrue(os.path.lexists(str(broken)))
        with self.assertRaisesRegex(M.M1335Error, "residue"):
            M.canonical_absent(broken)

    def test_05_compressed_record_seal_actual_mismatch_rejected(self):
        path, rows = self.ordered(); row = next(r for r in rows if r["payload"].get("retained"))
        (self.root / row["payload"]["compressed_fp32"]).write_bytes(zlib.compress(b"changed"))
        self.reject()

    def test_06_support_record_seal_actual_mismatch_rejected(self):
        _path, rows = self.ordered(); row = next(r for r in rows if r["payload"].get("retained"))
        (self.root / row["payload"]["support_sign"]).write_bytes(b"zz")
        self.reject()

    def test_07_raw_content_identity_rejected(self):
        path, rows = self.ordered(); row = next(r for r in rows if r["payload"].get("retained"))
        row["payload"]["raw_fp32_sha256"] = "0" * 64
        self.write_ordered(path, rows); self.reject()

    def test_08_support_plane_extent_rejected(self):
        path, rows = self.ordered(); row = next(r for r in rows if r["payload"].get("retained"))
        row["payload"]["positive_plane_bytes"] = 2
        self.write_ordered(path, rows); self.reject()

    def test_09_operator_invented_identity_rejected(self):
        path = self.root / "operator_runtime.json"; rows = json.loads(path.read_text())
        rows[0]["name"] = "invented.operator"; write_json(path, rows); self.reject()

    def test_10_operator_order_rejected(self):
        path = self.root / "operator_runtime.json"; rows = json.loads(path.read_text())
        rows[0], rows[1] = rows[1], rows[0]; write_json(path, rows); self.reject()

    def test_11_atlif_invented_identity_rejected(self):
        path = self.root / "atlif_activity.json"; rows = json.loads(path.read_text())
        rows[0]["name"] = "invented.atlif"; write_json(path, rows); self.reject()

    def test_12_atlif_order_rejected(self):
        path = self.root / "atlif_activity.json"; rows = json.loads(path.read_text())
        rows[0], rows[1] = rows[1], rows[0]; write_json(path, rows); self.reject()

    def test_13_attention_q_string_dtype_rejected(self):
        path, value = self.attention(); row = value["records"][0]
        self.rewrite_npz(row, q_bits_packed=np.array(["x", "y"]))
        write_json(path, value); self.reject()

    def test_14_attention_k_float_dtype_rejected(self):
        path, value = self.attention(); row = value["records"][0]
        self.rewrite_npz(row, k_bits_packed=np.array([1.0, 2.0], dtype=np.float32))
        write_json(path, value); self.reject()

    def test_15_attention_bool_gate_rejected(self):
        path, value = self.attention(); row = value["records"][0]
        self.rewrite_npz(row, gate_q17=np.array([[[True, False]]], dtype=np.bool_))
        write_json(path, value); self.reject()

    def test_16_attention_shape_metadata_rejected(self):
        path, value = self.attention(); row = value["records"][0]
        self.rewrite_npz(row, q_shape=np.array([2, 1, 1, 2, 4], dtype=np.int32))
        write_json(path, value); self.reject()

    def test_17_attention_gate_geometry_rejected(self):
        path, value = self.attention(); row = value["records"][0]
        self.rewrite_npz(row, gate_q17=np.zeros((1, 1, 3), dtype=np.uint16))
        write_json(path, value); self.reject()

    def test_18_attention_record_stat_rejected(self):
        path, value = self.attention(); value["records"][0]["q_active_bits"] += 1
        write_json(path, value); self.reject()


if __name__ == "__main__":
    unittest.main(verbosity=2)
