#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib


SOURCE = (Path(__file__).resolve().parent.parent / "scripts" /
          "build_m1321_ep34_decoder_capture_adapter_source.py")
SPEC = importlib.util.spec_from_file_location("m1321_source", SOURCE)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_payload(root, words, stem="x"):
    raw = b"".join(struct.pack("<I", word) for word in words)
    positive = bytearray((len(words) + 7) // 8)
    negative = bytearray((len(words) + 7) // 8)
    for index, word in enumerate(words):
        if word != 0:
            (negative if word & 0x80000000 else positive)[index >> 3] |= 1 << (index & 7)
    compressed = root / (stem + ".fp32.zlib")
    support = root / (stem + ".support_sign.le.bitpack")
    compressed.write_bytes(zlib.compress(raw))
    support.write_bytes(bytes(positive + negative))
    return compressed, support, hashlib.sha256(raw).hexdigest()


class PayloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m1321_")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def audit(self, words, module):
        compressed, support, raw_sha = write_payload(self.root, words)
        return M.audit_two_plane_payload(
            compressed, support, (1, 1, 1, 1, len(words)), module,
            raw_sha, sha(compressed), sha(support))

    def test_exact_binary_and_dynamic_theta_pass(self):
        binary = self.audit([0, M.ONE_WORD, 0, M.ONE_WORD, 0, 0, 0, 0, 0], 0)
        self.assertEqual(binary["active"], 2)
        theta = 0x3F400000
        scaled = self.audit([0, theta, 0, theta, 0, 0, 0, 0, 0], 1)
        self.assertEqual(scaled["theta_word_uint32"], theta)
        self.assertEqual(scaled["negative_count"], 0)

    def test_two_plane_extent_and_negative_rejected(self):
        compressed, support, raw_sha = write_payload(self.root, [0, M.ONE_WORD], "bad")
        support.write_bytes(support.read_bytes()[:-1])
        with self.assertRaisesRegex(M.AdapterError, "positive.*negative"):
            M.audit_two_plane_payload(compressed, support, (1, 1, 1, 1, 2), 0,
                                      raw_sha)
        compressed, support, raw_sha = write_payload(self.root, [0, 0xBF800000], "neg")
        with self.assertRaisesRegex(M.AdapterError, "negative plane"):
            M.audit_two_plane_payload(compressed, support, (1, 1, 1, 1, 2), 0,
                                      raw_sha)

    def test_binary_near_one_and_d1_multivalue_rejected(self):
        with self.assertRaisesRegex(M.AdapterError, "exact"):
            self.audit([0, 0x3F7FFFFF], 2)
        with self.assertRaisesRegex(M.AdapterError, "multiple nonzero theta"):
            self.audit([0, 0x3F400000, 0x3F000000], 1)
        all_zero = self.audit([0, 0, 0], 1)
        self.assertIsNone(all_zero["theta_word_uint32"])

    def test_padding_sha_and_trailing_stream_attacks_rejected(self):
        compressed, support, raw_sha = write_payload(self.root, [M.ONE_WORD], "pad")
        value = bytearray(support.read_bytes()); value[0] |= 0x80; support.write_bytes(value)
        with self.assertRaisesRegex(M.AdapterError, "padding"):
            M.audit_two_plane_payload(compressed, support, (1, 1, 1, 1, 1), 0,
                                      raw_sha)
        compressed, support, raw_sha = write_payload(self.root, [M.ONE_WORD], "trail")
        compressed.write_bytes(compressed.read_bytes() + b"junk")
        with self.assertRaisesRegex(M.AdapterError, "trailing"):
            M.audit_two_plane_payload(compressed, support, (1, 1, 1, 1, 1), 0,
                                      raw_sha)
        compressed, support, raw_sha = write_payload(self.root, [M.ONE_WORD], "sha")
        with self.assertRaisesRegex(M.AdapterError, "raw FP32 SHA"):
            M.audit_two_plane_payload(compressed, support, (1, 1, 1, 1, 1), 0,
                                      "0" * 64)


class PopulationAndWeightTests(unittest.TestCase):
    def make_ordered(self):
        rows = []
        order = 0
        for sample in range(40):
            for module in range(247):
                if module < 4:
                    ordinal = module
                    name = M.MODULES[ordinal]
                    category = "decoder_convtranspose"
                    shape = list(M.SHAPES[ordinal])
                    payload = {"retained": True,
                        "compressed_fp32": "payloads/a.fp32.zlib",
                        "compressed_sha256": "1" * 64,
                        "support_sign": "payloads/a.support_sign.le.bitpack",
                        "support_sign_sha256": "2" * 64,
                        "raw_fp32_sha256": "3" * 64,
                        "positive_plane_bytes": (M.product(shape) + 7) // 8,
                        "negative_plane_bytes": (M.product(shape) + 7) // 8}
                else:
                    name = "other.%d" % module; category = "atlif"
                    shape = [1]; payload = {"retained": False}
                rows.append({"global_order": order, "global_sample_id": sample,
                    "category": category, "name": name, "sequence": "s",
                    "sample_key": "k", "source_sha256": "4" * 64,
                    "input": {"shape": shape}, "payload": payload})
                order += 1
        return rows

    def test_decoder_population_selects_global_10_through_39(self):
        selected = M.decoder_rows_from_ordered(self.make_ordered())
        self.assertEqual(len(selected), 120)
        self.assertEqual((selected[0]["global_sample_id"],
                          selected[-1]["global_sample_id"]), (10, 39))
        self.assertEqual([row["module_ordinal"] for row in selected[:4]], [0, 1, 2, 3])

    def test_missing_call_and_wrong_total_rejected(self):
        rows = self.make_ordered()
        rows[10 * 247]["category"] = "atlif"
        with self.assertRaisesRegex(M.AdapterError, "four calls"):
            M.decoder_rows_from_ordered(rows)
        with self.assertRaisesRegex(M.AdapterError, "9880"):
            M.decoder_rows_from_ordered(self.make_ordered()[:-1])

    def test_weight_identity_interface(self):
        checkpoint = "a" * 64
        rows = []
        for ordinal, shape in enumerate(M.WEIGHT_SHAPES):
            rows.append({"module_ordinal": ordinal, "module": M.MODULES[ordinal],
                "checkpoint_sha256": checkpoint,
                "weight": {"shape": list(shape), "dtype": "torch.float32",
                    "layout": "C_ORDER_CONTIGUOUS", "byte_order": "little",
                    "content_bytes": M.product(shape) * 4,
                    "content_sha256": ("%x" % (ordinal + 1)) * 64},
                "bias": None})
        self.assertEqual(len(M.validate_weight_identities(rows, checkpoint)), 4)
        rows[1]["bias"] = {"shape": [1]}
        with self.assertRaisesRegex(M.AdapterError, "bias"):
            M.validate_weight_identities(rows, checkpoint)

    def test_cli_has_no_production_mode(self):
        with self.assertRaises(M.AdapterError):
            M.main([])


if __name__ == "__main__":
    unittest.main()
