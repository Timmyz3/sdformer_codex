#!/usr/bin/env python3
"""Small local tests for M1510; never reads the real M1458 capture."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib


SOURCE = (Path(__file__).resolve().parent.parent / "scripts" /
          "build_m1510_ep34_decoder_layer_constant_adapter_source.py")
SPEC = importlib.util.spec_from_file_location("test_m1510_source", SOURCE)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_payload(root: Path, words: list[int], stem: str = "payload") -> dict:
    raw = b"".join(struct.pack("<I", word) for word in words)
    plane_bytes = (len(words) + 7) // 8
    positive = bytearray(plane_bytes)
    negative = bytearray(plane_bytes)
    for index, word in enumerate(words):
        if word != 0:
            plane = negative if word & 0x80000000 else positive
            plane[index >> 3] |= 1 << (index & 7)
    compressed_relative = stem + ".fp32.zlib"
    support_relative = stem + ".support_sign.le.bitpack"
    compressed = root / compressed_relative
    support = root / support_relative
    compressed.write_bytes(zlib.compress(raw))
    support.write_bytes(bytes(positive + negative))
    return {
        "global_call_ordinal": 0, "global_order": 0,
        "global_sample_id": 10, "sequence": "s", "sample_key": "k",
        "source_sha256": "a" * 64, "module_ordinal": 0,
        "module": M.M1323.MODULES[0], "shape": [1, 1, 1, 1, len(words)],
        "compressed_fp32": compressed_relative,
        "compressed_sha256": digest(compressed),
        "support_sign": support_relative,
        "support_sign_sha256": digest(support),
        "raw_fp32_sha256": hashlib.sha256(raw).hexdigest(),
        "positive_plane_bytes": plane_bytes,
        "negative_plane_bytes": plane_bytes,
    }


class PayloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m1510_")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def call(self, ordinal: int, words: list[int], stem: str = "payload"):
        row = write_payload(self.root, words, stem)
        row["module_ordinal"] = ordinal
        row["module"] = M.M1323.MODULES[ordinal]
        return M.audit_call_payload(self.root, row)

    def test_01_four_exact_layer_words_pass(self):
        for ordinal, word in M.EXPECTED_WORDS.items():
            with self.subTest(ordinal=ordinal):
                result = self.call(ordinal, [0, word, 0, word], "p%d" % ordinal)
                self.assertEqual(result["positive_word_uint32"], word)
                self.assertEqual(result["negative_count"], 0)
                self.assertEqual(result["nonfinite_count"], 0)

    def test_02_d0_and_d1_multiple_theta_rejected(self):
        for ordinal in (0, 1):
            words = [0, M.EXPECTED_WORDS[ordinal], 0x3F000000]
            with self.subTest(ordinal=ordinal), self.assertRaisesRegex(
                    M.M1510Error, "multiple nonzero theta|exactly one unique"):
                self.call(ordinal, words, "multi%d" % ordinal)

    def test_03_negative_and_nonfinite_rejected(self):
        for word, message in ((0xBF800000, "negative"),
                              (0x7F800000, "positive finite|nonfinite")):
            with self.subTest(word=word), self.assertRaisesRegex(
                    M.M1510Error, message):
                self.call(0, [0, word], "bad%x" % word)

    def test_04_d2_and_d3_non_one_rejected(self):
        for ordinal in (2, 3):
            with self.subTest(ordinal=ordinal), self.assertRaisesRegex(
                    M.M1510Error, "exact|ONE"):
                self.call(ordinal, [0, 0x3F7FFFFF], "near%d" % ordinal)

    def test_05_sha_shape_and_padding_attacks_rejected(self):
        row = write_payload(self.root, [0, M.EXPECTED_WORDS[0]], "sha")
        row["compressed_sha256"] = "0" * 64
        with self.assertRaisesRegex(M.M1510Error, "SHA"):
            M.audit_call_payload(self.root, row)

        row = write_payload(self.root, [0, M.EXPECTED_WORDS[0]], "shape")
        row["shape"][-1] = 3
        with self.assertRaisesRegex(M.M1510Error, "extent|shape"):
            M.audit_call_payload(self.root, row)

        row = write_payload(self.root, [M.EXPECTED_WORDS[0]], "padding")
        support = self.root / row["support_sign"]
        value = bytearray(support.read_bytes())
        value[0] |= 0x80
        support.write_bytes(value)
        row["support_sign_sha256"] = digest(support)
        with self.assertRaisesRegex(M.M1510Error, "padding"):
            M.audit_call_payload(self.root, row)

    def test_06_cross_call_layer_word_drift_rejected(self):
        rows = []
        for sample in range(10, 40):
            for ordinal in range(4):
                rows.append({"global_sample_id": sample,
                             "module_ordinal": ordinal,
                             "positive_word_uint32": M.EXPECTED_WORDS[ordinal]})
        layers = M.summarize_layers(rows)
        self.assertEqual([row["calls"] for row in layers], [30] * 4)
        attacked = copy.deepcopy(rows)
        attacked[0]["positive_word_uint32"] = 0x3F000000
        with self.assertRaisesRegex(M.M1510Error, "drifts"):
            M.summarize_layers(attacked)

    def test_07_empty_positive_call_rejected(self):
        with self.assertRaisesRegex(M.M1510Error, "positive finite"):
            self.call(1, [0, 0, 0], "empty")


class PolicyTests(unittest.TestCase):
    def test_08_exact_predecessor_bindings(self):
        self.assertEqual(M.sha256(M.M1323_SOURCE), M.M1323_SHA256)
        self.assertEqual(M.sha256(M.M1501_SOURCE), M.M1501_SHA256)
        self.assertEqual(M.M1321.ONE_WORD, M.EXPECTED_WORDS[2])

    def test_09_source_has_no_action_or_performance_path(self):
        text = SOURCE.read_text(encoding="utf-8")
        for token in ("subprocess", "paramiko", "torch.cuda", "os.kill",
                      "ssh ", "vcs", "dc_shell", "pt_shell"):
            self.assertNotIn(token, text)
        self.assertNotIn('add_argument("--run"', text)
        self.assertFalse(M.CLAIM_BOUNDARY["bitplane"])
        self.assertFalse(M.CLAIM_BOUNDARY["cycles"])
        self.assertFalse(M.CLAIM_BOUNDARY["speedup"])


if __name__ == "__main__":
    unittest.main()
