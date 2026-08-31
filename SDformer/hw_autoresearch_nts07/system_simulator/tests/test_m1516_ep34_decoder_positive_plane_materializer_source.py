#!/usr/bin/env python3
"""Synthetic M1516 tests; never reads or materializes the real capture."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SOURCE = (Path(__file__).resolve().parent.parent / "scripts" /
          "build_m1516_ep34_decoder_positive_plane_materializer_source.py")
SPEC = importlib.util.spec_from_file_location("test_m1516_source", SOURCE)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)

TINY_SHAPES = ((1, 1, 1, 1, 9),) * 4
POSITIVE = bytes((0x05, 0x01))
NEGATIVE = bytes((0x00, 0x00))
POSITIVE_SHA = hashlib.sha256(POSITIVE).hexdigest()
NEGATIVE_SHA = hashlib.sha256(NEGATIVE).hexdigest()
SUPPORT_SHA = hashlib.sha256(POSITIVE + NEGATIVE).hexdigest()


def synthetic_audit() -> dict:
    calls = []
    for ordinal in range(120):
        sample = 10 + ordinal // 4
        module = ordinal % 4
        calls.append({
            "global_call_ordinal": ordinal,
            "global_order": sample * 247 + module,
            "global_sample_id": sample,
            "sequence": "sequence_%d" % (sample // 10),
            "sample_key": "sample_%02d" % sample,
            "module_ordinal": module,
            "module": M.M1510.M1323.MODULES[module],
            "shape": list(TINY_SHAPES[module]),
            "support_sign": "payloads/source_%03d.support_sign.le.bitpack" % ordinal,
            "support_sign_sha256": SUPPORT_SHA,
            "source_support_sign_sha256": SUPPORT_SHA,
            "positive_plane_sha256": POSITIVE_SHA,
            "negative_zero_plane_sha256": NEGATIVE_SHA,
            "positive_plane_bytes": 2,
            "negative_plane_bytes": 2,
            "plane_bytes": 2,
            "positive_word_uint32": M.EXPECTED_SCALE_WORDS[module],
            "negative_count": 0,
            "nonfinite_count": 0,
        })
    layers = [{
        "module_ordinal": ordinal,
        "module": M.M1510.M1323.MODULES[ordinal],
        "calls": 30,
        "word_uint32": M.EXPECTED_SCALE_WORDS[ordinal],
        "word_hex": "0x{:08x}".format(M.EXPECTED_SCALE_WORDS[ordinal]),
        "all_calls_same_word": True,
    } for ordinal in range(4)]
    return {
        "schema": M.M1510.SCHEMA,
        "status": M.M1510.STATUS,
        "capture_seal": {
            "sha256sums_sha256": M.CAPTURE_MANIFEST_SHA256,
            "outer_seal_sha256": M.CAPTURE_OUTER_SHA256,
        },
        "layer_scale_words": layers,
        "calls": calls,
    }


class ManifestTests(unittest.TestCase):
    def build(self, audit=None):
        with mock.patch.object(M.M1510.M1323, "SHAPES", TINY_SHAPES):
            return M.build_output_manifest(
                synthetic_audit() if audit is None else audit)

    def test_01_exact_30x4_manifest_and_encoding(self):
        value = self.build()
        self.assertEqual(len(value["records"]), 120)
        self.assertEqual([row["module_ordinal"] for row in value["records"][:4]],
                         [0, 1, 2, 3])
        self.assertEqual([row["global_sample_id"] for row in value["records"][::4]],
                         list(range(10, 40)))
        self.assertEqual([row["numeric_encoding"] for row in value["records"][:4]],
                         ["bit_times_layer_constant", "bit_times_layer_constant",
                          "exact_binary", "exact_binary"])
        self.assertTrue(all(row["negative_plane_output"] is None for row in
                            value["records"]))
        self.assertTrue(all(not row["weight_folding"] and not row["normalized"] and
                            not row["coerced"] for row in value["records"]))

    def test_02_scale_call_order_and_duplicate_attacks_rejected(self):
        audit = synthetic_audit()
        audit["layer_scale_words"][0]["word_uint32"] ^= 1
        with self.assertRaisesRegex(M.M1516Error, "scale word drift"):
            self.build(audit)
        audit = synthetic_audit()
        audit["calls"][5]["global_sample_id"] = 39
        with self.assertRaisesRegex(M.M1516Error, "identity/order"):
            self.build(audit)
        audit = synthetic_audit()
        audit["calls"][1]["global_order"] = audit["calls"][0]["global_order"]
        with self.assertRaisesRegex(M.M1516Error, "duplicate"):
            self.build(audit)
        audit = synthetic_audit()
        audit["calls"][1]["support_sign"] = audit["calls"][0]["support_sign"]
        with self.assertRaisesRegex(M.M1516Error, "duplicate"):
            self.build(audit)


class PlaneAndPathTests(unittest.TestCase):
    def test_03_positive_copy_split_and_no_negative_output(self):
        with tempfile.TemporaryDirectory(prefix="m1516_copy_") as directory:
            root = Path(directory)
            source = root / "support.bin"
            destination = root / "positive.bin"
            source.write_bytes(POSITIVE + NEGATIVE)
            M.copy_positive_plane_exclusive(
                source, destination, 9, 2, SUPPORT_SHA, POSITIVE_SHA, NEGATIVE_SHA)
            self.assertEqual(destination.read_bytes(), POSITIVE)
            self.assertEqual(sorted(path.name for path in root.iterdir()),
                             ["positive.bin", "support.bin"])
            with self.assertRaises(FileExistsError):
                M.copy_positive_plane_exclusive(
                    source, destination, 9, 2, SUPPORT_SHA, POSITIVE_SHA, NEGATIVE_SHA)

    def test_04_padding_and_negative_plane_attacks_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m1516_padding_") as directory:
            root = Path(directory)
            for name, payload, message in (
                    ("padding", bytes((0x05, 0x81)) + NEGATIVE, "padding"),
                    ("negative", POSITIVE + bytes((0x01, 0x00)), "negative")):
                source = root / (name + ".bin")
                source.write_bytes(payload)
                with self.subTest(name=name), self.assertRaisesRegex(
                        M.M1516Error, message):
                    M.copy_positive_plane_exclusive(
                        source, root / (name + ".out"), 9, 2,
                        hashlib.sha256(payload).hexdigest(),
                        hashlib.sha256(payload[:2]).hexdigest(),
                        hashlib.sha256(payload[2:]).hexdigest())

    def test_05_path_traversal_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory(prefix="m1516_path_") as directory:
            root = Path(directory)
            (root / "real.bin").write_bytes(b"x")
            with self.assertRaisesRegex(M.M1516Error, "unsafe"):
                M.safe_member(root, "../real.bin", "member")
            (root / "link.bin").symlink_to(root / "real.bin")
            with self.assertRaisesRegex(M.M1516Error, "symlink"):
                M.safe_member(root, "link.bin", "member")

    def test_06_atomic_no_replace_collision(self):
        with tempfile.TemporaryDirectory(prefix="m1516_rename_") as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            source.mkdir(); destination.mkdir()
            with self.assertRaisesRegex(M.M1516Error, "collision"):
                M.rename_noreplace(source, destination)
            self.assertTrue(source.exists() and destination.exists())


class OneShotAndSealTests(unittest.TestCase):
    def test_07_attempt_is_o_excl_and_no_retry(self):
        with tempfile.TemporaryDirectory(prefix="m1516_attempt_") as directory:
            attempt = Path(directory) / "attempt"
            M.consume_attempt(attempt)
            self.assertEqual(attempt.read_text(), M.ATTEMPT_TOKEN)
            with self.assertRaises(FileExistsError):
                M.consume_attempt(attempt)

    def make_sealed_staging(self, root: Path):
        (root / "payloads").mkdir(parents=True)
        with mock.patch.object(M.M1510.M1323, "SHAPES", TINY_SHAPES):
            manifest = M.build_output_manifest(synthetic_audit())
        for row in manifest["records"]:
            path = root.joinpath(*Path(row["positive_output"]).parts)
            M.write_exclusive(path, POSITIVE, 0o400)
        M.write_exclusive(root / "manifest.json",
                          (json.dumps(manifest, sort_keys=True) + "\n").encode(), 0o400)
        M.write_exclusive(root / "RUN_COMPLETE.txt", M.RUN_TOKEN.encode(), 0o400)
        return manifest

    def test_08_double_seal_population_and_manifest_sha_binding(self):
        with tempfile.TemporaryDirectory(prefix="m1516_seal_") as directory:
            root = Path(directory) / "stage"
            root.mkdir()
            self.make_sealed_staging(root)
            receipt = M.seal_staging(root)
            self.assertEqual(receipt["members"], 122)
            (root / "payloads/c000_s10_d0.positive.le.bitpack").chmod(0o600)
            (root / "payloads/c000_s10_d0.positive.le.bitpack").write_bytes(b"bad")
            with self.assertRaisesRegex(M.M1516Error, "drift"):
                M.verify_materialized_seal(root)

    def test_09_failure_preserves_attempt_and_stage(self):
        with tempfile.TemporaryDirectory(prefix="m1516_fail_") as directory:
            root = Path(directory)
            capture = root / "capture"
            (capture / "payloads").mkdir(parents=True)
            # Only call zero exists; call one fails after the exclusive stage exists.
            (capture / "payloads/source_000.support_sign.le.bitpack").write_bytes(
                POSITIVE + NEGATIVE)
            output = root / "output"
            attempt = root / "attempt"
            with mock.patch.object(M.M1510.M1323, "SHAPES", TINY_SHAPES), \
                    self.assertRaises(M.M1516Error):
                M.materialize_prepared_once(
                    capture, synthetic_audit(), output, attempt, ".stage.")
            self.assertTrue(attempt.exists())
            self.assertFalse(output.exists())
            stages = list(root.glob(".stage.*"))
            self.assertEqual(len(stages), 1)
            self.assertTrue((stages[0] / "payloads/c000_s10_d0.positive.le.bitpack").is_file())


class AuthorityAndPolicyTests(unittest.TestCase):
    def test_10_exact_capture_authority_chain(self):
        authority = M.verify_authorities()
        self.assertIn("M1512", authority["m1512"])
        self.assertIn("M1513", authority["m1513"])
        self.assertEqual(M.sha256(M.M1510_SOURCE), M.M1510_SOURCE_SHA256)
        self.assertEqual(M.sha256(M.M1510_CONTRACT), M.M1510_CONTRACT_SHA256)

    def test_11_release_shape_and_claim_boundary(self):
        release = {
            "schema": M.RELEASE_SCHEMA, "status": M.RELEASE_STATUS,
            "source_identity": {
                "source_path": str(M.SOURCE.relative_to(M.ROOT)),
                "source_sha256": M.sha256(M.SOURCE),
                "test_path": str(M.TEST.relative_to(M.ROOT)),
                "test_sha256": M.sha256(M.TEST),
                "contract_path": str(M.CONTRACT.relative_to(M.ROOT)),
                "contract_sha256": M.sha256(M.CONTRACT)},
            "m1517_source_hammer": {"path": "future", "review_sha256": "1" * 64,
                                     "manifest_sha256": "2" * 64,
                                     "outer_file_sha256": "3" * 64},
            "authority": {
                "m1510_source_sha256": M.M1510_SOURCE_SHA256,
                "m1510_contract_sha256": M.M1510_CONTRACT_SHA256,
                "m1512_review_manifest_outer": list(M.M1512_PINS),
                "m1513_review_manifest_outer": list(M.M1513_PINS),
                "capture_manifest_sha256": M.CAPTURE_MANIFEST_SHA256,
                "capture_outer_sha256": M.CAPTURE_OUTER_SHA256,
                "checkpoint_sha256": M.CHECKPOINT_SHA256},
            "one_shot": {"attempt_marker": str(M.ATTEMPT.relative_to(M.ROOT)),
                         "automatic_retry": False, "maximum_materializations": 1,
                         "failure_stage_preserved": True},
            "output": {"path": str(M.OUTPUT.relative_to(M.ROOT)),
                       "positive_plane_files": 120, "negative_plane_files": 0,
                       "atomic_no_replace": True, "recursive_double_seal": True},
            "claim_boundary": {"positive_plane_materialization": True,
                               "address_timed_replay": False, "cycles": False,
                               "traffic": False, "speedup": False,
                               "system_speedup": False, "energy": False,
                               "rtl": False, "eda": False, "ppa": False,
                               "table_a": False},
        }
        M.validate_release_shape(release)
        attacked = copy.deepcopy(release)
        attacked["one_shot"]["automatic_retry"] = True
        with self.assertRaisesRegex(M.M1516Error, "one-shot"):
            M.validate_release_shape(attacked)

    def test_12_cli_is_source_only_and_output_absent(self):
        with self.assertRaises(M.M1516Error):
            M.main([])
        text = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("--materialize", text)
        self.assertNotIn("--production", text)
        for token in ("subprocess", "paramiko", "torch.cuda", "ssh ",
                      "vcs", "dc_shell", "pt_shell"):
            self.assertNotIn(token, text)
        self.assertFalse(M.CLAIM_BOUNDARY["production"])
        self.assertFalse(M.CLAIM_BOUNDARY["cycles"])
        self.assertFalse(M.CLAIM_BOUNDARY["traffic"])


if __name__ == "__main__":
    unittest.main()
