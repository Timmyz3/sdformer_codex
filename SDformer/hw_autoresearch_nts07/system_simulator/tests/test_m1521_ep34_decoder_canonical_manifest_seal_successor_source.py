#!/usr/bin/env python3
"""Synthetic M1521 canonical re-derivation tests; no real capture reads."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SOURCE = (Path(__file__).resolve().parent.parent / "scripts" /
          "build_m1521_ep34_decoder_canonical_manifest_seal_successor_source.py")
SPEC = importlib.util.spec_from_file_location("test_m1521_source", SOURCE)
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


def synthetic_enriched() -> dict:
    calls = []
    for ordinal in range(120):
        sample = 10 + ordinal // 4
        module = ordinal % 4
        calls.append({
            "global_call_ordinal": ordinal,
            "global_order": sample * 247 + 200 + module,
            "global_sample_id": sample,
            "sequence": "sequence_%d" % (sample // 10),
            "sample_key": "sample_%02d" % sample,
            "module_ordinal": module,
            "module": M.M1516.M1510.M1323.MODULES[module],
            "shape": list(TINY_SHAPES[module]),
            "support_sign": "payloads/source_%03d.support_sign.le.bitpack" % ordinal,
            "support_sign_sha256": SUPPORT_SHA,
            "source_support_sign_sha256": SUPPORT_SHA,
            "positive_plane_sha256": POSITIVE_SHA,
            "negative_zero_plane_sha256": NEGATIVE_SHA,
            "positive_plane_bytes": 2,
            "negative_plane_bytes": 2,
            "plane_bytes": 2,
            "positive_word_uint32": M.M1516.EXPECTED_SCALE_WORDS[module],
            "negative_count": 0,
            "nonfinite_count": 0,
        })
    layers = [{
        "module_ordinal": ordinal,
        "module": M.M1516.M1510.M1323.MODULES[ordinal],
        "calls": 30,
        "word_uint32": M.M1516.EXPECTED_SCALE_WORDS[ordinal],
        "word_hex": "0x{:08x}".format(M.M1516.EXPECTED_SCALE_WORDS[ordinal]),
        "all_calls_same_word": True,
    } for ordinal in range(4)]
    return {
        "schema": M.M1516.M1510.SCHEMA,
        "status": M.M1516.M1510.STATUS,
        "capture_seal": {
            "sha256sums_sha256": M.M1516.CAPTURE_MANIFEST_SHA256,
            "outer_seal_sha256": M.M1516.CAPTURE_OUTER_SHA256,
        },
        "layer_scale_words": layers,
        "calls": calls,
    }


def expected_manifest() -> dict:
    with mock.patch.object(M.M1516.M1510.M1323, "SHAPES", TINY_SHAPES):
        return M.expected_manifest_from_enriched(synthetic_enriched())


def write_stage(root: Path, manifest: dict) -> None:
    (root / "payloads").mkdir(parents=True)
    for row in manifest["records"]:
        path = root.joinpath(*Path(row["positive_output"]).parts)
        M.M1516.write_exclusive(path, POSITIVE, 0o400)
    M.M1516.write_exclusive(root / "manifest.json",
                            (json.dumps(manifest, sort_keys=True) + "\n").encode(), 0o400)
    M.M1516.write_exclusive(root / "RUN_COMPLETE.txt", M.RUN_TOKEN.encode(), 0o400)


def weak_self_consistent_seal(root: Path) -> None:
    members = M._sealed_payload_files(root)
    lines = [M.sha256(path) + "  " + path.relative_to(root).as_posix()
             for path in members]
    M.M1516.write_exclusive(root / M.M1516.MANIFEST,
                            ("\n".join(lines) + "\n").encode(), 0o400)
    M.M1516.write_exclusive(
        root / M.M1516.OUTER,
        (M.sha256(root / M.M1516.MANIFEST) + "  " + M.M1516.MANIFEST + "\n").encode(),
        0o400)


class CanonicalSealTests(unittest.TestCase):
    def test_01_public_verifiers_do_not_accept_external_expected(self):
        self.assertEqual(list(inspect.signature(M.seal_staging).parameters), ["root"])
        self.assertEqual(list(inspect.signature(M.verify_materialized_seal).parameters),
                         ["root"])
        derive = inspect.getsource(M.derive_canonical_expected)
        self.assertIn("M1510.audit_capture", derive)
        self.assertIn("enrich_audit", derive)
        self.assertIn("expected_manifest_from_enriched", derive)

    def test_02_valid_preseal_and_postpublication_full_tree_pass(self):
        expected = expected_manifest()
        with tempfile.TemporaryDirectory(prefix="m1521_valid_") as directory:
            stage = Path(directory) / "stage"; stage.mkdir()
            write_stage(stage, expected)
            receipt = M._seal_against_expected(stage, expected)
            self.assertTrue(receipt["full_tree_equal"])
            self.assertEqual(receipt["canonical_paths"], 120)
            self.assertEqual(M._verify_against_expected(stage, expected)["members"], 122)

    def assert_preseal_rejects(self, mutation, message):
        expected = expected_manifest()
        observed = copy.deepcopy(expected)
        mutation(observed)
        with tempfile.TemporaryDirectory(prefix="m1521_preseal_") as directory:
            stage = Path(directory) / "stage"; stage.mkdir()
            write_stage(stage, observed)
            with self.assertRaisesRegex(M.M1521Error, message):
                M._seal_against_expected(stage, expected)
            self.assertFalse((stage / M.M1516.MANIFEST).exists())

    def test_03_preseal_rejects_m1517_semantic_forgery(self):
        attacks = (
            (lambda x: x["records"][0].update(
                layer_scale_word_uint32=0x3F800000), "value drift"),
            (lambda x: x["records"][0].update(numeric_encoding="exact_binary"),
             "value drift"),
            (lambda x: x["records"][0].update(weight_folding=True), "value drift"),
            (lambda x: x["records"][0].update(normalized=True), "value drift"),
            (lambda x: x["records"][0].update(coerced=True), "value drift"),
            (lambda x: x["records"][1].update(
                capture_global_order=x["records"][0]["capture_global_order"]),
             "value drift"),
            (lambda x: x["claim_boundary"].update(cycles=True), "value drift"),
        )
        for mutation, message in attacks:
            with self.subTest(mutation=repr(mutation)):
                self.assert_preseal_rejects(mutation, message)

    def test_04_preseal_rejects_renamed_but_self_consistent_path(self):
        self.assert_preseal_rejects(
            lambda x: x["records"][0].update(
                positive_output="payloads/renamed_attack.bin"), "value drift")

    def test_05_postpublication_rejects_semantic_forgery(self):
        expected = expected_manifest()
        forged = copy.deepcopy(expected)
        forged["records"][0].update(
            layer_scale_word_uint32=0x3F800000,
            numeric_encoding="exact_binary", weight_folding=True,
            normalized=True, coerced=True)
        forged["records"][1]["capture_global_order"] = forged["records"][0][
            "capture_global_order"]
        forged["claim_boundary"]["cycles"] = True
        with tempfile.TemporaryDirectory(prefix="m1521_post_") as directory:
            root = Path(directory) / "published"; root.mkdir()
            write_stage(root, forged); weak_self_consistent_seal(root)
            with self.assertRaisesRegex(M.M1521Error, "value drift"):
                M._verify_against_expected(root, expected)

    def test_06_postpublication_rejects_renamed_self_consistent_path(self):
        expected = expected_manifest()
        forged = copy.deepcopy(expected)
        forged["records"][0]["positive_output"] = "payloads/renamed_attack.bin"
        with tempfile.TemporaryDirectory(prefix="m1521_path_") as directory:
            root = Path(directory) / "published"; root.mkdir()
            write_stage(root, forged); weak_self_consistent_seal(root)
            with self.assertRaisesRegex(M.M1521Error, "value drift"):
                M._verify_against_expected(root, expected)

    def test_07_type_strict_bool_integer_alias_rejected(self):
        expected = expected_manifest()
        forged = copy.deepcopy(expected)
        forged["records"][0]["global_call_ordinal"] = False
        with tempfile.TemporaryDirectory(prefix="m1521_type_") as directory:
            root = Path(directory) / "published"; root.mkdir()
            write_stage(root, forged); weak_self_consistent_seal(root)
            with self.assertRaisesRegex(M.M1521Error, "type drift"):
                M._verify_against_expected(root, expected)

    def test_08_postpublication_rejects_payload_sha_drift(self):
        expected = expected_manifest()
        with tempfile.TemporaryDirectory(prefix="m1521_payload_") as directory:
            root = Path(directory) / "published"; root.mkdir()
            write_stage(root, expected); weak_self_consistent_seal(root)
            victim = root / expected["records"][0]["positive_output"]
            victim.chmod(0o600); victim.write_bytes(b"bad")
            with self.assertRaisesRegex(M.M1521Error, "member SHA drift"):
                M._verify_against_expected(root, expected)


class AuthorityReleaseAndPolicyTests(unittest.TestCase):
    def test_09_m1517_failure_is_exact_bound(self):
        review = M.verify_m1517_failure()
        self.assertEqual(review["status"], M.M1517_STATUS)
        self.assertFalse(review["authorization"]["m1518_release_authoring"])
        self.assertEqual(M.sha256(M.M1516_SOURCE), M.M1516_SOURCE_SHA256)

    def test_10_execute_requires_m1522_before_materialization(self):
        source = inspect.getsource(M.execute_once)
        self.assertLess(source.index("verify_m1522_hammer"),
                        source.index("materialize_canonical_once"))
        self.assertLess(source.index("verify_m1517_failure"),
                        source.index("materialize_canonical_once"))

    def test_11_cli_is_source_only_and_namespace_absent(self):
        with self.assertRaises(M.M1521Error):
            M.main([])
        text = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("--materialize", text)
        self.assertNotIn("--production", text)
        for token in ("subprocess", "paramiko", "torch.cuda", "ssh ",
                      "vcs", "dc_shell", "pt_shell"):
            self.assertNotIn(token, text)
        self.assertFalse(M.CLAIM_BOUNDARY["production"])
        self.assertFalse(M.CLAIM_BOUNDARY["cycles"])
        self.assertFalse(M.OUTPUT.exists())
        self.assertFalse(M.ATTEMPT.exists())


if __name__ == "__main__":
    unittest.main()
