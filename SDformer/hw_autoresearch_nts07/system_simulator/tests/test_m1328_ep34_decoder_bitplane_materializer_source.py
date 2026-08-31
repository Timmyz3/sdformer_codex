#!/usr/bin/env python3
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock


SOURCE = (Path(__file__).resolve().parent.parent / "scripts" /
          "build_m1328_ep34_decoder_bitplane_materializer_source.py")
SPEC = importlib.util.spec_from_file_location("m1328_source", SOURCE)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def synthetic_audit(theta=0x3F400000):
    calls = []
    for ordinal in range(120):
        sample = 10 + ordinal // 4
        module = ordinal % 4
        shape = list(M.M1323.SHAPES[module])
        plane_bytes = (math.prod(shape) + 7) // 8
        calls.append({
            "global_call_ordinal": ordinal,
            "global_order": sample * 247 + module,
            "global_sample_id": sample,
            "sequence": ("interlaken_01_a" if sample < 20 else
                         "thun_01_b" if sample < 30 else "zurich_city_12_a"),
            "sample_key": "sample_%02d.npy" % sample,
            "source_sha256": "1" * 64,
            "module_ordinal": module, "module": M.M1323.MODULES[module],
            "shape": shape, "elements": math.prod(shape),
            "support_sign": "payloads/source_%03d.support_sign.le.bitpack" % ordinal,
            "support_sign_sha256": "2" * 64,
            "raw_fp32_sha256": "3" * 64,
            "positive_plane_bytes": plane_bytes,
            "negative_plane_bytes": plane_bytes,
            "positive_plane_sha256": "4" * 64,
            "negative_plane_sha256": "5" * 64,
            "negative_count": 0,
        })
    return {
        "calls": calls,
        "d1": {"theta_word_uint32": theta,
               "theta_ieee754_le_hex": struct.pack("<I", theta).hex()},
        "ordered_jsonl_sha256": "6" * 64,
        "ordered_identity": {"ordered_rows": 9880},
    }


def synthetic_authority():
    return {
        "release": {
            "capture_result": {"path": str(M.M1327_CAPTURE.relative_to(M.ROOT)),
                               "manifest_sha256": "7" * 64,
                               "outer_file_sha256": "8" * 64,
                               "capture_manifest_sha256": "9" * 64,
                               "admission_sha256": "a" * 64},
            "capture_result_hammer": {"path": "hw_autoresearch_nts07/reviews/future",
                                      "manifest_sha256": "b" * 64,
                                      "outer_file_sha256": "c" * 64,
                                      "review_sha256": "d" * 64},
        },
        "result_hammer": {"identity": {"epoch": 34, "checkpoint_sha256": "e" * 64,
                                        "config_sha256": "f" * 64,
                                        "profile_sha256": "0" * 64}},
    }


class M1328Tests(unittest.TestCase):
    def test_01_m1323_and_m1324_authorities_exact(self):
        review = M.verify_m1324_hammer()
        template = M.verify_m1111dr2_template()
        self.assertEqual(review["status"], M.M1324_STATUS)
        self.assertTrue(review["authorization"]["actual_result_successor_authoring"])
        self.assertFalse(review["authorization"]["production_replay"])
        self.assertEqual(template["identity"]["runner_sha256"], M.M1111DR2_RUNNER_SHA256)

    def test_02_dynamic_theta_and_exact_30x4_manifest(self):
        theta = 0x3F400000
        manifest = M.build_output_manifest(synthetic_audit(theta), synthetic_authority())
        self.assertEqual(manifest["d1_dynamic_theta"]["word_uint32"], theta)
        self.assertEqual(manifest["d1_dynamic_theta"]["ieee754_le_hex"], "0000403f")
        self.assertEqual(len(manifest["records"]), 120)
        self.assertEqual([row["global_sample_id"] for row in manifest["records"][::4]],
                         list(range(10, 40)))
        self.assertEqual([row["module_ordinal"] for row in manifest["records"][:4]],
                         [0, 1, 2, 3])
        self.assertFalse(manifest["claim_boundary"]["decoder_replay"])

    def test_03_theta_and_call_identity_attacks_rejected(self):
        with self.assertRaisesRegex(M.M1328Error, "theta"):
            M.build_output_manifest(synthetic_audit(0), synthetic_authority())
        audit = synthetic_audit()
        audit["calls"][5]["global_sample_id"] = 39
        with self.assertRaisesRegex(M.M1328Error, "identity/order"):
            M.build_output_manifest(audit, synthetic_authority())
        audit = synthetic_audit()
        audit["calls"][1]["negative_count"] = 1
        with self.assertRaisesRegex(M.M1328Error, "sign"):
            M.build_output_manifest(audit, synthetic_authority())

    def test_04_output_plane_names_unique_and_bool_rejected(self):
        names = [M.output_plane_names(call, 10 + call // 4, call % 4)
                 for call in range(120)]
        self.assertEqual(len(set(item[0] for item in names)), 120)
        self.assertEqual(len(set(item[1] for item in names)), 120)
        with self.assertRaises(M.M1328Error):
            M.output_plane_names(True, 10, 0)

    def test_05_two_plane_copy_is_exact_and_exclusive(self):
        with tempfile.TemporaryDirectory(prefix="m1328_plane_") as directory:
            root = Path(directory)
            positive = bytes([0xA5, 0x01])
            negative = bytes([0x00, 0x00])
            source = root / "two_planes.bin"
            source.write_bytes(positive + negative)
            out_positive = root / "positive.bin"
            out_negative = root / "negative.bin"
            M.copy_plane_exclusive(source, 0, 2, out_positive, sha_bytes(positive))
            M.copy_plane_exclusive(source, 2, 2, out_negative, sha_bytes(negative))
            self.assertEqual(out_positive.read_bytes(), positive)
            self.assertEqual(out_negative.read_bytes(), negative)
            with self.assertRaises(FileExistsError):
                M.copy_plane_exclusive(source, 0, 2, out_positive, sha_bytes(positive))
            bad = root / "bad.bin"
            with self.assertRaisesRegex(M.M1328Error, "SHA"):
                M.copy_plane_exclusive(source, 0, 2, bad, "f" * 64)

    def test_06_attempt_is_o_excl_and_no_retry(self):
        with tempfile.TemporaryDirectory(prefix="m1328_attempt_") as directory:
            attempt = Path(directory) / "attempt"
            with mock.patch.object(M, "ATTEMPT", attempt):
                M.consume_attempt()
                self.assertEqual(attempt.read_text(), M.ATTEMPT_TOKEN)
                with self.assertRaises(FileExistsError):
                    M.consume_attempt()

    def test_07_recursive_atomic_seal_detects_mutation(self):
        with tempfile.TemporaryDirectory(prefix="m1328_seal_") as directory:
            root = Path(directory) / "staging"
            (root / "payloads").mkdir(parents=True)
            for index in range(240):
                M.write_exclusive(root / "payloads" / ("p%03d.bin" % index), b"x")
            materialized = {"population": {"calls": 120},
                            "records": [{} for _ in range(120)]}
            M.write_exclusive(root / "manifest.json", json.dumps(materialized).encode())
            M.write_exclusive(root / "RUN_COMPLETE.txt", b"PASS\n")
            receipt = M.seal_staging(root)
            self.assertEqual(receipt["members"], 242)
            (root / "payloads/p000.bin").chmod(0o600)
            (root / "payloads/p000.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(M.M1328Error, "drift"):
                M.verify_materialized_seal(root)

    def test_08_release_shape_binds_future_only_and_claim_boundary(self):
        release = {
            "schema": M.RELEASE_SCHEMA, "status": M.RELEASE_STATUS,
            "contract_path": str(M.FUTURE_RELEASE.relative_to(M.ROOT)),
            "release_identity": {
                "source_path": str(M.SOURCE_FILE.relative_to(M.ROOT)),
                "source_sha256": M.sha256(M.SOURCE_FILE),
                "test_path": str(M.TEST.relative_to(M.ROOT)), "test_sha256": M.sha256(M.TEST),
                "source_contract_path": str(M.SOURCE_CONTRACT.relative_to(M.ROOT)),
                "source_contract_sha256": M.sha256(M.SOURCE_CONTRACT)},
            "source_hammer": {"path": "future", "manifest_sha256": "1" * 64,
                              "outer_file_sha256": "2" * 64, "review_sha256": "3" * 64},
            "capture_result": synthetic_authority()["release"]["capture_result"],
            "capture_result_hammer": synthetic_authority()["release"]["capture_result_hammer"],
            "one_shot": {"attempt_marker": str(M.ATTEMPT.relative_to(M.ROOT)),
                         "automatic_retry": False, "maximum_materializations": 1},
            "output": {"path": str(M.OUTPUT.relative_to(M.ROOT)),
                       "atomic_no_replace": True, "recursive_double_seal": True},
            "claim_boundary": {"bitplane_materialization": True,
                               "decoder_replay": False, "cycles": False,
                               "traffic": False, "speedup": False,
                               "system_speedup": False, "energy": False,
                               "rtl": False, "eda": False, "ppa": False},
        }
        M.validate_release_shape(release, M.FUTURE_RELEASE)
        changed = copy.deepcopy(release); changed["one_shot"]["automatic_retry"] = True
        with self.assertRaisesRegex(M.M1328Error, "one-shot"):
            M.validate_release_shape(changed, M.FUTURE_RELEASE)

    def test_09_source_policy_has_no_fabricated_result_sha(self):
        policy = M.strict_json(M.SOURCE_CONTRACT)
        self.assertEqual(policy["actual_m1327_result"], {
            "present": False, "sha256_predeclared": False,
            "result_hammer_present": False})
        self.assertNotIn("capture_result_sha256", policy)
        self.assertFalse(policy["production_authorized"])

    def test_10_cli_is_inert_and_no_replay_gpu_eda(self):
        with self.assertRaises(M.M1328Error):
            M.main([])
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("--source-self-check", source)
        self.assertNotIn("--materialize", source)
        self.assertNotIn("exclusive_gpu_lease", source)
        self.assertNotIn("dc_shell", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
