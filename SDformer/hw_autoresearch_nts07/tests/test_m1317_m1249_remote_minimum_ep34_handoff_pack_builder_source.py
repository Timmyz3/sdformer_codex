from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "hw_autoresearch_nts07/system_handoff/scripts/"
    "build_m1317_m1249_remote_minimum_ep34_handoff_pack_source_only.py")
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m1317_m1249_remote_minimum_ep34_handoff_pack_builder_source_contract_r1_20260831.json")


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


M = load("m1317_pack_builder_under_test", SOURCE)


class M1317PackBuilderSourceTest(unittest.TestCase):
    def test_01_frozen_inventory_is_exact_and_closes_38_missing_files(self):
        self.assertEqual(sha(M.INVENTORY), M.INVENTORY_SHA256)
        rows = M.validate_inventory()
        self.assertEqual(len(rows), 38)
        self.assertEqual(len({row["path"] for row in rows}), 38)
        self.assertEqual(sum(row["size_bytes"] for row in rows), 223543)

    def test_02_base_payload_is_self_contained_before_release_and_manifest(self):
        rows = M.base_payload()
        self.assertEqual(len(rows), 60)
        self.assertEqual(len({row["path"] for row in rows}), 60)
        inventory = {row["path"] for row in M.validate_inventory()}
        additions = {row["path"] for row in rows} - inventory
        self.assertEqual(len(additions), 22)
        self.assertIn(M.M1313["contract"]["path"], additions)
        self.assertIn(M.M1313["checker"]["path"], additions)
        self.assertIn(M.M1313["test"]["path"], additions)
        self.assertIn(str(M.REMOTE_PREFLIGHT.relative_to(M.ROOT)), additions)
        self.assertIn(str(M.VERIFIER.relative_to(M.ROOT)), additions)
        self.assertEqual(sum(row["role"] == "M1314_independent_hammer" for row in rows), 10)

    def test_03_all_payload_members_are_regular_nonsymlink_and_exact_sha(self):
        for row in M.base_payload():
            with self.subTest(path=row["path"]):
                path = ROOT / row["path"]
                self.assertTrue(path.is_file())
                self.assertFalse(path.is_symlink())
                self.assertEqual(path.stat().st_size, row["size_bytes"])
                self.assertEqual(sha(path), row["sha256"])

    def test_04_path_traversal_absolute_noncanonical_and_backslash_are_rejected(self):
        attacks = ("../x", "/absolute", "a/../b", "a/./b", "a//b", "a\\b", "")
        for value in attacks:
            with self.subTest(value=value), self.assertRaises(M.M1317Error):
                M.safe_relative(value)

    def test_05_duplicate_inventory_path_fails_closed(self):
        inventory = M.strict_json(M.INVENTORY)
        attack = copy.deepcopy(inventory)
        attack["transfer_required"]["singleton_files"][1]["path"] = (
            attack["transfer_required"]["singleton_files"][0]["path"])
        attack["transfer_required"]["singleton_files"][1]["sha256"] = (
            attack["transfer_required"]["singleton_files"][0]["sha256"])
        attack["transfer_required"]["singleton_files"][1]["size_bytes"] = (
            attack["transfer_required"]["singleton_files"][0]["size_bytes"])
        with self.assertRaises(M.M1317Error):
            M.validate_inventory(attack)

    def test_06_inventory_sha_and_size_mutations_fail_closed(self):
        for key, value in (("sha256", "0" * 64), ("size_bytes", 1)):
            inventory = M.strict_json(M.INVENTORY)
            inventory["transfer_required"]["singleton_files"][0][key] = value
            with self.subTest(key=key), self.assertRaises(M.M1317Error):
                M.validate_inventory(inventory)

    def test_07_symlink_is_not_a_regular_payload_file(self):
        with tempfile.TemporaryDirectory(prefix="m1317_regular_") as name:
            root = Path(name)
            target = root / "target"
            link = root / "link"
            target.write_bytes(b"exact\n")
            link.symlink_to(target)
            with self.assertRaises(M.M1317Error):
                M.regular_exact(link, sha(target), target.stat().st_size, "symlink attack")

    def test_08_m1313_contract_checker_test_and_author_receipt_are_exact(self):
        rows = M.validate_m1313()
        self.assertEqual(len(rows), 10)
        receipt = M.M1313["author_receipt"]
        self.assertEqual(sha(ROOT / receipt["path"] / "SHA256SUMS"),
                         receipt["manifest_sha256"])
        self.assertEqual(sha(ROOT / receipt["path"] / "SHA256SUMS.seal.sha256"),
                         receipt["outer_file_sha256"])
        self.assertEqual(sha(ROOT / receipt["path"] / "author_receipt.json"),
                         receipt["author_receipt_sha256"])

    def test_09_future_m1314_shape_is_explicit_and_unhammered_release_is_rejected(self):
        release = {
            "schema": M.RELEASE_SCHEMA,
            "status": M.RELEASE_STATUS,
            "contract_path": str(M.RELEASE_PATH.relative_to(M.ROOT)),
            "builder_identity": M.expected_builder_identity(),
            "inventory": {"path": str(M.INVENTORY.relative_to(M.ROOT)),
                          "sha256": M.INVENTORY_SHA256},
            "m1313": copy.deepcopy(M.M1313),
            "m1314_hammer": {},
            "verifier": {"path": str(M.VERIFIER.relative_to(M.ROOT)),
                         "sha256": M.VERIFIER_SHA256},
            "payload_manifest": {"path": str(M.PAYLOAD_MANIFEST.relative_to(M.ROOT))},
            "one_shot": {"attempt_marker": str(M.ATTEMPT.relative_to(M.ROOT)),
                         "automatic_retry": False},
            "output": {"path": str(M.OUTPUT.relative_to(M.ROOT)),
                       "format": "deterministic_posix_tar"},
        }
        with self.assertRaises(M.M1317Error):
            M.validate_release(release, M.RELEASE_PATH)
        self.assertIn("m1314", M.M1314_SCHEMA)
        self.assertEqual(M.M1314_AUTHORIZATION["remote_capture_runs"], 1)
        self.assertFalse(M.M1314_AUTHORIZATION["automatic_retry"])

    def test_10_no_pack_no_attempt_no_partial_before_builder_execution(self):
        for path in (M.OUTPUT, M.ATTEMPT, M.PARTIAL):
            self.assertFalse(os.path.lexists(str(path)))

    def test_11_source_contract_is_source_only_and_forbids_pack_transfer_gpu(self):
        value = M.strict_json(CONTRACT)
        self.assertEqual(value["schema"], M.SOURCE_SCHEMA)
        self.assertEqual(value["status"], M.SOURCE_STATUS)
        self.assertFalse(value["production_release_created"])
        self.assertTrue(value["future_M1314_hammer_required"])
        self.assertEqual(value["author_execution"], {
            "pack_created": False,
            "remote_transfer": False,
            "remote_mutation": False,
            "gpu": False,
            "capture": False,
            "eda": False,
        })

    def test_12_builder_has_no_network_gpu_capture_or_eda_execution(self):
        text = SOURCE.read_text(encoding="utf-8")
        for forbidden in ("import subprocess", "paramiko", "import torch", "rsync ",
                          "scp ", "ssh ", "dc_shell", "vcs -full64"):
            self.assertNotIn(forbidden, text)
        self.assertIn("tarfile.open", text)
        self.assertIn("os.O_EXCL", text)
        self.assertIn("automatic_retry", text)

    def test_12b_remote_preflight_only_validates_and_keeps_canonical_log_fresh(self):
        text = M.REMOTE_PREFLIGHT.read_text(encoding="utf-8")
        self.assertIn("validate_production_launch", text)
        self.assertIn("CANONICAL_LOG", text)
        self.assertIn("canonical_log_fresh", text)
        for forbidden in (".execute_once(", ".run_capture(", "import subprocess", "paramiko",
                          "import torch", "write_text", "write_bytes", "open(\"w\""):
            self.assertNotIn(forbidden, text)

    def test_13_remote_existing_checkpoint_profile_cohort_are_not_in_payload(self):
        paths = {row["path"] for row in M.base_payload()}
        self.assertFalse(any(path.endswith("checkpoint_epoch34.pth") for path in paths))
        self.assertFalse(any(path.endswith("spike_profile.json") for path in paths))
        self.assertFalse(any(path.endswith(".npy") for path in paths))
        self.assertNotIn(
            "hw_autoresearch_nts07/results/m1257_motion_cross_run_final_checkpoint_selection_r5_20260830/final_checkpoint_selection.json",
            paths)

    def test_14_builder_import_is_inert_and_main_requires_release(self):
        self.assertFalse(M.OUTPUT.exists())
        self.assertEqual(M.ATTEMPT_TOKEN,
                         "M1317_PACK_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n")


if __name__ == "__main__":
    unittest.main(verbosity=2)
