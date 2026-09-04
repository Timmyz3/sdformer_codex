from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tarfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
VERIFY = HW / "system_handoff/scripts/verify_m1317_m1249_remote_minimum_ep34_handoff_pack.py"
ARCHIVE = HW / "system_handoff/packs/m1317_m1249_remote_minimum_ep34_handoff_r1_20260831.tar"
MANIFEST = HW / "system_handoff/packs/m1317_m1249_remote_minimum_ep34_handoff_manifest_r1_20260831.json"
SHA_FILE = HW / "system_handoff/packs/m1317_m1249_remote_minimum_ep34_handoff_r1_20260831.tar.sha256"
ATTEMPT = HW / "system_handoff/packs/.m1317_m1249_remote_minimum_ep34_handoff_r1_20260831.attempt_consumed"
ARCHIVE_SHA = "0023918b3e3395949a8897fdd40e1fc7f4b600994f9d7401c1d3d114c16ea8ba"
MANIFEST_SHA = "22e330d662a3ffb7751d1bc23c08fc88d96c3258daf9eeb7caabdbf655e6ddaa"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


V = load("m1317_final_pack_verifier_under_test", VERIFY)


class M1317FinalPackTest(unittest.TestCase):
    def test_01_exact_pack_verifies(self):
        result = V.verify(ARCHIVE, MANIFEST, SHA_FILE, ARCHIVE_SHA, MANIFEST_SHA)
        self.assertEqual(result["files"], 62)
        self.assertFalse(result["remote_transfer_executed"])
        self.assertFalse(result["capture_executed"])

    def test_02_archive_and_manifest_sha_are_exact(self):
        self.assertEqual(sha(ARCHIVE), ARCHIVE_SHA)
        self.assertEqual(sha(MANIFEST), MANIFEST_SHA)
        self.assertEqual(SHA_FILE.read_text(encoding="ascii"),
                         ARCHIVE_SHA + "  " + ARCHIVE.name + "\n")

    def test_03_pack_is_exact_62_unique_regular_members(self):
        with tarfile.open(ARCHIVE, "r:") as handle:
            members = handle.getmembers()
        self.assertEqual(len(members), 62)
        self.assertEqual(len({row.name for row in members}), 62)
        self.assertTrue(all(row.isfile() for row in members))

    def test_04_manifest_is_self_contained_and_in_pack(self):
        value = json.loads(MANIFEST.read_text(encoding="utf-8"))
        names = {row["path"] for row in value["nonmanifest_entries"]}
        self.assertEqual(len(names), 61)
        with tarfile.open(ARCHIVE, "r:") as handle:
            archive_names = {row.name for row in handle.getmembers()}
        self.assertIn(value["manifest_member_path"], archive_names)
        self.assertTrue(any("m1314_m1313" in name and name.endswith("review.json")
                            for name in names))
        self.assertTrue(any(name.endswith("production_release_r1_20260831.json")
                            for name in names))
        self.assertTrue(any("verify_m1317" in name for name in names))

    def test_05_wrong_archive_or_manifest_sha_fails_closed(self):
        with self.assertRaises(V.VerifyError):
            V.verify(ARCHIVE, MANIFEST, SHA_FILE, "0" * 64, MANIFEST_SHA)
        with self.assertRaises(V.VerifyError):
            V.verify(ARCHIVE, MANIFEST, SHA_FILE, ARCHIVE_SHA, "0" * 64)

    def test_06_traversal_and_noncanonical_paths_fail_closed(self):
        for value in ("../x", "/x", "a/../b", "a//b", "a\\b"):
            with self.subTest(value=value), self.assertRaises(V.VerifyError):
                V.safe_relative(value)

    def test_07_attempt_is_single_use_mode_0400_and_partial_absent(self):
        self.assertEqual(stat.S_IMODE(ATTEMPT.stat().st_mode), 0o400)
        self.assertEqual(ATTEMPT.read_text(encoding="ascii"),
                         "M1317_PACK_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n")
        partial = HW / "system_handoff/packs/.m1317_m1249_remote_minimum_ep34_handoff_r1_20260831.partial"
        self.assertFalse(os.path.lexists(str(partial)))

    def test_08_pack_sidecars_are_read_only(self):
        for path in (ARCHIVE, MANIFEST, SHA_FILE):
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o444)

    def test_09_m1249_capture_namespaces_remain_fresh(self):
        paths = (
            HW / "results/m1249_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830",
            HW / "results/.m1249_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830.attempt_consumed",
            HW / "results/.m1249_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830.production.log",
        )
        self.assertTrue(all(not os.path.lexists(str(path)) for path in paths))

    def test_10_no_remote_transfer_or_capture_code_in_verifier(self):
        text = VERIFY.read_text(encoding="utf-8")
        for forbidden in ("import subprocess", "paramiko", "rsync ", "scp ", "ssh ",
                          "import torch", "execute_once", "run_capture"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
