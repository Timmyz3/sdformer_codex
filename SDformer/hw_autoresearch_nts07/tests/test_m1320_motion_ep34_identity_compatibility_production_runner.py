#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1320_motion_ep34_identity_compatibility_production_runner_r1.py")


def load_source():
    spec = importlib.util.spec_from_file_location("test_m1320_runner", str(SOURCE))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_source()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M1320RunnerTests(unittest.TestCase):
    def test_01_exact_sealed_hammer(self):
        review = M.verify_m1320_hammer(M.M1320_HAMMER_ENTRY)
        self.assertEqual(review["independence"], {"different_author": True})
        self.assertFalse(any(review["hammer_execution"].values()))

    def test_02_source_contract_and_release_exact(self):
        policy = M.validate_source_contract()
        release = M.validate_release_static()
        self.assertFalse(policy["production_authorized_by_source_contract"])
        self.assertEqual(release["authorized_actor"], "root_agent")
        self.assertFalse(release["one_shot"]["automatic_retry"])

    def test_03_preflight_is_read_only_and_never_executes(self):
        release = {"status": M.RELEASE_STATUS}
        binding = {"identity": {
            "m1319_projection":
            "extended7_verified_then_frozen_keyset_temporarily_extended"}}
        with mock.patch.object(M, "validate_release_static", return_value=release), \
             mock.patch.object(M, "ensure_fresh_namespaces") as fresh, \
             mock.patch.object(M.M1319, "validate_exact_m1313_m1314",
                               return_value=({}, binding)) as validate, \
             mock.patch.object(M.M1319, "execute_once") as execute:
            result = M.read_only_preflight()
        self.assertTrue(result["namespaces_fresh"])
        self.assertEqual(fresh.call_count, 2)
        validate.assert_called_once_with(M.M1313_CONTRACT, M.M1314_ENTRY)
        execute.assert_not_called()

    def test_04_non_root_execution_rejected_before_temp_creation(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            canonical = root / "production.log"
            temp = root / "production.log.tmp.test"
            with mock.patch.object(M, "CANONICAL_LOG", canonical), \
                 mock.patch.object(M.os, "geteuid", return_value=1000), \
                 self.assertRaisesRegex(M.M1320Error, "root_agent"):
                M.execute_production_once(temp)
            self.assertFalse(temp.exists())

    def test_05_atomic_no_replace_publish_success(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            canonical = root / "production.log"
            temp = root / "production.log.tmp.unique"
            temp.write_bytes(b"sealed capture log\n")
            inode = temp.stat().st_ino
            with mock.patch.object(M, "CANONICAL_LOG", canonical):
                M.publish_temp_log_no_replace(temp)
            self.assertFalse(temp.exists())
            self.assertEqual(canonical.read_bytes(), b"sealed capture log\n")
            self.assertEqual(canonical.stat().st_ino, inode)

    def test_06_occupied_canonical_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            canonical = root / "production.log"
            temp = root / "production.log.tmp.unique"
            canonical.write_bytes(b"original\n")
            temp.write_bytes(b"new\n")
            with mock.patch.object(M, "CANONICAL_LOG", canonical), \
                 self.assertRaisesRegex(M.M1320Error, "already exists"):
                M.publish_temp_log_no_replace(temp)
            self.assertEqual(canonical.read_bytes(), b"original\n")
            self.assertEqual(temp.read_bytes(), b"new\n")

    def test_07_symlink_temp_and_wrong_directory_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            canonical = root / "production.log"
            target = root / "target"
            target.write_bytes(b"target")
            link = root / "production.log.tmp.link"
            link.symlink_to(target)
            with mock.patch.object(M, "CANONICAL_LOG", canonical), \
                 self.assertRaisesRegex(M.M1320Error, "regular non-symlink"):
                M.publish_temp_log_no_replace(link)
            other = root / "sub"
            other.mkdir()
            wrong = other / "production.log.tmp.wrong"
            wrong.write_bytes(b"wrong")
            with mock.patch.object(M, "CANONICAL_LOG", canonical), \
                 self.assertRaisesRegex(M.M1320Error, "share canonical"):
                M.publish_temp_log_no_replace(wrong)

    def test_08_source_cli_has_no_implicit_execution_mode(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("mutually_exclusive_group(required=True)", source)
        self.assertIn("--preflight", source)
        self.assertIn("--temporary-log", source)
        self.assertNotIn("automatic_retry = True", source)
        self.assertEqual(digest(M.DOCS359), M.DOCS359_SHA256)


if __name__ == "__main__":
    unittest.main()
