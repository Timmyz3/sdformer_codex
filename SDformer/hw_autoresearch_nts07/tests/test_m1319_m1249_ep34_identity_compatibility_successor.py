#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1319_motion_ep34_identity_compatibility_successor_r1.py")


def load_source():
    spec = importlib.util.spec_from_file_location("test_m1319_source", str(SOURCE))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load_source()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(path: Path) -> dict:
    observed = path.lstat()
    return {
        "absolute_path": str(path.resolve()),
        "size_bytes": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
        "sha256": digest(path),
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "mode": observed.st_mode,
    }


def seal(root: Path) -> tuple[str, str]:
    members = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join(
        f"{digest(root / name)}  {name}\n" for name in members), encoding="utf-8")
    outer = root / "SHA256SUMS.seal.sha256"
    outer.write_text(f"{digest(manifest)}  SHA256SUMS\n", encoding="utf-8")
    return digest(manifest), digest(outer)


class SelectionFixture:
    def __init__(self, checkpoint_extra: bool = False, profile_extra: bool = False):
        self.result_tmp = tempfile.TemporaryDirectory(prefix="m1319_result_", dir=HW / "results")
        self.hammer_tmp = tempfile.TemporaryDirectory(prefix="m1319_hammer_", dir=HW / "reviews")
        self.files_tmp = tempfile.TemporaryDirectory(prefix="m1319_files_", dir=HW / "results")
        self.result = Path(self.result_tmp.name)
        self.hammer = Path(self.hammer_tmp.name)
        files = Path(self.files_tmp.name)
        checkpoint = files / "checkpoint.pth"
        config = files / "config.yml"
        profile_path = files / "profile.json"
        checkpoint.write_bytes(b"checkpoint bytes\n")
        config.write_bytes(b"model: test\n")
        profile_path.write_text("{}\n", encoding="utf-8")
        checkpoint_identity = identity(checkpoint)
        if checkpoint_extra:
            checkpoint_identity["unexpected"] = 1
        profile = {
            **identity(profile_path),
            "samples": 825,
            "artifact_identity_exact": True,
            "load_audit_exact_zero": True,
            "module_counts": {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
            "descriptor_rooted_no_symlink_components": True,
            "hash_and_parse_same_bytes": True,
            "immutable_single_read": True,
            "post_parse_path_identity_frozen": True,
        }
        if profile_extra:
            profile["unexpected"] = True
        selection = {
            "schema": M.FROZEN_M1233.ALLOWED_SELECTION_SCHEMA,
            "status": M.FROZEN_M1233.ALLOWED_SELECTION_STATUS,
            "selected": {
                "candidate_id": "resume_ep34", "epoch": 34,
                "run_directory": str(files),
                "checkpoint": checkpoint_identity,
                "configuration": identity(config),
                "profile": profile,
                "accuracy_metrics": {}, "activity": {},
            },
        }
        selection_path = self.result / "final_checkpoint_selection.json"
        selection_path.write_text(json.dumps(selection, sort_keys=True) + "\n", encoding="utf-8")
        selection_sha = digest(selection_path)
        selection_manifest, selection_outer = seal(self.result)
        self.selection_entry = {
            "result_path": str(self.result.relative_to(ROOT)),
            "manifest_sha256": selection_manifest,
            "outer_file_sha256": selection_outer,
            "selection_member": selection_path.name,
            "selection_sha256": selection_sha,
        }
        authority = {
            "result_path": self.selection_entry["result_path"],
            "selection_member": selection_path.name,
            "selection_sha256": selection_sha,
            "selection_manifest_sha256": selection_manifest,
            "selection_outer_file_sha256": selection_outer,
            "selection_schema": M.FROZEN_M1233.ALLOWED_SELECTION_SCHEMA,
            "selection_status": M.FROZEN_M1233.ALLOWED_SELECTION_STATUS,
            "selected_candidate_id": "resume_ep34", "selected_epoch": 34,
            "selected_profile_sha256": profile["sha256"],
            "selected_checkpoint_sha256": checkpoint_identity["sha256"],
            "selected_config_sha256": selection["selected"]["configuration"]["sha256"],
        }
        review = {
            "schema": M.FROZEN_M1233.SELECTION_RESULT_HAMMER_SCHEMA,
            "status": M.FROZEN_M1233.SELECTION_RESULT_HAMMER_STATUS,
            "selection_authority": authority,
            "independence": {"different_author": True},
            "authorization": {
                "hardware_rebind_release_authoring": True, "production_capture": False},
        }
        review_path = self.hammer / "review.json"
        review_path.write_text(json.dumps(review, sort_keys=True) + "\n", encoding="utf-8")
        hammer_manifest, hammer_outer = seal(self.hammer)
        self.hammer_entry = {
            "path": str(self.hammer.relative_to(ROOT)),
            "manifest_sha256": hammer_manifest,
            "outer_file_sha256": hammer_outer,
            "review_sha256": digest(review_path),
        }

    def close(self):
        self.hammer_tmp.cleanup()
        self.result_tmp.cleanup()
        self.files_tmp.cleanup()


class M1319Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="m1319_identity_")
        self.root = Path(self.tmp.name)
        self.file = self.root / "artifact.bin"
        self.file.write_bytes(b"identity\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_01_extended_identity_positive(self):
        self.assertEqual(M.exact_extended_identity(identity(self.file), "artifact")["sha256"],
                         digest(self.file))

    def test_02_unexpected_key_rejected(self):
        value = identity(self.file)
        value["unexpected"] = 1
        with self.assertRaisesRegex(M.M1319Error, "exactly legacy4"):
            M.exact_extended_identity(value, "artifact")

    def test_03_mode_drift_rejected(self):
        value = identity(self.file)
        self.file.chmod(0o600)
        with self.assertRaisesRegex(M.M1319Error, "identity drift"):
            M.exact_extended_identity(value, "artifact")

    def test_04_relative_path_rejected(self):
        value = identity(self.file)
        value["absolute_path"] = "relative.bin"
        with self.assertRaisesRegex(M.M1319Error, "must be absolute"):
            M.exact_extended_identity(value, "artifact")

    def test_05_sha_drift_rejected(self):
        value = identity(self.file)
        value["sha256"] = "0" * 64
        with self.assertRaisesRegex(M.M1319Error, "identity drift"):
            M.exact_extended_identity(value, "artifact")

    def test_06_symlink_leaf_rejected(self):
        link = self.root / "link.bin"
        link.symlink_to(self.file)
        value = identity(self.file)
        value["absolute_path"] = str(link)
        with self.assertRaisesRegex(M.M1319Error, "regular non-symlink"):
            M.exact_extended_identity(value, "artifact")

    def test_07_frozen_validator_positive_and_keyset_restored(self):
        fixture = SelectionFixture()
        original = M.FROZEN_M1233.IDENTITY_KEYS
        try:
            binding = M.compat_validate_final_selection(
                fixture.selection_entry, fixture.hammer_entry)
            self.assertEqual(binding["identity"]["candidate_id"], "resume_ep34")
            self.assertEqual(binding["identity"]["m1319_projection"],
                             "extended7_verified_then_frozen_keyset_temporarily_extended")
            self.assertEqual(M.FROZEN_M1233.IDENTITY_KEYS, original)
        finally:
            fixture.close()

    def test_08_checkpoint_extra_key_rejected_and_keyset_restored(self):
        fixture = SelectionFixture(checkpoint_extra=True)
        original = M.FROZEN_M1233.IDENTITY_KEYS
        try:
            with self.assertRaisesRegex(M.M1319Error, "exactly legacy4"):
                M.compat_validate_final_selection(fixture.selection_entry, fixture.hammer_entry)
            self.assertEqual(M.FROZEN_M1233.IDENTITY_KEYS, original)
        finally:
            fixture.close()

    def test_09_profile_extra_key_rejected(self):
        fixture = SelectionFixture(profile_extra=True)
        try:
            with self.assertRaisesRegex(M.M1319Error, "profile keyset"):
                M.compat_validate_final_selection(fixture.selection_entry, fixture.hammer_entry)
        finally:
            fixture.close()

    def test_10_source_policy_and_cli_are_inert(self):
        policy = M.validate_source_policy()
        self.assertFalse(policy["production_authorized"])
        completed = subprocess.run(
            [sys.executable, str(SOURCE), "--source-self-check"],
            check=False, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn(M.PASS_TOKEN, completed.stdout)

    def test_11_cli_without_source_flag_cannot_launch(self):
        completed = subprocess.run(
            [sys.executable, str(SOURCE)], check=False, capture_output=True, text=True)
        self.assertNotEqual(completed.returncode, 0)
        self.assertFalse(os.path.lexists(M.M1249.CANONICAL_ATTEMPT))


if __name__ == "__main__":
    unittest.main()
