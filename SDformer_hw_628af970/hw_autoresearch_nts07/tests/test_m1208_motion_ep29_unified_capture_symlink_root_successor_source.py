#!/usr/bin/env python3
"""Controlled M1208 source tests; never remote, GPU, capture, or EDA."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1208_motion_ep29_unified_hardware_symlink_root_successor_r1.py")
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m1208_motion_ep29_unified_capture_symlink_root_successor_source_contract_r1_20260830.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


SPEC = importlib.util.spec_from_file_location("m1208_capture_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1208SourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def make_tree(self, root: Path, payload: bytes = b"exact") -> tuple[Path, Path, str]:
        target = root / "target"
        sample = target / "saved_flow_data/event_tensors/10bins/left/seq/sample.npy"
        sample.parent.mkdir(parents=True)
        sample.write_bytes(payload)
        (root / "repo/data/Datasets").mkdir(parents=True)
        (root / "repo/data/Datasets/DSEC").symlink_to(target)
        return root / "repo", target, digest(sample)

    def resolve(self, repo: Path, target: Path, sha: str, size: int = 5) -> Path:
        return M._resolve_whitelisted_sample(
            "data/Datasets/DSEC/saved_flow_data/event_tensors/10bins/left/seq/sample.npy",
            size, sha, repo_root=repo, pinned_root=target)

    def test_01_source_policy_exact_and_inert(self) -> None:
        self.assertEqual(self.policy["schema"], M.SOURCE_SCHEMA)
        self.assertEqual(self.policy["status"], M.SOURCE_STATUS)
        self.assertEqual(self.policy["source"]["sha256"], digest(SOURCE))
        self.assertEqual(self.policy["test_sha256"], digest(Path(__file__).resolve()))
        self.assertFalse(self.policy["claim_boundary"]["production_authorized"])

    def test_02_docs359_and_predecessor_are_pinned(self) -> None:
        frozen = ROOT / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"
        self.assertEqual(digest(frozen), self.policy["docs359_sha256"])
        self.assertEqual(digest(M.PREDECESSOR_PATH), M.PREDECESSOR_SHA256)
        self.assertEqual(digest(M.M1180_LAUNCH_CONTRACT), M.M1180_LAUNCH_SHA256)

    def test_03_original_resolver_rejects_whitelisted_link(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            repo, _, _ = self.make_tree(Path(name))
            with mock.patch.object(M.R1, "ROOT", repo):
                with self.assertRaisesRegex(M.R1.CaptureError, "symlink component rejected"):
                    M.R1.repo_path(
                        "data/Datasets/DSEC/saved_flow_data/event_tensors/10bins/left/seq/sample.npy")

    def test_04_successor_accepts_only_exact_root_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            repo, target, sha = self.make_tree(Path(name))
            self.assertEqual(self.resolve(repo, target, sha),
                             target / "saved_flow_data/event_tensors/10bins/left/seq/sample.npy")

    def test_05_raw_target_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            repo, target, sha = self.make_tree(Path(name))
            with self.assertRaisesRegex(M.M1208Error, "raw symlink target drift"):
                self.resolve(repo, target.parent / "wrong", sha)

    def test_06_escape_and_noncanonical_prefix_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            repo, target, sha = self.make_tree(Path(name))
            with self.assertRaisesRegex(M.M1208Error, "below exact pinned"):
                M._resolve_whitelisted_sample("data/Datasets/OTHER/sample.npy", 5, sha,
                                              repo_root=repo, pinned_root=target)
            with self.assertRaisesRegex(M.M1208Error, "below exact pinned"):
                M._resolve_whitelisted_sample("data/Datasets/DSEC/../escape.npy", 5, sha,
                                              repo_root=repo, pinned_root=target)

    def test_07_nested_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            repo, target, sha = self.make_tree(Path(name))
            real = target / "saved_flow_data/event_tensors/10bins/left/seq"
            moved = target / "real_seq"
            real.rename(moved)
            real.symlink_to(moved, target_is_directory=True)
            with self.assertRaisesRegex(M.M1208Error, "non-whitelisted symlink"):
                self.resolve(repo, target, sha)

    def test_08_leaf_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            repo, target, sha = self.make_tree(Path(name))
            leaf = target / "saved_flow_data/event_tensors/10bins/left/seq/sample.npy"
            real = leaf.with_name("real.npy")
            leaf.rename(real)
            leaf.symlink_to(real)
            with self.assertRaisesRegex(M.M1208Error, "non-whitelisted symlink"):
                self.resolve(repo, target, sha)

    def test_09_hash_and_size_drift_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            repo, target, sha = self.make_tree(Path(name))
            with self.assertRaisesRegex(M.M1208Error, "identity drift"):
                self.resolve(repo, target, "0" * 64)
            with self.assertRaisesRegex(M.M1208Error, "identity drift"):
                self.resolve(repo, target, sha, size=6)

    def test_10_override_is_restored_on_failure(self) -> None:
        original = M.R1.selected_samples
        def fail(_contract: object, _binding: object) -> Path:
            self.assertIs(M.R1.selected_samples, M.selected_samples)
            raise RuntimeError("controlled")
        with mock.patch.object(M, "frozen_inventory", return_value={}), \
             mock.patch.object(M.R1, "run_capture", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "controlled"):
                M.run_capture({"r1_compatible_binding": {}}, {"policy": {}})
        self.assertIs(M.R1.selected_samples, original)

    def test_11_source_contract_cannot_launch(self) -> None:
        with self.assertRaisesRegex(M.M1208Error, "cannot launch"):
            M.validate_launch_contract(self.policy, CONTRACT)

    def test_12_namespace_is_disjoint_and_failure_is_pinned(self) -> None:
        future = self.policy["future_launch_contract"]
        self.assertIn("m1208", future["canonical_attempt_marker"])
        self.assertIn("m1208", future["canonical_result"])
        self.assertIn("m1208", future["canonical_production_log"])
        self.assertNotEqual(M.CANONICAL_ATTEMPT, M.M1180_ATTEMPT)
        self.assertEqual(self.policy["prior_m1180_failure"]["attempt_token"],
                         "M1180_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
