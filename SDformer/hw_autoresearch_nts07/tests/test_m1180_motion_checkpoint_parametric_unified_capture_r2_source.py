#!/usr/bin/env python3
"""M1180 namespace and inherited technical hardening tests; never production."""
from __future__ import annotations

import ast
import copy
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
    "capture_m1180_motion_checkpoint_parametric_unified_hardware_r2.py")
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m1180_motion_checkpoint_parametric_unified_capture_source_contract_r2_20260830.json")
OLD_SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1177_motion_checkpoint_parametric_unified_hardware_r2.py")
OLD_CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m1177_motion_checkpoint_parametric_unified_capture_source_contract_r2_20260830.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


SPEC = importlib.util.spec_from_file_location("m1180_capture_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1180CaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.technical = M.load_technical_policy(self.policy)

    def test_01_source_contract_exact_and_source_only(self) -> None:
        self.assertEqual(self.policy["source"]["sha256"], digest(SOURCE))
        self.assertEqual(self.policy["schema"], M.SOURCE_SCHEMA)
        self.assertEqual(self.policy["status"], M.SOURCE_STATUS)
        self.assertFalse(self.policy["claim_boundary"]["production_authorized"])

    def test_02_failed_m1177_capture_package_is_unchanged_and_non_authoritative(self) -> None:
        self.assertEqual(digest(OLD_SOURCE), M.SUBSTRATE_SHA256)
        self.assertEqual(digest(OLD_CONTRACT),
                         self.policy["sealed_technical_policy"]["sha256"])
        self.assertEqual(self.policy["sealed_technical_substrate"]["role"],
                         "implementation_only__never_authority")

    def test_03_docs359_is_unchanged(self) -> None:
        frozen = ROOT / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"
        self.assertEqual(digest(frozen), self.policy["docs359_sha256"])

    def test_04_m1175_exact_admission_still_passes(self) -> None:
        review = M.validate_m1175()
        self.assertEqual(review["status"], "PASS")
        self.assertEqual(review["selection"]["epoch"], 29)

    def test_05_future_m1181_fail_status_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            hw = root / "hw_autoresearch_nts07"
            hammer = hw / "reviews/m1181_fake"
            hammer.mkdir(parents=True)
            review = {
                "schema": M.HAMMER_SCHEMA, "status": "FAIL",
                "source_sha256": digest(SOURCE),
                "contract_sha256": digest(CONTRACT),
                "test_sha256": self.policy["test_sha256"],
                "authorization": {"production_release": True},
            }
            (hammer / "review.json").write_text(
                json.dumps(review) + "\n", encoding="utf-8")
            M.canonical_write_double_seal(hammer)
            launch = {"inputs": {"m1180_source_hammer": {
                "path": "hw_autoresearch_nts07/reviews/m1181_fake",
                "manifest_sha256": digest(hammer / "SHA256SUMS"),
                "outer_file_sha256": digest(hammer / "SHA256SUMS.seal.sha256"),
                "review_sha256": digest(hammer / "review.json"),
            }}}
            with mock.patch.object(M, "ROOT", root), mock.patch.object(M, "HW", hw):
                with self.assertRaisesRegex(M.M1180Error, "semantic admission"):
                    M.validate_m1181_hammer(launch, self.policy)

    def test_06_source_policy_cannot_launch(self) -> None:
        with self.assertRaisesRegex(M.M1180Error, "cannot launch"):
            M.validate_launch_contract(self.policy, CONTRACT)

    def test_07_canonical_lease_is_literal(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        body = ast.get_source_segment(source, main) or ""
        self.assertIn("exclusive_gpu_lease(CANONICAL_LEASE)", body)
        self.assertNotIn("lease_path]", body)

    def test_08_attempt_result_log_and_tokens_are_exact_m1180(self) -> None:
        future = self.policy["future_launch_contract"]
        self.assertEqual(ROOT / future["canonical_attempt_marker"], M.CANONICAL_ATTEMPT)
        self.assertEqual(ROOT / future["canonical_result"], M.CANONICAL_RESULT)
        self.assertEqual(ROOT / future["canonical_production_log"], M.CANONICAL_LOG)
        self.assertEqual(future["attempt_token"], M.ATTEMPT_TOKEN.strip())
        self.assertEqual(future["success_token"], M.PASS_TOKEN)

    def test_09_exact_forty_sources_and_mutation_rejection(self) -> None:
        launch = {"cohort": {"samples": copy.deepcopy(self.technical["frozen_samples"])}}
        self.assertEqual(len(M.validate_fixed_samples(launch, self.technical)), 40)
        launch["cohort"]["samples"][1] = copy.deepcopy(
            launch["cohort"]["samples"][0])
        launch["cohort"]["samples"][1]["global_sample_id"] = 1
        with self.assertRaises(M.BASE.R2Error):
            M.validate_fixed_samples(launch, self.technical)

    def test_10_full_inventory_counts(self) -> None:
        inventory = M.frozen_inventory(self.technical)
        self.assertEqual({key: len(value) for key, value in inventory.items()}, {
            "c1_conv3x3": 4, "decoder_convtranspose": 4, "fc1": 12,
            "fc2": 12, "qkv": 24, "patch_embed": 8,
            "batch_norm": 78, "attention": 12,
        })

    def test_11_recursive_nested_seal_and_tamper_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payload/deep").mkdir(parents=True)
            item = root / "payload/deep/item.bin"
            item.write_bytes(b"exact")
            M.canonical_write_double_seal(root)
            self.assertEqual(set(M.canonical_verify_double_seal(root)),
                             {"payload/deep/item.bin"})
            item.write_bytes(b"tamper")
            with self.assertRaises(M.BASE.R2Error):
                M.canonical_verify_double_seal(root)

    def test_12_repository_m1180_authority_namespace_is_unique(self) -> None:
        roots = [
            ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints",
            ROOT / "hw_autoresearch_nts07/contracts",
            ROOT / "hw_autoresearch_nts07/tests",
            ROOT / "hw_autoresearch_nts07/system_handoff/scripts",
        ]
        hits: list[Path] = []
        for root in roots:
            for path in root.rglob("*"):
                if path.is_file() and path.suffix in {".py", ".json", ".sh"}:
                    text = path.read_text(encoding="utf-8", errors="strict")
                    if "m1180" in text.lower():
                        hits.append(path)
                        self.assertTrue(
                            "m1180_motion_checkpoint_parametric_unified_capture" in text.lower() or
                            "m1180_motion_ep29_unified_hardware_capture" in text.lower() or
                            "m1181_m1180_motion" in text.lower(),
                            "foreign M1180 milestone namespace: {}".format(path))
        self.assertEqual({path for path in hits if "m1180" in path.name.lower()},
                         {SOURCE, CONTRACT, Path(__file__).resolve()})
        all_text = "\n".join(path.read_text(encoding="utf-8") for path in hits)
        forbidden_foreign_namespace = "m1180_motion_ep29_" + "e1e8"
        self.assertNotIn(forbidden_foreign_namespace, all_text.lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
