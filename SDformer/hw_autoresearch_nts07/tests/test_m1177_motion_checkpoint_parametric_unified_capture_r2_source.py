#!/usr/bin/env python3
"""M1177 r2 controlled static/mutation tests; production is never called."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1177_motion_checkpoint_parametric_unified_hardware_r2.py")
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m1177_motion_checkpoint_parametric_unified_capture_source_contract_r2_20260830.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


SPEC = importlib.util.spec_from_file_location("m1177_r2_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class Handle:
    def remove(self) -> None:
        pass


def hookable(class_name: str):
    def register_forward_hook(self, hook):
        self.hook = hook
        return Handle()
    return type(class_name, (), {"register_forward_hook": register_forward_hook})()


class M1177R2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_source_contract_is_source_only_and_exact(self) -> None:
        self.assertEqual(self.policy["source"]["sha256"], digest(SOURCE))
        self.assertEqual(self.policy["status"],
                         "SOURCE_ONLY__R2_HAMMER_AND_RELEASE_REQUIRED__NO_GPU")
        self.assertFalse(self.policy["claim_boundary"]["production_authorized"])
        self.assertEqual(digest(ROOT / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"),
                         self.policy["docs359_sha256"])

    def test_m1175_exact_semantics_pass_and_schema_mutation_fails(self) -> None:
        review = M.validate_m1175()
        self.assertEqual(review["selection"]["epoch"], 29)
        with tempfile.TemporaryDirectory() as name:
            target = Path(name) / "m1175"
            shutil.copytree(M.M1175, target)
            data = json.loads((target / "review.json").read_text(encoding="utf-8"))
            data["schema"] = "attacker_schema"
            (target / "review.json").write_text(json.dumps(data) + "\n", encoding="utf-8")
            M.canonical_write_double_seal(target)
            with mock.patch.object(M, "M1175", target), \
                 mock.patch.object(M, "M1175_REVIEW_SHA256", digest(target / "review.json")), \
                 mock.patch.object(M, "M1175_MANIFEST_SHA256", digest(target / "SHA256SUMS")), \
                 mock.patch.object(M, "M1175_OUTER_FILE_SHA256", digest(target / "SHA256SUMS.seal.sha256")):
                with self.assertRaisesRegex(M.R2Error, "semantic admission"):
                    M.validate_m1175()

    def test_future_r2_hammer_status_mutation_fails(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            hw = root / "hw_autoresearch_nts07"
            hammer = hw / "reviews/fake_m1178"
            hammer.mkdir(parents=True)
            review = {
                "schema": "m1178_m1177_motion_unified_capture_source_hammer_r1_v1",
                "status": "FAIL",
                "source_sha256": digest(SOURCE),
                "contract_sha256": digest(CONTRACT),
                "test_sha256": self.policy["test_sha256"],
                "authorization": {"production_release": True},
            }
            (hammer / "review.json").write_text(json.dumps(review) + "\n", encoding="utf-8")
            M.canonical_write_double_seal(hammer)
            launch = {"inputs": {"m1177_source_hammer": {
                "path": "hw_autoresearch_nts07/reviews/fake_m1178",
                "manifest_sha256": digest(hammer / "SHA256SUMS"),
                "outer_file_sha256": digest(hammer / "SHA256SUMS.seal.sha256"),
                "review_sha256": digest(hammer / "review.json"),
            }}}
            with mock.patch.object(M, "ROOT", root), mock.patch.object(M, "HW", hw):
                with self.assertRaisesRegex(M.R2Error, "semantic admission"):
                    M.validate_r2_hammer(launch, self.policy)

    def test_canonical_lease_is_literal_and_redirect_not_consumed(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        main = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        text = ast.get_source_segment(SOURCE.read_text(encoding="utf-8"), main) or ""
        self.assertIn("exclusive_gpu_lease(CANONICAL_LEASE)", text)
        self.assertNotIn('gpu_ownership"]["lease_path', text)

    def test_exact_forty_sources_pass_duplicate_path_sha_and_order_mutations_fail(self) -> None:
        launch = {"cohort": {"samples": copy.deepcopy(self.policy["frozen_samples"])}}
        rows = M.validate_fixed_samples(launch, self.policy)
        self.assertEqual(len(rows), 40)
        for mutation in ("duplicate", "order", "cohort", "path", "sha"):
            bad = copy.deepcopy(launch)
            if mutation == "duplicate":
                bad["cohort"]["samples"][1] = copy.deepcopy(bad["cohort"]["samples"][0])
                bad["cohort"]["samples"][1]["global_sample_id"] = 1
            elif mutation == "order":
                bad["cohort"]["samples"][10], bad["cohort"]["samples"][11] = (
                    bad["cohort"]["samples"][11], bad["cohort"]["samples"][10])
            elif mutation == "cohort":
                bad["cohort"]["samples"][10]["cohort"] = "c1"
            elif mutation == "path":
                bad["cohort"]["samples"][0]["path"] = bad["cohort"]["samples"][1]["path"]
            else:
                bad["cohort"]["samples"][0]["sha256"] = "0" * 64
            with self.assertRaises(M.R2Error, msg=mutation):
                M.validate_fixed_samples(bad, self.policy)

    def test_frozen_inventory_counts_and_digests(self) -> None:
        inventory = M.frozen_inventory(self.policy)
        self.assertEqual({key: len(value) for key, value in inventory.items()}, {
            "c1_conv3x3": 4, "decoder_convtranspose": 4, "fc1": 12,
            "fc2": 12, "qkv": 24, "patch_embed": 8,
            "batch_norm": 78, "attention": 12,
        })

    def test_missing_one_c1_or_decoder_module_rejected(self) -> None:
        inventory = M.frozen_inventory(self.policy)
        M.StrictWriter.EXPECTED = inventory
        names = []
        for category, members in inventory.items():
            for name in members:
                if name in {M.C1_TARGETS[-1], M.DECODER_TARGETS[-1]}:
                    continue
                cls = {
                    "decoder_convtranspose": "ConvTranspose2d",
                    "batch_norm": "BatchNorm2d",
                    "attention": "ShiftmaxAttention",
                }.get(category, "Linear")
                names.append((name, hookable(cls)))
        for index in range(105):
            names.append(("atlif.{:03d}".format(index), hookable("ATLIFTernaryPSN")))
        model = type("Model", (), {"named_modules": lambda self: iter(names)})()
        writer = M.StrictWriter(object(), Path("/nonproduction"), {})
        with self.assertRaisesRegex(M.R2Error, "missing expected module"):
            writer.attach(model)

    def test_zero_and_partial_per_module_records_rejected(self) -> None:
        writer = object.__new__(M.StrictWriter)
        writer.EXPECTED = {category: [category + ".0"] for category in M.CATEGORIES}
        writer.records = []
        writer.handles = []
        writer._r2_attached = True
        with self.assertRaisesRegex(M.R2Error, "call coverage"):
            writer.close()
        writer.records = [{"category": category, "name": category + ".0"}
                          for category in M.CATEGORIES]
        with self.assertRaisesRegex(M.R2Error, "call coverage"):
            writer.close()

    def test_zero_partial_and_wrong_attention_cartesian_rejected(self) -> None:
        writer = object.__new__(M.StrictAttentionWriter)
        writer.records = []
        with self.assertRaisesRegex(M.R2Error, "40x12"):
            writer._assert_complete()
        writer.records = [{"sample_id": 0, "name": M.ATTENTION_ALIASES[0]}] * 480
        with self.assertRaisesRegex(M.R2Error, "Cartesian"):
            writer._assert_complete()

    def test_recursive_seal_nested_passes_and_tamper_partial_symlink_fail(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "payloads/deep").mkdir(parents=True)
            payload = root / "payloads/deep/item.bin"
            payload.write_bytes(b"payload")
            M.canonical_write_double_seal(root)
            rows = M.canonical_verify_double_seal(root)
            self.assertEqual(set(rows), {"payloads/deep/item.bin"})
            payload.write_bytes(b"tamper")
            with self.assertRaisesRegex(M.R2Error, "payload mismatch"):
                M.canonical_verify_double_seal(root)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "a").write_bytes(b"a")
            M.canonical_write_double_seal(root)
            (root / "a").unlink()
            with self.assertRaises(M.R2Error):
                M.canonical_verify_double_seal(root)
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "real").write_bytes(b"x")
            (root / "link").symlink_to(root / "real")
            with self.assertRaisesRegex(M.R2Error, "symlink"):
                M.canonical_write_double_seal(root)

    def test_source_only_contract_rejected_before_hammer_or_gpu(self) -> None:
        with self.assertRaisesRegex(M.R2Error, "cannot launch"):
            M.validate_launch_contract(self.policy, CONTRACT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
