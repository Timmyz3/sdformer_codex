#!/usr/bin/env python3
"""Controlled static/unit tests for M1174; never calls production main."""
from __future__ import annotations

import ast
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1174_motion_checkpoint_parametric_unified_hardware.py")
CONTRACT = ROOT / (
    "hw_autoresearch_nts07/contracts/"
    "m1174_motion_checkpoint_parametric_unified_capture_source_contract_r1_20260830.json")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


SPEC = importlib.util.spec_from_file_location("m1174_source_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1174SourceTests(unittest.TestCase):
    def test_contract_is_source_only_and_identity_bound(self) -> None:
        data = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(data["source"]["sha256"], digest(SOURCE))
        self.assertFalse(data["production_authorization"]["authorized_by_this_contract"])
        self.assertEqual(data["fixed_cohort"]["total_forward_samples"], 40)
        self.assertEqual(set(data["unified_capture"]["required_categories"]), M.CATEGORIES)
        self.assertEqual(digest(ROOT / data["protected_file"]["path"]),
                         data["protected_file"]["sha256"])

    def test_source_contract_fails_before_any_production_import(self) -> None:
        data = json.loads(CONTRACT.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(M.CaptureError, "not a production launch authority"):
            M.validate_launch_contract(data, CONTRACT)

    def test_ast_has_one_model_load_and_no_process_launch(self) -> None:
        tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
        build_calls = [node for node in ast.walk(tree)
                       if isinstance(node, ast.Call) and
                       isinstance(node.func, ast.Attribute) and
                       node.func.attr == "build_model"]
        self.assertEqual(len(build_calls), 1)
        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in {"run", "Popen", "call", "check_call", "check_output"}:
                    forbidden.append(node.func.attr)
        self.assertEqual(forbidden, [])

    def test_legacy_watcher_detection_includes_stopped_state(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            pid = root / "1234"
            pid.mkdir()
            (pid / "cmdline").write_bytes(
                b"python\x00capture_m511_h67_convtranspose_binary_inputs.py\x00")
            fields = ["1234", "(python)", "T"] + ["0"] * 20
            (pid / "stat").write_text(" ".join(fields), encoding="utf-8")
            found = M.running_legacy_watchers(root)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0]["state"], "T")

    def test_shared_lease_is_nonblocking(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            lock = Path(name) / "gpu.lock"
            first = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(first, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                with self.assertRaisesRegex(M.CaptureError, "lease is busy"):
                    with M.exclusive_gpu_lease(lock):
                        self.fail("second lease unexpectedly entered")
            finally:
                fcntl.flock(first, fcntl.LOCK_UN)
                os.close(first)

    def test_strict_json_rejects_duplicate_and_nonfinite(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "bad.json"
            path.write_text('{"a":1,"a":2}', encoding="utf-8")
            with self.assertRaisesRegex(M.CaptureError, "duplicate JSON key"):
                M.strict_json(path)
            path.write_text('{"a":NaN}', encoding="utf-8")
            with self.assertRaisesRegex(M.CaptureError, "non-standard JSON"):
                M.strict_json(path)

    def test_category_classifier_covers_declared_objects(self) -> None:
        class Dummy:
            pass
        writer = object.__new__(M.UnifiedHookWriter)
        cases = {
            M.C1_TARGETS[0]: "c1_conv3x3",
            "sttmultires_unet.decoders.0.0": "decoder_convtranspose",
            "x.atlif": "atlif",
            "x.fc1": "fc1",
            "x.fc2": "fc2",
            "x.patch_embed.proj": "patch_embed",
            "x.bn1": "batch_norm",
            "x.attn.q": "qkv",
            "x.attn": "attention",
        }
        classes = {
            "decoder_convtranspose": "ConvTranspose2d",
            "atlif": "ATLIFTernaryPSN",
            "batch_norm": "BatchNorm2d",
            "attention": "ShiftmaxAttention",
        }
        for module_name, expected in cases.items():
            cls_name = classes.get(expected, "Linear")
            module = type(cls_name, (), {})()
            self.assertEqual(writer._category(module_name, module), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
