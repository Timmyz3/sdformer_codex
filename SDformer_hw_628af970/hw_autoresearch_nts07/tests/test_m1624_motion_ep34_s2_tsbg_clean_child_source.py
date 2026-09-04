#!/usr/bin/env python3
"""Compile-free tests for the M1624 fixed clean-child source package."""

from __future__ import print_function

import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import stat
import sys
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1624_motion_ep34_s2_tsbg_clean_child_reduced_binary_"
    "successor_r1.py")
CONTRACT = HW / (
    "contracts/m1624_motion_ep34_s2_tsbg_clean_child_reduced_binary_"
    "source_contract_r1_20260901.json")
M1558 = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1558_motion_ep34_s2_tsbg_reduced_binary_source_r1.py")
M1598 = HW / (
    "reviews/m1598_m1582_m1574_tsbg_capture_permit_independent_rehammer_"
    "r1_20260901")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source():
    spec = importlib.util.spec_from_file_location("m1624_source_test", str(SOURCE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function_block(text, name, next_name):
    begin = text.index("def " + name + "(")
    end = text.index("def " + next_name + "(", begin)
    return text[begin:end]


def audit_source(text):
    tree = ast.parse(text)
    functions = dict((node.name, node) for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    assert "fixed_clean_child" in functions and "launch_parent" in functions
    assert len(functions["fixed_clean_child"].args.args) == 0
    assert len(functions["launch_parent"].args.args) == 0
    banned_arguments = {"provider", "permit", "free_space", "free_bytes",
                        "provenance", "callable", "callback", "registry"}
    for name in ("fixed_clean_child", "launch_parent"):
        assert not (set(arg.arg for arg in functions[name].args.args) & banned_arguments)
    banned_names = {"inspect", "globals", "eval", "exec"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not (set(alias.name for alias in node.names) & banned_names)
        if isinstance(node, ast.ImportFrom):
            assert node.module not in banned_names
        if isinstance(node, ast.Name):
            assert node.id not in banned_names
        if isinstance(node, ast.Attribute):
            assert node.attr not in {"__closure__", "__globals__"}

    parent = function_block(text, "launch_parent", "source_self_check")
    child = function_block(text, "fixed_clean_child", "launch_parent")
    assert parent.count("subprocess.run(") == 1
    assert "shell=True" not in parent
    assert 'command = [str(CHILD_PYTHON), "-I", str(SOURCE), "--fixed-clean-child"]' in parent
    for forbidden in ("provider", "permit", "free_space", "free_bytes",
                      "provenance", "callable", "PYTHONPATH"):
        assert forbidden not in parent
    assert 'stdin=subprocess.DEVNULL' in parent
    assert '"CUDA_VISIBLE_DEVICES": "0"' in parent

    required_child = [
        "verify_fixed_metadata(expect_future_absent=False)",
        "release = validate_future_authorities()",
        "require_fresh_namespaces()",
        "regular_exact(CHECKPOINT, CHECKPOINT_SHA256",
        "regular_exact(CONFIG, CONFIG_SHA256",
        "samples = m1434.R1.validate_cohort",
        "available = shutil.disk_usage",
        "with substrate.exclusive_gpu_lease",
        "consume_attempt(release)",
        "model = profile.build_model(config, CHECKPOINT, device)",
        "permit = m1558.issue_preload_permit(WORK)",
        "producer = m1558.ReducedBinaryProducer(",
        "write_child_receipt(WORK, release, load_audit, validation)",
        "WORK.rename(RESULT)",
        "WORK.rename(FAILURE)",
    ]
    for token in required_child:
        assert child.count(token) == 1, token
    order = [child.index(token) for token in required_child[:13]]
    assert order == sorted(order)
    assert child.index("consume_attempt(release)") < child.index(
        "profile.load_config(CONFIG)")
    assert child.index("consume_attempt(release)") < child.index(
        "m1558.issue_preload_permit(WORK)")
    assert child.index("consume_attempt(release)") < child.index(
        "m1558.ReducedBinaryProducer(")
    assert child.count("issue_preload_permit(") == 1
    assert child.count("ReducedBinaryProducer(") == 1
    assert "inspect" not in child and "globals(" not in child
    assert "__closure__" not in child and "__globals__" not in child

    attempt = function_block(text, "consume_attempt", "load_m1434")
    assert "os.O_EXCL" in attempt and "os.O_NOFOLLOW" in attempt
    assert "os.fsync(descriptor)" in attempt
    assert "unlink" not in attempt and "remove(" not in attempt
    assert text.count("subprocess.run(") == 1
    assert "automatic_retry\": True" not in text
    assert '"tsbg_dse": True' not in text
    assert '"aee": True' not in text
    assert '"rtl": True' not in text
    assert '"eda": True' not in text
    assert '"tsbg_dse": False' in text
    assert '"aee": False' in text
    assert '"rtl": False' in text
    assert '"eda": False' in text
    assignments = dict((node.targets[0].id, node.value) for node in tree.body
                       if isinstance(node, ast.Assign)
                       and len(node.targets) == 1
                       and isinstance(node.targets[0], ast.Name))
    expected_namespaces = {
        "RESULT": "results/m1624_motion_ep34_s2_tsbg_reduced_binary_"
                  "capture_s40_r1_20260901",
        "ATTEMPT": "results/.m1624_motion_ep34_s2_tsbg_reduced_binary_"
                   "capture_s40_r1_20260901.attempt_consumed",
        "WORK": "results/.m1624_motion_ep34_s2_tsbg_reduced_binary_"
                "capture_s40_r1_20260901.work",
        "FAILURE": "results/m1624_motion_ep34_s2_tsbg_reduced_binary_"
                   "capture_s40_r1_20260901.failed_no_retry",
    }
    for name, expected in expected_namespaces.items():
        value = assignments[name]
        assert isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div)
        assert isinstance(value.right, ast.Str) and value.right.s == expected


class M1624SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SOURCE.read_text(encoding="utf-8")
        cls.module = load_source()

    def test_01_regular_source_and_frozen_identities(self):
        for path in (SOURCE, CONTRACT, M1558, M1598 / "review.json", DOCS359):
            self.assertTrue(stat.S_ISREG(path.lstat().st_mode), str(path))
            self.assertFalse(path.is_symlink(), str(path))
        self.assertEqual(sha256(M1558),
            "e6686564064ae3acda2bfcfc8c2d75061eb9cb591bc739d090bc03911469b089")
        self.assertEqual(sha256(M1598 / "review.json"),
            "e887266475d28f7c2cfba3f69cbbbd103eed9db08905eebe042528f2baea1065")
        self.assertEqual(sha256(DOCS359),
            "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4")

    def test_02_import_is_inert(self):
        before = set(sys.modules)
        module = load_source()
        after = set(sys.modules)
        self.assertFalse("torch" in after - before)
        self.assertFalse("numpy" in after - before)
        self.assertFalse(module.RESULT.exists())
        self.assertFalse(module.ATTEMPT.exists())
        self.assertFalse(module.WORK.exists())
        self.assertFalse(module.FAILURE.exists())

    def test_03_source_structure_and_fixed_child_boundary(self):
        audit_source(self.text)

    def test_04_no_capability_parameter_crosses_parent_boundary(self):
        self.assertEqual(list(inspect.signature(self.module.launch_parent).parameters), [])
        self.assertEqual(list(inspect.signature(self.module.fixed_clean_child).parameters), [])
        parent = function_block(self.text, "launch_parent", "source_self_check")
        self.assertNotIn("os.environ.copy", parent)
        self.assertNotIn("sys.argv", parent)

    def test_05_source_self_check_reads_no_m1458_payload(self):
        allowed = {self.module.M1458_MANIFEST.resolve(),
                   self.module.M1458_SUMS.resolve(),
                   self.module.M1458_OUTER.resolve()}
        original_open = Path.open
        observed = []

        def guarded_open(path, *args, **kwargs):
            resolved = Path(path).resolve()
            try:
                resolved.relative_to(self.module.M1458_ROOT.resolve())
                observed.append(resolved)
                if resolved not in allowed:
                    raise AssertionError("M1458 payload opened: " + str(resolved))
            except ValueError:
                pass
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", guarded_open):
            value = self.module.source_self_check()
        self.assertEqual(value["status"],
            "PASS_M1624_SOURCE_SELF_CHECK__NO_PAYLOAD_NO_GPU_NO_CAPTURE")
        self.assertTrue(observed)
        self.assertTrue(set(observed) <= allowed)

    def test_06_source_contract_is_exact_and_non_authorizing(self):
        value = self.module.validate_source_contract()
        self.assertEqual(value["source"]["sha256"], sha256(SOURCE))
        self.assertEqual(value["test"]["sha256"], sha256(Path(__file__)))
        self.assertFalse(value["authorization"]["capture"])
        self.assertFalse(value["authorization"]["gpu"])
        self.assertTrue(value["authorization"]["different_author_review"])
        for key in ("tsbg_dse", "aee", "cycles", "traffic", "energy",
                    "speedup", "rtl", "eda", "paper_result"):
            self.assertFalse(value["claim_boundary"][key], key)

    def test_07_future_review_release_and_namespaces_are_fresh(self):
        for path in (self.module.FUTURE_REVIEW, self.module.FUTURE_RELEASE,
                     self.module.RESULT, self.module.ATTEMPT,
                     self.module.WORK, self.module.FAILURE):
            self.assertFalse(path.exists(), str(path))

    def test_08_fixed_metadata_binds_final_ep34_live93_and_m1598(self):
        value = self.module.verify_fixed_metadata(expect_future_absent=True)
        self.assertEqual(value, {
            "checkpoint_sha256":
                "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
            "samples": 40,
            "m1558_sha256":
                "e6686564064ae3acda2bfcfc8c2d75061eb9cb591bc739d090bc03911469b089",
            "m1598_review_sha256":
                "e887266475d28f7c2cfba3f69cbbbd103eed9db08905eebe042528f2baea1065",
        })

    def test_09_release_gate_remains_fail_closed(self):
        with self.assertRaises(self.module.M1624Error):
            self.module.validate_future_authorities()

    def test_10_attempt_precedes_model_permit_and_producer(self):
        child = function_block(self.text, "fixed_clean_child", "launch_parent")
        marker = child.index("consume_attempt(release)")
        for later in ("profile.load_config(CONFIG)", "profile.build_model(",
                      "m1558.issue_preload_permit(WORK)",
                      "m1558.ReducedBinaryProducer(", "np.load("):
            self.assertLess(marker, child.index(later), later)

    def test_11_one_parent_call_one_child_zero_retry(self):
        parent = function_block(self.text, "launch_parent", "source_self_check")
        self.assertEqual(parent.count("subprocess.run("), 1)
        self.assertNotIn("while ", parent)
        self.assertNotIn("for ", parent)
        self.assertIn("fixed clean child failed; no retry", parent)
        attempt = function_block(self.text, "consume_attempt", "load_m1434")
        self.assertNotIn("unlink", attempt)
        self.assertNotIn("remove", attempt)

    def test_12_result_receipt_keeps_tsbg_and_hardware_claims_false(self):
        receipt = function_block(self.text, "write_child_receipt",
                                 "close_failed_producer")
        for token in ('"tsbg_dse": False', '"aee": False',
                      '"cycles": False', '"traffic": False',
                      '"energy": False', '"speedup": False',
                      '"rtl": False', '"eda": False',
                      '"paper_result": False'):
            self.assertIn(token, receipt)
        self.assertIn("fresh_result_hammer_required", receipt)
        self.assertIn("seal_result(root)", receipt)

    def test_13_reflection_and_dynamic_provider_mutations_rejected(self):
        mutants = [
            self.text.replace("def fixed_clean_child():",
                              "def fixed_clean_child(provider=None):", 1),
            self.text.replace("def launch_parent():",
                              "def launch_parent(free_bytes=None):", 1),
            self.text + "\nimport inspect\n",
            self.text + "\nx = globals()\n",
            self.text + "\nx = fixed_clean_child.__closure__\n",
        ]
        for mutant in mutants:
            with self.assertRaises(AssertionError):
                audit_source(mutant)

    def test_14_budget_order_namespace_and_claim_mutations_rejected(self):
        child = function_block(self.text, "fixed_clean_child", "launch_parent")
        moved = child.replace("            consume_attempt(release)\n", "", 1)
        moved = moved.replace("            permit = m1558.issue_preload_permit(WORK)",
                              "            permit = m1558.issue_preload_permit(WORK)\n"
                              "            consume_attempt(release)", 1)
        mutants = [
            self.text.replace(child, moved, 1),
            self.text.replace("subprocess.run(command", "subprocess.run(command\n"
                              "    subprocess.run(command", 1),
            self.text.replace('"tsbg_dse": False', '"tsbg_dse": True', 1),
            self.text.replace("results/m1624_motion_ep34_s2_tsbg_",
                              "results/m1458_m1434_motion_ep34_live93_", 1),
        ]
        for mutant in mutants:
            with self.assertRaises((AssertionError, SyntaxError)):
                audit_source(mutant)

    def test_15_source_contains_no_capture_side_effect_in_static_modes(self):
        self.assertEqual(self.module.source_self_check()["child_processes"], 0)
        self.assertFalse(self.module.source_self_check()["gpu"])
        self.assertFalse(self.module.source_self_check()["capture"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
