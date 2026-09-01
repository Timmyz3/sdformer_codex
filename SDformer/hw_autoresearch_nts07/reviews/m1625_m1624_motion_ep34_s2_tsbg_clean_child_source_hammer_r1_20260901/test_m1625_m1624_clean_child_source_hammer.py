#!/usr/bin/env python3
"""Independent compile-free hammer for the M1624 clean-child source.

This test never launches the parent or child and never opens capture payloads.
It checks the fixed process boundary, one-shot ordering, identities, accounting,
publication/failure behavior and negative claim boundary from source text and
sealed metadata only.
"""

from __future__ import print_function

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import unittest


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1624_motion_ep34_s2_tsbg_clean_child_reduced_binary_"
    "successor_r1.py")
AUTHOR_TEST = HW / "tests/test_m1624_motion_ep34_s2_tsbg_clean_child_source.py"
CONTRACT = HW / (
    "contracts/m1624_motion_ep34_s2_tsbg_clean_child_reduced_binary_"
    "source_contract_r1_20260901.json")
AUTHOR = HW / (
    "reviews/m1624_motion_ep34_s2_tsbg_clean_child_source_author_receipt_"
    "r1_20260901")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

SOURCE_SHA = "ad36ab02b598f28458ed226f816b47281b7d388fddfe80bc7ea15155709ba76f"
TEST_SHA = "5b44434df85b2832435ded94258a9a9f038f902ed6e77de1f4b7d690c497891b"
CONTRACT_SHA = "2ba3445c2c40c437124c62f49881db1b8443344aa19afc504f4f45aa1c1eacd9"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CHECKPOINT_SHA = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(rows):
        value = {}
        for key, item in rows:
            assert key not in value
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          AssertionError("nonfinite " + token)))


def verify_tree(root):
    root = Path(root)
    sums = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    assert stat.S_ISREG(sums.lstat().st_mode) and not sums.is_symlink()
    assert stat.S_ISREG(outer.lstat().st_mode) and not outer.is_symlink()
    assert outer.read_text(encoding="ascii") == sha256(sums) + "  SHA256SUMS\n"
    members = set()
    for row in sums.read_text(encoding="utf-8").splitlines():
        fields = row.split("  ", 1)
        assert len(fields) == 2 and len(fields[0]) == 64
        relative = Path(fields[1])
        assert not relative.is_absolute() and ".." not in relative.parts
        member = root / relative
        assert stat.S_ISREG(member.lstat().st_mode) and not member.is_symlink()
        assert sha256(member) == fields[0]
        members.add(relative.as_posix())
    return members


def load_source():
    before = set(sys.modules)
    spec = importlib.util.spec_from_file_location("m1625_read_only_m1624", str(SOURCE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    added = set(sys.modules) - before
    assert "torch" not in added and "numpy" not in added
    return module


def function_block(text, name, next_name):
    begin = text.index("def " + name + "(")
    end = text.index("def " + next_name + "(", begin)
    return text[begin:end]


def audit_source(text):
    tree = ast.parse(text)
    functions = dict((node.name, node) for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)))
    for name in ("fixed_clean_child", "launch_parent"):
        assert name in functions
        node = functions[name]
        assert len(node.args.args) == 0 and node.args.vararg is None
        assert node.args.kwarg is None and len(node.args.kwonlyargs) == 0

    banned_names = {"inspect", "eval", "exec", "globals", "locals"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not (set(alias.name for alias in node.names) & banned_names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module not in banned_names
        elif isinstance(node, ast.Name):
            assert node.id not in banned_names
        elif isinstance(node, ast.Attribute):
            assert node.attr not in {"__closure__", "__globals__", "__code__"}

    parent = function_block(text, "launch_parent", "source_self_check")
    child = function_block(text, "fixed_clean_child", "launch_parent")
    attempt = function_block(text, "consume_attempt", "load_m1434")
    receipt = function_block(text, "write_child_receipt", "close_failed_producer")
    authority = function_block(text, "validate_future_authorities",
                               "require_fresh_namespaces")

    assert text.count("subprocess.run(") == 1
    assert parent.count("subprocess.run(") == 1
    assert 'command = [str(CHILD_PYTHON), "-I", str(SOURCE), "--fixed-clean-child"]' in parent
    assert "shell=True" not in parent and "os.environ.copy" not in parent
    assert "sys.argv" not in parent and "PYTHONPATH" not in parent
    assert "stdin=subprocess.DEVNULL" in parent and "check=False" in parent
    assert '"CUDA_VISIBLE_DEVICES": "0"' in parent
    for token in ("provider", "permit", "free_space", "free_bytes",
                  "provenance", "callable", "registry", "callback"):
        assert token not in parent
    assert parent.count("verify_fixed_metadata(expect_future_absent=False)") == 1
    assert parent.count("validate_future_authorities()") == 1
    assert parent.count("require_fresh_namespaces()") == 1
    assert parent.index("verify_fixed_metadata") < parent.index("subprocess.run(")
    assert parent.index("validate_future_authorities") < parent.index("subprocess.run(")
    assert parent.index("require_fresh_namespaces") < parent.index("subprocess.run(")
    assert "while " not in parent and "for " not in parent
    assert "fixed clean child failed; no retry" in parent

    required_order = [
        "verify_fixed_metadata(expect_future_absent=False)",
        "release = validate_future_authorities()",
        "require_fresh_namespaces()",
        "m1434 = load_m1434()",
        "m1558 = load_m1558()",
        "regular_exact(CHECKPOINT, CHECKPOINT_SHA256",
        "samples = m1434.R1.validate_cohort",
        "sample_order = m1558.M1552.verify_bindings()",
        "available = shutil.disk_usage",
        "with substrate.exclusive_gpu_lease",
        "consume_attempt(release)",
        "profile.load_config(CONFIG)",
        "model = profile.build_model(config, CHECKPOINT, device)",
        "permit = m1558.issue_preload_permit(WORK)",
        "producer = m1558.ReducedBinaryProducer(",
        "for row in samples:",
        "output = producer.finalize_source_result()",
        "validation = m1558.validate_binary_result(WORK, specs, sample_order)",
        "write_child_receipt(WORK, release, load_audit, validation)",
        "WORK.rename(RESULT)",
    ]
    for token in required_order:
        assert child.count(token) >= 1, token
    positions = [child.index(token) for token in required_order]
    assert positions == sorted(positions)
    assert child.count("consume_attempt(release)") == 1
    assert child.count("issue_preload_permit(WORK)") == 1
    assert child.count("ReducedBinaryProducer(") == 1
    assert child.count("WORK.rename(RESULT)") == 1
    assert child.count("WORK.rename(FAILURE)") == 1
    assert "if WORK.is_dir() and not os.path.lexists(str(FAILURE))" in child
    assert "if not published:" in child
    assert "close_failed_producer(producer)" in child
    assert "automatic_retry" not in child

    assert "os.O_EXCL" in attempt and "os.O_NOFOLLOW" in attempt
    assert "os.fsync(descriptor)" in attempt
    assert "unlink" not in attempt and "remove(" not in attempt
    assert "release_sha256=" in attempt and "source_sha256=" in attempt

    assert '"samples": 40' in receipt
    for key in ("frames", "fc_tokens", "patch_histogram_rows"):
        assert key in receipt
    for token in ('"clean_child_processes": 1', '"automatic_retry": False',
                  '"provider_crossed_parent_boundary": False',
                  '"permit_crossed_parent_boundary": False',
                  '"free_space_crossed_parent_boundary": False',
                  '"provenance_crossed_parent_boundary": False',
                  '"callable_crossed_parent_boundary": False'):
        assert token in receipt
    for token in ('"tsbg_dse": False', '"aee": False', '"cycles": False',
                  '"traffic": False', '"energy": False',
                  '"speedup": False', '"rtl": False', '"eda": False',
                  '"paper_result": False'):
        assert token in receipt
    assert "fresh_result_hammer_required" in receipt
    assert receipt.index("seal_result(root)") > receipt.index(
        '"paper_result": False')

    assert 'review.get("score", 0) >= 95' in authority
    assert 'review.get("p0_count") == 0' in authority
    assert 'review.get("p1_count") == 0' in authority
    assert '"release_authoring": True, "capture": False' in authority
    assert '"parent_calls": 1, "clean_child_processes": 1' in authority
    assert '"gpu_runs": 1, "production_captures": 1' in authority
    assert '"automatic_retry": False, "all_other_runs": 0' in authority
    assert '"tsbg_dse": False, "aee": False, "rtl": False' in authority
    assert '"performance": False' in authority

    constants = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name)):
            constants[node.targets[0].id] = node.value
    expected_suffixes = {
        "RESULT": "results/m1624_motion_ep34_s2_tsbg_reduced_binary_capture_s40_r1_20260901",
        "ATTEMPT": "results/.m1624_motion_ep34_s2_tsbg_reduced_binary_capture_s40_r1_20260901.attempt_consumed",
        "WORK": "results/.m1624_motion_ep34_s2_tsbg_reduced_binary_capture_s40_r1_20260901.work",
        "FAILURE": "results/m1624_motion_ep34_s2_tsbg_reduced_binary_capture_s40_r1_20260901.failed_no_retry",
    }
    for name, expected in expected_suffixes.items():
        value = constants[name]
        assert isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div)
        assert isinstance(value.right, ast.Str) and value.right.s == expected


def mutation_rejections(text):
    child = function_block(text, "fixed_clean_child", "launch_parent")
    moved = child.replace("            consume_attempt(release)\n", "", 1)
    moved = moved.replace("            permit = m1558.issue_preload_permit(WORK)",
                          "            permit = m1558.issue_preload_permit(WORK)\n"
                          "            consume_attempt(release)", 1)
    mutants = [
        text.replace("def fixed_clean_child():", "def fixed_clean_child(provider=None):", 1),
        text.replace("def launch_parent():", "def launch_parent(permit=None):", 1),
        text + "\nimport inspect\n",
        text + "\nx = globals()\n",
        text + "\nx = fixed_clean_child.__closure__\n",
        text.replace("subprocess.run(command", "subprocess.run(command\n    subprocess.run(command", 1),
        text.replace('"-I", str(SOURCE)', 'str(SOURCE)', 1),
        text.replace("stdin=subprocess.DEVNULL", "stdin=None", 1),
        text.replace("check=False", "check=True", 1),
        text.replace('"CUDA_VISIBLE_DEVICES": "0"', '"CUDA_VISIBLE_DEVICES": "1"', 1),
        text.replace("require_fresh_namespaces()\n    # No inherited", "# freshness removed\n    # No inherited", 1),
        text.replace(child, moved, 1),
        text.replace("os.O_EXCL", "0", 1),
        text.replace("os.O_NOFOLLOW", "0", 1),
        text.replace("os.fsync(descriptor)", "pass", 1),
        text.replace("WORK.rename(RESULT)", "shutil.copytree(WORK, RESULT)", 1),
        text.replace("WORK.rename(FAILURE)", "WORK.rename(RESULT)", 1),
        text.replace('"samples": 40', '"samples": 39'),
        text.replace('"clean_child_processes": 1', '"clean_child_processes": 2', 1),
        text.replace('"automatic_retry": False', '"automatic_retry": True'),
        text.replace('"provider_crossed_parent_boundary": False',
                     '"provider_crossed_parent_boundary": True', 1),
        text.replace('"tsbg_dse": False', '"tsbg_dse": True', 1),
        text.replace('"aee": False', '"aee": True', 1),
        text.replace('"speedup": False', '"speedup": True', 1),
        text.replace('review.get("score", 0) >= 95', 'review.get("score", 0) >= 0', 1),
        text.replace('review.get("p0_count") == 0', 'review.get("p0_count") >= 0'),
        text.replace('"release_authoring": True, "capture": False',
                     '"release_authoring": True, "capture": True', 1),
        text.replace('"gpu_runs": 1, "production_captures": 1',
                     '"gpu_runs": 2, "production_captures": 2', 1),
        text.replace('"automatic_retry": False, "all_other_runs": 0',
                     '"automatic_retry": True, "all_other_runs": 1', 1),
        text.replace("results/m1624_motion_ep34_s2_tsbg_reduced_binary_",
                     "results/m1458_m1434_motion_ep34_live93_", 1),
    ]
    rejected = 0
    survivors = []
    for index, mutant in enumerate(mutants):
        try:
            audit_source(mutant)
        except (AssertionError, SyntaxError, ValueError):
            rejected += 1
        else:
            survivors.append(index)
    return rejected, len(mutants), survivors


class M1625IndependentHammer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SOURCE.read_text(encoding="utf-8")
        cls.module = load_source()
        cls.contract = strict_json(CONTRACT)

    def test_01_exact_source_test_contract_and_docs_identity(self):
        self.assertEqual(sha256(SOURCE), SOURCE_SHA)
        self.assertEqual(sha256(AUTHOR_TEST), TEST_SHA)
        self.assertEqual(sha256(CONTRACT), CONTRACT_SHA)
        self.assertEqual(sha256(DOCS359), DOCS359_SHA)
        self.assertEqual(self.contract["source"]["sha256"], SOURCE_SHA)
        self.assertEqual(self.contract["test"]["sha256"], TEST_SHA)

    def test_02_author_receipt_is_double_sealed(self):
        members = verify_tree(AUTHOR)
        self.assertTrue({"review.json", "mechanical_checks.json",
                         "source_self_check.json", "cpython36_output.txt",
                         "cpython310_output.txt"} <= members)
        review = strict_json(AUTHOR / "review.json")
        self.assertEqual(review["status"],
            "PASS_AUTHOR_DUAL_RUNTIME_CLEAN_CHILD_SOURCE__DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_CAPTURE")
        self.assertEqual(review["paper_claims"], 0)

    def test_03_import_is_inert_and_namespaces_remain_fresh(self):
        for path in (self.module.RESULT, self.module.ATTEMPT,
                     self.module.WORK, self.module.FAILURE):
            self.assertFalse(path.exists(), str(path))

    def test_04_process_boundary_and_order_are_static_and_exact(self):
        audit_source(self.text)

    def test_05_no_parent_capability_or_retry_surface(self):
        parent = function_block(self.text, "launch_parent", "source_self_check")
        self.assertNotIn("os.environ.copy", parent)
        self.assertNotIn("PYTHONPATH", parent)
        self.assertNotIn("shell=True", parent)
        self.assertEqual(parent.count("subprocess.run("), 1)

    def test_06_child_rechecks_capture_and_checkpoint_identity(self):
        fixed = function_block(self.text, "verify_fixed_metadata",
                               "validate_future_authorities")
        self.assertIn("M1458_MANIFEST_SHA256", fixed)
        self.assertIn("M1458_SUMS_SHA256", fixed)
        self.assertIn("M1458_OUTER_SHA256", fixed)
        self.assertIn("SAMPLE_ORDER_SHA256", fixed)
        self.assertIn("selected.get(\"epoch\") == 34", fixed)
        self.assertIn("capture.get(\"cohort\", {}).get(\"population\") == 40", fixed)
        self.assertIn("capture.get(\"ordered_population\", {}).get(\"records\") == 9880", fixed)

    def test_07_child_population_and_topology_counts_are_bound(self):
        child = function_block(self.text, "fixed_clean_child", "launch_parent")
        self.assertIn("list(range(40))", self.text)
        self.assertIn('"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12', child)
        self.assertIn("array.shape == (10, 480, 640)", child)
        self.assertIn("for row in samples:", child)
        self.assertIn("producer.end_sample()", child)

    def test_08_receipt_summary_is_validation_derived(self):
        receipt = function_block(self.text, "write_child_receipt",
                                 "close_failed_producer")
        self.assertIn('"frames": int(validation["frames"])', receipt)
        self.assertIn('"fc_tokens": int(validation["fc_tokens"])', receipt)
        self.assertIn('"patch_histogram_rows": int(validation["patch_histogram_rows"])', receipt)
        self.assertIn("seal_result(root)", receipt)

    def test_09_atomic_attempt_publish_and_failure_quarantine(self):
        audit_source(self.text)
        child = function_block(self.text, "fixed_clean_child", "launch_parent")
        self.assertLess(child.index("consume_attempt(release)"),
                        child.index("profile.build_model("))
        self.assertLess(child.index("write_child_receipt("),
                        child.index("WORK.rename(RESULT)"))
        self.assertIn("WORK.rename(FAILURE)", child)

    def test_10_only_m1625_then_m1626_can_authorize_one_capture(self):
        authority = function_block(self.text, "validate_future_authorities",
                                   "require_fresh_namespaces")
        self.assertIn("REVIEW_STATUS", authority)
        self.assertIn("RELEASE_STATUS", authority)
        self.assertIn('"parent_calls": 1, "clean_child_processes": 1', authority)
        self.assertIn('"all_other_runs": 0', authority)

    def test_11_source_contract_authorizes_only_this_review(self):
        auth = self.contract["authorization"]
        self.assertTrue(auth["different_author_review"])
        for key in ("release_authoring", "parent_launch", "clean_child",
                    "checkpoint_load", "gpu", "capture", "remote",
                    "automatic_retry", "tsbg_dse", "rtl", "eda"):
            self.assertFalse(auth[key], key)

    def test_12_no_payload_gpu_remote_or_capture_has_happened(self):
        verification = self.contract["author_verification"]
        for key in ("payload_opened", "checkpoint_loaded", "gpu", "capture",
                    "remote", "aee", "rtl", "eda"):
            self.assertFalse(verification[key], key)
        for path in (self.module.RESULT, self.module.ATTEMPT,
                     self.module.WORK, self.module.FAILURE):
            self.assertFalse(path.exists(), str(path))

    def test_13_claim_boundary_has_no_measurement_or_paper_result(self):
        boundary = self.contract["claim_boundary"]
        for key in ("production_result", "hardware_quantization_authority",
                    "model_bit_exact", "tsbg_dse", "aee", "cycles",
                    "traffic", "energy", "speedup", "system_speedup",
                    "rtl", "eda", "paper_result"):
            self.assertFalse(boundary[key], key)

    def test_14_fixed_checkpoint_and_cohort_identity(self):
        frozen = self.contract["frozen_identity"]
        self.assertEqual(frozen["checkpoint_sha256"], CHECKPOINT_SHA)
        self.assertEqual(frozen["checkpoint_size_bytes"], 225504447)
        self.assertEqual(frozen["samples"], 40)
        self.assertEqual(frozen["docs359_sha256"], DOCS359_SHA)

    def test_15_clean_child_source_mutations_fail_closed(self):
        rejected, total, survivors = mutation_rejections(self.text)
        self.assertEqual(total, 30)
        self.assertEqual(rejected, total, "surviving mutation indices: " +
                         repr(survivors))


if __name__ == "__main__":
    unittest.main(verbosity=2)
