#!/usr/bin/env python3
"""No-remote dual-runtime tests for M1707 deployment closure."""
from __future__ import print_function

import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1707_motion_ep34_s2_tsbg_deployment_complete_successor_r1.py")
SPEC = importlib.util.spec_from_file_location("test_exact_m1707", str(SOURCE))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True,
                                     allow_nan=False) + "\n")


def seal_review(root, value):
    root.mkdir()
    review = root / "review.json"; write_json(review, value)
    manifest = root / "SHA256SUMS"
    manifest.write_text(M.sha256(review) + "  review.json\n", encoding="ascii")
    outer = root / "SHA256SUMS.seal.sha256"
    outer.write_text(M.sha256(manifest) + "  SHA256SUMS\n", encoding="ascii")
    return M.sha256(review), M.sha256(manifest), M.sha256(outer)


def seal_file(path, value):
    write_json(path, value)
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(M.sha256(path) + "  " + path.name + "\n",
                       encoding="ascii")
    Path(str(path) + ".sha256.seal.sha256").write_text(
        M.sha256(sidecar) + "  " + sidecar.name + "\n", encoding="ascii")


def base_review():
    return {"schema": "m1708_test_review", "status": M.REVIEW_STATUS,
        "score": 100, "p0_count": 0, "p1_count": 0,
        "identity": M.expected_review_identity(),
        "authorization": {"release_authoring": True, "capture": False,
            "gpu": False, "automatic_retry": False}}


def base_release(identity, interpreter):
    return {"schema": M.RELEASE_SCHEMA, "status": M.RELEASE_STATUS,
        "identity": identity,
        "authorization": {"parent_calls": 1, "clean_child_processes": 1,
            "gpu_runs": 1, "production_captures": 1,
            "automatic_retry": False, "all_other_runs": 0},
        "namespaces": {"result": str(M.RESULT.relative_to(M.ROOT)),
            "attempt": str(M.ATTEMPT.relative_to(M.ROOT)),
            "work": str(M.WORK.relative_to(M.ROOT)),
            "failure": str(M.FAILURE.relative_to(M.ROOT))},
        "pre_budget_runtime_closure": {
            "validator_path": str(M.RUNTIME_VALIDATOR.relative_to(M.ROOT)),
            "validator_sha256": M.RUNTIME_VALIDATOR_SHA256,
            "m1558_verify_bindings": True,
            "frozen_layer_specs": M.EXPECTED_LAYERS,
            "estimated_result_upper_bytes": M.EXPECTED_RESULT_UPPER_BYTES,
            "before_parent_subprocess_budget": True,
            "before_clean_child_gpu_attempt_budget": True},
        "remote_target": dict(M.REMOTE_TARGET),
        "claim_boundary": {"tsbg_dse": False, "aee": False,
            "rtl": False, "eda": False, "performance": False,
            "paper_result": False},
        "child_interpreter": {"path": str(interpreter),
            "sha256": M.sha256(interpreter)}}


@contextlib.contextmanager
def authority_fixture(review_mutator=None, release_mutator=None):
    with tempfile.TemporaryDirectory(prefix=".m1707-fixture-",
                                     dir=str(HW / "reviews")) as directory:
        root = Path(directory); review_root = root / "review"
        review = base_review()
        if review_mutator: review_mutator(review)
        review_sha, manifest_sha, outer_sha = seal_review(review_root, review)
        identity = dict(M.expected_review_identity(),
            review_sha256=review_sha, review_manifest_sha256=manifest_sha,
            review_outer_file_sha256=outer_sha)
        interpreter = Path(sys.executable).resolve()
        release = base_release(identity, interpreter)
        if release_mutator: release_mutator(release)
        release_path = root / "release.json"; seal_file(release_path, release)
        old = M.FUTURE_REVIEW, M.FUTURE_RELEASE, M.CHILD_INTERPRETER
        M.FUTURE_REVIEW, M.FUTURE_RELEASE = review_root, release_path
        M.CHILD_INTERPRETER = interpreter
        try:
            yield release
        finally:
            M.FUTURE_REVIEW, M.FUTURE_RELEASE, M.CHILD_INTERPRETER = old


class M1707Tests(unittest.TestCase):
    def test_01_m1692_failure_no_retry_is_exact(self):
        row = M.verify_predecessors()
        self.assertEqual(row["m1692_source_sha256"], M.M1692_SOURCE_SHA256)
        self.assertEqual(row["m1692_failure_receipt_sha256"],
                         M.M1692_FAILURE_RECEIPT_SHA256)
        self.assertTrue(row["m1692_no_retry"])
        self.assertEqual(row["failure"]["canonical_missing_member_sha256"],
                         M.RUNTIME_VALIDATOR_SHA256)

    def test_02_full_runtime_closure_is_real_and_exact(self):
        row = M.verify_runtime_closure()
        self.assertEqual(row["validator_sha256"], M.RUNTIME_VALIDATOR_SHA256)
        self.assertTrue(row["m1558_verify_bindings"])
        self.assertEqual(row["frozen_layer_specs"], 32)
        self.assertEqual(row["estimated_result_upper_bytes"], 7598737368)
        self.assertEqual(row["gpu_runs"], 0)
        self.assertEqual(row["attempt_writes"], 0)

    def test_03_missing_or_mismatched_validator_rejected(self):
        original_path, original_sha = M.RUNTIME_VALIDATOR, M.RUNTIME_VALIDATOR_SHA256
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.py"
            M.RUNTIME_VALIDATOR = missing
            try:
                with self.assertRaises(M.M1707Error): M.verify_runtime_closure()
            finally:
                M.RUNTIME_VALIDATOR = original_path
            bad = Path(directory) / "validator.py"; bad.write_text("bad\n")
            M.RUNTIME_VALIDATOR = bad
            try:
                with self.assertRaises(M.M1707Error): M.verify_runtime_closure()
            finally:
                M.RUNTIME_VALIDATOR = original_path
        M.RUNTIME_VALIDATOR_SHA256 = original_sha

    def test_04_m1558_layer_or_estimate_mutations_rejected(self):
        original = M.P.P.P.P.load_m1558
        class FakeM1552(object):
            @staticmethod
            def verify_bindings(): return {"samples": []}
        class Fake(object):
            M1552 = FakeM1552
            @staticmethod
            def frozen_layer_specs(): return [object()] * 31
            @staticmethod
            def estimate_from_specs(_specs, _samples):
                return {"result_upper_bytes": 7598737368}
        M.P.P.P.P.load_m1558 = lambda: Fake
        try:
            with self.assertRaises(M.M1707Error): M.verify_runtime_closure()
            Fake.frozen_layer_specs = staticmethod(lambda: [object()] * 32)
            Fake.estimate_from_specs = staticmethod(
                lambda _specs, _samples: {"result_upper_bytes": 1})
            with self.assertRaises(M.M1707Error): M.verify_runtime_closure()
        finally:
            M.P.P.P.P.load_m1558 = original

    def test_05_future_authority_exact_shape_passes(self):
        with authority_fixture() as expected:
            self.assertEqual(M.validate_future_authorities(), expected)

    def test_06_runtime_closure_release_mutations_rejected(self):
        mutations = (
            lambda row: row["pre_budget_runtime_closure"].update(
                validator_sha256="0" * 64),
            lambda row: row["pre_budget_runtime_closure"].update(
                frozen_layer_specs=31),
            lambda row: row["pre_budget_runtime_closure"].update(
                estimated_result_upper_bytes=1),
            lambda row: row["pre_budget_runtime_closure"].update(
                before_parent_subprocess_budget=False),
            lambda row: row["pre_budget_runtime_closure"].update(
                before_clean_child_gpu_attempt_budget=False),
            lambda row: row["authorization"].update(automatic_retry=True),
            lambda row: row["namespaces"].update(result="old-m1692"),
        )
        for mutation in mutations:
            with authority_fixture(release_mutator=mutation):
                with self.assertRaises(Exception):
                    M.validate_future_authorities()

    def test_07_parent_orders_closure_before_budget_delegate(self):
        events = []; originals = (M.verify_predecessors,
            M.verify_runtime_closure, M._bound_exact_m1692, M.P.launch_parent)
        @contextlib.contextmanager
        def bound(): events.append("bound"); yield
        M.verify_predecessors = lambda: events.append("predecessors")
        M.verify_runtime_closure = lambda: events.append("closure")
        M._bound_exact_m1692 = bound
        M.P.launch_parent = lambda: events.append("parent_budget")
        try: M.launch_parent()
        finally:
            (M.verify_predecessors, M.verify_runtime_closure,
             M._bound_exact_m1692, M.P.launch_parent) = originals
        self.assertEqual(events, ["predecessors", "closure", "bound",
                                  "parent_budget"])

    def test_08_child_orders_closure_before_gpu_attempt_delegate(self):
        events = []; originals = (M.verify_predecessors,
            M.verify_runtime_closure, M._bound_exact_m1692,
            M.P.fixed_clean_child)
        @contextlib.contextmanager
        def bound(): events.append("bound"); yield
        M.verify_predecessors = lambda: events.append("predecessors")
        M.verify_runtime_closure = lambda: events.append("closure")
        M._bound_exact_m1692 = bound
        M.P.fixed_clean_child = lambda: events.append("gpu_attempt_budget")
        try: M.fixed_clean_child()
        finally:
            (M.verify_predecessors, M.verify_runtime_closure,
             M._bound_exact_m1692, M.P.fixed_clean_child) = originals
        self.assertEqual(events, ["predecessors", "closure", "bound",
                                  "gpu_attempt_budget"])

    def test_09_bound_context_uses_fresh_m1707_and_restores(self):
        names = ("SOURCE", "FUTURE_REVIEW", "FUTURE_RELEASE", "RESULT",
                 "ATTEMPT", "WORK", "FAILURE", "validate_future_authorities")
        old = dict((name, getattr(M.P, name)) for name in names)
        with M._bound_exact_m1692():
            self.assertEqual(M.P.SOURCE, M.SOURCE)
            self.assertIs(M.P.validate_future_authorities,
                          M.validate_future_authorities)
            self.assertTrue(all("m1707_" in path.name for path in
                                (M.P.RESULT, M.P.ATTEMPT, M.P.WORK, M.P.FAILURE)))
        self.assertEqual(dict((name, getattr(M.P, name)) for name in names), old)

    def test_10_exact_engine_budget_and_no_retry_preserved(self):
        source = M.M1692_SOURCE.read_text()
        self.assertEqual(M.sha256(M.M1692_SOURCE), M.M1692_SOURCE_SHA256)
        self.assertIn("return P.fixed_clean_child()", source)
        self.assertIn("return P.launch_parent()", source)
        lower = M.P.P.P.P.SOURCE.read_text()
        body = lower[lower.index("def fixed_clean_child():"):]
        self.assertLess(body.index("m1434.build_runtime()"),
                        body.index("exclusive_gpu_lease"))
        self.assertLess(body.index("exclusive_gpu_lease"),
                        body.index("consume_attempt(release)"))
        self.assertLess(body.index("consume_attempt(release)"),
                        body.index("profile.load_config(CONFIG)"))

    def test_11_source_contract_self_check_and_fresh_namespaces(self):
        M.require_fresh_namespaces()
        contract = M.validate_source_contract()
        self.assertFalse(contract["authorization"]["automatic_retry"])
        row = M.source_self_check()
        self.assertEqual(row["runtime_closure"]["frozen_layer_specs"], 32)
        self.assertFalse(row["remote_connected"])
        self.assertEqual(row["attempt_writes"], 0)

    def test_12_receipt_binds_deployment_closure(self):
        with tempfile.TemporaryDirectory(prefix=".m1707-receipt-",
                                         dir=str(HW / "reviews")) as directory:
            root = Path(directory); release = root / "release.json"
            release.write_text("{}\n"); result = root / "result"; result.mkdir()
            old_release, old_seal = M.FUTURE_RELEASE, M.P.P.P.P.seal_result
            sealed = []; M.FUTURE_RELEASE = release
            M.P.P.P.P.seal_result = lambda path: sealed.append(Path(path))
            try:
                receipt = M.write_child_receipt(result, {},
                    {"missing_count": 0, "unexpected_count": 0,
                     "overlay_missing_count": 0, "overlay_unexpected_count": 0},
                    {"frames": 40, "fc_tokens": 120,
                     "patch_histogram_rows": 80})
            finally:
                M.FUTURE_RELEASE, M.P.P.P.P.seal_result = old_release, old_seal
            self.assertEqual(receipt["execution"]["frozen_layer_specs"], 32)
            self.assertEqual(receipt["execution"]["estimated_result_upper_bytes"],
                             7598737368)
            self.assertEqual(receipt["identity"]["runtime_validator_sha256"],
                             M.RUNTIME_VALIDATOR_SHA256)
            self.assertEqual(sealed, [result])


if __name__ == "__main__":
    unittest.main(verbosity=2)
