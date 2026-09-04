#!/usr/bin/env python3
"""Validator-in-the-loop tests for the inert M1692 TSBG successor."""
from __future__ import print_function

import contextlib
import copy
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
    "capture_m1692_motion_ep34_s2_tsbg_authority_shape_repair_"
    "successor_r1.py")


def load_source():
    spec = importlib.util.spec_from_file_location("test_exact_m1692", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_source()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")


def seal_review(root, value):
    root.mkdir()
    review = root / "review.json"
    write_json(review, value)
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
    outer = Path(str(path) + ".sha256.seal.sha256")
    outer.write_text(M.sha256(sidecar) + "  " + sidecar.name + "\n",
                     encoding="ascii")


def base_review():
    return {
        "schema": "m1693_m1692_tsbg_authority_shape_review_r1_v1",
        "status": M.REVIEW_STATUS,
        "score": 98,
        "p0_count": 0,
        "p1_count": 0,
        "identity": M.expected_review_identity(),
        "authorization": {
            "release_authoring": True,
            "capture": False,
            "gpu": False,
            "automatic_retry": False,
        },
    }


def base_release(identity, interpreter):
    return {
        "schema": M.RELEASE_SCHEMA,
        "status": M.RELEASE_STATUS,
        "identity": identity,
        "authorization": {
            "parent_calls": 1,
            "clean_child_processes": 1,
            "gpu_runs": 1,
            "production_captures": 1,
            "automatic_retry": False,
            "all_other_runs": 0,
        },
        "namespaces": {
            "result": str(M.RESULT.relative_to(M.ROOT)),
            "attempt": str(M.ATTEMPT.relative_to(M.ROOT)),
            "work": str(M.WORK.relative_to(M.ROOT)),
            "failure": str(M.FAILURE.relative_to(M.ROOT)),
        },
        "pre_budget_preflight": {
            "runtime_m1257_canonical": True,
            "current_entity_exact": True,
            "build_runtime_before_parent_subprocess": True,
            "build_runtime_before_child_gpu_attempt": True,
            "exact_remote_target": True,
            "exact_child_interpreter": True,
        },
        "remote_target": dict(M.REMOTE_TARGET),
        "claim_boundary": {
            "tsbg_dse": False,
            "aee": False,
            "rtl": False,
            "eda": False,
            "performance": False,
            "paper_result": False,
        },
        "child_interpreter": {
            "path": str(interpreter),
            "sha256": M.sha256(interpreter),
        },
    }


@contextlib.contextmanager
def authority_fixture(review_mutator=None, release_mutator=None):
    with tempfile.TemporaryDirectory(
            prefix=".m1692-validator-fixture-",
            dir=str(HW / "reviews")) as temporary:
        temporary = Path(temporary)
        review_root = temporary / "review"
        review = base_review()
        if review_mutator is not None:
            review_mutator(review)
        review_sha, manifest_sha, outer_sha = seal_review(review_root, review)
        identity = dict(M.expected_review_identity())
        identity.update({
            "review_sha256": review_sha,
            "review_manifest_sha256": manifest_sha,
            "review_outer_file_sha256": outer_sha,
        })
        interpreter = Path(sys.executable).resolve()
        release = base_release(identity, interpreter)
        if release_mutator is not None:
            release_mutator(release)
        release_path = temporary / "release.json"
        seal_file(release_path, release)
        old = M.FUTURE_REVIEW, M.FUTURE_RELEASE, M.CHILD_INTERPRETER
        M.FUTURE_REVIEW = review_root
        M.FUTURE_RELEASE = release_path
        M.CHILD_INTERPRETER = interpreter
        try:
            yield release
        finally:
            M.FUTURE_REVIEW, M.FUTURE_RELEASE, M.CHILD_INTERPRETER = old


class M1692SourceTests(unittest.TestCase):

    def test_01_predecessor_and_correction_are_exact(self):
        row = M.verify_predecessors()
        self.assertEqual(row["m1668_source_sha256"], M.M1668_SOURCE_SHA256)
        self.assertTrue(row["invalid_review_bound"])
        self.assertTrue(row["correction_bound"])

    def test_02_source_contract_and_new_authority_namespace(self):
        value = M.validate_source_contract()
        self.assertEqual(value["remote_target"], {
            "host": "ssh.sd5ai.scnet.cn",
            "port": 10037,
            "user": "root",
            "repository_root": "/root/private_data/work/sdformer_codex/SDformer",
        })
        self.assertEqual(value["child_interpreter_path"],
                         "/opt/conda/envs/sdformerflow/bin/python3.10")
        self.assertIn("m1693_", M.FUTURE_REVIEW.name)
        self.assertIn("m1694_", M.FUTURE_RELEASE.name)
        self.assertFalse(M.FUTURE_REVIEW.exists())
        self.assertFalse(M.FUTURE_RELEASE.exists())

    def test_03_source_self_check_is_inert_and_runtime_closed(self):
        row = M.source_self_check()
        self.assertEqual(row["status"],
                         "PASS_M1692_SOURCE_SELF_CHECK__AUTHORITY_SHAPE_REPAIRED__NO_CAPTURE")
        self.assertTrue(row["m1668_runtime_and_entity_gates_preserved"])
        self.assertEqual(row["runtime_handoff_files"], 9)
        self.assertEqual(row["runtime_canonical_files"], 7)
        self.assertEqual(row["remote_target"], M.REMOTE_TARGET)
        self.assertFalse(row["remote_connected"])
        self.assertFalse(row["checkpoint_loaded"])
        self.assertEqual(row["gpu_runs"], 0)
        self.assertEqual(row["capture_runs"], 0)
        self.assertEqual(row["attempt_writes"], 0)
        self.assertFalse(row["automatic_retry"])

    def test_04_validator_in_loop_positive_exact_shape(self):
        with authority_fixture() as expected:
            self.assertEqual(M.validate_future_authorities(), expected)

    def test_05_review_score_key_is_exact(self):
        def mutate(review):
            review["score_out_of_100"] = review.pop("score")
        with authority_fixture(review_mutator=mutate):
            with self.assertRaises(M.M1692Error):
                M.validate_future_authorities()

    def test_06_review_identity_rejects_extra_key(self):
        def mutate(review):
            review["identity"]["unsealed_alias"] = "forbidden"
        with authority_fixture(review_mutator=mutate):
            with self.assertRaises(M.M1692Error):
                M.validate_future_authorities()

    def test_07_review_authorization_rejects_extra_key(self):
        def mutate(review):
            review["authorization"]["remote_launch"] = False
        with authority_fixture(review_mutator=mutate):
            with self.assertRaises(M.M1692Error):
                M.validate_future_authorities()

    def _assert_release_rejected(self, mutator):
        with authority_fixture(release_mutator=mutator):
            with self.assertRaises(Exception):
                M.validate_future_authorities()

    def test_08_remote_host_is_exact(self):
        self._assert_release_rejected(
            lambda release: release["remote_target"].update(
                {"host": "wrong.example"}))

    def test_09_remote_port_is_exact(self):
        self._assert_release_rejected(
            lambda release: release["remote_target"].update({"port": 22}))

    def test_10_remote_user_is_exact(self):
        self._assert_release_rejected(
            lambda release: release["remote_target"].update({"user": "ubuntu"}))

    def test_11_remote_repository_root_is_exact(self):
        self._assert_release_rejected(
            lambda release: release["remote_target"].update(
                {"repository_root": "/root/wrong"}))

    def test_12_child_interpreter_path_is_exact(self):
        self._assert_release_rejected(
            lambda release: release["child_interpreter"].update(
                {"path": "/usr/bin/python"}))

    def test_13_child_interpreter_sha_is_exact(self):
        self._assert_release_rejected(
            lambda release: release["child_interpreter"].update(
                {"sha256": "0" * 64}))

    def test_14_release_authorization_is_exact(self):
        self._assert_release_rejected(
            lambda release: release["authorization"].update(
                {"remote_writes": 1}))

    def test_15_prebudget_runtime_and_order_gate_is_exact(self):
        self._assert_release_rejected(
            lambda release: release["pre_budget_preflight"].update(
                {"build_runtime_before_child_gpu_attempt": False}))

    def test_16_parent_wrapper_orders_runtime_before_delegate(self):
        events = []
        originals = (M.verify_predecessors, M.P.preflight_runtime_binding,
                     M._bound_exact_m1668, M.P.launch_parent)

        @contextlib.contextmanager
        def bound():
            events.append("bound")
            yield

        M.verify_predecessors = lambda: events.append("predecessors")
        M.P.preflight_runtime_binding = lambda: events.append("build_runtime")
        M._bound_exact_m1668 = bound
        M.P.launch_parent = lambda: events.append("parent_delegate")
        try:
            M.launch_parent()
        finally:
            (M.verify_predecessors, M.P.preflight_runtime_binding,
             M._bound_exact_m1668, M.P.launch_parent) = originals
        self.assertEqual(events, ["predecessors", "build_runtime", "bound",
                                  "parent_delegate"])

    def test_17_child_wrapper_orders_runtime_before_gpu_attempt_delegate(self):
        events = []
        originals = (M.verify_predecessors, M.P.preflight_runtime_binding,
                     M._bound_exact_m1668, M.P.fixed_clean_child)

        @contextlib.contextmanager
        def bound():
            events.append("bound")
            yield

        M.verify_predecessors = lambda: events.append("predecessors")
        M.P.preflight_runtime_binding = lambda: events.append("build_runtime")
        M._bound_exact_m1668 = bound
        M.P.fixed_clean_child = lambda: events.append("gpu_attempt_delegate")
        try:
            M.fixed_clean_child()
        finally:
            (M.verify_predecessors, M.P.preflight_runtime_binding,
             M._bound_exact_m1668, M.P.fixed_clean_child) = originals
        self.assertEqual(events, ["predecessors", "build_runtime", "bound",
                                  "gpu_attempt_delegate"])

    def test_18_bound_context_restores_m1668_module(self):
        names = ("SOURCE", "TEST", "SOURCE_CONTRACT", "FUTURE_REVIEW",
                 "FUTURE_RELEASE", "RESULT", "ATTEMPT", "WORK", "FAILURE",
                 "validate_future_authorities", "require_fresh_namespaces")
        old = dict((name, getattr(M.P, name)) for name in names)
        with M._bound_exact_m1668():
            self.assertEqual(M.P.SOURCE, M.SOURCE)
            self.assertEqual(M.P.FUTURE_REVIEW, M.FUTURE_REVIEW)
            self.assertIs(M.P.validate_future_authorities,
                          M.validate_future_authorities)
        self.assertEqual(dict((name, getattr(M.P, name)) for name in names), old)

    def test_19_inherited_lower_child_keeps_gpu_attempt_order(self):
        lower = M.P.P.P.SOURCE.read_text(encoding="utf-8")
        body = lower[lower.index("def fixed_clean_child():"):]
        self.assertIn("os.O_EXCL", lower)
        self.assertLess(body.index("m1434.build_runtime()"),
                        body.index("exclusive_gpu_lease"))
        self.assertLess(body.index("exclusive_gpu_lease"),
                        body.index("consume_attempt(release)"))
        self.assertLess(body.index("consume_attempt(release)"),
                        body.index("profile.load_config(CONFIG)"))

    def test_20_fresh_namespace_and_no_retry_contract(self):
        M.require_fresh_namespaces()
        contract = M.validate_source_contract()
        self.assertFalse(contract["authorization"]["automatic_retry"])
        self.assertFalse(contract["authorization"]["attempt_creation"])

    def test_21_capture_receipt_has_exact_future_evaluator_identity(self):
        with tempfile.TemporaryDirectory(
                prefix=".m1692-receipt-fixture-",
                dir=str(HW / "reviews")) as temporary:
            temporary = Path(temporary)
            release = temporary / "release.json"
            release.write_text("{}\n", encoding="utf-8")
            result = temporary / "result"
            result.mkdir()
            old_release = M.FUTURE_RELEASE
            old_seal = M.P.P.P.seal_result
            sealed = []
            M.FUTURE_RELEASE = release
            M.P.P.P.seal_result = lambda root: sealed.append(Path(root))
            try:
                receipt = M.write_child_receipt(
                    result,
                    {},
                    {"missing_count": 0, "unexpected_count": 0,
                     "overlay_missing_count": 0,
                     "overlay_unexpected_count": 0},
                    {"frames": 40, "fc_tokens": 120,
                     "patch_histogram_rows": 80})
            finally:
                M.FUTURE_RELEASE = old_release
                M.P.P.P.seal_result = old_seal
            observed = json.loads((result / M.RESULT_RECEIPT_NAME).read_text())
            self.assertEqual(observed, receipt)
            self.assertEqual(observed["schema"], M.RESULT_RECEIPT_SCHEMA)
            self.assertEqual(observed["status"], M.RESULT_RECEIPT_STATUS)
            self.assertEqual(observed["identity"]["source_sha256"],
                             M.sha256(M.SOURCE))
            self.assertEqual(observed["identity"]["source_contract_sha256"],
                             M.sha256(M.SOURCE_CONTRACT))
            self.assertEqual(observed["identity"]["release_sha256"],
                             M.sha256(release))
            self.assertEqual(observed["identity"][
                "m1669_correction_review_sha256"],
                M.M1669_CORRECTION_REVIEW_SHA256)
            self.assertEqual(sealed, [result])


if __name__ == "__main__":
    unittest.main()
