#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / ("neuron_experiments/H9_bipolar_self_attention/entrypoints/"
                 "capture_m1327_motion_ep34_consumed_namespace_bridge_r1.py")


def load_source():
    spec = importlib.util.spec_from_file_location("test_m1327_source", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load_source()


class OldNamespaceFixture:
    def __init__(self):
        self.temp = tempfile.TemporaryDirectory(prefix="m1327_old_")
        root = Path(self.temp.name)
        self.attempt = root / "old.attempt"
        self.result = root / "old.result"
        self.log = root / "old.log"
        self.attempt.write_bytes(M.M1249.ATTEMPT_TOKEN.encode("ascii"))
        self.attempt.chmod(0o400)
        self.patchers = [
            mock.patch.object(M.M1249, "CANONICAL_ATTEMPT", self.attempt),
            mock.patch.object(M.M1249, "CANONICAL_RESULT", self.result),
            mock.patch.object(M.M1249, "CANONICAL_LOG", self.log),
        ]
        for patcher in self.patchers:
            patcher.start()

    def close(self):
        for patcher in reversed(self.patchers):
            patcher.stop()
        self.temp.cleanup()


class M1327Tests(unittest.TestCase):
    def setUp(self):
        self.m1313 = M.strict_json(M.M1313_CONTRACT)

    def test_01_failed_predecessor_and_forensic_are_exact(self):
        failure = M.verify_m1326_failure()
        forensic = M.M1325.verify_m1324_forensic()
        self.assertEqual(failure["verdict"], "NO_GO_M1325_PRODUCTION_RELEASE")
        self.assertEqual(forensic["failed_execution_evidence"]["attempt_token"],
                         "M1249_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE")
        self.assertFalse(forensic["authorization"]["old_M1249_attempt_reuse"])

    def test_02_exact_consumed_state_passes_and_bridge_restores(self):
        fixture = OldNamespaceFixture()
        original = M.M1249.ensure_fresh_namespaces
        try:
            observed = M.verify_old_consumed_failure_state()
            self.assertEqual(observed["status"], "PASS_EXACT_OLD_CONSUMED_FAILURE_STATE")
            with M.old_consumed_freshness_bridge():
                self.assertIs(M.M1249.ensure_fresh_namespaces,
                              M.verify_old_consumed_failure_state)
                M.M1249.ensure_fresh_namespaces()
            self.assertIs(M.M1249.ensure_fresh_namespaces, original)
        finally:
            fixture.close()

    def test_03_missing_writable_or_wrong_old_attempt_rejected(self):
        fixture = OldNamespaceFixture()
        try:
            fixture.attempt.unlink()
            with self.assertRaisesRegex(M.M1327Error, "missing"):
                M.verify_old_consumed_failure_state()
            fixture.attempt.write_bytes(M.M1249.ATTEMPT_TOKEN.encode("ascii"))
            fixture.attempt.chmod(0o600)
            with self.assertRaisesRegex(M.M1327Error, "read-only"):
                M.verify_old_consumed_failure_state()
            fixture.attempt.chmod(0o400)
            fixture.attempt.unlink()
            fixture.attempt.write_bytes(b"wrong\n")
            fixture.attempt.chmod(0o400)
            with self.assertRaisesRegex(M.M1327Error, "token"):
                M.verify_old_consumed_failure_state()
        finally:
            fixture.close()

    def test_04_old_result_or_canonical_log_rejected(self):
        fixture = OldNamespaceFixture()
        try:
            fixture.result.write_text("unexpected")
            with self.assertRaisesRegex(M.M1327Error, "result"):
                M.verify_old_consumed_failure_state()
            fixture.result.unlink()
            fixture.log.write_text("unexpected")
            with self.assertRaisesRegex(M.M1327Error, "canonical log"):
                M.verify_old_consumed_failure_state()
        finally:
            fixture.close()

    def test_05_bridge_restores_after_failure(self):
        fixture = OldNamespaceFixture()
        original = M.M1249.ensure_fresh_namespaces
        try:
            fixture.attempt.chmod(0o600)
            with self.assertRaises(M.M1327Error):
                with M.old_consumed_freshness_bridge():
                    M.M1249.ensure_fresh_namespaces()
            self.assertIs(M.M1249.ensure_fresh_namespaces, original)
        finally:
            fixture.close()

    def test_06_identity_path_exercises_real_freshness_hook_not_mock(self):
        fixture = OldNamespaceFixture()
        original = M.M1249.ensure_fresh_namespaces
        calls = []
        binding = {"identity": {"m1327": "test"}}

        def identity_validator(path, entry):
            # This is the unchanged validator's exact freshness call site.  The
            # callback itself is real and reads the actual temporary old-state files.
            calls.append(M.M1249.ensure_fresh_namespaces())
            return copy.deepcopy(self.m1313), binding
        try:
            with mock.patch.object(M.M1319, "validate_exact_m1313_m1314",
                                   side_effect=identity_validator):
                runtime, observed = M.validate_identity_and_project()
            self.assertEqual(calls[0]["status"], "PASS_EXACT_OLD_CONSUMED_FAILURE_STATE")
            self.assertEqual(runtime["capture"], {"attention_windows_per_call": 100})
            self.assertEqual(runtime["output"],
                             {"path": str(M.CANONICAL_RESULT.relative_to(M.ROOT))})
            self.assertIs(observed, binding)
            self.assertIs(M.M1249.ensure_fresh_namespaces, original)
        finally:
            fixture.close()

    def test_07_runtime_exact_four_keys_and_deep_copy(self):
        runtime = M.build_runtime_contract(self.m1313)
        self.assertEqual(set(runtime), M.RUNTIME_KEYS)
        self.assertEqual(runtime["capture"], {"attention_windows_per_call": 100})
        before = copy.deepcopy(self.m1313)
        runtime["cohort"]["samples"][0]["bytes"] = 1
        self.assertEqual(self.m1313, before)
        changed = M.build_runtime_contract(self.m1313)
        changed["output"] = {"path": "old"}
        with self.assertRaises(M.M1327Error):
            M.validate_runtime_contract(changed, self.m1313)

    def test_08_new_namespaces_pairwise_disjoint_and_fresh_gate(self):
        self.assertEqual(len({M.CANONICAL_RESULT, M.CANONICAL_ATTEMPT, M.CANONICAL_LOG}), 3)
        self.assertTrue(all("m1327" in path.name for path in
                            (M.CANONICAL_RESULT, M.CANONICAL_ATTEMPT, M.CANONICAL_LOG)))
        self.assertFalse(any("m1249" in str(path) for path in
                             (M.CANONICAL_RESULT, M.CANONICAL_ATTEMPT, M.CANONICAL_LOG)))
        with tempfile.TemporaryDirectory(prefix="m1327_new_") as directory:
            root = Path(directory)
            paths = [root / "result", root / "attempt", root / "log"]
            with mock.patch.object(M, "CANONICAL_RESULT", paths[0]), \
                 mock.patch.object(M, "CANONICAL_ATTEMPT", paths[1]), \
                 mock.patch.object(M, "CANONICAL_LOG", paths[2]):
                M.require_fresh_m1327_namespaces()
                paths[1].write_text("occupied")
                with self.assertRaisesRegex(M.M1327Error, "not fresh"):
                    M.require_fresh_m1327_namespaces()

    def test_09_delegate_propagates_new_output_and_restores_on_exception(self):
        runtime = M.build_runtime_contract(self.m1313)
        binding = {"policy": {}, "verified_samples": [], "identity": {}, "selection": {},
                   "checkpoint_path": Path("checkpoint"), "config_path": Path("config")}
        old = M.M1249.CANONICAL_RESULT
        observed = []
        def success(contract, child_binding, substrate=None):
            observed.append((copy.deepcopy(contract), child_binding, substrate,
                             M.M1249.CANONICAL_RESULT))
            return M.M1249.CANONICAL_RESULT
        substrate = object()
        with mock.patch.object(M.M1249, "run_capture", side_effect=success):
            output = M.delegate_for_future_release(runtime, binding, substrate)
        self.assertEqual(output, M.CANONICAL_RESULT)
        self.assertEqual(observed[0][3], M.CANONICAL_RESULT)
        self.assertIs(M.M1249.CANONICAL_RESULT, old)
        with mock.patch.object(M.M1249, "run_capture", side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                M.delegate_for_future_release(runtime, binding, substrate)
        self.assertIs(M.M1249.CANONICAL_RESULT, old)

    def test_10_source_cli_is_inert_and_has_no_attempt_consumer(self):
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("--source-self-check", source)
        self.assertNotIn("--run", source)
        self.assertNotIn("consume_attempt()", source)
        self.assertNotIn("exclusive_gpu_lease", source)
        self.assertNotIn("os.O_EXCL", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
