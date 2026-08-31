#!/usr/bin/python3.12
"""Receipt-blind, local-only hammer for M1292.

This hammer reads the M1292 source/test/contract and frozen M1257 authorities.
It does not read the M1292 author receipt, connect to the remote host, select a
checkpoint, consume a production attempt, or invoke GPU/EDA work.
"""
from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1292_m1257_motion_final_checkpoint_binder_remote_python_compat_successor.py"
AUTHOR_TEST = ROOT / "hw_autoresearch_nts07/tests/test_run_m1292_m1257_motion_final_checkpoint_binder_remote_python_compat_successor.py"
CONTRACT = ROOT / "hw_autoresearch_nts07/contracts/m1292_m1257_motion_final_checkpoint_binder_remote_python_compat_successor_source_contract_r1_20260830.json"
DOCS359 = ROOT / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"

SOURCE_SHA = "76f04b076cef298799e2899670bf60f4671d2fcb1864cb63ee476e4c8f8c49e9"
AUTHOR_TEST_SHA = "589d1b8cfa9b58e8aa93052d78b1ade8497f101abb14f6a667d0850b7bc81db0"
CONTRACT_SHA = "b7dd35d704fc4c754ac76a1858febe2ce65ac23dc8576c30294737e354c21842"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


S = load("m1294_m1292_source", SOURCE)
A = load("m1294_m1292_author_test_fixture", AUTHOR_TEST)


class Hammer(unittest.TestCase):
    def setUp(self):
        self.fx = A.M1292CompatibilitySuccessorTest(
            methodName="test_01_import_is_inert_and_m1257_is_frozen")
        self.fx.setUp()

    def tearDown(self):
        self.fx.tearDown()

    def probe(self, *_):
        value = {key: True for key in S.RUNTIME_KEYS}
        value.update(interpreter=str(S.TARGET_INTERPRETER),
                     version=S.TARGET_PYTHON_VERSION)
        return value

    def prepared(self):
        return S.prepare(self.fx.policy, S.TARGET_INTERPRETER,
                         S.TARGET_PYTHON_VERSION, self.fx.old.fx.repo,
                         probe=self.probe)

    def test_01_reviewed_bytes_and_import_are_inert(self):
        self.assertEqual(sha(SOURCE), SOURCE_SHA)
        self.assertEqual(sha(AUTHOR_TEST), AUTHOR_TEST_SHA)
        self.assertEqual(sha(CONTRACT), CONTRACT_SHA)
        self.assertEqual(sha(DOCS359), DOCS359_SHA)
        self.assertFalse(self.fx.old.fx.attempt.exists())
        self.assertFalse(self.fx.old.fx.log.exists())

    def test_02_only_production_policy_interpreter_and_version_change(self):
        old, new = S.M.PRODUCTION_POLICY, S.PRODUCTION_POLICY
        self.assertEqual(new.base.interpreter, Path("/usr/bin/python3"))
        self.assertEqual(new.base.python_version, "3.12.3")
        for field in S.M.B.Policy.__dataclass_fields__:
            if field not in {"interpreter", "python_version"}:
                self.assertEqual(getattr(new.base, field), getattr(old.base, field))
        for field in S.M.Policy.__dataclass_fields__:
            if field != "base":
                self.assertEqual(getattr(new, field), getattr(old, field))
        self.assertEqual(S.M.B.artifact_map(new.base), S.M.B.artifact_map(old.base))

    def test_03_snapshots_children_full_seals_and_compile_survive(self):
        prepared = self.prepared()
        try:
            self.assertEqual(len(prepared.snapshots), 11)
            self.assertEqual(len(prepared.source_fds), 3)
            self.assertEqual(tuple(map(int, prepared.command[-4:-1])), prepared.source_fds)
            required = (fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW |
                        fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
            for descriptor in prepared.source_fds:
                self.assertEqual(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS), required)
                payload = os.pread(descriptor, 1 << 20, 0)
                compile(payload, "<m1294-sealed-child>", "exec")
                with self.assertRaises(OSError):
                    os.write(descriptor, b"attack")
        finally:
            prepared.close()

    def test_04_version_bool_alias_and_runtime_capability_attacks_fail(self):
        attacks = []
        value = self.probe(); value["version"] = True; attacks.append(value)
        value = self.probe(); value["version"] = "3.12"; attacks.append(value)
        value = self.probe(); value["os_memfd_create"] = 1; attacks.append(value)
        value = self.probe(); value["fcntl_add_seals"] = False; attacks.append(value)
        value = self.probe(); value.pop("fcntl_seal_seal"); attacks.append(value)
        value = self.probe(); value["alias"] = True; attacks.append(value)
        for value in attacks:
            with self.subTest(value=value):
                with self.assertRaises(S.M.B.ReleaseError):
                    S.validate_runtime_probe(value)

    def test_05_cwd_repo_and_interpreter_string_drift_fail_before_base(self):
        attacks = (
            (Path("/usr/bin/python3.12"), "3.12.3", self.fx.old.fx.repo),
            (S.TARGET_INTERPRETER, "3.12.4", self.fx.old.fx.repo),
            (S.TARGET_INTERPRETER, True, self.fx.old.fx.repo),
            (S.TARGET_INTERPRETER, "3.12.3", self.fx.old.fx.repo / "alias"),
        )
        for executable, version, cwd in attacks:
            with self.subTest(executable=executable, version=version, cwd=cwd):
                with mock.patch.object(S.M, "prepare") as inherited:
                    with self.assertRaises(S.M.B.ReleaseError):
                        S.prepare(self.fx.policy, executable, version, cwd,
                                  probe=self.probe)
                    inherited.assert_not_called()

    def test_06_claim_promotion_alias_and_integer_false_fail(self):
        attacks = []
        value = dict(S.CONTRACT_CLAIM_BOUNDARY); value["hardware_speedup"] = True; attacks.append(value)
        value = dict(S.CONTRACT_CLAIM_BOUNDARY); value["paper_metric"] = 0; attacks.append(value)
        value = dict(S.CONTRACT_CLAIM_BOUNDARY); value["remote_execution_authorized_alias"] = False; attacks.append(value)
        for value in attacks:
            with self.assertRaises(S.M.B.ReleaseError):
                S.validate_contract_claim_boundary(value)

    def test_07_probe_and_compile_fail_before_attempt_consumption(self):
        bad = self.probe(); bad["os_memfd_create"] = False
        with self.assertRaises(S.M.B.ReleaseError):
            S.execute_once(self.fx.policy, S.TARGET_INTERPRETER,
                           S.TARGET_PYTHON_VERSION, self.fx.old.fx.repo,
                           probe=lambda *_: bad)
        self.assertFalse(self.fx.old.fx.attempt.exists())
        self.assertFalse(self.fx.old.fx.log.exists())

    def test_08_attempt_is_O_EXCL_and_failed_child_has_no_retry(self):
        calls = []
        def failed(command, cwd, pass_fds):
            calls.append(tuple(command))
            return subprocess.CompletedProcess(command, 23, stdout="", stderr="failure")
        with self.assertRaisesRegex(S.M.B.ReleaseError, "no retry"):
            S.execute_once(self.fx.policy, S.TARGET_INTERPRETER,
                           S.TARGET_PYTHON_VERSION, self.fx.old.fx.repo,
                           runner=failed, probe=self.probe)
        self.assertEqual(len(calls), 1)
        self.assertTrue(self.fx.old.fx.attempt.exists())
        self.assertTrue(self.fx.old.fx.log.exists())
        with self.assertRaises(S.M.B.ReleaseError):
            S.execute_once(self.fx.policy, S.TARGET_INTERPRETER,
                           S.TARGET_PYTHON_VERSION, self.fx.old.fx.repo,
                           runner=failed, probe=self.probe)
        self.assertEqual(len(calls), 1)

    def test_09_interpreter_symlink_entity_is_not_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            link = Path(tmp) / "python3"
            link.symlink_to("/usr/bin/python3.12")
            original_target = S.TARGET_INTERPRETER
            try:
                S.TARGET_INTERPRETER = link
                policy = S.rebind_interpreter(self.fx.old.policy)
                # This accepting call demonstrates the finding: lstat/realpath,
                # dev+ino, hash and an opened interpreter fd are never checked.
                S._validate_identity(link, S.TARGET_PYTHON_VERSION,
                                     self.fx.old.fx.repo, policy)
                self.assertTrue(link.is_symlink())
                self.assertNotEqual(link, link.resolve())
            finally:
                S.TARGET_INTERPRETER = original_target

    def test_10_contract_is_closed_and_production_stays_forbidden(self):
        value = json.loads(CONTRACT.read_text(encoding="utf-8"))
        self.assertEqual(value["source"]["sha256"], sha(SOURCE))
        self.assertEqual(value["test"]["sha256"], sha(AUTHOR_TEST))
        S.validate_contract_claim_boundary(value["claim_boundary"])
        self.assertFalse(value["current_readiness"]["production_execution_authorized"])
        self.assertFalse(value["current_readiness"]["attempt_consumed"])

    def test_11_source_has_no_remote_gpu_eda_or_checkpoint_action(self):
        text = SOURCE.read_text(encoding="utf-8")
        for forbidden in ("paramiko", "ssh ", "scp ", "rsync ", "import torch",
                          "nvidia-smi", "dc_shell", "vcs -full64"):
            self.assertNotIn(forbidden, text)
        self.assertFalse(self.fx.old.fx.attempt.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
