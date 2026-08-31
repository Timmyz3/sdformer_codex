#!/usr/bin/python3.12
"""Receipt-blind local hammer for the M1297 fd-bound interpreter successor."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1297_m1292_motion_final_checkpoint_binder_interpreter_entity_successor.py"
AUTHOR_TEST = ROOT / "hw_autoresearch_nts07/tests/test_run_m1297_m1292_motion_final_checkpoint_binder_interpreter_entity_successor.py"
CONTRACT = ROOT / "hw_autoresearch_nts07/contracts/m1297_m1292_motion_final_checkpoint_binder_interpreter_entity_successor_source_contract_r1_20260830.json"
M1292_CONTRACT = ROOT / "hw_autoresearch_nts07/contracts/m1292_m1257_motion_final_checkpoint_binder_remote_python_compat_successor_source_contract_r1_20260830.json"
DOCS359 = ROOT / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"

SOURCE_SHA = "336195e40cf07cfa273be650f2edd0cf1c537c8c70b1c39e68515c087ca81899"
AUTHOR_TEST_SHA = "1dddabbc1334ed98633e556898cf5df74a4b49089f70c253cae2cf5e408563de"
CONTRACT_SHA = "ace730ff38df4ba5025afb46edcd90e6913ef9806058570e2db2db04fdf35cb2"
M1292_CONTRACT_SHA = "b7dd35d704fc4c754ac76a1858febe2ce65ac23dc8576c30294737e354c21842"
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


S = load("m1298_m1297_source", SOURCE)
A = load("m1298_m1297_author_fixture", AUTHOR_TEST)


class Hammer(unittest.TestCase):
    def setUp(self):
        self.fx = A.M1297InterpreterEntitySuccessorTest(
            methodName="test_01_import_inert_and_m1292_frozen")
        self.fx.setUp()

    def tearDown(self):
        self.fx.tearDown()

    def test_01_reviewed_bytes_import_and_docs_are_frozen(self):
        self.assertEqual(sha(SOURCE), SOURCE_SHA)
        self.assertEqual(sha(AUTHOR_TEST), AUTHOR_TEST_SHA)
        self.assertEqual(sha(CONTRACT), CONTRACT_SHA)
        self.assertEqual(sha(M1292_CONTRACT), M1292_CONTRACT_SHA)
        self.assertEqual(sha(DOCS359), DOCS359_SHA)
        self.assertEqual(sha(S.M1292_SOURCE), S.M1292_SOURCE_SHA256)
        self.assertFalse(self.fx.old.old.fx.attempt.exists())

    def test_02_m1292_and_m1257_policy_values_do_not_drift(self):
        old, new = S.M.PRODUCTION_POLICY, S.PRODUCTION_POLICY
        for field in S.M.M.B.Policy.__dataclass_fields__:
            self.assertEqual(getattr(new.base, field), getattr(old.base, field))
        for field in S.M.M.Policy.__dataclass_fields__:
            if field != "base":
                self.assertEqual(getattr(new, field), getattr(old, field))
        self.assertEqual(S.M.M.B.artifact_map(new.base), S.M.M.B.artifact_map(old.base))
        self.assertEqual(new.base.execution_pins, old.base.execution_pins)

    def test_03_all_entity_fields_types_version_and_capability_fail_closed(self):
        for key in S.ENTITY_KEYS:
            bad = copy.deepcopy(self.fx.expected)
            if key == "memfd_and_all_seals":
                bad[key] = False
            elif type(bad[key]) is int:
                bad[key] += 1
            else:
                bad[key] += "_drift"
            with self.subTest(key=key):
                with self.assertRaises(S.M.M.B.ReleaseError):
                    S.open_interpreter_entity(self.fx.link, self.fx.real, bad)
        for key in ("device", "inode", "mode", "size_bytes", "mtime_sec"):
            bad = copy.deepcopy(self.fx.expected); bad[key] = True
            with self.assertRaises(S.M.M.B.ReleaseError):
                S.validate_entity(self.fx.expected, bad)
        bad = copy.deepcopy(self.fx.expected); bad["alias"] = False
        with self.assertRaises(S.M.M.B.ReleaseError):
            S.validate_entity(self.fx.expected, bad)

    def test_04_logical_symlink_and_realpath_attacks_fail_before_snapshots(self):
        self.fx.link.unlink(); self.fx.link.symlink_to("/bin/false")
        with mock.patch.object(S.M.M, "prepare") as inherited:
            with self.assertRaises(S.M.M.B.ReleaseError):
                S.prepare(self.fx.policy, self.fx.old.old.fx.repo, self.fx.link,
                          self.fx.real, self.fx.expected)
            inherited.assert_not_called()
        self.assertFalse(self.fx.old.old.fx.attempt.exists())

    def test_05_path_replacement_after_prepare_fails_before_attempt(self):
        prepared = self.fx.prepared()
        try:
            self.fx.link.unlink(); self.fx.link.symlink_to("/bin/false")
            with self.assertRaises(S.M.M.B.ReleaseError):
                S.revalidate_entity(prepared.interpreter, self.fx.expected)
            self.assertFalse(self.fx.old.old.fx.attempt.exists())
        finally:
            prepared.close()

    def test_06_interpreter_fd_close_and_replacement_fail_before_attempt(self):
        for attack in ("close", "replace"):
            with self.subTest(attack=attack):
                prepared = self.fx.prepared()
                fd = prepared.interpreter.descriptor
                os.close(fd)
                replacement = None
                if attack == "replace":
                    replacement = os.open("/bin/false", os.O_RDONLY | os.O_CLOEXEC)
                    self.assertEqual(replacement, fd)
                try:
                    with self.assertRaises((OSError, S.M.M.B.ReleaseError)):
                        S.revalidate_entity(prepared.interpreter, self.fx.expected)
                    self.assertFalse(self.fx.old.old.fx.attempt.exists())
                finally:
                    prepared.close()

    def test_07_proc_fd_child_requires_and_accepts_exact_pass_fd(self):
        fd = os.open(self.fx.real, os.O_RDONLY | os.O_CLOEXEC)
        command = ["/proc/self/fd/{}".format(fd), "-I", "-B", "-c",
                   "print('M1298_FD_CHILD_PASS')"]
        try:
            good = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, pass_fds=(fd,), check=False)
            self.assertEqual(good.returncode, 0, good.stderr)
            self.assertEqual(good.stdout.strip(), "M1298_FD_CHILD_PASS")
            with self.assertRaises(FileNotFoundError):
                subprocess.run(command, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, pass_fds=(), check=False)
        finally:
            os.close(fd)
        prepared = self.fx.prepared()
        try:
            self.assertEqual(prepared.pass_fds,
                             prepared.source_fds + (prepared.interpreter.descriptor,))
            self.assertEqual(len(prepared.pass_fds), 4)
            self.assertEqual(prepared.command[0],
                             "/proc/self/fd/{}".format(prepared.interpreter.descriptor))
        finally:
            prepared.close()

    def test_08_fd_probe_defeats_local_version_spoof(self):
        fd = os.open(self.fx.real, os.O_RDONLY | os.O_CLOEXEC)
        try:
            runtime = S.probe_fd_runtime(fd)
            observed = S.entity_from_fd(fd, runtime["version"],
                                        runtime["memfd_and_all_seals"])
            spoof = copy.deepcopy(observed); spoof["version"] = "3.12.3-spoof"
            with self.assertRaises(S.M.M.B.ReleaseError):
                S.validate_entity(observed, spoof)
            self.assertEqual(runtime["version"], self.fx.runtime["version"])
        finally:
            os.close(fd)

    def test_09_attempt_digest_binds_pre_and_post_revalidated_entity(self):
        prepared = self.fx.prepared()
        try:
            self.assertFalse(self.fx.old.old.fx.attempt.exists())
            S.revalidate_entity(prepared.interpreter, self.fx.expected)
            S.consume_attempt(prepared)
            body = self.fx.old.old.fx.attempt.read_text(encoding="utf-8")
            digest = S.M.M.B.sha256_bytes(json.dumps(
                prepared.interpreter.identity, sort_keys=True,
                separators=(",", ":")).encode())
            self.assertIn("interpreter_entity_sha256=" + digest, body)
            self.assertIn("automatic_retry=false", body)
            with self.assertRaises(FileExistsError):
                S.consume_attempt(prepared)
        finally:
            prepared.close()

    def test_10_failed_child_is_once_only_and_claims_do_not_promote(self):
        calls = []
        def failed(command, cwd, pass_fds):
            calls.append((tuple(command), tuple(pass_fds)))
            return subprocess.CompletedProcess(command, 17, stdout="", stderr="failed")
        with self.assertRaisesRegex(S.M.M.B.ReleaseError, "no retry"):
            S.execute_once(self.fx.policy, self.fx.old.old.fx.repo, self.fx.link,
                           self.fx.real, self.fx.expected, runner=failed)
        self.assertEqual(len(calls), 1)
        with self.assertRaises(Exception):
            S.execute_once(self.fx.policy, self.fx.old.old.fx.repo, self.fx.link,
                           self.fx.real, self.fx.expected, runner=failed)
        self.assertEqual(len(calls), 1)
        inherited = json.loads(M1292_CONTRACT.read_text())["claim_boundary"]
        for key, value in inherited.items():
            self.assertIs(type(value), bool)
            self.assertFalse(value)

    def test_11_contract_claim_keyset_drift_is_reproduced(self):
        inherited = json.loads(M1292_CONTRACT.read_text())["claim_boundary"]
        successor = json.loads(CONTRACT.read_text())["claim_boundary"]
        self.assertNotEqual(set(successor), set(inherited))
        self.assertEqual(set(inherited) - set(successor),
                         {"checkpoint_selected_now", "remote_execution_authorized"})
        self.assertEqual(set(successor) - set(inherited), {"paper_ppa_ready"})
        self.assertTrue(all(type(value) is bool and value is False
                            for value in successor.values()))

    def test_12_source_is_local_inert_zero_arg_and_no_remote_gpu_eda(self):
        import inspect
        self.assertEqual(len(inspect.signature(S.main).parameters), 0)
        text = SOURCE.read_text(encoding="utf-8")
        for forbidden in ("paramiko", "ssh ", "scp ", "rsync ", "import torch",
                          "nvidia-smi", "dc_shell", "vcs -full64"):
            self.assertNotIn(forbidden, text)
        self.assertFalse(self.fx.old.old.fx.attempt.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
