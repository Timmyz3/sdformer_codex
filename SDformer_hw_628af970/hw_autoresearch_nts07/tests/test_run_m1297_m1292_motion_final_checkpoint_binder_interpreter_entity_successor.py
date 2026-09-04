from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/run_m1297_m1292_motion_final_checkpoint_binder_interpreter_entity_successor.py"
OLD_TEST = ROOT / "tests/test_run_m1292_m1257_motion_final_checkpoint_binder_remote_python_compat_successor.py"
CONTRACT = ROOT / "contracts/m1297_m1292_motion_final_checkpoint_binder_interpreter_entity_successor_source_contract_r1_20260830.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


S = load("m1297_source_under_test", SOURCE)
T = load("m1297_m1292_fixture", OLD_TEST)


class M1297InterpreterEntitySuccessorTest(unittest.TestCase):
    def setUp(self):
        self.old = T.M1292CompatibilitySuccessorTest(
            methodName="test_01_import_is_inert_and_m1257_is_frozen")
        self.old.setUp()
        self.temp = tempfile.TemporaryDirectory(prefix="m1297.entity.")
        self.real = Path("/usr/bin/python3.12")
        self.link = Path(self.temp.name) / "python3"
        self.link.symlink_to(self.real)
        descriptor = os.open(self.real, os.O_RDONLY | os.O_CLOEXEC)
        try:
            self.runtime = S.probe_fd_runtime(descriptor)
            self.expected = S.entity_from_fd(
                descriptor, self.runtime["version"], self.runtime["memfd_and_all_seals"])
        finally:
            os.close(descriptor)
        self.policy = S.rebind_policy(self.old.old.policy, self.link,
                                      self.expected["version"])

    def tearDown(self):
        self.temp.cleanup(); self.old.tearDown()

    def prepared(self):
        return S.prepare(self.policy, self.old.old.fx.repo, self.link,
                         self.real, copy.deepcopy(self.expected))

    def test_01_import_inert_and_m1292_frozen(self):
        self.assertEqual(sha(S.M1292_SOURCE), S.M1292_SOURCE_SHA256)
        self.assertFalse(self.old.old.fx.attempt.exists())
        self.assertFalse(self.old.old.output.exists())

    def test_02_remote_production_entity_is_exactly_pinned(self):
        self.assertEqual(S.TARGET_LINK, Path("/usr/bin/python3"))
        self.assertEqual(S.TARGET_REALPATH, Path("/usr/bin/python3.12"))
        self.assertEqual(S.TARGET_ENTITY, {
            "device": 1048625, "inode": 1347357695, "mode": 0x81ED,
            "size_bytes": 8020928, "mtime_sec": 1774292672,
            "sha256": "e1efa562c2cc2e35521a5c9c9b9939921001ff8ca9708a13ef15ace68cc2ccd7",
            "version": "3.12.3", "memfd_and_all_seals": True})

    def test_03_fd_probe_measures_running_entity_not_caller_label(self):
        descriptor = os.open(self.real, os.O_RDONLY | os.O_CLOEXEC)
        try:
            measured = S.probe_fd_runtime(descriptor)
        finally:
            os.close(descriptor)
        self.assertEqual(measured["version"], platform.python_version())
        self.assertTrue(measured["memfd_and_all_seals"])

    def test_04_open_uses_realpath_and_retained_regular_executable_fd(self):
        handle = S.open_interpreter_entity(self.link, self.real, self.expected)
        try:
            self.assertEqual(os.fstat(handle.descriptor).st_ino, self.expected["inode"])
            self.assertTrue(stat.S_ISREG(os.fstat(handle.descriptor).st_mode))
            self.assertEqual(S.sha_fd(handle.descriptor), self.expected["sha256"])
        finally:
            handle.close()

    def test_05_all_entity_fields_and_exact_types_are_enforced(self):
        for key in S.ENTITY_KEYS:
            bad = copy.deepcopy(self.expected)
            bad[key] = (not bad[key]) if key == "memfd_and_all_seals" else (
                bad[key] + 1 if type(bad[key]) is int else str(bad[key]) + "x")
            with self.subTest(key=key):
                with self.assertRaises(S.M.M.B.ReleaseError):
                    S.open_interpreter_entity(self.link, self.real, bad)
        bad = copy.deepcopy(self.expected); bad["device"] = True
        with self.assertRaises(S.M.M.B.ReleaseError):
            S.open_interpreter_entity(self.link, self.real, bad)

    def test_06_symlink_retarget_after_open_is_detected(self):
        handle = S.open_interpreter_entity(self.link, self.real, self.expected)
        try:
            self.link.unlink(); self.link.symlink_to("/bin/false")
            with self.assertRaises(S.M.M.B.ReleaseError):
                S.revalidate_entity(handle, self.expected)
        finally:
            handle.close()

    def test_07_prepare_is_fd_bound_and_preserves_three_sealed_sources(self):
        prepared = self.prepared()
        try:
            self.assertEqual(prepared.command[0],
                             "/proc/self/fd/{}".format(prepared.interpreter.descriptor))
            self.assertEqual(len(prepared.source_fds), 3)
            self.assertEqual(set(prepared.pass_fds),
                             set(prepared.source_fds) | {prepared.interpreter.descriptor})
            self.assertEqual(len(prepared.snapshots), 11)
        finally:
            prepared.close()

    def test_08_attempt_binds_entity_and_is_persistent_o_excl(self):
        prepared = self.prepared()
        try:
            S.revalidate_entity(prepared.interpreter, self.expected)
            S.consume_attempt(prepared)
            text = self.old.old.fx.attempt.read_text()
            wanted = S.M.M.B.sha256_bytes(json.dumps(
                prepared.interpreter.identity, sort_keys=True,
                separators=(",", ":")).encode())
            self.assertIn("interpreter_entity_sha256=" + wanted, text)
            with self.assertRaises(FileExistsError):
                S.consume_attempt(prepared)
        finally:
            prepared.close()

    def test_09_failed_child_runs_once_by_fd_and_no_retry(self):
        calls = []
        def failed(command, cwd, pass_fds):
            calls.append((tuple(command), tuple(pass_fds)))
            self.assertTrue(command[0].startswith("/proc/self/fd/"))
            self.assertIn(int(command[0].rsplit("/", 1)[1]), pass_fds)
            return subprocess.CompletedProcess(command, 9, stdout="", stderr="fail")
        with self.assertRaisesRegex(S.M.M.B.ReleaseError, "no retry"):
            S.execute_once(self.policy, self.old.old.fx.repo, self.link,
                           self.real, self.expected, runner=failed)
        self.assertEqual(len(calls), 1)
        with self.assertRaises(Exception):
            S.execute_once(self.policy, self.old.old.fx.repo, self.link,
                           self.real, self.expected, runner=failed)
        self.assertEqual(len(calls), 1)

    def test_10_path_retarget_before_attempt_consumes_nothing(self):
        prepared = self.prepared()
        try:
            self.link.unlink(); self.link.symlink_to("/bin/false")
            with self.assertRaises(S.M.M.B.ReleaseError):
                S.revalidate_entity(prepared.interpreter, self.expected)
            self.assertFalse(self.old.old.fx.attempt.exists())
        finally:
            prepared.close()

    def test_11_production_entry_zero_arg_and_no_remote_action(self):
        import inspect
        self.assertEqual(len(inspect.signature(S.main).parameters), 0)
        text = SOURCE.read_text()
        for forbidden in ("paramiko", "ssh ", "scp ", "rsync ", "import torch",
                          "nvidia-smi", "dc_shell", "vcs -full64"):
            self.assertNotIn(forbidden, text)

    def test_12_contract_identity_when_present_and_docs_frozen(self):
        docs = ROOT / "docs/359_DATE终局冻结_20260813.md"
        self.assertEqual(sha(docs), S.M.M.B.DOCS359_SHA256)
        if CONTRACT.exists():
            value = json.loads(CONTRACT.read_text())
            self.assertEqual(value["source"]["sha256"], sha(SOURCE))
            self.assertEqual(value["test"]["sha256"], sha(Path(__file__).resolve()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
