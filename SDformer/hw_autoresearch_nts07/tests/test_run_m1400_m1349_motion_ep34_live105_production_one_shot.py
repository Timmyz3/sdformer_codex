#!/usr/bin/env python3
from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1400_m1349_motion_ep34_live105_production_one_shot.py"
spec = importlib.util.spec_from_file_location("test_m1400_source", SOURCE)
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)


def completed(stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


class Tests(unittest.TestCase):
    def controller_fixture(self, root: Path, state="T", ppid=1, count=1):
        roots = []
        for offset in range(count):
            pid = root / str(500 + offset)
            pid.mkdir()
            (pid / "cmdline").write_bytes(b"\0".join(x.encode() for x in M.CONTROLLER_ARGV) + b"\0")
            fields = [state, str(ppid)] + ["0"] * 17 + [str(7000 + offset)]
            (pid / "stat").write_text(f"{pid.name} (python) " + " ".join(fields) + "\n")
            (pid / "cwd").touch(); (pid / "exe").touch(); roots.append(pid)
        return roots

    def readlink(self, path):
        return M.CONTROLLER_EXE if Path(path).name == "exe" else str(M.REMOTE_ROOT)

    def gpu_runner(self, command, **_kwargs):
        if "--query-gpu=index,uuid,name,memory.used,memory.total" in command:
            return completed(f"0, {M.GPU_UUID}, {M.GPU_NAME}, 0, {M.GPU_TOTAL_MIB}\n")
        return completed("")

    def test_01_prerequisite_author_and_blind_seals(self):
        M.verify_prerequisites()

    def test_02_source_contract_and_command_self_proof(self):
        policy = M.validate_source_contract()
        self.assertFalse(policy["launch_authorized"])
        self.assertEqual(policy["commands"], M.source_commands())

    def test_03_source_absent_positive_no_attempt(self):
        with mock.patch.object(M, "future_paths", return_value=(ROOT / ".m1400_abs1",)), \
             mock.patch.object(M, "inspect_gpu") as gpu, \
             mock.patch.object(M, "consume_attempt") as consume:
            M.source_absent_self_check()
        gpu.assert_not_called(); consume.assert_not_called()

    def test_04_source_absent_rejects_future_residue(self):
        with tempfile.TemporaryDirectory() as raw:
            residue = Path(raw) / "blind"; residue.write_text("x")
            with mock.patch.object(M, "future_paths", return_value=(residue,)), \
                 self.assertRaisesRegex(M.M1400Error, "future"):
                M.source_absent_self_check()

    def test_05_strict_json_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "x.json"; path.write_text('{"a":1,"a":2}')
            with self.assertRaisesRegex(M.M1400Error, "duplicate"):
                M.strict_json(path)

    def test_06_external_sha_absence_fails(self):
        with self.assertRaisesRegex(M.M1400Error, "external SHA"):
            M.external_bindings({})

    def test_07_exact_controller_passes(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); self.controller_fixture(root)
            with mock.patch.object(M.os, "readlink", side_effect=self.readlink):
                value = M.inspect_controller(root)
            self.assertEqual(value["state"], "T"); self.assertEqual(value["ppid"], 1)

    def test_08_running_controller_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); self.controller_fixture(root, state="S")
            with mock.patch.object(M.os, "readlink", side_effect=self.readlink), \
                 self.assertRaisesRegex(M.M1400Error, "not exact"):
                M.inspect_controller(root)

    def test_09_duplicate_controller_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); self.controller_fixture(root, count=2)
            with mock.patch.object(M.os, "readlink", side_effect=self.readlink), \
                 self.assertRaisesRegex(M.M1400Error, "exactly one"):
                M.inspect_controller(root)

    def test_10_wrong_controller_ppid_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); self.controller_fixture(root, ppid=2)
            with mock.patch.object(M.os, "readlink", side_effect=self.readlink), \
                 self.assertRaisesRegex(M.M1400Error, "not exact"):
                M.inspect_controller(root)

    def test_11_gpu_idle_passes(self):
        value = M.inspect_gpu(self.gpu_runner)
        self.assertEqual(value["compute_apps"], []); self.assertEqual(value["memory_used_mib"], 0)

    def test_12_gpu_app_rejected(self):
        def runner(command, **kwargs):
            if "--query-gpu=index,uuid,name,memory.used,memory.total" in command:
                return self.gpu_runner(command, **kwargs)
            return completed(f"123, {M.GPU_UUID}\n")
        with self.assertRaisesRegex(M.M1400Error, "compute"):
            M.inspect_gpu(runner)

    def test_13_gpu_query_failure_rejected(self):
        with self.assertRaisesRegex(M.M1400Error, "query failed"):
            M.inspect_gpu(lambda *a, **k: completed(returncode=1))

    def test_14_gpu_memory_busy_rejected(self):
        def runner(command, **_kwargs):
            if "--query-gpu=index,uuid,name,memory.used,memory.total" in command:
                return completed(f"0, {M.GPU_UUID}, {M.GPU_NAME}, 65, {M.GPU_TOTAL_MIB}\n")
            return completed("")
        with self.assertRaisesRegex(M.M1400Error, "idleness"):
            M.inspect_gpu(runner)

    def test_15_attempt_is_structured_O_EXCL_no_retry(self):
        controller = {"pid": 1, "state": "T"}
        values = {"M1400_EXPECTED_RUNNER_SHA256": "a" * 64}
        with tempfile.TemporaryDirectory() as raw:
            marker = Path(raw) / "attempt"
            with mock.patch.object(M, "CANONICAL_ATTEMPT", marker):
                M.consume_attempt(controller, values)
                data = M.strict_json(marker)
                self.assertFalse(data["automatic_retry"])
                self.assertFalse(data["controller_restore_permitted"])
                with self.assertRaises(FileExistsError): M.consume_attempt(controller, values)

    def test_16_failure_and_success_restore_boundary(self):
        controller = {"pid": 1, "state": "T"}
        failed = json.loads(M.log_payload("FAIL", controller, "x"))
        passed = json.loads(M.log_payload("PASS", controller, "x"))
        self.assertFalse(failed["controller_restore_permitted"])
        self.assertTrue(passed["controller_restore_permitted_after_success"])
        self.assertFalse(passed["controller_restored_by_runner"])

    def test_17_atomic_log_no_replace(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); canonical = root / "p.log"; temp = root / "p.log.tmp.x"
            with mock.patch.object(M, "CANONICAL_LOG", canonical):
                M.publish_log(temp, b"one")
                self.assertEqual(canonical.read_bytes(), b"one")
                with self.assertRaises(M.M1400Error): M.publish_log(root / "p.log.tmp.y", b"two")
            self.assertEqual(canonical.read_bytes(), b"one")

    def test_18_namespaces_collision_rejected(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); result = root / "result"; result.mkdir()
            with mock.patch.object(M, "CANONICAL_RESULT", result), \
                 mock.patch.object(M, "CANONICAL_ATTEMPT", root / "a"), \
                 mock.patch.object(M, "CANONICAL_LOG", root / "l"), \
                 self.assertRaisesRegex(M.M1400Error, "not fresh"):
                M.namespaces_fresh()

    def test_19_remote_preflight_rechecks_controller_and_gpu(self):
        runtime, binding = {"x": 1}, {"checkpoint_path": "x", "config_path": "y"}
        controller = {"pid": 1, "state": "T"}; values = {"x": "y"}
        with mock.patch.object(M, "ROOT", M.REMOTE_ROOT), mock.patch.object(M.Path, "cwd", return_value=M.REMOTE_ROOT), \
             mock.patch.object(M, "verify_prerequisites"), mock.patch.object(M, "validate_source_contract"), \
             mock.patch.object(M, "external_bindings", return_value=values), mock.patch.object(M, "validate_future_authorities"), \
             mock.patch.object(M, "namespaces_fresh") as fresh, mock.patch.object(M, "inspect_controller", return_value=controller) as ctl, \
             mock.patch.object(M, "inspect_gpu", return_value={}) as gpu, mock.patch.object(M.M1349, "build_runtime", return_value=(runtime, binding)), \
             mock.patch.object(M, "validate_bound_capture_files"):
            observed = M.remote_preflight({})
        self.assertEqual(observed, (runtime, binding, controller, values))
        self.assertEqual(fresh.call_count, 2); self.assertEqual(ctl.call_count, 2); self.assertEqual(gpu.call_count, 2)

    def test_20_no_controller_signal_or_restore_primitive(self):
        source = SOURCE.read_text(encoding="utf-8")
        for forbidden in ("os.kill", "SIGCONT", "send_signal", "kill("):
            self.assertNotIn(forbidden, source)
        self.assertIn("controller_restored_by_runner", source)

    def test_21_execute_failure_before_attempt_writes_no_log(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw); log = root / "p.log"; temp = root / "p.log.tmp.x"
            with mock.patch.object(M, "CANONICAL_LOG", log), \
                 mock.patch.object(M, "remote_preflight", side_effect=M.M1400Error("pre")), \
                 self.assertRaises(M.M1400Error):
                M.execute_once(temp)
            self.assertFalse(log.exists())

    def test_22_m1349_exact_population_binding(self):
        self.assertEqual(M.M1349.EXPECTED_ATLIF_COUNT, 105)
        self.assertEqual(M.M1349.EXPECTED_ORDERED_RECORDS, 10360)
        self.assertEqual(M.M1349.EXPECTED_PAYLOAD, 640)


if __name__ == "__main__":
    unittest.main(verbosity=2)
