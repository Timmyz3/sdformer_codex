#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1206 local-only tests: no SSH, SCP, GPU, capture, or EDA."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1206_m1182_m1180_missing2_secure_mktemp_monotonic_transport_source.py"
SPEC = importlib.util.spec_from_file_location("m1206_secure", SOURCE)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


class M1206Tests(unittest.TestCase):
    def row(self, payload: bytes, path: str = "a/one") -> dict:
        return {"path": path, "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest()}

    def write(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)

    def archive(self, path: Path, rows: list[dict], payloads: list[bytes],
                names: list[str] | None = None, symlink: bool = False) -> None:
        with tarfile.open(path, "w") as archive:
            for index, (row, payload) in enumerate(zip(rows, payloads)):
                info = tarfile.TarInfo((names or [r["path"] for r in rows])[index])
                if symlink and index == 0:
                    info.type = tarfile.SYMTYPE; info.linkname = "elsewhere"; info.size = 0
                    archive.addfile(info)
                else:
                    info.size = len(payload); archive.addfile(info, io.BytesIO(payload))

    def completed(self, code: int = 0, stdout: bytes = b"", stderr: bytes = b""):
        return subprocess.CompletedProcess(["mock"], code, stdout, stderr)

    def test_01_contract_authorities_and_docs359(self) -> None:
        contract = M.load_contract(); M.verify_policy(contract)
        self.assertEqual(len(M.exact_members(contract)), 2)
        self.assertEqual(M.sha256(ROOT / M.DOCS359_REL), M.DOCS359_SHA256)

    def test_02_mktemp_stdout_exact_anchor_only(self) -> None:
        good = b"/tmp/m1206_m1180.Ab12Cd34Ef56\n"
        self.assertEqual(M.validate_temp_path_text(good), Path(good.decode().strip()))
        attacks = [b"relative\n", b"/tmp/m1206_m1180.short\n",
                   good + b"/tmp/m1206_m1180.Zz99Yy88Xx77\n",
                   b"noise /tmp/m1206_m1180.Ab12Cd34Ef56\n",
                   b"/tmp/m1206_m1180.Ab12Cd34Ef56", b"\xff\n"]
        for attack in attacks:
            with self.assertRaises(M.TransportError):
                M.validate_temp_path_text(attack)

    def test_03_temp_lstat_type_owner_and_mode_attacks(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m1206_parent_") as parent:
            root = Path(parent); path = root / "m1206_m1180.Ab12Cd34Ef56"
            old_pattern = M.REMOTE_TEMP_RE
            M.REMOTE_TEMP_RE = __import__("re").compile(
                r"\A" + __import__("re").escape(path.as_posix()) + r"\Z")
            try:
                path.mkdir(mode=0o700)
                M.validate_temp_directory(path, os.getuid())
                with self.assertRaises(M.TransportError):
                    M.validate_temp_directory(path, os.getuid() + 1)
                path.chmod(0o755)
                with self.assertRaises(M.TransportError):
                    M.validate_temp_directory(path, os.getuid())
                path.chmod(0o700); path.rmdir(); path.write_bytes(b"not-dir")
                with self.assertRaises(M.TransportError):
                    M.validate_temp_directory(path, os.getuid())
                path.unlink(); target = root / "target"; target.mkdir()
                path.symlink_to(target, target_is_directory=True)
                with self.assertRaises(M.TransportError):
                    M.validate_temp_directory(path, os.getuid())
            finally:
                M.REMOTE_TEMP_RE = old_pattern

    def test_04_archive_lstat_size_sha_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m1206_archive_path_") as temporary:
            root = Path(temporary); archive = root / M.REMOTE_ARCHIVE_BASENAME
            archive.write_bytes(b"archive")
            M.validate_archive_path(archive, 7, hashlib.sha256(b"archive").hexdigest())
            for size, digest in ((8, hashlib.sha256(b"archive").hexdigest()),
                                 (7, "0" * 64)):
                with self.assertRaises(M.TransportError):
                    M.validate_archive_path(archive, size, digest)
            archive.unlink(); target = root / "target"; target.write_bytes(b"archive")
            archive.symlink_to(target)
            with self.assertRaises(M.TransportError):
                M.validate_archive_path(archive, 7, hashlib.sha256(b"archive").hexdigest())

    def test_05_archive_member_attacks(self) -> None:
        payloads = [b"one", b"two-two"]
        rows = [self.row(payloads[0], "a/one"), self.row(payloads[1], "b/two")]
        with tempfile.TemporaryDirectory(prefix="m1206_tar_") as temporary:
            root = Path(temporary); good = root / "good.tar"
            self.archive(good, rows, payloads)
            staged = M.validate_archive_to_stage(good, root / "good-stage", rows)
            self.assertEqual([M.exact_state(p, r) for p, r in zip(staged, rows)],
                             ["EXACT", "EXACT"])
            extra = root / "extra.tar"; extra_rows = rows + [self.row(b"x", "c/x")]
            self.archive(extra, extra_rows, payloads + [b"x"])
            traversal = root / "traversal.tar"
            self.archive(traversal, rows, payloads, names=["../escape", "b/two"])
            link = root / "link.tar"; self.archive(link, rows, payloads, symlink=True)
            for index, attack in enumerate((extra, traversal, link)):
                with self.assertRaises(M.TransportError):
                    M.validate_archive_to_stage(attack, root / ("bad-stage-" + str(index)), rows)

    def test_06_preexisting_exact_subset_is_idempotent(self) -> None:
        payloads = [b"one", b"two"]
        rows = [self.row(payloads[0], "a/one"), self.row(payloads[1], "b/two")]
        with tempfile.TemporaryDirectory(prefix="m1206_subset_") as temporary:
            root = Path(temporary); staged = [root / "s1", root / "s2"]
            destinations = [root / "a/one", root / "b/two"]
            for path, payload in zip(staged, payloads): self.write(path, payload)
            self.write(destinations[0], payloads[0]); destinations[1].parent.mkdir(parents=True)
            inode = destinations[0].stat().st_ino
            states = M.reconcile_exact_files(staged, destinations, rows, "Ab12Cd34Ef56")
            self.assertEqual(states, ["EXACT", "EXACT"])
            self.assertEqual(destinations[0].stat().st_ino, inode)
            self.assertEqual(M.reconcile_exact_files(staged, destinations, rows,
                                                     "Ab12Cd34Ef56"),
                             ["EXACT", "EXACT"])

    def test_07_wrong_symlink_and_publish_failure_stay_safe(self) -> None:
        payloads = [b"one", b"two"]
        rows = [self.row(payloads[0], "a/one"), self.row(payloads[1], "b/two")]
        for attack in ("wrong", "symlink"):
            with tempfile.TemporaryDirectory(prefix="m1206_target_") as temporary:
                root = Path(temporary); staged = [root / "s1", root / "s2"]
                destinations = [root / "a/one", root / "b/two"]
                for path, payload in zip(staged, payloads): self.write(path, payload)
                for path in destinations: path.parent.mkdir(parents=True, exist_ok=True)
                if attack == "wrong": destinations[0].write_bytes(b"bad")
                else: destinations[0].symlink_to(root / "missing")
                with self.assertRaises(M.TransportError):
                    M.reconcile_exact_files(staged, destinations, rows, "Ab12Cd34Ef56")
                self.assertFalse(destinations[1].exists())
        with tempfile.TemporaryDirectory(prefix="m1206_publish_") as temporary:
            root = Path(temporary); staged = [root / "s1", root / "s2"]
            destinations = [root / "a/one", root / "b/two"]
            for path, payload in zip(staged, payloads): self.write(path, payload)
            for path in destinations: path.parent.mkdir(parents=True)
            calls = 0
            def fail_second(source: Path, destination: Path) -> None:
                nonlocal calls; calls += 1
                if calls == 2: raise OSError("INJECT_PUBLISH_FAILURE")
                os.replace(source, destination)
            with self.assertRaises(OSError):
                M.reconcile_exact_files(staged, destinations, rows, "Ab12Cd34Ef56",
                                        publish=fail_second)
            self.assertEqual([M.exact_state(p, r) for p, r in zip(destinations, rows)],
                             ["EXACT", "ABSENT"])

    def test_08_capture_marker_race_fails_with_absent_or_exact_targets(self) -> None:
        payloads = [b"one", b"two"]
        rows = [self.row(payloads[0], "a/one"), self.row(payloads[1], "b/two")]
        with tempfile.TemporaryDirectory(prefix="m1206_race_") as temporary:
            root = Path(temporary); staged = [root / "s1", root / "s2"]
            destinations = [root / "a/one", root / "b/two"]; marker = root / "capture"
            for path, payload in zip(staged, payloads): self.write(path, payload)
            for path in destinations: path.parent.mkdir(parents=True)
            def control() -> None:
                if marker.exists(): raise M.TransportError("capture marker race")
            def race(index: int) -> None:
                if index == 0: marker.write_text("race")
            with self.assertRaises(M.TransportError):
                M.reconcile_exact_files(staged, destinations, rows, "Ab12Cd34Ef56",
                                        after_publish=race, control_absent=control)
            self.assertEqual([M.exact_state(p, r) for p, r in zip(destinations, rows)],
                             ["EXACT", "ABSENT"])

    def test_09_cleanup_failure_never_invalidates_exact_targets(self) -> None:
        payloads = [b"one", b"two"]
        rows = [self.row(payloads[0], "a/one"), self.row(payloads[1], "b/two")]
        with tempfile.TemporaryDirectory(prefix="m1206_cleanup_") as temporary:
            root = Path(temporary); staged = [root / "s1", root / "s2"]
            destinations = [root / "a/one", root / "b/two"]
            for path, payload in zip(staged, payloads): self.write(path, payload)
            for path in destinations: path.parent.mkdir(parents=True)
            def keep_temp(source: Path, destination: Path) -> None: os.link(source, destination)
            def fail_cleanup(path: Path) -> None: raise OSError("INJECT_CLEANUP_FAILURE")
            with self.assertRaises(M.TransportError):
                M.reconcile_exact_files(staged, destinations, rows, "Ab12Cd34Ef56",
                                        publish=keep_temp, cleanup=fail_cleanup)
            self.assertEqual([M.exact_state(p, r) for p, r in zip(destinations, rows)],
                             ["EXACT", "EXACT"])

    def test_10_scp_failure_runs_cleanup_and_never_writes_result(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m1206_run_scp_") as temporary:
            root = Path(temporary); attempt = root / "attempt"; result = root / "result"
            td = "/tmp/m1206_m1180.Ab12Cd34Ef56"
            pre = (json.dumps({"status": "PASS_M1206_SECURE_TEMP_PREFLIGHT",
                               "temp": td, "states": ["ABSENT", "ABSENT"],
                               "attempt_result_absent": True}, sort_keys=True) + "\n").encode()
            clean = (json.dumps({"status": "PASS_M1206_UNIQUE_TEMP_CLEANUP",
                                 "final": ["ABSENT", "ABSENT"],
                                 "attempt_result_absent": True}, sort_keys=True) + "\n").encode()
            calls = [self.completed(stdout=(td + "\n").encode()),
                     self.completed(stdout=pre), self.completed(stdout=clean)]
            with mock.patch.object(M, "LOCAL_ATTEMPT", attempt), \
                 mock.patch.object(M, "LOCAL_RESULT", result), \
                 mock.patch.object(M, "load_contract", return_value={}), \
                 mock.patch.object(M, "verify_policy"), \
                 mock.patch.object(M, "exact_members", return_value=[]), \
                 mock.patch.object(M, "verify_future_hammer"), \
                 mock.patch.object(M, "build_archive", return_value=(123, "0" * 64)), \
                 mock.patch.object(M, "run_ssh", side_effect=calls) as ssh_run, \
                 mock.patch.object(M.subprocess, "run", return_value=self.completed(code=1)):
                with self.assertRaises(M.TransportError): M.run()
            self.assertEqual(ssh_run.call_count, 3)
            self.assertTrue(attempt.exists()); self.assertFalse(result.exists())

    def test_11_cleanup_failure_blocks_success_after_exact_reconcile(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m1206_run_cleanup_") as temporary:
            root = Path(temporary); attempt = root / "attempt"; result = root / "result"
            td = "/tmp/m1206_m1180.Ab12Cd34Ef56"
            line = lambda value: (json.dumps(value, sort_keys=True) + "\n").encode()
            calls = [self.completed(stdout=(td + "\n").encode()),
                     self.completed(stdout=line({"status": "PASS_M1206_SECURE_TEMP_PREFLIGHT",
                         "temp": td, "states": ["ABSENT", "ABSENT"],
                         "attempt_result_absent": True})),
                     self.completed(stdout=line({"status": "PASS_M1206_REMOTE_MONOTONIC_EXACT2",
                         "initial": ["ABSENT", "ABSENT"], "final": ["EXACT", "EXACT"],
                         "attempt_result_absent": True})),
                     self.completed(code=1, stderr=b"INJECT_CLEANUP_FAILURE")]
            with mock.patch.object(M, "LOCAL_ATTEMPT", attempt), \
                 mock.patch.object(M, "LOCAL_RESULT", result), \
                 mock.patch.object(M, "load_contract", return_value={}), \
                 mock.patch.object(M, "verify_policy"), \
                 mock.patch.object(M, "exact_members", return_value=[]), \
                 mock.patch.object(M, "verify_future_hammer"), \
                 mock.patch.object(M, "build_archive", return_value=(123, "0" * 64)), \
                 mock.patch.object(M, "run_ssh", side_effect=calls), \
                 mock.patch.object(M.subprocess, "run", return_value=self.completed()):
                with self.assertRaises(M.TransportError): M.run()
            self.assertTrue(attempt.exists()); self.assertFalse(result.exists())

    def test_12_program_boundary_and_inert_namespaces(self) -> None:
        members = M.exact_members(M.load_contract())
        td = Path("/tmp/m1206_m1180.Ab12Cd34Ef56")
        source = SOURCE.read_text(encoding="utf-8")
        programs = (M.mktemp_program() + M.temp_preflight_program(td, members) +
                    M.reconciler_program(td, members, 123, "0" * 64) +
                    M.cleanup_program(td, members)).decode()
        for token in ("mktemp", "temp symlink/non-directory", "temp owner", "temp mode",
                      "archive symlink/nonregular", "archive owner/size/SHA",
                      "archive extra/path/order attack", "final both-exact gate",
                      "M1180 attempt/result must remain absent", "unique temp cleanup failed"):
            self.assertIn(token, source + programs)
        self.assertNotIn("/tmp/m1203_m1180_missing2_monotonic_transport_r1.tar", source)
        self.assertIn("shell=False", source); self.assertNotIn("shell=True", source)
        for forbidden in ("nvidia-smi", "dc_shell", "simv", "torch.cuda"):
            self.assertNotIn(forbidden, source)
        self.assertFalse(M.LOCAL_ATTEMPT.exists() or M.LOCAL_ATTEMPT.is_symlink())
        self.assertFalse(M.LOCAL_RESULT.exists() or M.LOCAL_RESULT.is_symlink())


if __name__ == "__main__":
    unittest.main(verbosity=2)
