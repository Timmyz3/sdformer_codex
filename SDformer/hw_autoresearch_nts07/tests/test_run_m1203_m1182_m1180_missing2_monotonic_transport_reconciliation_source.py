#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1203 local-only source tests.  No SSH, transfer, GPU, capture, or EDA."""
from __future__ import annotations

import importlib.util
import io
import os
from pathlib import Path
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1203_m1182_m1180_missing2_monotonic_transport_reconciliation_source.py"
SPEC = importlib.util.spec_from_file_location("m1203_monotonic", SOURCE)
assert SPEC and SPEC.loader
M = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(M)


class M1203Tests(unittest.TestCase):
    def row(self, payload: bytes, path: str = "a/one") -> dict:
        import hashlib
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

    def test_01_contract_and_exact_authorities(self) -> None:
        contract = M.load_contract(); M.verify_policy(contract)
        self.assertEqual(len(M.exact_members(contract)), 2)
        self.assertEqual(M.sha256(ROOT / M.DOCS359_REL), M.DOCS359_SHA256)

    def test_02_state_absent_exact_wrong_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m1203_state_") as temporary:
            root = Path(temporary); path = root / "target"; row = self.row(b"exact")
            self.assertEqual(M.exact_state(path, row), "ABSENT")
            path.write_bytes(b"exact"); self.assertEqual(M.exact_state(path, row), "EXACT")
            path.write_bytes(b"wrong")
            with self.assertRaises(M.ReconcileError): M.exact_state(path, row)
            path.unlink(); path.symlink_to(root / "missing")
            with self.assertRaises(M.ReconcileError): M.exact_state(path, row)

    def test_03_archive_exact_and_attacks(self) -> None:
        payloads = [b"one", b"two-two"]
        rows = [self.row(payloads[0], "a/one"), self.row(payloads[1], "b/two")]
        with tempfile.TemporaryDirectory(prefix="m1203_archive_") as temporary:
            root = Path(temporary); good = root / "good.tar"
            self.archive(good, rows, payloads)
            staged = M.validate_archive_to_stage(good, root / "stage-good", rows)
            self.assertEqual([M.exact_state(p, r) for p, r in zip(staged, rows)],
                             ["EXACT", "EXACT"])
            extra = root / "extra.tar"
            extra_rows = rows + [self.row(b"x", "c/extra")]
            self.archive(extra, extra_rows, payloads + [b"x"])
            with self.assertRaises(M.ReconcileError):
                M.validate_archive_to_stage(extra, root / "stage-extra", rows)
            traversal = root / "traversal.tar"
            self.archive(traversal, rows, payloads, names=["../escape", "b/two"])
            with self.assertRaises(M.ReconcileError):
                M.validate_archive_to_stage(traversal, root / "stage-traversal", rows)
            link = root / "link.tar"; self.archive(link, rows, payloads, symlink=True)
            with self.assertRaises(M.ReconcileError):
                M.validate_archive_to_stage(link, root / "stage-link", rows)

    def test_04_preexisting_exact_subset_is_idempotent(self) -> None:
        payloads = [b"one", b"two"]
        rows = [self.row(payloads[0], "a/one"), self.row(payloads[1], "b/two")]
        with tempfile.TemporaryDirectory(prefix="m1203_subset_") as temporary:
            root = Path(temporary); staged = [root / "s1", root / "s2"]
            destinations = [root / "a/one", root / "b/two"]
            for p, data in zip(staged, payloads): self.write(p, data)
            self.write(destinations[0], payloads[0]); destinations[1].parent.mkdir(parents=True)
            inode = destinations[0].stat().st_ino
            states = M.reconcile_exact_files(staged, destinations, rows)
            self.assertEqual(states, ["EXACT", "EXACT"])
            self.assertEqual(destinations[0].stat().st_ino, inode)
            self.assertEqual(M.reconcile_exact_files(staged, destinations, rows),
                             ["EXACT", "EXACT"])

    def test_05_wrong_and_symlink_preexisting_rejected_before_publish(self) -> None:
        payloads = [b"one", b"two"]
        rows = [self.row(payloads[0], "a/one"), self.row(payloads[1], "b/two")]
        for attack in ("wrong", "symlink"):
            with tempfile.TemporaryDirectory(prefix="m1203_preexisting_") as temporary:
                root = Path(temporary); staged = [root / "s1", root / "s2"]
                destinations = [root / "a/one", root / "b/two"]
                for p, data in zip(staged, payloads): self.write(p, data)
                destinations[0].parent.mkdir(parents=True); destinations[1].parent.mkdir(parents=True)
                if attack == "wrong": destinations[0].write_bytes(b"bad")
                else: destinations[0].symlink_to(root / "missing")
                with self.assertRaises(M.ReconcileError):
                    M.reconcile_exact_files(staged, destinations, rows)
                self.assertFalse(destinations[1].exists())

    def test_06_publish_failure_leaves_only_absent_or_exact(self) -> None:
        payloads = [b"one", b"two"]
        rows = [self.row(payloads[0], "a/one"), self.row(payloads[1], "b/two")]
        with tempfile.TemporaryDirectory(prefix="m1203_publish_fail_") as temporary:
            root = Path(temporary); staged = [root / "s1", root / "s2"]
            destinations = [root / "a/one", root / "b/two"]
            for p, data in zip(staged, payloads): self.write(p, data)
            for p in destinations: p.parent.mkdir(parents=True)
            calls = 0
            def fail_second(source: Path, destination: Path) -> None:
                nonlocal calls; calls += 1
                if calls == 2: raise OSError("INJECT_PUBLISH_FAILURE")
                os.replace(source, destination)
            with self.assertRaises(OSError):
                M.reconcile_exact_files(staged, destinations, rows, publish=fail_second)
            self.assertEqual([M.exact_state(p, r) for p, r in zip(destinations, rows)],
                             ["EXACT", "ABSENT"])

    def test_07_async_window_partial_is_safe_and_recoverable(self) -> None:
        payloads = [b"one", b"two"]
        rows = [self.row(payloads[0], "a/one"), self.row(payloads[1], "b/two")]
        with tempfile.TemporaryDirectory(prefix="m1203_async_") as temporary:
            root = Path(temporary); staged = [root / "s1", root / "s2"]
            destinations = [root / "a/one", root / "b/two"]
            for p, data in zip(staged, payloads): self.write(p, data)
            for p in destinations: p.parent.mkdir(parents=True)
            def interrupt(index: int) -> None:
                if index == 0: raise RuntimeError("INJECT_ASYNC_WINDOW_EQUIVALENT")
            with self.assertRaises(RuntimeError):
                M.reconcile_exact_files(staged, destinations, rows, after_publish=interrupt)
            self.assertEqual([M.exact_state(p, r) for p, r in zip(destinations, rows)],
                             ["EXACT", "ABSENT"])
            self.assertEqual(M.reconcile_exact_files(staged, destinations, rows),
                             ["EXACT", "EXACT"])

    def test_08_cleanup_failure_does_not_invalidate_exact_targets(self) -> None:
        payloads = [b"one", b"two"]
        rows = [self.row(payloads[0], "a/one"), self.row(payloads[1], "b/two")]
        with tempfile.TemporaryDirectory(prefix="m1203_cleanup_") as temporary:
            root = Path(temporary); staged = [root / "s1", root / "s2"]
            destinations = [root / "a/one", root / "b/two"]
            for p, data in zip(staged, payloads): self.write(p, data)
            for p in destinations: p.parent.mkdir(parents=True)
            def keep_temp(source: Path, destination: Path) -> None: os.link(source, destination)
            def fail_cleanup(path: Path) -> None: raise OSError("INJECT_CLEANUP_FAILURE")
            with self.assertRaises(M.ReconcileError):
                M.reconcile_exact_files(staged, destinations, rows,
                                        publish=keep_temp, cleanup=fail_cleanup)
            self.assertEqual([M.exact_state(p, r) for p, r in zip(destinations, rows)],
                             ["EXACT", "EXACT"])

    def test_09_program_boundary_and_final_gate(self) -> None:
        members = M.exact_members(M.load_contract())
        source = SOURCE.read_text(encoding="utf-8")
        program = M.reconciler_program(members, "0" * 64).decode()
        for token in ("final both-exact gate", "target symlink/nonregular",
                      "target wrong size/SHA", "archive extra/path/order attack",
                      "publish temporary preexists", "cleanup failed after reconciliation"):
            self.assertIn(token, source + program)
        for forbidden in ("nvidia-smi", "dc_shell", "simv", "torch.cuda"):
            self.assertNotIn(forbidden, source)
        self.assertIn("shell=False", source)
        self.assertNotIn("shell=True", source)

    def test_10_runtime_namespaces_are_inert(self) -> None:
        self.assertFalse(M.LOCAL_ATTEMPT.exists() or M.LOCAL_ATTEMPT.is_symlink())
        self.assertFalse(M.LOCAL_RESULT.exists() or M.LOCAL_RESULT.is_symlink())


if __name__ == "__main__":
    unittest.main(verbosity=2)
