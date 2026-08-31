#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1200 inert exact-two atomic transport tests; local sandboxes only."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1200_m1182_m1180_missing2_atomic_transport_repair_source.py"
SPEC = importlib.util.spec_from_file_location("m1200_atomic_transport", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1200Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = M.load_contract()

    def test_01_contract_inventory_stop_and_exact2(self) -> None:
        M.verify_policy(self.contract)
        rows = M.exact_members(self.contract)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row["size_bytes"] for row in rows], [15961, 34816039])
        self.assertEqual(M.sha256(ROOT / M.STOP_REL / "review.json"),
                         self.contract["m1197_stop_authority"]["review_sha256"])

    def test_02_all_exact2_contract_mutations_rejected(self) -> None:
        rows = self.contract["missing2"]
        for index in range(2):
            for key in sorted(rows[index]):
                changed = copy.deepcopy(rows)
                value = changed[index][key]
                changed[index][key] = (value + "_ATTACK") if isinstance(value, str) else value + 1
                with self.assertRaisesRegex(M.RepairError, "identity/order"):
                    M.validate_expected_rows(changed)
        for changed in (rows[:1], rows + [copy.deepcopy(rows[0])], list(reversed(rows))):
            with self.assertRaises(M.RepairError):
                M.validate_expected_rows(changed)

    def test_03_archive_exact_order_regular_size_sha(self) -> None:
        rows = M.exact_members(self.contract)
        with tempfile.TemporaryDirectory(prefix="m1200_archive_") as temporary:
            path = Path(temporary) / "exact2.tar"
            digest = M.build_archive(path, rows)
            self.assertEqual(digest, M.sha256(path))
            with tarfile.open(path, "r:") as archive:
                members = archive.getmembers()
                self.assertEqual([item.name for item in members], [row["path"] for row in rows])
                self.assertTrue(all(item.isfile() and not item.issym() and
                                    not item.islnk() for item in members))
                self.assertEqual([item.size for item in members],
                                 [row["size_bytes"] for row in rows])

    def test_04_atomic_publish_success(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m1200_publish_") as temporary:
            root = Path(temporary)
            staged = [root / "s0", root / "s1"]
            destinations = [root / "d0", root / "d1"]
            for index, path in enumerate(staged):
                path.write_bytes(bytes([index + 1]))
            M.publish_exact2_atomic(staged, destinations,
                                     lambda: self.assertTrue(all(path.exists() for path in destinations)))
            self.assertEqual([path.read_bytes() for path in destinations], [b"\x01", b"\x02"])

    def test_05_second_publication_failure_rolls_back_both(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m1200_second_fail_") as temporary:
            root = Path(temporary)
            staged = [root / "s0", root / "s1"]
            destinations = [root / "d0", root / "d1"]
            for path in staged:
                path.write_bytes(b"sealed")
            calls = 0
            def fail_second(source: Path, destination: Path) -> None:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("INJECT_SECOND_PUBLICATION_FAILURE")
                os.link(source, destination)
            with self.assertRaisesRegex(OSError, "INJECT_SECOND"):
                M.publish_exact2_atomic(staged, destinations, lambda: None, fail_second)
            self.assertEqual(calls, 2)
            self.assertTrue(all(not path.exists() and not path.is_symlink()
                                for path in destinations))

    def test_06_postverify_failure_rolls_back_both(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m1200_verify_fail_") as temporary:
            root = Path(temporary)
            staged = [root / "s0", root / "s1"]
            destinations = [root / "d0", root / "d1"]
            for path in staged:
                path.write_bytes(b"sealed")
            def fail_verify() -> None:
                raise M.RepairError("INJECT_POST_SHA_OR_M1180_POSTCONDITION_FAILURE")
            with self.assertRaisesRegex(M.RepairError, "INJECT_POST"):
                M.publish_exact2_atomic(staged, destinations, fail_verify)
            self.assertTrue(all(not path.exists() and not path.is_symlink()
                                for path in destinations))

    def test_07_preexisting_destination_rejected_untouched(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m1200_preexisting_") as temporary:
            root = Path(temporary)
            staged = [root / "s0", root / "s1"]
            destinations = [root / "d0", root / "d1"]
            for path in staged:
                path.write_bytes(b"sealed")
            destinations[0].write_bytes(b"preexisting")
            with self.assertRaisesRegex(M.RepairError, "preexists"):
                M.publish_exact2_atomic(staged, destinations, lambda: None)
            self.assertEqual(destinations[0].read_bytes(), b"preexisting")
            self.assertFalse(destinations[1].exists())

    @staticmethod
    def make_tiny_archive(path: Path, rows: list[dict], *, extra: bool = False,
                          symlink_index: int | None = None) -> str:
        with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as archive:
            for index, row in enumerate(rows):
                info = tarfile.TarInfo(row["path"])
                data = bytes([65 + index]) * row["size_bytes"]
                if symlink_index == index:
                    info.type = tarfile.SYMTYPE
                    info.linkname = "escape"
                    info.size = 0
                    archive.addfile(info)
                else:
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
            if extra:
                info = tarfile.TarInfo("extra/member")
                info.size = 1
                archive.addfile(info, io.BytesIO(b"x"))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def run_tiny_remote(self, mutate: str) -> tuple[subprocess.CompletedProcess[bytes], list[Path]]:
        temporary = tempfile.TemporaryDirectory(prefix="m1200_remote_attack_")
        self.addCleanup(temporary.cleanup)
        base = Path(temporary.name)
        root = base / "repo"
        root.mkdir()
        rows = [
            {"path": "a/x", "size_bytes": 3,
             "sha256": hashlib.sha256(b"AAA").hexdigest()},
            {"path": "b/y", "size_bytes": 4,
             "sha256": hashlib.sha256(b"BBBB").hexdigest()},
        ]
        for row in rows:
            (root / Path(row["path"]).parent).mkdir(parents=True, exist_ok=True)
        attempt = root / M.M1180_ATTEMPT_REL
        result = root / M.M1180_RESULT_REL
        archive = base / "payload.tar"
        stage = root / ".stage"
        extra = mutate == "extra"
        symlink = 1 if mutate == "symlink" else None
        if mutate == "traversal":
            rows[1]["path"] = "../escape"
        archive_sha = self.make_tiny_archive(archive, rows, extra=extra,
                                               symlink_index=symlink)
        if mutate == "sha":
            rows[1]["sha256"] = "0" * 64
        if mutate == "destination":
            (root / rows[0]["path"]).write_bytes(b"preexisting")
        if mutate == "attempt":
            attempt.parent.mkdir(parents=True, exist_ok=True)
            attempt.write_text("race")
        program = M.remote_program(rows, archive_sha, root=root, archive=archive,
                                   stage=stage, interpreter=sys.executable,
                                   python_version=sys.version.split()[0])
        completed = subprocess.run([sys.executable, "-I", "-"], input=program,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   shell=False, check=False)
        destinations = [root / row["path"] for row in rows if ".." not in Path(row["path"]).parts]
        return completed, destinations

    def test_08_m1197_remote_attacks_fail_closed(self) -> None:
        for attack in ("extra", "traversal", "sha", "symlink", "destination", "attempt"):
            completed, destinations = self.run_tiny_remote(attack)
            self.assertNotEqual(completed.returncode, 0, attack)
            if attack not in {"destination"}:
                self.assertTrue(all(not path.exists() and not path.is_symlink()
                                    for path in destinations), attack)

    def test_09_fixed_argv_and_static_fail_closed_boundaries(self) -> None:
        self.assertEqual(M.fixed_ssh_argv()[0], "/usr/bin/ssh")
        self.assertEqual(M.fixed_scp_argv(Path("/fixed/exact2.tar"))[0], "/usr/bin/scp")
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("shell=False", source)
        self.assertNotIn("shell=True", source)
        for token in ("extra/path/order member", "symlink/type/size",
                      "remote member SHA mismatch", "preexisting destination",
                      "M1180 attempt postcondition", "M1200_REMOTE_ROLLBACK_FAILED"):
            self.assertIn(token, source)
        self.assertNotIn("nvidia-smi", source)
        self.assertNotIn("dc_shell", source)
        self.assertNotIn("simv", source)

    def test_10_docs359_and_runtime_namespaces_untouched(self) -> None:
        self.assertEqual(M.sha256(ROOT / M.DOCS359_REL), M.DOCS359_SHA256)
        self.assertFalse(M.LOCAL_ATTEMPT.exists())
        self.assertFalse(M.LOCAL_RESULT.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
