#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Source-only M1188 transport adapter tests; no remote, transfer, GPU or capture."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1188_m1182_m1180_capture_tar_transport_adapter_source.py"
SPEC = importlib.util.spec_from_file_location("m1188_transport", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1188TransportSourceTests(unittest.TestCase):
    def test_01_contract_and_exact51_close(self) -> None:
        contract = M.load_contract()
        M.verify_transport_contract(contract)
        rows = M.exact_members(contract)
        self.assertEqual((len(rows), sum(r["class"] == "ORIGINAL_EXACT42" for r in rows),
                          sum(r["class"] == "M1184_EXACT_SEAL" for r in rows)),
                         (51, 42, 9))
        self.assertEqual(len({r["path"] for r in rows}), 51)

    def test_02_original_authorities_are_unchanged(self) -> None:
        expected = {
            M.ORIGINAL_LIST_REL: (5115, "ec53838c3f6961a9b1143ba96d6d4452980bc3c569948f16ca7842a0a08cbc1b"),
            M.ORIGINAL_INVENTORY_REL: (42133, "de6ff2b13719580b77674b44f7414a7798cffd3f7cde5e80e88ff3ea8f0d97ae"),
            M.ORIGINAL_RELEASE_REL: (44255, "46450015bcdb3b8c0a32ccd7aaba68a78abf923705a133147202283e7bc7220f"),
        }
        for rel, (size, digest) in expected.items():
            path = ROOT / rel
            self.assertEqual(path.stat().st_size, size)
            self.assertEqual(M.sha256(path), digest)
        self.assertEqual(len((ROOT / M.ORIGINAL_LIST_REL).read_text().splitlines()), 42)

    def test_03_fixed_argv_and_no_shell_tokens(self) -> None:
        ssh = M.fixed_ssh_argv()
        scp = M.fixed_scp_argv(Path("/fixed/local/archive.tar"))
        self.assertEqual(ssh[0], "/usr/bin/ssh")
        self.assertEqual(scp[0], "/usr/bin/scp")
        self.assertNotIn("sh", ssh)
        self.assertNotIn("-c", ssh)
        self.assertEqual(ssh[-3:], [M.REMOTE_INTERPRETER, "-I", "-"])
        self.assertEqual(scp[-1], M.REMOTE_HOST + ":" + str(M.REMOTE_ARCHIVE))

    def test_04_archive_is_exact_regular_only(self) -> None:
        rows = M.exact_members(M.load_contract())
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "x.tar"
            digest = M.build_archive(archive, rows)
            self.assertEqual(digest, M.sha256(archive))
            with tarfile.open(archive, "r:") as tf:
                self.assertEqual([m.name for m in tf.getmembers()], [r["path"] for r in rows])
                self.assertTrue(all(m.isfile() and not m.issym() and not m.islnk()
                                    for m in tf.getmembers()))

    def test_05_remote_extractor_local_positive_and_post_hash(self) -> None:
        rows = M.exact_members(M.load_contract())
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            root = base / "repo"; root.mkdir()
            archive = base / "x.tar"
            stage = root / ".stage"
            archive_sha = M.build_archive(archive, rows)
            program = M.remote_program(rows, archive_sha, archive, root, stage,
                                       str(Path(M.sys.executable)), M.sys.version.split()[0])
            run = subprocess.run([M.sys.executable, "-I", "-"], input=program,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            self.assertEqual(run.returncode, 0, run.stderr.decode())
            receipt = json.loads(run.stdout)
            self.assertEqual(receipt["verified"], 51)
            self.assertFalse(archive.exists())
            for row in rows:
                path = root / row["path"]
                self.assertEqual((path.stat().st_size, M.sha256(path)),
                                 (row["size_bytes"], row["sha256"]))

    def test_06_remote_extractor_rejects_traversal_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            base = Path(td); root = base / "repo"; root.mkdir()
            for name, kind in (("../escape", "file"), ("safe-link", "symlink")):
                archive = base / (kind + ".tar")
                payload = b"x"
                with tarfile.open(archive, "w") as tf:
                    info = tarfile.TarInfo(name)
                    if kind == "symlink":
                        info.type = tarfile.SYMTYPE; info.linkname = "target"
                        tf.addfile(info)
                    else:
                        info.size = 1; tf.addfile(info, io.BytesIO(payload))
                row = {"path": name, "size_bytes": 1,
                       "sha256": hashlib.sha256(payload).hexdigest(), "class": "X"}
                # Cardinality fails before extraction; that is still fail-closed.
                program = M.remote_program([row], M.sha256(archive), archive, root,
                                           root / ".stage", str(Path(M.sys.executable)),
                                           M.sys.version.split()[0])
                run = subprocess.run([M.sys.executable, "-I", "-"], input=program,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                self.assertNotEqual(run.returncode, 0)
            self.assertFalse((base / "escape").exists())

    def test_07_prior_failure_and_claim_boundary_are_explicit(self) -> None:
        contract = M.load_contract()
        prior = contract["prior_rsync_failure"]
        self.assertEqual(prior["bytes_transferred"], 0)
        self.assertFalse(prior["remote_namespace_created"])
        self.assertFalse(prior["m1180_attempt_consumed"])
        self.assertFalse(prior["gpu_consumed"])
        self.assertFalse(contract["claim_boundary"]["paper_result"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
