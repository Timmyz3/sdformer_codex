#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1195 exact-two transport source tests; no remote/transfer/GPU/capture/EDA."""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1195_m1182_m1180_missing2_transport_repair_source.py"
SPEC = importlib.util.spec_from_file_location("m1195_missing2", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1195Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = M.load_contract()

    def test_01_inventory_exact2_identity(self) -> None:
        rows = M.exact_members(self.contract)
        self.assertEqual(rows, [
            {"path": "hw_autoresearch_nts07/contracts/m699_h67_ep35_multisequence_decoder_payload_contract_r1_20260828.json",
             "size_bytes": 15961,
             "sha256": "43d3b024c1a78d8bc2422af3846c9a376a67bedbecb2ff7396a17bc51ec68fc7"},
            {"path": "hw_autoresearch_nts07/results/h67_ep35_dependency_dag_s1_20260822/dependency_events.jsonl",
             "size_bytes": 34816039,
             "sha256": "e1d2007195a036eedcee1e49d960955b3508ffe590ba3d075a3877a501a62f6b"},
        ])

    def test_02_transport_contract_and_tools(self) -> None:
        M.verify_transport_contract(self.contract)
        self.assertEqual(M.fixed_ssh_argv()[0], "/usr/bin/ssh")
        self.assertEqual(M.fixed_scp_argv(Path("/fixed/exact2.tar"))[0], "/usr/bin/scp")

    def test_03_archive_exact_order_type_size_sha(self) -> None:
        rows = M.exact_members(self.contract)
        with tempfile.TemporaryDirectory(prefix="m1195_test_") as temporary:
            path = Path(temporary) / "exact2.tar"
            digest = M.build_archive(path, rows)
            self.assertEqual(len(digest), 64)
            with tarfile.open(path, "r:") as archive:
                members = archive.getmembers()
                self.assertEqual([member.name for member in members],
                                 [row["path"] for row in rows])
                self.assertTrue(all(member.isfile() and not member.issym() and
                                    not member.islnk() for member in members))
                self.assertEqual([member.size for member in members],
                                 [row["size_bytes"] for row in rows])

    def test_04_safe_paths_and_duplicate_json(self) -> None:
        for text in ("/absolute", "../escape", "a/../escape"):
            with self.assertRaisesRegex(M.RepairError, "unsafe"):
                M.repo_relative(text)
        with self.assertRaisesRegex(M.RepairError, "duplicate JSON key"):
            M.strict_json_bytes(b'{"a":1,"a":2}')

    def test_05_remote_program_is_exact2_and_fail_closed(self) -> None:
        rows = M.exact_members(self.contract)
        preflight = M.preflight_program(rows).decode()
        remote = M.remote_program(rows, "0" * 64).decode()
        for token in (str(M.M1180_ATTEMPT_REL), str(M.M1180_RESULT_REL)):
            self.assertIn(token, preflight)
            self.assertIn(token, remote)
        for row in rows:
            self.assertIn(row["path"], preflight)
            self.assertIn(row["path"], remote)
        self.assertIn("len(rows)!=2", remote)
        self.assertIn("post-install identity", remote)
        self.assertIn("finally:", remote)

    def test_06_source_static_no_shell_no_gpu_no_eda(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("shell=False", source)
        self.assertNotIn("shell=True", source)
        self.assertNotIn("nvidia-smi", source)
        self.assertNotIn("dc_shell", source)
        self.assertNotIn("simv", source)
        self.assertNotIn("run_m1182_m1180_motion_ep29_unified_capture_remote_one_shot_source.py\")", source)

    def test_07_docs359_and_runtime_namespaces_untouched(self) -> None:
        self.assertEqual(M.sha256(ROOT / M.DOCS359_REL), M.DOCS359_SHA256)
        self.assertFalse(M.LOCAL_ATTEMPT.exists())
        self.assertFalse(M.LOCAL_RESULT.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
