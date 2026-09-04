#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Controlled M1215 successor tests; no network, remote, GPU, capture, EDA."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1215_m1208_motion_ep29_unified_capture_successor_secure_remote_one_shot_source.py"
SUCCESSOR = ROOT / "hw_autoresearch_nts07/scripts/run_m1215_motion_ep29_unified_capture_remote_one_shot_successor_source.py"
SPEC = importlib.util.spec_from_file_location("m1215_release_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = M; SPEC.loader.exec_module(M)
SUCCESSOR_SPEC = importlib.util.spec_from_file_location("m1215_remote_successor_test", SUCCESSOR)
assert SUCCESSOR_SPEC is not None and SUCCESSOR_SPEC.loader is not None
S = importlib.util.module_from_spec(SUCCESSOR_SPEC)
sys.modules[SUCCESSOR_SPEC.name] = S
SUCCESSOR_SPEC.loader.exec_module(S)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Tests(unittest.TestCase):
    def test_01_import_is_inert_and_docs359_pinned(self) -> None:
        self.assertFalse(M.LOCAL_ATTEMPT.exists())
        self.assertEqual(sha(ROOT / M.DOCS359_REL), M.DOCS359_SHA)

    def test_02_namespace_is_disjoint_and_m1180_is_immutable(self) -> None:
        self.assertIn("m1208", M.M1208_ATTEMPT_REL.as_posix())
        self.assertIn("m1180", M.M1180_ATTEMPT_REL.as_posix())
        self.assertNotEqual(M.M1208_ATTEMPT_REL, M.M1180_ATTEMPT_REL)
        source = SOURCE.read_text(encoding="utf-8")
        self.assertNotIn("unlink(M1180", source)
        self.assertNotIn("rmtree(M1180", source)
        self.assertTrue(M.M1210_FAILED_ATTEMPT.is_file())
        self.assertEqual(sha(M.M1210_FAILED_ATTEMPT), M.M1210_FAILED_SHA)

    def test_03_remote_temp_path_is_anchored(self) -> None:
        self.assertIsNotNone(M.REMOTE_TEMP_RE.fullmatch("/tmp/m1215_m1208.A1b2C3d4E5f6"))
        for bad in ("/tmp/m1215_m1208.bad", "/tmp/x.A1b2C3d4E5f6", "/tmp/m1215_m1208.A1b2C3d4E5f6/x"):
            self.assertIsNone(M.REMOTE_TEMP_RE.fullmatch(bad))

    def test_04_exact_state_rejects_wrong_regular_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); target = root / "x"; target.write_bytes(b"ok")
            row = {"path": "x", "size_bytes": 2, "sha256": sha(target)}
            self.assertEqual(M.exact_state(target, row), "EXACT")
            target.write_bytes(b"bad")
            with self.assertRaisesRegex(M.ReleaseError, "drift"):
                M.exact_state(target, row)
            target.unlink(); (root / "real").write_bytes(b"ok"); target.symlink_to(root / "real")
            with self.assertRaisesRegex(M.ReleaseError, "symlink"):
                M.exact_state(target, row)

    def test_05_archive_is_exact_order_and_regular(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); old_root = M.ROOT
            try:
                M.ROOT = root
                for rel, data in (("a/x", b"x"), ("b/y", b"yy")):
                    p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
                rows = [{"path": rel, "size_bytes": (root / rel).stat().st_size,
                         "sha256": sha(root / rel)} for rel in ("a/x", "b/y")]
                archive = root / "out.tar"; M.make_archive(archive, rows)
                with tarfile.open(archive, "r:") as stream:
                    self.assertEqual(stream.getnames(), ["a/x", "b/y"])
                    self.assertTrue(all(member.isfile() for member in stream.getmembers()))
            finally:
                M.ROOT = old_root

    def test_06_one_child_no_retry_static_gate(self) -> None:
        source = SOURCE.read_text(encoding="utf-8"); tree = ast.parse(source)
        execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "execute_once")
        body = ast.get_source_segment(source, execute) or ""
        self.assertEqual(body.count("launch_code"), 2)
        self.assertEqual(body.count("launched = runner("), 1)
        self.assertNotIn("while ", body)
        self.assertIn("LOCAL_ATTEMPT", body)

    def test_07_remote_helper_checks_old_deps_and_namespaces_before_publish(self) -> None:
        helper = M.REMOTE_HELPER
        self.assertLess(helper.index("old_dependencies"), helper.index("stage=temp/'stage'"))
        self.assertLess(helper.index("m1208_attempt"), helper.index("stage=temp/'stage'"))
        self.assertIn("target_drift", helper)
        self.assertIn("os.replace(tmp,dst)", helper)

    def test_08_contract_inventory_and_fixed_members_close(self) -> None:
        source_contract, inventory, rows = M.load_release()
        self.assertEqual(len(rows), 1)
        self.assertEqual(inventory["fixed_transfer_member_count"], 1)
        self.assertFalse(source_contract["claim_boundary"]["remote"])

    def test_09_successor_admits_exact_actual_m1211(self) -> None:
        source = SUCCESSOR.read_text(encoding="utf-8")
        self.assertIn("PASS_M1210_SECURE_TRANSFER_AND_ONE_M1208_REMOTE_LAUNCH_AUTHORIZED", source)
        self.assertIn('review.get("verdict") == "GO"', source)
        self.assertIn('review.get("score") >= 95', source)
        self.assertIn('review.get("p0_count") == 0', source)
        self.assertIn('review.get("p1_count") == 0', source)
        self.assertNotIn('review.get("status") == "PASS"', source)

    def test_10_actual_m1211_review_passes_successor_admission(self) -> None:
        old_repo = S.REPO
        class Capture:
            @staticmethod
            def canonical_verify_double_seal(path: Path):
                return M.canonical_verify_double_seal(path)
        try:
            S.REPO = ROOT
            review = S.validate_release_hammer(
                Capture(), ROOT / S.LAUNCH_CONTRACT_REL)
            self.assertEqual(review["status"], S.RELEASE_HAMMER_STATUS)
            self.assertEqual(review["verdict"], "GO")
            self.assertGreaterEqual(review["score"], 95)
        finally:
            S.REPO = old_repo


if __name__ == "__main__":
    unittest.main(verbosity=2)
