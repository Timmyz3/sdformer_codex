#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Controlled M1217 source tests; no remote, network, GPU, capture, or EDA."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1217_m1215_m1208_final_monotonic_transport_successor_source.py"
SPEC = importlib.util.spec_from_file_location("m1217_source_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = M; SPEC.loader.exec_module(M)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Tests(unittest.TestCase):
    def test_01_import_inert_and_new_namespace(self) -> None:
        self.assertFalse(M.LOCAL_ATTEMPT.exists())
        self.assertNotEqual(M.LOCAL_ATTEMPT, M.M1210_ATTEMPT)
        self.assertNotEqual(M.LOCAL_ATTEMPT, M.M1215_ATTEMPT)
        self.assertEqual(sha(ROOT / M.DOCS359_REL), M.DOCS359_SHA)

    def test_02_both_consumed_markers_are_exact(self) -> None:
        M.verify_marker(M.M1210_ATTEMPT, M.M1210_SHA, M.M1210_TOKEN, "m1210")
        M.verify_marker(M.M1215_ATTEMPT, M.M1215_SHA, M.M1215_TOKEN, "m1215")

    def test_03_launcher_rel_population_exact(self) -> None:
        found = M.launcher_rel_constants(ROOT / M.LAUNCHER_REL)
        self.assertEqual(set(found), M.EXACT_REL_NAMES | M.ABSENT_REL_NAMES | M.OPTIONAL_REL_NAMES)
        self.assertEqual(len(found), 13)

    def test_04_inventory_expands_to_complete_requirement_closure(self) -> None:
        _, inventory, rows, closure, authority = M.load_release()
        self.assertEqual(len(inventory["transfer_roots"]), 9)
        self.assertEqual(len(rows), 40)
        self.assertEqual(len(closure["required_exact"]), 41)
        self.assertEqual(len(closure["sealed_roots"]), 4)
        self.assertEqual(len(closure["runtime_absent"]), 3)
        self.assertEqual(len(authority), 143)

    def test_05_forensic_full_recursive_seal_is_in_package(self) -> None:
        inventory = M.strict_json(ROOT / M.INVENTORY_REL)
        rows = M.expand_transfer_roots(inventory)
        prefix = "hw_autoresearch_nts07/reviews/m1215_m1210_m1208_first_launch_failure_forensic_r1_20260830/"
        forensic = [row for row in rows if row["path"].startswith(prefix)]
        self.assertEqual(len(forensic), 9)
        self.assertEqual({Path(row["path"]).name for row in forensic},
                         {p.name for p in (ROOT / prefix).iterdir() if p.is_file()})

    def test_06_missing_forensic_root_is_rejected(self) -> None:
        inventory = M.strict_json(ROOT / M.INVENTORY_REL)
        changed = copy.deepcopy(inventory)
        changed["transfer_roots"] = [row for row in changed["transfer_roots"]
                                     if "first_launch_failure_forensic" not in row["path"]]
        with self.assertRaisesRegex(M.ReleaseError, "root population"):
            M.expand_transfer_roots(changed)

    def test_07_missing_launcher_rel_is_rejected(self) -> None:
        inventory = M.strict_json(ROOT / M.INVENTORY_REL)
        rows = M.expand_transfer_roots(inventory)
        old = M.strict_json(ROOT / M.OLD_INVENTORY_REL)["dependencies"]
        changed = copy.deepcopy(inventory)
        changed["launcher_required_rel"]["exact_prerequisite_constants"].pop("FORENSIC_REL")
        with self.assertRaisesRegex(M.ReleaseError, "every launcher REL"):
            M.validate_launcher_coverage(changed, rows, old)

    def test_08_archive_order_type_and_hash(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); old = M.ROOT
            try:
                M.ROOT = root
                for rel, data in (("a/x", b"x"), ("b/y", b"yy")):
                    path = root / rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
                rows = [M.row_for(rel) for rel in ("a/x", "b/y")]
                archive = root / "x.tar"; M.make_archive(archive, rows)
                with tarfile.open(archive, "r:") as stream:
                    self.assertEqual(stream.getnames(), ["a/x", "b/y"])
                    self.assertTrue(all(member.isfile() for member in stream.getmembers()))
            finally:
                M.ROOT = old

    def test_09_remote_helper_preflights_then_absent_only_links(self) -> None:
        helper = M.REMOTE_HELPER
        self.assertLess(helper.index("for row in plan['old_dependencies']"),
                        helper.index("with tarfile.open(archive"))
        self.assertLess(helper.index("for row in plan['members']:\n p=repo"),
                        helper.index("for row in plan['members']:\n rel=pathlib"))
        self.assertIn("os.link(tmp,dst)", helper)
        self.assertNotIn("os.replace(tmp,dst)", helper)
        self.assertIn("for row in plan['post_publish_authority']", helper)
        self.assertIn("if exact_count!=143", helper)
        self.assertIn("for sealed in plan['sealed_roots']", helper)

    def test_10_exactly_one_existing_launcher_and_no_retry_loop(self) -> None:
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == "execute_once")
        body = ast.get_source_segment(source, execute) or ""
        self.assertEqual(body.count("launched = runner("), 1)
        self.assertEqual(body.count("launch_code ="), 1)
        self.assertNotIn("while ", body)
        self.assertIn("M1215 remote launch failed; no retry authorized", body)
        self.assertEqual(sha(ROOT / M.LAUNCHER_REL),
                         "30936622b629439d6d6c112d17bfc16881ae45d293660f615ba99309a5a3d98c")

    def test_11_no_remote_gpu_capture_eda_during_tests(self) -> None:
        self.assertFalse(M.LOCAL_ATTEMPT.exists())
        self.assertEqual(M.command_run.__name__, "command_run")

    def test_12_remote_read_only_audit_is_exactly_bound(self) -> None:
        contract = M.strict_json(ROOT / M.SOURCE_CONTRACT_REL)
        frozen = contract["remote_dependency_read_only_audit"]
        audit = ROOT / M.DEPENDENCY_AUDIT_REL
        self.assertEqual(frozen["review_sha256"], sha(audit / "review.json"))
        self.assertEqual(frozen["manifest_sha256"], sha(audit / "SHA256SUMS"))
        self.assertEqual(frozen["outer_seal_file_sha256"],
                         sha(audit / "SHA256SUMS.seal.sha256"))
        self.assertEqual((frozen["authority_unique_files"], frozen["remote_exact_files"],
                          frozen["remote_missing_files"], frozen["remote_drift_files"]),
                         (143, 134, 9, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
