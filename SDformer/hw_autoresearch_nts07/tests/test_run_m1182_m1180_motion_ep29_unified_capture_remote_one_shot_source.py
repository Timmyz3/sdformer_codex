#!/usr/bin/env python3
"""Controlled tests for the inert M1182/M1180 remote release; no production."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "hw_autoresearch_nts07/scripts/run_m1182_m1180_motion_ep29_unified_capture_remote_one_shot_source.py"
CONTRACT = ROOT / "hw_autoresearch_nts07/contracts/m1182_m1180_motion_ep29_unified_capture_launch_release_r1_20260830.json"
CAPTURE = ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m1180_motion_checkpoint_parametric_unified_hardware_r2.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


SPEC = importlib.util.spec_from_file_location("m1182_release_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)
CSPEC = importlib.util.spec_from_file_location("m1182_capture_sealer", CAPTURE)
assert CSPEC is not None and CSPEC.loader is not None
C = importlib.util.module_from_spec(CSPEC)
sys.modules[CSPEC.name] = C
CSPEC.loader.exec_module(C)


class M1182ReleaseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_01_contract_and_runtime_are_exact_and_inert(self) -> None:
        self.assertEqual(self.contract["schema"], C.LAUNCH_SCHEMA)
        self.assertEqual(self.contract["status"], C.LAUNCH_STATUS)
        self.assertFalse(self.contract["release_hammer_gate"]["present_now"])
        self.assertFalse(self.contract["claim_boundary"]["remote_execution_authorized_now"])
        self.assertEqual(self.contract["inputs"]["remote_launcher"]["sha256"], sha(SOURCE))
        self.assertEqual(self.contract["inputs"]["launcher"]["sha256"], sha(CAPTURE))

    def test_02_exact_forty_cohort_and_binding(self) -> None:
        rows = self.contract["cohort"]["samples"]
        self.assertEqual(rows, self.contract["r1_compatible_binding"]["cohort"]["samples"])
        self.assertEqual(len(rows), 40)
        self.assertEqual([row["global_sample_id"] for row in rows], list(range(40)))
        self.assertEqual(len({row["path"] for row in rows}), 40)
        self.assertEqual(len({row["sha256"] for row in rows}), 40)
        self.assertEqual([row["cohort"] for row in rows[:10]], ["c1"] * 10)
        self.assertEqual([row["sequence"] for row in rows[10:20]], ["interlaken_01_a"] * 10)
        self.assertEqual([row["sequence"] for row in rows[20:30]], ["thun_01_b"] * 10)
        self.assertEqual([row["sequence"] for row in rows[30:]], ["zurich_city_12_a"] * 10)

    def test_03_canonical_lease_attempt_result_log_and_no_retry(self) -> None:
        self.assertEqual(self.contract["gpu_ownership"]["lease_path"], str(M.LEASE_REL))
        self.assertEqual(self.contract["one_shot"]["attempt_marker"], str(M.ATTEMPT_REL))
        self.assertEqual(self.contract["output"]["path"], str(M.RESULT_REL))
        self.assertEqual(self.contract["production_log"]["path"], str(M.LOG_REL))
        self.assertFalse(self.contract["one_shot"]["automatic_retry"])
        source = SOURCE.read_text(encoding="utf-8")
        tree = ast.parse(source)
        execute = next(node for node in tree.body
                       if isinstance(node, ast.FunctionDef) and node.name == "execute_once")
        body = ast.get_source_segment(source, execute) or ""
        self.assertEqual(body.count("runner(command, policy.repo)"), 1)
        self.assertNotIn("while ", body)

    def test_04_m1175_and_source_hammer_exact_bindings(self) -> None:
        source_hammer = self.contract["inputs"]["m1180_source_hammer"]
        self.assertEqual(source_hammer["review_sha256"], M.SOURCE_HAMMER_REVIEW_SHA)
        self.assertEqual(source_hammer["manifest_sha256"], M.SOURCE_HAMMER_MANIFEST_SHA)
        self.assertEqual(source_hammer["outer_file_sha256"], M.SOURCE_HAMMER_OUTER_FILE_SHA)
        m1175 = self.contract["inputs"]["m1175_final_checkpoint_result_hammer"]
        self.assertEqual(m1175["review_sha256"], C.BASE.M1175_REVIEW_SHA256)
        self.assertEqual(m1175["manifest_sha256"], C.BASE.M1175_MANIFEST_SHA256)
        self.assertEqual(m1175["outer_file_sha256"], C.BASE.M1175_OUTER_FILE_SHA256)

    def test_05_release_hammer_missing_fail_then_exact_pass_and_mutation_fail(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            repo = Path(name)
            (repo / M.RELEASE_HAMMER_REL).mkdir(parents=True)
            (repo / M.CONTRACT_REL).parent.mkdir(parents=True, exist_ok=True)
            (repo / M.CONTRACT_REL).write_bytes(CONTRACT.read_bytes())
            dependency = {
                "label": "launch_contract", "path": M.CONTRACT_REL.as_posix(),
                "size_bytes": (repo / M.CONTRACT_REL).stat().st_size,
                "sha256": sha(repo / M.CONTRACT_REL), "disposition": "transfer_required",
            }
            inventory = {
                "schema": "m1182_m1180_motion_ep29_unified_capture_remote_dependency_inventory_r1_v1",
                "status": "COMPLETE_EXACT_REMOTE_PREFLIGHT_INVENTORY",
                "remote_repository": str(repo), "remote_interpreter": "/fake/python",
                "required_labels": ["launch_contract"], "dependencies": [dependency],
            }
            (repo / M.DEPENDENCY_INVENTORY_REL).write_text(
                json.dumps(inventory) + "\n", encoding="utf-8")
            transfer = sorted([M.CONTRACT_REL.as_posix(), M.DEPENDENCY_INVENTORY_REL.as_posix(),
                               M.TRANSFER_LIST_REL.as_posix()])
            (repo / M.TRANSFER_LIST_REL).write_text("\n".join(transfer) + "\n", encoding="utf-8")
            release_author = repo / M.RELEASE_AUTHOR_REL
            release_author.mkdir(parents=True)
            (release_author / "author_receipt.json").write_text("{}\n", encoding="utf-8")
            C.canonical_write_double_seal(release_author)
            policy = M.Policy(repo=repo, interpreter=Path("/fake/python"), python_version="3.10.20")
            review = {
                "schema": M.RELEASE_HAMMER_SCHEMA, "status": "PASS",
                "bindings": {
                    "launcher_sha256": sha(SOURCE),
                    "launch_contract_sha256": sha(CONTRACT),
                    "capture_source_sha256": M.SOURCE_SHA,
                    "capture_source_contract_sha256": M.SOURCE_CONTRACT_SHA,
                    "capture_source_test_sha256": M.SOURCE_TEST_SHA,
                    "capture_author_manifest_sha256": M.AUTHOR_MANIFEST_SHA,
                    "capture_author_outer_file_sha256": M.AUTHOR_OUTER_FILE_SHA,
                    "source_hammer_review_sha256": M.SOURCE_HAMMER_REVIEW_SHA,
                    "source_hammer_manifest_sha256": M.SOURCE_HAMMER_MANIFEST_SHA,
                    "source_hammer_outer_file_sha256": M.SOURCE_HAMMER_OUTER_FILE_SHA,
                    "dependency_inventory_sha256": sha(repo / M.DEPENDENCY_INVENTORY_REL),
                    "transfer_list_sha256": sha(repo / M.TRANSFER_LIST_REL),
                    "release_author_manifest_sha256": sha(release_author / "SHA256SUMS"),
                    "release_author_outer_file_sha256": sha(release_author / "SHA256SUMS.seal.sha256"),
                },
                "authorization": {"exact_remote_launch": True, "automatic_retry": False},
            }
            hammer = repo / M.RELEASE_HAMMER_REL
            (hammer / "review.json").write_text(json.dumps(review) + "\n", encoding="utf-8")
            C.canonical_write_double_seal(hammer)
            admitted = M.validate_release_hammer(C, policy)
            self.assertEqual(admitted["status"], "PASS")
            self.assertEqual(M.validate_dependency_inventory(policy, admitted)["status"],
                             "COMPLETE_EXACT_REMOTE_PREFLIGHT_INVENTORY")
            bad = copy.deepcopy(review); bad["status"] = "FAIL"
            (hammer / "review.json").write_text(json.dumps(bad) + "\n", encoding="utf-8")
            C.canonical_write_double_seal(hammer)
            with self.assertRaisesRegex(M.ReleaseError, "semantic admission"):
                M.validate_release_hammer(C, policy)

    def test_06_gpu_idle_parser_rejects_busy_and_malformed(self) -> None:
        def run(stdout: str, code: int = 0):
            return lambda *args, **kwargs: subprocess.CompletedProcess(args[0], code, stdout, "")
        self.assertEqual(M.gpu_compute_pids(run("")), [])
        self.assertEqual(M.gpu_compute_pids(run("123\n456\n")), [123, 456])
        with self.assertRaisesRegex(M.ReleaseError, "malformed"):
            M.gpu_compute_pids(run("not-a-pid\n"))
        with self.assertRaisesRegex(M.ReleaseError, "cannot prove"):
            M.gpu_compute_pids(run("", 1))

    def test_07_production_log_is_atomic_and_non_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "production.log"
            done = subprocess.CompletedProcess(["child"], 0, "PASS\n", "")
            M.write_production_log(path, ["child"], done)
            self.assertIn("automatic_retry=false", path.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                M.write_production_log(path, ["child"], done)

    def test_08_failure_calls_child_once_and_propagates_no_retry(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            policy = M.Policy(repo=Path(name), interpreter=Path("/fake/python"), python_version="3.10.20")
            calls = []
            def fake_preflight(*args):
                return ["child"], object()
            def fake_runner(command, cwd):
                calls.append((list(command), cwd))
                return subprocess.CompletedProcess(command, 7, "", "fail")
            with mock.patch.object(M, "preflight", fake_preflight):
                with self.assertRaisesRegex(M.ReleaseError, "no retry authorized"):
                    M.execute_once(policy, policy.interpreter, policy.python_version,
                                   policy.repo, fake_runner)
            self.assertEqual(len(calls), 1)

    def test_09_one_model_load_and_zero_argument_entry(self) -> None:
        r1 = C.BASE.R1_PATH
        self.assertEqual(r1.read_text(encoding="utf-8").count("profile.build_model("), 1)
        source = SOURCE.read_text(encoding="utf-8")
        self.assertIn("require(len(sys.argv) == 1", source)
        self.assertNotIn("shell=True", source)

    def test_10_docs359_and_claim_boundary(self) -> None:
        docs = ROOT / "hw_autoresearch_nts07/docs/359_DATE终局冻结_20260813.md"
        self.assertEqual(sha(docs), M.DOCS359_SHA)
        boundary = self.contract["claim_boundary"]
        self.assertTrue(boundary["release_source_only"])
        for key in ("capture_complete", "hardware_speedup", "system_speedup",
                    "energy", "ppa", "paper_citable_result"):
            self.assertFalse(boundary[key])


if __name__ == "__main__":
    unittest.main(verbosity=2)
