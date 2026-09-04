#!/usr/bin/env python3
"""Directed source-only tests for M1074; never advance the full generator."""
from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HW = Path(__file__).resolve().parents[2]
ENGINE = HW / "system_simulator/scripts/execute_m1074_m1072_c1_full_exact_1rw_one_shot.py"
RUNNER = HW / "system_simulator/scripts/run_m1074_m1072_c1_full_exact_1rw_one_shot.sh"
SPEC = importlib.util.spec_from_file_location("m1074_test_engine", ENGINE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load M1074 engine")
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class M1074Tests(unittest.TestCase):
    def setUp(self):
        self.raw = M.synthetic_raw_result()

    def test_source_self_test_never_advances_full(self):
        value = M.source_self_test()
        self.assertEqual(value["status"],
                         "PASS_M1074_SOURCE_SELF_TEST__NO_FULL_REPLAY_NO_ATTEMPT")
        self.assertFalse(value["m1072_generator_advanced"])
        self.assertFalse(value["canonical_rows_opened_or_hashed"])

    def test_contract_and_m1073_are_frozen_no_launch(self):
        value = M.validate_source_contract(require_fresh=True)
        self.assertEqual(value["contract_sha256"], M.CONTRACT_SHA)
        contract = M.strict_json(M.CONTRACT)
        self.assertFalse(contract["launch_now"])
        self.assertEqual(contract["max_attempts_now"], 0)
        self.assertEqual(M.sha256(M.M1073 / "SHA256SUMS.seal.sha256"),
                         M.M1073_ID[2])

    def test_only_one_exact_zero_argument_cycle_call(self):
        source = inspect.getsource(M.execute_full)
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        target = [node for node in calls
                  if isinstance(node.func, ast.Attribute) and
                  node.func.attr == "iter_canonical_full_replay_results"]
        self.assertEqual(len(target), 1)
        self.assertEqual(target[0].args, [])
        self.assertEqual(target[0].keywords, [])
        self.assertNotIn("403922", source)

    def test_synthetic_full_normalization(self):
        value = M.normalize_full_result(self.raw)
        self.assertEqual(len(value["sample_boundaries"]), 10)
        self.assertEqual(value["sample_boundaries"][0], {
            "sample": 0, "first_task_id": 0,
            "last_task_id": M.M1072.TASKS_PER_SAMPLE - 1,
        })
        self.assertEqual(value["sample_boundaries"][-1]["last_task_id"],
                         M.M1072.TASKS - 1)
        self.assertEqual(value["aggregate"]["candidate"]["cycles"], 10045)
        self.assertEqual(value["aggregate"]["candidate"]["delayed_accesses"], 45)
        self.assertEqual(value["row_work_execution_provenance_digest_sha256"],
                         "a" * 64)

    def test_partial_duplicate_reordered_extra_samples_reject(self):
        variants = []
        partial = copy.deepcopy(self.raw); partial["samples"].pop(); variants.append(partial)
        duplicate = copy.deepcopy(self.raw); duplicate["samples"][1] = duplicate["samples"][0]; variants.append(duplicate)
        reordered = copy.deepcopy(self.raw); reordered["samples"][0], reordered["samples"][1] = reordered["samples"][1], reordered["samples"][0]; variants.append(reordered)
        extra = copy.deepcopy(self.raw); extra["samples"].append(copy.deepcopy(extra["samples"][-1])); variants.append(extra)
        for value in variants:
            with self.assertRaisesRegex(RuntimeError, "population|boundary"):
                M.normalize_full_result(value)

    def test_port_stall_missing_negative_bool_or_forged_reject(self):
        for replacement in (None, -1, True):
            value = copy.deepcopy(self.raw)
            row = value["samples"][0]["designs"]["candidate"]
            if replacement is None:
                del row["delayed_accesses"]
            else:
                row["delayed_accesses"] = replacement
            with self.assertRaisesRegex(RuntimeError, "cycle/stall"):
                M.normalize_full_result(value)

    def test_coverage_service_and_row_work_forgery_reject(self):
        mutations = []
        value = copy.deepcopy(self.raw); value["coverage"]["full_coverage_pass"] = False; mutations.append(value)
        value = copy.deepcopy(self.raw); value["coverage"]["service_digests"]["candidate"] = "0" * 64; mutations.append(value)
        value = copy.deepcopy(self.raw); value["coverage"]["execution_provenance_digest_sha256"] = "bad"; mutations.append(value)
        value = copy.deepcopy(self.raw); value["coverage"]["parent"]["candidate"]["reads"] += 1; mutations.append(value)
        for value in mutations:
            with self.assertRaises(RuntimeError):
                M.normalize_full_result(value)

    def test_capacity_is_internal_and_nonadmitted(self):
        value = M.normalize_full_result(self.raw)["capacity"]
        self.assertEqual(value["derived_total_bytes"], 214912)
        self.assertEqual(value["derived_margin_bytes"], 30848)
        self.assertFalse(value["capacity_only_214912B_admitted"])
        forged = copy.deepcopy(self.raw)
        forged["capacity"]["derived_total_bytes"] = 1
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            M.normalize_full_result(forged)

    def test_atomic_seal_detects_payload_change_and_collision(self):
        with tempfile.TemporaryDirectory(prefix="m1074_seal_") as temp:
            parent = Path(temp)
            root = parent / "payload"; root.mkdir()
            M.write_exclusive(root / "data", b"one\n")
            M.atomic_seal(root)
            (root / "data").write_bytes(b"two\n")
            with self.assertRaisesRegex(RuntimeError, "member"):
                M.verify_atomic_seal(root)
            source = parent / "source"; source.mkdir()
            destination = parent / "destination"; destination.mkdir()
            with self.assertRaisesRegex(RuntimeError, "no-replace"):
                M.rename_noreplace(source, destination)
            self.assertTrue(source.exists() and destination.exists())

    def test_atomic_seal_rejects_symlink(self):
        with tempfile.TemporaryDirectory(prefix="m1074_link_") as temp:
            root = Path(temp) / "payload"; root.mkdir()
            (root / "real").write_bytes(b"x")
            (root / "link").symlink_to(root / "real")
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                M.atomic_seal(root)

    def test_attempt_is_atomic_one_shot_before_rows(self):
        authority = {"m1075_outer_seal_file_sha256": "b" * 64}
        with tempfile.TemporaryDirectory(prefix="m1074_attempt_") as temp:
            parent = Path(temp)
            first = M.consume_attempt(authority, parent)
            final = parent / M.ATTEMPT.name
            self.assertTrue(final.is_dir())
            self.assertFalse(first["receipt"][
                "canonical_rows_opened_or_hashed_before_attempt"])
            with self.assertRaisesRegex(RuntimeError, "collision"):
                M.consume_attempt(authority, parent)
            interrupted = parent / "interrupted_attempt"
            interrupted.mkdir()
            closed = M.finalize_interrupted_attempt(interrupted)
            self.assertTrue((interrupted / "ATTEMPT_INTERRUPTED.json").is_file())
            self.assertEqual(closed, M.verify_atomic_seal(interrupted))

    def test_failure_after_attempt_is_recursively_sealed(self):
        with tempfile.TemporaryDirectory(prefix="m1074_failure_") as temp:
            parent = Path(temp)
            work = parent / (M.WORK_PREFIX + "test"); work.mkdir()
            nested = work / "nested"; nested.mkdir()
            (nested / "partial").write_bytes(b"partial")
            interrupted_seal = parent / (work.name + ".m1074_seal_stage.injected")
            interrupted_seal.mkdir()
            (interrupted_seal / "SHA256SUMS").write_bytes(b"partial seal\n")
            quarantine = parent / (M.FAILURE_PREFIX + "test")
            value = M.quarantine_work(work, quarantine, 130, "TEST", parent)
            self.assertEqual(value["status"],
                             "PASS_M1074_SEALED_FAILURE_QUARANTINE")
            self.assertFalse(work.exists())
            self.assertTrue((quarantine / "partial_result/nested/partial").is_file())
            self.assertTrue((quarantine /
                "partial_result_seal_stages/attempt_000/SHA256SUMS").is_file())
            M.verify_atomic_seal(quarantine)

    def test_runner_gate_order_and_no_cycle_arguments(self):
        text = RUNNER.read_text(encoding="utf-8")
        order = [text.index(token) for token in (
            "--validate-source", "--validate-authority", "m1074_process_gate\n",
            "m1074_resource_gate\n", "--consume-attempt", "--execute-full",
            "--publish",
        )]
        self.assertEqual(order, sorted(order))
        self.assertIn('[[ "$#" -eq 0 ]]', text)
        self.assertNotIn("--work-cycles", text)
        self.assertNotIn("--capacity", text)
        self.assertNotIn("403922", text)

    def test_runner_without_m1075_authority_cannot_consume(self):
        before = (M.ATTEMPT.exists(), M.RESULT.exists())
        result = subprocess.run([str(RUNNER)], text=True,
                                capture_output=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pin exact runner SHA", result.stderr)
        self.assertEqual((M.ATTEMPT.exists(), M.RESULT.exists()), before)

    def test_claim_boundary_and_namespaces_remain_closed(self):
        contract = M.strict_json(M.CONTRACT)
        boundary = contract["claim_boundary"]
        for key in ("launch_now", "attempt_consumed",
                    "full_51840000_replay_executed", "raw_result_created",
                    "capacity_only_214912B_admitted",
                    "full_trace_port_feasibility", "matched_cycles_admitted",
                    "speedup_admitted", "rtl_cycles", "paper_ppa_ready",
                    "eda_gpu_remote_used"):
            self.assertFalse(boundary[key])
        self.assertFalse(M.ATTEMPT.exists())
        self.assertFalse(M.RESULT.exists())
        self.assertEqual(M.sha256(M.DOCS359), M.DOCS359_SHA)


if __name__ == "__main__":
    unittest.main()
