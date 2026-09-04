#!/usr/bin/env python3
"""Bounded M1094r2 source tests; never open canonical rows or run replay."""
from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


HW = Path(__file__).resolve().parents[2]
ENGINE = HW / "system_simulator/scripts/execute_m1094_m1086_c1_zero_work_exact_1rw_full_replay_one_shot.py"
RUNNER = HW / "system_simulator/scripts/run_m1094_m1086_c1_zero_work_exact_1rw_full_replay_one_shot.sh"
SPEC = importlib.util.spec_from_file_location("m1094r2_test_engine", ENGINE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load M1094r2 atomic library")
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def authority() -> dict[str, str]:
    return {
        "status": "PASS_DIFFERENT_AUTHOR_HARDCODED_LAUNCH_AUTHORITY",
        "m1095_review_sha256": "1" * 64,
        "m1095_manifest_sha256": "2" * 64,
        "m1095_outer_seal_file_sha256": "3" * 64,
        "m1095_launch_wrapper_sha256": "4" * 64,
        "m1094_engine_sha256": M.sha256(ENGINE),
        "m1094_contract_sha256": M.CONTRACT_SHA,
        "m1086_source_sha256": M.M1086_SHA,
        "m1087r3_outer_seal_file_sha256": M.M1087R3_ID[2],
    }


class M1094r2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.preflight = M.synthetic_preflight()
        self.raw = M.synthetic_raw_result()

    def test_source_validation_never_calls_production_interfaces(self):
        old_preflight = M.M1086.canonical_work_domain_preflight
        old_iterator = M.M1086.iter_canonical_full_replay_results

        def forbidden_preflight():
            raise AssertionError("production preflight was called")

        def forbidden_iterator():
            raise AssertionError("production iterator was called")
            yield None

        try:
            M.M1086.canonical_work_domain_preflight = forbidden_preflight
            M.M1086.iter_canonical_full_replay_results = forbidden_iterator
            result = M.source_self_test()
        finally:
            M.M1086.canonical_work_domain_preflight = old_preflight
            M.M1086.iter_canonical_full_replay_results = old_iterator
        self.assertEqual(result["status"],
                         "PASS_M1094R2_SOURCE_SELF_TEST__NO_ATTEMPT_NO_PAYLOAD")
        self.assertFalse(result["production_preflight_called"])
        self.assertFalse(result["production_iterator_called"])

    def test_contract_repairs_caller_selected_trust_root(self):
        contract = M.strict_json(M.CONTRACT)
        self.assertEqual(contract["status"],
            "PASS_M1094R2_ATOMIC_LIBRARY_SOURCE_CONTRACT__NO_EXECUTABLE_LAUNCH")
        self.assertFalse(contract["launch_now"])
        self.assertEqual(contract["max_attempts_now"], 0)
        self.assertFalse(contract["claim_boundary"]["executable_launch_present"])
        self.assertTrue(contract["future_execution_topology"][
            "m1095_launch_wrapper_must_hardcode_exact_authority_paths_and_digests_in_source"])

    def test_cli_has_no_mutating_or_caller_authority_mode(self):
        source = ENGINE.read_text(encoding="utf-8")
        main = textwrap.dedent(inspect.getsource(M.main))
        for forbidden in ("validate-authority", "consume-attempt", "execute-full",
                          "--publish", "quarantine-work", "expected-m1095",
                          "EXPECTED_M1095"):
            self.assertNotIn(forbidden, main)
            self.assertNotIn(forbidden, source)
        self.assertIn('(\"self-test\", \"validate-source\", \"verify-published\")',
                      main)

    def test_execute_function_orders_exact_zero_argument_production_calls(self):
        source = textwrap.dedent(inspect.getsource(M.execute_full))
        tree = ast.parse(source)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        targets = [(node.func.attr, node.lineno, node) for node in calls
                   if isinstance(node.func, ast.Attribute) and node.func.attr in {
                       "canonical_work_domain_preflight",
                       "iter_canonical_full_replay_results"}]
        self.assertEqual([item[0] for item in sorted(targets, key=lambda x: x[1])],
                         ["canonical_work_domain_preflight",
                          "iter_canonical_full_replay_results"])
        for _, _, call in targets:
            self.assertEqual(call.args, [])
            self.assertEqual(call.keywords, [])
        self.assertLess(source.index("canonical_work_domain_preflight()"),
                        source.index("iter_canonical_full_replay_results()"))

    def test_preflight_exact_population_accepts_and_mutations_reject(self):
        self.assertEqual(M.validate_preflight(self.preflight)["values_checked"],
                         2_436_480)
        mutations = []
        value = copy.deepcopy(self.preflight); value["values_checked"] -= 1; mutations.append(value)
        value = copy.deepcopy(self.preflight); value["counts"]["candidate"]["zero"] += 1; mutations.append(value)
        value = copy.deepcopy(self.preflight); value["counts"]["candidate"]["zero"] = True; mutations.append(value)
        value = copy.deepcopy(self.preflight); value["task_design_work_digest_sha256"] = "A" * 64; mutations.append(value)
        value = copy.deepcopy(self.preflight); value["cycles_derived_or_exported"] = True; mutations.append(value)
        for value in mutations:
            with self.assertRaises(RuntimeError):
                M.validate_preflight(value)

    def test_raw_result_normalization_and_mutations(self):
        normalized = M.normalize_raw(self.raw)
        self.assertEqual(normalized["aggregate"]["candidate"]["cycles"], 10045)
        mutations = []
        value = copy.deepcopy(self.raw); value["samples"].pop(); mutations.append(value)
        value = copy.deepcopy(self.raw); value["samples"][1] = value["samples"][0]; mutations.append(value)
        value = copy.deepcopy(self.raw); value["samples"][0]["designs"]["candidate"]["cycles_after_commit"] = True; mutations.append(value)
        value = copy.deepcopy(self.raw); value["coverage"]["full_coverage_pass"] = False; mutations.append(value)
        value = copy.deepcopy(self.raw); value["coverage"]["service_digests"]["candidate"] = "A" * 64; mutations.append(value)
        value = copy.deepcopy(self.raw); value["capacity"]["derived_total_bytes"] = 1; mutations.append(value)
        value = copy.deepcopy(self.raw); value["claim_boundary"]["speedup_admitted"] = True; mutations.append(value)
        for value in mutations:
            with self.assertRaises(RuntimeError):
                M.normalize_raw(value)

    def test_atomic_seal_and_no_replace(self):
        with tempfile.TemporaryDirectory(prefix="m1094r2_seal_") as temp:
            parent = Path(temp)
            payload = parent / "payload"; payload.mkdir()
            M.write_exclusive(payload / "value.json", b"{}\n")
            M.atomic_seal(payload)
            self.assertGreater(M.verify_atomic_seal(payload)["members"], 0)
            (payload / "value.json").write_bytes(b"changed\n")
            with self.assertRaisesRegex(RuntimeError, "member"):
                M.verify_atomic_seal(payload)
            source = parent / "source"; source.mkdir()
            destination = parent / "destination"; destination.mkdir()
            with self.assertRaisesRegex(RuntimeError, "no-replace"):
                M.rename_noreplace(source, destination)

    def test_attempt_is_one_shot_and_requires_closed_authority_shape(self):
        with tempfile.TemporaryDirectory(prefix="m1094r2_attempt_") as temp:
            parent = Path(temp)
            first = M.consume_attempt(authority(), parent)
            self.assertEqual(first["receipt"]["maximum_attempts"], 1)
            self.assertFalse(first["receipt"][
                "canonical_payload_opened_or_hashed_before_attempt"])
            with self.assertRaisesRegex(RuntimeError, "collision"):
                M.consume_attempt(authority(), parent)
            forged = authority(); forged["metric"] = "1.75x"
            with tempfile.TemporaryDirectory(prefix="m1094r2_auth_") as other:
                with self.assertRaisesRegex(RuntimeError, "authority"):
                    M.consume_attempt(forged, Path(other))

    def test_failure_quarantine_is_recursive_and_no_replace(self):
        with tempfile.TemporaryDirectory(prefix="m1094r2_failure_") as temp:
            parent = Path(temp)
            work = parent / (M.WORK_PREFIX + "test"); work.mkdir()
            nested = work / "nested"; nested.mkdir()
            (nested / "partial").write_bytes(b"partial")
            stage = parent / (work.name + ".m1094_seal_stage.injected")
            stage.mkdir(); (stage / "partial").write_bytes(b"seal stage")
            quarantine = parent / (M.FAILURE_PREFIX + "test")
            result = M.quarantine_work(work, quarantine, 130, "TEST", parent)
            self.assertEqual(result["status"],
                             "PASS_M1094_SEALED_FAILURE_QUARANTINE")
            self.assertTrue((quarantine / "partial_result/nested/partial").is_file())
            self.assertTrue((quarantine /
                "partial_result_seal_stages/attempt_000/partial").is_file())
            M.verify_atomic_seal(quarantine)

    def test_non_launch_stub_fails_before_attempt(self):
        before = (M.ATTEMPT.exists(), M.RESULT.exists())
        result = subprocess.run([str(RUNNER)], text=True, capture_output=True,
                                check=False)
        self.assertEqual(result.returncode, 86, result.stderr)
        self.assertIn("DIFFERENT_AUTHOR_M1095_HARDCODED_WRAPPER_REQUIRED",
                      result.stderr)
        self.assertEqual((M.ATTEMPT.exists(), M.RESULT.exists()), before)

    def test_namespaces_claims_and_docs359_remain_closed(self):
        self.assertFalse(M.ATTEMPT.exists())
        self.assertFalse(M.RESULT.exists())
        self.assertEqual(M.sha256(M.DOCS359), M.DOCS359_SHA)
        contract = M.strict_json(M.CONTRACT)
        for key in ("executable_launch_present", "attempt_consumed",
                    "preflight_executed", "full_replay_executed",
                    "matched_cycles_admitted", "speedup_admitted", "rtl_cycles",
                    "paper_citable", "paper_ppa_ready"):
            self.assertFalse(contract["claim_boundary"][key])


if __name__ == "__main__":
    unittest.main()
