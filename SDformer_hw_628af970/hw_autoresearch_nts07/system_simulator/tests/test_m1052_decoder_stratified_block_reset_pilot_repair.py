#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Synthetic/source-only M1052 regression and M1049 attack replay."""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import shutil
import unittest


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
SOURCE = HERE.parent / "scripts/execute_m1052_decoder_stratified_block_reset_pilot_repair.py"
RUNNER = HERE.parent / "scripts/run_m1054_m1052_decoder_stratified_block_reset_pilot_one_shot.sh"
SPEC = importlib.util.spec_from_file_location("m1052_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1052RepairTest(unittest.TestCase):
    def envelope(self):
        return {
            "schema": M.ENVELOPE_SCHEMA,
            "status": "CYCLE_CI_AT_MOST_5_PERCENT_NO_DERIVED_VALUES",
            "state": "CANDIDATE_AT_MOST_5_PERCENT",
            "bounds": {"candidate_total_cycles_ci95": [1.0, 2.0],
                       "baseline_total_cycles_ci95": [1.0, 2.0]},
            "uncertainty": {"candidate_cycles_relative_halfwidth": 0.04,
                "baseline_cycles_relative_halfwidth": 0.04,
                "maximum_relative_halfwidth": 0.04, "t_critical": 2.365},
            "coverage": {"strata": [{"stratum": name,
                "population_blocks": 8, "sample_blocks": 8,
                "finite_population_fraction": 1.0} for name in M.NONCENSUS]},
            "identity": {"metric":
                "serial block-reset executable schedule raw cycles"},
            "admission": {"derived_values_emitted": False,
                          "paper_citable": False},
        }

    def test_m1048_synthetic_partition_and_selector_preserved(self):
        result = M.self_test()
        self.assertEqual(result["m1048_transactions"], 85)
        self.assertFalse(result["real_payload_members_opened"])
        self.assertFalse(result["real_window_execution"])

    def test_preattempt_does_not_call_full_payload_verifier(self):
        original = M.M785.verify_sealed_directory
        calls = []
        def tripwire(path):
            calls.append(str(path))
            raise RuntimeError("PAYLOAD_MEMBER_VERIFIER_FORBIDDEN_PREATTEMPT")
        M.M785.verify_sealed_directory = tripwire
        try:
            value = M.validate_pre_attempt_source(M.CONTRACT, RUNNER)
        finally:
            M.M785.verify_sealed_directory = original
        self.assertEqual(calls, [])
        self.assertFalse(value["payload_members_opened"])
        self.assertFalse(value["payload_members_statted"])
        self.assertFalse(value["payload_members_hashed"])

    def test_contract_d1_exact_and_extra_fields_rejected(self):
        canonical = M.strict_json(M.CONTRACT)
        self.assertTrue(M.validate_contract(canonical))
        attacks = []
        attack = copy.deepcopy(canonical)
        attack["d1"]["scheduler_allowed"] = True
        attacks.append(attack)
        attack = copy.deepcopy(canonical)
        attack["d1"]["extra"] = False
        attacks.append(attack)
        attack = copy.deepcopy(canonical)
        attack["candidate_mean_cycles"] = 1.0
        attacks.append(attack)
        attack = copy.deepcopy(canonical)
        attack["sampling"]["nested"] = {"point_speedup": 2.0}
        attacks.append(attack)
        for value in attacks:
            with self.assertRaises(RuntimeError):
                M.validate_contract(value)

    def test_all_forbidden_semantic_aliases_rejected_at_depth(self):
        aliases = ("candidate_mean_cycles", "baselineMean", "point_speedup",
                   "speedups", "normalizedCycles", "runtimeEstimate",
                   "throughput", "FPS", "averageLatency")
        for alias in aliases:
            attack = copy.deepcopy(self.envelope())
            attack["coverage"]["strata"][0]["nested"] = {
                "deeper": {alias: 1.0}}
            with self.subTest(alias=alias):
                with self.assertRaisesRegex(RuntimeError,
                                            "forbidden derived semantic key"):
                    M.validate_envelope(attack)

    def test_envelope_exact_schema_types_ranges_and_identities(self):
        canonical = self.envelope()
        self.assertTrue(M.validate_envelope(canonical))
        attacks = []
        for value in ([1.0], [2.0, 1.0], [False, 2.0],
                      [1.0, math.nan], [1.0, math.inf]):
            attack = copy.deepcopy(canonical)
            attack["bounds"]["candidate_total_cycles_ci95"] = value
            attacks.append(attack)
        for value in (False, [0.1], math.nan, math.inf, -0.1):
            attack = copy.deepcopy(canonical)
            attack["uncertainty"]["maximum_relative_halfwidth"] = value
            attacks.append(attack)
        attack = copy.deepcopy(canonical)
        attack["uncertainty"]["maximum_relative_halfwidth"] = 0.2
        attacks.append(attack)
        attack = copy.deepcopy(canonical)
        attack["admission"]["derived_values_emitted"] = True
        attacks.append(attack)
        attack = copy.deepcopy(canonical)
        attack["extra"] = 1
        attacks.append(attack)
        for value in attacks:
            with self.assertRaises((RuntimeError, TypeError)):
                M.validate_envelope(value)

    def test_m1049_assemble_injections_rejected_without_sealing(self):
        base = M.RESULTS / (
            "." + M.RESULT_NAME + ".work.m1052test")
        if base.exists():
            shutil.rmtree(base)
        base.mkdir(mode=0o700)
        try:
            payload = {"schema": M.PAYLOAD_SCHEMA,
                "status": "PASS_M1054_POSTATTEMPT_FULL_PAYLOAD_IDENTITY",
                "attempt_receipt_sha256": "0" * 64,
                "m699_manifest_sha256": "1" * 64,
                "m699_root_manifest_sha256": "2" * 64,
                "m699_outer_seal_file_sha256": "3" * 64,
                "selected_records": [{"layer": layer,
                    "sequence": M.M1048.SEQUENCE, "sample_id": 0,
                    "module_index": M.M1048.MODULE_BY_LAYER[layer],
                    "route": "EXACT_BINARY_BITPACK", "relative_path": "x",
                    "packed_sha256": "4" * 64} for layer in M.LAYERS],
                "payload_members_verified": True, "post_attempt": True,
                "d1_scheduled": False, "paper_citable": False}
            M.atomic_json(base / "payload_validation.json", payload)
            raw = {"schema": M.RAW_SCHEMA,
                "status": "PASS_M1054_RAW_CYCLES__RESULT_HAMMER_REQUIRED",
                "candidate_mean_cycles": 12.5}
            M.atomic_json(base / "raw_windows.json", raw)
            M.atomic_json(base / "result.json", {"schema": M.RESULT_SCHEMA,
                "point_speedup": 2.0})
            (base / "RUN_COMPLETE.txt").write_text("FORGED\n")
            with self.assertRaisesRegex(RuntimeError,
                                        "forbidden derived semantic key"):
                M.assemble(base)
            self.assertFalse((base / "SHA256SUMS").exists())
            self.assertFalse((base / "SHA256SUMS.seal.sha256").exists())
        finally:
            shutil.rmtree(base)

    def test_direct_runtime_requires_attempt_and_independent_authority(self):
        with self.assertRaises(RuntimeError):
            M.run_pilot(
                M.RESULTS / M.ATTEMPT_NAME,
                M.RESULTS / ("." + M.RESULT_NAME + ".work.direct"), {})

    def test_wrong_contract_pin_cannot_consume_attempt(self):
        attempt = M.RESULTS / M.ATTEMPT_NAME
        self.assertFalse(attempt.exists())
        with self.assertRaisesRegex(RuntimeError,
                                    "attempt source/contract identity drift"):
            M.consume_attempt(attempt, RUNNER, "0" * 64, {})
        self.assertFalse(attempt.exists())

    def test_wrong_runtime_namespaces_rejected(self):
        attacks = ((M.RESULTS / ".wrong-attempt", "attempt"),
                   (M.RESULTS / "wrong-result", "result"),
                   (M.RESULTS / ".wrong-work", "work"),
                   (M.RESULTS / "wrong-quarantine", "quarantine"),
                   (Path("/tmp/m1052-outside"), "work"))
        for path, role in attacks:
            with self.subTest(path=path, role=role):
                with self.assertRaises(RuntimeError):
                    M.safe_path(path, role)


if __name__ == "__main__":
    unittest.main()
