#!/usr/bin/env python3
"""Fault-injection tests closing every M1017 P0 in M1023 r2."""

from dataclasses import replace
import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "scripts/analyze_m1023_decoder_stratified_block_reset_windows_source_r2.py"
SPEC = importlib.util.spec_from_file_location("m1023_under_test", SOURCE)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1023R2P0RepairTest(unittest.TestCase):
    def test_p0_1_cycle_aliases_case_and_nested_paths_rejected(self):
        attacks = (
            {"total_cycles": 9}, {"TOTAL_CyClEs": 9},
            {"latency_ns": 3}, {"ElapsedTime": 3},
            {"metrics": {"runtime": 3}},
            {"diagnostic": [{"SpEeDuP": 2.0}]},
        )
        for mutation in attacks:
            row = {"block_id": "attack", "compute_count": 1}
            row.update(mutation)
            with self.subTest(mutation=mutation):
                with self.assertRaisesRegex(RuntimeError,
                                            "semantic field forbidden"):
                    M.deterministic_select([row], "COMPUTE_REGULAR", 1)

    def test_p0_1_unknown_noncycle_field_and_nested_value_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "unknown pre-cycle"):
            M.deterministic_select([{
                "block_id": "a", "compute_count": 1, "harmless": 2}],
                "COMPUTE_REGULAR", 1)
        with self.assertRaisesRegex(RuntimeError, "nested metadata"):
            M.deterministic_select([{
                "block_id": "a", "compute_count": 1,
                "destination": [1, 2]}], "COMPUTE_REGULAR", 1)

    def test_p0_1_legal_allowlist_selection_is_deterministic(self):
        rows = [{"block_id": "b{:02d}".format(i), "compute_count": 1,
                 "layer": "D2", "sample_id": 0, "timestep": 0,
                 "expanded_request_count": 10} for i in range(12)]
        lhs = M.deterministic_select(rows, "COMPUTE_REGULAR", 8)
        rhs = M.deterministic_select(list(reversed(rows)),
                                     "COMPUTE_REGULAR", 8)
        self.assertEqual([row["block_id"] for row in lhs],
                         [row["block_id"] for row in rhs])

    def _expect_reset_mutation_rejected(self, mutator):
        body = M.M890.synthetic_transactions(448)
        spec = M.WindowSpec("reset-attack", "D0", "COMMIT_TAIL", 1)
        original = M.block_reset_transactions

        def attack(body_arg, spec_arg, side_arg):
            rows, metadata = original(body_arg, spec_arg, side_arg)
            if side_arg == "baseline":
                rows = list(rows)
                rows[0] = mutator(rows[0])
            return rows, metadata

        M.block_reset_transactions = attack
        try:
            with self.assertRaisesRegex(RuntimeError,
                                        "reset semantic or cycle charge"):
                M.paired_replay(body, body, spec)
        finally:
            M.block_reset_transactions = original

    def test_p0_2_reset_kind_service_mutation_rejected(self):
        self._expect_reset_mutation_rejected(
            lambda tx: replace(tx, kind="external_read"))

    def test_p0_2_reset_address_and_byte_mutations_rejected(self):
        self._expect_reset_mutation_rejected(
            lambda tx: replace(tx, base_address=tx.base_address + 1))
        self._expect_reset_mutation_rejected(
            lambda tx: replace(tx, width_bytes=2))

    def test_p0_2_reset_issue_cycle_and_bank_mutations_rejected(self):
        self._expect_reset_mutation_rejected(
            lambda tx: replace(tx, earliest_issue_cycle=1))
        self._expect_reset_mutation_rejected(
            lambda tx: replace(tx, bank_pattern=(1,)))

    def test_p0_2_normal_pair_has_exact_reset_charge_sequence(self):
        body = M.M890.synthetic_transactions(448)
        spec = M.WindowSpec("normal", "D0", "COMMIT_TAIL", 1)
        result = M.paired_replay(body, body, spec)
        self.assertTrue(result["paired_reset_exact_equal"])
        self.assertEqual([row["role"] for row in
                          result["paired_reset_service_cycle_sequence"]],
                         ["boundary", "fill", "drain"])
        self.assertEqual(result["candidate_cycles"],
                         result["baseline_cycles"])

    def test_p0_3_above_10pct_suppresses_all_point_estimates(self):
        result = M.estimate_paired_totals([{
            "stratum": "COMPUTE_REGULAR", "population_blocks": 1000,
            "candidate_cycles": [1, 100, 1, 100, 1, 100, 1, 100],
            "baseline_cycles": [100, 1, 100, 1, 100, 1, 100, 1],
        }])
        self.assertGreater(result["ci95_relative_halfwidth_max"], 0.10)
        self.assertTrue(result["status"].startswith("NO_POINT_ESTIMATE"))
        self.assertIsNone(result["candidate_total_cycles_estimate"])
        self.assertIsNone(result["baseline_total_cycles_estimate"])
        self.assertIsNone(result["paired_speedup_estimate"])
        self.assertFalse(result["point_estimate_admitted"])

    def test_p0_3_five_to_ten_is_diagnostic_only(self):
        result = M.apply_ci_publication_gate({
            "candidate_total_cycles_estimate": 100.0,
            "candidate_ci95": [94.0, 106.0],
            "baseline_total_cycles_estimate": 120.0,
            "baseline_ci95": [114.0, 126.0],
            "paired_speedup_estimate": 1.2,
            "paired_speedup_ci95": [1.128, 1.272],
        })
        self.assertGreater(result["ci95_relative_halfwidth_max"], 0.05)
        self.assertLessEqual(result["ci95_relative_halfwidth_max"], 0.10)
        self.assertTrue(result["status"].startswith("DIAGNOSTIC_POINT_ONLY"))
        self.assertFalse(result["point_estimate_admitted"])
        self.assertEqual(result["adaptive_action"],
                         "ADAPT_SAMPLE_BY_VARIANCE_BELOW_CAP")

    def test_p0_3_at_most_five_is_precise(self):
        result = M.estimate_paired_totals([{
            "stratum": "COMPUTE_REGULAR", "population_blocks": 8,
            "candidate_cycles": [10] * 8,
            "baseline_cycles": [20] * 8,
        }])
        self.assertEqual(result["status"],
                         "PRECISE_POINT_ELIGIBLE_FOR_LATER_RELEASE")
        self.assertTrue(result["point_estimate_admitted"])
        self.assertFalse(result["paper_citable"])

    def test_source_self_test_is_synthetic_only(self):
        result = M.self_test()
        self.assertEqual(result["status"],
                         "PASS_M1023_R2_SMALL_SYNTHETIC_P0_REPAIR_SELFTEST")
        self.assertFalse(result["real_payload_opened"])
        self.assertFalse(result["real_window_execution"])
        self.assertFalse(result["launch_now"])


if __name__ == "__main__":
    unittest.main()
