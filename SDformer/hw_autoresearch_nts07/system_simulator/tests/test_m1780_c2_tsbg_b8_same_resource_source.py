#!/usr/bin/env python3
"""Author static/reference tests for source-only M1780."""
from __future__ import print_function

import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1780_c2_tsbg_b8_same_resource_source.py"
SPEC = importlib.util.spec_from_file_location("m1780_checker", str(CHECKER))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class M1780SourceTest(unittest.TestCase):
    def test_01_source_boundary_and_identity(self):
        value = CHECK.validate_sources()
        self.assertEqual(value["status"],
                         "PASS_M1780_TSBG_B8_SAME_RESOURCE_SOURCE_ONLY_NO_EDA")
        self.assertFalse(any(value["claim_boundary"].values()))
        self.assertEqual(value["author_execution"], {
            "vcs_runs": 0, "simv_runs": 0, "dc_runs": 0,
            "ptpx_runs": 0, "license_queries": 0,
            "attempts": 0, "results": 0})

    def test_02_ordinary_lru8_fair_directed_ledger(self):
        active = [[True] * 12 for _ in range(8)]
        baseline = CHECK.lru_miss_ledger(active, "token")
        candidate = CHECK.lru_miss_ledger(active, "tsbg")
        self.assertEqual(baseline, {"accesses": 96, "hits": 0,
                                   "misses": 96, "weight_beats": 1152})
        self.assertEqual(candidate, {"accesses": 96, "hits": 84,
                                    "misses": 12, "weight_beats": 144})
        self.assertEqual(baseline["accesses"], candidate["accesses"])

    def test_03_lru_is_persistent_and_capacity_sensitive(self):
        active = [[True] * 12 for _ in range(8)]
        cap8 = CHECK.lru_miss_ledger(active, "token", 8)
        cap12 = CHECK.lru_miss_ledger(active, "token", 12)
        self.assertEqual(cap8["misses"], 96)
        self.assertEqual(cap12["misses"], 12)
        self.assertEqual(cap12["hits"], 84)

    def test_04_signed_reference_preserves_products_and_commits(self):
        value = CHECK.directed_accumulators()
        self.assertEqual(value["issues"], 1152)
        self.assertEqual(value["signed_products"], 18432)
        self.assertEqual(value["commits"], 48)
        flattened = [number for context in value["accumulators"]
                     for output_slice in context for number in output_slice]
        self.assertTrue(any(number < 0 for number in flattened))
        self.assertTrue(any(number > 0 for number in flattened))

    def test_05_per_context_sign_changes_result_without_weight_change(self):
        weights = [CHECK.directed_weight(0, 0, 0, bank, 0)
                   for bank in range(8)]
        codes0 = CHECK.directed_codes(0, 0)[:8]
        codes1 = CHECK.directed_codes(1, 0)[:8]
        sum0 = sum(value * weight for value, weight in zip(codes0, weights))
        sum1 = sum(value * weight for value, weight in zip(codes1, weights))
        self.assertNotEqual(sum0, sum1)
        self.assertEqual(weights,
                         [CHECK.directed_weight(0, 0, 0, bank, 0)
                          for bank in range(8)])

    def test_06_resource_price_exceeds_m1763_lower_bound(self):
        value = CHECK.resource_account()
        self.assertEqual(value["ordinary_lru_rows"], 8)
        self.assertEqual(value["banks"], 8)
        self.assertEqual(value["shared_row_cache_bytes"], 12288)
        self.assertEqual(value["acc24_context_bytes"], 2304)
        self.assertEqual(value["source_fifo_bytes"], 6144)
        self.assertEqual(value["m1763_incremental_state_lower_bound_bytes"],
                         2128)
        self.assertGreater(value["explicit_state_bytes_excluding_control"],
                           value["m1763_incremental_state_lower_bound_bytes"])

    def test_07_rtl_has_no_product_reuse_or_lossy_drop(self):
        active = CHECK.strip_sv_comments(CHECK.RTL.read_text())
        for forbidden in ("issue_source_value[bank] *", "reuse_product",
                          "approx", "epsilon", "drop_source"):
            self.assertNotIn(forbidden, active)
        self.assertIn("delta = delta +", active)
        self.assertIn("delta = delta -", active)

    def test_08_protocol_and_backpressure_coverage_present(self):
        sva = CHECK.SVA.read_text()
        for token in ("ap_mem_req_stable", "ap_issue_payload_stable",
                      "ap_commit_payload_stable", "ap_fault_closes_load",
                      "cp_negative_source", "cp_positive_source",
                      "cp_issue_stall", "cp_memory_stall",
                      "cp_commit_stall", "cp_protocol_attack"):
            self.assertIn(token, sva)
        tb = CHECK.TB.read_text()
        self.assertIn("load_source_value[0] = 8'sd2", tb)
        self.assertIn("directed same-resource local gate below 1.15x", tb)

    def test_09_predecessor_stays_screening_only(self):
        decision = CHECK.strict_json(CHECK.M1763 / "decision.json")
        rows = [row for row in decision["tsbg"]["rows"]
                if row["bundle"] == 8 and row["scope_type"] == "all"]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertFalse(row["same_resource_claim"])
        self.assertFalse(row["full_area_energy_pricing_complete"])
        self.assertFalse(row["fetch_ratio_is_cycle_speedup"])
        self.assertFalse(row["aggregate_cycle_gate_ge_1p15"])
        self.assertTrue(row["diagnostic_aggregate_cycle_gate_ge_1p15"])


if __name__ == "__main__":
    unittest.main()
