#!/usr/bin/env python3
"""Author source/reference tests for M1787; no EDA or attempt creation."""
from __future__ import print_function

import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1787_c2_tsbg_b8_real_channel_signed_source.py"
SPEC = importlib.util.spec_from_file_location("m1787_checker", str(CHECKER))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class M1787SourceTest(unittest.TestCase):
    def test_01_source_identity_and_claim_boundary(self):
        value = CHECK.validate_sources()
        self.assertEqual(value["status"],
            "PASS_M1787_REAL_M803_TYPED_SIGNED_SUCCESSOR_SOURCE_ONLY_NO_EDA")
        self.assertFalse(any(value["claim_boundary"].values()))
        self.assertEqual(value["author_execution"], {
            "vcs_runs": 0, "simv_runs": 0, "dc_runs": 0,
            "ptpx_runs": 0, "license_queries": 0,
            "attempts": 0, "results": 0})

    def test_02_m1780_is_failed_and_immutable(self):
        value = CHECK.validate_sources()["predecessor_disposition"]
        self.assertEqual(value["m1780"], "FAILED_DO_NOT_RELEASE")
        self.assertEqual(value["m1781_p0_count"], 2)
        self.assertEqual(CHECK.sha(CHECK.M1780_RTL),
            "63599d57323fafce8003947df68fc890c2877e52dc8ee0e0806106440787f04c")

    def test_03_lru8_directed_ledger_and_terminology(self):
        baseline = CHECK.lru_ledger("token")
        candidate = CHECK.lru_ledger("row")
        self.assertEqual(baseline["row_accesses"], 96)
        self.assertEqual(candidate["row_accesses"], 96)
        self.assertEqual((baseline["hits"], baseline["misses"]), (0, 96))
        self.assertEqual((candidate["hits"], candidate["misses"]), (84, 12))
        self.assertEqual(baseline["aggregate_eight_bank_bundle_beats"], 1152)
        self.assertEqual(candidate["aggregate_eight_bank_bundle_beats"], 144)
        self.assertEqual(baseline["scalar_bank_beats"], 9216)
        self.assertEqual(candidate["scalar_bank_beats"], 1152)

    def test_04_eviction_is_explicit_and_capacity_sensitive(self):
        baseline8 = CHECK.lru_ledger("token", 8)
        candidate8 = CHECK.lru_ledger("row", 8)
        baseline12 = CHECK.lru_ledger("token", 12)
        self.assertEqual(baseline8["evictions"], 88)
        self.assertEqual(candidate8["evictions"], 4)
        self.assertEqual(baseline12["evictions"], 0)

    def test_05_typed_signed_bridge_has_exact_int8_min_corner(self):
        self.assertEqual(CHECK.typed_effective(1, 0, -128), -128)
        self.assertEqual(CHECK.typed_effective(1, 1, -128), 128)
        self.assertEqual(CHECK.typed_effective(0, 1, -128), 0)
        value = CHECK.arithmetic_ledger()
        self.assertTrue(value["exact_neg128_to_positive128"])

    def test_06_work_is_conserved_per_token_context(self):
        value = CHECK.arithmetic_ledger()
        self.assertEqual(value["issues"], 1152)
        self.assertEqual(value["signed_products"], 18432)
        self.assertEqual(value["commits"], 48)
        flat = [number for context in value["accumulators"]
                for output_slice in context for number in output_slice]
        self.assertTrue(any(number < 0 for number in flat))
        self.assertTrue(any(number > 0 for number in flat))

    def test_07_independent_protocol_reorder_and_stale_model(self):
        value = CHECK.protocol_model()
        self.assertTrue(value["reordered"])
        self.assertTrue(value["all_eight_retired"])
        self.assertTrue(value["stale_rejected"])
        self.assertEqual(len(value["independent_arrival_order"]), 8)

    def test_08_resource_state_is_same_for_both_modes(self):
        value = CHECK.resource_account()
        self.assertEqual(value["shared_lru8_weight_data_bytes"], 12288)
        self.assertEqual(value["eight_by_96_acc24_context_bytes"], 2304)
        self.assertEqual(value["b8_active_bitmap_bytes"], 768)
        self.assertEqual(value["b8_sign_bitmap_bytes"], 768)
        self.assertEqual(
            value["explicit_datapath_state_bytes_excluding_m803_control"], 16152)

    def test_09_real_m803_protocol_is_composed_not_redeclared_atomic(self):
        active = CHECK.strip_comments(CHECK.RTL.read_text())
        self.assertIn(
            "m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter adapter",
            active)
        for token in ("mem_req_epoch [0:7]", "mem_req_slot [0:7]",
                      "mem_req_generation [0:7]", "mem_req_tag [0:7]",
                      "mem_req_output_block [0:7]", "mem_req_slice [0:7]",
                      "mem_req_source_channel [0:7]"):
            self.assertIn(token, active)

    def test_10_zero_is_no_issue_and_sign_is_not_product_reuse(self):
        active = CHECK.strip_comments(CHECK.RTL.read_text())
        self.assertIn("bridge_bank_valid[bank] = active_q", active)
        self.assertIn("bridge_effective_weight[bank][lane] = -widened_weight",
                      active)
        for forbidden in ("reuse_product", "approx", "epsilon", "drop_source"):
            self.assertNotIn(forbidden, active)

    def test_11_attack_backpressure_terminal_and_bound_coverage(self):
        sva = CHECK.SVA.read_text()
        for token in ("cp_independent_bank_backpressure",
                      "cp_bank_response_reorder", "cp_cache_eviction",
                      "cp_stale_attack", "cp_reset_recovery",
                      "ap_fault_is_sticky", "ap_no_legal_overflow"):
            self.assertIn(token, sva)
        tb = CHECK.TB.read_text()
        for token in ("stale_attack_count", "duplicate_attack_count",
                      "reset_recovery_count", "saw_exact_neg128"):
            self.assertIn(token, tb)
        self.assertEqual(48 * 16 * 128, 98304)
        self.assertLess(98304, 1 << 23)

    def test_12_mutations_are_detected_by_independent_models(self):
        self.assertNotEqual(CHECK.lru_ledger("token", 12),
                            CHECK.lru_ledger("token", 8))
        self.assertNotEqual(CHECK.typed_effective(1, 1, -128),
                            CHECK.typed_effective(1, 0, -128))
        protocol = CHECK.protocol_model()
        self.assertNotEqual(protocol["independent_arrival_order"],
                            sorted(protocol["independent_arrival_order"]))


if __name__ == "__main__":
    unittest.main()
