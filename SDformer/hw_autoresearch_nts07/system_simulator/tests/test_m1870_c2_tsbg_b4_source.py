#!/usr/bin/env python3
"""Author source/reference tests for M1870 B4; never launch EDA."""
from __future__ import print_function

import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1870_c2_tsbg_b4_source.py"
SPEC = importlib.util.spec_from_file_location("m1870_checker", str(CHECKER))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class M1870B4SourceTest(unittest.TestCase):
    def test_01_source_identity_claim_boundary_and_no_eda(self):
        value = CHECK.validate_sources()
        self.assertEqual(value["status"],
                         "PASS_M1870_C2_TSBG_B4_SOURCE_STATIC_NO_EDA")
        self.assertTrue(value["claim_boundary"]["source_only"])
        self.assertFalse(value["claim_boundary"]["same_area"])
        self.assertFalse(value["claim_boundary"]["paper_admitted"])
        self.assertEqual(value["author_execution"], {
            "vcs_runs": 0, "simv_runs": 0, "dc_runs": 0,
            "ptpx_runs": 0, "license_queries": 0, "attempts": 0,
            "results": 0, "releases": 0})

    def test_02_m1794_b8_and_m803_are_immutable(self):
        self.assertEqual(CHECK.sha(CHECK.M1794_RTL),
                         CHECK.FIXED[CHECK.M1794_RTL])
        self.assertEqual(CHECK.sha(CHECK.M803), CHECK.FIXED[CHECK.M803])

    def test_03_only_b4_lru4_specializes_compute_commit_rtl(self):
        candidate = CHECK.RTL.read_text()
        self.assertEqual(CHECK.normalize_candidate_rtl_to_m1794(candidate),
                         CHECK.M1794_RTL.read_text())

    def test_04_b4_lru4_parameter_point(self):
        rtl = CHECK.RTL.read_text()
        for token in ("parameter int BUNDLE = 4",
                      "parameter int CACHE_ROWS = 4",
                      "BUNDLE == 4 && SOURCES_PER_GROUP == 16",
                      "CACHE_ROWS == 4 && LANES == 16"):
            self.assertIn(token, rtl)
        CHECK.validate_rtl_text(rtl)

    def test_05_token_major_ordinary_lru4_exact_ledger(self):
        self.assertEqual(CHECK.lru_ledger("token"), {
            "rows": 48, "hits": 0, "misses": 48, "evictions": 44,
            "aggregate_eight_bank_bundle_beats": 576,
            "scalar_bank_beats": 4608})

    def test_06_group_major_tsbg_lru4_exact_ledger(self):
        self.assertEqual(CHECK.lru_ledger("group"), {
            "rows": 48, "hits": 36, "misses": 12, "evictions": 8,
            "aggregate_eight_bank_bundle_beats": 144,
            "scalar_bank_beats": 1152})

    def test_07_same_work_conservation(self):
        value = CHECK.arithmetic_ledger()
        self.assertEqual((value["issues"], value["products"], value["commits"]),
                         (576, 9216, 24))

    def test_08_mixed_signed_and_int8_min_corner(self):
        value = CHECK.arithmetic_ledger()
        self.assertTrue(value["positive"])
        self.assertTrue(value["negative"])
        self.assertTrue(value["exact_neg128_to_positive128"])
        self.assertEqual(-CHECK.directed_weight(0, 0, 0, 0, 0), 128)

    def test_09_production_acc24_bound_stays_48_groups(self):
        rtl = CHECK.RTL.read_text()
        self.assertIn("PRODUCTION_SOURCE_GROUPS = 48", rtl)
        self.assertIn("PRODUCTION_ACC24_ABS_BOUND == 98304", rtl)
        self.assertLess(98304, 1 << 23)

    def test_10_four_acc24_contexts_and_eight_physical_banks(self):
        rtl = CHECK.RTL.read_text()
        self.assertIn("acc_q [0:BUNDLE-1][0:OUTPUT_SLICES-1][0:LANES-1]", rtl)
        self.assertIn("mem_req_source_channel [0:7]", rtl)
        self.assertEqual(CHECK.resource_account()[
            "four_by_96_acc24_context_bytes"], 1152)

    def test_11_resource_account(self):
        self.assertEqual(CHECK.resource_account(), {
            "shared_lru4_weight_data_bytes": 6144,
            "four_by_96_acc24_context_bytes": 1152,
            "b4_active_bitmap_bytes": 384,
            "b4_sign_bitmap_bytes": 384,
            "context_tag_bytes": 12,
            "explicit_datapath_state_bytes_excluding_m803_control": 8076})

    def test_12_no_lossy_or_product_reuse_path(self):
        active = CHECK.strip_comments(CHECK.RTL.read_text())
        for token in ("approx", "epsilon", "drop_source", "reuse_product",
                      "reuse_effective_weight", "product_cache"):
            self.assertNotIn(token, active)
        self.assertIn("bridge_effective_weight[bank][lane] = -widened_weight",
                      active)

    def test_13_tb_exact_full_workload_ledgers(self):
        text = CHECK.TB.read_text()
        CHECK.validate_tb_text(text)
        for token in ("base.cache_miss_count != 48",
                      "tsbg.cache_miss_count != 12",
                      "base.cache_eviction_count != 44",
                      "tsbg.cache_eviction_count != 8"):
            self.assertIn(token, text)

    def test_14_backpressure_and_stall_coverage(self):
        sva = CHECK.SVA.read_text()
        for token in ("cp_independent_bank_backpressure",
                      "cp_bridge_stall", "cp_commit_stall",
                      "ap_bank_request_stable", "ap_commit_payload_stable"):
            self.assertIn(token, sva)

    def test_15_stale_and_retired_replay_fail_closed(self):
        tb = CHECK.TB.read_text()
        for token in ("saved_rsp_epoch <= tsbg.mem_rsp_epoch[3]",
                      "tsbg.replay_epoch[3] = saved_rsp_epoch",
                      "retired legal identity replay was accepted",
                      "replay_accept_count != 0",
                      "bogus stale response was accepted"):
            self.assertIn(token, tb)

    def test_16_reset_recovery_runs_complete_b4_service(self):
        tb = CHECK.TB.read_text()
        for token in ("repeat (3) @(posedge clk_core)",
                      "load_minimal_legal_workload()",
                      "post_reset_legal_service_count",
                      "base.row_access_count != 4",
                      "base.commit_count != 24",
                      "terminal_base != 4"):
            self.assertIn(token, tb)
        self.assertIn("cp_reset_recovery_minimum_one_cycle", CHECK.SVA.read_text())

    def test_17_context_range_sva(self):
        sva = CHECK.SVA.read_text()
        CHECK.validate_sva_text(sva)
        for token in ("ap_load_context_is_b4", "ap_bridge_context_is_b4",
                      "ap_commit_context_is_b4"):
            self.assertIn(token, sva)

    def test_18_filelist_exact_order(self):
        rows = [row.strip() for row in CHECK.FILELIST.read_text().splitlines()
                if row.strip()]
        self.assertEqual(rows, [str(path.relative_to(CHECK.ROOT)) for path in
                                (CHECK.M803, CHECK.RTL, CHECK.SVA, CHECK.TB)])

    def test_19_parameter_and_ledger_mutations_are_rejected(self):
        rtl = CHECK.RTL.read_text()
        tb = CHECK.TB.read_text()
        with self.assertRaises(CHECK.CheckFailure):
            CHECK.validate_rtl_text(rtl.replace("parameter int BUNDLE = 4",
                                                "parameter int BUNDLE = 8", 1))
        with self.assertRaises(CHECK.CheckFailure):
            CHECK.validate_rtl_text(rtl.replace("parameter int CACHE_ROWS = 4",
                                                "parameter int CACHE_ROWS = 8", 1))
        with self.assertRaises(CHECK.CheckFailure):
            CHECK.validate_tb_text(tb.replace("base.cache_miss_count != 48",
                                              "base.cache_miss_count != 47", 1))
        with self.assertRaises(CHECK.CheckFailure):
            CHECK.validate_tb_text(tb.replace("tsbg.cache_hit_count != 36",
                                              "tsbg.cache_hit_count != 35", 1))

    def test_20_future_authority_is_review_then_release(self):
        value = CHECK.validate_contract()
        self.assertEqual(value["future_authority"], {
            "different_author_source_review": "M1871",
            "exact_launch_release": "M1872",
            "review_required_before_release": True,
            "postrun_different_author_result_review_required": True})

    def test_21_m1866_independent_ruling_selects_only_b4_source(self):
        value = CHECK.strict_json(CHECK.M1866 / "review.json")
        self.assertEqual(value["evidence_quality"], {
            "p0_count": 0, "p1_count": 0, "p2_count": 0,
            "score_over_100": 99, "status": "PASS"})
        self.assertEqual(value["rtl_source_ruling"]["single_selected_bundle"], 4)
        self.assertTrue(value["authorization"][
            "b4_new_fail_closed_source_contract_may_be_authored"])
        self.assertFalse(value["authorization"]["b4_rtl_execution"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
