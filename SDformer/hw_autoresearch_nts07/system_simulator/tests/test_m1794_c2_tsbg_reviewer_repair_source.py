#!/usr/bin/env python3
"""Author source/reference tests for additive M1794; never launch EDA."""
from __future__ import print_function

import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1794_c2_tsbg_reviewer_repair_source.py"
SPEC = importlib.util.spec_from_file_location("m1794_checker", str(CHECKER))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


class M1794SourceTest(unittest.TestCase):
    def test_01_source_identity_claim_boundary_and_no_eda(self):
        value = CHECK.validate_sources()
        self.assertEqual(value["status"],
            "PASS_M1794_M1788_REVIEWER_REPAIR_SOURCE_ONLY_NO_EDA")
        self.assertFalse(any(value["claim_boundary"].values()))
        self.assertEqual(value["author_execution"], {
            "vcs_runs": 0, "simv_runs": 0, "dc_runs": 0,
            "ptpx_runs": 0, "license_queries": 0,
            "attempts": 0, "results": 0})

    def test_02_m1787_is_failed_and_immutable(self):
        value = CHECK.validate_sources()["predecessor_disposition"]
        self.assertEqual(value["m1787"], "FAILED_DO_NOT_RELEASE")
        self.assertEqual(value["m1788_findings"], {"p0": 1, "p1": 2, "p2": 1})
        self.assertEqual(CHECK.sha(CHECK.M1787_RTL),
            "f7119779cd5e9adab98cb6252f6a946fd903f68ca341163baa6643033150be94")

    def test_03_directed_tuple_is_legal_under_fixed_production_bound(self):
        value = CHECK.elaborated_parameter_ledger(
            CHECK.RTL.read_text(), CHECK.TB.read_text())
        self.assertEqual(value["production_source_groups"], 48)
        self.assertEqual(value["production_acc24_abs_bound"], 98304)
        self.assertEqual(value["directed_source_groups"], 12)
        self.assertEqual(value["directed_acc24_abs_bound"], 24576)
        self.assertTrue(value["all_dut_tuples_legal"])
        self.assertFalse(value["time_zero_parameter_fatal"])
        self.assertLessEqual(24576, 98304)
        self.assertLess(98304, 1 << 23)

    def test_04_parameter_mutations_detect_time_zero_fatal(self):
        rtl = CHECK.RTL.read_text()
        tb = CHECK.TB.read_text()
        bad_minimum = CHECK.elaborated_parameter_ledger(
            rtl.replace("SOURCE_GROUPS >= 1", "SOURCE_GROUPS >= 48", 1), tb)
        bad_tb = CHECK.elaborated_parameter_ledger(
            rtl, tb.replace("BUNDLE=8, GROUPS=12", "BUNDLE=8, GROUPS=49", 1))
        bad_production = CHECK.elaborated_parameter_ledger(
            rtl.replace("PRODUCTION_SOURCE_GROUPS = 48",
                        "PRODUCTION_SOURCE_GROUPS = 12", 1), tb)
        self.assertTrue(bad_minimum["time_zero_parameter_fatal"])
        self.assertTrue(bad_tb["time_zero_parameter_fatal"])
        self.assertTrue(bad_production["time_zero_parameter_fatal"])

    def test_05_lru8_work_and_beat_terminology(self):
        baseline = CHECK.lru_ledger("token")
        candidate = CHECK.lru_ledger("row")
        self.assertEqual((baseline["row_accesses"], candidate["row_accesses"]),
                         (96, 96))
        self.assertEqual((baseline["hits"], baseline["misses"]), (0, 96))
        self.assertEqual((candidate["hits"], candidate["misses"]), (84, 12))
        self.assertEqual((baseline["aggregate_eight_bank_bundle_beats"],
                          candidate["aggregate_eight_bank_bundle_beats"]),
                         (1152, 144))
        self.assertEqual((baseline["scalar_bank_beats"],
                          candidate["scalar_bank_beats"]), (9216, 1152))

    def test_06_exact_typed_signed_bridge(self):
        self.assertEqual(CHECK.typed_effective(1, 0, -128), -128)
        self.assertEqual(CHECK.typed_effective(1, 1, -128), 128)
        self.assertEqual(CHECK.typed_effective(0, 1, -128), 0)
        self.assertTrue(CHECK.arithmetic_ledger()[
            "exact_neg128_to_positive128"])

    def test_07_work_conservation(self):
        value = CHECK.arithmetic_ledger()
        self.assertEqual(value["issues"], 1152)
        self.assertEqual(value["signed_products"], 18432)
        self.assertEqual(value["commits"], 48)

    def test_08_retired_legal_identity_replay_and_bogus_stale(self):
        value = CHECK.protocol_model()
        self.assertTrue(value["reordered"])
        self.assertTrue(value["all_eight_retired"])
        self.assertEqual(value["saved_accepted_bank"], 3)
        self.assertTrue(value["retired_legal_identity_replay_rejected"])
        self.assertTrue(value["bogus_stale_rejected"])
        tb = CHECK.TB.read_text()
        for token in ("saved_rsp_epoch <= tsbg.mem_rsp_epoch[3]",
                      "tsbg.replay_epoch[3] = saved_rsp_epoch",
                      "retired legal identity replay was accepted",
                      "replay_accept_count != 0"):
            self.assertIn(token, tb)

    def test_09_reset_recovery_runs_complete_legal_service(self):
        tb = CHECK.TB.read_text()
        for token in ("repeat (3) @(posedge clk_core)",
                      "load_minimal_legal_workload()",
                      "post_reset_legal_service_count",
                      "base.issue_count != 96",
                      "base.commit_count != 48",
                      "terminal_base != 8"):
            self.assertIn(token, tb)
        sva = CHECK.SVA.read_text()
        self.assertIn("cp_reset_recovery_minimum_one_cycle", sva)
        self.assertIn("rst_core[*1:8]", sva)
        self.assertIn("commit_accept && commit_terminal", sva)

    def test_10_real_m803_protocol_and_equal_contexts_remain(self):
        rtl = CHECK.strip_comments(CHECK.RTL.read_text())
        self.assertEqual(rtl.count(
            "m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter adapter"),
            1)
        self.assertIn("acc_q [0:BUNDLE-1][0:OUTPUT_SLICES-1][0:LANES-1]",
                      rtl)
        self.assertIn("if (SCHEDULE_MODE == 0)", rtl)

    def test_11_resource_account(self):
        value = CHECK.resource_account()
        self.assertEqual(value["shared_lru8_weight_data_bytes"], 12288)
        self.assertEqual(value["eight_by_96_acc24_context_bytes"], 2304)
        self.assertEqual(value["b8_active_bitmap_bytes"], 768)
        self.assertEqual(value["b8_sign_bitmap_bytes"], 768)
        self.assertEqual(
            value["explicit_datapath_state_bytes_excluding_m803_control"], 16152)

    def test_12_no_lossy_or_product_reuse_path(self):
        rtl = CHECK.strip_comments(CHECK.RTL.read_text())
        for token in ("approx", "epsilon", "drop_source", "reuse_product"):
            self.assertNotIn(token, rtl)
        self.assertIn("bridge_effective_weight[bank][lane] = -widened_weight",
                      rtl)

    def test_13_coverage_obligations_are_source_visible(self):
        sva = CHECK.SVA.read_text()
        for token in ("cp_independent_bank_backpressure",
                      "cp_bank_response_reorder", "cp_cache_eviction",
                      "cp_stale_attack", "ap_fault_is_sticky",
                      "ap_no_legal_overflow"):
            self.assertIn(token, sva)

    def test_14_full_validation_reports_all_parameter_mutations(self):
        value = CHECK.validate_sources()
        self.assertTrue(all(value["parameter_mutations_detected"].values()))
        self.assertEqual(len(value["parameter_mutations_detected"]), 3)


if __name__ == "__main__":
    unittest.main()
