#!/usr/bin/env python3
"""Author static and fail-closed parser tests for source-only M1782."""
from __future__ import print_function

import importlib.util
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1782_c1_expected_macro_leaf_blackbox_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1782_checker_test", str(CHECKER))
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def inventory(rows=None):
    if rows is None:
        rows = ["name=%s ref=%s is_hierarchical=false is_black_box=true" %
                (name, CHECK.MACRO_REF) for name in CHECK.EXPECTED_MACRO_NAMES]
    return "\n".join(["black_box_count=9", "expected_macro_count=9"] + rows) + "\n"


class M1782SourceTest(unittest.TestCase):
    def test_01_source_boundary_and_fresh_budget(self):
        value = CHECK.validate_sources()
        self.assertEqual(value["status"],
                         "PASS_M1782_EXACT_EXPECTED_MACRO_LEAF_BLACKBOX_SOURCE_ONLY_NO_EDA")
        self.assertEqual(value["black_box_policy"],
                         "exact_9_expected_linked_sram_liberty_leaves_only")
        self.assertEqual(value["fresh_execution_budget"], {
            "ptpx_runs": 1, "saif_files": 1, "simv_runs": 1,
            "vcs_compiles": 1, "reuse_m1772_private_build": False})
        self.assertTrue(all(item is False
                            for item in value["claim_boundary"].values()))

    def test_02_m1772_failure_is_bound_before_pt_saif_read(self):
        value = CHECK.validate_m1772_failure()
        self.assertEqual(value["failure_phase"], "PTPX_post_link_pre_SAIF")
        self.assertEqual(value["measurement_cycles"], 253)
        self.assertEqual(value["saif_activity_forms_per_tag"], 117690)
        self.assertEqual(value["saif_tx_nonzero"], 0)
        self.assertFalse(value["ptpx_power_result"])
        self.assertFalse(value["automatic_retry"])

    def test_03_accepts_exact_nine_expected_leaf_macros(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "inventory.rpt"
            path.write_text(inventory())
            value = CHECK.validate_black_box_inventory(path)
        self.assertEqual(value["count"], 9)
        self.assertEqual(value["unexpected_black_boxes"], 0)
        self.assertEqual(value["missing_expected_macros"], 0)
        self.assertEqual(set(value["names"]), set(CHECK.EXPECTED_MACRO_NAMES))

    def test_04_rejects_missing_extra_wrong_ref_hierarchical_or_nonblackbox(self):
        good = ["name=%s ref=%s is_hierarchical=false is_black_box=true" %
                (name, CHECK.MACRO_REF) for name in CHECK.EXPECTED_MACRO_NAMES]
        mutations = [
            good[:-1],
            good + ["name=u_extra ref=OTHER is_hierarchical=false is_black_box=true"],
            [good[0].replace(CHECK.MACRO_REF, "WRONG_REF")] + good[1:],
            [good[0].replace("is_hierarchical=false", "is_hierarchical=true")] + good[1:],
            [good[0].replace("is_black_box=true", "is_black_box=false")] + good[1:],
            [good[0]] + good,
        ]
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "inventory.rpt"
            for rows in mutations:
                path.write_text(inventory(rows))
                with self.assertRaises(RuntimeError):
                    CHECK.validate_black_box_inventory(path)

    def test_05_tcl_keeps_blackbox_gate_and_whole_component_accounting(self):
        text = CHECK.PT_TCL.read_text()
        active = CHECK.strip_tcl_comments(text)
        self.assertIn('get_cells -hierarchical -filter "is_black_box==true"', text)
        self.assertIn("M1782_FAIL_UNEXPECTED_BLACK_BOX_", text)
        self.assertIn("M1782_FAIL_EXPECTED_MACRO_BLACK_BOX_MISSING_", text)
        self.assertNotIn(
            'sizeof_collection [get_cells -hierarchical -filter "is_black_box==true"]] != 0',
            active)
        self.assertNotIn("report_power $macro_cells", active)
        self.assertIn("ptpx_whole_mapped_c1_including_9macro_liberty.rpt", text)
        self.assertIn("unresolved_or_unexpected_black_box_allowed=false", text)

    def test_06_runner_is_fresh_and_does_not_reuse_m1772_private_build(self):
        text = CHECK.RUNNER.read_text()
        self.assertEqual(text.count('"+define+UNIT_DELAY"'), 1)
        self.assertIn('saif = candidate / "m1782_c1_directed_component.saif"', text)
        self.assertIn("validate_black_box_inventory(", text)
        self.assertNotIn(
            "m1772_c1_two_bank_public_warmup_energy_r1_20260902.private_build",
            text)
        for forbidden in ("+notimingcheck", "+no_notifier", "+nospecify",
                          "+initreg", "ignore" + "_black_box"):
            self.assertNotIn(forbidden, text)

    def test_07_predecessor_runtime_saif_and_power_parsers_remain_active(self):
        good = (
            "COVERAGE_M1772_TWO_BANK_PUBLIC_WARMUP bank0_epoch=5943 "
            "bank1_epoch=5944 public_backpressure=1 hierarchy_drive=0\n"
            "M1772_PUBLIC_COUNTERS cycles=253 issue_accepts=96 "
            "parent_edges=48 macro_reads=46 macro_writes=34 forwards=2 "
            "dead_write_elisions=30 psum_commits=64 row_completions=64\n"
            "PASS_M1772_C1_M1701_TWO_BANK_WARMUP_MAPPED_DIRECTED_COMPONENT_ACTIVITY\n")
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "sim.log"
            path.write_text(good)
            value = CHECK.validate_runtime(path)
            self.assertEqual(value["measurement_cycles"], 253)
            path.write_text(good.replace("macro_reads=46", "macro_reads=45"))
            with self.assertRaises(RuntimeError):
                CHECK.validate_runtime(path)


if __name__ == "__main__":
    unittest.main()
