#!/usr/bin/env python3
"""Author static and two-path runtime tests for source-only M1750."""
from __future__ import print_function

import importlib.util
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1750_c1_m1701_public_port_mapped_whole_component_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1750_checker", CHECKER)
CHECK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECK)


def power_report(switching, internal, leakage, total=None):
    if total is None:
        total = switching + internal + leakage
    return ("Report : Averaged Power\nCommand : report_power -unit mW\n"
            "Net Switching Power = %.9f\nCell Internal Power = %.9f\n"
            "Cell Leakage Power = %.9f\nTotal Power = %.9f\n"
            % (switching, internal, leakage, total))


class M1750SourceTest(unittest.TestCase):
    def test_01_source_boundary_and_identity(self):
        value = CHECK.validate_sources()
        self.assertEqual(value["status"], "PASS_M1750_SOURCE_ONLY_NO_EDA")
        self.assertTrue(value["public_port_only"])
        self.assertFalse(value["new_rtl_wrapper"])
        self.assertTrue(all(item is False
                            for item in value["claim_boundary"].values()))
        active = CHECK.strip_sv_comments(CHECK.TB.read_text()).lower()
        self.assertNotIn("force ", active)
        self.assertNotIn("release ", active)
        self.assertNotIn("dut.", active)
        pt = CHECK.PT_TCL.read_text()
        self.assertIn("corner_classification=mixed_corner_component_estimate", pt)
        self.assertIn("standard_cell_power_library=TT_0p9V_25C", pt)
        self.assertIn("parent_sram_macro_liberty=SSG_0p9V_125C", pt)
        self.assertNotIn("report_power $macro_cells", CHECK.strip_sv_comments(pt))

    def test_02_public_counter_runtime_accepts_only_conserved_trace(self):
        good = (
            "M1739_PUBLIC_COUNTERS cycles=777 issue_accepts=145 "
            "parent_edges=20 macro_reads=13 macro_writes=9 forwards=7 "
            "dead_write_elisions=55 psum_commits=64 row_completions=64\n"
            "PASS_M1739_C1_M1701_PUBLIC_PORT_MAPPED_DIRECTED_COMPONENT_ACTIVITY\n")
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "sim.log"
            path.write_text(good)
            value = CHECK.validate_runtime(path)
            self.assertEqual(value["measurement_cycles"], 777)
            self.assertEqual(value["macro_reads"] + value["forwards"],
                             value["parent_edges"])
            path.write_text(good.replace("macro_reads=13", "macro_reads=12"))
            with self.assertRaises(RuntimeError):
                CHECK.validate_runtime(path)
            path.write_text(good.replace("macro_writes=9", "macro_writes=8"))
            with self.assertRaises(RuntimeError):
                CHECK.validate_runtime(path)

    def test_03_whole_component_and_separate_sram_sensitivity(self):
        with tempfile.TemporaryDirectory() as name:
            top = Path(name) / "top.rpt"
            top.write_text(power_report(3.0, 6.0, 1.0))
            value = CHECK.whole_component_power(top, 100, 5, 3)
        whole = value["ptpx_whole_mapped_c1_including_9macro_liberty"]
        self.assertAlmostEqual(whole["total_power_mw"], 10.0)
        self.assertAlmostEqual(whole["directed_window_energy_pj"], 3000.0)
        self.assertTrue(whole["component_total_conserved"])
        parent = value["parent_sram_datasheet_alternative_sensitivity"]
        expected_parent = (5 * 94.57074 + 3 * 90.65763
                           + 0.54009423 * 300.0)
        self.assertAlmostEqual(parent["alternative_sram_energy_pj"],
                               expected_parent)
        self.assertFalse(parent["added_to_ptpx_whole_component"])
        self.assertFalse(value["ptpx_plus_datasheet_sram_combined"])
        boundary = value["claim_boundary"]
        self.assertFalse(boundary["standard_cell_logic_total"])
        self.assertFalse(boundary["top_minus_macro"])
        self.assertFalse(boundary["ptpx_plus_datasheet_sram_combined"])
        self.assertFalse(boundary["total_c1_schedule_energy"])
        self.assertFalse(boundary["energy_per_frame"])
        self.assertFalse(boundary["system_energy"])

    def test_04_rejects_nonconserved_whole_component_summary(self):
        with tempfile.TemporaryDirectory() as name:
            top = Path(name) / "top.rpt"
            top.write_text(power_report(3.0, 6.0, 1.0, total=9.0))
            with self.assertRaises(RuntimeError):
                CHECK.whole_component_power(top, 100, 5, 3)

    def test_05_exact_window_saif_accepts_scope_and_rejects_tx_or_duration(self):
        template = """(SAIFILE
 (DURATION {duration})
 (INSTANCE tb_m1739_c1_m1701_public_port_mapped_production_energy
  (INSTANCE dut
   (NET (clk_core (T0 150) (T1 150) (TX {tx}) (TC 100))))))
"""
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "dut.saif"
            path.write_text(template.format(duration=300, tx=0))
            value = CHECK.validate_saif(path, 100)
            self.assertEqual(value["duration_ns"], 300.0)
            path.write_text(template.format(duration=297, tx=0))
            with self.assertRaises(RuntimeError):
                CHECK.validate_saif(path, 100)
            path.write_text(template.format(duration=300, tx=1))
            with self.assertRaises(RuntimeError):
                CHECK.validate_saif(path, 100)

    def test_06_m1745_fail_and_m1743_timing_are_exactly_pinned(self):
        self.assertEqual(CHECK.sha(CHECK.M1745_REVIEW),
                         "44fca21fde5163ae39f249f5a485c5f2d4953910d8ff76e911aff6a543373359")
        failed = CHECK.strict_json(CHECK.M1745_REVIEW)
        self.assertEqual(failed["p0_count"], 1)
        self.assertFalse(failed["m1746_authorized"])
        self.assertEqual(CHECK.sha(CHECK.M1743),
                         "3c623618115c4ecf2e4bfec6efe167c90296825428ce87e16e6d52bd79216921")
        self.assertEqual(CHECK.sha(CHECK.TIMING_RECEIPT),
                         "0b3ee22f9369a38eb83f674a4f1eb73fac39757ee85a3e1aeebe032bd0c76a1e")
        timing = CHECK.strict_json(CHECK.TIMING_RECEIPT)
        self.assertEqual(timing["prime_time"]["clock_period_ns"], "3.000")
        self.assertEqual(timing["prime_time"]["setup_wns_ns"], "0.027871")
        self.assertEqual(timing["prime_time"]["hold_wns_ns"], "0.001827")
        self.assertEqual(timing["formality"]["passing_compare_points"], 16549)
        self.assertEqual(timing["scope"]["macro_count"], 9)
        self.assertTrue(timing["claim_boundary"]["formality"])
        self.assertTrue(timing["claim_boundary"]["independent_pt"])
        self.assertFalse(timing["claim_boundary"]["power"])
        self.assertFalse(timing["claim_boundary"]["energy"])


if __name__ == "__main__":
    unittest.main()
