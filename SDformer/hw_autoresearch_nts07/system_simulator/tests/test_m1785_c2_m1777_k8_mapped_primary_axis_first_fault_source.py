#!/usr/bin/env python3
"""Mutation tests for the M1785 source-only first-fault diagnostic."""

from __future__ import print_function

import importlib.util
from pathlib import Path
import unittest


HW = Path(__file__).resolve().parents[2]
CHECKER = HW / "system_simulator/scripts/check_m1785_c2_m1777_k8_mapped_primary_axis_first_fault_source.py"
SPEC = importlib.util.spec_from_file_location("m1785_check", str(CHECKER))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1785Tests(unittest.TestCase):
    def test_01_full_source_check(self):
        value = M.main()
        self.assertEqual(value["status"],
                         "PASS_M1785_SOURCE_ONLY_NO_EDA_NO_ATTEMPT")
        self.assertEqual(value["m1777_counts"], {
            "ptpx_runs": 0, "saif_files": 0,
            "simv_runs": 1, "vcs_compiles": 1})

    def test_02_tb_removing_settle_is_rejected(self):
        text = M.TB.read_text(encoding="utf-8")
        with self.assertRaises(M.Failure):
            M.audit_tb(text.replace("#1ps;", "", 1))

    def test_03_tb_removing_original_wrapper_is_rejected(self):
        text = M.TB.read_text(encoding="utf-8")
        with self.assertRaises(M.Failure):
            M.audit_tb(text.replace(
                "tb_m1684_c2_m1609_fresh_mapped_production_energy sealed();",
                "", 1))

    def test_04_tb_x_suppression_is_rejected(self):
        text = M.TB.read_text(encoding="utf-8")
        with self.assertRaises(M.Failure):
            M.audit_tb(text.replace("ignore_x=0", "ignore_x=1", 1))

    def test_05_filelist_rtl_injection_is_rejected(self):
        text = M.FILELIST.read_text(encoding="utf-8")
        with self.assertRaises(M.Failure):
            M.audit_filelist(text + "rtl_m803/injected.sv\n")

    def test_06_filelist_drops_exact_assertion_is_rejected(self):
        text = M.FILELIST.read_text(encoding="utf-8")
        with self.assertRaises(M.Failure):
            M.audit_filelist(text.replace(str(M.M1684_ASSERT), "", 1))

    def test_07_runtime_parser_localizes_protocol(self):
        log = "\n".join([
            "M1785_FIRST_UNKNOWN code=1 class=FAULT field=protocol_error time_ps=25501",
            "M1684 mapped fault vector contains X/Z",
            "M1785_FINAL first_unknown_seen=1 first_unknown_code=1 first_unknown_time_ps=25501 settled_samples=4 exact_m1684_assertion_preserved=1 initreg=0 force=0 ignore_x=0",
        ])
        value = M.check_runtime_text(log)
        self.assertEqual(value["classification"], "PROTOCOL_ERROR_PUBLIC_XZ")

    def test_08_runtime_parser_rejects_missing_exact_failure(self):
        log = "\n".join([
            "M1785_FIRST_UNKNOWN code=2 class=FAULT field=numeric_overflow time_ps=25501",
            "M1785_FINAL first_unknown_seen=1 first_unknown_code=2 first_unknown_time_ps=25501 settled_samples=4 exact_m1684_assertion_preserved=1 initreg=0 force=0 ignore_x=0",
        ])
        with self.assertRaises(M.Failure):
            M.check_runtime_text(log)

    def test_09_runtime_parser_rejects_private_only_localization(self):
        log = "\n".join([
            "M1785_FIRST_UNKNOWN code=37 class=DIAGNOSTIC_TAP field=registered_fault_taps time_ps=25501",
            "M1684 mapped fault vector contains X/Z",
            "M1785_FINAL first_unknown_seen=1 first_unknown_code=37 first_unknown_time_ps=25501 settled_samples=4 exact_m1684_assertion_preserved=1 initreg=0 force=0 ignore_x=0",
        ])
        with self.assertRaises(M.Failure):
            M.check_runtime_text(log)

    def test_10_runner_force_or_initreg_injection_is_rejected(self):
        text = M.RUNNER.read_text(encoding="utf-8")
        for mutant in (text + "\nforce dut.q 0\n",
                       text + "\n+vcs+initreg+0\n"):
            with self.assertRaises(M.Failure):
                M.audit_runner(mutant)


if __name__ == "__main__":
    unittest.main(verbosity=2)
