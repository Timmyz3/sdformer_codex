#!/usr/bin/env python3
"""No-EDA author tests for the M1684 C2 production energy source package."""
from __future__ import print_function

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1684_c2_m1609_fresh_mapped_production_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1684_source_checker", str(CHECKER))
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


class M1684Tests(unittest.TestCase):
    def test_01_predecessor_and_fresh_netlist_chain(self):
        M.validate_predecessors()
        self.assertEqual(M.sha(M.BASE / "k8" / M.NET_REL),
                         M.AXES["k8"]["net_sha"])
        self.assertEqual(M.sha(M.BASE / "k1x8" / M.NET_REL),
                         M.AXES["k1x8"]["net_sha"])

    def test_02_exact_equal_workload_filelists(self):
        for axis in M.AXES:
            M.validate_filelist(axis)
            self.assertEqual(M.active_lines(M.FILELISTS[axis]),
                             M.expected_filelist(axis))

    def test_03_old_netlist_or_axis_mutation_rejected(self):
        text = M.FILELISTS["k8"].read_text()
        changed = text.replace(M.AXES["k8"]["net_sha"], "unused")
        # The SHA is not literally in the filelist; mutate the fresh path and
        # separately mutate the active axis define.
        changed = changed.replace("m1661_m1652", "m872_m803", 1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.f"
            path.write_text(changed)
            with self.assertRaisesRegex(RuntimeError, "filelist/order"):
                M.validate_filelist("k8", path)
            path.write_text(text.replace("M979_AXIS_K8", "M979_AXIS_K1X8", 1))
            with self.assertRaisesRegex(RuntimeError, "filelist/order"):
                M.validate_filelist("k8", path)

    def test_04_registered_fault_and_half_cycle_gates(self):
        text = M.ASSERT.read_text()
        for token in ("ap_public_fault_binary",
                      "ap_registered_public_fault_zero",
                      "@(negedge clk_core)",
                      "accepted_sources != expected_sources(case_id)",
                      "registered_fault_public_zero=1"):
            self.assertIn(token, text)
        self.assertNotRegex(text.lower(), r"(?m)^\s*force\s")

    def test_05_runner_geometry_order_and_no_state_masking_flag(self):
        text = M.RUNNER.read_text()
        self.assertNotIn("initreg", text.lower())
        self.assertGreaterEqual(text.count('for axis in ("k8", "k1x8"):'), 3)
        self.assertGreaterEqual(text.count("for case_id in range(5):"), 2)
        self.assertLess(text.index("all ten mapped production SAIF gates"),
                        text.index('state["phase"] = "PTPX_"'))
        self.assertIn('"vcs_compiles": 2', text)
        self.assertIn('"ptpx_runs": 10', text)

    def test_06_runtime_log_denominator_and_fault_token(self):
        template = (
            "PASS M1334 coverage case=0 source=1 endpoint=120 commit=6 stall=3 done=1 unknown=0 fatal=0\n"
            "PASS M1684 M1609 binary-clean production case=0 accepted_sources=20 source_packets=1 endpoint_accepts=120 result_accepts=6 done_accepts=1 fault_binary_clean=1 registered_fault_public_zero=1\n"
            "PASS M979 mapped replay axis=K8 case=0 events=20 cycles=51 saif_duration_ns=153 numeric_mismatches=0 tuple_mismatches=0 weight_mismatches=0 accepted_unknowns=0 protocol_errors=0\n")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.log"
            path.write_text(template)
            value = M.validate_runtime_log(path, "k8", 0)
            self.assertEqual(value["accepted_sources"], 20)
            path.write_text(template.replace("accepted_sources=20",
                                             "accepted_sources=19"))
            with self.assertRaisesRegex(RuntimeError, "source denominator drift"):
                M.validate_runtime_log(path, "k8", 0)

    def test_07_power_parser_and_aggregate_math(self):
        report = """Report : Averaged Power
    -unit mW
Net Switching Power  = 1.00000000e+00
Cell Internal Power  = 2.00000000e+00
Cell Leakage Power   = 1.00000000e-01
Total Power          = 3.10000000e+00
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "power.rpt"
            path.write_text(report)
            parsed = M.parse_power_report(path)
            self.assertEqual(parsed["total_mw"], 3.1)
        rows = []
        for axis, power in (("k8", 2.0), ("k1x8", 4.0)):
            for case_id in range(5):
                rows.append({"axis": axis, "case": case_id,
                             "cycles": M.AXES[axis]["cycles"][case_id],
                             "accepted_sources": M.EVENTS[case_id],
                             "total_mw": power})
        metrics = M.aggregate_metrics(rows)
        self.assertAlmostEqual(
            metrics["equal_bandwidth_cycle_speedup_k8_vs_k1x8"],
            1945.0 / 1913.0)
        self.assertAlmostEqual(
            metrics["equal_bandwidth_energy_ratio_k1x8_over_k8"],
            (4.0 * 1945.0) / (2.0 * 1913.0))
        self.assertEqual(metrics["axes"]["k8"]["accepted_sources"], 261)

    def test_08_duplicate_and_nonfinite_json_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"x":1,"x":2}')
            with self.assertRaisesRegex(RuntimeError, "duplicate JSON"):
                M.strict_json(path)
            path.write_text('{"x":NaN}')
            with self.assertRaisesRegex(RuntimeError, "nonfinite JSON"):
                M.strict_json(path)

    def test_09_complete_source_contract_and_no_execution(self):
        value = M.validate_sources()
        self.assertEqual(value["status"], "PASS_M1684_SOURCE_ONLY_NO_EDA")
        self.assertEqual(value["accepted_sources_per_axis"], 261)
        self.assertTrue(all(item is False
                            for item in value["claim_boundary"].values()))
        for path in (M.M1685, M.M1686,
                     M.HW / "results/.m1684_c2_mapped_production_energy_attempt_consumed",
                     M.HW / "results/m1684_c2_mapped_production_energy_r1_20260901"):
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
