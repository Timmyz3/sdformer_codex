#!/usr/bin/env python3
"""Mutation tests for the source-only M1777 two-axis C2 campaign."""
from __future__ import print_function

import copy
import importlib.util
import math
from pathlib import Path
import unittest


HW = Path(__file__).resolve().parents[2]
CHECKER = HW / "system_simulator/scripts/check_m1777_m1776_c2_two_axis_equal_bandwidth_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1777_source_check", str(CHECKER))
M = importlib.util.module_from_spec(SPEC)
if SPEC.loader is None:
    raise RuntimeError("checker loader unavailable")
SPEC.loader.exec_module(M)


class M1777Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = M.RUNNER.read_text()
        cls.contract = M.strict_json(M.CONTRACT)

    def rejected(self, function, value):
        with self.assertRaises(RuntimeError):
            function(value)

    def rows(self):
        rows = []
        power = {"k8": 2.0, "k1x8": 8.0}
        for axis in M.AXES:
            for case_id, cycles in enumerate(M.AXES[axis]["cycles"]):
                total = power[axis] + case_id * 0.1
                rows.append({"axis": axis, "case": case_id, "cycles": cycles,
                             "accepted_sources": M.EVENTS[case_id],
                             "net_switching_mw": total * 0.3,
                             "cell_internal_mw": total * 0.6,
                             "cell_leakage_mw": total * 0.1,
                             "total_mw": total})
        return rows

    def test_01_live_source(self):
        value = M.validate_sources()
        self.assertEqual(value["status"], "PASS_M1777_SOURCE_ONLY_NO_EDA")
        self.assertEqual(value["axes"], ["k8", "k1x8"])

    def test_02_k1_reintroduction_rejected(self):
        self.rejected(M.validate_runner_text,
                      self.text.replace('AXES = ("k8", "k1x8")',
                                        'AXES = ("k1", "k8", "k1x8")'))

    def test_03_budget_mutation_rejected(self):
        value = copy.deepcopy(self.contract)
        value["future_budget"]["simv_runs"] = 9
        self.rejected(M.validate_contract_value, value)

    def test_04_order_mutation_rejected(self):
        text = self.text.replace(
            'state["phase"] = "PTPX_" + axis + "_" + str(case_id)',
            'state["phase"] = "PTPX_MUTATED_" + axis + "_" + str(case_id)', 1)
        self.rejected(M.validate_runner_text, text)

    def test_05_interpreter_drift_rejected(self):
        value = copy.deepcopy(self.contract)
        value["interpreter_identity"]["version"] = "3.12.12"
        self.rejected(M.validate_contract_value, value)

    def test_06_partial_publication_rejected(self):
        self.rejected(M.validate_runner_text,
                      self.text.replace('"partial_axis_citable": False',
                                        '"partial_axis_citable": True'))

    def test_07_retry_rejected(self):
        value = copy.deepcopy(self.contract)
        value["future_budget"]["automatic_retry"] = True
        self.rejected(M.validate_contract_value, value)

    def test_08_assertion_weakening_rejected(self):
        value = copy.deepcopy(self.contract)
        value["fault_integrity"]["assertion_sha256"] = "0" * 64
        self.rejected(M.validate_contract_value, value)

    def test_09_missing_primary_axis_rejected(self):
        rows = [row for row in self.rows() if row["axis"] != "k1x8"]
        self.rejected(M.aggregate_metrics, rows)

    def test_10_extra_k1_axis_rejected(self):
        rows = self.rows()
        extra = copy.deepcopy(rows[0])
        extra["axis"] = "k1"
        rows.append(extra)
        self.rejected(M.aggregate_metrics, rows)

    def test_11_cycle_mutation_rejected(self):
        rows = self.rows()
        rows[0]["cycles"] += 1
        self.rejected(M.aggregate_metrics, rows)

    def test_12_power_decomposition_rejected(self):
        rows = self.rows()
        rows[0]["total_mw"] += 1.0
        self.rejected(M.aggregate_metrics, rows)

    def test_13_two_axis_metrics_and_joint_disclosure(self):
        value = M.aggregate_metrics(self.rows())
        self.assertEqual(value["axes"]["k8"]["cycles"], 1913)
        self.assertEqual(value["axes"]["k1x8"]["cycles"], 1945)
        self.assertTrue(math.isclose(
            value["equal_bandwidth_cycle_speedup_k8_vs_k1x8"], 1945.0 / 1913.0))
        self.assertTrue(math.isclose(
            value["equal_bandwidth_throughput_per_mm2_k8_vs_k1x8"],
            (1945.0 * M.AXES["k1x8"]["area_um2"])
            / (1913.0 * M.AXES["k8"]["area_um2"])))
        self.assertTrue(value["joint_disclosure_required"])
        self.assertEqual(value["k1_dc_role"], "DIAGNOSTIC_ONLY")


if __name__ == "__main__":
    unittest.main()
