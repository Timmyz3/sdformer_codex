#!/usr/bin/env python3
"""CPU-only tests for the M1753 C2 three-axis energy source."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
CHECKER = HERE.parent / "scripts/check_m1753_m1715_c2_three_axis_mapped_directed_component_energy_source.py"
SPEC = importlib.util.spec_from_file_location("m1753_checker_test", CHECKER)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


class M1753Tests(unittest.TestCase):
    def rows(self):
        rows = []
        power = {"k1": 3.0, "k8": 2.0, "k1x8": 8.0}
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

    def test_source(self):
        value = M.validate_sources()
        self.assertEqual(value["status"], "PASS_M1753_SOURCE_ONLY_NO_EDA")
        self.assertEqual(value["axes"], ["k1", "k8", "k1x8"])
        self.assertEqual(value["accepted_sources_per_axis"], 261)
        self.assertEqual(value["workload_class"], "DIRECTED_COMPONENT_NOT_PRODUCTION")

    def test_aggregate_three_axes_and_joint_disclosure(self):
        value = M.aggregate_metrics(self.rows())
        self.assertEqual(value["axes"]["k1"]["cycles"], 11732)
        self.assertEqual(value["axes"]["k8"]["cycles"], 1913)
        self.assertEqual(value["axes"]["k1x8"]["cycles"], 1945)
        self.assertTrue(math.isclose(
            value["equal_bandwidth_cycle_speedup_k8_vs_k1x8"], 1945.0 / 1913.0))
        self.assertTrue(math.isclose(
            value["equal_bandwidth_throughput_per_mm2_k8_vs_k1x8"],
            (1945.0 * M.AREAS_UM2["k1x8"]) / (1913.0 * M.AREAS_UM2["k8"])))
        self.assertTrue(value["joint_disclosure_required"])
        self.assertTrue(value["k8_vs_single_k1_headline_forbidden"])

    def test_missing_axis_rejected(self):
        rows = [row for row in self.rows() if row["axis"] != "k1"]
        with self.assertRaises(RuntimeError):
            M.aggregate_metrics(rows)

    def test_cycle_mutation_rejected(self):
        rows = self.rows()
        rows[0]["cycles"] += 1
        with self.assertRaises(RuntimeError):
            M.aggregate_metrics(rows)

    def test_denominator_mutation_rejected(self):
        rows = self.rows()
        rows[0]["accepted_sources"] += 1
        with self.assertRaises(RuntimeError):
            M.aggregate_metrics(rows)

    def test_power_decomposition_mutation_rejected(self):
        rows = self.rows()
        rows[0]["total_mw"] += 1.0
        with self.assertRaises(RuntimeError):
            M.aggregate_metrics(rows)

    def test_power_report_requires_all_four_fields(self):
        good = """Report : Averaged Power
-unit mW
Net Switching Power = 1.00000000
Cell Internal Power = 2.00000000
Cell Leakage Power = 0.10000000
Total Power = 3.10000000
"""
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "power.rpt"
            path.write_text(good)
            value = M.parse_power_report(path)
            self.assertEqual(value["report_scope"], "WHOLE_MAPPED_COMPONENT")
            self.assertEqual(value["total_mw"], 3.1)
            path.write_text(good.replace("Total Power = 3.10000000\n", ""))
            with self.assertRaises(RuntimeError):
                M.parse_power_report(path)


if __name__ == "__main__":
    unittest.main()
