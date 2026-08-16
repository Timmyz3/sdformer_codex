#!/usr/bin/env python3

from __future__ import annotations

import unittest

from scripts.analyze_h67_class_gate_cache_baseline_v1 import analyze


def fixture() -> dict:
    row = {
        "row": 0,
        "active": 8,
        "fixed_exp": 11,
        "rqtb_exp": 8,
        "fixed_cycles": 20,
        "rqtb_cycles": 16,
    }
    return {
        "schema": "h67_rqtb_strong_baseline_v1",
        "status": "PASS",
        "scope": "test",
        "input_identity": {"vector_sha256": "a" * 64},
        "work": {"fixed_exp": 11 * 138, "rqtb_exp": 8 * 138},
        "rows_2s": [dict(row, row=index) for index in range(138)],
    }


class H67ClassGateCacheBaselineTest(unittest.TestCase):
    def test_cache_removes_exp_and_gate_compute_differential(self) -> None:
        result = analyze(fixture())
        self.assertEqual(result["baseline_counts"]["unique_class_transactions"], 414)
        self.assertEqual(result["baseline_counts"]["fixed_active_descriptors"], 1104)
        self.assertEqual(result["baseline_counts"]["rqtb_active_descriptors"], 690)
        self.assertEqual(result["class_exp_cache"]["rqtb_exp_lut_reduction_vs_fixed"], 0.0)
        self.assertEqual(result["class_gate_cache"]["rqtb_gate_quant_reduction_vs_fixed"], 0.0)
        self.assertEqual(result["class_exp_cache"]["compact_storage_bits"], 163 * 10)
        self.assertEqual(
            result["class_exp_cache"]["rtl_interface_storage_bits"],
            163 * 17,
        )
        self.assertEqual(result["class_gate_cache"]["storage_bits"], 163 * 10)

    def test_rejects_invalid_rqtb_descriptor_count(self) -> None:
        value = fixture()
        value["rows_2s"][0]["rqtb_exp"] = 20
        with self.assertRaisesRegex(ValueError, "闭式"):
            analyze(value)

    def test_rejects_unbound_report(self) -> None:
        value = fixture()
        value["input_identity"] = {}
        with self.assertRaisesRegex(ValueError, "身份"):
            analyze(value)


if __name__ == "__main__":
    unittest.main()
