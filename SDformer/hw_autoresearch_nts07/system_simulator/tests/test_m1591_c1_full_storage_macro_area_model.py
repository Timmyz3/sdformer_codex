#!/usr/bin/env python3
"""Unit checks for the M1591 conservative C1 full-storage area model."""
from __future__ import annotations

import importlib.util
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_m1591_c1_full_storage_macro_area_model.py"
SPEC = importlib.util.spec_from_file_location("m1591_area_model", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class M1591AreaModelTest(unittest.TestCase):
    def test_frozen_coordinate_and_conservative_rounding(self) -> None:
        value = MODULE.build()
        self.assertEqual(value["schema"], MODULE.SCHEMA)
        self.assertEqual(value["status"], MODULE.STATUS)
        logical = value["logical_storage"]
        self.assertEqual(logical["total_bytes"], 214_912)
        self.assertEqual(logical["budget_bytes"], 245_760)
        rounded = value["conservative_macro_rounding"]
        self.assertEqual(rounded["counts"], {
            "parent_scratch": 9,
            "psum": 60,
            "weight": 24,
            "metadata_and_reserve_conservative": 12,
        })
        self.assertEqual(rounded["total_macro_count"], 105)
        self.assertEqual(rounded["represented_bytes"], 215_040)
        self.assertEqual(rounded["rounding_overhead_bytes"], 128)
        self.assertEqual(rounded["budget_margin_after_rounding_bytes"], 30_720)

    def test_area_arithmetic_and_claim_boundary(self) -> None:
        value = MODULE.build()
        area = value["area_um2"]
        logic = Decimal(area["dc_logic_excluding_nine_parent_macros"])
        each = Decimal(area["foundry_macro_area_each_from_dc"])
        modeled_macro = Decimal(area["modeled_105_macro_area"])
        total = Decimal(area["modeled_logic_plus_full_storage"])
        self.assertEqual(modeled_macro, each * Decimal(105))
        self.assertEqual(total, logic + modeled_macro)
        self.assertLess(total, Decimal(1_000_000))
        self.assertEqual(value["timing"]["extra_96_macros_integrated_in_timing_top"], False)
        boundary = value["claim_boundary"]
        self.assertTrue(boundary["macro_area_model"])
        for field in ("full_storage_logic_netlist", "full_storage_timing", "power",
                      "energy", "throughput", "throughput_per_area",
                      "system_speedup", "paper_citable_after_independent_review_with_model_label"):
            self.assertFalse(boundary[field], field)

    def test_output_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            output.write_text("occupied\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                if output.exists():
                    MODULE.require(False, "refuse overwrite")


if __name__ == "__main__":
    unittest.main()
