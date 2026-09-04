#!/usr/bin/env python3
from __future__ import annotations

from decimal import Decimal
import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_m1607_ep34_c1_parent_partial_energy_model.py"
SPEC = importlib.util.spec_from_file_location("m1607_energy", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


class M1607EnergyTest(unittest.TestCase):
    def test_exact_activation_and_energy_arithmetic(self) -> None:
        value = M.build()
        parent = value["parent_sram"]
        self.assertEqual(parent["read_macro_activations"], 1_044_464_328)
        self.assertEqual(parent["write_macro_activations"], 653_094_432)
        energy = value["energy"]
        self.assertEqual(Decimal(energy["parent_dynamic_mj_per_sample"]),
                         Decimal("1.755375086376432"))
        self.assertEqual(Decimal(energy["full_105macro_capacity_leakage_mj_per_sample"]),
                         Decimal("0.72371030841550350"))
        self.assertEqual(Decimal(energy["known_partial_parent_dynamic_plus_full_capacity_leakage_mj_per_sample"]),
                         Decimal("2.47908539479193550"))

    def test_claim_remains_partial(self) -> None:
        boundary = M.build()["claim_boundary"]
        self.assertTrue(boundary["component_energy_model"])
        for field in ("weight_dynamic", "psum_dynamic", "metadata_dynamic",
                      "logic_dynamic_or_leakage", "dram_energy", "total_c1_energy",
                      "energy_per_full_frame", "system_energy", "measured_power",
                      "paper_citable_after_independent_review"):
            self.assertFalse(boundary[field], field)


if __name__ == "__main__":
    unittest.main()
