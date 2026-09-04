#!/usr/bin/env python3
"""Regression guards for the M462R2 site-only full-FFN gate correction."""

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (ROOT / "hw_autoresearch_nts07/system_simulator/scripts/"
          "analyze_m462r2_h67_g8_site_gate_postcompute_oracle_cycles.py")


def load_module():
    spec = importlib.util.spec_from_file_location("m462r2_test", str(SCRIPT))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_module()


class M462R2GateTests(unittest.TestCase):

    def test_literal_s10_t10_and_gate_population(self):
        self.assertEqual(M.M.SAMPLES, 10)
        self.assertEqual(M.M.TIMESTEPS, 10)
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('if row["mask_mode"] == "t10_all_spatial_site"', source)
        self.assertIn('if row["mask_mode"] == "strict_token_tnhw"', source)
        self.assertIn('eligible and row["total_accounted_postcompute_oracle_saved_cycles"]',
                      source)

    def test_token_rows_cannot_drive_site_gate(self):
        rows = [
            {"mask_mode": "strict_token_tnhw",
             "total_accounted_postcompute_oracle_saved_cycles": 200_000_000},
            {"mask_mode": "t10_all_spatial_site",
             "total_accounted_postcompute_oracle_saved_cycles": 0},
        ]
        site_rows = [row for row in rows
                     if row["mask_mode"] == "t10_all_spatial_site"]
        self.assertEqual(max(row[
            "total_accounted_postcompute_oracle_saved_cycles"]
            for row in site_rows), 0)
        self.assertLess(0, M.M.GATES[0][1])

    def test_no_executable_or_system_admission(self):
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"executable_skip": False', source)
        self.assertIn('"system_speedup": False', source)
        self.assertIn('"delta_aee": False', source)
        self.assertNotIn('"system_speedup": True', source)


if __name__ == "__main__":
    unittest.main()
