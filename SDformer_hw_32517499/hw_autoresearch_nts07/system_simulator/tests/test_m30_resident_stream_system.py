#!/usr/bin/env python3

import importlib.util
import math
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_m30_resident_stream_system.py"
SPEC = importlib.util.spec_from_file_location("m30_dse", str(SCRIPT))
M30 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M30)


class M30ResidentStreamSystemTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = M30.build_report(M30.DEFAULT_M26)
        cls.rows = {
            row["name"]: row for row in cls.report["port_candidates"]
        }

    def test_t2_population_and_m26_packing_correction(self):
        correction = self.report["m26_t2_correction"]
        self.assertEqual(correction["t2_dense_products"], 580608000)
        self.assertEqual(correction["t2_vectors"], 145152000)
        self.assertEqual(correction["t2_tiles_at_lanes16"], 9072000)
        self.assertEqual(correction["m26_ideal_packed24_cycles"], 6048000)
        self.assertEqual(correction["minimum_sustained_input_bits_for_packed24"], 384)
        frozen = self.report["frozen_resources"]
        self.assertEqual(frozen["live_parameter_state_bits"], 37449)
        self.assertEqual(frozen["live_parameter_state_bytes"], 4682)
        self.assertEqual(frozen["t10_requant_shift_bits"], 225)

    def test_port_dse_reproduces_corrected_cycle_points(self):
        one256 = self.rows["256b_independent_output_lanes16"]
        self.assertEqual(one256["t2_lanes_per_cycle"], 16)
        self.assertEqual(one256["t2_cycles"], 9072000)
        self.assertEqual(one256["local_cycles"] - one256["parameter_cold_fill_cycles"], 308071124)
        self.assertAlmostEqual(one256["local_speedup_vs_fixed"], 2.0153, places=4)
        self.assertTrue(one256["crosses_2x_local"])

        one128 = self.rows["128b_independent_output"]
        self.assertEqual(one128["t2_cycles"], 18144000)
        self.assertFalse(one128["crosses_2x_local"])
        self.assertAlmostEqual(one128["local_speedup_vs_fixed"], 1.9577, places=4)

        dual256 = self.rows["dual256b_independent_output_packed24"]
        self.assertEqual(dual256["t2_lanes_per_cycle"], 24)
        self.assertEqual(dual256["t2_product_slots_used"], 96)
        self.assertEqual(dual256["t2_cycles"], 6048000)
        self.assertEqual(dual256["local_cycles"] - dual256["parameter_cold_fill_cycles"], 305047124)

    def test_shared_bus_and_threshold_payload_are_not_free(self):
        shared = self.rows["256b_shared_with_bitpack_output"]
        self.assertEqual(shared["t2_lanes_per_cycle"], 14)
        self.assertEqual(shared["t2_product_slots_used"], 56)
        traffic = self.report["threshold_bitplane_forwarding"]
        self.assertAlmostEqual(traffic["output_payload_reduction"], 24.0)
        self.assertAlmostEqual(traffic["boundary_payload_reduction"], 3.5555555556)
        self.assertIn("PENDING", traffic["semantic_admission"])

    def test_claim_boundary_remains_closed(self):
        self.assertFalse(self.report["headline_admitted"])
        self.assertIn("NO_ACCURACY", self.report["status"])
        self.assertTrue(self.report["claim_boundary"]["forbidden"])


if __name__ == "__main__":
    unittest.main()
