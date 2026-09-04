import unittest

from scripts.model_local5_cross_head_spatial_stripe import build_model


class SpatialStripeModelTest(unittest.TestCase):
    def setUp(self):
        self.source = {
            "physical_width": {"accumulator_payload_bits": 460800},
            "population": {"groups": 100},
            "correctness": {"acc32_mismatch": 0},
            "cycles": {"rolling_qsilent": 155791},
        }

    def test_r1_max_head_fails_pre_registered_gates(self):
        report = build_model(self.source)
        summary = report["max_head_r1_summary"]
        self.assertEqual(report["status"], "NO_GO_NO_RTL")
        self.assertAlmostEqual(summary["acc_payload_reduction"], 15.0)
        self.assertLess(summary["resident_all_head_state_reduction"], 2.0)
        self.assertEqual(
            summary["resident_memory_bit_activity_reduction_upper_bound"], 0.0
        )
        self.assertEqual(summary["reload_external_weight_read_ratio"], 15)
        self.assertGreater(summary["reload_cycle_slowdown_lower_bound"], 1.05)

    def test_model_fails_closed_on_wrong_source_identity(self):
        bad = dict(self.source)
        bad["physical_width"] = {"accumulator_payload_bits": 1}
        with self.assertRaises(ValueError):
            build_model(bad)


if __name__ == "__main__":
    unittest.main()
