from __future__ import annotations

import unittest

from model_gcmp_multicast_dse import row_cycle_estimate


class GcmpMulticastDseTest(unittest.TestCase):
    def test_multicast_can_hide_product_generation(self):
        result = row_cycle_estimate(
            active_lanes=16,
            class_channel_terms=4,
            active_classes=2,
            delivery_transactions=6,
            output_channels=64,
            output_lanes=16,
            product_engines=1,
            class_slots=4,
        )
        self.assertFalse(result["overflow"])
        self.assertEqual(result["direct_cycles"], 64)
        self.assertEqual(result["product_cycles"], 16)
        self.assertEqual(result["delivery_cycles"], 24)
        self.assertEqual(result["candidate_cycles"], 24)

    def test_class_overflow_falls_back_to_direct(self):
        result = row_cycle_estimate(
            active_lanes=16,
            class_channel_terms=4,
            active_classes=5,
            delivery_transactions=6,
            output_channels=64,
            output_lanes=16,
            product_engines=2,
            class_slots=4,
        )
        self.assertTrue(result["overflow"])
        self.assertEqual(result["candidate_cycles"], result["direct_cycles"])

    def test_invalid_hardware_parameters_fail(self):
        with self.assertRaises(ValueError):
            row_cycle_estimate(
                active_lanes=1,
                class_channel_terms=1,
                active_classes=1,
                delivery_transactions=1,
                output_channels=32,
                output_lanes=0,
                product_engines=1,
                class_slots=1,
            )


if __name__ == "__main__":
    unittest.main()
