from __future__ import annotations

import unittest

from model_gatestack_full_projection import (
    head_backend_cycles,
    overlap_two_contexts,
)


class GateStackFullProjectionModelTest(unittest.TestCase):
    def test_obi_skips_empty_directory_cells(self):
        row = head_backend_cycles(
            active_lanes=40,
            class_terms=6,
            active_classes=2,
            delivery_transactions=9,
            class_slots=4,
            head_dim=32,
            product_engines=1,
            multicast_width=4,
            obi_issue_width=1,
            delivery_efficiency=1.0,
            pipeline_fill=4,
        )
        self.assertFalse(row["overflow"])
        self.assertEqual(row["fixed_scan"], 132)
        self.assertEqual(row["obi_replay"], 13)
        self.assertLess(row["obi_replay"], row["direct"])

    def test_overflow_is_exact_direct_cost(self):
        row = head_backend_cycles(
            active_lanes=31,
            class_terms=9,
            active_classes=5,
            delivery_transactions=11,
            class_slots=4,
            head_dim=32,
            product_engines=1,
            multicast_width=4,
            obi_issue_width=1,
            delivery_efficiency=0.85,
            pipeline_fill=4,
        )
        self.assertTrue(row["overflow"])
        self.assertEqual(row["obi_replay"], row["direct"])
        self.assertEqual(row["fixed_scan"], row["direct"])

    def test_two_context_sequence_keeps_fill_and_drain(self):
        self.assertEqual(overlap_two_contexts([10], [30]), 40)
        self.assertEqual(overlap_two_contexts([10, 10, 10], [30, 30, 30]), 100)
        self.assertEqual(overlap_two_contexts([30, 30], [10, 10]), 70)

    def test_invalid_efficiency_fails(self):
        with self.assertRaises(ValueError):
            head_backend_cycles(
                active_lanes=1,
                class_terms=1,
                active_classes=1,
                delivery_transactions=1,
                class_slots=4,
                head_dim=32,
                product_engines=1,
                multicast_width=4,
                obi_issue_width=1,
                delivery_efficiency=0.0,
                pipeline_fill=0,
            )


if __name__ == "__main__":
    unittest.main()
