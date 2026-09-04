from __future__ import annotations

import unittest

from model_gate_window_group_dse import state_cost_bits


class GateWindowGroupDseTest(unittest.TestCase):
    def test_state_cost_scales_with_window_group(self):
        g1 = state_cost_bits(group_windows=1, class_slots=4, output_lanes=16)
        g4 = state_cost_bits(group_windows=4, class_slots=4, output_lanes=16)
        self.assertEqual(g4["destination_bitmap_bits"], 4 * g1["destination_bitmap_bits"])
        self.assertEqual(g4["accumulator_tile_bits"], 4 * g1["accumulator_tile_bits"])
        self.assertEqual(g4["group_state_bits"], 4 * g1["group_state_bits"])

    def test_class_slots_only_scale_destination_bitmap(self):
        s2 = state_cost_bits(group_windows=2, class_slots=2, output_lanes=8)
        s8 = state_cost_bits(group_windows=2, class_slots=8, output_lanes=8)
        self.assertEqual(s8["destination_bitmap_bits"], 4 * s2["destination_bitmap_bits"])
        self.assertEqual(s8["accumulator_tile_bits"], s2["accumulator_tile_bits"])


if __name__ == "__main__":
    unittest.main()
