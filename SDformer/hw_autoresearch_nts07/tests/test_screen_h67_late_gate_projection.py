#!/usr/bin/env python3

import unittest

from scripts.screen_h67_late_gate_projection import (
    project_direct,
    project_late_gate,
    row_counts,
    traffic_model,
)


class LateGateProjectionTest(unittest.TestCase):
    def test_exact_and_counts(self) -> None:
        vectors = [
            {"k": 0b0101, "gate": 7},
            {"k": 0b0011, "gate": 7},
            {"k": 0, "gate": 11},
            {"k": 0b1000, "gate": 3},
        ]
        weights = [2, -3, 5, -7] + [0] * 28
        self.assertEqual(project_direct(vectors, weights), 21)
        self.assertEqual(project_late_gate(vectors, weights), 21)
        self.assertEqual(
            row_counts(vectors),
            {
                "active_tokens": 3,
                "active_lane_events": 5,
                "final_gate_lane_terms": 4,
            },
        )

    def test_traffic_model(self) -> None:
        model = traffic_model(
            active_tokens=3,
            active_lane_events=5,
            gate_lane_terms=2,
            out_dim=2,
        )
        self.assertEqual(model["direct"]["multiply_starts"], 10)
        self.assertEqual(model["late_gate"]["multiply_starts"], 6)
        self.assertEqual(
            model["row_resident_gate_weight"]["multiply_starts"], 4
        )
        self.assertAlmostEqual(
            model["ratios"]["late_multiply_reduction_vs_direct"], 0.4
        )


if __name__ == "__main__":
    unittest.main()
