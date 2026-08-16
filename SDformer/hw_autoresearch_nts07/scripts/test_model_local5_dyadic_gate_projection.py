#!/usr/bin/env python3
"""Unit tests for the Local5 exact dyadic gate projection screen."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_local5_dyadic_gate_projection import (  # noqa: E402
    COMMON_GATES,
    classify_gate,
    dyadic_product,
    exhaustive_numeric_check,
    frontier_escape_max,
    lru_stats,
    output_width_activity,
)


class DyadicGateProjectionTest(unittest.TestCase):
    def test_gate_classification(self) -> None:
        self.assertEqual(classify_gate(16), "shift")
        self.assertEqual(classify_gate(32), "shift")
        self.assertEqual(classify_gate(15), "shift_sub")
        self.assertEqual(classify_gate(31), "shift_sub")
        self.assertEqual(classify_gate(29), "escape")
        self.assertEqual(COMMON_GATES, {15, 16, 31, 32})

    def test_lru_stats(self) -> None:
        terms = [(0, 15), (0, 16), (0, 15), (1, 31), (1, 31)]
        self.assertEqual(lru_stats(terms, ways=1), (1, 4))
        self.assertEqual(lru_stats(terms, ways=2), (2, 3))

    def test_dyadic_product_and_width(self) -> None:
        for weight in (-128, -1, 0, 1, 127):
            for gate in (15, 16, 31, 32, 29):
                self.assertEqual(dyadic_product(gate, weight), gate * weight)
        check = exhaustive_numeric_check()
        self.assertEqual(check["vectors"], 131072)
        self.assertEqual(check["mismatches"], 0)
        self.assertGreaterEqual(check["minimum"], -(1 << 16))
        self.assertLess(check["maximum"], 1 << 16)

    def test_frontier_escape_capacity(self) -> None:
        valid = [0] * 450
        gates = [0] * 450
        # One escape in each plane; both must be seen, but never in one
        # plane-local frontier at the same time.
        valid[0] = 1
        gates[0] = 29
        valid[225] = 1
        gates[225] = 14
        self.assertEqual(frontier_escape_max(valid, gates), 1)

    def test_storage_baseline_isolation(self) -> None:
        result = output_width_activity(
            out_dim=2,
            terms=10,
            hits=8,
            misses=2,
            weight_row_reads=4,
            pinned_lane_fills=2,
            term_class={"shift": 6, "shift_sub": 3, "escape": 1},
            raw_relation_storage_bits=1000,
            symbol_relation_storage_bits=600,
        )
        storage = result["system_storage_bits"]
        self.assertEqual(storage["raw_relation_plus_w4_cache"], 6922)
        self.assertEqual(storage["raw_relation_plus_dyadic_arithmetic"], 1000)
        self.assertEqual(storage["symbol_relation_plus_dyadic_arithmetic"], 600)
        self.assertAlmostEqual(storage["symbol_ratio_vs_raw_dyadic"], 0.6)
        self.assertEqual(storage["symbol_relation_plus_pinned_four_product"], 3954)


if __name__ == "__main__":
    unittest.main()
