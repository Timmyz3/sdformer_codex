#!/usr/bin/env python3
"""ECGB模型的确定性单元测试。"""

import unittest

from scripts.model_exact_cross_context_gate_batching import (
    ecgb_bits,
    lru_misses,
    lru_misses_by_group,
    make_batches,
    pingpong_cycles,
    reorder_batches,
)


def row(seq: int, context: int, lane: int, gate: int) -> dict[str, int]:
    return {
        "seq": seq,
        "plane": 0,
        "y": 0,
        "x": context,
        "lane": lane,
        "gate": gate,
        "mask": 1,
        "window_last": 0,
    }


class TestExactCrossContextGateBatching(unittest.TestCase):
    def test_batching_groups_equal_keys(self) -> None:
        contexts = [
            [row(0, 0, 0, 1), row(1, 0, 0, 2)],
            [row(2, 1, 0, 1), row(3, 1, 0, 2)],
        ]
        original = contexts[0] + contexts[1]
        reordered, capacity, slots = reorder_batches(contexts, 2)
        self.assertEqual(lru_misses(original, 1), 4)
        self.assertEqual(lru_misses(reordered, 1), 2)
        self.assertEqual(capacity, 4)
        self.assertEqual(slots, 2)

    def test_storage_grows_with_output_width(self) -> None:
        out4 = ecgb_bits(batch=4, capacity=128, slots=6, out_dim=4)
        out32 = ecgb_bits(batch=4, capacity=128, slots=6, out_dim=32)
        self.assertGreater(out32["total_bits"], out4["total_bits"])
        self.assertEqual(
            out32["total_bits"] - out4["total_bits"], 28 * 17
        )

    def test_pingpong_has_finite_fill_cost(self) -> None:
        contexts = [
            [row(0, 0, 0, 1), row(1, 0, 0, 2)],
            [row(2, 1, 0, 1), row(3, 1, 0, 2)],
        ]
        groups, _, _ = make_batches(contexts, 1)
        misses = lru_misses_by_group(groups, 1)
        cycles = pingpong_cycles(groups, misses, 1)
        self.assertGreater(cycles, sum(map(len, groups)))


if __name__ == "__main__":
    unittest.main()
