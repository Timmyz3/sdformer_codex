#!/usr/bin/env python3

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_local5_ecgb_population import lru_misses, make_batches


class EcgbPopulationTest(unittest.TestCase):
    def test_reorder_coalesces_exact_lane_gate_keys(self) -> None:
        contexts = [
            [(0, 10), (1, 20)],
            [(0, 11), (0, 10), (1, 20)],
        ]
        batch = make_batches(contexts, 2)[0]
        self.assertEqual(batch["terms"], 5)
        self.assertEqual(batch["slots"], 2)
        self.assertEqual(batch["reordered_w1_misses"], 3)
        self.assertEqual(lru_misses([term for c in contexts for term in c], 4), 3)

    def test_batches_preserve_term_count(self) -> None:
        contexts = [[(0, 1)], [(1, 2), (1, 3)], [(2, 4)]]
        batches = make_batches(contexts, 2)
        self.assertEqual(sum(int(row["terms"]) for row in batches), 4)
        self.assertEqual(len(batches), 2)


if __name__ == "__main__":
    unittest.main()
