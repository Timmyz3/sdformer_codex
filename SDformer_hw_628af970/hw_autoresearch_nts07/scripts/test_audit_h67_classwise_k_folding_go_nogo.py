from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_h67_classwise_k_folding_go_nogo import evaluate_counterexample


class H67ClasswiseKFoldingAuditTest(unittest.TestCase):
    def test_token_axis_is_lost_without_destination_bitmap(self) -> None:
        result = evaluate_counterexample()
        self.assertEqual(result["token_indexed_output"], [8, 12])
        self.assertEqual(result["folded_single_accumulator"], 20)
        self.assertEqual(result["invalid_broadcast_mismatches"], 2)
        self.assertEqual(result["destination_bitmap_mismatches"], 0)
        pair = result["indistinguishable_count_pair"]
        self.assertEqual(pair["shared_class_lane_count"], [1, 1])
        self.assertEqual(pair["token_output_a"], [8, 12])
        self.assertEqual(pair["token_output_b"], [12, 8])


if __name__ == "__main__":
    unittest.main()
