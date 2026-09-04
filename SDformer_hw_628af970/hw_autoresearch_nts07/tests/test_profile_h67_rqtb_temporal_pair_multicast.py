from __future__ import annotations

import unittest

from scripts.generate_h67_checkpoint_row_vectors import score_q7
from scripts.profile_h67_rqtb_temporal_pair_multicast import analyze_row


class ProfileH67RqtbTemporalPairMulticastTest(unittest.TestCase):
    def test_equal_score_pairs_share_overlapped_k_lanes(self) -> None:
        q = 0
        k0 = 0b0011
        k1 = 0b0110
        score0 = score_q7(q, k0, k1)
        score1 = score_q7(q, k1, k0)
        self.assertEqual(score0, score1)
        row = analyze_row([(q, k0, k1, 100), (q, k1, k0, 100)])
        self.assertEqual(row["baseline_commands"], 4)
        self.assertEqual(row["paired_commands"], 3)
        self.assertEqual(row["saved_commands"], 1)

    def test_different_score_pair_is_not_merged(self) -> None:
        row = analyze_row([(0, 0, 0, 100), (1, 1, 0, 101)])
        self.assertEqual(row["equal_pairs"], 0)
        self.assertEqual(row["baseline_commands"], row["paired_commands"])


if __name__ == "__main__":
    unittest.main()
