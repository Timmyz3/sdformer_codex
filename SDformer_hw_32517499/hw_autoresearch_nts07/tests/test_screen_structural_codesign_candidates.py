import unittest

import numpy as np

from scripts.screen_structural_codesign_candidates import (
    TOKENS,
    local_orbit_counts,
    round_div_even,
    score_q7,
    structural_shared_score,
)


class StructuralCodesignCandidateTest(unittest.TestCase):
    def test_round_div_even(self) -> None:
        self.assertEqual(round_div_even(8, 16), 0)
        self.assertEqual(round_div_even(24, 16), 2)
        self.assertEqual(round_div_even(25, 16), 2)

    def test_shared_score_preserves_identical_pair(self) -> None:
        q = 0x13579BDF
        k = 0x2468ACE0
        expected = score_q7(q, k, k)
        self.assertEqual(structural_shared_score(q, k, q, k), expected)

    def test_all_one_local_orbits(self) -> None:
        counts = local_orbit_counts(np.full((1, TOKENS), 0xFFFFFFFF, dtype=np.uint32))
        self.assertEqual(counts["orbit_terms"], 3 * TOKENS * 32)
        self.assertEqual(counts["edge_lane_terms"], 2 * (225 + 420 + 420) * 32)


if __name__ == "__main__":
    unittest.main()
