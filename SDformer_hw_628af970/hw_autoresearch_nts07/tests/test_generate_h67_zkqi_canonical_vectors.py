import random
import unittest

from scripts.generate_h67_checkpoint_row_vectors import score_q7
from scripts.generate_h67_zkqi_canonical_vectors import canonical_pair


class CanonicalH67VectorTest(unittest.TestCase):
    def test_random_realizable_counts_round_trip(self) -> None:
        rng = random.Random(20260809)
        for index in range(1000):
            q0 = rng.getrandbits(32)
            q1 = rng.getrandbits(32)
            k0 = rng.getrandbits(32)
            k1 = rng.getrandbits(32)
            counts = (
                q0.bit_count(), q1.bit_count(), k0.bit_count(), k1.bit_count(),
                (q0 & k0).bit_count(), (q1 & k1).bit_count(),
                (k0 ^ k1).bit_count(),
            )
            cq0, cq1, ck0, ck1 = canonical_pair(*counts, rotate=index)
            self.assertEqual(
                (
                    cq0.bit_count(), cq1.bit_count(), ck0.bit_count(), ck1.bit_count(),
                    (cq0 & ck0).bit_count(), (cq1 & ck1).bit_count(),
                    (ck0 ^ ck1).bit_count(),
                ),
                counts,
            )
            self.assertEqual(score_q7(cq0, ck0, ck1), score_q7(q0, k0, k1))
            self.assertEqual(score_q7(cq1, ck1, ck0), score_q7(q1, k1, k0))

    def test_impossible_k_motion_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "奇偶"):
            canonical_pair(0, 0, 1, 1, 0, 0, 1)
        with self.assertRaisesRegex(ValueError, "集合关系"):
            canonical_pair(0, 0, 31, 31, 0, 0, 30)


if __name__ == "__main__":
    unittest.main()
