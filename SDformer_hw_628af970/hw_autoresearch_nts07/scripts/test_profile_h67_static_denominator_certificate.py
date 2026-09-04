#!/usr/bin/env python3

import unittest

from scripts.profile_h67_static_denominator_certificate import (
    max_score_from_qkm,
    max_score_from_qcount,
    rne_fraction16,
)


class H67StaticDenominatorCertificateTest(unittest.TestCase):
    def test_boundary(self) -> None:
        self.assertEqual(max_score_from_qcount(15), 93)
        self.assertEqual(max_score_from_qcount(16), 97)
        self.assertGreater(256 + 449 * 152, 1 << 16)
        self.assertLess(256 + 449 * 145, 1 << 16)

    def test_score_bound_exhaustive_aggregate_counts(self) -> None:
        for q_count in range(33):
            bound = max_score_from_qcount(q_count)
            observed = 0
            for overlap in range(q_count + 1):
                for same_zero in range(33 - q_count):
                    # The temporal peer can complement current K on every lane.
                    observed = max(
                        observed,
                        rne_fraction16(4 * overlap + 32, same_zero),
                    )
            self.assertEqual(observed, bound)

    def test_invalid_q_count_fails(self) -> None:
        with self.assertRaises(ValueError):
            max_score_from_qcount(33)

    def test_qkm_bound_exhaustive_aggregate_counts(self) -> None:
        for q_count in range(33):
            for k_count in range(33):
                overlap_min = max(0, q_count + k_count - 32)
                overlap_max = min(q_count, k_count)
                for motion_count in range(33):
                    bound = max_score_from_qkm(
                        q_count, k_count, motion_count
                    )
                    observed = max(
                        rne_fraction16(
                            4 * overlap + motion_count,
                            32 - q_count - k_count + overlap,
                        )
                        for overlap in range(overlap_min, overlap_max + 1)
                    )
                    self.assertEqual(observed, bound)

    def test_invalid_qkm_count_fails(self) -> None:
        with self.assertRaises(ValueError):
            max_score_from_qkm(0, 0, 33)


if __name__ == "__main__":
    unittest.main()
