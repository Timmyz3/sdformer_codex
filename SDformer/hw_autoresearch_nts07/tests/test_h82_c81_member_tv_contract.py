#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from scripts.h82_c81_member_tv_contract import demo_loss_tracks_jaccard, member_tv_loss
from scripts.h82_class_file_reference import TOKENS, q7_codes


class MemberTvContractTests(unittest.TestCase):
    def test_identical_rows_have_zero_tv_and_unit_codes_ok(self) -> None:
        codes = q7_codes(np.linspace(-0.2, 0.2, TOKENS))
        self.assertEqual(member_tv_loss(codes, codes), 0.0)

    def test_full_roster_swap_is_one(self) -> None:
        a = np.zeros(TOKENS, dtype=np.int64)
        b = np.ones(TOKENS, dtype=np.int64)
        # no surviving class → 0 by contract
        self.assertEqual(member_tv_loss(a, b), 0.0)
        b[0] = 0
        self.assertGreater(member_tv_loss(a, b), 0.0)

    def test_demo_churn_increases_tv_and_drops_jaccard(self) -> None:
        demo = demo_loss_tracks_jaccard()
        self.assertLess(demo["stable_tv"], demo["churn_tv"])
        self.assertGreater(demo["stable_jaccard"], demo["churn_jaccard"])


if __name__ == "__main__":
    unittest.main()
