#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from scripts.h82_class_file_reference import TOKENS, window_codes
from scripts.h86_window_member_delta_reference import (
    apply_patch,
    expand_dest_owned,
    member_patch,
    motion_boundary_contract,
)


class H86WindowMemberDeltaTests(unittest.TestCase):
    def test_patch_roundtrip_on_adjacent_windows(self) -> None:
        rng = np.random.default_rng(86)
        field = rng.normal(0.0, 0.4, size=(16, 16, 2)).clip(-2, 2)
        prev = window_codes(field, 0, 0)
        curr = window_codes(field, 0, 1)
        ops = tuple(op for op in member_patch(prev, curr) if op.kind != "stay")
        rebuilt = apply_patch(prev, ops)
        self.assertTrue(np.array_equal(rebuilt, curr))

    def test_dest_owned_expand_differs_for_c7_and_one_vote(self) -> None:
        scores = np.zeros(TOKENS, dtype=np.float64)
        scores[:8] = 0.0
        scores[8] = 1.0
        k = np.ones((TOKENS, 1), dtype=np.float64)
        h82 = expand_dest_owned(scores, k, one_vote=True)
        c7 = expand_dest_owned(scores, k, one_vote=False)
        self.assertGreater(float(np.max(np.abs(h82 - c7))), 1.0e-4)
        self.assertTrue(np.allclose(h82[:8], h82[0]))

    def test_motion_boundary_keeps_dest_identity(self) -> None:
        contract = motion_boundary_contract()
        self.assertTrue(contract["keeps_rqtb_quotient"])
        self.assertIn("dest i", contract["destination_identity"])
        self.assertTrue(contract["not_motion_xor"])
        self.assertTrue(contract["not_local5"])


if __name__ == "__main__":
    unittest.main()
