#!/usr/bin/env python3

import unittest

from prove_local5_valid_gate_nonzero import exp_weight, round_to_nearest_even


class Local5GateProofTest(unittest.TestCase):
    def test_exp_weight_bound(self) -> None:
        weights = [exp_weight(delta) for delta in range(513)]
        self.assertEqual(min(weights), 16)
        self.assertEqual(max(weights), 256)

    def test_minimum_gate_is_one_lsb(self) -> None:
        self.assertEqual(round_to_nearest_even(16 << 7, 11), 1)


if __name__ == "__main__":
    unittest.main()
