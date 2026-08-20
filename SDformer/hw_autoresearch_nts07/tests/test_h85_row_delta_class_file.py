#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from scripts.h85_row_delta_class_file_reference import (
    SPATIAL,
    TOKENS,
    build_row_files,
    compare_operators,
    expand_without_token_gate,
    h85_row_gates,
    row_deltas,
    storage_bits,
)


class H85RowDeltaTests(unittest.TestCase):
    def test_h85_is_not_h82_window_operator(self) -> None:
        scores = np.zeros(TOKENS, dtype=np.float64)
        scores[0:20] = 0.0
        scores[20:40] = 1.0
        scores[225:245] = 0.25
        cmp = compare_operators(scores)
        self.assertGreater(cmp["maxabs_h82_vs_h85"], 1.0e-4)

    def test_expand_covers_every_token_without_450_gate(self) -> None:
        rng = np.random.default_rng(3)
        scores = rng.normal(0, 0.4, size=TOKENS).clip(-2, 2)
        k = rng.normal(size=(TOKENS, 3))
        files = build_row_files(scores)
        attn, beats = expand_without_token_gate(scores, k, files)
        self.assertEqual(attn.shape, (TOKENS, 3))
        self.assertEqual(len(beats), TOKENS)
        gates = h85_row_gates(files)
        expected = k * gates[:, None]
        self.assertTrue(np.allclose(attn, expected, atol=1.0e-9))

    def test_class_set_reuse_does_not_imply_member_stability(self) -> None:
        scores = np.zeros(TOKENS, dtype=np.float64)
        # two rows, same two classes, swapped columns
        scores[0:8] = 0.0
        scores[8:15] = 1.0
        scores[15:22] = 1.0
        scores[22:30] = 0.0
        files = build_row_files(scores)
        deltas = [d for d in row_deltas(files) if d.time_idx == 0 and d.curr_row == 1]
        self.assertEqual(len(deltas), 1)
        self.assertGreaterEqual(deltas[0].class_set_jaccard, 0.99)
        self.assertLess(deltas[0].member_jaccard_surviving, 0.5)

    def test_member_delta_is_the_directory_cost(self) -> None:
        rng = np.random.default_rng(7)
        scores = rng.normal(0, 0.5, size=TOKENS).clip(-2, 2)
        files = build_row_files(scores)
        bits = storage_bits(files, row_deltas(files))
        self.assertGreater(
            bits["class_and_member_delta_bits"], bits["class_set_delta_bits"]
        )
        self.assertGreater(bits["full_row_files_bits"], 0)
        self.assertEqual(SPATIAL, 15)


if __name__ == "__main__":
    unittest.main()
