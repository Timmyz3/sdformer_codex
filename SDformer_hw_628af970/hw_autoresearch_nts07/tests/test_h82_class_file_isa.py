#!/usr/bin/env python3
"""Class File ISA is a new execution object, not C7 multiplicity Shiftmax."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.h82_class_file_reference import (
    N_BINS,
    PAIRS,
    TOKENS,
    build_class_file,
    c7_multiplicity_weighted_gates,
    class_center,
    integer_c7_gates,
    integer_class_major_gates,
    member_jaccard_surviving,
    q7_codes,
    sliding_window_study,
    storage_object_model,
    token_shiftmax_gates,
)


class H82ClassFileIsaTests(unittest.TestCase):
    def test_q7_grid_is_513_not_h67_162(self) -> None:
        self.assertEqual(N_BINS, 513)
        self.assertEqual(q7_codes(np.asarray([-2.0, 0.0, 2.0])).tolist(), [0, 256, 512])
        self.assertAlmostEqual(class_center(256), 0.0)
        self.assertAlmostEqual(class_center(0), -2.0)

    def test_unequal_multiplicity_is_not_c7_or_token(self) -> None:
        scores = np.zeros(TOKENS, dtype=np.float64)
        scores[0:3] = 0.0
        scores[3] = 1.0
        h82 = build_class_file(scores, preserve_mean=False).gate_tokens()
        c7 = c7_multiplicity_weighted_gates(scores, preserve_mean=False)
        tok = token_shiftmax_gates(scores, preserve_mean=False)
        self.assertGreater(float(np.max(np.abs(h82 - tok))), 1.0e-4)
        self.assertGreater(float(np.max(np.abs(h82 - c7))), 1.0e-4)
        self.assertLess(float(np.max(np.abs(c7 - tok))), 1.0e-6)

    def test_singleton_classes_match_token_shiftmax(self) -> None:
        scores = np.asarray([class_center(index) for index in range(TOKENS)], dtype=np.float64)
        codes = q7_codes(scores)
        self.assertEqual(int(np.unique(codes).size), TOKENS)
        h82 = build_class_file(scores, preserve_mean=False).gate_tokens()
        tok = token_shiftmax_gates(scores, preserve_mean=False)
        self.assertTrue(np.allclose(h82, tok, atol=1.0e-6))

    def test_equal_multiplicity_greater_than_one_is_not_token_equal(self) -> None:
        scores = np.zeros(TOKENS, dtype=np.float64)
        scores[PAIRS:] = 1.0
        h82 = build_class_file(scores, preserve_mean=False).gate_tokens()
        tok = token_shiftmax_gates(scores, preserve_mean=False)
        self.assertGreater(float(np.max(np.abs(h82 - tok))), 1.0e-4)
        ratio = h82[0] / tok[0]
        self.assertAlmostEqual(ratio, h82[PAIRS] / tok[PAIRS], places=6)
        self.assertGreater(ratio, 1.0)

    def test_class_file_covers_pairs_and_masks(self) -> None:
        scores = np.zeros(TOKENS, dtype=np.float64)
        scores[0] = 0.25
        scores[PAIRS] = 0.25
        scores[1] = 0.75
        class_file = build_class_file(scores, preserve_mean=True)
        self.assertEqual(sum(record.multiplicity for record in class_file.records), TOKENS)
        same = next(record for record in class_file.records if record.class_id == int(q7_codes(np.asarray([0.25]))[0]))
        self.assertEqual(same.temporal_mask, 0b11)
        pair0 = next(member for member in same.members if member.pair_id == 0)
        self.assertEqual(pair0.k_mask, 0b11)
        self.assertTrue(np.all(class_file.gate_tokens() > 0.0))

    def test_preserve_mean_keeps_class_broadcast(self) -> None:
        scores = np.zeros(TOKENS, dtype=np.float64)
        scores[0:5] = 0.5
        scores[10] = 1.0
        class_file = build_class_file(scores, preserve_mean=True)
        gates = class_file.gate_tokens()
        self.assertTrue(np.allclose(gates[0:5], gates[0]))
        self.assertGreater(float(abs(gates[10] - gates[0])), 1.0e-8)

    def test_integer_one_vote_differs_from_integer_c7(self) -> None:
        codes = np.full(TOKENS, 256, dtype=np.int64)
        codes[0:8] = 256
        codes[8:10] = 384
        h82 = integer_class_major_gates(codes, preserve_mean=True)
        c7 = integer_c7_gates(codes, preserve_mean=True)
        self.assertTrue(np.any(h82 != c7))
        self.assertTrue(np.all(h82[0:8] == h82[0]))
        self.assertTrue(np.all(h82[8:10] == h82[8]))

    def test_member_jaccard_detects_roster_churn(self) -> None:
        a = np.zeros(TOKENS, dtype=np.int64)
        b = np.zeros(TOKENS, dtype=np.int64)
        a[0:10] = 1
        b[5:15] = 1
        stats = member_jaccard_surviving(a, b)
        self.assertEqual(stats["n_surviving"], 2)
        self.assertLess(stats["member_jaccard_surviving"], 0.8)
        self.assertGreater(stats["class_set_jaccard"], 0.9)

    def test_storage_model_saves_score_ram_not_membership(self) -> None:
        model = storage_object_model(n_occupied=10, n_members=450)
        self.assertGreater(model["old_h67_h81_bits"]["active_scored_list"], 0)
        self.assertGreater(model["h82_class_file_bits"]["saved_vs_old_active"], 0)
        self.assertLess(
            model["h82_class_file_bits"]["occupied_records"],
            model["h82_class_file_bits"]["member_csr_scoreless"],
        )

    def test_spatial_tv_proxy_does_not_invent_frozen_jaccard(self) -> None:
        study = sliding_window_study(seed=82, height=18, width=18)
        raw = next(item for item in study["fields"] if item["field"] == "raw")
        tv = next(item for item in study["fields"] if item["field"] == "spatial_tv")
        self.assertGreater(tv["mean_member_jaccard_east"], raw["mean_member_jaccard_east"])
        self.assertLess(tv["mean_occupied"], raw["mean_occupied"])


if __name__ == "__main__":
    unittest.main()
