#!/usr/bin/env python3
from __future__ import annotations

import unittest

import numpy as np

from scripts.generate_h82_class_file_golden_vectors import rows
from scripts.h82_class_file_tlm import run_row, tokens_covered


class ClassFileTlmTests(unittest.TestCase):
    def test_expand_covers_every_token_and_is_class_major(self) -> None:
        report = run_row(rows()["mixed_pair_mask"])
        self.assertFalse(report.protocol_error)
        self.assertEqual(tokens_covered(report), 450)
        self.assertEqual(report.n_shiftmax, report.class_file.n_occupied)
        class_ids = [beat.class_id for beat in report.beats]
        self.assertEqual(class_ids, sorted(class_ids))
        # same class keeps one integer gate
        by_class = {}
        for beat in report.beats:
            by_class.setdefault(beat.class_id, beat.gate_c_q17)
            self.assertEqual(by_class[beat.class_id], beat.gate_c_q17)

    def test_c_max_protocol_error(self) -> None:
        report = run_row(rows()["singletons"], c_max=64)
        self.assertTrue(report.protocol_error)
        self.assertEqual(report.beats, ())

    def test_pair_zero_has_both_times_when_scores_match(self) -> None:
        report = run_row(rows()["mixed_pair_mask"])
        pair0 = [beat for beat in report.beats if beat.pair_id == 0]
        self.assertTrue(any(beat.k_mask & 0b1 for beat in pair0))
        self.assertTrue(any(beat.k_mask & 0b10 for beat in pair0))


if __name__ == "__main__":
    unittest.main()
