#!/usr/bin/env python3
"""Synthetic pre-payload recovery tests for M463r2."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "system_simulator/scripts/analyze_m463r2_beta16_destination_stationary_dse.py"
SPEC = importlib.util.spec_from_file_location("m463r2_under_test", str(ANALYZER))
M463 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M463)


class M463R2SyntheticTest(unittest.TestCase):
    def test_drop_set_cover_semantics(self):
        weight = np.full((16, 96), 100, dtype=np.int8)
        weight[0, :32] = 0
        weight[1, 32:64] = 0
        weight[2, 64:] = 0
        minimum, reachable = M463.minimum_drop_set_cover_le4(weight, 16)
        self.assertTrue(reachable)
        self.assertEqual(minimum, 3)
        weight[2, 95] = 100
        minimum, reachable = M463.minimum_drop_set_cover_le4(weight, 16)
        self.assertFalse(reachable)
        self.assertIsNone(minimum)

    def test_dense_destination_cost(self):
        correction = np.asarray([0, 1, 3, 0xffff], dtype=np.uint16)
        keep = np.full((8, 96), 0xffff, dtype=np.uint16)
        observed = M463.destination_cost(correction, keep)
        self.assertTrue(np.all(
            observed == M463.POPCOUNT[correction][:, None]))

    def test_pwp_direct_pruned_miter(self):
        originals = np.asarray([0, 3, 5, 15], dtype=np.uint16)
        counts = np.asarray([2, 3, 5, 7], dtype=np.int64)
        centers = np.asarray([0, 3] + [7] * 30, dtype=np.uint16)
        weights = (np.arange(16 * 8 * 96, dtype=np.int16).reshape(
            16, 8, 96) % 31 - 15).astype(np.int8)
        all_keep = np.full((8, 96), 0xffff, dtype=np.uint16)
        no_keep = np.zeros((8, 96), dtype=np.uint16)
        phase = M463.phase_metrics(
            originals, counts, centers, {0: all_keep, 16: no_keep}, weights)
        self.assertEqual(phase["source_rows"], 17)
        self.assertEqual(phase["beta16_correction_work_by_block"], [0] * 8)


if __name__ == "__main__":
    unittest.main()
