#!/usr/bin/env python3
"""Unit tests for Local5 source-owned/cache strong-baseline profiling."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/profile_local5_source_owned_cache_baselines.py"
SPEC = importlib.util.spec_from_file_location("source_cache_profile", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SourceOwnedCacheBaselineTest(unittest.TestCase):
    def test_unique_gate_order_and_validity(self) -> None:
        self.assertEqual(
            MODULE.ordered_unique_nonzero_gates([7, 0, 7, 9, 11], 0b01111),
            [7, 9],
        )

    def test_lru_reuses_across_descriptors(self) -> None:
        rows = [(0, 7), (0, 9), (0, 7), (0, 11), (0, 9)]
        self.assertEqual(MODULE.lru_product_starts(rows, 2), 4)
        self.assertEqual(MODULE.lru_product_starts(rows, 3), 3)

    def test_group_attains_descriptor_local_issue_minimum(self) -> None:
        report = MODULE.analyze_group(
            group_index=0,
            item_offsets=np.array([0, 5]),
            item_lanes=np.array([0, 0, 0, 1, 1]),
            item_gates=np.array([7, 7, 9, 7, 9]),
            item_multiplicity=np.ones(5, dtype=np.uint8),
            descriptor_offsets=np.array([0, 1]),
            descriptor_gates=np.array([[7, 7, 9, 0, 0]]),
            descriptor_valid_masks=np.array([0b00111]),
            descriptor_k_bitmaps=np.array([0b11], dtype=np.uint64),
            ways=(2,),
        )
        self.assertEqual(report["raw_relation_lane_issues"], 5)
        self.assertEqual(report["destination_mfep_issues"], 5)
        self.assertEqual(report["source_owned_issues"], 4)
        self.assertEqual(report["descriptor_local_issue_lower_bound"], 4)
        self.assertEqual(report["epoch_unique_product_keys"], 4)

    def test_rejects_nonpositive_cache(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            MODULE.lru_product_starts([], 0)


if __name__ == "__main__":
    unittest.main()
