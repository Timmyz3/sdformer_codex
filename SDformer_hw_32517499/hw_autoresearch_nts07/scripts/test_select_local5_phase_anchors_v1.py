#!/usr/bin/env python3
"""Local5 phase anchor 选择规则单元测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_local5_phase_anchors_v1 as selector  # noqa: E402


def make_rows() -> tuple[list[dict[str, object]], list[str]]:
    rows: list[dict[str, object]] = []
    sequence_keys: list[str] = []
    heads_by_stage = [3, 6, 12, 24]
    for sample in range(18):
        sequence_keys.append(f"sequence_{sample:02d}")
        for stage, heads in enumerate(heads_by_stage):
            value = 1000 + sample * 10 + stage
            rows.append(
                {
                    "sample": sample,
                    "stage": stage,
                    "block": 0,
                    "window": sample + stage,
                    "heads": heads,
                    "tokens": 450,
                    "term_items": value * heads,
                    "term_items_per_head": float(value),
                    "active_source_ratio": 0.5,
                    "source_terms": value,
                    "source_deliveries": value * 2,
                    "service_cycles": (value + stage) * heads,
                    "service_cycles_per_head": float(value + stage),
                }
            )
    return rows, sequence_keys


class AnchorSelectionTest(unittest.TestCase):
    def test_covers_clusters_and_heads(self) -> None:
        rows, sequences = make_rows()
        anchors = selector.build_anchor_plan(rows, sequences)
        reasons = [reason for row in anchors for reason in row["reasons"]]
        self.assertEqual(
            {row["sequence_key"] for row in anchors}, set(sequences)
        )
        self.assertEqual({row["heads"] for row in anchors}, {3, 6, 12, 24})
        for heads in (3, 6, 12, 24):
            self.assertTrue(any(reason == f"H{heads}_NONZERO_RANDOM_BACKPRESSURE_ANCHOR" for reason in reasons))
            self.assertTrue(any(reason == f"H{heads}_term_items_per_head_MIN" for reason in reasons))
            self.assertTrue(any(reason == f"H{heads}_term_items_per_head_MAX" for reason in reasons))

    def test_is_deterministic(self) -> None:
        rows, sequences = make_rows()
        first = selector.build_anchor_plan(rows, sequences)
        rows, sequences = make_rows()
        second = selector.build_anchor_plan(list(reversed(rows)), sequences)
        self.assertEqual(first, second)

    def test_rejects_missing_sequence_cluster(self) -> None:
        rows, sequences = make_rows()
        sequences[-1] = sequences[-2]
        with self.assertRaisesRegex(ValueError, "18"):
            selector.build_anchor_plan(rows, sequences)

    def test_rejects_non_contiguous_samples(self) -> None:
        rows, sequences = make_rows()
        rows = [row for row in rows if row["sample"] != 7]
        with self.assertRaisesRegex(ValueError, "sample"):
            selector.build_anchor_plan(rows, sequences)

    def test_backpressure_anchor_is_nonzero(self) -> None:
        rows, sequences = make_rows()
        for row in rows:
            if row["heads"] == 6:
                row["term_items_per_head"] = 0.0
                row["service_cycles_per_head"] = 0.0
                row["active_source_ratio"] = 0.0
        active = next(row for row in rows if row["heads"] == 6)
        active["term_items_per_head"] = 5.0
        active["service_cycles_per_head"] = 7.0
        active["active_source_ratio"] = 0.1
        anchors = selector.build_anchor_plan(rows, sequences)
        chosen = [
            row
            for row in anchors
            if "H6_NONZERO_RANDOM_BACKPRESSURE_ANCHOR" in row["reasons"]
        ]
        self.assertEqual(len(chosen), 1)
        self.assertGreater(chosen[0]["active_source_ratio"], 0)
        self.assertGreater(chosen[0]["service_cycles_per_head"], 0)

    def test_aggregate_rejects_silent_descriptor_truncation(self) -> None:
        groups = [
            {
                "tag": 0,
                "sample": 0,
                "stage": 0,
                "block": 0,
                "window": 0,
                "head": 0,
                "heads": 3,
                "tokens": 1,
            },
            {
                "tag": 1,
                "sample": 0,
                "stage": 0,
                "block": 0,
                "window": 0,
                "head": 1,
                "heads": 3,
                "tokens": 1,
            },
            {
                "tag": 2,
                "sample": 0,
                "stage": 0,
                "block": 0,
                "window": 0,
                "head": 2,
                "heads": 3,
                "tokens": 1,
            },
        ]
        arrays = {
            "group_offsets": np.array([0, 1, 2, 3], dtype=np.int64),
            "descriptor_group_offsets": np.array([0, 1, 2, 3], dtype=np.int64),
            "item_mode_multiset": np.zeros(3, dtype=np.uint8),
            "source_term_count": np.zeros(2, dtype=np.uint16),
            "source_delivery_count": np.zeros(3, dtype=np.uint16),
            "source_service_cycles_pipelined": np.zeros(3, dtype=np.uint16),
        }

        class FakeArchive(dict):
            @property
            def files(self) -> list[str]:
                return list(self)

        with self.assertRaisesRegex(ValueError, "source_term_count"):
            selector.aggregate_windows(groups, FakeArchive(arrays))

    def test_aggregate_rejects_dtype_drift(self) -> None:
        arrays = {
            "group_offsets": np.array([0], dtype=np.int32),
            "descriptor_group_offsets": np.array([0], dtype=np.int64),
            "item_mode_multiset": np.zeros(0, dtype=np.uint8),
            "source_term_count": np.zeros(0, dtype=np.uint16),
            "source_delivery_count": np.zeros(0, dtype=np.uint16),
            "source_service_cycles_pipelined": np.zeros(0, dtype=np.uint16),
        }

        class FakeArchive(dict):
            @property
            def files(self) -> list[str]:
                return list(self)

        with self.assertRaisesRegex(ValueError, "dtype"):
            selector.aggregate_windows([], FakeArchive(arrays))

    def test_aggregate_rejects_item_terminal_mismatch(self) -> None:
        arrays = {
            "group_offsets": np.array([0, 2], dtype=np.int64),
            "descriptor_group_offsets": np.array([0, 1], dtype=np.int64),
            "item_mode_multiset": np.zeros(1, dtype=np.uint8),
            "source_term_count": np.zeros(1, dtype=np.uint16),
            "source_delivery_count": np.zeros(1, dtype=np.uint16),
            "source_service_cycles_pipelined": np.zeros(1, dtype=np.uint16),
        }

        class FakeArchive(dict):
            @property
            def files(self) -> list[str]:
                return list(self)

        group = {
            "tag": 0,
            "sample": 0,
            "stage": 0,
            "block": 0,
            "window": 0,
            "head": 0,
            "heads": 3,
            "tokens": 1,
        }
        with self.assertRaisesRegex(ValueError, "item"):
            selector.aggregate_windows([group], FakeArchive(arrays))

    def test_aggregate_rejects_non_1d_array(self) -> None:
        arrays = {
            "group_offsets": np.array([[0]], dtype=np.int64),
            "descriptor_group_offsets": np.array([0], dtype=np.int64),
            "item_mode_multiset": np.zeros(0, dtype=np.uint8),
            "source_term_count": np.zeros(0, dtype=np.uint16),
            "source_delivery_count": np.zeros(0, dtype=np.uint16),
            "source_service_cycles_pipelined": np.zeros(0, dtype=np.uint16),
        }

        class FakeArchive(dict):
            @property
            def files(self) -> list[str]:
                return list(self)

        with self.assertRaisesRegex(ValueError, "一维"):
            selector.aggregate_windows([], FakeArchive(arrays))

    def test_rejects_no_nonzero_backpressure_candidate(self) -> None:
        rows, sequences = make_rows()
        for row in rows:
            if row["heads"] == 24:
                row["term_items_per_head"] = 0.0
                row["service_cycles_per_head"] = 0.0
                row["active_source_ratio"] = 0.0
        with self.assertRaisesRegex(ValueError, "H24"):
            selector.build_anchor_plan(rows, sequences)


if __name__ == "__main__":
    unittest.main(verbosity=2)
