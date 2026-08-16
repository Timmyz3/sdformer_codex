#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).with_name(
    "profile_local5_source_projection_hoisting_v2.py"
)
SPEC = importlib.util.spec_from_file_location("local5_hoist_v1", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Local5HoistingProfileTest(unittest.TestCase):
    def test_extract_top_level_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.json"
            path.write_text(
                json.dumps({"records": [{"summary": 1}], "summary": {"x": 7}}),
                encoding="utf-8",
            )
            # Pretty-print is part of the producer contract used by the extractor.
            path.write_text(
                json.dumps(
                    {"records": [{"summary": 1}], "summary": {"x": 7}},
                    indent=2,
                ),
                encoding="utf-8",
            )
            self.assertEqual(MODULE.extract_top_level_object(path, "summary"), {"x": 7})

    def test_descriptor_counts_preserve_unique_gate_multicast(self) -> None:
        k_bitmap = np.asarray([0b11, 0b11111, 0], dtype=np.uint64)
        kpop = np.asarray([2, 5, 0], dtype=np.uint8)
        gates = np.asarray(
            [[4, 4, 8, 0, 0], [2, 3, 2, 3, 4], [7, 7, 7, 0, 0]],
            dtype=np.uint16,
        )
        masks = np.asarray([0b00111, 0b11111, 0b00111], dtype=np.uint8)
        gate_count = np.asarray([2, 3, 0], dtype=np.uint8)
        terms = np.asarray([4, 15, 0], dtype=np.uint16)
        delivery = np.asarray([6, 25, 0], dtype=np.uint16)
        result = MODULE.analyze_descriptors(
            k_bitmap, kpop, gates, masks, gate_count, terms, delivery
        )
        self.assertEqual(result["active_sources"], 2)
        self.assertEqual(result["source_quotient_product_rows"], 19)
        # 两个destination使用gate=4时只做一次wide scale，而不是按degree重复。
        self.assertEqual(result["project_first_wide_gate_scales"], 5)
        self.assertEqual(result["project_first_total_vector_ops"], 10)
        self.assertEqual(result["project_first_favorable_sources"], 2)

    def test_row_mode_compares_project_first_with_dqfs(self) -> None:
        k_bitmap = np.asarray([0b11] + [0] * 449, dtype=np.uint64)
        gates = np.zeros((450, 5), dtype=np.uint16)
        gates[0] = [4, 4, 8, 0, 0]
        masks = np.zeros(450, dtype=np.uint8)
        masks[0] = 0b00111
        gate_count = np.zeros(450, dtype=np.uint8)
        gate_count[0] = 2
        result = MODULE.analyze_row_mode_oracle(
            k_bitmap,
            gates,
            masks,
            np.asarray([0, 450], dtype=np.int64),
            np.asarray([0] * 225 + [1] * 225, dtype=np.uint8),
            np.asarray([index // 15 for index in range(225)] * 2, dtype=np.uint16),
            gate_count,
        )
        self.assertEqual(result["row_segments"], 30)
        self.assertEqual(result["dqfs_weight_reads_and_narrow_products"], 4)
        self.assertEqual(result["project_first_weight_reads"], 2)
        self.assertEqual(result["project_first_13b_vector_adds"], 1)
        self.assertEqual(result["project_first_13x9_wide_products"], 2)
        self.assertEqual(
            result["weight_only_free_compute_oracle"]["selected_weight_reads"],
            2,
        )

    def test_rejects_term_count_tamper(self) -> None:
        with self.assertRaises(ValueError):
            MODULE.analyze_descriptors(
                np.asarray([1], dtype=np.uint64),
                np.asarray([1], dtype=np.uint8),
                np.asarray([[2, 0, 0, 0, 0]], dtype=np.uint16),
                np.asarray([1], dtype=np.uint8),
                np.asarray([1], dtype=np.uint8),
                np.asarray([2], dtype=np.uint16),
                np.asarray([1], dtype=np.uint16),
            )

    def test_full_profile_lower_bound(self) -> None:
        result = MODULE.full_profile_bounds(
            {
                "source_resident_active_k_lanes": 100,
                "source_active_instances": 20,
                "source_gate_lane_terms": 120,
                "dqfs_row_value_product_computes": 80,
                "naive_active_edge_products": 300,
                "zero_gate_entries": 0,
                "source_gate_cardinality_histogram": [0, 10, 10, 0, 0, 0],
            }
        )
        self.assertEqual(result["project_first_unique_gate_scale_vectors"], 30)
        self.assertEqual(result["project_first_total_vector_ops"], 110)
        self.assertGreater(result["project_first_op_change_vs_dqfs"], 0.0)


if __name__ == "__main__":
    unittest.main()
