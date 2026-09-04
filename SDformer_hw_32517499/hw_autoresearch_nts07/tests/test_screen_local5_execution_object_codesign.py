#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "screen_local5_execution_object_codesign",
    ROOT / "scripts/screen_local5_execution_object_codesign.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ScreenLocal5ExecutionObjectCodesignTest(unittest.TestCase):
    def fixture(self) -> dict[str, np.ndarray]:
        return {
            "group_offsets": np.array([0, 3, 4], dtype=np.int64),
            "item_destination": np.array([0, 0, 1, 2], dtype=np.uint16),
            "item_gate_code": np.array([5, 7, 9, 11], dtype=np.uint16),
            "item_lane_id": np.array([0, 0, 1, 2], dtype=np.uint16),
            "item_multiplicity": np.array([1, 2, 1, 1], dtype=np.uint8),
            "descriptor_group_offsets": np.array([0, 2, 3], dtype=np.int64),
            "source_gate_count": np.array([1, 3, 1], dtype=np.uint8),
            "source_k_popcount": np.array([2, 3, 1], dtype=np.uint8),
            "source_term_count": np.array([2, 9, 1], dtype=np.uint16),
            "source_delivery_count": np.array([2, 2, 1], dtype=np.uint16),
        }

    def test_exact_ledgers(self) -> None:
        report = MODULE.analyze_arrays(self.fixture(), [0, 1])
        totals = report["totals"]
        self.assertEqual(totals["source_owned_terms"], 12)
        self.assertEqual(totals["dual_gate_issue_cycles"], 9)
        self.assertEqual(totals["ideal_one_gate_issue_cycles"], 6)
        self.assertEqual(totals["coefficient_nonzero_terms"], 3)
        self.assertEqual(totals["destination_updates"], 5)
        self.assertEqual(totals["maximum_coefficient"], 19)
        self.assertEqual(totals["dual_faster_groups"], 1)
        self.assertEqual(totals["dual_equal_groups"], 1)

    def test_source_term_mismatch_fails_closed(self) -> None:
        fixture = self.fixture()
        fixture["source_term_count"] = np.array([2, 8, 1], dtype=np.uint16)
        with self.assertRaisesRegex(ValueError, "source term formula mismatch"):
            MODULE.analyze_arrays(fixture, [0, 1])


if __name__ == "__main__":
    unittest.main()
