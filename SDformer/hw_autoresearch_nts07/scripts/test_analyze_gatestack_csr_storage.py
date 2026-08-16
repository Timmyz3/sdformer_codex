from __future__ import annotations

import unittest

from analyze_gatestack_csr_storage import (
    classify_head_slot,
    physical_storage_by_stage,
)


class GateStackCsrStorageTest(unittest.TestCase):
    def test_sparse_head_uses_csr(self):
        row = classify_head_slot(
            active_lanes=60, class_terms=11, active_classes=2
        )
        self.assertEqual(row["mode"], "TERM_CSR")
        self.assertLess(row["stored_bits"], row["raw_bits"])

    def test_class_overflow_forces_raw(self):
        row = classify_head_slot(
            active_lanes=10, class_terms=10, active_classes=5
        )
        self.assertEqual(row["mode"], "RAW_CLASS_OVERFLOW")
        self.assertEqual(row["stored_bits"], row["raw_bits"])

    def test_capacity_overflow_forces_raw(self):
        row = classify_head_slot(
            active_lanes=1000, class_terms=100, active_classes=4
        )
        self.assertEqual(row["mode"], "RAW_CAPACITY_OVERFLOW")
        self.assertEqual(row["stored_bits"], row["raw_bits"])

    def test_stage3_physical_storage_stays_below_old_bitmap(self):
        stage3 = physical_storage_by_stage()[3]
        self.assertLess(stage3["total_kib"], stage3["bitmap_design_kib"])
        self.assertLess(stage3["total_kib"], 80.0)


if __name__ == "__main__":
    unittest.main()
