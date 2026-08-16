from __future__ import annotations

import unittest

from analyze_temporal_signature_reuse import recover_row


class TemporalSignatureReuseAnalyzerTest(unittest.TestCase):
    def test_intersection_recovery(self):
        row = {
            "k_count_histogram": [0, 1, 1],
            "motion_histogram": [0, 1, 0],
        }
        result = recover_row(row)
        self.assertEqual(result["baseline_weight_row_reads"], 3)
        self.assertEqual(result["intersection_reused_reads"], 1)
        self.assertEqual(result["union_weight_row_reads"], 2)

    def test_odd_difference_fails(self):
        with self.assertRaises(ValueError):
            recover_row({"k_count_histogram": [0, 1], "motion_histogram": [0, 0]})

    def test_old_schema_density_recovery(self):
        result = recover_row({
            "k_active_density": 3 / 8,
            "batch_windows": 1,
            "num_heads": 1,
            "tokens": 2,
            "head_dim": 4,
            "k_temporal_toggle_elements": 1,
        })
        self.assertEqual(result["baseline_weight_row_reads"], 3)
        self.assertEqual(result["intersection_reused_reads"], 1)
        self.assertEqual(result["recovery_method"], "density_shape_plus_exact_xor")


if __name__ == "__main__":
    unittest.main()
