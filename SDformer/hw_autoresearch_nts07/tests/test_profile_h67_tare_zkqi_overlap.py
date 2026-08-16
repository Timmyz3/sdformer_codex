import unittest

import numpy as np

from scripts.profile_h67_tare_zkqi_overlap import (
    DEFAULT_MANIFEST,
    DEFAULT_STRONG_BASELINE,
    analyze,
    candidate_metrics,
    rne_div16_array,
)


class TestH67TareZkqiOverlap(unittest.TestCase):
    def test_rne_ties_to_even(self):
        values = np.asarray([0, 7, 8, 9, 23, 24, 25, 40, 41], dtype=np.int64)
        self.assertEqual(rne_div16_array(values).tolist(), [0, 0, 0, 1, 1, 2, 2, 2, 3])

    def test_candidate_model(self):
        hist = np.zeros(33, dtype=np.int64)
        hist[0] = 1
        hist[2] = 2
        hist[5] = 1
        result = candidate_metrics(hist, 4, 4)
        self.assertEqual(result["zero_pairs"], 1)
        self.assertEqual(result["sparse_pairs"], 2)
        self.assertEqual(result["dense_fallback_pairs"], 1)
        self.assertEqual(result["candidate_score_lane_work"], 32 + 2 * 34 + 64)
        self.assertAlmostEqual(result["score_throughput_ratio_vs_two_direct32"], 0.8)

    def test_frozen_trace_identity_and_selection(self):
        result = analyze(DEFAULT_MANIFEST, DEFAULT_STRONG_BASELINE)
        self.assertEqual(result["identity_calibration"]["pairs"], 31050)
        self.assertEqual(result["identity_calibration"]["zkqi_active_pairs"], 14554)
        self.assertEqual(result["exactness"]["raw16_mismatches"], 0)
        self.assertEqual(result["exactness"]["q7_mismatches"], 0)
        self.assertEqual(result["selection"]["selected_width"], 16)
        self.assertEqual(result["selection"]["rtl_screen_widths"], [8, 16])
        self.assertFalse(result["selection"]["width_frozen"])
        self.assertFalse(result["identity_calibration"]["pairwise_active_bitmap_compared"])
        tare16 = next(row for row in result["candidates"] if row["residual_width"] == 16)
        self.assertEqual(tare16["dense_fallback_pairs"], 251)
        self.assertGreater(tare16["score_lane_work_reduction"], 0.40)
        self.assertGreater(tare16["lane_only_area_normalized_score_throughput"], 1.30)


if __name__ == "__main__":
    unittest.main()
