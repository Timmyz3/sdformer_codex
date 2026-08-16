import unittest

import numpy as np

from scripts.profile_h67_fcip_real_trace import (
    build_row_relation,
    reconstruct_h67_score_class,
    rne_div16,
)


class H67FcipRealTraceTest(unittest.TestCase):
    def test_rne_div16_ties_to_even(self):
        values = np.asarray([8, 24, 25, 40], dtype=np.int32)
        np.testing.assert_array_equal(
            rne_div16(values),
            np.asarray([0, 2, 2, 2], dtype=np.int32),
        )

    def test_reconstruct_layout_and_motion(self):
        q = np.zeros((2, 1, 1, 2, 4), dtype=bool)
        k = np.zeros_like(q)
        q[0, 0, 0, 0, 0] = True
        k[0, 0, 0, 0, 0] = True
        k[1, 0, 0, 0, 1] = True
        score, row_k = reconstruct_h67_score_class(q, k)
        self.assertEqual(score.shape, (1, 1, 4))
        self.assertEqual(row_k.shape, (1, 1, 4, 4))
        self.assertGreater(int(score[0, 0, 0]), int(score[0, 0, 2]))

    def test_relation_preserves_alias_and_segments(self):
        score = np.asarray([1, 2, 1, 2, 3], dtype=np.int16)
        gate = np.asarray([64, 64, 64, 64, 32], dtype=np.int16)
        k = np.zeros((5, 2), dtype=bool)
        k[0, 0] = True
        k[1, 0] = True
        k[4, 1] = True
        row = build_row_relation(score, k, gate, segment_tokens=2)
        self.assertEqual(row["active_score_classes"], 3)
        self.assertEqual(row["active_final_gates"], 2)
        self.assertEqual(row["max_classes_per_final_gate"], 2)
        self.assertEqual(row["class_lane_terms"], 3)
        self.assertEqual(row["final_gate_lane_terms"], 2)
        self.assertGreaterEqual(
            row["final_gate_lane_segments"],
            row["final_gate_lane_terms"],
        )

    def test_rejects_nonfunctional_class_gate_map(self):
        score = np.asarray([1, 1], dtype=np.int16)
        gate = np.asarray([2, 3], dtype=np.int16)
        k = np.ones((2, 1), dtype=bool)
        with self.assertRaises(ValueError):
            build_row_relation(score, k, gate)


if __name__ == "__main__":
    unittest.main()
