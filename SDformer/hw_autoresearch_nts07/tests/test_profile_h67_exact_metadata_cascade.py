import tempfile
import unittest
from pathlib import Path

from scripts.profile_h67_exact_metadata_cascade import (
    classify_pair,
    exhaustive_zero_k_closure,
    full_profile_cost_model,
    percentile,
    score_q7,
    simulate_decoupled_bundle_queue,
)


class ExactMetadataCascadeProfileTest(unittest.TestCase):
    def test_percentile_linear_interpolation(self):
        self.assertEqual(percentile([0, 10], 0.5), 5.0)
        self.assertEqual(percentile([0, 10, 20], 0.95), 19.0)

    def test_exact_pair_classes(self):
        self.assertEqual(classify_pair(0, 0, 0, 0), "empty")
        self.assertEqual(classify_pair(1, 0, 0, 0), "kzero_nonempty")
        self.assertEqual(classify_pair(1, 2, 3, 3), "motionzero_nonkzero")
        self.assertEqual(classify_pair(1, 2, 3, 4), "full")
        self.assertEqual(score_q7(0, 0, 0), 2)
        closure = exhaustive_zero_k_closure()
        self.assertEqual(closure["classes"], [0, 1, 2])
        self.assertEqual(
            closure["ranges"],
            [
                {"qcount_min": 0, "qcount_max": 8, "score_q7": 2},
                {"qcount_min": 9, "qcount_max": 23, "score_q7": 1},
                {"qcount_min": 24, "qcount_max": 32, "score_q7": 0},
            ],
        )

    def test_cost_model_is_disjoint_and_conservative(self):
        profile = {
            "pair_empty": 0.60,
            "both_kzero": 0.75,
            "no_k_motion": 0.80,
            "per_token_kzero": 0.85,
        }
        model = full_profile_cost_model(profile, bundle_size=8)
        self.assertAlmostEqual(
            sum(model["disjoint_categories"].values()), 1.0, places=9
        )
        score = model["score_boolean_lane_work"]
        self.assertGreater(
            score["conservative_qcount_ratio"], score["metadata_assisted_ratio"]
        )
        payload = model["payload_model"]["header_sensitivity"]
        self.assertGreater(payload["64"]["bits_per_pair"], payload["32"]["bits_per_pair"])

    def test_decoupled_queue_consumes_every_active_pair(self):
        result = simulate_decoupled_bundle_queue([0, 3, 0, 2])
        self.assertEqual(result["cycles"], 6)
        self.assertEqual(result["max_bundle_descriptors"], 2)
        self.assertEqual(result["max_active_pair_backlog"], 3)


if __name__ == "__main__":
    unittest.main()
