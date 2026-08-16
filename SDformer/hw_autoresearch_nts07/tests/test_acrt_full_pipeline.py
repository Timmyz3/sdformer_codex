import unittest

from scripts.model_acrt_full_pipeline import (
    acrt_class_cycles,
    aenr_cycles,
    allclass_replay_cycles,
    current_scs_g1_cycles,
    segment_major_intersection,
)


def make_row() -> dict:
    return {
        "tokens": 8,
        "lanes": 4,
        "segments": 2,
        "active_tokens": 3,
        "active_lane_events": 4,
        "active_nonzero_gate_lane_events": 4,
        "active_gatezero_tokens": 0,
        "all_score_classes": 3,
        "active_score_classes": 2,
        "kzero_score_classes": 1,
        "final_gate_lane_terms": 3,
        "class_words": {
            "1": [0b0011, 0],
            "2": [0, 0b0011],
        },
        "k_words": [
            [0b0001, 0],
            [0b0010, 0],
            [0, 0b0001],
            [0, 0],
        ],
        "final_gate_groups": {"64": [1, 2]},
    }


class AcrtFullPipelineTest(unittest.TestCase):
    def test_segment_major_emits_exact_terms(self):
        result = segment_major_intersection(
            make_row(),
            lane_width=4,
            read_latency=1,
            ready_percent=100,
        )
        self.assertEqual(result["emitted_terms"], 3)
        self.assertEqual(result["gate_reads"], 2)
        self.assertEqual(result["lane_groups"], 1)

    def test_current_contract_matches_fsm_shape(self):
        result = current_scs_g1_cycles(
            make_row(),
            ready_percent=100,
        )
        self.assertEqual(result["sum_active"], 3)
        self.assertEqual(result["sum_fold"], 2)
        self.assertEqual(result["emit_active_to_g1"], 3)
        self.assertEqual(result["cycles"], 11)

    def test_class_mode_includes_two_pass_and_read_drain(self):
        result = acrt_class_cycles(
            make_row(),
            lane_width=4,
            read_latency=1,
            ready_percent=100,
        )
        self.assertEqual(result["denominator"], 6)
        self.assertEqual(result["gate_fold"], 3)
        self.assertEqual(result["intersection"], 6)
        self.assertEqual(result["cycles"], 15)

    def test_allclass_replay_width_is_explicit(self):
        w1 = allclass_replay_cycles(
            make_row(),
            replay_width=1,
            ready_percent=100,
        )
        w4 = allclass_replay_cycles(
            make_row(),
            replay_width=4,
            ready_percent=100,
        )
        self.assertEqual(w1["denominator"], w4["denominator"])
        self.assertGreater(w1["cycles"], w4["cycles"])

    def test_event_prefix_switch_is_not_oracle(self):
        sparse = aenr_cycles(
            make_row(),
            event_threshold=4,
            lane_width=4,
            read_latency=1,
            ready_percent=100,
        )
        dense = aenr_cycles(
            make_row(),
            event_threshold=3,
            lane_width=4,
            read_latency=1,
            ready_percent=100,
        )
        self.assertEqual(sparse["mode"], "singleton")
        self.assertEqual(dense["mode"], "class")


if __name__ == "__main__":
    unittest.main()
