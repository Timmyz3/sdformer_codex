import unittest

from scripts.model_fcip_equal_bandwidth import (
    b1_cycles,
    ncfip_cycles,
    paired_comparison,
    relation_plane_cycles,
    simulate_bounded_assembler,
    singleton_event_cycles,
    sink_service_cycles,
)


def make_row() -> dict:
    return {
        "tokens": 8,
        "lanes": 2,
        "segments": 2,
        "active_tokens": 4,
        "active_score_classes": 2,
        "final_gate_lane_terms": 2,
        "class_words": {
            "1": [0b0011, 0],
            "2": [0, 0b0011],
        },
        "k_words": [
            [0b0001, 0b0010],
            [0b0010, 0],
        ],
        "final_gate_groups": {"64": [1, 2]},
    }


class FcipEqualBandwidthTest(unittest.TestCase):
    def test_sink_backpressure(self):
        self.assertEqual(sink_service_cycles(9, 90), 9)
        self.assertEqual(sink_service_cycles(10, 90), 11)
        self.assertEqual(sink_service_cycles(4, 75), 5)

    def test_bounded_assembler_conserves_terms(self):
        report = simulate_bounded_assembler(
            [
                {"read_work": 2, "produces_term": True},
                {"read_work": 1, "produces_term": False},
                {"read_work": 1, "produces_term": True},
            ],
            contexts=2,
            read_width=2,
            read_latency=1,
            ready_percent=100,
        )
        self.assertEqual(report["emitted_terms"], 2)
        self.assertEqual(report["empty_retired"], 1)
        self.assertEqual(report["reads"], 4)

    def test_fcip_and_b2_emit_exact_final_terms(self):
        row = make_row()
        for architecture in ("fcip", "b2"):
            report = relation_plane_cycles(
                row,
                architecture=architecture,
                ingress_width=1,
                read_width=1,
                contexts=2,
                read_latency=1,
                ready_percent=100,
            )
            self.assertEqual(report["emitted_terms"], 2)
            self.assertFalse(report["fallback"])

    def test_overflow_pays_abort_and_replay(self):
        row = make_row()
        row["active_score_classes"] = 17
        result = relation_plane_cycles(
            row,
            architecture="fcip",
            ingress_width=1,
            read_width=1,
            contexts=2,
            read_latency=1,
            ready_percent=100,
        )
        replay = b1_cycles(
            row,
            ingress_width=1,
            ready_percent=100,
        )
        self.assertTrue(result["fallback"])
        self.assertEqual(
            result["cycles"],
            row["active_tokens"] + 1 + replay["cycles"],
        )

    def test_ncfip_fusion_removes_only_transduction_cycles(self):
        row = make_row()
        explicit = ncfip_cycles(
            row,
            ingress_width=1,
            read_width=1,
            contexts=2,
            read_latency=1,
            ready_percent=100,
            transduction_overlapped=False,
        )
        fused = ncfip_cycles(
            row,
            ingress_width=1,
            read_width=1,
            contexts=2,
            read_latency=1,
            ready_percent=100,
            transduction_overlapped=True,
        )
        self.assertEqual(
            explicit["cycles"] - fused["cycles"],
            explicit["transduction_cycles"],
        )
        self.assertEqual(explicit["emitted_terms"], fused["emitted_terms"])

    def test_singleton_mode_uses_nonzero_gate_events(self):
        row = make_row()
        row["active_nonzero_gate_lane_events"] = 3
        self.assertEqual(
            singleton_event_cycles(row, ready_percent=100),
            3,
        )

    def test_paired_tail_is_rowwise(self):
        result = paired_comparison([10, 100], [20, 90])
        self.assertAlmostEqual(
            result["paired_slowdown"]["p99"],
            1.989,
            places=3,
        )
        self.assertEqual(result["rows_over_10pct_slower_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
