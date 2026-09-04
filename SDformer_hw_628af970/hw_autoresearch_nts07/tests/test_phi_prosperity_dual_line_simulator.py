import unittest
import base64
import zlib
from dataclasses import replace

from scripts.phi_prosperity_dual_line_simulator import (
    ArchConfig,
    SampleWorkload,
    architecture_dse,
    ceil_div,
    decode_count_trace,
    evaluate_line,
    local_hist_service,
    ordered_decoupled_cycles,
    percentile,
    resolve_motion_temporal_tokens,
    scale_workload,
    simulate_sample,
    workloads_at_tokens,
)


class PhiProsperityDualLineSimulatorTest(unittest.TestCase):
    def setUp(self):
        self.cfg = ArchConfig()
        self.work = SampleWorkload(
            line="Motion",
            sample_id=0,
            tokens_per_window=162,
            vector_count=100,
            pattern_vectors=100,
            direct_score_lane_work=6400,
            anchor_score_lane_work=3200,
            residual_lane_work=64,
            online_match_lane_work=6400,
            direct_k_bits=3200,
            anchor_k_bits=1800,
            direct_projection_products=1000,
            exact_projection_terms=100,
            exact_destination_count=300,
            packed_delivery_commands=120,
            term_scan_entries=350,
            static_score_cycles_w2=140,
            static_score_cycles_w4=105,
            static_score_cycles_w8=101,
            bundle_total=25,
            bundle_empty=20,
            profile_source="synthetic",
            evidence="[test]",
        )

    def test_ceil_div_and_percentile(self):
        self.assertEqual(ceil_div(33, 32), 2)
        self.assertEqual(percentile([1, 2, 3, 4], 0.5), 2.5)

    def test_decoder_accepts_adaptive_int32_trace(self):
        encoded = {
            "shape": [1],
            "dtype": "int32_le",
            "codec": "zlib_base64",
            "data": base64.b64encode(
                zlib.compress(b"\x40\x9c\x00\x00")
            ).decode("ascii"),
        }
        self.assertEqual(decode_count_trace(encoded), [40000])

    def test_t450_scale_is_25_over_9(self):
        scaled = scale_workload(self.work, 450)
        self.assertEqual(scaled.tokens_per_window, 450)
        self.assertEqual(scaled.vector_count, round(100 * 25 / 9))
        self.assertEqual(scaled.evidence, "[外推-25/9]")

    def test_t162_preserves_measured_evidence(self):
        same = workloads_at_tokens([self.work], 162)
        self.assertIs(same[0], self.work)
        result = evaluate_line([self.work], self.cfg, 162, [0.5])
        self.assertEqual(result["evidence"], "[test]")

    def test_legacy_motion_profile_derives_consistent_record_tokens(self):
        records = [{"tokens": 162}, {"tokens": 162}]
        self.assertEqual(resolve_motion_temporal_tokens({}, records), 162)

    def test_motion_profile_rejects_protocol_record_token_mismatch(self):
        records = [{"tokens": 450}, {"tokens": 450}]
        with self.assertRaisesRegex(ValueError, "T与h60_records不一致"):
            resolve_motion_temporal_tokens({"tokens_per_window": 162}, records)

    def test_motion_profile_rejects_mixed_record_tokens(self):
        with self.assertRaisesRegex(ValueError, "T不唯一"):
            resolve_motion_temporal_tokens({}, [{"tokens": 162}, {"tokens": 450}])

    def test_phi_hit_rate_monotonically_reduces_cycles(self):
        low = simulate_sample(
            self.work, self.cfg, "phi_pattern_residual", phi_hit_rate=0.5
        )
        high = simulate_sample(
            self.work, self.cfg, "phi_pattern_residual", phi_hit_rate=0.9
        )
        self.assertLess(high["total_cycles"], low["total_cycles"])

    def test_static_anchor_uses_exact_terms_and_metadata(self):
        direct = simulate_sample(self.work, self.cfg, "direct")
        ours = simulate_sample(self.work, self.cfg, "static_anchor_term")
        self.assertGreater(ours["metadata_bits"], 0)
        self.assertLess(
            ours["components"]["projection"]["payload_bits"],
            direct["components"]["projection"]["payload_bits"],
        )

    def test_static_anchor_uses_ordered_or_hist_cycle_field(self):
        ours = simulate_sample(self.work, self.cfg, "static_anchor_term")
        self.assertEqual(
            ours["components"]["score"]["compute_cycles"],
            self.work.static_score_cycles_w4,
        )

    def test_prosperity_oracle_gets_exact_projection_reuse(self):
        prosperity = simulate_sample(self.work, self.cfg, "prosperity_online")
        direct = simulate_sample(self.work, self.cfg, "direct")
        self.assertLess(
            prosperity["components"]["projection"]["payload_bits"],
            direct["components"]["projection"]["payload_bits"],
        )

    def test_fixed64_uses_profiled_packed_commands(self):
        motion = simulate_sample(self.work, self.cfg, "static_anchor_term")
        self.assertEqual(
            motion["components"]["projection"]["compute_cycles"],
            self.work.packed_delivery_commands,
        )

    def test_term_and_destination_granularity_are_conserved(self):
        result = simulate_sample(self.work, self.cfg, "static_anchor_hierdesc")
        projection = result["components"]["projection"]
        self.assertEqual(
            projection["payload_bits"],
            self.work.exact_projection_terms
            * self.cfg.output_lanes
            * self.cfg.weight_bits,
        )
        self.assertEqual(
            projection["compute_cycles"],
            self.work.exact_destination_count,
        )
        expected_metadata = (
            self.work.exact_projection_terms * self.cfg.compact_header_bits
            + (self.work.exact_destination_count - self.work.exact_projection_terms)
            * (
                self.cfg.compact_delta_continuation_bits
                + self.cfg.compact_continuation_flags
            )
        )
        self.assertEqual(projection["metadata_bits"], expected_metadata)

    def test_ordered_backlog_and_local_hist_service(self):
        self.assertEqual(ordered_decoupled_cycles([0, 9, 0], 4), 4)
        record = {
            "up_delta_histogram": [1, 0, 2],
            "down_delta_histogram": [0, 1],
            "left_delta_histogram": [1],
            "right_delta_histogram": [0, 0, 1],
        }
        self.assertEqual(local_hist_service([record], 2), 4)

    def test_hierarchical_descriptor_reduces_metadata(self):
        fixed = simulate_sample(self.work, self.cfg, "static_anchor_term")
        compact = simulate_sample(self.work, self.cfg, "static_anchor_hierdesc")
        self.assertLess(compact["metadata_bits"], fixed["metadata_bits"])
        self.assertLess(compact["fabric_bits"], fixed["fabric_bits"])

    def test_local5_missing_ordered_tail_blocks_formal_promotion(self):
        result = architecture_dse(
            [replace(self.work, line="Local5")], self.cfg, 162
        )
        self.assertEqual(result["passing_count"], 0)
        self.assertTrue(
            all(
                rec["passes_tail_contract"] is None
                and not rec["passes"]
                for rec in result["all_candidates"]
            )
        )

    def test_motion_t450_extrapolation_is_not_ordered_evidence(self):
        measured = replace(self.work, evidence="[prof-ordered]")
        result = architecture_dse([measured], self.cfg, 450)
        self.assertTrue(
            all(
                rec["passes_tail_contract"] is None
                and rec["tail_evidence"] == "[缺失-ordered]"
                for rec in result["all_candidates"]
            )
        )

    def test_measured_motion_t450_preserves_ordered_tail_evidence(self):
        measured = replace(
            self.work,
            tokens_per_window=450,
            evidence="[prof-ordered-fullres]",
        )
        result = architecture_dse([measured], self.cfg, 450)
        self.assertTrue(
            all(
                rec["passes_tail_contract"] is not None
                and rec["tail_evidence"] == "[prof-ordered]"
                for rec in result["all_candidates"]
            )
        )


if __name__ == "__main__":
    unittest.main()
