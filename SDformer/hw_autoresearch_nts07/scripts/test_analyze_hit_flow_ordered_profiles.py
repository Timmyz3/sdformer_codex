from __future__ import annotations

import unittest
import base64
import zlib

from analyze_hit_flow_ordered_profiles import (
    atlif_quant_summary,
    csd_nonzero_digits,
    decode_count_trace,
    pair_and_bank_summary,
    stage_value_summary,
)


class OrderedProfileAnalyzerTest(unittest.TestCase):
    def test_csd_digit_count_reduces_runs_of_ones(self):
        self.assertEqual(csd_nonzero_digits(0), 0)
        self.assertEqual(csd_nonzero_digits(1), 1)
        self.assertEqual(csd_nonzero_digits(7), 2)
        self.assertEqual(csd_nonzero_digits(255), 2)

    def test_decode_count_trace_checks_shape(self):
        raw = b"\x01\x00\xfe\xff"
        encoded = {
            "shape": [2],
            "dtype": "int16_le",
            "codec": "zlib_base64",
            "data": base64.b64encode(zlib.compress(raw)).decode("ascii"),
        }
        self.assertEqual(decode_count_trace(encoded), [1, -2])

    def test_decode_count_trace_supports_int32(self):
        raw = b"\x40\x9c\x00\x00"
        encoded = {
            "shape": [1],
            "dtype": "int32_le",
            "codec": "zlib_base64",
            "data": base64.b64encode(zlib.compress(raw)).decode("ascii"),
        }
        self.assertEqual(decode_count_trace(encoded), [40000])

    def test_stage_value_summary_uses_finite_count_weights(self):
        rows = [
            {
                "name": "S0.skip", "kind": "stage_skip_predownsample",
                "elements": 4, "finite_count": 4,
                "value_min": 0.0, "value_max": 1.0, "value_absmax": 1.0,
                "near_integer_ratio": 1.0, "binary01_ratio": 1.0, "ternary_ratio": 1.0,
            },
            {
                "name": "S0.skip", "kind": "stage_skip_predownsample",
                "elements": 2, "finite_count": 2,
                "value_min": -2.0, "value_max": 2.0, "value_absmax": 2.0,
                "near_integer_ratio": 0.0, "binary01_ratio": 0.0, "ternary_ratio": 0.0,
            },
        ]
        summary = stage_value_summary(rows)[0]
        self.assertEqual(summary["value_min"], -2.0)
        self.assertEqual(summary["value_absmax"], 2.0)
        self.assertAlmostEqual(summary["binary01_ratio"], 4 / 6)

    def test_atlif_quant_summary_excludes_dead_results(self):
        base = {
            "quant_sample_events": 100,
            "recomputed_reference_mismatch": 0,
            "parameter_q4_event_mismatch": 10,
            "parameter_q6_event_mismatch": 2,
            "parameter_q8_event_mismatch": 0,
            "margin_abs_le_1_128": 1,
            "margin_abs_le_1_64": 2,
            "margin_abs_le_1_32": 4,
            "margin_abs_le_1_16": 8,
        }
        rows = [
            {**base, "deployment_dead_result": False},
            {**base, "deployment_dead_result": True, "parameter_q8_event_mismatch": 50},
        ]
        summary = atlif_quant_summary(rows)
        self.assertEqual(summary["live_modules"], 1)
        self.assertEqual(summary["sample_events"], 100)
        self.assertAlmostEqual(summary["q4_event_mismatch_ratio"], 0.10)
        self.assertEqual(summary["q8_event_mismatches"], 0)

    def test_pair_summary_reports_conditional_pccc_coverage(self):
        pair = {
            "pair_empty_ratio": 0.5,
            "pair_kzero_both": 80,
            "pair_kzero_both_ratio": 0.8,
            "pair_kzero_same_class_h67": 60,
            "pair_kzero_same_class_h67_ratio": 0.6,
            "spatial_persistence_ratio": 0.7,
            "spatial_change_ratio": 0.3,
            "k_temporal_union_read_ratio": 0.9,
            "k_temporal_exact_reuse_ratio": 0.1,
            "projection_class_channel_ratio_ttx": 0.6,
            "projection_class_channel_ratio_h67": 0.7,
            "projection_gate_class_channel_ratio_deploy": 0.5,
        }
        for direction in ("horizontal", "vertical", "diag_down", "diag_up"):
            pair[f"spatial_{direction}_adjacent_ratio"] = 0.25
        for banks in (4, 8):
            for mapping in ("rowmajor", "diagonal", "xor"):
                pair[f"spatial_bank{banks}_{mapping}_cycles_mean"] = float(banks)
        data = {"summary": {"binary_temporal_pairs": pair}}
        summary = pair_and_bank_summary(data)
        self.assertAlmostEqual(summary["both_kzero_same_class_h67_ratio_conditional"], 0.75)
        self.assertAlmostEqual(summary["projection_class_channel_ratio_h67"], 0.7)
        self.assertAlmostEqual(summary["projection_gate_class_channel_ratio_deploy"], 0.5)
        self.assertEqual(len(summary["bank_mappings"]), 6)


if __name__ == "__main__":
    unittest.main()
