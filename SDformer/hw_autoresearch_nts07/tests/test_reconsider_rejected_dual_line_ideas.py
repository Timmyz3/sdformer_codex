#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(
    0,
    str(
        Path(__file__).resolve().parents[1]
        / "scripts"
    ),
)

from reconsider_rejected_dual_line_ideas import (
    DEFAULT_ADAPTIVE_FORMAT,
    DEFAULT_DUAL_PROFILE,
    DEFAULT_LOCAL5_PROFILE,
    DEFAULT_MOTION_PROFILE,
    DEFAULT_PPDI_PROFILE,
    DEFAULT_RESOLUTION,
    build_result,
    combined_exact_sparse_coverage,
    destination_encoding_cost,
    histogram_quantile,
    ideal_dual_destination_commands,
    validate_inputs,
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class ReconsiderRejectedIdeasTest(unittest.TestCase):
    def test_histogram_helpers(self) -> None:
        histogram = [0, 2, 1, 1]
        self.assertEqual(histogram_quantile(histogram, 0.5), 1)
        self.assertEqual(histogram_quantile(histogram, 0.75), 2)
        self.assertEqual(ideal_dual_destination_commands(histogram), 5)
        self.assertAlmostEqual(
            combined_exact_sparse_coverage(0.75, 0.80),
            0.95,
        )

    def test_destination_encoding(self) -> None:
        histogram = [0, 1, 1, 0, 1]
        result = destination_encoding_cost(
            histogram,
            tokens=16,
            destination_id_bits=4,
        )
        self.assertEqual(result["terms"], 3)
        self.assertEqual(result["deliveries"], 7)
        self.assertEqual(result["list_bits"], 28)
        self.assertEqual(result["bitmap_bits"], 48)
        self.assertEqual(result["adaptive_bits_including_tag"], 31)

    def test_current_profile_reopens_three_candidates(self) -> None:
        result = build_result(
            load(DEFAULT_DUAL_PROFILE),
            load(DEFAULT_MOTION_PROFILE),
            load(DEFAULT_LOCAL5_PROFILE),
            load(DEFAULT_PPDI_PROFILE),
            load(DEFAULT_ADAPTIVE_FORMAT),
            load(DEFAULT_RESOLUTION),
        )
        self.assertEqual(len(result["reopened"]), 3)
        self.assertTrue(result["input_audit"]["all_passed"])
        self.assertEqual(result["reopened"][0]["decision"], "DEFER_AFTER_ET3")
        self.assertEqual(
            result["reopened"][1]["decision"],
            "RTL_PROTOTYPE_DONE_PROFILE_PENDING",
        )
        self.assertGreater(
            result["derived"]["motion_zero_or_list4_coverage"],
            0.90,
        )
        self.assertGreater(
            result["derived"][
                "local5_exact_or_list4_coverage_pre_g0"
            ],
            0.90,
        )
        self.assertGreater(
            result["derived"]["local5_mpet_fanout_mean_pre_g0"],
            10.0,
        )
        self.assertGreater(
            result["derived"][
                "local5_native_multiset_command_reduction_pre_g0"
            ],
            0.18,
        )
        self.assertGreater(
            result["derived"][
                "local5_mpet_product_compute_reduction_pre_g0"
            ],
            0.92,
        )
        self.assertEqual(
            result["derived"]["local5_mpet_fanout_p95_pre_g0"],
            45,
        )
        self.assertGreater(
            result["derived"][
                "local5_ideal_pair_command_reduction_pre_g0"
            ],
            0.45,
        )
        self.assertGreater(
            result["derived"]["local5_w9_destination_encoding"][
                "adaptive_reduction_vs_list"
            ],
            0.25,
        )
        self.assertEqual(
            result["held_or_rejected"][0]["decision"],
            "KEEP_REJECTED",
        )

    def test_rejects_cohort_mismatch(self) -> None:
        dual = load(DEFAULT_DUAL_PROFILE)
        motion = load(DEFAULT_MOTION_PROFILE)
        local5 = load(DEFAULT_LOCAL5_PROFILE)
        ppdi = load(DEFAULT_PPDI_PROFILE)
        resolution = load(DEFAULT_RESOLUTION)
        local5["cohort"]["sample_key_sha256"] = "bad"
        with self.assertRaises(ValueError):
            validate_inputs(dual, motion, local5, ppdi, resolution)

    def test_resolution_id_widths(self) -> None:
        result = build_result(
            load(DEFAULT_DUAL_PROFILE),
            load(DEFAULT_MOTION_PROFILE),
            load(DEFAULT_LOCAL5_PROFILE),
            load(DEFAULT_PPDI_PROFILE),
            load(DEFAULT_ADAPTIVE_FORMAT),
            load(DEFAULT_RESOLUTION),
        )
        self.assertEqual(
            result["derived"]["local5_w9_destination_encoding"][
                "destination_id_bits"
            ],
            8,
        )
        self.assertEqual(
            result["derived"][
                "local5_w15_sensitivity_using_w9_fanout"
            ]["destination_id_bits"],
            9,
        )


if __name__ == "__main__":
    unittest.main()
