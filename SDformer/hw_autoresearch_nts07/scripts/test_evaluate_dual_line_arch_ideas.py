#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluate_dual_line_arch_ideas import (
    DEFAULT_LOCAL5_DETAIL,
    DEFAULT_MOTION_DETAIL,
    DEFAULT_PROFILE,
    build_result,
    combined_exact_or_sparse_coverage,
)


class DualLineIdeaScreenTest(unittest.TestCase):
    def test_combined_coverage_bounds(self) -> None:
        self.assertEqual(combined_exact_or_sparse_coverage(1.0, 0.0), 1.0)
        self.assertEqual(combined_exact_or_sparse_coverage(0.0, 1.0), 1.0)
        self.assertAlmostEqual(
            combined_exact_or_sparse_coverage(0.75, 0.80), 0.95
        )

    def test_current_profile_contract(self) -> None:
        profile = json.loads(
            Path(DEFAULT_PROFILE).read_text(encoding="utf-8")
        )
        motion_detail = json.loads(
            Path(DEFAULT_MOTION_DETAIL).read_text(encoding="utf-8")
        )
        local5_detail = json.loads(
            Path(DEFAULT_LOCAL5_DETAIL).read_text(encoding="utf-8")
        )
        result = build_result(profile, motion_detail, local5_detail)
        self.assertEqual(result["mainline"]["current"], "H67 Motion")
        self.assertEqual(len(result["ideas"]), 6)
        self.assertGreater(
            result["derived"]["motion_tare4_zero_or_list4_coverage"], 0.90
        )
        self.assertGreater(
            result["derived"][
                "local5_tare4_exact_or_list4_coverage_pre_g0"
            ],
            0.90,
        )
        self.assertIsNone(
            next(
                idea
                for idea in result["ideas"]
                if idea["id"] == "I2"
            )["local5"]["opportunity"]
        )
        absolute = result["derived"]["absolute_work"]
        self.assertTrue(absolute["available"])
        self.assertGreater(
            absolute["local5_over_motion_active_k_reads"], 1.0
        )
        self.assertGreater(
            absolute["local5_over_motion_projection_terms"], 1.0
        )
        self.assertFalse(
            result["mainline"]["current_switch_decision"]["pass"]
        )


if __name__ == "__main__":
    unittest.main()
