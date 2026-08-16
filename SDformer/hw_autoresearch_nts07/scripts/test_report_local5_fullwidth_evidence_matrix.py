#!/usr/bin/env python3
"""Tests for the Local5 full-width evidence matrix."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.report_local5_fullwidth_evidence_matrix import build_report


ROOT = Path(__file__).resolve().parents[1]


class Local5FullWidthEvidenceMatrixTest(unittest.TestCase):
    def test_current_receipts(self) -> None:
        report = build_report(
            ROOT / "results/local5_out32_population_sensitivity_20260814/report.json",
            ROOT / "results/local5_erep_integrated_cross_head_canary_v5_tagfix_20260811/merge_report.json",
            {
                1: ROOT / "results/local5_erep_integrated_stage1_h6_smoke_20260811/smoke_report.json",
                2: ROOT / "results/local5_erep_integrated_stage2_h12_smoke_20260811/smoke_report.json",
                3: ROOT / "results/local5_erep_integrated_stage3_h24_smoke_v2_20260811/smoke_report.json",
            },
        )
        self.assertEqual(report["optimized_front_population"]["acc32_checked"], 1_440_000)
        self.assertEqual(
            report["integrated_hxh_recompute_canaries"]["final_acc32_checked"],
            648_000,
        )
        self.assertEqual(report["integrated_hxh_recompute_canaries"]["mismatch"], 0)


if __name__ == "__main__":
    unittest.main()
