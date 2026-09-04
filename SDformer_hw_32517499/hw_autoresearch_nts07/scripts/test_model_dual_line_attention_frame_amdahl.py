#!/usr/bin/env python3
"""Tests for the scoped dual-line attention/Amdahl model."""

from __future__ import annotations

import unittest
from pathlib import Path

from scripts.model_dual_line_attention_frame_amdahl import amdahl, build_report


ROOT = Path(__file__).resolve().parents[1]


class DualLineAttentionAmdahlTest(unittest.TestCase):
    def test_amdahl_edges(self) -> None:
        self.assertEqual(amdahl(1.2, 0.0), 1.0)
        self.assertAlmostEqual(amdahl(1.2, 1.0), 1.2)
        with self.assertRaises(ValueError):
            amdahl(0.0, 0.5)

    def test_current_evidence_receipts(self) -> None:
        report = build_report(
            ROOT / "results/h67_fair_merge_population_20260813/ep35_fair_merge.log",
            ROOT / "tb_qfit/vectors/local5_joint_ep29_score_projection_realw_sample100_population_out32_v1_20260814/manifest.json",
            ROOT / "results/local5_out32_population_sensitivity_20260814/t450_qsilent_verilator_assert.log",
            ROOT / "results/local5_out32_population_sensitivity_20260814/rolling_qsilent_verilator_assert.log",
        )
        motion = report["motion"]["frame_attention_row_model"]
        self.assertEqual(motion["fixed2s_cycles"], 4_137_640)
        self.assertEqual(motion["rqtb2s_cycles"], 3_448_960)
        self.assertAlmostEqual(motion["component_speedup"], 1.1996775845478058)
        local = report["local5"]["frame_attention_one_output_tile_model"]
        self.assertAlmostEqual(local["t450_cycles"], 8_625_420.131868131)
        self.assertAlmostEqual(local["rolling_cycles"], 7_725_115.032967033)
        self.assertAlmostEqual(local["component_speedup"], 1.1165426140652699)
        replay = report["local5"]["packed_pipeline_tile_replay_model"]
        self.assertAlmostEqual(replay["component_speedup"], 1.1571910839412565)


if __name__ == "__main__":
    unittest.main()
