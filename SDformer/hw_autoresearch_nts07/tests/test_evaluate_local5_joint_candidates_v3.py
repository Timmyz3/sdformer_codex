from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_local5_joint_candidates_v3 as evaluator


class EvaluateLocal5JointCandidatesV3Test(unittest.TestCase):
    def test_build_comparison_constant_speedup(self) -> None:
        baseline = np.arange(1, 101, dtype=np.float64) * 2
        candidate = baseline / 2
        windows_base = np.full(1200, 20.0)
        windows_candidate = np.full(1200, 10.0)
        weights = np.ones(1200)
        stages = np.tile(np.arange(4), 300)
        sequences = [f"seq{index % 18}" for index in range(100)]
        row = evaluator.build_comparison(
            baseline,
            candidate,
            windows_base,
            windows_candidate,
            weights,
            stages,
            sequences,
        )
        self.assertEqual(row["scenario_gate"], "PASS_SCENARIO")
        self.assertEqual(row["sample_bootstrap"]["ratio_of_means"], 2.0)

    def test_aggregate_requires_both_fixed_scenarios(self) -> None:
        passed = {"scenario_gate": "PASS_SCENARIO"}
        failed = {"scenario_gate": "FAIL_SCENARIO"}
        self.assertEqual(
            evaluator.aggregate_candidate_gate(
                {
                    "calibrated_median_459": passed,
                    "calibration_max_475": passed,
                }
            ),
            "PROMOTE_TO_MINIMAL_RTL",
        )
        self.assertEqual(
            evaluator.aggregate_candidate_gate(
                {
                    "calibrated_median_459": passed,
                    "calibration_max_475": failed,
                }
            ),
            "REJECT_MODEL_PROMOTION",
        )

    def test_aggregate_rejects_missing_scenario(self) -> None:
        with self.assertRaises(ValueError):
            evaluator.aggregate_candidate_gate(
                {"calibrated_median_459": {"scenario_gate": "PASS_SCENARIO"}}
            )


if __name__ == "__main__":
    unittest.main()
