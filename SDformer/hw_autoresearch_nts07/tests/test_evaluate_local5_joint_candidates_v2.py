from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import evaluate_local5_joint_candidates_v2 as evaluator


class EvaluateLocal5JointCandidatesTest(unittest.TestCase):
    def test_paired_sample_bootstrap_identity(self) -> None:
        values = np.arange(1, 101, dtype=np.float64)
        result = evaluator.bootstrap_ratio(
            values,
            values,
            trials=100,
            seed=7,
            clusters=None,
        )
        self.assertEqual(result["ratio_of_means"], 1.0)
        self.assertEqual(result["one_sided_familywise_lower"], 1.0)
        self.assertAlmostEqual(result["one_sided_alpha"], 0.05 / 3)

    def test_sequence_bootstrap_constant_speedup(self) -> None:
        baseline = np.arange(1, 101, dtype=np.float64) * 2
        candidate = baseline / 2
        clusters = [f"s{index % 10}" for index in range(100)]
        result = evaluator.bootstrap_ratio(
            baseline,
            candidate,
            trials=100,
            seed=9,
            clusters=clusters,
        )
        self.assertEqual(result["ratio_of_means"], 2.0)
        self.assertEqual(result["one_sided_familywise_lower"], 2.0)

    def test_bootstrap_rejects_invalid_alpha(self) -> None:
        values = np.ones(100, dtype=np.float64)
        with self.assertRaises(ValueError):
            evaluator.bootstrap_ratio(
                values,
                values,
                trials=10,
                seed=1,
                clusters=None,
                one_sided_alpha=0.5,
            )


if __name__ == "__main__":
    unittest.main()
