#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_local5_joint_relation_memo import (  # noqa: E402
    JointWindow,
    sequence_cluster_ratio_ci,
    evaluate,
)


class Local5JointRelationMemoAnalysisTest(unittest.TestCase):
    def test_cluster_ci_preserves_constant_ratio(self) -> None:
        baseline = np.asarray([200.0, 400.0, 600.0])
        candidate = baseline / 2.0
        result = sequence_cluster_ratio_ci(
            baseline,
            candidate,
            ["a", "a", "b"],
            trials=200,
            seed=7,
        )
        self.assertEqual(result["ratio_of_means"], 2.0)
        self.assertEqual(result["ci95_lower"], 2.0)
        self.assertEqual(result["ci95_upper"], 2.0)

    def test_evaluate_uses_joint_head_vectors(self) -> None:
        windows = []
        for sample in range(100):
            for stage, heads in enumerate((3, 6, 12, 24)):
                depth = (2, 2, 6, 2)[stage]
                for block in range(depth):
                    windows.append(
                        JointWindow(
                            sample=sample,
                            stage=stage,
                            block=block,
                            window=0,
                            analysis_weight=float((440, 120, 30, 10)[stage]),
                            service_cycles=np.full(heads, 20, dtype=np.int64),
                            packet_storage_bits=np.full(
                                heads, 112, dtype=np.int64
                            ),
                        )
                    )
        result = evaluate(
            windows,
            capacity_kib=7,
            policy="critical_only",
            trials=100,
            seed=11,
            sequence_keys=[f"seq-{index // 10}" for index in range(100)],
        )
        self.assertEqual(result["cluster_bootstrap"]["samples"], 100)
        self.assertEqual(
            sum(row["joint_windows"] for row in result["per_stage"]),
            1200,
        )
        self.assertGreater(
            result["cluster_bootstrap"]["ratio_of_means"], 1.0
        )
        self.assertEqual(
            result["cluster_bootstrap"]["sequence_clusters"], 10
        )


if __name__ == "__main__":
    unittest.main()
