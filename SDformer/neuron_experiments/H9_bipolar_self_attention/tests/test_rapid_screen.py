from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "SDformerFlow"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "overlay"))


class RapidScreenTest(unittest.TestCase):
    def test_confirm_stage_requires_confirm_steps_and_promote_samples(self):
        from neuron_experiments.H9_bipolar_self_attention.entrypoints.rapid_screen import compact_summary

        args = Namespace(
            confirm_steps=360,
            promote_samples=40,
            promote_aee=1.58,
            promote_aae=7.9,
            promote_sops_g=3.35,
            max_zero_neg_modules=4.0,
            max_worst_pos_neg_ratio=40.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "sops_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "samples": 40,
                        "estimated_total_sops": 3.1e9,
                        "global_firing_rate": 0.07,
                        "metrics": {"AEE": 1.4, "AAE": 7.0},
                    }
                ),
                encoding="utf-8",
            )
            train_log = root / "train.log"
            train_log.write_text(
                "[H9] ATLIFTernaryPSN summary: "
                "{'threshold_mean': 1.0, 'threshold_min': 0.5, 'threshold_max': 1.5, "
                "'ternary_activity_mean': 0.03, 'ternary_pos_neg_ratio': 1.2, "
                "'ternary_worst_pos_neg_ratio': 2.0, 'ternary_zero_pos_modules': 0, "
                "'ternary_zero_neg_modules': 0, 'target_rate_control_modules': 12, "
                "'target_rate_bidirectional_modules': 0}\n",
                encoding="utf-8",
            )

            early = compact_summary("candidate_steps120_valid40", 120, summary_path, train_log, 1.0, 1.0, args)
            confirmed = compact_summary("candidate_steps360_valid40", 360, summary_path, train_log, 1.0, 1.0, args)

        self.assertEqual(early["stage"], "screen")
        self.assertEqual(early["gate"], "pass")
        self.assertEqual(confirmed["stage"], "confirm")
        self.assertEqual(confirmed["gate"], "pass")


if __name__ == "__main__":
    unittest.main()
