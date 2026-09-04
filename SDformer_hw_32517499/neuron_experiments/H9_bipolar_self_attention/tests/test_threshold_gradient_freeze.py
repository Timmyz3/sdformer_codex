from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
OVERLAY = EXPERIMENT_ROOT / "overlay"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "SDformerFlow"))
sys.path.insert(0, str(OVERLAY))

from models.STSwinNet_SNN.atlif_ternary_psn.atlif_ternary_psn import ATLIFTernaryPSN
from models.STSwinNet_SNN.h28_optimizer import freeze_threshold_gradients


class ThresholdGradientFreezeTest(unittest.TestCase):
    def test_gradient_freeze_is_explicit_and_step_bounded(self) -> None:
        model = nn.Sequential(ATLIFTernaryPSN(T=2))
        model[0].thresh.grad = torch.ones_like(model[0].thresh)
        config = {
            "atlif_ternary_psn": {
                "freeze_threshold_grad_after_step": True,
                "threshold_freeze_after_step": 10,
            }
        }

        self.assertEqual(freeze_threshold_gradients(model, 9, config), 0)
        self.assertIsNotNone(model[0].thresh.grad)
        self.assertEqual(freeze_threshold_gradients(model, 10, config), 1)
        self.assertIsNone(model[0].thresh.grad)

    def test_historical_config_does_not_freeze_gradient(self) -> None:
        model = nn.Sequential(ATLIFTernaryPSN(T=2))
        model[0].thresh.grad = torch.ones_like(model[0].thresh)
        config = {
            "atlif_ternary_psn": {
                "threshold_freeze_after_step": 0,
            }
        }
        self.assertEqual(freeze_threshold_gradients(model, 10, config), 0)
        self.assertIsNotNone(model[0].thresh.grad)


if __name__ == "__main__":
    unittest.main()
