#!/usr/bin/env python3
"""H82 Class-Major TTX is a new operator, not a C7 rewrite."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXPERIMENT_ROOT.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "third_party" / "SDformerFlow"))
sys.path.insert(0, str(EXPERIMENT_ROOT / "overlay"))

from models.STSwinNet_SNN.bsa_attention import (
    ShiftmaxAttentionConfig,
    _class_major_shiftmax_gate,
    shiftmax,
)


def _cfg(**kwargs) -> ShiftmaxAttentionConfig:
    values = dict(
        hardware_score_step=1.0 / 128.0,
        hardware_score_min=-2.0,
        hardware_score_max=2.0,
        preserve_mean=False,
        eps=1.0e-6,
        class_stability_regularization_weight=0.0,
    )
    values.update(kwargs)
    return ShiftmaxAttentionConfig(**values)


class ClassMajorTTXTests(unittest.TestCase):
    def test_class_major_is_not_multiplicity_weighted_shiftmax(self) -> None:
        scores = torch.tensor(
            [[[[0.0], [0.0], [0.0], [1.0]]]],
            dtype=torch.float32,
        )
        token_gate = shiftmax(scores, dim=2)
        class_gate, stats = _class_major_shiftmax_gate(scores, _cfg())
        self.assertEqual(int(round(float(stats["n_occupied_classes"]))), 2)
        self.assertGreater(
            float((class_gate - token_gate).abs().max()),
            1.0e-4,
        )

    def test_equal_multiplicity_matches_token_shiftmax(self) -> None:
        scores = torch.tensor(
            [[[[0.0], [1.0]]]],
            dtype=torch.float32,
        )
        token_gate = shiftmax(scores, dim=2)
        class_gate, _stats = _class_major_shiftmax_gate(scores, _cfg())
        self.assertTrue(torch.allclose(class_gate, token_gate, atol=1.0e-6))

    def test_stability_proxy_is_captured(self) -> None:
        scores = torch.linspace(-0.5, 0.5, 450, dtype=torch.float32).view(1, 1, 450, 1)
        _gate, stats = _class_major_shiftmax_gate(
            scores,
            _cfg(class_stability_regularization_weight=0.01),
        )
        self.assertIn("stability_proxy", stats)
        self.assertTrue(torch.isfinite(stats["stability_proxy"]))


if __name__ == "__main__":
    unittest.main()
