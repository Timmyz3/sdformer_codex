#!/usr/bin/env python3
"""H83 Class File is the executed ISA, not a discarded H82 sidecar."""

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
    _class_file_from_scores,
    _expand_k_from_class_file,
    shiftmax,
)


def _cfg() -> ShiftmaxAttentionConfig:
    return ShiftmaxAttentionConfig(
        hardware_score_step=1.0 / 128.0,
        hardware_score_min=-2.0,
        hardware_score_max=2.0,
        preserve_mean=False,
        eps=1.0e-6,
    )


class ClassFileISATests(unittest.TestCase):
    def test_class_file_keeps_occupied_records(self) -> None:
        scores = torch.tensor([[[[0.0], [0.0], [0.0], [1.0]]]], dtype=torch.float32)
        class_file = _class_file_from_scores(scores, _cfg())
        self.assertEqual(int(class_file["occupied"].sum().item()), 2)
        self.assertIn("temporal_pair_mask", class_file)
        self.assertIn("gate_c", class_file)
        self.assertIn("multiplicity", class_file)

    def test_expand_uses_class_gate_not_token_shiftmax(self) -> None:
        scores = torch.tensor([[[[0.0], [0.0], [0.0], [1.0]]]], dtype=torch.float32)
        k = torch.ones(1, 1, 4, 3)
        class_file = _class_file_from_scores(scores, _cfg())
        attn, token_gate = _expand_k_from_class_file(k, class_file)
        token_shiftmax = shiftmax(scores, dim=2)
        self.assertGreater(float((token_gate - token_shiftmax).abs().max()), 1.0e-4)
        self.assertTrue(torch.allclose(attn, k.mul(token_gate)))

    def test_no_preserve_mean_times_n(self) -> None:
        scores = torch.tensor([[[[0.0], [1.0]]]], dtype=torch.float32)
        class_file = _class_file_from_scores(scores, _cfg())
        _attn, token_gate = _expand_k_from_class_file(torch.ones(1, 1, 2, 1), class_file)
        self.assertLess(float(token_gate.max()), 1.5)

    def test_member_jaccard_defined_for_t2_window(self) -> None:
        scores = torch.zeros(1, 1, 450, 1)
        scores[:, :, :225] = 0.0
        scores[:, :, 225:] = 1.0
        class_file = _class_file_from_scores(scores, _cfg())
        self.assertTrue(torch.isfinite(class_file["member_jaccard_t0t1"]))


if __name__ == "__main__":
    unittest.main()
