#!/usr/bin/env python3
"""H84 packed Class File is the only expand object."""

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
    _adjacent_row_class_jaccard,
    _expand_k_from_packed_class_file,
    _pack_occupied_class_file,
    _q7_class_grid,
    _soft_hard_membership,
    shiftmax,
)


def _cfg() -> ShiftmaxAttentionConfig:
    return ShiftmaxAttentionConfig(
        hardware_score_step=1.0 / 128.0,
        hardware_score_min=-2.0,
        hardware_score_max=2.0,
        eps=1.0e-6,
    )


def _file_from_scores(scores: torch.Tensor) -> dict:
    cfg = _cfg()
    step, lo, _hi, n_bins = _q7_class_grid(cfg)
    squeezed = scores.squeeze(-1)
    centers = lo + step * torch.arange(n_bins, device=scores.device, dtype=scores.dtype)
    member, hard = _soft_hard_membership(squeezed, centers, step)
    return _pack_occupied_class_file(member, hard, squeezed, centers, cfg)


class PackedClassFileTests(unittest.TestCase):
    def test_expand_forbids_codes_key(self) -> None:
        scores = torch.tensor([[[[0.0], [1.0]]]], dtype=torch.float32)
        class_file = _file_from_scores(scores)
        class_file["codes"] = torch.zeros(1, 1, 2, dtype=torch.long)
        with self.assertRaises(RuntimeError):
            _expand_k_from_packed_class_file(torch.ones(1, 1, 2, 1), class_file)

    def test_expand_uses_only_member_mask(self) -> None:
        scores = torch.tensor([[[[0.0], [0.0], [0.0], [1.0]]]], dtype=torch.float32)
        class_file = _file_from_scores(scores)
        self.assertNotIn("codes", class_file)
        k = torch.ones(1, 1, 4, 2)
        attn, token_gate = _expand_k_from_packed_class_file(k, class_file)
        token_shiftmax = shiftmax(scores, dim=2)
        self.assertGreater(float((token_gate - token_shiftmax).abs().max()), 1.0e-4)
        rebuilt = (class_file["member_mask"] * class_file["gate_c"].unsqueeze(-1)).sum(2)
        self.assertTrue(torch.allclose(token_gate.squeeze(-1), rebuilt, atol=1.0e-6))
        self.assertTrue(torch.allclose(attn, k.mul(token_gate)))

    def test_row_jaccard_has_ste_grad(self) -> None:
        leaf = torch.linspace(-0.2, 0.2, 450, requires_grad=True)
        scores = leaf.view(1, 1, 450, 1)
        class_file = _file_from_scores(scores)
        jacc = _adjacent_row_class_jaccard(class_file["member_mask"], class_file["valid"])
        (1.0 - jacc).backward()
        self.assertIsNotNone(leaf.grad)
        self.assertGreater(float(leaf.grad.abs().sum()), 0.0)
        self.assertTrue(torch.isfinite(jacc))


if __name__ == "__main__":
    unittest.main()
