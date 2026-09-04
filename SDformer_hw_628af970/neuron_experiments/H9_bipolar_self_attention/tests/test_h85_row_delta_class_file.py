#!/usr/bin/env python3
"""H85 per-row Class File: no T450 token gate, delta is a file field."""

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
    _build_h85_row_files,
    _expand_k_from_h85_row_files,
)


def _cfg() -> ShiftmaxAttentionConfig:
    return ShiftmaxAttentionConfig(
        hardware_score_step=1.0 / 128.0,
        hardware_score_min=-2.0,
        hardware_score_max=2.0,
        eps=1.0e-6,
    )


class RowDeltaClassFileTests(unittest.TestCase):
    def test_member_storage_is_in_row_not_t450(self) -> None:
        scores = torch.zeros(1, 1, 450, 1)
        class_file = _build_h85_row_files(scores, _cfg())
        self.assertEqual(class_file["member_idx"].shape[-1], 15)
        self.assertNotEqual(class_file["member_idx"].numel(), 0)
        self.assertNotIn("token_gate", class_file)
        self.assertNotIn("codes", class_file)
        self.assertNotIn("member_mask", class_file)

    def test_expand_forbids_t450_tensors(self) -> None:
        scores = torch.zeros(1, 1, 450, 1)
        class_file = _build_h85_row_files(scores, _cfg())
        k = torch.ones(1, 1, 450, 2)
        for key in ("codes", "token_gate", "member_mask"):
            bad = dict(class_file)
            bad[key] = torch.zeros(1, 1, 450)
            with self.assertRaises(RuntimeError):
                _expand_k_from_h85_row_files(k, bad)

    def test_same_class_set_writes_reuse_delta(self) -> None:
        scores = torch.zeros(1, 1, 450, 1)
        class_file = _build_h85_row_files(scores, _cfg())
        self.assertGreater(float(class_file["reuse_set"].to(dtype=torch.float32).mean()), 0.9)
        self.assertIn("shared_ids", class_file)
        self.assertIn("insert_ids", class_file)
        self.assertIn("delete_ids", class_file)

    def test_expand_uses_class_id_and_returns_token_output(self) -> None:
        scores = torch.zeros(1, 1, 450, 1)
        scores[0, 0, 0, 0] = 1.0
        class_file = _build_h85_row_files(scores, _cfg())
        k = torch.randn(1, 1, 450, 3)
        attn = _expand_k_from_h85_row_files(k, class_file)
        self.assertEqual(attn.shape, k.shape)
        self.assertTrue(torch.isfinite(attn).all())
        self.assertGreater(int(class_file["class_id"].sum().item()), 0)

    def test_expand_does_not_return_t450_token_gate(self) -> None:
        scores = torch.zeros(1, 1, 450, 1)
        class_file = _build_h85_row_files(scores, _cfg())
        attn = _expand_k_from_h85_row_files(torch.ones(1, 1, 450, 2), class_file)
        self.assertIsInstance(attn, torch.Tensor)
        self.assertEqual(tuple(attn.shape), (1, 1, 450, 2))


if __name__ == "__main__":
    unittest.main()
