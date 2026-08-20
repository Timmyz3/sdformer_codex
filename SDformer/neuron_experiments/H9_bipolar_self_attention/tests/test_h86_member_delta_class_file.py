#!/usr/bin/env python3
"""H86: window class-major + member insert/delete as the expand object."""

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
    _build_h86_member_delta_file,
    _class_major_shiftmax_gate,
    _expand_k_from_h86_member_delta,
)


def _cfg() -> ShiftmaxAttentionConfig:
    return ShiftmaxAttentionConfig(
        hardware_score_step=1.0 / 128.0,
        hardware_score_min=-2.0,
        hardware_score_max=2.0,
        preserve_mean=False,
        eps=1.0e-6,
    )


def _grid_scores(values: torch.Tensor) -> torch.Tensor:
    """values [T,S,S] -> scores [1,1,450,1]."""

    return values.reshape(1, 1, -1, 1)


class MemberDeltaClassFileTests(unittest.TestCase):
    def test_file_has_no_t450_or_513_operands(self) -> None:
        scores = torch.zeros(1, 1, 450, 1)
        class_file = _build_h86_member_delta_file(scores, _cfg())
        forbidden = {
            "codes",
            "token_gate",
            "member_mask",
            "member_idx",
            "shared_ids",
            "insert_ids",
            "delete_ids",
            "occupied",
            "n_bins",
            "reuse_set",
        }
        for key in forbidden:
            self.assertNotIn(key, class_file)
        for key, value in class_file.items():
            if torch.is_tensor(value) and value.ndim > 0:
                self.assertNotEqual(int(value.shape[-1]), 513, msg=key)
        self.assertEqual(class_file["row0_member_idx"].shape[-1], 15)
        self.assertEqual(class_file["member_insert"].shape[-1], 15)
        self.assertEqual(class_file["member_delete"].shape[-1], 15)
        self.assertEqual(
            tuple(class_file["member_insert"].shape[-4:]),
            (2, 14, class_file["class_id"].shape[-1], 15),
        )

    def test_expand_forbids_t450_and_513(self) -> None:
        scores = torch.zeros(1, 1, 450, 1)
        class_file = _build_h86_member_delta_file(scores, _cfg())
        k = torch.ones(1, 1, 450, 2)
        for key in ("codes", "token_gate", "member_mask", "member_idx"):
            bad = dict(class_file)
            bad[key] = torch.zeros(1, 1, 450)
            with self.assertRaises(RuntimeError):
                _expand_k_from_h86_member_delta(k, bad)
        bad = dict(class_file)
        bad["occupied"] = torch.zeros(1, 1, 513)
        with self.assertRaises(RuntimeError):
            _expand_k_from_h86_member_delta(k, bad)

    def test_expand_matches_window_class_major(self) -> None:
        codes = torch.zeros(2, 15, 15)
        codes[:, :, :4] = 8
        codes[:, :, 4:] = 24
        scores = _grid_scores(codes / 128.0)
        class_file = _build_h86_member_delta_file(scores, _cfg())
        k = torch.randn(1, 1, 450, 3)
        attn = _expand_k_from_h86_member_delta(k, class_file)
        token_gate, _stats = _class_major_shiftmax_gate(scores, _cfg())
        self.assertTrue(torch.allclose(attn, k * token_gate, atol=1.0e-5, rtol=1.0e-5))

    def test_insert_is_executed(self) -> None:
        codes = torch.zeros(2, 15, 15)
        codes[:, :, :3] = 8
        scores = _grid_scores(codes / 128.0)
        class_file = _build_h86_member_delta_file(scores, _cfg())
        k = torch.ones(1, 1, 450, 2)
        attn_ref = _expand_k_from_h86_member_delta(k, class_file)
        row0_idx = class_file["row0_member_idx"][0, 0, 0]
        row0_ok = class_file["row0_member_ok"][0, 0, 0]
        slot = int(((row0_idx == 0) & row0_ok).any(dim=-1).nonzero(as_tuple=False).flatten()[0])
        mutated = dict(class_file)
        mutated["member_insert"] = class_file["member_insert"].clone()
        mutated["member_insert_ok"] = class_file["member_insert_ok"].clone()
        mutated["member_insert"][0, 0, 0, 0, slot, 0] = 7
        mutated["member_insert_ok"][0, 0, 0, 0, slot, 0] = True
        attn_mut = _expand_k_from_h86_member_delta(k, mutated)
        self.assertGreater(float((attn_ref - attn_mut).abs().max()), 1.0e-6)

    def test_surviving_member_jaccard_not_class_name_reuse(self) -> None:
        codes = torch.zeros(2, 15, 15)
        codes[:, :, :] = 16
        codes[:, 0, :6] = 4
        codes[:, 1, 3:9] = 4
        scores = _grid_scores(codes / 128.0)
        class_file = _build_h86_member_delta_file(scores, _cfg())
        self.assertTrue(bool(class_file["class_shared"].any()))
        self.assertLess(float(class_file["member_jaccard_surviving"]), 0.99)
        self.assertGreater(float(class_file["member_jaccard_surviving"]), 0.0)

    def test_rows_after_zero_have_no_full_member_table(self) -> None:
        scores = torch.zeros(1, 1, 450, 1)
        class_file = _build_h86_member_delta_file(scores, _cfg())
        self.assertIn("row0_member_idx", class_file)
        self.assertNotIn("member_idx", class_file)
        self.assertEqual(class_file["row0_member_idx"].shape[3], class_file["class_id"].shape[-1])


if __name__ == "__main__":
    unittest.main()
