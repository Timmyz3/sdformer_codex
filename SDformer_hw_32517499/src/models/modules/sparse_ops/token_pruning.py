"""Structured sparsity helpers."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from src.models.registry import register_module


@register_module("sparse_ops", "structured_token")
class StructuredTokenPruner(nn.Module):
    """
    Applies deterministic top-k pruning over flattened spatial tokens per timestep.

    Input: `[B, T, C, H, W]`
    Output:
        `tensor`: pruned tensor with the same shape
        `mask`: `[B, T, H, W]`
    """

    def __init__(self, keep_ratio: float = 1.0) -> None:
        super().__init__()
        self.keep_ratio = float(keep_ratio)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, steps, chans, height, width = x.shape
        tokens = x.abs().mean(dim=2).reshape(bsz, steps, height * width)
        keep = max(1, int(tokens.shape[-1] * self.keep_ratio))
        topk = torch.topk(tokens, k=keep, dim=-1).indices
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        mask.scatter_(dim=-1, index=topk, value=True)
        mask_2d = mask.reshape(bsz, steps, height, width)
        pruned = x * mask_2d[:, :, None]
        return {"tensor": pruned, "mask": mask_2d, "token_mask": mask_2d}


@register_module("sparse_ops", "activity_stats")
class ActivityStats(nn.Module):
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        active = x != 0
        return {
            "density": active.float().mean(),
            "per_timestep": active.float().mean(dim=(0, 2, 3, 4)),
        }
