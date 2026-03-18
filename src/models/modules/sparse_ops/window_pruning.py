"""Window-level sparse gates."""

from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.registry import register_module


@register_module("sparse_ops", "window_topk")
class WindowTopKPruner(nn.Module):
    """
    Applies deterministic top-k pruning over local spatial windows per timestep.

    Input: `[B, T, C, H, W]`
    Output:
        `tensor`: pruned tensor with the same shape
        `token_mask`: `[B, T, H, W]`
        `window_mask`: `[B, T, H_w, W_w]`
    """

    def __init__(self, keep_ratio: float = 1.0, window_size: Sequence[int] = (8, 8), min_keep: int = 1) -> None:
        super().__init__()
        if len(window_size) != 2:
            raise ValueError("window_size must be a 2-element sequence")
        self.keep_ratio = float(keep_ratio)
        self.window_size = (int(window_size[0]), int(window_size[1]))
        self.min_keep = max(1, int(min_keep))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() != 5:
            raise ValueError(f"expected [B, T, C, H, W], got {tuple(x.shape)}")

        batch, steps, _, height, width = x.shape
        win_h, win_w = self.window_size
        padded_h = int(math.ceil(height / win_h) * win_h)
        padded_w = int(math.ceil(width / win_w) * win_w)

        energy = x.abs().mean(dim=2)
        if padded_h != height or padded_w != width:
            energy = F.pad(energy, (0, padded_w - width, 0, padded_h - height))

        pooled = F.avg_pool2d(
            energy.reshape(batch * steps, 1, padded_h, padded_w),
            kernel_size=self.window_size,
            stride=self.window_size,
        )
        pooled = pooled.reshape(batch, steps, -1)
        keep = max(self.min_keep, int(pooled.shape[-1] * self.keep_ratio))
        keep = min(keep, pooled.shape[-1])

        topk = torch.topk(pooled, k=keep, dim=-1).indices
        window_mask = torch.zeros_like(pooled, dtype=torch.bool)
        window_mask.scatter_(dim=-1, index=topk, value=True)
        window_mask = window_mask.reshape(batch, steps, padded_h // win_h, padded_w // win_w)

        token_mask = window_mask.repeat_interleave(win_h, dim=2).repeat_interleave(win_w, dim=3)
        token_mask = token_mask[:, :, :height, :width]
        pruned = x * token_mask[:, :, None]
        return {"tensor": pruned, "token_mask": token_mask, "window_mask": window_mask}
