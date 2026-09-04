"""Timestep-level sparse gates."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from src.models.registry import register_module


@register_module("sparse_ops", "timestep_budget")
class TimestepBudgetPruner(nn.Module):
    """
    Keeps active timesteps according to a threshold or a top-k budget.

    Input: `[B, T, C, H, W]`
    Output:
        `tensor`: masked tensor with the same shape
        `timestep_mask`: `[B, T]`
    """

    def __init__(
        self,
        keep_ratio: float | None = None,
        threshold: float | None = None,
        min_keep: int = 1,
    ) -> None:
        super().__init__()
        self.keep_ratio = None if keep_ratio is None else float(keep_ratio)
        self.threshold = None if threshold is None else float(threshold)
        self.min_keep = max(1, int(min_keep))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() != 5:
            raise ValueError(f"expected [B, T, C, H, W], got {tuple(x.shape)}")

        scores = x.abs().mean(dim=(2, 3, 4))
        steps = scores.shape[1]
        keep = max(self.min_keep, int(steps * self.keep_ratio)) if self.keep_ratio is not None else self.min_keep

        if self.threshold is not None:
            mask = scores > self.threshold
            topk = torch.topk(scores, k=min(steps, keep), dim=1).indices
            topk_mask = torch.zeros_like(mask)
            topk_mask.scatter_(dim=1, index=topk, value=True)
            mask = mask | topk_mask
        else:
            topk = torch.topk(scores, k=min(steps, keep), dim=1).indices
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(dim=1, index=topk, value=True)

        masked = x * mask[:, :, None, None, None]
        return {"tensor": masked, "timestep_mask": mask}
