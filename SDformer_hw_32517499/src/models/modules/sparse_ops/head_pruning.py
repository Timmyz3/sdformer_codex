"""Head-group sparse gates."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from src.models.registry import register_module


@register_module("sparse_ops", "head_group")
class HeadGroupPruner(nn.Module):
    """
    Approximates head pruning by gating channel groups.

    Input: `[B, T, C, H, W]`
    Output:
        `tensor`: gated tensor with the same shape
        `head_mask`: `[B, T, G]`
    """

    def __init__(self, keep_ratio: float = 1.0, num_groups: int = 1, min_keep: int = 1) -> None:
        super().__init__()
        self.keep_ratio = float(keep_ratio)
        self.num_groups = max(1, int(num_groups))
        self.min_keep = max(1, int(min_keep))

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() != 5:
            raise ValueError(f"expected [B, T, C, H, W], got {tuple(x.shape)}")

        num_groups = min(self.num_groups, x.shape[2])
        groups = torch.tensor_split(x, num_groups, dim=2)
        scores = torch.stack([group.abs().mean(dim=(2, 3, 4)) for group in groups], dim=-1)
        keep = max(self.min_keep, int(scores.shape[-1] * self.keep_ratio))
        keep = min(keep, scores.shape[-1])

        topk = torch.topk(scores, k=keep, dim=-1).indices
        head_mask = torch.zeros_like(scores, dtype=torch.bool)
        head_mask.scatter_(dim=-1, index=topk, value=True)

        gated_groups = []
        for group_idx, group in enumerate(groups):
            gate = head_mask[..., group_idx][:, :, None, None, None]
            gated_groups.append(group * gate)
        gated = torch.cat(gated_groups, dim=2)
        return {"tensor": gated, "head_mask": head_mask}
