"""Attention reuse approximations for event streams."""

from __future__ import annotations

import torch

from src.models.registry import register_module

from .base import ExternalModuleBase, ExternalModuleOutput


@register_module("external_inspirations", "temporal_attention_reuse")
class TemporalAttentionReuse(ExternalModuleBase):
    """
    Reuses previous-step features in low-motion regions to approximate attention reuse.
    """

    def __init__(self, reuse_threshold: float = 0.02, momentum: float = 0.8) -> None:
        super().__init__()
        self.reuse_threshold = float(reuse_threshold)
        self.momentum = float(momentum)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._ensure_5d(x)
        energy = x.abs().mean(dim=2)
        motion = torch.zeros_like(energy)
        motion[:, 1:] = (x[:, 1:] - x[:, :-1]).abs().mean(dim=2)

        reuse_mask = motion <= self.reuse_threshold
        output = x.clone()
        for step in range(1, x.shape[1]):
            mask = reuse_mask[:, step][:, None]
            reused = self.momentum * output[:, step - 1] + (1.0 - self.momentum) * x[:, step]
            output[:, step] = torch.where(mask, reused, x[:, step])

        result = ExternalModuleOutput(
            tensor=output,
            metadata={
                "reuse_mask": reuse_mask,
                "reuse_ratio": reuse_mask.float().mean(),
            },
        )
        return result.asdict()

