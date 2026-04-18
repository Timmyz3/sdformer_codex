"""Input spike encoders used before the upstream baseline."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from src.models.registry import register_module


@register_module("spike_encoding", "voxel")
class VoxelSpikeEncoder(nn.Module):
    """
    Pass-through encoder for polarity-split voxel tensors.

    Input: `[B, T, 2, H, W]`
    Output: `[B, T, 2, H, W]`
    """

    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        super().__init__()
        self.cfg = cfg or {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


@register_module("spike_encoding", "temporal_contrast")
class TemporalContrastEncoder(nn.Module):
    """
    Emphasizes temporal changes between adjacent event bins.

    Input: `[B, T, 2, H, W]`
    Output: `[B, T, 2, H, W]`
    """

    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        super().__init__()
        self.cfg = cfg or {}
        self.eps = float(self.cfg.get("temporal_contrast_eps", 1.0e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(x)
        delta[:, 1:] = x[:, 1:] - x[:, :-1]
        return delta / (x.abs().amax(dim=(2, 3, 4), keepdim=True) + self.eps)


@register_module("spike_encoding", "latency")
class LatencySpikeEncoder(nn.Module):
    """
    Converts activity magnitude into a monotonic latency-like code.

    Input: `[B, T, 2, H, W]`
    Output: `[B, T, 2, H, W]`
    """

    def __init__(self, cfg: Dict[str, Any] | None = None) -> None:
        super().__init__()
        self.cfg = cfg or {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ranking = torch.argsort(torch.argsort(-x.abs(), dim=1), dim=1)
        return 1.0 / (ranking.to(x.dtype) + 1.0)

