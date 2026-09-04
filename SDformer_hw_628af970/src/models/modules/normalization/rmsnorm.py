"""RMSNorm implementation used by the project variants."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
import torch.nn as nn

from src.models.registry import register_module


@register_module("normalization", "RMSNorm")
class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps: float = 1.0e-6) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape: Tuple[int, ...] = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        variance = x.pow(2).mean(dim=dims, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight
