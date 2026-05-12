"""Shiftmax: hardware-friendly softmax replacement for ternary spiking attention.

BSA (NeurIPS 2025) proposes Shiftmax as a bit-shift-based softmax approximation
for bipolar spiking attention. Our version simplifies to clamp + normalize:

  shiftmax(x) = clamp(x, 0) / sum(clamp(x, 0))

Hardware: 1 comparator (clamp) + adder tree (sum) + divider. No exponentiation.
"""

import torch
import torch.nn as nn


def shiftmax(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """Clamp negative scores to zero, then L1-normalize.

    Args:
        x: attention scores [..., N, N]
        dim: normalization dimension
        eps: numerical stability
    """
    pos = torch.clamp(x, min=0.0)
    denom = pos.sum(dim=dim, keepdim=True) + eps
    return pos / denom


class Shiftmax(nn.Module):
    """Module wrapper for shiftmax with optional temperature."""

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return shiftmax(x, dim=self.dim)
