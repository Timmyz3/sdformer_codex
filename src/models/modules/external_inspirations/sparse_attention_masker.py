"""Sparse attention masking helpers."""

from __future__ import annotations

import math
from typing import Sequence

import torch

from src.models.registry import register_module

from .base import ExternalModuleBase, ExternalModuleOutput


@register_module("external_inspirations", "block_sparse_attention_masker")
class BlockSparseAttentionMasker(ExternalModuleBase):
    """
    Builds a fixed-radius block-sparse attention mask over windows.

    The mask is emitted as metadata for downstream attention layers or hardware
    simulators. The input tensor is left unchanged.
    """

    def __init__(self, window_size: Sequence[int] = (8, 8), radius: int = 1) -> None:
        super().__init__()
        if len(window_size) != 2:
            raise ValueError("window_size must have 2 elements")
        self.window_size = (int(window_size[0]), int(window_size[1]))
        self.radius = int(radius)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._ensure_5d(x)
        batch, steps, _, height, width = x.shape
        win_h, win_w = self.window_size
        windows_h = int(math.ceil(height / win_h))
        windows_w = int(math.ceil(width / win_w))
        num_windows = windows_h * windows_w

        coords = torch.cartesian_prod(
            torch.arange(windows_h, device=x.device),
            torch.arange(windows_w, device=x.device),
        )
        delta = coords[:, None, :] - coords[None, :, :]
        local = (delta.abs().max(dim=-1).values <= self.radius)
        attention_mask = local[None, None].expand(batch, steps, num_windows, num_windows)
        output = ExternalModuleOutput(
            tensor=x,
            metadata={
                "attention_mask": attention_mask,
                "attention_sparsity": 1.0 - attention_mask.float().mean(),
            },
        )
        return output.asdict()
