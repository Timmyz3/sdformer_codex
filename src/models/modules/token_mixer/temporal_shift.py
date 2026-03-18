"""Lightweight temporal token mixing modules."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.registry import register_module


@register_module("token_mixer", "temporal_shift")
class TemporalShiftTokenMixer(nn.Module):
    """
    Zero-parameter temporal mixing inspired by token/feature shift papers.

    Input: `[B, T, C, H, W]`
    Output: `[B, T, C, H, W]`
    """

    def __init__(self, shift_div: int = 8, mode: str = "bidirectional") -> None:
        super().__init__()
        self.shift_div = max(1, int(shift_div))
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"expected [B, T, C, H, W], got {tuple(x.shape)}")

        channels = x.shape[2]
        if channels == 0:
            return x

        fold = max(1, channels // self.shift_div)
        fold = min(fold, channels)
        out = x.clone()

        if self.mode in {"left", "bidirectional"}:
            out[:, 1:, :fold] = x[:, :-1, :fold]
        if self.mode in {"right", "bidirectional"} and fold < channels:
            upper = min(channels, 2 * fold)
            out[:, :-1, fold:upper] = x[:, 1:, fold:upper]

        return out
