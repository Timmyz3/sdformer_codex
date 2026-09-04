"""Placeholder token mixer interface."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.registry import register_module


@register_module("token_mixer", "identity")
class IdentityTokenMixer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
