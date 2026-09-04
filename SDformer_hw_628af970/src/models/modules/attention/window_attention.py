"""Attention descriptors used by config-driven model variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.models.registry import register_module


@register_module("attention", "baseline")
@dataclass
class BaselineAttentionSpec:
    window_size: Sequence[int] = (2, 9, 9)


@register_module("attention", "window_spike")
@dataclass
class WindowSpikeAttentionSpec:
    window_size: Sequence[int] = (2, 8, 8)
    hardware_friendly: bool = True

