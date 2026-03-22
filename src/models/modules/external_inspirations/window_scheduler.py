"""Structured window scheduling modules."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

from src.models.registry import register_module

from .base import ExternalModuleBase, ExternalModuleOutput


@register_module("external_inspirations", "activity_window_scheduler")
class ActivityWindowScheduler(ExternalModuleBase):
    """
    Structured window scheduler with temporal hysteresis to reduce flicker.
    """

    def __init__(
        self,
        window_size: Sequence[int] = (8, 8),
        keep_ratio: float = 0.75,
        hysteresis: float = 0.5,
        min_keep: int = 1,
    ) -> None:
        super().__init__()
        if len(window_size) != 2:
            raise ValueError("window_size must have 2 elements")
        self.window_size = (int(window_size[0]), int(window_size[1]))
        self.keep_ratio = float(keep_ratio)
        self.hysteresis = float(hysteresis)
        self.min_keep = int(min_keep)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._ensure_5d(x)
        batch, steps, _, height, width = x.shape
        win_h, win_w = self.window_size
        padded_h = int(math.ceil(height / win_h) * win_h)
        padded_w = int(math.ceil(width / win_w) * win_w)

        energy = x.abs().mean(dim=2)
        if padded_h != height or padded_w != width:
            energy = F.pad(energy, (0, padded_w - width, 0, padded_h - height))

        pooled = F.avg_pool2d(
            energy.reshape(batch * steps, 1, padded_h, padded_w),
            kernel_size=self.window_size,
            stride=self.window_size,
        ).reshape(batch, steps, padded_h // win_h, padded_w // win_w)

        smoothed = pooled.clone()
        if steps > 1:
            smoothed[:, 1:] = torch.maximum(smoothed[:, 1:], self.hysteresis * pooled[:, :-1])
            smoothed[:, :-1] = torch.maximum(smoothed[:, :-1], self.hysteresis * pooled[:, 1:])

        mask = self._structured_topk_mask(smoothed, keep_ratio=self.keep_ratio, min_keep=self.min_keep)
        token_mask = mask.repeat_interleave(win_h, dim=2).repeat_interleave(win_w, dim=3)
        token_mask = token_mask[:, :, :height, :width]
        output = ExternalModuleOutput(
            tensor=x * token_mask[:, :, None],
            metadata={
                "window_mask": mask,
                "token_mask": token_mask,
                "scheduled_window_density": mask.float().mean(),
            },
        )
        return output.asdict()

