"""Token merging modules inspired by token fusion literature."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

from src.models.registry import register_module

from .base import ExternalModuleBase, ExternalModuleOutput


@register_module("external_inspirations", "similarity_token_merger")
class SimilarityTokenMerger(ExternalModuleBase):
    """
    Replaces low-activity windows with their local centroid while preserving shape.

    The current SDFormer hook requires fixed `[B, T, C, H, W]` shape, so this
    module models merge behavior without changing token count yet.
    """

    def __init__(
        self,
        window_size: Sequence[int] = (4, 4),
        merge_ratio: float = 0.5,
        blend: float = 1.0,
    ) -> None:
        super().__init__()
        if len(window_size) != 2:
            raise ValueError("window_size must have 2 elements")
        self.window_size = (int(window_size[0]), int(window_size[1]))
        self.merge_ratio = float(merge_ratio)
        self.blend = float(blend)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._ensure_5d(x)
        batch, steps, channels, height, width = x.shape
        win_h, win_w = self.window_size
        padded_h = int(math.ceil(height / win_h) * win_h)
        padded_w = int(math.ceil(width / win_w) * win_w)

        padded = x
        if padded_h != height or padded_w != width:
            padded = F.pad(x, (0, padded_w - width, 0, padded_h - height))

        energy = padded.abs().mean(dim=2)
        windows_h = padded_h // win_h
        windows_w = padded_w // win_w
        window_energy = F.avg_pool2d(
            energy.reshape(batch * steps, 1, padded_h, padded_w),
            kernel_size=self.window_size,
            stride=self.window_size,
        ).reshape(batch, steps, windows_h, windows_w)

        flat = window_energy.reshape(batch, steps, -1)
        merge_windows = max(1, int(flat.shape[-1] * self.merge_ratio))
        lowk = torch.topk(flat, k=merge_windows, dim=-1, largest=False).indices
        merge_mask = torch.zeros_like(flat, dtype=torch.bool)
        merge_mask.scatter_(dim=-1, index=lowk, value=True)
        merge_mask = merge_mask.reshape(batch, steps, windows_h, windows_w)

        windows = padded.reshape(batch, steps, channels, windows_h, win_h, windows_w, win_w)
        windows = windows.permute(0, 1, 3, 5, 2, 4, 6)
        centroids = windows.mean(dim=(-1, -2), keepdim=True)
        merged = torch.where(
            merge_mask[:, :, :, :, None, None, None],
            (1.0 - self.blend) * windows + self.blend * centroids,
            windows,
        )
        merged = merged.permute(0, 1, 4, 2, 5, 3, 6).reshape(batch, steps, channels, padded_h, padded_w)
        merged = merged[:, :, :, :height, :width]
        token_mask = (~merge_mask).repeat_interleave(win_h, dim=2).repeat_interleave(win_w, dim=3)
        token_mask = token_mask[:, :, :height, :width]

        output = ExternalModuleOutput(
            tensor=merged,
            metadata={
                "window_mask": ~merge_mask,
                "token_mask": token_mask,
                "merge_ratio_realized": merge_mask.float().mean(),
            },
        )
        return output.asdict()

