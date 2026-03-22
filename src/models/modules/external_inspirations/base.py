"""Common helpers for external inspiration plug-ins."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import torch
import torch.nn as nn


@dataclass
class ExternalModuleOutput:
    tensor: torch.Tensor
    metadata: Dict[str, torch.Tensor] = field(default_factory=dict)

    def asdict(self) -> Dict[str, torch.Tensor]:
        result = {"tensor": self.tensor}
        result.update(self.metadata)
        return result


class ExternalModuleBase(nn.Module):
    """Base class for `[B, T, C, H, W]` plug-ins."""

    def _ensure_5d(self, x: torch.Tensor) -> None:
        if x.dim() != 5:
            raise ValueError(f"expected [B, T, C, H, W], got {tuple(x.shape)}")

    def _structured_topk_mask(self, scores: torch.Tensor, keep_ratio: float, min_keep: int = 1) -> torch.Tensor:
        flat = scores.reshape(*scores.shape[:2], -1)
        keep = min(flat.shape[-1], max(int(flat.shape[-1] * keep_ratio), int(min_keep)))
        topk = torch.topk(flat, k=keep, dim=-1).indices
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask.scatter_(dim=-1, index=topk, value=True)
        return mask.reshape_as(scores)

