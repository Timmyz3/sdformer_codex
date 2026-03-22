"""Token selector inspired by dynamic token retention literature."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.registry import register_module

from .base import ExternalModuleBase, ExternalModuleOutput


@register_module("external_inspirations", "motion_guided_selector")
class MotionGuidedTokenSelector(ExternalModuleBase):
    """
    Produces a training-free token mask using activity, motion, and edge proxies.

    This is a conservative front-end selector: it preserves tensor shape and emits
    masks for downstream pruners or schedulers.
    """

    def __init__(
        self,
        keep_ratio: float = 0.75,
        motion_weight: float = 0.5,
        edge_weight: float = 0.25,
        min_keep: int = 1,
    ) -> None:
        super().__init__()
        self.keep_ratio = float(keep_ratio)
        self.motion_weight = float(motion_weight)
        self.edge_weight = float(edge_weight)
        self.min_keep = int(min_keep)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._ensure_5d(x)
        energy = x.abs().mean(dim=2)
        motion = torch.zeros_like(energy)
        motion[:, 1:] = (x[:, 1:] - x[:, :-1]).abs().mean(dim=2)

        pad_h = F.pad(energy, (0, 0, 1, 0))
        pad_w = F.pad(energy, (1, 0, 0, 0))
        grad_h = (energy - pad_h[:, :, :-1]).abs()
        grad_w = (energy - pad_w[:, :, :, :-1]).abs()
        edge = grad_h + grad_w

        scores = energy + self.motion_weight * motion + self.edge_weight * edge
        mask = self._structured_topk_mask(scores, keep_ratio=self.keep_ratio, min_keep=self.min_keep)
        output = ExternalModuleOutput(
            tensor=x,
            metadata={
                "token_mask": mask,
                "selector_scores": scores,
                "selector_density": mask.float().mean(),
            },
        )
        return output.asdict()

