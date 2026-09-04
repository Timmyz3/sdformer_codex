"""Token pruning modules inspired by graph-based token importance methods."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.registry import register_module

from .base import ExternalModuleBase, ExternalModuleOutput


@register_module("external_inspirations", "graph_token_pruner")
class GraphImportanceTokenPruner(ExternalModuleBase):
    """
    A training-free structured token pruner that approximates attention-graph
    importance with local spatial-temporal message passing.
    """

    def __init__(
        self,
        keep_ratio: float = 0.6,
        self_weight: float = 0.6,
        neighborhood_weight: float = 0.4,
        min_keep: int = 1,
    ) -> None:
        super().__init__()
        self.keep_ratio = float(keep_ratio)
        self.self_weight = float(self_weight)
        self.neighborhood_weight = float(neighborhood_weight)
        self.min_keep = int(min_keep)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._ensure_5d(x)
        energy = x.abs().mean(dim=2)

        spatial_context = F.avg_pool2d(
            energy.reshape(-1, 1, energy.shape[-2], energy.shape[-1]),
            kernel_size=3,
            stride=1,
            padding=1,
        ).reshape_as(energy)

        temporal_context = energy.clone()
        temporal_context[:, 1:] += energy[:, :-1]
        temporal_context[:, :-1] += energy[:, 1:]
        temporal_context = temporal_context / 3.0

        scores = self.self_weight * energy + self.neighborhood_weight * 0.5 * (spatial_context + temporal_context)
        mask = self._structured_topk_mask(scores, keep_ratio=self.keep_ratio, min_keep=self.min_keep)
        pruned = x * mask[:, :, None]
        output = ExternalModuleOutput(
            tensor=pruned,
            metadata={
                "token_mask": mask,
                "importance_scores": scores,
                "token_keep_ratio_realized": mask.float().mean(),
            },
        )
        return output.asdict()

