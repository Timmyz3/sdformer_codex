"""Budget controllers for structured sparsity experiments."""

from __future__ import annotations

import torch

from src.models.registry import register_module

from .base import ExternalModuleBase, ExternalModuleOutput


@register_module("external_inspirations", "structured_latency_controller")
class StructuredLatencyPruningController(ExternalModuleBase):
    """
    Emits budget recommendations using simple workload-aware heuristics.
    """

    def __init__(
        self,
        target_token_keep_ratio: float = 0.6,
        target_window_keep_ratio: float = 0.75,
        min_timestep_keep_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.target_token_keep_ratio = float(target_token_keep_ratio)
        self.target_window_keep_ratio = float(target_window_keep_ratio)
        self.min_timestep_keep_ratio = float(min_timestep_keep_ratio)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        self._ensure_5d(x)
        timestep_activity = x.abs().mean(dim=(2, 3, 4))
        normalized = timestep_activity / timestep_activity.amax(dim=1, keepdim=True).clamp_min(1e-6)
        timestep_mask = normalized >= self.min_timestep_keep_ratio

        result = ExternalModuleOutput(
            tensor=x,
            metadata={
                "timestep_mask": timestep_mask,
                "recommended_token_keep_ratio": x.new_tensor(self.target_token_keep_ratio),
                "recommended_window_keep_ratio": x.new_tensor(self.target_window_keep_ratio),
                "controller_active_steps": timestep_mask.float().mean(),
            },
        )
        return result.asdict()

