"""Lightweight sparsity preprocessing for SDFormerFlow event voxels.

These operate on tensors of shape [B, T, C, H, W] (batch-first) and return
tensors of the same shape but with selected elements zeroed out.

All modules are training-aware: during training they apply stochastic sparsity
(gumbel-softmax or random masking), during eval they use deterministic thresholds.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TimestepBudget(nn.Module):
    """Zero out low-activity timesteps.

    Computes mean absolute activity per timestep, then zeroes out timesteps
    below a threshold fraction of the max activity.
    """

    def __init__(self, threshold: float = 0.02, stochastic: bool = False):
        super().__init__()
        self.threshold = threshold
        self.stochastic = stochastic

    def forward(self, chunk: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # chunk: [B, T, C, H, W]
        B, T, C, H, W = chunk.shape
        activity = chunk.abs().mean(dim=(2, 3, 4))  # [B, T]
        max_activity = activity.max(dim=1, keepdim=True).values  # [B, 1]
        keep = activity > (self.threshold * max_activity)  # [B, T]

        if self.stochastic and self.training:
            noise = torch.rand_like(activity) * 0.1
            keep = activity > (self.threshold * max_activity - noise)

        mask = keep.float().view(B, T, 1, 1, 1)
        chunk = chunk * mask
        stats = {
            "sparse_timestep_keep_ratio": keep.float().mean().item(),
            "sparse_timestep_dropped": (~keep).float().sum().item(),
        }
        return chunk, stats


class TokenPruning(nn.Module):
    """Zero out low-energy spatial tokens.

    Computes mean energy per spatial position (averaged across T and C),
    keeps the top keep_ratio fraction of tokens.
    """

    def __init__(self, keep_ratio: float = 0.75, stochastic: bool = False):
        super().__init__()
        self.keep_ratio = keep_ratio
        self.stochastic = stochastic

    def forward(self, chunk: torch.Tensor) -> tuple[torch.Tensor, dict]:
        # chunk: [B, T, C, H, W]
        B, T, C, H, W = chunk.shape
        energy = chunk.abs().mean(dim=(1, 2))  # [B, H, W]
        flat_energy = energy.view(B, -1)  # [B, H*W]
        num_keep = max(1, int(H * W * self.keep_ratio))

        if self.stochastic and self.training:
            noise = torch.rand_like(flat_energy) * 0.05
            _, indices = (flat_energy + noise).topk(num_keep, dim=1)
        else:
            _, indices = flat_energy.topk(num_keep, dim=1)

        mask = torch.zeros_like(flat_energy)
        mask.scatter_(1, indices, 1.0)
        mask = mask.view(B, 1, 1, H, W)  # [B, 1, 1, H, W]
        chunk = chunk * mask
        stats = {
            "sparse_token_keep_ratio": self.keep_ratio,
            "sparse_token_actual_keep": mask.float().mean().item(),
        }
        return chunk, stats


class WindowPruning(nn.Module):
    """Zero out low-energy spatial windows.

    Divides the spatial grid into windows of size (win_h, win_w),
    computes mean energy per window, keeps top keep_ratio windows.
    """

    def __init__(self, window_size: tuple[int, int] = (8, 8), keep_ratio: float = 0.75):
        super().__init__()
        self.window_size = window_size
        self.keep_ratio = keep_ratio

    def forward(self, chunk: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, T, C, H, W = chunk.shape
        wh, ww = self.window_size

        # Pad spatial dims to be divisible by window size
        pad_h = (wh - H % wh) % wh
        pad_w = (ww - W % ww) % ww
        if pad_h > 0 or pad_w > 0:
            chunk = torch.nn.functional.pad(chunk, (0, pad_w, 0, pad_h))
        _, _, _, H_pad, W_pad = chunk.shape

        # Reshape to windows: [B, T, C, H/wh, wh, W/ww, ww]
        chunk_win = chunk.view(B, T, C, H_pad // wh, wh, W_pad // ww, ww)
        # Window energy: mean across T, C, and within-window dims
        win_energy = chunk_win.abs().mean(dim=(1, 2, 4, 6))  # [B, H/wh, W/ww]
        num_windows = (H_pad // wh) * (W_pad // ww)
        num_keep = max(1, int(num_windows * self.keep_ratio))

        flat_energy = win_energy.view(B, -1)
        _, indices = flat_energy.topk(num_keep, dim=1)
        mask_flat = torch.zeros_like(flat_energy)
        mask_flat.scatter_(1, indices, 1.0)
        mask = mask_flat.view(B, 1, 1, H_pad // wh, 1, W_pad // ww, 1)
        # Expand mask back to full resolution
        mask = mask.expand(B, 1, 1, H_pad // wh, wh, W_pad // ww, ww)
        mask = mask.reshape(B, 1, 1, H_pad, W_pad)

        if pad_h > 0 or pad_w > 0:
            mask = mask[:, :, :, :H, :W]

        chunk = chunk[:, :, :, :H, :W] * mask
        stats = {
            "sparse_window_keep_ratio": self.keep_ratio,
            "sparse_window_actual_keep": mask.float().mean().item(),
        }
        return chunk, stats


class SparsityPipeline(nn.Module):
    """Sequential application of multiple sparsity preprocessors."""

    def __init__(self, stages: list[nn.Module]):
        super().__init__()
        self.stages = nn.ModuleList(stages)

    def forward(self, chunk: torch.Tensor) -> tuple[torch.Tensor, dict]:
        all_stats = {}
        for stage in self.stages:
            chunk, stats = stage(chunk)
            all_stats.update(stats)
        return chunk, all_stats


def build_sparsity_pipeline(config: dict) -> nn.Module | None:
    """Build sparsity preprocessing pipeline from config."""
    sparsity_cfg = config.get("sparsity", {})
    if not sparsity_cfg.get("enabled", False):
        return None

    modules = []

    ts_cfg = sparsity_cfg.get("timestep_budget", {})
    if ts_cfg.get("enabled", False):
        modules.append(TimestepBudget(
            threshold=ts_cfg.get("threshold", 0.02),
            stochastic=ts_cfg.get("stochastic", True),
        ))

    token_cfg = sparsity_cfg.get("token_pruning", {})
    if token_cfg.get("enabled", False):
        modules.append(TokenPruning(
            keep_ratio=token_cfg.get("keep_ratio", 0.75),
            stochastic=token_cfg.get("stochastic", True),
        ))

    window_cfg = sparsity_cfg.get("window_pruning", {})
    if window_cfg.get("enabled", False):
        modules.append(WindowPruning(
            window_size=tuple(window_cfg.get("window_size", [8, 8])),
            keep_ratio=window_cfg.get("keep_ratio", 0.75),
        ))

    if not modules:
        return None
    return SparsityPipeline(modules)
