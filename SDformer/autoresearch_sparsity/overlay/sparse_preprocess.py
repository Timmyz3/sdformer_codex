"""Lightweight sparsity and voxel preprocessing for SDFormerFlow event voxels.

These operate on tensors of shape [B, T, C, H, W] (batch-first) and return
tensors of the same shape but with selected elements zeroed out.

All modules keep the input shape unchanged: [B, T, C, H, W]. That makes them
safe to insert before the baseline SDFormerFlow model without changing PSN
weights or the checkpoint format.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_mean_nonzero(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mask = value != 0
    if not bool(mask.any()):
        return value.new_tensor(1.0)
    return value[mask].abs().mean().clamp_min(eps)


def _rescale_like_reference(reference: torch.Tensor, transformed: torch.Tensor) -> torch.Tensor:
    """Match nonzero mean magnitude to reduce train/profile distribution shift."""
    ref_scale = _safe_mean_nonzero(reference)
    out_scale = _safe_mean_nonzero(transformed)
    return transformed * (ref_scale / out_scale)


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


class SparseSpikFormerTokenPruning(nn.Module):
    """SparseSpikFormer-style foreground token selector.

    The paper idea is to score image tokens by average spike firing/activity and
    let only high-importance foreground tokens participate in later computation.
    In this pre-model adapter we cannot shrink tensors without rewriting the
    Swin/QKFormer blocks, so we use structured zero masking with the same token
    score. This preserves checkpoint compatibility while giving a faithful
    short-test proxy for token sparsity and SOP reduction.
    """

    def __init__(
        self,
        keep_ratio: float = 0.85,
        min_keep_ratio: float = 0.20,
        window_size: tuple[int, int] | None = None,
        stochastic: bool = False,
        noise_scale: float = 0.02,
    ):
        super().__init__()
        self.keep_ratio = float(keep_ratio)
        self.min_keep_ratio = float(min_keep_ratio)
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        if not 0.0 < self.min_keep_ratio <= 1.0:
            raise ValueError(f"min_keep_ratio must be in (0, 1], got {min_keep_ratio}")
        if self.min_keep_ratio > self.keep_ratio:
            raise ValueError(
                f"min_keep_ratio ({min_keep_ratio}) cannot exceed keep_ratio ({keep_ratio})"
            )
        if window_size is not None:
            if len(window_size) != 2 or int(window_size[0]) <= 0 or int(window_size[1]) <= 0:
                raise ValueError(f"window_size must be two positive integers, got {window_size}")
            window_size = (int(window_size[0]), int(window_size[1]))
        self.window_size = window_size
        self.stochastic = bool(stochastic)
        self.noise_scale = float(noise_scale)

    def forward(self, chunk: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, T, C, H, W = chunk.shape
        activity = chunk.abs().mean(dim=(1, 2))  # [B, H, W]

        if self.window_size is not None:
            wh, ww = self.window_size
            pooled = F.avg_pool2d(activity.unsqueeze(1), kernel_size=(wh, ww), stride=(wh, ww), ceil_mode=True)
            score = F.interpolate(pooled, size=(H, W), mode="nearest").squeeze(1)
        else:
            score = activity

        flat_score = score.reshape(B, -1)
        keep_ratio = min(1.0, max(self.min_keep_ratio, self.keep_ratio))
        num_keep = max(1, int(flat_score.shape[1] * keep_ratio))
        if self.stochastic and self.training:
            flat_score = flat_score + torch.rand_like(flat_score) * self.noise_scale
        _, indices = flat_score.topk(num_keep, dim=1)
        mask = torch.zeros_like(flat_score)
        mask.scatter_(1, indices, 1.0)
        mask = mask.view(B, 1, 1, H, W)
        return chunk * mask, {
            "sparsespikformer_keep_ratio": keep_ratio,
            "sparsespikformer_actual_keep": mask.mean().item(),
        }


class QPSNNSVSPruning(nn.Module):
    """QP-SNN-inspired structured pruning by spatiotemporal activity rank.

    QP-SNN uses the singular value of spatiotemporal spike activities as a
    structured pruning criterion. Here each temporal-polarity slice is a small
    structured unit. We score a slice by spectral energy over its spatial map
    and zero low-score slices. This is deliberately coarse and hardware-friendly:
    pruning whole T x polarity planes maps to fewer dense slice reads.
    """

    def __init__(
        self,
        keep_ratio: float = 0.90,
        remove_dc: bool = False,
        preserve_dc: bool | None = None,
    ):
        super().__init__()
        self.keep_ratio = float(keep_ratio)
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
        # Backward compatibility with early configs: preserve_dc=True means do
        # not subtract the spatial DC component before scoring.
        if preserve_dc is not None:
            remove_dc = not bool(preserve_dc)
        self.remove_dc = bool(remove_dc)

    def forward(self, chunk: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, T, C, H, W = chunk.shape
        units = chunk.reshape(B, T * C, H, W)
        centered = units - units.mean(dim=(2, 3), keepdim=True) if self.remove_dc else units
        # A fast SVS proxy: Frobenius energy plus row/column concentration.
        # It tracks the dominant singular direction without calling SVD per unit.
        fro = centered.square().mean(dim=(2, 3))
        row = centered.mean(dim=3).square().mean(dim=2)
        col = centered.mean(dim=2).square().mean(dim=2)
        score = fro + row + col
        num_units = T * C
        num_keep = max(1, min(num_units, math.ceil(num_units * min(1.0, max(0.0, self.keep_ratio)))))
        _, indices = score.topk(num_keep, dim=1)
        mask = torch.zeros_like(score)
        mask.scatter_(1, indices, 1.0)
        mask = mask.view(B, T, C, 1, 1)
        return chunk * mask, {
            "qpsnn_svs_keep_ratio": self.keep_ratio,
            "qpsnn_svs_actual_keep": mask.mean().item(),
            "qpsnn_svs_remove_dc": float(self.remove_dc),
        }


class TemporalDifferenceVoxel(nn.Module):
    """EDCFlow-style temporal difference voxel adapter.

    It keeps the original voxel stream as the carrier and injects adjacent-bin
    temporal differences, then rescales magnitude to stay close to baseline
    minmax-normalized input statistics.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        mode: str = "residual",
        rescale: bool = True,
        clamp: bool = True,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.mode = str(mode)
        self.rescale = bool(rescale)
        self.clamp = bool(clamp)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def forward(self, chunk: torch.Tensor) -> tuple[torch.Tensor, dict]:
        diff = torch.zeros_like(chunk)
        diff[:, 1:] = chunk[:, 1:] - chunk[:, :-1]
        if self.mode == "diff":
            out = diff
        elif self.mode == "absdiff":
            out = diff.abs()
        elif self.mode == "residual_abs":
            out = chunk + self.alpha * diff.abs()
        else:
            out = chunk + self.alpha * diff
        if self.rescale:
            out = _rescale_like_reference(chunk, out)
        if self.clamp:
            out = out.clamp(min=self.clamp_min, max=self.clamp_max)
        return out, {
            "temporal_diff_alpha": self.alpha,
            "temporal_diff_abs_mean": diff.abs().mean().item(),
            "temporal_diff_clamp": float(self.clamp),
        }


class EventPillarsLite(nn.Module):
    """EventPillars-inspired dense feature adapter with unchanged shape.

    EventPillars uses temporal range, polarity activity, and event density. This
    lightweight version derives those cues from precomputed voxels and blends
    them back into the original tensor.
    """

    def __init__(
        self,
        density_alpha: float = 0.15,
        range_alpha: float = 0.10,
        rescale: bool = True,
        preserve_zero: bool = True,
        clamp: bool = True,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
    ):
        super().__init__()
        self.density_alpha = float(density_alpha)
        self.range_alpha = float(range_alpha)
        self.rescale = bool(rescale)
        self.preserve_zero = bool(preserve_zero)
        self.clamp = bool(clamp)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

    def forward(self, chunk: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, T, C, H, W = chunk.shape
        density = chunk.abs().mean(dim=1, keepdim=True)  # [B,1,C,H,W]
        active = chunk.abs() > 0
        time_index = torch.linspace(0, 1, T, device=chunk.device, dtype=chunk.dtype).view(1, T, 1, 1, 1)
        masked_time = torch.where(active, time_index, torch.zeros_like(time_index))
        t_max = masked_time.max(dim=1, keepdim=True).values
        t_min = torch.where(active, time_index, torch.ones_like(time_index)).min(dim=1, keepdim=True).values
        temporal_range = (t_max - t_min).clamp_min(0.0)
        additive = self.density_alpha * density + self.range_alpha * temporal_range
        if self.preserve_zero:
            additive = additive * active.float()
        out = chunk + additive
        if self.rescale:
            out = _rescale_like_reference(chunk, out)
        if self.clamp:
            out = out.clamp(min=self.clamp_min, max=self.clamp_max)
        if self.preserve_zero:
            out = torch.where(active, out, torch.zeros_like(out))
        return out, {
            "eventpillars_density_mean": density.mean().item(),
            "eventpillars_range_mean": temporal_range.mean().item(),
            "eventpillars_preserve_zero": float(self.preserve_zero),
            "eventpillars_clamp": float(self.clamp),
        }


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

    ssf_cfg = sparsity_cfg.get("sparsespikformer_token_pruning", {})
    if ssf_cfg.get("enabled", False):
        window_size = ssf_cfg.get("window_size")
        modules.append(SparseSpikFormerTokenPruning(
            keep_ratio=ssf_cfg.get("keep_ratio", 0.85),
            min_keep_ratio=ssf_cfg.get("min_keep_ratio", 0.20),
            window_size=tuple(window_size) if window_size else None,
            stochastic=ssf_cfg.get("stochastic", True),
            noise_scale=ssf_cfg.get("noise_scale", 0.02),
        ))

    qpsnn_cfg = sparsity_cfg.get("qpsnn_svs_pruning", {})
    if qpsnn_cfg.get("enabled", False):
        modules.append(QPSNNSVSPruning(
            keep_ratio=qpsnn_cfg.get("keep_ratio", 0.90),
            remove_dc=qpsnn_cfg.get("remove_dc", False),
            preserve_dc=qpsnn_cfg.get("preserve_dc", None),
        ))

    voxel_cfg = sparsity_cfg.get("voxel_adapter", {})
    if voxel_cfg.get("enabled", False):
        method = voxel_cfg.get("method", "temporal_diff")
        if method in {"temporal_diff", "edcflow_temporal_diff"}:
            modules.append(TemporalDifferenceVoxel(
                alpha=voxel_cfg.get("alpha", 0.25),
                mode=voxel_cfg.get("mode", "residual"),
                rescale=voxel_cfg.get("rescale", True),
                clamp=voxel_cfg.get("clamp", True),
                clamp_min=voxel_cfg.get("clamp_min", 0.0),
                clamp_max=voxel_cfg.get("clamp_max", 1.0),
            ))
        elif method == "eventpillars_lite":
            modules.append(EventPillarsLite(
                density_alpha=voxel_cfg.get("density_alpha", 0.15),
                range_alpha=voxel_cfg.get("range_alpha", 0.10),
                rescale=voxel_cfg.get("rescale", True),
                preserve_zero=voxel_cfg.get("preserve_zero", True),
                clamp=voxel_cfg.get("clamp", True),
                clamp_min=voxel_cfg.get("clamp_min", 0.0),
                clamp_max=voxel_cfg.get("clamp_max", 1.0),
            ))
        else:
            raise ValueError(f"Unknown sparsity.voxel_adapter.method: {method}")

    if not modules:
        return None
    return SparsityPipeline(modules)
