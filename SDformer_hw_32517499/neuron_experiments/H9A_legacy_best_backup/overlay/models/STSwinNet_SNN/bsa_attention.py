"""H9 Shiftmax compatibility layer for SDFormerFlow QK attention.

This is a paper-formula reproduction of the BSA Shiftmax idea, adapted as a
minimal compatibility gate around SDFormerFlow's existing QK attention path.
It does not replace the baseline attention module with a full QK^T V operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Any, Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ShiftmaxAttentionConfig:
    enabled: bool = False
    stage_selection: str = "all"
    target_blocks: tuple[str, ...] = ()
    mode: str = "compat_qk_product"
    score_scale: float = 1.0
    center_scores: bool = True
    preserve_mean: bool = True
    eps: float = 1.0e-6


def config_from_dict(raw: dict | None) -> ShiftmaxAttentionConfig:
    raw = raw or {}
    target_blocks = raw.get("target_blocks", ())
    if isinstance(target_blocks, str):
        target_blocks = [target_blocks]
    return ShiftmaxAttentionConfig(
        enabled=bool(raw.get("enabled", False)),
        stage_selection=str(raw.get("stage_selection", "all")),
        target_blocks=tuple(str(item) for item in target_blocks),
        mode=str(raw.get("mode", "compat_qk_product")),
        score_scale=float(raw.get("score_scale", 1.0)),
        center_scores=bool(raw.get("center_scores", True)),
        preserve_mean=bool(raw.get("preserve_mean", True)),
        eps=float(raw.get("eps", 1.0e-6)),
    )


def shiftmax(scores: torch.Tensor, dim: int = -1, eps: float = 1.0e-6) -> torch.Tensor:
    """Shiftmax from BSA: 2^x normalized by the next power-of-two row sum.

    The returned row sum is bounded by (0.5, 1] before any optional rescaling.
    """

    shifted = scores - scores.amax(dim=dim, keepdim=True)
    numerator = torch.pow(2.0, shifted)
    denom_power = torch.ceil(torch.log2(numerator.sum(dim=dim, keepdim=True).clamp_min(eps)))
    denominator = torch.pow(2.0, denom_power)
    return numerator / denominator


def _iter_attention_modules(model: nn.Module, cfg: ShiftmaxAttentionConfig) -> Iterable[tuple[str, nn.Module]]:
    if not hasattr(model, "sttmultires_unet"):
        return []
    swin3d = model.sttmultires_unet.encoders.swin3d
    layers = list(swin3d.layers)
    if cfg.target_blocks:
        wanted = set(cfg.target_blocks)
        pairs = []
        found: set[str] = set()
        for stage_idx, stage in enumerate(layers):
            for block_idx, block in enumerate(stage.swin_blocks):
                key = f"{stage_idx}:{block_idx}"
                if key in wanted:
                    found.add(key)
                    pairs.append((f"layers.{stage_idx}.swin_blocks.{block_idx}.attn", block.attn))
        missing = wanted - found
        if missing:
            raise KeyError(f"Could not find H9 target_blocks: {sorted(missing)}")
        return pairs
    if cfg.stage_selection == "all":
        stage_ids = range(len(layers))
    elif cfg.stage_selection.startswith("stage"):
        stage_ids = [int(cfg.stage_selection.replace("stage", ""))]
    else:
        raise ValueError("bsa_attention.stage_selection must be all or stage{index}")
    pairs = []
    for stage_idx in stage_ids:
        for block_idx, block in enumerate(layers[stage_idx].swin_blocks):
            pairs.append((f"layers.{stage_idx}.swin_blocks.{block_idx}.attn", block.attn))
    return pairs


def _qk_shiftmax_gate_forward(self, x, mask=None):
    """Patched forward for Spiking_QK_WindowAttention3D.

    It preserves the original QK attention carrier and adds a Shiftmax gate
    computed from signed Q/K token compatibility.
    """

    del mask
    cfg: ShiftmaxAttentionConfig = self._h9_shiftmax_cfg
    T, B_, H, W, C = x.shape
    head_dim = C // self.num_heads

    x = self.proj_sn(x.float())
    q = self.linear_q(x)
    if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
        q = self.bn_q(q.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
    q = self.sn_q(q)

    k = self.linear_k(x).float()
    if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
        k = self.bn_k(k.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
    positional_encoding = self.positional_encoding.reshape(T, 1, H, W, C)
    k = self.sn_k(k + positional_encoding)

    q_orig = q.reshape(T, B_, self.num_heads, -1, head_dim)
    k_orig = k.reshape(B_, self.num_heads, -1, head_dim)
    n_tokens = k_orig.shape[2]

    if cfg.mode in {"qkformer_spike_shift", "spike_shift"}:
        # QKFormer-preserving variant.
        #
        # Baseline SDFormerFlow does:
        #   att_token = sn2_q(sum_channel(q))
        #   attn = k * att_token
        #
        # Keep that exact sparse token-spike carrier, then use Shiftmax only as
        # a positive token reweighting term. This avoids the H10 failure mode
        # where Shiftmax replaced sn2_q and destroyed signed sparse token gating.
        att_token = q_orig.sum(dim=-1, keepdim=True)
        att_token = self.sn2_q(att_token)
        att_gate = att_token.reshape(B_, self.num_heads, n_tokens, 1)
        attn = k_orig.mul(att_gate)

        scores = att_gate * cfg.score_scale
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        gate = shiftmax(scores, dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        attn = attn * gate
    elif cfg.mode in {"qkformer_token", "token"}:
        # SDFormerFlow uses QKFormer-style token gating, not standard QK^T V.
        # Keep that carrier: normalize the native signed token score from Q,
        # then apply the positive Shiftmax gate to the ternary K carrier.
        scores = q_orig.sum(dim=-1, keepdim=True).reshape(B_, self.num_heads, n_tokens, 1)
        scores = scores * cfg.score_scale
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        gate = shiftmax(scores, dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        attn = k_orig.mul(gate)
    elif cfg.mode in {"compat_qk_product", "legacy"}:
        # Historical H9 path. This is not strict BSA and is kept only so older
        # checkpoints/configs remain loadable and comparable.
        att_token = q_orig.sum(dim=-1, keepdim=True)
        att_token = self.sn2_q(att_token)
        attn = k_orig.mul(att_token.reshape(B_, self.num_heads, -1, 1))

        q_token = q_orig.permute(1, 2, 0, 3, 4).reshape(B_, self.num_heads, n_tokens, head_dim)
        scores = (q_token * k_orig).sum(dim=-1, keepdim=True) * cfg.score_scale
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        gate = shiftmax(scores, dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        attn = attn * gate
    else:
        raise ValueError(
            "bsa_attention.mode must be qkformer_spike_shift/spike_shift, "
            "qkformer_token/token, or compat_qk_product/legacy"
        )

    with torch.no_grad():
        self.h9_shiftmax_row_sum_mean = float(row_sum.detach().mean().cpu())
        self.h9_shiftmax_row_sum_min = float(row_sum.detach().min().cpu())
        self.h9_shiftmax_row_sum_max = float(row_sum.detach().max().cpu())
        self.h9_shiftmax_gate_mean = float(gate.detach().mean().cpu())

    attn = self.attn_drop(attn)
    x = attn.reshape(B_, self.num_heads, T, H, W, head_dim)
    x = x.permute(2, 0, 3, 4, 1, 5).reshape(T, B_, H, W, C).float()
    attn = self.attn_sn(x)
    x = self.proj(x)
    if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
        x = self.proj_bn(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
    x = x.reshape(B_, n_tokens, C)
    return x, attn


def install_shiftmax_attention(model: nn.Module, raw_config: dict | None) -> list[str]:
    cfg = config_from_dict(raw_config)
    if not cfg.enabled:
        return []
    installed: list[str] = []
    for name, module in _iter_attention_modules(model, cfg):
        if module.__class__.__name__ != "Spiking_QK_WindowAttention3D":
            continue
        if not hasattr(module, "_h9_original_forward"):
            module._h9_original_forward = module.forward
        module._h9_shiftmax_cfg = cfg
        module.forward = MethodType(_qk_shiftmax_gate_forward, module)
        installed.append(name)
    return installed


def register_shiftmax_pickle_compat() -> None:
    """Expose the patched forward name needed by full-module checkpoints."""

    try:
        from models.STSwinNet_SNN.Spiking_swin_transformer3D import Spiking_QK_WindowAttention3D
    except Exception:
        return
    setattr(Spiking_QK_WindowAttention3D, "_qk_shiftmax_gate_forward", _qk_shiftmax_gate_forward)


def shiftmax_attention_summary(model: nn.Module) -> dict[str, float | int]:
    modules = [
        module
        for module in model.modules()
        if hasattr(module, "_h9_shiftmax_cfg")
    ]
    if not modules:
        return {"num_modules": 0}
    row_means = [float(getattr(module, "h9_shiftmax_row_sum_mean", 0.0)) for module in modules]
    gate_means = [float(getattr(module, "h9_shiftmax_gate_mean", 0.0)) for module in modules]
    return {
        "num_modules": len(modules),
        "row_sum_mean": sum(row_means) / len(row_means),
        "gate_mean": sum(gate_means) / len(gate_means),
    }
