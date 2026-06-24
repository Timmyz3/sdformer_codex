"""H9/H10 Shiftmax compatibility layer for SDFormerFlow QK attention.

The legacy H9 modes are compatibility gates around SDFormerFlow's original
QKFormer-style token carrier. The H10c qk_bsa mode builds a true ternary Q/K
score matrix before Shiftmax, then uses K as the value carrier because this
baseline attention block has no separate V projection.
"""

from __future__ import annotations

import copy
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
    consensus_score_norm: str = "head_dim"
    consensus_bias: float = 0.0
    matrix_diag_bias: float = 0.0
    value_mode: str = "threshold"
    value_branch: str = "reuse_k"
    value_init: str = "copy_k"
    alpha0: float = 0.05
    mismatch_penalty: float = 0.5
    single_active_penalty: float = 0.0
    single_active_penalty_grad: str = "hard"
    single_active_ste_slope: float = 4.0
    single_active_ste_margin: float = 0.25
    relu_k_floor: float = 0.0
    residual_alpha: float = 0.3
    bipolar_mu: float = 0.5
    bipolar_lambda: float = 0.8
    bipolar_gate_min: float | None = None
    bipolar_gate_max: float | None = None
    deadzone_epsilon: float = 0.0
    confidence_enabled: bool = False
    k_consistency_mod: bool = False
    k_magnitude_alpha: float = 0.0  # K magnitude correction: score += α × sign(Q) × |K_before_sign|
    temporal_consistency_alpha: float = 0.0  # S3: penalty on gate time variation
    motion_weight_alpha: float = 0.0  # S1: scale motion magnitude bonus
    directional_channels_enabled: bool = False  # S2: split Q/K by x/y direction
    directional_merge_mode: str = "sum"  # S2: "sum" or "mean"
    confidence_min_active: int = 0  # FAPS: sparse K_mag only when active channels >= tau
    flow_disagreement_gamma: float = 0.0  # FAPS: penalize |S_x - S_y| when directional
    directional_residual_gamma: float = 0.0  # H62: confidence-gated directional residual strength
    confidence_floor: float = 0.0  # H62: minimum residual confidence
    kmag_quantize_bits: int = 2  # FAPS: quantize threshold-margin lane to N-bit levels
    sc_warm_start_gate: bool = False
    sc_mu_schedule_enabled: bool = False
    sc_mu_start_step: int = 0
    sc_mu_warmup_steps: int = 0
    sc_mu_start: float = 0.0
    hardware_quant_enabled: bool = False
    hardware_mu_pow2_shift: int = 0
    hardware_score_step: float = 0.0
    hardware_score_min: float | None = None
    hardware_score_max: float | None = None
    hardware_gate_step: float = 0.0
    hardware_gate_min: float | None = None
    hardware_gate_max: float | None = None


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
        consensus_score_norm=str(raw.get("consensus_score_norm", "head_dim")),
        consensus_bias=float(raw.get("consensus_bias", 0.0)),
        matrix_diag_bias=float(raw.get("matrix_diag_bias", 0.0)),
        value_mode=str(raw.get("value_mode", "threshold")),
        value_branch=str(raw.get("value_branch", "reuse_k")),
        value_init=str(raw.get("value_init", "copy_k")),
        alpha0=float(raw.get("alpha0", 0.05)),
        mismatch_penalty=float(raw.get("mismatch_penalty", 0.5)),
        single_active_penalty=float(raw.get("single_active_penalty", 0.0)),
        single_active_penalty_grad=str(raw.get("single_active_penalty_grad", "hard")),
        single_active_ste_slope=float(raw.get("single_active_ste_slope", 4.0)),
        single_active_ste_margin=float(raw.get("single_active_ste_margin", 0.25)),
        relu_k_floor=float(raw.get("relu_k_floor", 0.0)),
        residual_alpha=float(raw.get("residual_alpha", 0.3)),
        bipolar_mu=float(raw.get("bipolar_mu", raw.get("residual_alpha", 0.5))),
        bipolar_lambda=float(raw.get("bipolar_lambda", 0.8)),
        bipolar_gate_min=None if raw.get("bipolar_gate_min") is None else float(raw.get("bipolar_gate_min")),
        bipolar_gate_max=None if raw.get("bipolar_gate_max") is None else float(raw.get("bipolar_gate_max")),
        deadzone_epsilon=float(raw.get("deadzone_epsilon", 0.0)),
        confidence_enabled=bool(raw.get("confidence_enabled", False)),
        k_consistency_mod=bool(raw.get("k_consistency_mod", False)),
        k_magnitude_alpha=float(raw.get("k_magnitude_alpha", 0.0)),
        temporal_consistency_alpha=float(raw.get("temporal_consistency_alpha", 0.0)),
        motion_weight_alpha=float(raw.get("motion_weight_alpha", 0.0)),
        directional_channels_enabled=bool(raw.get("directional_channels_enabled", False)),
        directional_merge_mode=str(raw.get("directional_merge_mode", "sum")),
        confidence_min_active=int(raw.get("confidence_min_active", 0) or 0),
        flow_disagreement_gamma=float(raw.get("flow_disagreement_gamma", 0.0)),
        directional_residual_gamma=float(raw.get("directional_residual_gamma", 0.0)),
        confidence_floor=float(raw.get("confidence_floor", 0.0)),
        kmag_quantize_bits=int(raw.get("kmag_quantize_bits", 2) or 2),
        sc_warm_start_gate=bool(raw.get("sc_warm_start_gate", False)),
        sc_mu_schedule_enabled=bool(raw.get("sc_mu_schedule_enabled", False)),
        sc_mu_start_step=int(raw.get("sc_mu_start_step", 0) or 0),
        sc_mu_warmup_steps=int(raw.get("sc_mu_warmup_steps", 0) or 0),
        sc_mu_start=float(raw.get("sc_mu_start", 0.0)),
        hardware_quant_enabled=bool(raw.get("hardware_quant_enabled", False)),
        hardware_mu_pow2_shift=int(raw.get("hardware_mu_pow2_shift", 0) or 0),
        hardware_score_step=float(raw.get("hardware_score_step", 0.0) or 0.0),
        hardware_score_min=None if raw.get("hardware_score_min") is None else float(raw.get("hardware_score_min")),
        hardware_score_max=None if raw.get("hardware_score_max") is None else float(raw.get("hardware_score_max")),
        hardware_gate_step=float(raw.get("hardware_gate_step", 0.0) or 0.0),
        hardware_gate_min=None if raw.get("hardware_gate_min") is None else float(raw.get("hardware_gate_min")),
        hardware_gate_max=None if raw.get("hardware_gate_max") is None else float(raw.get("hardware_gate_max")),
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


def _quantize_ste(value: torch.Tensor, step: float) -> torch.Tensor:
    if step <= 0.0:
        return value
    quantized = torch.round(value / float(step)) * float(step)
    return value + (quantized - value).detach()


def _apply_hardware_score_quant(scores: torch.Tensor, cfg: ShiftmaxAttentionConfig) -> torch.Tensor:
    if not cfg.hardware_quant_enabled:
        return scores
    if cfg.hardware_score_min is not None or cfg.hardware_score_max is not None:
        min_value = -float("inf") if cfg.hardware_score_min is None else float(cfg.hardware_score_min)
        max_value = float("inf") if cfg.hardware_score_max is None else float(cfg.hardware_score_max)
        scores = scores.clamp(min=min_value, max=max_value)
    return _quantize_ste(scores, float(cfg.hardware_score_step))


def _apply_hardware_gate_quant(gate: torch.Tensor, cfg: ShiftmaxAttentionConfig) -> torch.Tensor:
    if not cfg.hardware_quant_enabled:
        return gate
    if cfg.hardware_gate_min is not None or cfg.hardware_gate_max is not None:
        min_value = -float("inf") if cfg.hardware_gate_min is None else float(cfg.hardware_gate_min)
        max_value = float("inf") if cfg.hardware_gate_max is None else float(cfg.hardware_gate_max)
        gate = gate.clamp(min=min_value, max=max_value)
    return _quantize_ste(gate, float(cfg.hardware_gate_step))


def _apply_hardware_mu_quant(mu: float, cfg: ShiftmaxAttentionConfig) -> float:
    if not cfg.hardware_quant_enabled or int(cfg.hardware_mu_pow2_shift) <= 0:
        return float(mu)
    return 1.0 / float(1 << int(cfg.hardware_mu_pow2_shift))


def _safe_float_stat(tensor: torch.Tensor | None, op: str) -> float | None:
    if tensor is None:
        return None
    data = tensor.detach().float()
    if data.numel() == 0:
        return 0.0
    if op == "mean":
        return float(data.mean().item())
    if op == "std":
        return float(data.std(unbiased=False).item())
    if op == "min":
        return float(data.min().item())
    if op == "max":
        return float(data.max().item())
    raise ValueError(op)


def _maybe_emit_h60_profile(
    module: nn.Module,
    *,
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    tx_scores: torch.Tensor,
    sc_scores: torch.Tensor,
    fused_scores: torch.Tensor,
    gate: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> None:
    collector = getattr(module, "_h9_profile_collector", None)
    if collector is None:
        return
    with torch.no_grad():
        gate_data = gate.detach().float()
        token_dim = 2 if gate_data.ndim >= 3 else -2
        gate_abs = gate_data.abs()
        gate_sum = gate_abs.sum(dim=token_dim, keepdim=True).clamp_min(float(cfg.eps))
        prob = gate_abs / gate_sum
        entropy = -(prob.clamp_min(float(cfg.eps)).log2() * prob).sum(dim=token_dim)
        sorted_prob = torch.sort(prob, dim=token_dim, descending=True).values
        top1_mass = sorted_prob.narrow(token_dim, 0, 1).sum(dim=token_dim)
        top4 = min(4, sorted_prob.shape[token_dim])
        top4_mass = sorted_prob.narrow(token_dim, 0, top4).sum(dim=token_dim)
        eff_tokens = 1.0 / (prob.square().sum(dim=token_dim).clamp_min(float(cfg.eps)))

        q_active = q_orig.detach().ne(0).float()
        k_active = k_orig.detach().ne(0).float()
        q_token_active = q_active.any(dim=-1).float()
        k_token_active = k_active.any(dim=-1).float()

        bundle_stats: dict[str, float] = {}
        if q_token_active.ndim >= 4:
            t_len = q_token_active.shape[0]
            for bundle_t in (1, 2, 4):
                usable = (t_len // bundle_t) * bundle_t
                if usable <= 0:
                    continue
                grouped = q_token_active[:usable].reshape(bundle_t, -1, *q_token_active.shape[1:]).amax(dim=0)
                density = grouped.float().mean(dim=-1)
                bundle_stats[f"ttb{bundle_t}_empty_ratio"] = float((density <= 0).float().mean().item())
                bundle_stats[f"ttb{bundle_t}_low_density_ratio"] = float(((density > 0) & (density <= 0.125)).float().mean().item())
                bundle_stats[f"ttb{bundle_t}_high_density_ratio"] = float((density >= 0.5).float().mean().item())

        stats = {
            "mode": str(cfg.mode),
            "stage": int(getattr(module, "_h9_stage", -1)),
            "block": int(getattr(module, "_h9_block", -1)),
            "num_heads": int(getattr(module, "num_heads", 0)),
            "tokens": int(k_orig.shape[2]) if k_orig.ndim >= 3 else 0,
            "head_dim": int(k_orig.shape[-1]) if k_orig.ndim >= 1 else 0,
            "tx_mean": _safe_float_stat(tx_scores, "mean"),
            "tx_std": _safe_float_stat(tx_scores, "std"),
            "tx_min": _safe_float_stat(tx_scores, "min"),
            "tx_max": _safe_float_stat(tx_scores, "max"),
            "sc_mean": _safe_float_stat(sc_scores, "mean"),
            "sc_std": _safe_float_stat(sc_scores, "std"),
            "sc_min": _safe_float_stat(sc_scores, "min"),
            "sc_max": _safe_float_stat(sc_scores, "max"),
            "fused_mean": _safe_float_stat(fused_scores, "mean"),
            "fused_std": _safe_float_stat(fused_scores, "std"),
            "fused_min": _safe_float_stat(fused_scores, "min"),
            "fused_max": _safe_float_stat(fused_scores, "max"),
            "gate_mean": _safe_float_stat(gate_data, "mean"),
            "gate_std": _safe_float_stat(gate_data, "std"),
            "gate_min": _safe_float_stat(gate_data, "min"),
            "gate_max": _safe_float_stat(gate_data, "max"),
            "gate_entropy_mean": _safe_float_stat(entropy, "mean"),
            "top1_mass_mean": _safe_float_stat(top1_mass, "mean"),
            "top4_mass_mean": _safe_float_stat(top4_mass, "mean"),
            "effective_tokens_mean": _safe_float_stat(eff_tokens, "mean"),
            "q_active_density": _safe_float_stat(q_active, "mean"),
            "k_active_density": _safe_float_stat(k_active, "mean"),
            "q_token_active_density": _safe_float_stat(q_token_active, "mean"),
            "k_token_active_density": _safe_float_stat(k_token_active, "mean"),
            **bundle_stats,
        }
    collector(module, stats)


def shiftmax_raw(scores: torch.Tensor, dim: int = -1, eps: float = 1.0e-6) -> torch.Tensor:
    """Shiftmax ablation without subtracting the row maximum."""

    numerator = torch.pow(2.0, scores)
    denom_power = torch.ceil(torch.log2(numerator.sum(dim=dim, keepdim=True).clamp_min(eps)))
    denominator = torch.pow(2.0, denom_power)
    return numerator / denominator


def shiftnorm(nonnegative_scores: torch.Tensor, dim: int = -1, eps: float = 1.0e-6) -> torch.Tensor:
    """Power-of-two normalization for nonnegative integer-like scores.

    This is a Shiftmax sibling for hardware-oriented ablations: the numerator is
    a popcount-style score instead of 2^score, and the denominator is rounded up
    to the next power of two so division can be approximated by shifting.
    """

    numerator = nonnegative_scores.clamp_min(0)
    row_sum = numerator.sum(dim=dim, keepdim=True)
    empty = row_sum <= eps
    if empty.any():
        numerator = torch.where(empty.expand_as(numerator), torch.ones_like(numerator), numerator)
        row_sum = numerator.sum(dim=dim, keepdim=True)
    denom_power = torch.ceil(torch.log2(row_sum.clamp_min(eps)))
    denominator = torch.pow(2.0, denom_power)
    return numerator / denominator


def l1norm(nonnegative_scores: torch.Tensor, dim: int = -1, eps: float = 1.0e-6) -> torch.Tensor:
    """Exact L1 normalization for popcount-only attention ablations."""

    numerator = nonnegative_scores.clamp_min(0)
    row_sum = numerator.sum(dim=dim, keepdim=True)
    empty = row_sum <= eps
    if empty.any():
        numerator = torch.where(empty.expand_as(numerator), torch.ones_like(numerator), numerator)
        row_sum = numerator.sum(dim=dim, keepdim=True)
    return numerator / row_sum.clamp_min(eps)


def _ternary_sign_ste(x: torch.Tensor) -> torch.Tensor:
    """Hard {-1,0,+1} event in forward, identity gradient in backward."""

    hard = x.sign()
    return (hard - x).detach() + x


def _single_active_proxy(event: torch.Tensor, cfg: ShiftmaxAttentionConfig) -> torch.Tensor:
    """Hard active mask in forward with optional soft proxy gradient."""

    mode = cfg.single_active_penalty_grad
    if mode in {"hard", "none"}:
        return event.ne(0).to(dtype=event.dtype)
    if mode not in {"ste", "proxy", "soft"}:
        raise ValueError("bsa_attention.single_active_penalty_grad must be hard or ste")
    hard_active = event.ne(0).to(dtype=event.dtype)
    soft_active = torch.sigmoid(float(cfg.single_active_ste_slope) * (event.abs() - float(cfg.single_active_ste_margin)))
    return (hard_active - soft_active).detach() + soft_active


def _qkformer_token_q(q_orig: torch.Tensor) -> torch.Tensor:
    return q_orig.permute(1, 2, 0, 3, 4).reshape(
        q_orig.shape[1],
        q_orig.shape[2],
        q_orig.shape[0] * q_orig.shape[3],
        q_orig.shape[4],
    )


def _temporal_motion_from_q_orig(
    q_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor | None:
    """Q temporal saliency per token, layout [B, num_heads, T*N, 1].

    ``q_orig`` is ``[T, B, num_heads, N, head_dim]`` inside one Swin window.
    Saliency is the channel-mean ``|Q_t - Q_{t-1}|``; ``t=0`` is zero because
    there is no prior step inside the window. Normalization is per batch item
    and head over time and window tokens (not one global max across heads).

    With default ``window_size[0]=2`` only one inter-step difference exists
    within the window; this is not long-range temporal context.
    """

    if float(cfg.motion_weight_alpha) <= 0.0:
        return None
    if q_orig.ndim != 5:
        raise ValueError("q_orig must be [T, B, num_heads, N, head_dim] for motion saliency")

    t_steps, batch, num_heads, n_tokens, _ = q_orig.shape
    if t_steps < 2:
        return torch.zeros(
            batch,
            num_heads,
            t_steps * n_tokens,
            1,
            device=q_orig.device,
            dtype=q_orig.dtype,
        )

    q_diff = (q_orig[1:] - q_orig[:-1]).abs().mean(dim=-1)  # [T-1, B, H, N]
    q_diff = torch.cat([torch.zeros_like(q_diff[:1]), q_diff], dim=0)  # [T, B, H, N]
    q_diff = q_diff / q_diff.amax(dim=(0, 3), keepdim=True).clamp_min(1e-6)
    motion = q_diff.permute(1, 2, 0, 3).reshape(batch, num_heads, t_steps * n_tokens, 1)
    return motion.to(dtype=q_orig.dtype)


def _ensure_independent_value_branch(attn: nn.Module, cfg: ShiftmaxAttentionConfig) -> None:
    """Add a trainable V branch to SDFormerFlow's QK-only block when requested.

    The baseline class has commented-out V code but no live `linear_v/sn_v`
    modules. For strict QKV ablations we attach separate modules in the overlay
    after checkpoint load and before optimizer construction. `copy_k` gives a
    stable continuation point while still creating independent trainable V
    parameters.
    """

    if hasattr(attn, "linear_v") and hasattr(attn, "sn_v"):
        return
    if not hasattr(attn, "linear_k") or not hasattr(attn, "sn_k"):
        raise AttributeError("independent V branch requires linear_k and sn_k on the attention module")
    if cfg.value_init not in {"copy_k"}:
        raise ValueError("bsa_attention.value_init currently supports only copy_k")
    attn.linear_v = copy.deepcopy(attn.linear_k)
    if getattr(attn, "norm_layer", None) in {"BN", "BNTT", "tdBN", "IN"} and hasattr(attn, "bn_k"):
        attn.bn_v = copy.deepcopy(attn.bn_k)
    attn.sn_v = copy.deepcopy(attn.sn_k)


def _uses_independent_value_branch(cfg: ShiftmaxAttentionConfig) -> bool:
    return cfg.mode in {
        "strict_bsa_qkv_shiftmax",
        "bsa_qkv_shiftmax",
        "bsa_true_qkv_shiftmax",
        "a2os2a_qkv_l1",
        "a2os2a_true_qkv_l1",
        "ternary_alpha_xnor_ssa_qkv_linear",
        "alpha_xnor_ssa_qkv_linear",
        "ternary_alpha_xnor_qkv",
        "h42c",
        "ternary_alpha_xnor_ssa_qkv_shiftmax",
        "alpha_xnor_ssa_qkv_shiftmax",
        "h42d",
    }


def sync_independent_value_branch_from_k(model: nn.Module, raw_config: dict | None) -> int:
    """Initialize overlay V branches from the currently loaded K branches.

    QKV ablations are usually resumed from a baseline QK checkpoint. The V
    modules must exist before `load_state_dict` so overlay checkpoints can load,
    but a baseline checkpoint has no V keys. In that case this function copies
    the already-loaded K branch into V after checkpoint load, preserving the
    intended `copy_k` continuation point without overwriting trained V when a
    later overlay checkpoint provides V parameters.
    """

    cfg = config_from_dict(raw_config)
    if not cfg.enabled or not _uses_independent_value_branch(cfg):
        return 0
    synced = 0
    for _, module in _iter_attention_modules(model, cfg):
        if module.__class__.__name__ != "Spiking_QK_WindowAttention3D":
            continue
        _ensure_independent_value_branch(module, cfg)
        module.linear_v.load_state_dict(module.linear_k.state_dict())
        if (
            getattr(module, "norm_layer", None) in {"BN", "BNTT", "tdBN", "IN"}
            and hasattr(module, "bn_k")
            and hasattr(module, "bn_v")
        ):
            module.bn_v.load_state_dict(module.bn_k.state_dict())
        module.sn_v.load_state_dict(module.sn_k.state_dict())
        module._h9_v_initialized_from_loaded_k = True
        synced += 1
    return synced


def _independent_value_tokens(
    attn: nn.Module,
    x: torch.Tensor,
    T: int,
    B_: int,
    H: int,
    W: int,
    C: int,
    head_dim: int,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    _ensure_independent_value_branch(attn, cfg)
    v = attn.linear_v(x).float()
    if getattr(attn, "norm_layer", None) in ["BN", "BNTT", "tdBN", "IN"]:
        v = attn.bn_v(v.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
    v = attn.sn_v(v)
    return v.reshape(B_, attn.num_heads, T * H * W, head_dim)


def _signed_consensus_token_scores(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    """Token-wise signed Q/K consensus score for H13 modes.

    Forward values are integer-like agreement-minus-conflict popcounts:
    same-positive and same-negative channels add +1, opposite polarity channels
    add -1, and silent/silent channels add 0. When enabled, single-active
    channels (one side silent, the other side nonzero) are penalized separately.
    Optional normalization keeps Shiftmax from becoming too sharp while
    remaining power-of-two friendly for head_dim values.
    """

    _motion = _temporal_motion_from_q_orig(q_orig, cfg)

    q_event = _ternary_sign_ste(_qkformer_token_q(q_orig))
    k_event = _ternary_sign_ste(k_orig)
    q_active = q_event.ne(0)
    k_active = k_event.ne(0)
    if cfg.directional_channels_enabled:
        # S2: split head_dim into two halves (x/y direction channels), compute SC separately
        d = q_event.shape[-1]
        d2 = d // 2
        q_x, q_y = q_event[..., :d2], q_event[..., d2:]
        k_x, k_y = k_event[..., :d2], k_event[..., d2:]
        score_x = (q_x * k_x).sum(dim=-1, keepdim=True)
        score_y = (q_y * k_y).sum(dim=-1, keepdim=True)
        if cfg.directional_merge_mode == "mean":
            score = (score_x + score_y) / 2.0
        else:
            score = score_x + score_y
    else:
        score = (q_event * k_event).sum(dim=-1, keepdim=True)
    if _motion is not None:
        score = score * (1.0 + float(cfg.motion_weight_alpha) * _motion)
    if cfg.k_magnitude_alpha:
        k_mag = torch.relu(k_orig - k_event.detach())
        mag_correction = (q_event * k_mag).sum(dim=-1, keepdim=True)
        score = score + float(cfg.k_magnitude_alpha) * mag_correction
    if cfg.single_active_penalty:
        if cfg.single_active_penalty_grad in {"ste", "proxy", "soft"}:
            q_active_proxy = _single_active_proxy(q_event, cfg)
            k_active_proxy = _single_active_proxy(k_event, cfg)
            single_active = (
                q_active_proxy * (1.0 - k_active_proxy) + (1.0 - q_active_proxy) * k_active_proxy
            ).sum(dim=-1, keepdim=True)
        else:
            single_active = (q_active ^ k_active).sum(dim=-1, keepdim=True).to(dtype=score.dtype)
        score = score - float(cfg.single_active_penalty) * single_active
    norm = cfg.consensus_score_norm
    if norm in {"head_dim", "dim"}:
        score = score / float(max(1, q_event.shape[-1]))
    elif norm in {"sqrt_head_dim", "sqrt_dim"}:
        score = score / float(max(1, q_event.shape[-1]) ** 0.5)
    elif norm == "active":
        if cfg.single_active_penalty:
            if cfg.single_active_penalty_grad in {"ste", "proxy", "soft"}:
                q_active_proxy = _single_active_proxy(q_event, cfg)
                k_active_proxy = _single_active_proxy(k_event, cfg)
                active = (q_active_proxy + k_active_proxy - q_active_proxy * k_active_proxy).sum(
                    dim=-1, keepdim=True
                ).clamp_min(1)
            else:
                active = (q_active | k_active).sum(dim=-1, keepdim=True).clamp_min(1)
        else:
            active = (q_active & k_active).sum(dim=-1, keepdim=True).clamp_min(1)
        score = score / active.to(dtype=score.dtype)
    elif norm in {"none", "raw"}:
        pass
    else:
        raise ValueError("bsa_attention.consensus_score_norm must be head_dim, sqrt_head_dim, active, or none")
    return score * cfg.score_scale


def _normalize_consensus_score(
    score: torch.Tensor,
    head_dim: int,
    cfg: ShiftmaxAttentionConfig,
    active: torch.Tensor | None = None,
) -> torch.Tensor:
    norm = cfg.consensus_score_norm
    if norm in {"head_dim", "dim"}:
        score = score / float(max(1, head_dim))
    elif norm in {"sqrt_head_dim", "sqrt_dim"}:
        score = score / float(max(1, head_dim) ** 0.5)
    elif norm == "active":
        if active is None:
            raise ValueError("active consensus normalization requires an active-count tensor")
        score = score / active.clamp_min(1).to(dtype=score.dtype)
    elif norm in {"none", "raw"}:
        pass
    else:
        raise ValueError(
            "bsa_attention.consensus_score_norm must be head_dim, sqrt_head_dim, active, or none"
        )
    return score * cfg.score_scale


def _strict_bsa_matrix_attention(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
    value_orig: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BSA-style ternary QK^T attention.

    Q and K are converted to {-1, 0, +1} events before the score matrix, matching
    the BSA ternary matrix product. If `value_orig` is provided, it is a separate
    trainable V stream; otherwise K is reused as V for legacy no-V ablations.
    `value_mode=sign` keeps the value path event/add/shift friendly, while
    `value_mode=threshold` preserves the ATLIF scalar firing amplitude.
    """

    q_event = _ternary_sign_ste(_qkformer_token_q(q_orig))
    k_event = _ternary_sign_ste(k_orig)
    scores = torch.matmul(q_event, k_event.transpose(-2, -1))
    active = None
    if cfg.consensus_score_norm == "active":
        active = torch.matmul(
            q_event.detach().ne(0).to(dtype=scores.dtype),
            k_event.detach().ne(0).to(dtype=scores.dtype).transpose(-2, -1),
        )
    scores = _normalize_consensus_score(scores, q_event.shape[-1], cfg, active=active)
    if cfg.center_scores:
        scores = scores - scores.mean(dim=-1, keepdim=True)

    gate = shiftmax(scores, dim=-1, eps=cfg.eps)
    row_sum = gate.sum(dim=-1)
    if cfg.preserve_mean:
        gate = gate * float(k_event.shape[-2])

    value_source = k_orig if value_orig is None else value_orig
    if cfg.value_mode in {"sign", "event", "ternary"}:
        value = _ternary_sign_ste(value_source)
    elif cfg.value_mode in {"threshold", "theta", "atlif"}:
        value = value_source
    else:
        raise ValueError("bsa_attention.value_mode must be threshold/theta/atlif or sign/event/ternary")
    return torch.matmul(gate, value), row_sum, gate


def _ternary_alpha_xnor_token_scores(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
    beta: float | None = None,
) -> torch.Tensor:
    """Ternary extension of CVPR 2025 alpha-XNOR spike similarity.

    The paper's binary alpha-XNOR gives spike-spike matches more weight than
    non-spike/non-spike matches and distinguishes them from mismatches. Our
    ATLIF path is signed ternary, so same nonzero polarity is a strong match,
    same silence gets the paper's small alpha reward, and opposite polarity is
    penalized because it is harmful for flow direction.
    """

    q_event = _ternary_sign_ste(_qkformer_token_q(q_orig))
    k_event = _ternary_sign_ste(k_orig)
    q_active = q_event.ne(0)
    k_active = k_event.ne(0)
    same_nonzero = (q_event == k_event) & q_active & k_active
    same_zero = (~q_active) & (~k_active)
    opposite = (q_event == -k_event) & q_active & k_active
    if cfg.single_active_penalty and cfg.single_active_penalty_grad in {"ste", "proxy", "soft"}:
        q_active_proxy = _single_active_proxy(q_event, cfg)
        k_active_proxy = _single_active_proxy(k_event, cfg)
        single_active = q_active_proxy * (1.0 - k_active_proxy) + (1.0 - q_active_proxy) * k_active_proxy
    else:
        single_active = (q_active ^ k_active).to(dtype=q_orig.dtype)
    _mismatch = beta if beta is not None else float(cfg.mismatch_penalty)
    score = (
        same_nonzero.to(dtype=q_orig.dtype)
        + float(cfg.alpha0) * same_zero.to(dtype=q_orig.dtype)
        - _mismatch * opposite.to(dtype=q_orig.dtype)
        - float(cfg.single_active_penalty) * single_active.to(dtype=q_orig.dtype)
    ).sum(dim=-1, keepdim=True)
    # ── NTX-11: K magnitude correction (additive, off by default) ──
    if cfg.k_magnitude_alpha:
        k_mag = torch.relu(k_orig - k_event.detach())  # |K| before sign binarization
        mag_correction = (q_event * k_mag).sum(dim=-1, keepdim=True)
        score = score + float(cfg.k_magnitude_alpha) * mag_correction
    # ── end K magnitude ──
    active = None
    if cfg.consensus_score_norm == "active":
        active = (q_active | k_active).sum(dim=-1, keepdim=True).clamp_min(1)
    return _normalize_consensus_score(score, q_event.shape[-1], cfg, active=active)


def _dual_channel_token_scores(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    """Excitation/inhibition token selector for signed ternary Q/K events.

    Positive and negative spikes are handled as separate channels. Same-polarity
    evidence excites the token, opposite-polarity and one-sided activity inhibit
    it. This keeps H49's QKFormer-native K carrier while making the signed
    ternary semantics explicit instead of folding everything into one XNOR term.
    """

    q_event = _ternary_sign_ste(_qkformer_token_q(q_orig))
    k_event = _ternary_sign_ste(k_orig)
    q_pos = torch.relu(q_event)
    q_neg = torch.relu(-q_event)
    k_pos = torch.relu(k_event)
    k_neg = torch.relu(-k_event)
    q_zero = torch.clamp(1.0 - q_event.abs(), min=0.0, max=1.0)
    k_zero = torch.clamp(1.0 - k_event.abs(), min=0.0, max=1.0)

    excite = (q_pos * k_pos + q_neg * k_neg).sum(dim=-1, keepdim=True)
    inhibit = (q_pos * k_neg + q_neg * k_pos).sum(dim=-1, keepdim=True)
    if cfg.single_active_penalty:
        q_active = q_pos + q_neg
        k_active = k_pos + k_neg
        one_sided = (q_active * k_zero + q_zero * k_active).sum(dim=-1, keepdim=True)
        inhibit = inhibit + float(cfg.single_active_penalty) * one_sided
    same_zero = (q_zero * k_zero).sum(dim=-1, keepdim=True)
    score = excite - float(cfg.mismatch_penalty) * inhibit + float(cfg.alpha0) * same_zero

    active = None
    if cfg.consensus_score_norm == "active":
        active = ((q_pos + q_neg) + (k_pos + k_neg)).sum(dim=-1, keepdim=True).clamp_min(1)
    return _normalize_consensus_score(score, q_event.shape[-1], cfg, active=active)


def _bipolar_token_score_components(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return TX, same-polarity, and opposite-polarity token scores.

    H54 uses the same ternary evidence as TX, but keeps the positive and
    negative evidence separated until after normalization. That lets the final
    selector become signed while every Shiftmax branch remains nonnegative.
    """

    q_event = _ternary_sign_ste(_qkformer_token_q(q_orig))
    k_event = _ternary_sign_ste(k_orig)
    q_pos = torch.relu(q_event)
    q_neg = torch.relu(-q_event)
    k_pos = torch.relu(k_event)
    k_neg = torch.relu(-k_event)
    q_zero = torch.clamp(1.0 - q_event.abs(), min=0.0, max=1.0)
    k_zero = torch.clamp(1.0 - k_event.abs(), min=0.0, max=1.0)

    same_nonzero = (q_pos * k_pos + q_neg * k_neg).sum(dim=-1, keepdim=True)
    opposite = (q_pos * k_neg + q_neg * k_pos).sum(dim=-1, keepdim=True)
    same_zero = (q_zero * k_zero).sum(dim=-1, keepdim=True)
    one_sided = torch.zeros_like(opposite)
    if cfg.single_active_penalty:
        q_active = q_pos + q_neg
        k_active = k_pos + k_neg
        one_sided = (q_active * k_zero + q_zero * k_active).sum(dim=-1, keepdim=True)

    same_score = same_nonzero + float(cfg.alpha0) * same_zero
    opp_score = opposite + float(cfg.single_active_penalty) * one_sided
    tx_score = same_score - float(cfg.mismatch_penalty) * opposite - float(cfg.single_active_penalty) * one_sided

    active = None
    if cfg.consensus_score_norm == "active":
        active = ((q_pos + q_neg) + (k_pos + k_neg)).sum(dim=-1, keepdim=True).clamp_min(1)
    return (
        _normalize_consensus_score(tx_score, q_event.shape[-1], cfg, active=active),
        _normalize_consensus_score(same_score, q_event.shape[-1], cfg, active=active),
        _normalize_consensus_score(opp_score, q_event.shape[-1], cfg, active=active),
    )


def _center_token_scores(score: torch.Tensor, cfg: ShiftmaxAttentionConfig) -> torch.Tensor:
    if cfg.center_scores:
        return score - score.mean(dim=2, keepdim=True)
    return score


def _tx_sc_fusion_score_pair(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """TX/SC fusion scores with K_mag applied on TX only (not SC)."""

    k_mag = float(cfg.k_magnitude_alpha)
    if k_mag > 0:
        object.__setattr__(cfg, "k_magnitude_alpha", 0.0)
    sc_scores = _signed_consensus_token_scores(q_orig, k_orig, cfg)
    if k_mag > 0:
        object.__setattr__(cfg, "k_magnitude_alpha", k_mag)
    tx_scores = _ternary_alpha_xnor_token_scores(q_orig, k_orig, cfg)
    return tx_scores, sc_scores


def _faps_dyadic_channel_score(
    q_event: torch.Tensor,
    k_event: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    """Unified dyadic popcount for one x or y channel group.

    Compile-time weights absorb TX(alpha0/mismatch/single) + SC(mu=1/8):
    +4 same-sign, +1 silence, -1 softened opposite, -4 single-active.
    """

    q_active = q_event.ne(0)
    k_active = k_event.ne(0)
    same_nonzero = (q_event == k_event) & q_active & k_active
    same_zero = (~q_active) & (~k_active)
    opposite = (q_event == -k_event) & q_active & k_active
    if cfg.single_active_penalty and cfg.single_active_penalty_grad in {"ste", "proxy", "soft"}:
        q_active_proxy = _single_active_proxy(q_event, cfg)
        k_active_proxy = _single_active_proxy(k_event, cfg)
        single_active = q_active_proxy * (1.0 - k_active_proxy) + (1.0 - q_active_proxy) * k_active_proxy
    else:
        single_active = (q_active ^ k_active).to(dtype=q_event.dtype)
    score = (
        4.0 * same_nonzero.to(dtype=q_event.dtype)
        + 1.0 * same_zero.to(dtype=q_event.dtype)
        - 1.0 * opposite.to(dtype=q_event.dtype)
        - 4.0 * single_active
    ).sum(dim=-1, keepdim=True)
    return score


def _quantize_margin_bits(margin: torch.Tensor, bits: int) -> torch.Tensor:
    levels = max(2, int(bits))
    max_margin = margin.amax(dim=-1, keepdim=True).clamp_min(1.0e-6)
    normalized = (margin / max_margin).clamp(0.0, 1.0)
    quantized = torch.round(normalized * float(levels - 1)) / float(levels - 1)
    return quantized * max_margin


def _faps_sparse_k_magnitude(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    q_event: torch.Tensor,
    k_event: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    """Sparse 2-bit threshold-margin lane on high-confidence tokens only."""

    k_mag = torch.relu(k_orig - k_event.detach())
    k_mag = _quantize_margin_bits(k_mag, cfg.kmag_quantize_bits)
    mag_correction = (q_event * k_mag).sum(dim=-1, keepdim=True)
    min_active = int(cfg.confidence_min_active)
    if min_active > 0:
        active_count = (q_event.ne(0) | k_event.ne(0)).sum(dim=-1, keepdim=True)
        mask = (active_count >= min_active).to(dtype=mag_correction.dtype)
        mag_correction = mag_correction * mask
    return float(cfg.k_magnitude_alpha) * mag_correction


def _faps_flow_aligned_token_scores(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    """FAPS: flow-aligned unified dyadic popcount with optional sparse K_mag."""

    q_event = _ternary_sign_ste(_qkformer_token_q(q_orig))
    k_event = _ternary_sign_ste(k_orig)
    if cfg.directional_channels_enabled:
        head_dim = q_event.shape[-1]
        half = head_dim // 2
        score_x = _faps_dyadic_channel_score(q_event[..., :half], k_event[..., :half], cfg)
        score_y = _faps_dyadic_channel_score(q_event[..., half:], k_event[..., half:], cfg)
        gamma = float(cfg.flow_disagreement_gamma)
        if gamma > 0.0:
            score = score_x + score_y - gamma * (score_x - score_y).abs()
        elif cfg.directional_merge_mode == "mean":
            score = (score_x + score_y) / 2.0
        else:
            score = score_x + score_y
    else:
        score = _faps_dyadic_channel_score(q_event, k_event, cfg)
    if cfg.k_magnitude_alpha:
        score = score + _faps_sparse_k_magnitude(q_orig, k_orig, q_event, k_event, cfg)
    active = None
    if cfg.consensus_score_norm == "active":
        active = (q_event.ne(0) | k_event.ne(0)).sum(dim=-1, keepdim=True).clamp_min(1)
    return _normalize_consensus_score(score, q_event.shape[-1], cfg, active=active)


def _event_agree_confidence(
    q_event: torch.Tensor,
    k_event: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    """Hardware-friendly residual confidence from active and agree popcounts."""

    q_active = q_event.ne(0)
    k_active = k_event.ne(0)
    active = (q_active | k_active).sum(dim=-1, keepdim=True).to(dtype=q_event.dtype)
    agree = ((q_event == k_event) & q_active & k_active).sum(dim=-1, keepdim=True).to(dtype=q_event.dtype)
    active_frac = active / float(max(1, q_event.shape[-1]))
    agree_ratio = agree / active.clamp_min(1.0)
    conf = torch.sqrt(active_frac.clamp_min(0.0)) * agree_ratio.clamp(0.0, 1.0)
    floor = float(cfg.confidence_floor)
    if floor > 0.0:
        conf = torch.clamp(conf, min=floor)
    return conf


def _faps_directional_residual_score(
    q_event: torch.Tensor,
    k_event: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    """Small x/y group residual used by H62, normalized like the other scores."""

    head_dim = q_event.shape[-1]
    half = head_dim // 2
    if half <= 0 or half == head_dim:
        score = _faps_dyadic_channel_score(q_event, k_event, cfg)
    else:
        score_x = _faps_dyadic_channel_score(q_event[..., :half], k_event[..., :half], cfg)
        score_y = _faps_dyadic_channel_score(q_event[..., half:], k_event[..., half:], cfg)
        score = (score_x + score_y) / 2.0
    active = None
    if cfg.consensus_score_norm == "active":
        active = (q_event.ne(0) | k_event.ne(0)).sum(dim=-1, keepdim=True).clamp_min(1)
    return _normalize_consensus_score(score, head_dim, cfg, active=active)


def _confidence_calibrated_nts_scores(
    module: nn.Module,
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    """H62: NTS score with confidence-gated SC and optional directional residual.

    The base TX score stays identical to NTS/H60. SC and directional evidence are
    only injected when Q/K have enough active, same-sign event support, making the
    residual easy to implement with popcount counters and safer than full FAPS.
    """

    tx_scores, sc_scores = _tx_sc_fusion_score_pair(q_orig, k_orig, cfg)
    q_event = _ternary_sign_ste(_qkformer_token_q(q_orig))
    k_event = _ternary_sign_ste(k_orig)
    conf = _event_agree_confidence(q_event, k_event, cfg)
    mu = _scheduled_bipolar_mu(module, cfg)
    residual = mu * sc_scores
    gamma = float(cfg.directional_residual_gamma)
    if gamma != 0.0:
        residual = residual + gamma * _faps_directional_residual_score(q_event, k_event, cfg)
    return tx_scores + conf * residual


def _maybe_clamp_bipolar_gate(gate: torch.Tensor, cfg: ShiftmaxAttentionConfig) -> torch.Tensor:
    if cfg.bipolar_gate_min is None and cfg.bipolar_gate_max is None:
        return gate
    min_value = -float("inf") if cfg.bipolar_gate_min is None else float(cfg.bipolar_gate_min)
    max_value = float("inf") if cfg.bipolar_gate_max is None else float(cfg.bipolar_gate_max)
    return gate.clamp(min=min_value, max=max_value)


def _scheduled_bipolar_mu(module: nn.Module, cfg: ShiftmaxAttentionConfig) -> float:
    final_mu = float(cfg.bipolar_mu)
    if not cfg.sc_mu_schedule_enabled:
        return final_mu
    step = getattr(module, "_h9_global_step", None)
    if step is None:
        return final_mu
    step = int(step)
    start_step = max(0, int(cfg.sc_mu_start_step))
    warmup_steps = max(0, int(cfg.sc_mu_warmup_steps))
    start_mu = float(cfg.sc_mu_start)
    if step <= start_step:
        return start_mu
    if warmup_steps <= 0:
        return final_mu
    progress = min(1.0, max(0.0, float(step - start_step) / float(warmup_steps)))
    return start_mu + (final_mu - start_mu) * progress


def _sc_agree_disagree_gate(
    scores: torch.Tensor,
    n_tokens: int,
    head_dim: int,
    cfg: ShiftmaxAttentionConfig,
    q_event: torch.Tensor | None = None,
    k_event: torch.Tensor | None = None,
) -> torch.Tensor:
    """SC-native signed gate: split popcount score into agree/disagree.

    agree    = max(score, 0)   — same-polarity evidence
    disagree = max(-score, 0)  — opposite-polarity evidence

    gate = Shiftmax(agree) - λ × Shiftmax(disagree)

    Optional add-ons (all SC-native, no extra scoring):
      - deadzone: |score|<ε tokens get uniform 1/N weight
      - confidence: low-activity tokens regress toward 1/N
    """

    agree = torch.relu(scores)
    disagree = torch.relu(-scores)

    dead_mask = None
    dead_fraction = 0.0
    if cfg.deadzone_epsilon > 0:
        dead_mask = scores.abs() < float(cfg.deadzone_epsilon)
        dead_fraction = dead_mask.float().sum(dim=2, keepdim=True) / float(n_tokens)
        agree = agree * (~dead_mask).float()
        disagree = disagree * (~dead_mask).float()

    agree_gate = shiftmax(agree, dim=2, eps=cfg.eps)
    disagree_gate = shiftmax(disagree, dim=2, eps=cfg.eps)
    if cfg.preserve_mean:
        agree_gate = agree_gate * float(n_tokens)
        disagree_gate = disagree_gate * float(n_tokens)

    live_scale = 1.0 - dead_fraction
    gate = (agree_gate - float(cfg.bipolar_lambda) * disagree_gate) * live_scale

    if dead_mask is not None:
        gate = gate + dead_mask.float() * (1.0 / float(n_tokens))

    if cfg.confidence_enabled and q_event is not None and k_event is not None:
        q_active = q_event.ne(0)
        k_active = k_event.ne(0)
        active = (q_active | k_active).sum(dim=-1, keepdim=True).float().clamp_min(1)
        confidence = torch.sqrt(active / float(head_dim))
        gate = confidence * gate + (1.0 - confidence) * (1.0 / float(n_tokens))

    return _maybe_clamp_bipolar_gate(gate, cfg)


def _a2os2a_token_scores(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    """A2OS2A-inspired hybrid Q/K score for SDFormerFlow's no-V attention.

    CVPR 2025 A2OS2A uses binary Q, ReLU/nonnegative K, and ternary V, with no
    softmax/scaling. This adapter keeps the native QKFormer carrier elsewhere,
    but computes a compatible auxiliary gate from binary Q and nonnegative K.
    """

    q_event = (_qkformer_token_q(q_orig) > 0).to(dtype=q_orig.dtype)
    k_nonnegative = k_orig.clamp_min(float(cfg.relu_k_floor))
    score = (q_event * k_nonnegative).sum(dim=-1, keepdim=True)
    return _normalize_consensus_score(score, q_event.shape[-1], cfg, active=None)


def _ternary_alpha_xnor_matrix_scores(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    """Token-token alpha-XNOR matrix for signed ternary Q/K events."""

    q_event = _ternary_sign_ste(_qkformer_token_q(q_orig))
    k_event = _ternary_sign_ste(k_orig)
    q_pos = q_event.gt(0).to(dtype=q_orig.dtype)
    q_neg = q_event.lt(0).to(dtype=q_orig.dtype)
    q_zero = q_event.eq(0).to(dtype=q_orig.dtype)
    k_pos = k_event.gt(0).to(dtype=q_orig.dtype)
    k_neg = k_event.lt(0).to(dtype=q_orig.dtype)
    k_zero = k_event.eq(0).to(dtype=q_orig.dtype)
    same_nonzero = torch.matmul(q_pos, k_pos.transpose(-2, -1)) + torch.matmul(
        q_neg, k_neg.transpose(-2, -1)
    )
    same_zero = torch.matmul(q_zero, k_zero.transpose(-2, -1))
    opposite = torch.matmul(q_pos, k_neg.transpose(-2, -1)) + torch.matmul(q_neg, k_pos.transpose(-2, -1))
    score = same_nonzero + float(cfg.alpha0) * same_zero - float(cfg.mismatch_penalty) * opposite
    q_active = q_event.ne(0).to(dtype=q_orig.dtype)
    k_active = k_event.ne(0).to(dtype=q_orig.dtype)
    if cfg.single_active_penalty:
        single_active = torch.matmul(q_active, k_zero.transpose(-2, -1)) + torch.matmul(
            q_zero, k_active.transpose(-2, -1)
        )
        score = score - float(cfg.single_active_penalty) * single_active
    active = None
    if cfg.consensus_score_norm == "active":
        both_active = torch.matmul(q_active, k_active.transpose(-2, -1))
        if cfg.single_active_penalty:
            q_active_count = q_active.sum(dim=-1, keepdim=True)
            k_active_count = k_active.sum(dim=-1, keepdim=True).transpose(-2, -1)
            active = q_active_count + k_active_count - both_active
        else:
            active = both_active
    return _normalize_consensus_score(score, q_event.shape[-1], cfg, active=active)


def _ternary_alpha_xnor_matrix_scores_ste(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    """Differentiable ternary alpha-XNOR matrix for direct SSA branches.

    Boolean comparisons make the alpha-XNOR score non-differentiable with
    respect to Q/K. For H42 direct-attention modes the score is the main
    attention object, so keep the same forward values using ReLU arithmetic on
    STE ternary events while preserving surrogate gradients.
    """

    q_event = _ternary_sign_ste(_qkformer_token_q(q_orig))
    k_event = _ternary_sign_ste(k_orig)
    q_pos = torch.relu(q_event)
    q_neg = torch.relu(-q_event)
    k_pos = torch.relu(k_event)
    k_neg = torch.relu(-k_event)
    q_zero = torch.clamp(1.0 - q_event.abs(), min=0.0, max=1.0)
    k_zero = torch.clamp(1.0 - k_event.abs(), min=0.0, max=1.0)
    same_nonzero = torch.matmul(q_pos, k_pos.transpose(-2, -1)) + torch.matmul(
        q_neg, k_neg.transpose(-2, -1)
    )
    same_zero = torch.matmul(q_zero, k_zero.transpose(-2, -1))
    opposite = torch.matmul(q_pos, k_neg.transpose(-2, -1)) + torch.matmul(q_neg, k_pos.transpose(-2, -1))
    score = same_nonzero + float(cfg.alpha0) * same_zero - float(cfg.mismatch_penalty) * opposite
    q_active = q_pos + q_neg
    k_active = k_pos + k_neg
    if cfg.single_active_penalty:
        single_active = torch.matmul(q_active, k_zero.transpose(-2, -1)) + torch.matmul(
            q_zero, k_active.transpose(-2, -1)
        )
        score = score - float(cfg.single_active_penalty) * single_active
    active = None
    if cfg.consensus_score_norm == "active":
        both_active = torch.matmul(q_active, k_active.transpose(-2, -1))
        if cfg.single_active_penalty:
            q_active_count = q_active.sum(dim=-1, keepdim=True)
            k_active_count = k_active.sum(dim=-1, keepdim=True).transpose(-2, -1)
            active = q_active_count + k_active_count - both_active
        else:
            active = both_active
    return _normalize_consensus_score(score, q_event.shape[-1], cfg, active=active)


def _add_matrix_diag_bias(scores: torch.Tensor, cfg: ShiftmaxAttentionConfig) -> torch.Tensor:
    """Add a same-token prior to direct token-token attention scores."""

    bias = float(cfg.matrix_diag_bias)
    if bias == 0.0:
        return scores
    n_query, n_key = scores.shape[-2], scores.shape[-1]
    diag = torch.eye(n_query, n_key, device=scores.device, dtype=scores.dtype)
    return scores + bias * diag


def _binary_alpha_xnor_matrix_scores(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    """Binary alpha-XNOR token-token score without ternary polarity semantics.

    This is the conservative paper-facing variant: positive events are 1,
    everything else is 0. Silent/silent matches receive alpha weight; mismatch
    penalty is configurable but set to 0 in strict binary configs.
    """

    q_event = (_qkformer_token_q(q_orig) > 0).to(dtype=q_orig.dtype)
    k_event = (k_orig > 0).to(dtype=q_orig.dtype)
    q_silent = 1.0 - q_event
    k_silent = 1.0 - k_event
    same_spike = torch.matmul(q_event, k_event.transpose(-2, -1))
    same_silent = torch.matmul(q_silent, k_silent.transpose(-2, -1))
    score = same_spike + float(cfg.alpha0) * same_silent
    if cfg.mismatch_penalty:
        mismatch = torch.matmul(q_event, k_silent.transpose(-2, -1)) + torch.matmul(
            q_silent, k_event.transpose(-2, -1)
        )
        score = score - float(cfg.mismatch_penalty) * mismatch
    active = None
    if cfg.consensus_score_norm == "active":
        active = torch.matmul(q_event, k_event.transpose(-2, -1))
    return _normalize_consensus_score(score, q_event.shape[-1], cfg, active=active)


def _a2os2a_matrix_scores(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    """A2OS2A-style token-token score: binary Q against nonnegative K."""

    q_event = (_qkformer_token_q(q_orig) > 0).to(dtype=q_orig.dtype)
    k_nonnegative = k_orig.clamp_min(float(cfg.relu_k_floor))
    score = torch.matmul(q_event, k_nonnegative.transpose(-2, -1))
    return _normalize_consensus_score(score, q_event.shape[-1], cfg, active=None)


def _hamming_linear_attention(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
    ternary_active: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """SpikeVideoFormer-style Hamming attention adapted to no-V QK blocks.

    Official SpikeVideoFormer uses binary spikes:

        x = (2K - 1)^T V
        x = (2Q - 1) x / (2 * dim)

    SDFormerFlow has no separate V path, so K is reused as the value stream.
    `ternary_active=True` keeps silence at zero and only uses polarity for
    active events; the default follows the paper's binary 0/1 to -1/+1 mapping.
    """

    q_token = _qkformer_token_q(q_orig)
    if ternary_active:
        q_h = _ternary_sign_ste(q_token)
        k_h = _ternary_sign_ste(k_orig)
    else:
        q_h = (q_token > 0).to(dtype=q_orig.dtype).mul(2.0).sub(1.0)
        k_h = (k_orig > 0).to(dtype=q_orig.dtype).mul(2.0).sub(1.0)
    value = _ternary_sign_ste(k_orig) if cfg.value_mode in {"sign", "event", "ternary"} else k_orig
    kv = torch.matmul(k_h.transpose(-2, -1), value)
    attn = torch.matmul(q_h, kv) / float(max(1, 2 * q_h.shape[-1]))
    row_sum = q_h.abs().sum(dim=-1)
    gate = q_h
    return attn, row_sum, gate


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

    if cfg.mode in {"strict_bsa_qkv_shiftmax", "bsa_qkv_shiftmax", "bsa_true_qkv_shiftmax"}:
        # Reviewed strict-BSA candidate: BSA-style ternary matrix product over
        # QK^T with a separate trainable V branch. SDFormerFlow ships this
        # branch commented out, so the overlay attaches it after checkpoint load
        # and initializes from K for a stable fine-tuning start.
        value_orig = _independent_value_tokens(self, x, T, B_, H, W, C, head_dim, cfg)
        attn, row_sum, gate = _strict_bsa_matrix_attention(q_orig, k_orig, cfg, value_orig=value_orig)
    elif cfg.mode in {"strict_bsa_shiftmax", "bsa_matrix_shiftmax", "bsa_qkt_shiftmax"}:
        # BSA-style adapted candidate: ternary QK^T with K reused as V because
        # the original SDFormerFlow attention block has no live V projection.
        #
        # Unlike H13's token-wise consensus gate, the normalized object here is
        # a real token-token attention matrix whose entries come from sign-only
        # ternary events. Because SDFormerFlow's block has no separate V branch,
        # K is reused as V; cfg.value_mode chooses sign-only K or thresholded K.
        attn, row_sum, gate = _strict_bsa_matrix_attention(q_orig, k_orig, cfg)
    elif cfg.mode in {"binary_alpha_xnor_matrix_shiftmax", "strict_binary_alpha_xnor_shiftmax"}:
        # Conservative alpha-XNOR reproduction: binary positive spike events
        # only, no ternary negative polarity semantics. This is separate from
        # the more aggressive ternary alpha-XNOR adaptation used by H18/H34.
        scores = _binary_alpha_xnor_matrix_scores(q_orig, k_orig, cfg)
        if cfg.center_scores:
            scores = scores - scores.mean(dim=-1, keepdim=True)
        gate = shiftmax(scores, dim=-1, eps=cfg.eps)
        row_sum = gate.sum(dim=-1)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        value = _ternary_sign_ste(k_orig) if cfg.value_mode in {"sign", "event", "ternary"} else k_orig
        attn = torch.matmul(gate, value)
    elif cfg.mode in {"binary_alpha_xnor_matrix_l1", "strict_binary_alpha_xnor_l1"}:
        scores = _binary_alpha_xnor_matrix_scores(q_orig, k_orig, cfg)
        gate = l1norm(torch.relu(scores + cfg.consensus_bias), dim=-1, eps=cfg.eps)
        row_sum = gate.sum(dim=-1)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        value = _ternary_sign_ste(k_orig) if cfg.value_mode in {"sign", "event", "ternary"} else k_orig
        attn = torch.matmul(gate, value)
    elif cfg.mode in {"alpha_xnor_matrix_shiftmax", "ternary_alpha_xnor_matrix_shiftmax", "h18c"}:
        # Direct H18c: replace QKFormer's token carrier with a true
        # token-token alpha-XNOR attention matrix. This is intentionally bold:
        # the original carrier is not preserved, and K is reused as V because
        # the baseline QK block has no separate value projection.
        scores = _ternary_alpha_xnor_matrix_scores(q_orig, k_orig, cfg)
        scores = _add_matrix_diag_bias(scores, cfg)
        if cfg.center_scores:
            scores = scores - scores.mean(dim=-1, keepdim=True)
        gate = shiftmax(scores, dim=-1, eps=cfg.eps)
        row_sum = gate.sum(dim=-1)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        value = _ternary_sign_ste(k_orig) if cfg.value_mode in {"sign", "event", "ternary"} else k_orig
        attn = torch.matmul(gate, value)
    elif cfg.mode in {
        "ternary_alpha_xnor_ssa_linear",
        "alpha_xnor_ssa_linear",
        "ternary_alpha_xnor_ssa",
        "h42b",
    }:
        # Paper-facing ternary alpha-XNOR SSA adaptation.
        #
        # TX/H18a keeps SDFormerFlow's original QKFormer token carrier and uses
        # alpha-XNOR as an auxiliary gate. This branch is cleaner: alpha-XNOR
        # similarity is the attention object itself, followed by a linear
        # score transform and a value matmul. The baseline block has no live V
        # branch, so K is intentionally reused as V.
        scores = _ternary_alpha_xnor_matrix_scores_ste(q_orig, k_orig, cfg)
        scores = _add_matrix_diag_bias(scores, cfg)
        if cfg.center_scores:
            scores = scores - scores.mean(dim=-1, keepdim=True)
        gate = scores + float(cfg.consensus_bias)
        row_sum = gate.sum(dim=-1)
        value = _ternary_sign_ste(k_orig) if cfg.value_mode in {"sign", "event", "ternary"} else k_orig
        attn = torch.matmul(gate, value)
    elif cfg.mode in {
        "ternary_alpha_xnor_ssa_qkv_linear",
        "alpha_xnor_ssa_qkv_linear",
        "ternary_alpha_xnor_qkv",
        "h42c",
    }:
        # QKV version of the paper-facing ternary alpha-XNOR SSA adaptation.
        #
        # Q/K build the ternary alpha-XNOR similarity matrix, while V is a
        # separate trainable branch attached by the overlay. The branch is
        # initialized from K when installed, giving stable baseline continuation
        # while allowing V to specialize during fine-tuning.
        scores = _ternary_alpha_xnor_matrix_scores_ste(q_orig, k_orig, cfg)
        scores = _add_matrix_diag_bias(scores, cfg)
        if cfg.center_scores:
            scores = scores - scores.mean(dim=-1, keepdim=True)
        gate = scores + float(cfg.consensus_bias)
        row_sum = gate.sum(dim=-1)
        value_orig = _independent_value_tokens(self, x, T, B_, H, W, C, head_dim, cfg)
        value = _ternary_sign_ste(value_orig) if cfg.value_mode in {"sign", "event", "ternary"} else value_orig
        attn = torch.matmul(gate, value)
    elif cfg.mode in {
        "ternary_alpha_xnor_ssa_qkv_shiftmax",
        "alpha_xnor_ssa_qkv_shiftmax",
        "h42d",
    }:
        # Standard Shiftmax QKV version of ternary alpha-XNOR SSA.
        #
        # This keeps the paper-facing Q/K/V attention structure while using the
        # stable BSA-style Shiftmax normalization already used by existing H
        # experiments. V is an independent overlay branch initialized from K.
        scores = _ternary_alpha_xnor_matrix_scores_ste(q_orig, k_orig, cfg)
        scores = _add_matrix_diag_bias(scores, cfg)
        if cfg.center_scores:
            scores = scores - scores.mean(dim=-1, keepdim=True)
        gate = shiftmax(scores, dim=-1, eps=cfg.eps)
        row_sum = gate.sum(dim=-1)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        value_orig = _independent_value_tokens(self, x, T, B_, H, W, C, head_dim, cfg)
        value = _ternary_sign_ste(value_orig) if cfg.value_mode in {"sign", "event", "ternary"} else value_orig
        attn = torch.matmul(gate, value)
    elif cfg.mode in {
        "ternary_alpha_xnor_ssa_kreuse_shiftmax",
        "alpha_xnor_ssa_kreuse_shiftmax",
        "h45",
    }:
        # H45: same ternary alpha-XNOR + Shiftmax attention as H44, but reuse K
        # as the value stream. This removes the independent V branch while
        # preserving the relaxed ATLIF/training setup for a controlled ablation.
        scores = _ternary_alpha_xnor_matrix_scores_ste(q_orig, k_orig, cfg)
        if cfg.center_scores:
            scores = scores - scores.mean(dim=-1, keepdim=True)
        gate = shiftmax(scores, dim=-1, eps=cfg.eps)
        row_sum = gate.sum(dim=-1)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        value = _ternary_sign_ste(k_orig) if cfg.value_mode in {"sign", "event", "ternary"} else k_orig
        attn = torch.matmul(gate, value)
    elif cfg.mode in {"alpha_xnor_matrix_l1", "ternary_alpha_xnor_matrix_l1", "h18d"}:
        # Direct H18d: same alpha-XNOR matrix, but with add/L1 normalization
        # instead of Shiftmax. This is the hardware-cleanest alpha-XNOR test.
        scores = _ternary_alpha_xnor_matrix_scores(q_orig, k_orig, cfg)
        gate = l1norm(torch.relu(scores + cfg.consensus_bias), dim=-1, eps=cfg.eps)
        row_sum = gate.sum(dim=-1)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        value = _ternary_sign_ste(k_orig) if cfg.value_mode in {"sign", "event", "ternary"} else k_orig
        attn = torch.matmul(gate, value)
    elif cfg.mode in {"a2os2a_direct", "a2os2a_matrix_l1", "h18e"}:
        # Direct H18e: A2OS2A-style matrix replacement.
        #
        # Q is binary, K is nonnegative, V is represented by the thresholded
        # ternary K stream. No Shiftmax/softmax is used; scores are L1
        # normalized to keep magnitude bounded inside this no-V SDFormer block.
        scores = _a2os2a_matrix_scores(q_orig, k_orig, cfg)
        gate = l1norm(torch.relu(scores + cfg.consensus_bias), dim=-1, eps=cfg.eps)
        row_sum = gate.sum(dim=-1)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        value = _ternary_sign_ste(k_orig) if cfg.value_mode in {"sign", "event", "ternary"} else k_orig
        attn = torch.matmul(gate, value)
    elif cfg.mode in {"a2os2a_qkv_l1", "a2os2a_true_qkv_l1"}:
        # Reviewed A2OS2A-style QKV candidate: binary Q, nonnegative K, and an
        # independent ternary/threshold V stream. It is still an SDFormerFlow
        # adaptation, but unlike H18e it exercises all three Q/K/V paths.
        scores = _a2os2a_matrix_scores(q_orig, k_orig, cfg)
        gate = l1norm(torch.relu(scores + cfg.consensus_bias), dim=-1, eps=cfg.eps)
        row_sum = gate.sum(dim=-1)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        value_orig = _independent_value_tokens(self, x, T, B_, H, W, C, head_dim, cfg)
        value = _ternary_sign_ste(value_orig) if cfg.value_mode in {"sign", "event", "ternary"} else value_orig
        attn = torch.matmul(gate, value)
    elif cfg.mode in {"hamming_binary_direct", "spikevideoformer_hamming", "h21a"}:
        # H21a: direct SpikeVideoFormer Hamming attention.
        #
        # This is not QKFormer carrier-preserving and not N^2 softmax. It
        # follows the official ICML 2025 code path with binary {0,1} spikes
        # mapped to {-1,+1}; K is reused as V in this no-V block.
        attn, row_sum, gate = _hamming_linear_attention(q_orig, k_orig, cfg, ternary_active=False)
    elif cfg.mode in {"hamming_ternary_active_direct", "ternary_hamming_direct", "h21b"}:
        # H21b: ternary-safe Hamming attention.
        #
        # Same linear Hamming structure, but silence remains 0 and only active
        # signed events contribute. This avoids treating absence of events as a
        # strong negative signal, which is a risk for sparse event data.
        attn, row_sum, gate = _hamming_linear_attention(q_orig, k_orig, cfg, ternary_active=True)
    elif cfg.mode in {"qk_bsa", "bsa_qk", "ternary_matrix"}:
        # BSA-style QK path for SDFormerFlow's no-V attention block.
        #
        # The normalized object is a real attention matrix built from ternary
        # Q/K events. Since the baseline module has no V projection, the ternary
        # K carrier is used as the value stream:
        #
        #   scores = Q_ternary @ K_ternary^T
        #   weights = Shiftmax(scores)
        #   attn = weights @ K_ternary
        #
        # This is not the original QKFormer carrier; it is the closest BSA-style
        # replacement that keeps the local module interface unchanged.
        q_token = q_orig.permute(1, 2, 0, 3, 4).reshape(B_, self.num_heads, n_tokens, head_dim)
        scores = torch.matmul(q_token, k_orig.transpose(-2, -1)) * cfg.score_scale
        if cfg.center_scores:
            scores = scores - scores.mean(dim=-1, keepdim=True)
        gate = shiftmax(scores, dim=-1, eps=cfg.eps)
        row_sum = gate.sum(dim=-1)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        attn = torch.matmul(gate, k_orig)
    elif cfg.mode in {"qkformer_spike_shift", "spike_shift"}:
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
    elif cfg.mode in {"signed_consensus_shiftmax", "consensus_shiftmax", "h13b"}:
        # H13b: ternary-native token gating.
        #
        # The score is a hardware-friendly signed popcount over Q/K events:
        # same polarity contributes +1, opposite polarity -1, silence/silence 0,
        # and optional single-active mismatches get their own penalty. Shiftmax is
        # retained as the BSA-style normalization, but its input is now a
        # sign-consensus score instead of a theta-weighted real-valued product.
        scores = _signed_consensus_token_scores(q_orig, k_orig, cfg)
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        gate = shiftmax(scores, dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        attn = k_orig.mul(gate)
    elif cfg.mode in {"signed_consensus_shiftmax_raw", "consensus_shiftmax_raw", "h13b_raw"}:
        # SC raw-Shiftmax ablation: keep signed-consensus scores, but remove
        # the max-subtraction used by numerically stable Shiftmax.
        scores = _signed_consensus_token_scores(q_orig, k_orig, cfg)
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        gate = shiftmax_raw(scores, dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        attn = k_orig.mul(gate)
    elif cfg.mode in {"signed_consensus_shiftnorm", "consensus_shiftnorm", "h13c"}:
        # H13c: fully shift-normalized variant.
        #
        # This keeps the same signed consensus score, but replaces 2^score with
        # a ReLU/bias numerator and a next-power-of-two denominator. It is less
        # expressive than Shiftmax but maps to popcount + add/sub + shift.
        scores = _signed_consensus_token_scores(q_orig, k_orig, cfg)
        nonnegative_scores = torch.relu(scores + cfg.consensus_bias)
        gate = shiftnorm(nonnegative_scores, dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        attn = k_orig.mul(gate)
    elif cfg.mode in {"signed_consensus_popcount_l1", "popcount_l1", "soc_l1", "h13t"}:
        # H13t: no Shiftmax, no exponent/LUT.
        #
        # Convert signed consensus into a nonnegative popcount-style evidence
        # score, then use exact L1 normalization across tokens. This isolates
        # whether H13's gains require Shiftmax or mainly need signed popcount.
        scores = _signed_consensus_token_scores(q_orig, k_orig, cfg)
        nonnegative_scores = torch.relu(scores + cfg.consensus_bias)
        gate = l1norm(nonnegative_scores, dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        attn = k_orig.mul(gate)
    elif cfg.mode in {"ternary_alpha_xnor_qkselector_shiftmax", "tx_qkselector_shiftmax", "h49"}:
        # H49: QKFormer-native ternary selector.
        #
        # Unlike H45/H47, this does not form an N x N attention matrix and does
        # not mix K across tokens. It replaces QKFormer's Q-only token selector
        # with a same-token ternary Q/K consistency selector:
        #
        #   score_i = TX(q_i, k_i)
        #   selector = Shiftmax(score over tokens)
        #   attn_i = k_i * selector_i
        #
        # This keeps the linear-complexity QKFormer carrier while avoiding the
        # older H41 pattern of multiplying the native sn2(sum(Q)) gate by an
        # extra TX gate.
        # ── NTX-11 stage-β: override mismatch_penalty per stage (additive) ──
        _stage_betas = getattr(cfg, "stage_mismatch_penalty", None)
        if _stage_betas is not None:
            _stage = getattr(self, "_h9_stage", 0)
            if _stage < len(_stage_betas):
                _override_beta = float(_stage_betas[_stage])
            else:
                _override_beta = float(cfg.mismatch_penalty)
        else:
            _override_beta = float(cfg.mismatch_penalty)
        scores = _ternary_alpha_xnor_token_scores(q_orig, k_orig, cfg, beta=_override_beta)
        # ── end stage-β ──
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        gate = shiftmax(scores, dim=2, eps=cfg.eps)
        # ── NTX-11 gate smooth: EMA across timesteps (optional, additive) ──
        _alpha = getattr(cfg, "gate_smooth_alpha", 0.0)
        if _alpha > 0 and _alpha < 1:
            _prev = getattr(self, "_h9_prev_gate", None)
            if _prev is not None and _prev.shape == gate.shape:
                gate = _alpha * gate + (1.0 - _alpha) * _prev
            self._h9_prev_gate = gate.detach()
        # ── end gate smooth ──
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        attn = k_orig.mul(gate)
    elif cfg.mode in {"ternary_alpha_xnor_local_shiftmax", "tx_local_shiftmax", "h59_local"}:
        # H59_local / NTX-10: H49 selector + local spatial neighbor interaction.
        #
        # Same-token TX selector as H49, augmented with a lightweight
        # pairwise consistency term over spatial neighbors within the SWIN window:
        #
        #   self_i    = TX(q_i, k_i)
        #   neighbor_i = mean_{j in N(i)} TX(q_i, k_j)   [local pairwise]
        #   score_i   = self_i + lambda * neighbor_i
        #   selector  = Shiftmax(score)
        #   attn_i    = k_i * selector_i
        #
        # The window_size = [T, H, W] spatial layout is recovered to identify
        # local neighbors. lambda=0 degenerates to plain H49.  No new parameters.
        scores = _ternary_alpha_xnor_token_scores(q_orig, k_orig, cfg)
        # ── local neighbor interaction ──
        local_lambda = float(getattr(cfg, "local_lambda", 0.2))
        if local_lambda > 0:
            import math
            N = scores.shape[2]  # total tokens in window
            T_dim, H_dim, W_dim = 2, 9, 9  # SWIN window_size
            if T_dim * H_dim * W_dim == N:
                try:
                    # scores may have leading T dim (from q_orig) or not.
                    # Reshape to ensure (T, B_, heads, H, W, 1)
                    if scores.shape[0] == T_dim:
                        ss = scores.reshape(T_dim, -1, scores.shape[1], H_dim, W_dim, 1)
                    else:
                        ss = scores.reshape(-1, scores.shape[1], T_dim, H_dim, W_dim, 1).permute(2, 0, 1, 3, 4, 5)
                    # Neighbor average: roll along H dim (3) and W dim (4)
                    nbr = torch.zeros_like(ss)
                    n = 0
                    for d in (1, -1):
                        nbr = nbr + torch.roll(ss, shifts=d, dims=3)  # H neighbors
                        nbr = nbr + torch.roll(ss, shifts=d, dims=4)  # W neighbors
                        n += 2
                    nbr = nbr / n
                    # Flatten back to match scores shape
                    if scores.shape[0] == T_dim:
                        nbr = nbr.reshape(scores.shape)
                    else:
                        nbr = nbr.permute(1, 2, 0, 3, 4, 5).reshape(scores.shape)
                    scores = scores + local_lambda * nbr
                except Exception:
                    pass
        # ── end local interaction ──
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        gate = shiftmax(scores, dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        attn = k_orig.mul(gate)
    elif cfg.mode in {"bipolar_qkselector_shiftmax", "tx_bipolar_two_score_shiftmax", "h54a"}:
        # H54a: two-score bipolar selector.
        #
        # Split ternary TX evidence into same-polarity and opposite-polarity
        # Shiftmax branches. The final modulation is signed:
        #
        #   gate = g_same - lambda * g_opp
        #   attn = gate * K
        #
        # Shiftmax itself remains nonnegative and hardware-friendly, but the
        # effective K carrier can now be attenuated or polarity-flipped.
        _, same_scores, opp_scores = _bipolar_token_score_components(q_orig, k_orig, cfg)
        same_scores = _center_token_scores(same_scores, cfg)
        opp_scores = _center_token_scores(opp_scores, cfg)
        same_gate = shiftmax(same_scores, dim=2, eps=cfg.eps)
        opp_gate = shiftmax(opp_scores, dim=2, eps=cfg.eps)
        if cfg.preserve_mean:
            same_gate = same_gate * float(n_tokens)
            opp_gate = opp_gate * float(n_tokens)
        gate = same_gate - float(cfg.bipolar_lambda) * opp_gate
        gate = _maybe_clamp_bipolar_gate(gate, cfg)
        row_sum = gate.abs().sum(dim=2)
        scores = same_scores - float(cfg.bipolar_lambda) * opp_scores
        attn = k_orig.mul(gate)
    elif cfg.mode in {"tx_bipolar_qkselector_shiftmax", "tx_bipolar_three_score_shiftmax", "h54b"}:
        # H54b: three-score TX + bipolar correction selector.
        #
        # The normal TX selector stays as the stable carrier, while same/opposite
        # evidence supplies a signed correction:
        #
        #   gate = g_tx + mu * (g_same - lambda * g_opp)
        #
        # This can still flip K when the opposite evidence is strong, but it
        # degenerates back to H49 when mu=0.
        tx_scores, same_scores, opp_scores = _bipolar_token_score_components(q_orig, k_orig, cfg)
        tx_scores = _center_token_scores(tx_scores, cfg)
        same_scores = _center_token_scores(same_scores, cfg)
        opp_scores = _center_token_scores(opp_scores, cfg)
        tx_gate = shiftmax(tx_scores, dim=2, eps=cfg.eps)
        same_gate = shiftmax(same_scores, dim=2, eps=cfg.eps)
        opp_gate = shiftmax(opp_scores, dim=2, eps=cfg.eps)
        if cfg.preserve_mean:
            tx_gate = tx_gate * float(n_tokens)
            same_gate = same_gate * float(n_tokens)
            opp_gate = opp_gate * float(n_tokens)
        gate = tx_gate + float(cfg.bipolar_mu) * (same_gate - float(cfg.bipolar_lambda) * opp_gate)
        gate = _maybe_clamp_bipolar_gate(gate, cfg)
        row_sum = gate.abs().sum(dim=2)
        scores = tx_scores + float(cfg.bipolar_mu) * (same_scores - float(cfg.bipolar_lambda) * opp_scores)
        attn = k_orig.mul(gate)
    elif cfg.mode in {"dual_channel_qkselector_shiftmax", "excite_inhibit_qkselector_shiftmax", "h51"}:
        # H51: signed dual-channel selector.
        #
        # It keeps H49's linear token selector form, but separates positive and
        # negative spikes into excitation/inhibition evidence. The K carrier is
        # still present, so this is a conservative attention change rather than
        # a direct QKV replacement.
        scores = _dual_channel_token_scores(q_orig, k_orig, cfg)
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        gate = shiftmax(scores, dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        attn = k_orig.mul(gate)
    elif cfg.mode in {"a2os2a_kasv_shiftmax", "key_as_value_a2os2a_shiftmax", "kasv_shiftmax", "h52"}:
        # H52: Key-as-Proxy V A2OS2A adapter.
        #
        # A2OS2A's useful inductive bias is binary Q against nonnegative K. To
        # avoid H47's unstable independent V branch, this branch uses K itself
        # as the value proxy. It is a short-test candidate, not the current
        # full-training mainline.
        scores = _a2os2a_matrix_scores(q_orig, k_orig, cfg)
        if cfg.center_scores:
            scores = scores - scores.mean(dim=-1, keepdim=True)
        gate = shiftmax(scores, dim=-1, eps=cfg.eps)
        row_sum = gate.sum(dim=-1)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        value = _ternary_sign_ste(k_orig) if cfg.value_mode in {"sign", "event", "ternary"} else k_orig
        attn = torch.matmul(gate, value)
    elif cfg.mode in {"sc_agree_disagree_shiftmax", "h56a"}:
        # H56a: SC-native agree/disagree signed gate.
        #
        # The baseline SC score = Σ sign(q)·sign(k)/d is a clean agree-minus-
        # disagree net. This mode splits it by sign, runs two Shiftmax branches,
        # and subtracts: gate = g_agree - λ·g_disagree.
        #
        # Unlike TX bipolar (H54b), no extra score computation — just split the
        # existing popcount score.
        scores = _signed_consensus_token_scores(q_orig, k_orig, cfg)
        scores = _center_token_scores(scores, cfg)
        gate = _sc_agree_disagree_gate(scores, n_tokens, head_dim, cfg)
        row_sum = gate.abs().sum(dim=2)
        attn = k_orig.mul(gate)
    elif cfg.mode in {"sc_agree_disagree_residual_shiftmax", "h56r"}:
        # H56r: residual SC agree/disagree gate.
        #
        # Keep the original QKFormer carrier and use SC agree/disagree as a
        # bounded modulation. This tests whether H56a failed because the pure
        # signed gate replaced the carrier too aggressively.
        att_token = q_orig.sum(dim=-1, keepdim=True)
        att_token = self.sn2_q(att_token)
        attn_carrier = k_orig.mul(att_token.reshape(B_, self.num_heads, n_tokens, 1))

        scores = _signed_consensus_token_scores(q_orig, k_orig, cfg)
        scores = _center_token_scores(scores, cfg)
        gate = _sc_agree_disagree_gate(scores, n_tokens, head_dim, cfg)
        row_sum = gate.abs().sum(dim=2)
        alpha = float(cfg.residual_alpha)
        attn = attn_carrier * (1.0 + alpha * (gate - 1.0))
    elif cfg.mode in {"sc_ad_carrier_blend_shiftmax", "sc_agree_disagree_carrier_blend_shiftmax", "h56m"}:
        # H56m: carrier/gate blend for the repaired SC route.
        #
        # H56r modulates the native QKFormer carrier multiplicatively, so an
        # inactive carrier cannot be corrected by SC evidence. This branch keeps
        # the same no-new-weights SC score, but blends the native carrier token
        # with the signed agree/disagree gate before multiplying K:
        #
        #   output = K * ((1 - mu) * carrier_q + mu * sc_signed_gate)
        #
        # It is compatible with old checkpoints and only activates when a new
        # config explicitly selects this mode.
        att_token = q_orig.sum(dim=-1, keepdim=True)
        att_token = self.sn2_q(att_token)
        carrier_gate = att_token.reshape(B_, self.num_heads, n_tokens, 1)

        scores = _signed_consensus_token_scores(q_orig, k_orig, cfg)
        scores = _center_token_scores(scores, cfg)
        sc_gate = _sc_agree_disagree_gate(scores, n_tokens, head_dim, cfg)
        mu = float(cfg.bipolar_mu)
        gate = (1.0 - mu) * carrier_gate + mu * sc_gate
        gate = _maybe_clamp_bipolar_gate(gate, cfg)
        row_sum = gate.abs().sum(dim=2)
        attn = k_orig.mul(gate)
    elif cfg.mode in {
        "tx_sc_residual_selector_shiftmax",
        "tx_sc_hybrid_selector_shiftmax",
        "tx_sc_late_residual_selector_shiftmax",
        "tx_sc_score_residual_shiftmax",
        "h57",
        "h57a",
        "h58",
        "h58a",
        "h59",
        "h59a",
    }:
        # H57: NTX-01-compatible TX selector with a small SC residual.
        #
        # NSC-04/05 showed that replacing the carrier with a signed SC gate is
        # too aggressive. This branch keeps the same native QKFormer carrier as
        # NTX-01, keeps the TX gate as the stable base, and only blends in SC
        # agree/disagree as a residual selector:
        #
        #   carrier = K * sn2_q(sum(Q))
        #   gate    = (1 - mu) * TX(Q,K) + mu * SC_agree_disagree(Q,K)
        #   output  = carrier * gate
        #
        # mu=0 is the NTX-01/TX gate; small mu tests whether SC evidence helps
        # without forcing a new module to learn the whole attention behavior.
        att_token = q_orig.sum(dim=-1, keepdim=True)
        att_token = self.sn2_q(att_token)
        att_gate = att_token.reshape(B_, self.num_heads, n_tokens, 1)
        attn_carrier = k_orig.mul(att_gate)

        tx_scores, sc_scores = _tx_sc_fusion_score_pair(q_orig, k_orig, cfg)
        tx_scores = _center_token_scores(tx_scores, cfg)
        tx_gate = shiftmax(tx_scores, dim=2, eps=cfg.eps)
        if cfg.preserve_mean:
            tx_gate = tx_gate * float(n_tokens)

        sc_scores = _center_token_scores(sc_scores, cfg)
        mu = _scheduled_bipolar_mu(self, cfg)
        if cfg.mode in {"tx_sc_score_residual_shiftmax", "h59", "h59a"}:
            scores = tx_scores + mu * sc_scores
            gate = shiftmax(scores, dim=2, eps=cfg.eps)
            if cfg.preserve_mean:
                gate = gate * float(n_tokens)
            gate = _maybe_clamp_bipolar_gate(gate, cfg)
            row_sum = gate.abs().sum(dim=2)
            attn = attn_carrier.mul(gate)
        else:
            q_event = _ternary_sign_ste(_qkformer_token_q(q_orig)) if cfg.confidence_enabled else None
            k_event = _ternary_sign_ste(k_orig) if cfg.confidence_enabled else None
            sc_gate = _sc_agree_disagree_gate(sc_scores, n_tokens, head_dim, cfg, q_event=q_event, k_event=k_event)
            if cfg.k_consistency_mod:
                consistency = torch.clamp(sc_scores + 1.0, 0.0, 2.0) / 2.0
                sc_gate = sc_gate * consistency
            gate = (1.0 - mu) * tx_gate + mu * sc_gate
            gate = _maybe_clamp_bipolar_gate(gate, cfg)
            row_sum = gate.abs().sum(dim=2)
            scores = (1.0 - mu) * tx_scores + mu * sc_scores
            attn = attn_carrier.mul(gate)
    elif cfg.mode in {"faps", "h61", "flow_aligned_popcount_selector"}:
        # FAPS: unified dyadic popcount + flow-aligned x/y channels + sparse K_mag.
        #   scores = unified_dyadic(Q,K) [+ sparse 2-bit K_mag on active>=tau]
        #   gate = Shiftmax(scores)
        #   attn = K * gate                         (NO carrier)
        scores = _faps_flow_aligned_token_scores(q_orig, k_orig, cfg)
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        gate = shiftmax(scores, dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        attn = k_orig.mul(gate)
    elif cfg.mode in {"h62", "nts_conf_residual_shiftmax", "confidence_calibrated_nts"}:
        # H62: NTS base score + confidence-calibrated SC/FAPS residual.
        #   conf  = sqrt(active/head_dim) * agree/active
        #   score = TX + conf * (mu * SC + gamma * DIR)
        #   gate  = Shiftmax(score)
        #   attn  = K * gate                         (NO carrier)
        scores = _confidence_calibrated_nts_scores(self, q_orig, k_orig, cfg)
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        gate = shiftmax(scores, dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        attn = k_orig.mul(gate)
    elif cfg.mode in {"h60", "tx_sc_k_mag_no_carrier_shiftmax"}:
        # h60: H49 no-carrier + SC score residual + K magnitude.
        #   tx_scores = TX(q,k) [+ K_mag if cfg.k_magnitude_alpha]
        #   sc_scores = SC(q,k)
        #   scores = tx_scores + mu * sc_scores    (score-level fusion, not gate)
        #   gate = Shiftmax(scores)
        #   attn = K * gate                         (NO carrier)
        mu = _apply_hardware_mu_quant(_scheduled_bipolar_mu(self, cfg), cfg)
        tx_scores, sc_scores = _tx_sc_fusion_score_pair(q_orig, k_orig, cfg)
        scores = tx_scores + mu * sc_scores
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        scores = _apply_hardware_score_quant(scores, cfg)
        gate = shiftmax(scores, dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        gate = _apply_hardware_gate_quant(gate, cfg)
        _maybe_emit_h60_profile(
            self,
            q_orig=q_orig,
            k_orig=k_orig,
            tx_scores=tx_scores,
            sc_scores=sc_scores,
            fused_scores=scores,
            gate=gate,
            cfg=cfg,
        )
        attn = k_orig.mul(gate)
    elif cfg.mode in {
        "sc_ad_confidence_carrier_blend_shiftmax",
        "sc_agree_disagree_confidence_carrier_blend_shiftmax",
        "h56mc",
    }:
        # H56mc: confidence/dead-zone SC gate blended with the native carrier.
        #
        # NSC-04's H56m lets SC evidence correct an inactive carrier, but its
        # SC branch treats weak low-activity votes like confident votes. This
        # mode keeps H56m's no-new-weights carrier blend while enabling the
        # confidence/dead-zone path from H56c:
        #
        #   sc_gate = confidence * SC(agree, disagree) + (1-confidence)/N
        #   output  = K * ((1 - mu) * carrier_q + mu * sc_gate)
        q_event = _ternary_sign_ste(_qkformer_token_q(q_orig))
        k_event = _ternary_sign_ste(k_orig)
        att_token = q_orig.sum(dim=-1, keepdim=True)
        att_token = self.sn2_q(att_token)
        carrier_gate = att_token.reshape(B_, self.num_heads, n_tokens, 1)

        scores = _signed_consensus_token_scores(q_orig, k_orig, cfg)
        scores = _center_token_scores(scores, cfg)
        sc_gate = _sc_agree_disagree_gate(scores, n_tokens, head_dim, cfg, q_event=q_event, k_event=k_event)
        if cfg.k_consistency_mod:
            consistency = torch.clamp(scores + 1.0, 0.0, 2.0) / 2.0
            sc_gate = sc_gate * consistency
        mu = float(cfg.bipolar_mu)
        gate = (1.0 - mu) * carrier_gate + mu * sc_gate
        gate = _maybe_clamp_bipolar_gate(gate, cfg)
        row_sum = gate.abs().sum(dim=2)
        attn = k_orig.mul(gate)
    elif cfg.mode in {"sc_ad_deadzone_shiftmax", "h56b"}:
        # H56b: SC agree/disagree + dead-zone.
        #
        # Tokens with |score| < epsilon are treated as "no opinion" and given
        # uniform 1/N weight. The remaining tokens share (1 - dead_fraction)
        # of the attention budget via agree/disagree Shiftmax.
        scores = _signed_consensus_token_scores(q_orig, k_orig, cfg)
        scores = _center_token_scores(scores, cfg)
        gate = _sc_agree_disagree_gate(scores, n_tokens, head_dim, cfg)
        row_sum = gate.abs().sum(dim=2)
        attn = k_orig.mul(gate)
    elif cfg.mode in {"sc_ad_confidence_shiftmax", "h56c"}:
        # H56c: SC agree/disagree + dead-zone + confidence gating.
        #
        # Low-activity tokens (few channels voting) have their gate regressed
        # toward 1/N: effective = confidence×gate + (1-confidence)/N.
        # Confidence = sqrt(active_channels / head_dim).
        q_event = _ternary_sign_ste(_qkformer_token_q(q_orig))
        k_event = _ternary_sign_ste(k_orig)
        scores = _signed_consensus_token_scores(q_orig, k_orig, cfg)
        scores = _center_token_scores(scores, cfg)
        gate = _sc_agree_disagree_gate(scores, n_tokens, head_dim, cfg, q_event=q_event, k_event=k_event)
        row_sum = gate.abs().sum(dim=2)
        attn = k_orig.mul(gate)
    elif cfg.mode in {"sc_ad_confidence_kmod_shiftmax", "h56d"}:
        # H56d: SC agree/disagree + dead-zone + confidence + K consistency mod.
        #
        # Before gating, K is modulated by consistency = clamp(score+1,0,2)/2.
        # score=+1 → K fully trusted; score=0 → K halved; score=-1 → K zeroed.
        q_event = _ternary_sign_ste(_qkformer_token_q(q_orig))
        k_event = _ternary_sign_ste(k_orig)
        scores = _signed_consensus_token_scores(q_orig, k_orig, cfg)
        scores = _center_token_scores(scores, cfg)
        gate = _sc_agree_disagree_gate(scores, n_tokens, head_dim, cfg, q_event=q_event, k_event=k_event)
        row_sum = gate.abs().sum(dim=2)
        consistency = torch.clamp(scores + 1.0, 0.0, 2.0) / 2.0
        attn = k_orig.mul(gate * consistency)
    elif cfg.mode in {"sc_ad_activenorm_shiftmax", "h56e"}:
        # H56e: SC agree/disagree + active-norm denominator.
        #
        # Score is divided by the actual number of active channels instead of
        # the fixed head_dim=32. Fewer active channels → score magnitude is
        # naturally penalized without needing explicit confidence gating.
        scores = _signed_consensus_token_scores(q_orig, k_orig, cfg)
        scores = _center_token_scores(scores, cfg)
        gate = _sc_agree_disagree_gate(scores, n_tokens, head_dim, cfg)
        row_sum = gate.abs().sum(dim=2)
        attn = k_orig.mul(gate)
    elif cfg.mode in {"ternary_alpha_xnor_shiftmax", "alpha_xnor_shiftmax", "h18a"}:
        # H18a: CVPR 2025 alpha-XNOR-inspired ternary gate.
        #
        # Keep the native QKFormer token carrier, then reweight it with a
        # spike-similarity gate that distinguishes same-polarity events,
        # silence/silence matches, and opposite-polarity conflicts.
        att_token = q_orig.sum(dim=-1, keepdim=True)
        att_token = self.sn2_q(att_token)
        att_gate = att_token.reshape(B_, self.num_heads, n_tokens, 1)
        attn = k_orig.mul(att_gate)

        scores = _ternary_alpha_xnor_token_scores(q_orig, k_orig, cfg)
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        gate = shiftmax(scores, dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        attn = attn * gate
    elif cfg.mode in {"ternary_alpha_xnor_shiftmax_residual", "h48"}:
        # H48: residual TX gate preserves the original QKFormer carrier and adds
        # the TX attention as an additive correction: attn = carrier * (1 + α*(gate-1)).
        # α=0 means pure baseline carrier; α=1 means pure TX gate (same as h18a).
        att_token = q_orig.sum(dim=-1, keepdim=True)
        att_token = self.sn2_q(att_token)
        att_gate = att_token.reshape(B_, self.num_heads, n_tokens, 1)
        attn_carrier = k_orig.mul(att_gate)

        scores = _ternary_alpha_xnor_token_scores(q_orig, k_orig, cfg)
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        gate = shiftmax(scores, dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        alpha = float(cfg.residual_alpha)
        attn = attn_carrier * (1.0 + alpha * (gate - 1.0))
    elif cfg.mode in {"ternary_alpha_xnor_l1", "alpha_xnor_l1", "h18a_l1"}:
        # Same alpha-XNOR evidence, but exact L1 normalization instead of
        # Shiftmax. This tests whether the exponent-like normalization is the
        # AAE failure source.
        att_token = q_orig.sum(dim=-1, keepdim=True)
        att_token = self.sn2_q(att_token)
        att_gate = att_token.reshape(B_, self.num_heads, n_tokens, 1)
        attn = k_orig.mul(att_gate)

        scores = _ternary_alpha_xnor_token_scores(q_orig, k_orig, cfg)
        gate = l1norm(torch.relu(scores + cfg.consensus_bias), dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        attn = attn * gate
    elif cfg.mode in {"a2os2a_gate", "a2os2a_qk_adapter", "h18b"}:
        # H18b: A2OS2A-inspired auxiliary gate.
        #
        # Paper pattern: binary Q, nonnegative/ReLU K, ternary V, no softmax or
        # scaling. The baseline block has no separate V projection, so this
        # conservative adapter keeps the original QKFormer K carrier and uses a
        # nonnegative A2OS2A-style score only as a gate.
        att_token = q_orig.sum(dim=-1, keepdim=True)
        att_token = self.sn2_q(att_token)
        att_gate = att_token.reshape(B_, self.num_heads, n_tokens, 1)
        attn = k_orig.mul(att_gate)

        scores = _a2os2a_token_scores(q_orig, k_orig, cfg)
        gate = l1norm(torch.relu(scores + cfg.consensus_bias), dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        attn = attn * gate
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
            "bsa_attention.mode must be strict_bsa_shiftmax/bsa_matrix_shiftmax, "
            "qk_bsa/bsa_qk/ternary_matrix, "
            "qkformer_spike_shift/spike_shift, qkformer_token/token, "
            "signed_consensus_shiftmax/h13b, signed_consensus_shiftmax_raw/h13b_raw, "
            "signed_consensus_shiftnorm/h13c, "
            "signed_consensus_popcount_l1/h13t, "
            "ternary_alpha_xnor_qkselector_shiftmax/h49, "
            "bipolar_qkselector_shiftmax/h54a, tx_bipolar_qkselector_shiftmax/h54b, "
            "dual_channel_qkselector_shiftmax/h51, a2os2a_kasv_shiftmax/h52, "
            "sc_agree_disagree_shiftmax/h56a, sc_ad_deadzone_shiftmax/h56b, "
            "sc_ad_confidence_shiftmax/h56c, sc_ad_confidence_kmod_shiftmax/h56d, "
            "sc_ad_activenorm_shiftmax/h56e, sc_agree_disagree_residual_shiftmax/h56r, "
            "sc_ad_carrier_blend_shiftmax/h56m, "
            "tx_sc_residual_selector_shiftmax/h57, tx_sc_late_residual_selector_shiftmax/h58, "
            "tx_sc_score_residual_shiftmax/h59, "
            "faps/h61/flow_aligned_popcount_selector, "
            "h62/nts_conf_residual_shiftmax, "
            "h60/tx_sc_k_mag_no_carrier_shiftmax, "
            "ternary_alpha_xnor_local_shiftmax/h59_local, "
            "sc_ad_confidence_carrier_blend_shiftmax/h56mc, "
            "ternary_alpha_xnor_shiftmax/h18a, ternary_alpha_xnor_shiftmax_residual/h48, "
            "ternary_alpha_xnor_l1/h18a_l1, "
            "ternary_alpha_xnor_ssa_linear/h42b, ternary_alpha_xnor_ssa_qkv_linear/h42c, "
            "ternary_alpha_xnor_ssa_qkv_shiftmax/h42d, ternary_alpha_xnor_ssa_kreuse_shiftmax/h45, "
            "binary_alpha_xnor_matrix_shiftmax/l1, "
            "a2os2a_gate/h18b, alpha_xnor_matrix_shiftmax/h18c, "
            "alpha_xnor_matrix_l1/h18d, a2os2a_direct/h18e, a2os2a_qkv_l1, "
            "hamming_binary_direct/h21a, hamming_ternary_active_direct/h21b, "
            "or compat_qk_product/legacy"
        )

    with torch.no_grad():
        self.h9_shiftmax_row_sum_mean = float(row_sum.detach().mean().cpu())
        self.h9_shiftmax_row_sum_min = float(row_sum.detach().min().cpu())
        self.h9_shiftmax_row_sum_max = float(row_sum.detach().max().cpu())
        self.h9_shiftmax_gate_mean = float(gate.detach().mean().cpu())
        self.h13_consensus_score_mean = float(
            locals().get("scores", torch.zeros((), device=x.device)).detach().mean().cpu()
        )

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
        if cfg.mode in {
            "strict_bsa_qkv_shiftmax",
            "bsa_qkv_shiftmax",
            "bsa_true_qkv_shiftmax",
            "a2os2a_qkv_l1",
            "a2os2a_true_qkv_l1",
            "ternary_alpha_xnor_ssa_qkv_linear",
            "alpha_xnor_ssa_qkv_linear",
            "ternary_alpha_xnor_qkv",
            "h42c",
            "ternary_alpha_xnor_ssa_qkv_shiftmax",
            "alpha_xnor_ssa_qkv_shiftmax",
            "h42d",
        }:
            _ensure_independent_value_branch(module, cfg)
        if not hasattr(module, "_h9_original_forward"):
            module._h9_original_forward = module.forward
        module._h9_shiftmax_cfg = cfg
        # ── NTX-11: store stage index for per-stage enhancements ──
        _stage = int(name.split(".")[1]) if name.startswith("layers.") else 0
        module._h9_stage = _stage
        # ── end stage index ──
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
    score_means = [float(getattr(module, "h13_consensus_score_mean", 0.0)) for module in modules]
    return {
        "num_modules": len(modules),
        "row_sum_mean": sum(row_means) / len(row_means),
        "gate_mean": sum(gate_means) / len(gate_means),
        "score_mean": sum(score_means) / len(score_means),
    }


def set_shiftmax_attention_step(model: nn.Module, step: int) -> int:
    count = 0
    for module in model.modules():
        if hasattr(module, "_h9_shiftmax_cfg"):
            module._h9_global_step = int(step)
            count += 1
    return count
