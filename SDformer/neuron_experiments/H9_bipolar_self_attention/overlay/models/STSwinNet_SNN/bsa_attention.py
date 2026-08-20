"""H9/H10 Shiftmax compatibility layer for SDFormerFlow QK attention.

The legacy H9 modes are compatibility gates around SDFormerFlow's original
QKFormer-style token carrier. The H10c qk_bsa mode builds a true ternary Q/K
score matrix before Shiftmax, then uses K as the value carrier because this
baseline attention block has no separate V projection.
"""

from __future__ import annotations

import base64
import copy
import math
import zlib
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
    binary_motion_xor_alpha: float = 0.0  # H67: dyadic temporal K XOR bias
    castling_matrix_aux_weight: float = 0.0  # H68: training-only full-matrix branch
    castling_matrix_aux_end_step: int = 0
    event_temperature_enabled: bool = False  # H70: activity-conditioned dyadic inverse-temperature
    event_temperature_max_shift: int = 3
    context_broadcast_enabled: bool = False  # H71: parameter-free window context mixing
    match_code_seed: int = 6701
    match_code_weight_quant_enabled: bool = False
    match_code_weight_step: float = 1.0 / 128.0
    match_code_weight_min: float = -1.0
    match_code_weight_max: float = 127.0 / 128.0
    lc4_coefficient_quant_enabled: bool = False
    lc4_coefficient_step: float = 1.0 / 64.0
    lc4_coefficient_min: float = -1.0
    lc4_coefficient_max: float = 1.0
    cf10_beta_step: float = 1.0 / 64.0
    cf10_beta_min: float = -1.0
    cf10_beta_max: float = 1.0
    directional_channels_enabled: bool = False  # S2: split Q/K by x/y direction
    directional_merge_mode: str = "sum"  # S2: "sum" or "mean"
    confidence_min_active: int = 0  # FAPS: sparse K_mag only when active channels >= tau
    flow_disagreement_gamma: float = 0.0  # FAPS: penalize |S_x - S_y| when directional
    faps_same_nonzero_weight: float = 4.0
    faps_same_zero_weight: float = 1.0
    faps_opposite_weight: float = 1.0
    faps_single_active_weight: float = 4.0
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
    hardware_rtl_shiftmax_enabled: bool = False
    hardware_mask_invalid_candidates: bool = False
    source_gate_cardinality_regularization_weight: float = 0.0
    source_gate_cardinality_proxy_mode: str = "mean_collapse"
    class_stability_regularization_weight: float = 0.0
    direct_shiftmax_groups: int = 1
    direct_shiftmax_center_output: bool = False
    direct_shiftmax_signed_events: bool = False
    temporal_quotient_steps: int = 0  # D1 (h87): SNN num_steps for T=5 grouping (10 for w15 fullres)
    temporal_quotient_len: int = 5  # D1 (h87): quintuple time-quotient length (contract-pinned)
    temporal_quotient_batch: int = 0  # D1 (h87): explicit batch for batch-dim window decomposition (0=auto)
    a3s_delta_bins: int = 0  # D3 (h88): A3S 方向场分数偏移（Q7 1/128 网格档；8 = Δ=1/16；0 = Δ=0 恒等锚点）
    a3s_delta_warmup_steps: int = 0  # D3 (h88): Δ 注入式渐增步数（0 = 立即满档；>0 时从 0 线性渐增至满档）
    sw12_window_size: int = 0  # D2 (h89): stride-12 重叠滑窗窗口边长（0 = 默认 15 = 现网 Swin 窗）
    sw12_stride: int = 0  # D2 (h89): 重叠滑窗步长（0 = 默认 12；15 = 退化解 = 稠密非重叠基线）
    sw12_num_steps: int = 0  # D2 (h89): SNN num_steps（w15 fullres 为 10）
    sw12_batch: int = 0  # D2 (h89): batch 维窗口分解显式偏好（0 = 自动；= 每 field 窗口数）
    sw12_window_grid: tuple[int, int] = (0, 0)  # D2 (h89): (n_y, n_x) field 网格显式钉死（(0,0) = 自动）


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
        binary_motion_xor_alpha=float(raw.get("binary_motion_xor_alpha", 0.0)),
        castling_matrix_aux_weight=float(raw.get("castling_matrix_aux_weight", 0.0)),
        castling_matrix_aux_end_step=int(raw.get("castling_matrix_aux_end_step", 0) or 0),
        event_temperature_enabled=bool(raw.get("event_temperature_enabled", False)),
        event_temperature_max_shift=int(raw.get("event_temperature_max_shift", 3) or 0),
        context_broadcast_enabled=bool(raw.get("context_broadcast_enabled", False)),
        match_code_seed=int(raw.get("match_code_seed", 6701) or 6701),
        match_code_weight_quant_enabled=bool(raw.get("match_code_weight_quant_enabled", False)),
        match_code_weight_step=float(raw.get("match_code_weight_step", 1.0 / 128.0) or 0.0),
        match_code_weight_min=float(raw.get("match_code_weight_min", -1.0)),
        match_code_weight_max=float(raw.get("match_code_weight_max", 127.0 / 128.0)),
        lc4_coefficient_quant_enabled=bool(raw.get("lc4_coefficient_quant_enabled", False)),
        lc4_coefficient_step=float(raw.get("lc4_coefficient_step", 1.0 / 64.0) or 0.0),
        lc4_coefficient_min=float(raw.get("lc4_coefficient_min", -1.0)),
        lc4_coefficient_max=float(raw.get("lc4_coefficient_max", 1.0)),
        cf10_beta_step=float(raw.get("cf10_beta_step", 1.0 / 64.0) or 0.0),
        cf10_beta_min=float(raw.get("cf10_beta_min", -1.0)),
        cf10_beta_max=float(raw.get("cf10_beta_max", 1.0)),
        directional_channels_enabled=bool(raw.get("directional_channels_enabled", False)),
        directional_merge_mode=str(raw.get("directional_merge_mode", "sum")),
        confidence_min_active=int(raw.get("confidence_min_active", 0) or 0),
        flow_disagreement_gamma=float(raw.get("flow_disagreement_gamma", 0.0)),
        faps_same_nonzero_weight=float(raw.get("faps_same_nonzero_weight", 4.0)),
        faps_same_zero_weight=float(raw.get("faps_same_zero_weight", 1.0)),
        faps_opposite_weight=float(raw.get("faps_opposite_weight", 1.0)),
        faps_single_active_weight=float(raw.get("faps_single_active_weight", 4.0)),
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
        hardware_rtl_shiftmax_enabled=bool(raw.get("hardware_rtl_shiftmax_enabled", False)),
        hardware_mask_invalid_candidates=bool(
            raw.get("hardware_mask_invalid_candidates", False)
        ),
        source_gate_cardinality_regularization_weight=float(
            raw.get("source_gate_cardinality_regularization_weight", 0.0) or 0.0
        ),
        source_gate_cardinality_proxy_mode=str(
            raw.get("source_gate_cardinality_proxy_mode", "mean_collapse")
        ),
        class_stability_regularization_weight=float(
            raw.get("class_stability_regularization_weight", 0.0) or 0.0
        ),
        direct_shiftmax_groups=int(raw.get("direct_shiftmax_groups", 1) or 1),
        direct_shiftmax_center_output=bool(raw.get("direct_shiftmax_center_output", False)),
        direct_shiftmax_signed_events=bool(raw.get("direct_shiftmax_signed_events", False)),
        temporal_quotient_steps=int(raw.get("temporal_quotient_steps", 0) or 0),
        temporal_quotient_len=int(raw.get("temporal_quotient_len", 5) or 5),
        temporal_quotient_batch=int(raw.get("temporal_quotient_batch", 0) or 0),
        a3s_delta_bins=int(raw.get("a3s_delta_bins", 0) or 0),
        a3s_delta_warmup_steps=int(raw.get("a3s_delta_warmup_steps", 0) or 0),
        sw12_window_size=int(raw.get("sw12_window_size", 0) or 0),
        sw12_stride=int(raw.get("sw12_stride", 0) or 0),
        sw12_num_steps=int(raw.get("sw12_num_steps", 0) or 0),
        sw12_batch=int(raw.get("sw12_batch", 0) or 0),
        sw12_window_grid=(
            tuple(int(item) for item in raw.get("sw12_window_grid", (0, 0)))
            if raw.get("sw12_window_grid")
            else (0, 0)
        ),
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


def _hardware_score_clip_stats(
    scores: torch.Tensor | None,
    cfg: ShiftmaxAttentionConfig,
) -> dict[str, int | float]:
    """Count deployment score clipping without changing the quantized path."""

    if scores is None or not cfg.hardware_quant_enabled:
        return {}
    data = scores.detach()
    total = int(data.numel())
    low = (
        int((data < float(cfg.hardware_score_min)).sum().item())
        if cfg.hardware_score_min is not None
        else 0
    )
    high = (
        int((data > float(cfg.hardware_score_max)).sum().item())
        if cfg.hardware_score_max is not None
        else 0
    )
    clipped = low + high
    return {
        "score_quant_total": total,
        "score_clip_low": low,
        "score_clip_high": high,
        "score_clip_ratio": clipped / total if total else 0.0,
    }


def _apply_hardware_gate_quant(gate: torch.Tensor, cfg: ShiftmaxAttentionConfig) -> torch.Tensor:
    if not cfg.hardware_quant_enabled:
        return gate
    if cfg.hardware_gate_min is not None or cfg.hardware_gate_max is not None:
        min_value = -float("inf") if cfg.hardware_gate_min is None else float(cfg.hardware_gate_min)
        max_value = float("inf") if cfg.hardware_gate_max is None else float(cfg.hardware_gate_max)
        gate = gate.clamp(min=min_value, max=max_value)
    return _quantize_ste(gate, float(cfg.hardware_gate_step))


def _rtl_shiftmax_gate_q17(
    scores: torch.Tensor,
    *,
    dim: int,
    preserve_mean: bool,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Bit-exact model of the RTL LUT Shiftmax and unsigned Q1.7 gate.

    ``scores`` must already be quantized to Q7. The RTL subtracts the row
    maximum, approximates ``2**delta`` with a 16-entry Q8 LUT, normalizes by
    the next power-of-two integer row sum, and rounds the final gate to nearest
    with ties to even. The output is saturated to the deployment range [0, 2].
    """

    score_q7 = torch.round(scores * 128.0).to(dtype=torch.int64)
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=scores.device, dtype=torch.bool)
        valid_mask = torch.broadcast_to(valid_mask, scores.shape)
        score_min = torch.iinfo(score_q7.dtype).min // 4
        row_max_q7 = score_q7.masked_fill(~valid_mask, score_min).amax(
            dim=dim, keepdim=True
        )
    else:
        row_max_q7 = score_q7.amax(dim=dim, keepdim=True)
    delta_q7 = score_q7 - row_max_q7
    abs_delta = (-delta_q7).clamp_min(0)
    integer_shift = torch.bitwise_right_shift(abs_delta, 7).clamp_max(8)
    fraction_q7 = torch.bitwise_and(abs_delta, 127)
    fraction_index = torch.div(fraction_q7 + 7, 8, rounding_mode="floor").clamp_max(15)
    lut = torch.tensor(
        [256, 245, 234, 224, 215, 205, 196, 188, 181, 173, 165, 158, 152, 145, 139, 133],
        dtype=torch.int64,
        device=scores.device,
    )
    exp_q8 = torch.bitwise_right_shift(lut[fraction_index], integer_shift)
    if valid_mask is not None:
        exp_q8 = exp_q8.masked_fill(~valid_mask, 0)
    row_sum_q8 = exp_q8.sum(dim=dim, keepdim=True)

    probe = (row_sum_q8 - 1).clamp_min(0)
    denominator_shift = torch.zeros_like(probe)
    for _ in range(32):
        denominator_shift = denominator_shift + probe.ne(0).to(dtype=torch.int64)
        probe = torch.bitwise_right_shift(probe, 1)

    token_scale = scores.shape[dim] if preserve_mean else 1
    scaled = exp_q8 * int(token_scale) * 128
    quotient = torch.bitwise_right_shift(scaled, denominator_shift)
    remainder = scaled - torch.bitwise_left_shift(quotient, denominator_shift)
    half = torch.bitwise_left_shift(
        torch.ones_like(denominator_shift),
        (denominator_shift - 1).clamp_min(0),
    )
    increment = denominator_shift.ne(0) & (
        remainder.gt(half) | (remainder.eq(half) & torch.bitwise_and(quotient, 1).ne(0))
    )
    gate_q17 = (quotient + increment.to(dtype=torch.int64)).clamp(min=0, max=256)
    if valid_mask is not None:
        gate_q17 = gate_q17.masked_fill(~valid_mask, 0)
    return gate_q17.to(dtype=scores.dtype) / 128.0


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


def _encode_ordered_count_trace(counts: torch.Tensor) -> dict[str, Any]:
    """Encode a profiling-only ordered count tensor without retaining Python lists."""

    values = counts.detach().to(device="cpu", dtype=torch.long).contiguous()
    if values.numel() == 0:
        min_value = max_value = 0
    else:
        min_value = int(values.amin().item())
        max_value = int(values.amax().item())
    if -(1 << 15) <= min_value and max_value < (1 << 15):
        cpu = values.to(dtype=torch.int16)
        dtype = "int16_le"
        payload_bytes = cpu.numpy().astype("<i2", copy=False).tobytes()
    elif -(1 << 31) <= min_value and max_value < (1 << 31):
        cpu = values.to(dtype=torch.int32)
        dtype = "int32_le"
        payload_bytes = cpu.numpy().astype("<i4", copy=False).tobytes()
    else:
        raise OverflowError(
            f"ordered count trace超出int32范围: [{min_value}, {max_value}]"
        )
    payload = zlib.compress(payload_bytes, level=6)
    return {
        "shape": list(cpu.shape),
        "dtype": dtype,
        "codec": "zlib_base64",
        "data": base64.b64encode(payload).decode("ascii"),
    }


def _delta_locality_stats(
    q_toggle: torch.Tensor,
    k_toggle: torch.Tensor,
    *,
    include_ordered_trace: bool = False,
) -> dict[str, Any]:
    """Return raw, element-weightable locality counts for exact Delta-TTX."""

    if q_toggle.dtype != torch.bool or k_toggle.dtype != torch.bool:
        raise ValueError("Delta-TTX toggle tensors must be boolean")
    if q_toggle.shape != k_toggle.shape or q_toggle.ndim != 4:
        raise ValueError("Delta-TTX toggles must share [B, heads, tokens, lanes] shape")
    update = q_toggle | k_toggle
    update_count = update.sum(dim=-1)
    lanes = int(update.shape[-1])
    changed_token = update_count > 0
    token_heads = int(update_count.numel())
    changed_tokens = int(changed_token.sum().item())
    previous = torch.nn.functional.pad(changed_token[..., :-1], (1, 0), value=False)
    run_starts = changed_token & ~previous
    stats = {
        "delta_token_heads": token_heads,
        "delta_zero_update_token_heads": token_heads - changed_tokens,
        "delta_changed_token_heads": changed_tokens,
        "delta_changed_token_runs": int(run_starts.sum().item()),
        "delta_update_count_0": int((update_count == 0).sum().item()),
        "delta_update_count_1": int((update_count == 1).sum().item()),
        "delta_update_count_2": int((update_count == 2).sum().item()),
        "delta_update_count_3_4": int(((update_count >= 3) & (update_count <= 4)).sum().item()),
        "delta_update_count_5_8": int(((update_count >= 5) & (update_count <= 8)).sum().item()),
        "delta_update_count_9_16": int(((update_count >= 9) & (update_count <= 16)).sum().item()),
        "delta_update_count_17_plus": int((update_count >= 17).sum().item()),
        "delta_update_histogram": torch.bincount(
            update_count.reshape(-1), minlength=lanes + 1
        ).cpu().tolist(),
    }
    for threshold in (2, 4, 8, 12, 16):
        sparse = (update_count > 0) & (update_count <= threshold)
        stats[f"delta_active_le{threshold}"] = int(sparse.sum().item())
        stats[f"delta_active_lane_sum_le{threshold}"] = int(update_count[sparse].sum().item())
    for bundle in (4, 8):
        token_count = int(changed_token.shape[-1])
        groups = (token_count + bundle - 1) // bundle
        padded = torch.nn.functional.pad(changed_token, (0, groups * bundle - token_count), value=False)
        bundle_changed = padded.reshape(*padded.shape[:-1], groups, bundle).any(dim=-1)
        stats[f"delta_bundle{bundle}_total"] = int(bundle_changed.numel())
        stats[f"delta_bundle{bundle}_empty"] = int((~bundle_changed).sum().item())
    if include_ordered_trace:
        stats["delta_update_ordered_trace"] = _encode_ordered_count_trace(update_count)
    return stats


def _token_time_bundle_stats(
    q_binary: torch.Tensor,
    k_binary: torch.Tensor,
    *,
    include_ordered_trace: bool = False,
) -> dict[str, Any]:
    """Raw counts for true T=2 by spatial-token hardware bundles.

    Activity routing uses ``Q OR K``. K-zero bundles are reported separately
    because they permit exact value/projection gating but do not, by themselves,
    permit dropping a score from the window-wide Shiftmax denominator.
    """

    if q_binary.dtype != torch.bool or k_binary.dtype != torch.bool:
        raise ValueError("TTB tensors must be boolean")
    if q_binary.ndim != 5 or k_binary.ndim != 5:
        raise ValueError("TTB tensors must use [T,B,heads,tokens,lanes]")
    if q_binary.shape != k_binary.shape or q_binary.shape[0] != 2:
        raise ValueError("TTB requires matching Q/K tensors with T=2")

    t_steps, batch, heads, tokens, lanes = q_binary.shape
    union = q_binary | k_binary
    motion = k_binary[0] ^ k_binary[1]
    stats: dict[str, Any] = {}
    for bundle in (1, 2, 4, 8):
        groups = (tokens + bundle - 1) // bundle
        pad_tokens = groups * bundle - tokens
        union_padded = torch.nn.functional.pad(union, (0, 0, 0, pad_tokens), value=False)
        k_padded = torch.nn.functional.pad(k_binary, (0, 0, 0, pad_tokens), value=False)
        motion_padded = torch.nn.functional.pad(motion, (0, 0, 0, pad_tokens), value=False)
        union_count = union_padded.reshape(
            t_steps, batch, heads, groups, bundle, lanes
        ).sum(dim=(0, 4, 5))
        k_count = k_padded.reshape(
            t_steps, batch, heads, groups, bundle, lanes
        ).sum(dim=(0, 4, 5))
        motion_count = motion_padded.reshape(
            batch, heads, groups, bundle, lanes
        ).sum(dim=(3, 4))
        prefix = f"ttb_tok{bundle}"
        stats[f"{prefix}_total"] = int(union_count.numel())
        stats[f"{prefix}_empty"] = int((union_count == 0).sum().item())
        stats[f"{prefix}_active_lanes"] = int(union_count.sum().item())
        stats[f"{prefix}_capacity_lanes"] = int(t_steps * batch * heads * tokens * lanes)
        stats[f"{prefix}_kzero"] = int((k_count == 0).sum().item())
        stats[f"{prefix}_motion_zero"] = int((motion_count == 0).sum().item())
        stats[f"{prefix}_active_histogram"] = torch.bincount(
            union_count.reshape(-1), minlength=t_steps * bundle * lanes + 1
        ).cpu().tolist()
        if include_ordered_trace and bundle in (4, 8):
            stats[f"{prefix}_active_ordered_trace"] = _encode_ordered_count_trace(union_count)
            stats[f"{prefix}_k_ordered_trace"] = _encode_ordered_count_trace(k_count)
            stats[f"{prefix}_motion_ordered_trace"] = _encode_ordered_count_trace(motion_count)
        for threshold in (2, 4, 8, 12, 16, 32):
            sparse = (union_count > 0) & (union_count <= threshold)
            stats[f"{prefix}_active_le{threshold}"] = int(sparse.sum().item())
            stats[f"{prefix}_active_lane_sum_le{threshold}"] = int(
                union_count[sparse].sum().item()
            )
    return stats


def _spatial_pair_locality_stats(
    q_binary: torch.Tensor,
    k_binary: torch.Tensor,
) -> dict[str, Any]:
    """Measure exact spatial locality and sparse-bank pressure for a T=2 window."""

    if q_binary.shape != k_binary.shape or q_binary.ndim != 5 or q_binary.shape[0] != 2:
        raise ValueError("spatial pair tensors must share [2,B,H,N,D] shape")
    tokens = int(q_binary.shape[-2])
    side = math.isqrt(tokens)
    if side * side != tokens:
        return {}

    token_active = (q_binary | k_binary).any(dim=-1)
    union = token_active.any(dim=0)
    persistent = token_active[0] & token_active[1]
    changed = token_active[0] ^ token_active[1]
    grid = union.reshape(*union.shape[:-1], side, side)
    row_total = int(union.shape[0] * union.shape[1])
    union_count = union.sum(dim=-1).to(dtype=torch.long)

    stats: dict[str, Any] = {
        "spatial_row_total": row_total,
        "spatial_union_tokens": int(union.sum().item()),
        "spatial_persistent_tokens": int(persistent.sum().item()),
        "spatial_changed_tokens": int(changed.sum().item()),
        "spatial_union_count_histogram": torch.bincount(
            union_count.reshape(-1), minlength=tokens + 1
        ).cpu().tolist(),
    }
    adjacency = {
        "horizontal": (grid[..., :, :-1], grid[..., :, 1:]),
        "vertical": (grid[..., :-1, :], grid[..., 1:, :]),
        "diag_down": (grid[..., :-1, :-1], grid[..., 1:, 1:]),
        "diag_up": (grid[..., 1:, :-1], grid[..., :-1, 1:]),
    }
    for name, (left, right) in adjacency.items():
        stats[f"spatial_{name}_adjacent_active"] = int((left & right).sum().item())
        stats[f"spatial_{name}_adjacent_total"] = int(left.numel())

    rows = torch.arange(side, device=union.device).view(side, 1).expand(side, side)
    cols = torch.arange(side, device=union.device).view(1, side).expand(side, side)
    linear = torch.arange(tokens, device=union.device).reshape(side, side)
    for banks in (4, 8):
        mappings = {
            "rowmajor": torch.remainder(linear, banks),
            "diagonal": torch.remainder(rows + cols, banks),
            "xor": torch.remainder(torch.bitwise_xor(rows, cols), banks),
        }
        for name, mapping in mappings.items():
            # CUDA does not implement Long matmul; use float32 for counting.
            bank_select = torch.nn.functional.one_hot(
                mapping.reshape(-1), num_classes=banks
            ).to(dtype=torch.float32)
            loads = union.to(dtype=torch.float32) @ bank_select
            cycles = loads.amax(dim=-1).to(dtype=torch.long)
            stats[f"spatial_bank{banks}_{name}_cycles_sum"] = int(cycles.sum().item())
            stats[f"spatial_bank{banks}_{name}_cycles_histogram"] = torch.bincount(
                cycles.reshape(-1), minlength=tokens + 1
            ).cpu().tolist()
    return stats


def _binary_temporal_pair_stats(
    q_binary: torch.Tensor,
    k_binary: torch.Tensor,
    *,
    gate_q17_code: torch.Tensor | None = None,
    windows_per_sample: int | None = None,
    include_ordered_trace: bool = False,
) -> dict[str, Any]:
    """Return sufficient statistics for TTX and H67 temporal-pair hardware.

    The input layout is ``[T=2, B, heads, spatial_tokens, lanes]``.  Per-time
    Q/K cardinalities and intersections are sufficient to reconstruct the
    dyadic TTX score.  The K temporal XOR cardinality adds the H67 motion term.
    """

    if q_binary.dtype != torch.bool or k_binary.dtype != torch.bool:
        raise ValueError("binary temporal-pair tensors must be boolean")
    if q_binary.shape != k_binary.shape or q_binary.ndim != 5 or q_binary.shape[0] != 2:
        raise ValueError("binary temporal-pair tensors must share [2,B,H,N,D] shape")

    lanes = int(q_binary.shape[-1])
    q_count = q_binary.sum(dim=-1).to(dtype=torch.long)
    k_count = k_binary.sum(dim=-1).to(dtype=torch.long)
    overlap = (q_binary & k_binary).sum(dim=-1).to(dtype=torch.long)
    same_zero = lanes - q_count - k_count + overlap
    motion = (k_binary[0] ^ k_binary[1]).sum(dim=-1).to(dtype=torch.long)
    k_temporal_intersection = (k_binary[0] & k_binary[1]).sum(dim=-1).to(dtype=torch.long)
    k_temporal_union = (k_binary[0] | k_binary[1]).sum(dim=-1).to(dtype=torch.long)
    update = ((q_binary[0] ^ q_binary[1]) | (k_binary[0] ^ k_binary[1])).sum(
        dim=-1
    ).to(dtype=torch.long)
    four_vector_events = (q_count + k_count).sum(dim=0)
    four_vector_union = (
        q_binary[0] | q_binary[1] | k_binary[0] | k_binary[1]
    ).sum(dim=-1).to(dtype=torch.long)

    # Q7 deployment units: round-to-nearest-even((64*overlap + same_zero
    # + 16*motion)/16).  Omitting the final term gives TTX/H68 deployment.
    ttx_numerator = 64 * overlap + same_zero
    h67_numerator = ttx_numerator + 16 * motion.unsqueeze(0)

    def rne_div_pow2(numerator: torch.Tensor, denominator: int) -> torch.Tensor:
        if denominator <= 0 or denominator & (denominator - 1):
            raise ValueError("RNE denominator must be a positive power of two")
        quotient = torch.div(numerator, denominator, rounding_mode="floor")
        remainder = torch.remainder(numerator, denominator)
        half = denominator // 2
        increment = remainder.gt(half) | (
            remainder.eq(half) & quotient.bitwise_and(1).ne(0)
        )
        return quotient + increment.to(dtype=quotient.dtype)

    ttx_score_q7 = rne_div_pow2(ttx_numerator, 16)
    h67_score_q7 = rne_div_pow2(h67_numerator, 16)
    # The exact H67 numerator represents score * 2^11. These counters expose
    # fractional-precision sensitivity without changing the model forward.
    # They are workload statistics, not claims that the Q5/Q6/Q8 RTL exists.
    h67_scores_by_fractional_bits = {
        bits: rne_div_pow2(h67_numerator, 1 << (11 - bits))
        for bits in (5, 6, 7, 8)
    }
    pair_empty = four_vector_events.eq(0)
    kzero_mask = k_count[0].eq(0).to(dtype=torch.long) | (
        k_count[1].eq(0).to(dtype=torch.long) << 1
    )
    score_pair_equal_ttx = ttx_score_q7[0].eq(ttx_score_q7[1])
    score_pair_equal_h67 = h67_score_q7[0].eq(h67_score_q7[1])
    score_pair_equal_h67_by_fractional_bits = {
        bits: scores[0].eq(scores[1])
        for bits, scores in h67_scores_by_fractional_bits.items()
    }
    row_scores_ttx = ttx_score_q7.permute(1, 2, 0, 3).reshape(
        q_binary.shape[1], q_binary.shape[2], -1
    )
    row_scores_h67 = h67_score_q7.permute(1, 2, 0, 3).reshape(
        q_binary.shape[1], q_binary.shape[2], -1
    )
    row_k_binary = k_binary.permute(1, 2, 0, 3, 4).reshape(
        q_binary.shape[1], q_binary.shape[2], -1, lanes
    )

    def projection_class_channel_terms(
        row_scores: torch.Tensor,
        num_classes: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
        """Count unique class-channel products and their maximum token fanout."""

        rows = int(row_scores.shape[0] * row_scores.shape[1])
        score_index = row_scores.reshape(rows, -1).clamp(
            min=0, max=num_classes - 1
        )
        k_lanes = row_k_binary.reshape(rows, -1, lanes).to(dtype=torch.int32)
        class_channel_counts = torch.zeros(
            rows,
            num_classes,
            lanes,
            dtype=torch.int32,
            device=row_scores.device,
        )
        class_channel_counts.scatter_add_(
            1,
            score_index.unsqueeze(-1).expand(-1, -1, lanes),
            k_lanes,
        )
        terms = class_channel_counts.ne(0).sum(dim=(1, 2)).reshape(
            row_scores.shape[0], row_scores.shape[1]
        ).to(dtype=torch.long)
        max_fanout = class_channel_counts.amax(dim=(1, 2)).reshape(
            row_scores.shape[0], row_scores.shape[1]
        ).to(dtype=torch.long)
        delivery_cycles = {
            width: torch.div(
                class_channel_counts + width - 1, width, rounding_mode="floor"
            ).sum(dim=(1, 2)).reshape(
                row_scores.shape[0], row_scores.shape[1]
            ).to(dtype=torch.long)
            for width in (1, 2, 4, 8, 16)
        }
        return terms, max_fanout, delivery_cycles

    def projection_factorized_segment_stats(
        row_scores: torch.Tensor,
        num_classes: int,
        *,
        segment_tokens: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """统计 class×lane 因子化位平面的物理分段扫描工作。"""

        rows = int(row_scores.shape[0] * row_scores.shape[1])
        tokens = int(row_scores.shape[-1])
        score_index = row_scores.reshape(rows, tokens).clamp(
            min=0, max=num_classes - 1
        )
        k_lanes = row_k_binary.reshape(rows, tokens, lanes).to(dtype=torch.int32)
        class_segments = torch.zeros(
            rows, dtype=torch.long, device=row_scores.device
        )
        class_lane_segments = torch.zeros_like(class_segments)
        for start in range(0, tokens, segment_tokens):
            stop = min(tokens, start + segment_tokens)
            segment_counts = torch.zeros(
                rows,
                num_classes,
                lanes,
                dtype=torch.int32,
                device=row_scores.device,
            )
            segment_counts.scatter_add_(
                1,
                score_index[:, start:stop].unsqueeze(-1).expand(
                    -1, -1, lanes
                ),
                k_lanes[:, start:stop],
            )
            presence = segment_counts.ne(0)
            class_segments += presence.any(dim=2).sum(dim=1)
            class_lane_segments += presence.sum(dim=(1, 2))
        return (
            class_segments.reshape(row_scores.shape[0], row_scores.shape[1]),
            class_lane_segments.reshape(
                row_scores.shape[0], row_scores.shape[1]
            ),
        )

    def projection_gate_group_terms(
        gate_code: torch.Tensor,
    ) -> dict[str, Any]:
        """统计最终Q1.7 gate码驱动的逐row和跨窗口唯一乘积项。"""

        if gate_code.shape != row_scores_ttx.shape:
            raise ValueError("gate_q17_code必须与[B,heads,2N] row布局一致")
        batch_windows, heads, tokens = gate_code.shape
        sample_windows = (
            batch_windows if windows_per_sample is None else int(windows_per_sample)
        )
        if sample_windows <= 0 or batch_windows % sample_windows != 0:
            raise ValueError(
                "windows_per_sample必须为正且整除batch_windows："
                f"{sample_windows} vs {batch_windows}"
            )
        rows = batch_windows * heads
        gate_index = gate_code.reshape(rows, tokens).clamp(min=0, max=256)
        k_lanes = row_k_binary.reshape(rows, tokens, lanes).to(dtype=torch.int32)
        # gate=0 的 gated-K 输出恒为零，不应进入 term、delivery 或 PPDI。
        k_lanes = k_lanes * gate_index.ne(0).unsqueeze(-1)
        class_channel_counts = torch.zeros(
            rows, 257, lanes, dtype=torch.int32, device=gate_code.device
        )
        class_channel_counts.scatter_add_(
            1,
            gate_index.unsqueeze(-1).expand(-1, -1, lanes),
            k_lanes,
        )
        destination_id = torch.arange(
            tokens,
            device=gate_code.device,
            dtype=torch.long,
        )
        parity_counts = torch.zeros(
            rows, 2, 257, lanes, dtype=torch.int32, device=gate_code.device
        )
        for parity in (0, 1):
            parity_lanes = k_lanes * destination_id.bitwise_and(1).eq(
                parity
            ).reshape(1, tokens, 1)
            parity_counts[:, parity].scatter_add_(
                1,
                gate_index.unsqueeze(-1).expand(-1, -1, lanes),
                parity_lanes,
            )
        ppdi_delivery = parity_counts.amax(dim=1).sum(dim=(1, 2)).reshape(
            batch_windows, heads
        ).to(torch.long)
        presence = class_channel_counts.ne(0)
        row_terms = presence.sum(dim=(1, 2)).reshape(batch_windows, heads).to(torch.long)
        max_fanout = class_channel_counts.amax(dim=(1, 2)).reshape(
            batch_windows, heads
        ).to(torch.long)
        active_classes = presence.any(dim=2).sum(dim=1).reshape(
            batch_windows, heads
        ).to(torch.long)
        delivery_cycles = {
            width: torch.div(
                class_channel_counts + width - 1, width, rounding_mode="floor"
            ).sum(dim=(1, 2)).reshape(batch_windows, heads).to(torch.long)
            for width in (1, 2, 4, 8, 16)
        }
        term_gate_histogram = presence.sum(dim=(0, 2)).to(torch.long)
        active_lane_gate_histogram = torch.zeros(
            257, dtype=torch.long, device=gate_code.device
        )
        active_lane_gate_histogram.scatter_add_(
            0,
            gate_index.reshape(-1),
            k_lanes.sum(dim=-1).reshape(-1).to(torch.long),
        )
        presence = presence.reshape(batch_windows, heads, 257, lanes)
        counts_by_window = class_channel_counts.reshape(batch_windows, heads, 257, lanes)
        parity_counts_by_window = parity_counts.reshape(
            batch_windows, heads, 2, 257, lanes
        )
        grouped: dict[int, dict[str, Any]] = {}
        for group_windows in (1, 2, 4, 8, 16):
            term_chunks = []
            active_lane_chunks = []
            class_chunks = []
            fanout_chunks = []
            window_count_chunks = []
            ppdi_delivery_chunks = []
            delivery_chunks: dict[int, list[torch.Tensor]] = {
                width: [] for width in (1, 2, 4, 8, 16)
            }
            for sample_start in range(0, batch_windows, sample_windows):
                sample_end = sample_start + sample_windows
                for start in range(sample_start, sample_end, group_windows):
                    grouped_counts = counts_by_window[
                        start : min(start + group_windows, sample_end)
                    ].sum(dim=0)
                    grouped_parity_counts = parity_counts_by_window[
                        start : min(start + group_windows, sample_end)
                    ].sum(dim=0)
                    valid_windows = min(start + group_windows, sample_end) - start
                    grouped_presence = grouped_counts.ne(0)
                    term_chunks.append(grouped_presence.sum(dim=(1, 2)))
                    active_lane_chunks.append(grouped_counts.sum(dim=(1, 2)))
                    class_chunks.append(grouped_presence.any(dim=2).sum(dim=1))
                    fanout_chunks.append(grouped_counts.amax(dim=(1, 2)))
                    window_count_chunks.append(
                        torch.full(
                            (heads,),
                            valid_windows,
                            dtype=torch.long,
                            device=gate_code.device,
                        )
                    )
                    ppdi_delivery_chunks.append(
                        grouped_parity_counts.amax(dim=1).sum(dim=(1, 2))
                    )
                    for width in delivery_chunks:
                        delivery_chunks[width].append(
                            torch.div(
                                grouped_counts + width - 1,
                                width,
                                rounding_mode="floor",
                            ).sum(dim=(1, 2))
                        )
            grouped[group_windows] = {
                "terms": torch.stack(term_chunks).to(torch.long),
                "active_lanes": torch.stack(active_lane_chunks).to(torch.long),
                "active_classes": torch.stack(class_chunks).to(torch.long),
                "max_fanout": torch.stack(fanout_chunks).to(torch.long),
                "window_count": torch.stack(window_count_chunks).to(torch.long),
                "ppdi_delivery": torch.stack(ppdi_delivery_chunks).to(torch.long),
                "delivery_cycles": {
                    width: torch.stack(chunks).to(torch.long)
                    for width, chunks in delivery_chunks.items()
                },
            }
        noninteger_count = gate_code.lt(0).sum() + gate_code.gt(256).sum()
        return {
            "row_terms": row_terms,
            "max_fanout": max_fanout,
            "active_classes": active_classes,
            "delivery_cycles": delivery_cycles,
            "ppdi_delivery": ppdi_delivery,
            "grouped_terms": grouped,
            "term_gate_histogram": term_gate_histogram,
            "active_lane_gate_histogram": active_lane_gate_histogram,
            "out_of_range": noninteger_count.to(torch.long),
        }

    projection_baseline_active_lanes_by_row = row_k_binary.sum(dim=(2, 3)).to(
        dtype=torch.long
    )
    (
        projection_class_channel_terms_ttx_by_row,
        projection_class_channel_max_fanout_ttx_by_row,
        projection_multicast_delivery_ttx_by_width,
    ) = projection_class_channel_terms(
        row_scores_ttx, 4 * lanes + 3
    )
    (
        projection_class_channel_terms_h67_by_row,
        projection_class_channel_max_fanout_h67_by_row,
        projection_multicast_delivery_h67_by_width,
    ) = projection_class_channel_terms(
        row_scores_h67, 5 * lanes + 3
    )
    projection_baseline_active_lanes = int(projection_baseline_active_lanes_by_row.sum().item())
    projection_class_channel_terms_ttx = int(
        projection_class_channel_terms_ttx_by_row.sum().item()
    )
    projection_class_channel_terms_h67 = int(
        projection_class_channel_terms_h67_by_row.sum().item()
    )
    (
        projection_h67_factor_class_segments_by_row,
        projection_h67_factor_class_lane_segments_by_row,
    ) = projection_factorized_segment_stats(
        row_scores_h67,
        5 * lanes + 3,
    )
    projection_h67_factor_class_segments = int(
        projection_h67_factor_class_segments_by_row.sum().item()
    )
    projection_h67_factor_class_lane_segments = int(
        projection_h67_factor_class_lane_segments_by_row.sum().item()
    )
    if projection_class_channel_terms_ttx > projection_baseline_active_lanes:
        raise RuntimeError("TTX类通道投影项不能超过活动K lane基线")
    if projection_class_channel_terms_h67 > projection_baseline_active_lanes:
        raise RuntimeError("H67类通道投影项不能超过活动K lane基线")
    if (
        projection_h67_factor_class_lane_segments
        < projection_class_channel_terms_h67
    ):
        raise RuntimeError("因子化class-lane分段数不能小于全row term数")
    if (
        projection_h67_factor_class_segments
        > projection_h67_factor_class_lane_segments
    ):
        raise RuntimeError("因子化class分段数不能超过class-lane分段数")
    gate_projection_stats = None
    if gate_q17_code is not None:
        gate_projection_stats = projection_gate_group_terms(gate_q17_code.to(torch.long))
    score_class_one_hot_ttx = torch.nn.functional.one_hot(
        row_scores_ttx.clamp(min=0, max=4 * lanes + 2),
        num_classes=4 * lanes + 3,
    ).bool()
    score_class_one_hot_h67 = torch.nn.functional.one_hot(
        row_scores_h67.clamp(min=0, max=4 * lanes + lanes + 2),
        num_classes=5 * lanes + 3,
    ).bool()
    all_class_presence_ttx = score_class_one_hot_ttx.any(dim=2)
    all_class_presence_h67 = score_class_one_hot_h67.any(dim=2)
    kzero_token = k_count.eq(0)
    row_kzero = kzero_token.permute(1, 2, 0, 3).reshape(
        q_binary.shape[1], q_binary.shape[2], -1
    )
    kzero_class_presence_ttx = score_class_one_hot_ttx & row_kzero.unsqueeze(-1)
    kzero_class_presence_h67 = score_class_one_hot_h67 & row_kzero.unsqueeze(-1)
    active_class_presence_ttx = score_class_one_hot_ttx & (~row_kzero).unsqueeze(-1)
    active_class_presence_h67 = score_class_one_hot_h67 & (~row_kzero).unsqueeze(-1)
    all_occupied_classes_ttx = all_class_presence_ttx.sum(dim=-1).to(dtype=torch.long)
    all_occupied_classes_h67 = all_class_presence_h67.sum(dim=-1).to(dtype=torch.long)
    kzero_fold_classes_ttx = kzero_class_presence_ttx.any(dim=2).sum(dim=-1).to(dtype=torch.long)
    kzero_fold_classes_h67 = kzero_class_presence_h67.any(dim=2).sum(dim=-1).to(dtype=torch.long)
    active_projection_classes_ttx = active_class_presence_ttx.any(dim=2).sum(dim=-1).to(
        dtype=torch.long
    )
    active_projection_classes_h67 = active_class_presence_h67.any(dim=2).sum(dim=-1).to(
        dtype=torch.long
    )
    row_span_ttx = row_scores_ttx.amax(dim=-1) - row_scores_ttx.amin(dim=-1)
    row_span_h67 = row_scores_h67.amax(dim=-1) - row_scores_h67.amin(dim=-1)
    both_kzero = kzero_token[0] & kzero_token[1]
    both_active = ~kzero_token[0] & ~kzero_token[1]

    stats: dict[str, Any] = {
        "pair_total": int(pair_empty.numel()),
        "pair_empty": int(pair_empty.sum().item()),
        "pair_motion_zero": int(motion.eq(0).sum().item()),
        "pair_update_zero": int(update.eq(0).sum().item()),
        "pair_score_equal_ttx": int(score_pair_equal_ttx.sum().item()),
        "pair_score_equal_h67": int(score_pair_equal_h67.sum().item()),
        **{
            f"pair_score_equal_h67_qf{bits}": int(equal.sum().item())
            for bits, equal in score_pair_equal_h67_by_fractional_bits.items()
        },
        "pair_kzero_both": int(kzero_mask.eq(3).sum().item()),
        "pair_kzero_one": int(((kzero_mask == 1) | (kzero_mask == 2)).sum().item()),
        "pair_both_active": int(both_active.sum().item()),
        "k_temporal_baseline_reads": int(k_count.sum().item()),
        "k_temporal_union_reads": int(k_temporal_union.sum().item()),
        "k_temporal_intersection_reuse": int(k_temporal_intersection.sum().item()),
        "projection_baseline_active_lanes": projection_baseline_active_lanes,
        "projection_class_channel_terms_ttx": projection_class_channel_terms_ttx,
        "projection_class_channel_terms_h67": projection_class_channel_terms_h67,
        "projection_h67_factor_segment_tokens": 64,
        "projection_h67_factor_class_segments": (
            projection_h67_factor_class_segments
        ),
        "projection_h67_factor_class_lane_segments": (
            projection_h67_factor_class_lane_segments
        ),
        "pair_kzero_same_class_ttx": int((both_kzero & score_pair_equal_ttx).sum().item()),
        "pair_kzero_same_class_h67": int((both_kzero & score_pair_equal_h67).sum().item()),
        "pair_kzero_dual_class_ttx": int((both_kzero & ~score_pair_equal_ttx).sum().item()),
        "pair_kzero_dual_class_h67": int((both_kzero & ~score_pair_equal_h67).sum().item()),
        "token_total": int(q_count.numel()),
        "token_kzero": int(k_count.eq(0).sum().item()),
        "row_total": int(kzero_fold_classes_h67.numel()),
        "row_all_occupied_classes_sum_ttx": int(all_occupied_classes_ttx.sum().item()),
        "row_all_occupied_classes_sum_h67": int(all_occupied_classes_h67.sum().item()),
        "row_kzero_fold_classes_sum_ttx": int(kzero_fold_classes_ttx.sum().item()),
        "row_kzero_fold_classes_sum_h67": int(kzero_fold_classes_h67.sum().item()),
        "row_active_projection_classes_sum_ttx": int(active_projection_classes_ttx.sum().item()),
        "row_active_projection_classes_sum_h67": int(active_projection_classes_h67.sum().item()),
        "q_count_histogram": torch.bincount(q_count.reshape(-1), minlength=lanes + 1).cpu().tolist(),
        "k_count_histogram": torch.bincount(k_count.reshape(-1), minlength=lanes + 1).cpu().tolist(),
        "overlap_histogram": torch.bincount(overlap.reshape(-1), minlength=lanes + 1).cpu().tolist(),
        "same_zero_histogram": torch.bincount(same_zero.reshape(-1), minlength=lanes + 1).cpu().tolist(),
        "motion_histogram": torch.bincount(motion.reshape(-1), minlength=lanes + 1).cpu().tolist(),
        "k_temporal_intersection_histogram": torch.bincount(
            k_temporal_intersection.reshape(-1), minlength=lanes + 1
        ).cpu().tolist(),
        "k_temporal_union_histogram": torch.bincount(
            k_temporal_union.reshape(-1), minlength=lanes + 1
        ).cpu().tolist(),
        "update_histogram": torch.bincount(update.reshape(-1), minlength=lanes + 1).cpu().tolist(),
        "four_vector_event_histogram": torch.bincount(
            four_vector_events.reshape(-1), minlength=4 * lanes + 1
        ).cpu().tolist(),
        "four_vector_union_histogram": torch.bincount(
            four_vector_union.reshape(-1), minlength=lanes + 1
        ).cpu().tolist(),
        "ttx_score_q7_histogram": torch.bincount(
            ttx_score_q7.reshape(-1), minlength=4 * lanes + 3
        ).cpu().tolist(),
        "h67_score_q7_histogram": torch.bincount(
            h67_score_q7.reshape(-1), minlength=5 * lanes + 3
        ).cpu().tolist(),
        "row_all_occupied_classes_ttx_histogram": torch.bincount(
            all_occupied_classes_ttx.reshape(-1), minlength=4 * lanes + 4
        ).cpu().tolist(),
        "row_all_occupied_classes_h67_histogram": torch.bincount(
            all_occupied_classes_h67.reshape(-1), minlength=5 * lanes + 4
        ).cpu().tolist(),
        "row_kzero_fold_classes_ttx_histogram": torch.bincount(
            kzero_fold_classes_ttx.reshape(-1), minlength=4 * lanes + 4
        ).cpu().tolist(),
        "row_kzero_fold_classes_h67_histogram": torch.bincount(
            kzero_fold_classes_h67.reshape(-1), minlength=5 * lanes + 4
        ).cpu().tolist(),
        "row_active_projection_classes_ttx_histogram": torch.bincount(
            active_projection_classes_ttx.reshape(-1), minlength=4 * lanes + 4
        ).cpu().tolist(),
        "row_active_projection_classes_h67_histogram": torch.bincount(
            active_projection_classes_h67.reshape(-1), minlength=5 * lanes + 4
        ).cpu().tolist(),
        "projection_h67_factor_class_segments_histogram": torch.bincount(
            projection_h67_factor_class_segments_by_row.reshape(-1)
        ).cpu().tolist(),
        "projection_h67_factor_class_lane_segments_histogram": torch.bincount(
            projection_h67_factor_class_lane_segments_by_row.reshape(-1)
        ).cpu().tolist(),
        "row_score_span_ttx_histogram": torch.bincount(
            row_span_ttx.reshape(-1), minlength=4 * lanes + 3
        ).cpu().tolist(),
        "row_score_span_h67_histogram": torch.bincount(
            row_span_h67.reshape(-1), minlength=5 * lanes + 3
        ).cpu().tolist(),
        **_spatial_pair_locality_stats(q_binary, k_binary),
    }
    if gate_projection_stats is not None:
        gate_terms_by_row = gate_projection_stats["row_terms"]
        gate_max_fanout_by_row = gate_projection_stats["max_fanout"]
        gate_group_terms = gate_projection_stats["grouped_terms"]
        gate_term_histogram = gate_projection_stats["term_gate_histogram"]
        active_lane_gate_histogram = gate_projection_stats["active_lane_gate_histogram"]
        gate_code_out_of_range = gate_projection_stats["out_of_range"]
        gate_terms = int(gate_terms_by_row.sum().item())
        if gate_terms > projection_baseline_active_lanes:
            raise RuntimeError("最终gate类通道投影项不能超过活动K lane基线")
        stats.update({
            "projection_gate_class_channel_terms_deploy": gate_terms,
            "projection_gate_class_channel_max_fanout_deploy": int(
                gate_max_fanout_by_row.amax().item()
            ),
            "row_active_projection_gate_classes_sum_deploy": int(
                gate_projection_stats["active_classes"].sum().item()
            ),
            "projection_gate_q17_out_of_range": int(gate_code_out_of_range.item()),
            "projection_gate_ppdi_delivery_exact": int(
                gate_projection_stats["ppdi_delivery"].sum().item()
            ),
            "projection_gate_class_channel_term_histogram": gate_term_histogram.cpu().tolist(),
            "projection_active_lane_gate_q17_histogram": active_lane_gate_histogram.cpu().tolist(),
        })
        for group_windows, values in gate_group_terms.items():
            stats[f"projection_gate_group_terms_g{group_windows}"] = int(
                values["terms"].sum().item()
            )
            stats[f"projection_gate_group_active_lanes_g{group_windows}"] = int(
                values["active_lanes"].sum().item()
            )
            stats[f"projection_gate_group_active_classes_g{group_windows}"] = int(
                values["active_classes"].sum().item()
            )
            stats[f"projection_gate_group_max_fanout_g{group_windows}"] = int(
                values["max_fanout"].amax().item()
            )
            stats[f"projection_gate_group_window_count_g{group_windows}"] = int(
                values["window_count"].sum().item()
            )
            stats[f"projection_gate_group_ppdi_delivery_g{group_windows}"] = int(
                values["ppdi_delivery"].sum().item()
            )
            for width, delivery in values["delivery_cycles"].items():
                stats[
                    f"projection_gate_group_delivery_g{group_windows}_m{width}"
                ] = int(delivery.sum().item())
        for width, values in gate_projection_stats["delivery_cycles"].items():
            stats[f"projection_gate_multicast_delivery_m{width}"] = int(values.sum().item())
    if include_ordered_trace:
        stats.update({
            "pair_q_count_ordered_trace": _encode_ordered_count_trace(q_count),
            "pair_k_count_ordered_trace": _encode_ordered_count_trace(k_count),
            "pair_overlap_ordered_trace": _encode_ordered_count_trace(overlap),
            "pair_motion_ordered_trace": _encode_ordered_count_trace(motion),
            "pair_k_temporal_intersection_ordered_trace": _encode_ordered_count_trace(
                k_temporal_intersection
            ),
            "pair_k_temporal_union_ordered_trace": _encode_ordered_count_trace(k_temporal_union),
            "pair_update_ordered_trace": _encode_ordered_count_trace(update),
            "pair_four_vector_union_ordered_trace": _encode_ordered_count_trace(four_vector_union),
            "projection_baseline_active_lanes_ordered_trace": _encode_ordered_count_trace(
                projection_baseline_active_lanes_by_row
            ),
            "projection_class_channel_terms_ttx_ordered_trace": _encode_ordered_count_trace(
                projection_class_channel_terms_ttx_by_row
            ),
            "projection_class_channel_terms_h67_ordered_trace": _encode_ordered_count_trace(
                projection_class_channel_terms_h67_by_row
            ),
            "projection_class_channel_max_fanout_ttx_ordered_trace": _encode_ordered_count_trace(
                projection_class_channel_max_fanout_ttx_by_row
            ),
            "projection_class_channel_max_fanout_h67_ordered_trace": _encode_ordered_count_trace(
                projection_class_channel_max_fanout_h67_by_row
            ),
            "projection_active_classes_ttx_ordered_trace": _encode_ordered_count_trace(
                active_projection_classes_ttx
            ),
            "projection_active_classes_h67_ordered_trace": _encode_ordered_count_trace(
                active_projection_classes_h67
            ),
            "projection_h67_factor_class_segments_ordered_trace": (
                _encode_ordered_count_trace(
                    projection_h67_factor_class_segments_by_row
                )
            ),
            "projection_h67_factor_class_lane_segments_ordered_trace": (
                _encode_ordered_count_trace(
                    projection_h67_factor_class_lane_segments_by_row
                )
            ),
        })
        for width in (1, 2, 4, 8, 16):
            stats[f"projection_multicast_delivery_ttx_m{width}_ordered_trace"] = (
                _encode_ordered_count_trace(projection_multicast_delivery_ttx_by_width[width])
            )
            stats[f"projection_multicast_delivery_h67_m{width}_ordered_trace"] = (
                _encode_ordered_count_trace(projection_multicast_delivery_h67_by_width[width])
            )
        if gate_projection_stats is not None:
            stats["projection_gate_class_channel_terms_deploy_ordered_trace"] = (
                _encode_ordered_count_trace(gate_projection_stats["row_terms"])
            )
            stats["projection_gate_class_channel_max_fanout_deploy_ordered_trace"] = (
                _encode_ordered_count_trace(gate_projection_stats["max_fanout"])
            )
            stats["projection_active_gate_classes_deploy_ordered_trace"] = (
                _encode_ordered_count_trace(gate_projection_stats["active_classes"])
            )
            stats["projection_gate_ppdi_delivery_exact_ordered_trace"] = (
                _encode_ordered_count_trace(gate_projection_stats["ppdi_delivery"])
            )
            for width, values in gate_projection_stats["delivery_cycles"].items():
                stats[f"projection_gate_multicast_delivery_m{width}_ordered_trace"] = (
                    _encode_ordered_count_trace(values)
                )
            for group_windows, values in gate_projection_stats["grouped_terms"].items():
                stats[f"projection_gate_group_terms_g{group_windows}_ordered_trace"] = (
                    _encode_ordered_count_trace(values["terms"])
                )
                stats[
                    f"projection_gate_group_active_lanes_g{group_windows}_ordered_trace"
                ] = _encode_ordered_count_trace(values["active_lanes"])
                stats[
                    f"projection_gate_group_active_classes_g{group_windows}_ordered_trace"
                ] = _encode_ordered_count_trace(values["active_classes"])
                stats[
                    f"projection_gate_group_max_fanout_g{group_windows}_ordered_trace"
                ] = _encode_ordered_count_trace(values["max_fanout"])
                stats[
                    f"projection_gate_group_window_count_g{group_windows}_ordered_trace"
                ] = _encode_ordered_count_trace(values["window_count"])
                stats[
                    f"projection_gate_group_ppdi_delivery_g{group_windows}_ordered_trace"
                ] = _encode_ordered_count_trace(values["ppdi_delivery"])
                for width, delivery in values["delivery_cycles"].items():
                    stats[
                        f"projection_gate_group_delivery_g{group_windows}_m{width}_ordered_trace"
                    ] = _encode_ordered_count_trace(delivery)
    return stats


def _maybe_emit_h60_profile(
    module: nn.Module,
    *,
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    tx_scores: torch.Tensor,
    sc_scores: torch.Tensor,
    fused_scores: torch.Tensor,
    pre_quant_scores: torch.Tensor | None,
    gate: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> None:
    collector = getattr(module, "_h9_profile_collector", None)
    bit_trace_collector = getattr(module, "_h9_bit_trace_collector", None)
    if collector is None and bit_trace_collector is None:
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

        head_dim = int(k_orig.shape[-1])
        q_tokenized_active = _qkformer_token_q(q_orig.detach()).ne(0)
        k_tokenized_active = k_orig.detach().ne(0)
        q_active_count = q_tokenized_active.sum(dim=-1).to(dtype=torch.long)
        k_zero_token = ~k_tokenized_active.any(dim=-1)
        zaf_class_presence = torch.nn.functional.one_hot(
            q_active_count.clamp(min=0, max=head_dim),
            num_classes=head_dim + 1,
        ).bool() & k_zero_token.unsqueeze(-1)
        zaf_fold_classes = zaf_class_presence.any(dim=2).sum(dim=-1).float()
        zaf_active_entries = (~k_zero_token).sum(dim=2).float()

        temporal_stats: dict[str, Any] = {}
        if q_orig.ndim == 5 and q_orig.shape[0] == 2:
            q_binary = q_orig.detach().gt(0)
            batch, heads, total_tokens, head_dim = k_orig.shape
            spatial_tokens = q_orig.shape[3]
            if total_tokens == 2 * spatial_tokens:
                k_binary = k_orig.detach().gt(0).reshape(batch, heads, 2, spatial_tokens, head_dim)
                q_toggle = q_binary[0] ^ q_binary[1]
                k_toggle = k_binary[:, :, 0] ^ k_binary[:, :, 1]
                lane_elements = int(q_toggle.numel())
                q_toggle_elements = int(q_toggle.sum().item())
                k_toggle_elements = int(k_toggle.sum().item())
                update_elements = int((q_toggle | k_toggle).sum().item())
                include_ordered_trace = bool(
                    getattr(module, "_h9_profile_ordered_trace", False)
                )
                temporal_stats = {
                    "temporal_lane_elements": lane_elements,
                    "q_temporal_toggle_elements": q_toggle_elements,
                    "k_temporal_toggle_elements": k_toggle_elements,
                    "qk_temporal_update_elements": update_elements,
                    "q_temporal_toggle_density": q_toggle_elements / lane_elements,
                    "k_temporal_toggle_density": k_toggle_elements / lane_elements,
                    "qk_temporal_update_density": update_elements / lane_elements,
                    **_delta_locality_stats(
                        q_toggle,
                        k_toggle,
                        include_ordered_trace=include_ordered_trace,
                    ),
                    **_token_time_bundle_stats(
                        q_binary,
                        k_binary.permute(2, 0, 1, 3, 4),
                        include_ordered_trace=include_ordered_trace,
                    ),
                    **_binary_temporal_pair_stats(
                        q_binary,
                        k_binary.permute(2, 0, 1, 3, 4),
                        gate_q17_code=(
                            torch.round(gate_data.squeeze(-1) * 128.0).to(torch.long)
                            if gate_data.ndim == 4
                            and gate_data.shape[-1] == 1
                            and tuple(gate_data.shape[:-1])
                            == (batch, heads, total_tokens)
                            else None
                        ),
                        windows_per_sample=getattr(
                            module, "_h9_windows_per_sample", None
                        ),
                        include_ordered_trace=include_ordered_trace,
                    ),
                }

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
            "batch_windows": int(k_orig.shape[0]) if k_orig.ndim >= 1 else 0,
            "windows_per_sample": int(
                getattr(module, "_h9_windows_per_sample", k_orig.shape[0])
            ),
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
            **_hardware_score_clip_stats(pre_quant_scores, cfg),
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
            "zaf_kzero_token_ratio": _safe_float_stat(k_zero_token.float(), "mean"),
            "zaf_active_entries_mean": _safe_float_stat(zaf_active_entries, "mean"),
            "zaf_fold_classes_mean": _safe_float_stat(zaf_fold_classes, "mean"),
            **temporal_stats,
            **bundle_stats,
        }
    if collector is not None:
        collector(module, stats)
    if bit_trace_collector is not None:
        bit_trace_collector(
            module,
            q_orig=q_orig.detach(),
            k_orig=k_orig.detach(),
            gate=gate.detach(),
        )


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
    if cfg.binary_motion_xor_alpha:
        score = score + float(cfg.binary_motion_xor_alpha) * _binary_temporal_k_xor_popcount(
            q_orig, k_orig
        )
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


def _binary_temporal_k_xor_popcount(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
) -> torch.Tensor:
    """Per-token binary motion evidence from paired temporal K events.

    The Swin attention input uses ``q_orig=[T,B,H,N,D]`` and
    ``k_orig=[B,H,T*N,D]``. Each token receives the XOR popcount against the
    same spatial position in the other time slice. The arithmetic form keeps
    the binary forward value while preserving surrogate gradients.
    """

    if q_orig.ndim != 5 or k_orig.ndim != 4:
        raise ValueError("motion XOR requires q_orig=[T,B,H,N,D] and k_orig=[B,H,T*N,D]")
    t_steps, batch, heads, spatial_tokens, head_dim = q_orig.shape
    if t_steps != 2:
        raise ValueError("motion XOR currently requires a two-slice temporal window")
    if tuple(k_orig.shape) != (batch, heads, t_steps * spatial_tokens, head_dim):
        raise ValueError("k_orig shape is inconsistent with q_orig temporal/spatial layout")

    k_event = _binary_event_ste(k_orig).reshape(batch, heads, t_steps, spatial_tokens, head_dim)
    paired = k_event.flip(dims=(2,))
    return (k_event - paired).abs().sum(dim=-1, keepdim=True).reshape(
        batch, heads, t_steps * spatial_tokens, 1
    )


# ─────────────────────────────────────────────────────────────────────────────
# D1 (h87): Motion T=5 quintuple time quotient
#
# Contract: CLAUDE_OPERATOR_CONTRACT_DRAFTS_20260818.md 的 D1（Motion T>2 时间商）。
# 每时间槽位 t 的规范融合式（合同钉死，硬件与部署同式）：
#   s_t = min(RNE16(64·o_t + sz_t + 16·m̄_t), 162)
#   q_t = popcount(Q_t), k_t = popcount(K_t), o_t = popcount(Q_t & K_t)
#   sz_t = 32 − q_t − k_t + o_t（容斥界：max(0,q+k−32) ≤ o ≤ min(q,k)）
#   m̄_t = 槽位 t 的运动边（K_{t-1}⊕K_t 的 popcount；组内首槽 t≡0 采用组内第 1 条边，
#         与 H67 T=2 pair 的“pair 内唯一运动边同时喂两个槽位”语义一致，见 I4/I5 恒等式）
# 槽位分解 s_t = 4·o_t + r_t, r_t ∈ {0,1,2} 唯一（I2；物理域内无 s%4==3）。
#
# 布局约定（与 _binary_temporal_k_xor_popcount 相同）：
#   q_orig = [T=2, B*, H, N, D]，k_orig = [B*, H, 2N, D]
#   Swin window_partition_v2 把全部时间对窗堆叠进 batch 维：
#   B* = B × n_pairs × n_sw，行序 row = (b·n_pairs + wd)·n_sw + s。
#   因此 T=5 分组（跨窗时间槽）可在算子内部完成：10 bin → 2 组 × 5 槽，
#   组 g 覆盖 bin [5g, 5g+4]，跨组边 (4,5) 不可见（I7：8/9 时间边覆盖）。
# 实现方式选择：保持 Swin 分窗 (2,15,15) 与全部模型参数不动（续训起点
# Motion ep35 checkpoint 可直接加载，对比口径 = 纯算子消融），时间商分组
# 在算子内完成（选项 b；理由见 D1_MOTION_T5_IMPLEMENTATION_20260818.md）。
# ─────────────────────────────────────────────────────────────────────────────

# D1 全分辨率 (2,15,15) 分窗族各 stage 的空间窗数（含 ceil padding），
# 用于 batch 维自动分解 B* = B × n_pairs × n_sw 的候选（按 n_sw 降序试解，
# 首个整除解即采用；显式 cfg.temporal_quotient_batch 优先）。
_D1_SPATIAL_WINDOW_CANDIDATES = (1376, 352, 88, 24, 6, 2)


def _rne16_div_pow2_ste(
    numerator: torch.Tensor,
    denominator: int = 16,
) -> torch.Tensor:
    """Round-to-nearest-even 除以 2 的幂（与 RTL/analyze_binary_temporal_pair_arch 同式）。

    ``numerator`` 必须是整值张量（由 STE 二值事件的整数组合给出）；除法在
    int64 上逐位精确（Q7 档值域远小于 2^53，float32 无精度损失）。STE
    backward 按真实导数 1/denominator 直通（F2 修复，2026-08-19）：forward
    恒为 RNE16 商（与 RTL 逐位一致），梯度路径 ÷16，消除恒等 backward 造成
    的 o 项系数 65/16 梯度放大（D1 漂移诊断 §3.4/§6-F2）。
    """

    rounded = torch.round(numerator)
    # floor 商转 int64（分子 <= 2592 << 2^53，精确）：bitwise 奇偶判定需整型
    quotient = torch.div(rounded, denominator, rounding_mode="floor").to(torch.int64)
    remainder = torch.remainder(rounded, denominator)
    half = denominator // 2
    increment = remainder.gt(half) | (
        remainder.eq(half) & quotient.bitwise_and(1).ne(0)
    )
    result = (quotient + increment.to(dtype=quotient.dtype)).to(dtype=numerator.dtype)
    return numerator / float(denominator) + (
        result - numerator / float(denominator)
    ).detach()


def _d1_decompose_temporal_batch(
    batch_total: int,
    n_pairs: int,
    cfg: ShiftmaxAttentionConfig,
) -> tuple[int, int]:
    """把 attention batch 维分解为 (B, n_sw)：B* = B × n_pairs × n_sw。

    行序 row = (b·n_pairs + wd)·n_sw + s 由 window_partition_v2 固定
    （batch 维 = B × 时间对窗 × 空间窗）。在 D1 w15 全分辨率族各 stage
    的空间窗数候选中按 n_sw 降序试解；``temporal_quotient_batch`` 是
    **偏好**而非覆盖：训练 bs2 与评测 bs1 共用同一配置时，优先取与配置
    batch 一致的候选分解，无匹配时回退到首个整除解（评测时 batch 变化）。
    """

    candidates = []
    for n_sw in _D1_SPATIAL_WINDOW_CANDIDATES:
        if n_pairs * n_sw > 0 and batch_total % (n_pairs * n_sw) == 0:
            batch = batch_total // (n_pairs * n_sw)
            if n_sw * n_pairs * batch == batch_total:
                candidates.append((batch, n_sw))
    if not candidates:
        raise ValueError(
            "D1 batch decomposition failed: batch_total={} cannot be factored as "
            "B × {} (n_pairs) × n_sw for any D1 spatial-window count {}; set "
            "bsa_attention.temporal_quotient_batch explicitly".format(
                batch_total, n_pairs, _D1_SPATIAL_WINDOW_CANDIDATES
            )
        )
    explicit = int(cfg.temporal_quotient_batch or 0)
    if explicit > 0:
        for batch, n_sw in candidates:
            if batch == explicit:
                return batch, n_sw
    return candidates[0]


def _binary_t5_quotient_token_scores(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
    """D1 规范融合式 T=5 时间商逐 token 分数，布局 [B*, H, 2N, 1]。

    返回 (scores, rle_stats, slot_views)：
      scores     [B*, H, 2N, 1] —— 与 q/k 原 token 布局一致
                 （token (t_local, n) 取槽位 2·wd(idx) + t_local 的分数）
      rle_stats  dict —— 时间维 run-length 广播账（I6：每位置独立门数、
                 eq 边率、广播节省；供验证实验 3 的 dump 裁决）
      slot_views dict —— 5-slot 时间商视图（scores/overlap/remainder，
                 [B*, H, num_steps, N]，每位置 5 槽，I2 槽位分解 r=s%4）
    """

    if q_orig.ndim != 5 or k_orig.ndim != 4:
        raise ValueError("D1 T5 quotient requires q_orig=[T,B,H,N,D] and k_orig=[B,H,T*N,D]")
    t_steps, batch, heads, spatial_tokens, head_dim = q_orig.shape
    if t_steps != 2:
        raise ValueError("D1 T5 quotient requires the two-slice temporal window (2,15,15)")
    if tuple(k_orig.shape) != (batch, heads, t_steps * spatial_tokens, head_dim):
        raise ValueError("k_orig shape is inconsistent with q_orig temporal/spatial layout")
    num_steps = int(cfg.temporal_quotient_steps)
    quotient_len = int(cfg.temporal_quotient_len)
    if num_steps <= 0 or num_steps % t_steps != 0:
        raise ValueError(
            "D1 requires bsa_attention.temporal_quotient_steps > 0 and divisible by "
            f"the window T=2; got {num_steps}"
        )
    if quotient_len != 5:
        raise ValueError(f"D1 contract pins the quintuple length to 5; got {quotient_len}")
    n_pairs = num_steps // t_steps
    if num_steps % quotient_len != 0:
        raise ValueError(
            f"D1 requires num_steps % 5 == 0 (num_steps=10 -> 2 groups); got {num_steps}"
        )
    n_groups = num_steps // quotient_len

    q_event = _binary_event_ste(_qkformer_token_q(q_orig)).reshape(
        batch, heads, t_steps, spatial_tokens, head_dim
    )
    k_event = _binary_event_ste(k_orig).reshape(batch, heads, t_steps, spatial_tokens, head_dim)

    # ── 跨窗时间槽分组（I7 语义）：行序 row=(b·n_pairs+wd)·n_sw+s ──
    batch_actual, n_sw = _d1_decompose_temporal_batch(batch, n_pairs, cfg)
    if batch_actual * n_pairs * n_sw != batch:
        raise ValueError(
            f"D1 batch decomposition inconsistent: {batch_actual} × {n_pairs} × {n_sw} != {batch}"
        )
    row = torch.arange(batch, device=q_orig.device)
    wd_of_row = (row // n_sw) % n_pairs
    s_of_row = row % n_sw
    b_of_row = row // (n_pairs * n_sw)
    base_row = b_of_row * (n_pairs * n_sw) + s_of_row  # 同 (b, s) 的 pair 0 行号

    # 按全局 bin 重排：slot tb 的 K/Q = 行 base_row + (tb//2)·n_sw 处、
    # pair 内 t_local = tb%2 的事件。结果 [batch, H, num_steps, N, D]。
    q_slot_list = []
    k_slot_list = []
    for tb in range(num_steps):
        pair_row = base_row + (tb // 2) * n_sw
        # q_event 布局 [B*, H, 2, N, D]：时间维是第 3 维（tb % 2），
        # 第 1 维是 batch（行）。错误写成 q_event[tb % 2] 会取错行/越界。
        q_slot_list.append(q_event[:, :, tb % 2].index_select(0, pair_row))
        k_slot_list.append(k_event[:, :, tb % 2].index_select(0, pair_row))
    q_slot = torch.stack(q_slot_list, dim=2)  # [batch, H, num_steps, N, D]
    k_slot = torch.stack(k_slot_list, dim=2)

    # ── 每槽统计（I2 的物理域：容斥界）──
    q_count = q_slot.sum(dim=-1)  # [batch, H, num_steps, N]
    k_count = k_slot.sum(dim=-1)
    overlap = (q_slot * k_slot).sum(dim=-1)  # o_t = popcount(Q_t & K_t)
    same_zero = head_dim - q_count - k_count + overlap  # sz_t
    # 运动边：edge[tb] = popcount(K_{tb} ⊕ K_{tb+1})，tb=0..num_steps-2；
    # 槽位 t 采用边 min(t,1)-1 语义：首槽 t=0 采用组内第 1 条边（I4 与
    # H67 pair 的“同一运动边喂两个槽位”一致），t>=1 采用边 (t-1,t)。
    k_diff = (k_slot[:, :, :-1] - k_slot[:, :, 1:]).abs().sum(dim=-1)  # [batch, H, num_steps-1, N]
    slot_edge_index = torch.zeros(num_steps, dtype=torch.long, device=q_orig.device)
    for g in range(n_groups):
        slot_edge_index[g * quotient_len] = g * quotient_len  # 首槽复用组内第 1 条边
        slot_edge_index[g * quotient_len + 1 : (g + 1) * quotient_len] = torch.arange(
            g * quotient_len, (g + 1) * quotient_len - 1, device=q_orig.device
        )
    motion = k_diff.index_select(2, slot_edge_index)  # [batch, H, num_steps, N]

    # ── 规范融合式（I1：RNE16(64o+sz+16m̄) 为唯一规范，拆解式平局翻转处差 1 档）──
    numerator = 64.0 * overlap + same_zero + 16.0 * motion
    slot_scores = torch.clamp(_rne16_div_pow2_ste(numerator), max=162.0)

    # ── 时间维 run-length 广播账（I6：eq 边沿 T 广播）──
    group_scores = slot_scores.unflatten(2, (n_groups, quotient_len))
    eq_edge = group_scores[:, :, :, :-1].eq(group_scores[:, :, :, 1:])  # [batch, H, ng, 4, N]
    runs = 1 + (quotient_len - 1 - eq_edge.sum(dim=3).to(dtype=torch.float))  # [batch, H, ng, N]
    mean_runs = float(runs.float().mean().item())
    eq_rate = float(eq_edge.float().mean().item())
    rle_stats = {
        "mean_runs_per_position": mean_runs,
        "independent_gate_ratio": mean_runs / float(quotient_len),
        "broadcast_saving": 1.0 - mean_runs / float(quotient_len),
        "eq_edge_rate": eq_rate,
        "num_steps": num_steps,
        "quotient_len": quotient_len,
        "batch_decomposition": (batch_actual, n_pairs, n_sw),
    }

    # ── 写回原 token 布局：token (t_local, n) ← 槽 2·wd(idx) + t_local ──
    tb_of_token = (wd_of_row * 2).unsqueeze(-1) + torch.arange(
        2, device=q_orig.device
    ).view(1, 2)  # [batch, 2]
    gather_index = tb_of_token.unsqueeze(-1).expand(
        batch, 2, spatial_tokens
    ).unsqueeze(1).expand(batch, heads, 2, spatial_tokens)
    scores = slot_scores.gather(2, gather_index).reshape(
        batch, heads, t_steps * spatial_tokens, 1
    )

    slot_views = {
        "scores": slot_scores,
        "overlap": overlap,
        "remainder": slot_scores % 4,
        "motion": motion,
        "same_zero": same_zero,
    }
    return scores, rle_stats, slot_views


# ─────────────────────────────────────────────────────────────────────────────
# B2 (h87b): Motion T=4 + pad wildcard 时间商（D1 的 plan B 预案）
#
# Contract: D1_VARIANT_SEARCH_20260819.md §4.1（T=4+pad12）。num_steps=10 →
# 3 组 T=4 四元组：(0,1,2,3)、(4,5,6,7)、(8,9,pad,pad)——末组 2 个 pad 槽。
#
# pad 槽 wildcard 掩码语义（与敏感度账 len-2 口径等价）：
#   * pad 槽不参与商组：不贡献 run-length 统计（不产生 eq 边、不产生 run
#     断点——wildcard 合并；(pad,pad) 恒等），广播时按掩码跳过；
#   * pad 槽不进 slot 融合式（无 (o, sz, m̄) 统计、无分数）；
#   * 真实槽的融合式与 D1 逐位一致：s_t = min(RNE16(64·o_t + sz_t + 16·m̄_t), 162)，
#     组内首槽采用组内第 1 条边（I4）；跨组边 (3,4)/(7,8) 不可见（I7：
#     7/9 时间边覆盖——(8,9) 仍是 within-pair 边 0.9808）。
#
# 位账（逐边模型，敏感度账口径）：E[独立门]/位置 = Σ_g (1 + Σ_{组内真实边}(1−eq))
#   = (1+3(1−p̄₁)) + (1+3(1−p̄₂)) + (1+(1−p_w8,9)) = 3 + 7·(1−p̄)，p̄=0.879
#   → 1 − 3.851/10 = −61.5%（合同 −61.4%，全组口径 −64.6% 不采用：第三组
#   pad 不得与真实槽合并）。rle_stats.mean_runs_per_position 即 10 槽序列的
#   每位置总独立门数（非 D1 的每 (组,位置) 口径——pad 槽无门数可言）。
#
# 实现选择 (a)：独立新函数 + 新 mode motion_t4_pad_quotient/h87b，不触碰
# h87 函数与 I1-I7 既有单测（h87 的 quotient_len==5 / steps%5==0 校验原样
# 保留）；B2 与 D1 同族候选 n_sw、同一 batch 分解与布局写回。
# ─────────────────────────────────────────────────────────────────────────────


def _binary_t4_pad_quotient_token_scores(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> tuple[torch.Tensor, dict[str, Any], dict[str, torch.Tensor]]:
    """B2 (h87b) 规范融合式 T=4+pad 时间商逐 token 分数，布局 [B*, H, 2N, 1]。

    返回 (scores, rle_stats, slot_views)：
      scores     [B*, H, 2N, 1] —— 与 q/k 原 token 布局一致（真实槽）
      rle_stats  dict —— 含 wildcard pad 的 run-length 广播账（真实边 7 条、
                 group_lengths=(4,4,2)、pad_slots、coverage_edges=7）
      slot_views dict —— scores/overlap/remainder/motion/same_zero 仅真实槽
                 [B*, H, num_steps, N]；pad_mask [B*, H, n_groups, 4, N]
                 （True = pad，wildcard 掩码）；grouped_runs [B*, H, n_groups, N]
    """

    if q_orig.ndim != 5 or k_orig.ndim != 4:
        raise ValueError(
            "B2 T4 pad quotient requires q_orig=[T,B,H,N,D] and k_orig=[B,H,T*N,D]"
        )
    t_steps, batch, heads, spatial_tokens, head_dim = q_orig.shape
    if t_steps != 2:
        raise ValueError("B2 T4 pad quotient requires the two-slice temporal window (2,15,15)")
    if tuple(k_orig.shape) != (batch, heads, t_steps * spatial_tokens, head_dim):
        raise ValueError("k_orig shape is inconsistent with q_orig temporal/spatial layout")
    num_steps = int(cfg.temporal_quotient_steps)
    quotient_len = int(cfg.temporal_quotient_len)
    if num_steps <= 0 or num_steps % t_steps != 0:
        raise ValueError(
            "B2 requires bsa_attention.temporal_quotient_steps > 0 and divisible by "
            f"the window T=2; got {num_steps}"
        )
    if quotient_len != 4:
        raise ValueError(f"B2 contract pins the quadruple length to 4; got {quotient_len}")
    if num_steps % quotient_len == 0:
        raise ValueError(
            f"B2 requires num_steps % 4 != 0 (pad wildcard group; num_steps=10 -> 2 pad "
            f"slots); got {num_steps}; exact quadruple division belongs to h87"
        )
    pad_slots = quotient_len - (num_steps % quotient_len)
    n_pairs = num_steps // t_steps
    n_groups = (num_steps + quotient_len - 1) // quotient_len

    q_event = _binary_event_ste(_qkformer_token_q(q_orig)).reshape(
        batch, heads, t_steps, spatial_tokens, head_dim
    )
    k_event = _binary_event_ste(k_orig).reshape(batch, heads, t_steps, spatial_tokens, head_dim)

    # ── 跨窗时间槽分组（I7 语义，与 D1 同一行序）──
    batch_actual, n_sw = _d1_decompose_temporal_batch(batch, n_pairs, cfg)
    if batch_actual * n_pairs * n_sw != batch:
        raise ValueError(
            f"B2 batch decomposition inconsistent: {batch_actual} × {n_pairs} × {n_sw} != {batch}"
        )
    row = torch.arange(batch, device=q_orig.device)
    wd_of_row = (row // n_sw) % n_pairs
    s_of_row = row % n_sw
    b_of_row = row // (n_pairs * n_sw)
    base_row = b_of_row * (n_pairs * n_sw) + s_of_row  # 同 (b, s) 的 pair 0 行号

    q_slot_list = []
    k_slot_list = []
    for tb in range(num_steps):
        pair_row = base_row + (tb // 2) * n_sw
        # q_event 布局 [B*, H, 2, N, D]：时间维是第 3 维（tb % 2）
        q_slot_list.append(q_event[:, :, tb % 2].index_select(0, pair_row))
        k_slot_list.append(k_event[:, :, tb % 2].index_select(0, pair_row))
    q_slot = torch.stack(q_slot_list, dim=2)  # [batch, H, num_steps, N, D]
    k_slot = torch.stack(k_slot_list, dim=2)

    # ── 每真实槽统计（I2 物理域，与 D1 同式）──
    q_count = q_slot.sum(dim=-1)
    k_count = k_slot.sum(dim=-1)
    overlap = (q_slot * k_slot).sum(dim=-1)  # o_t
    same_zero = head_dim - q_count - k_count + overlap  # sz_t
    k_diff = (k_slot[:, :, :-1] - k_slot[:, :, 1:]).abs().sum(dim=-1)  # 边 (t,t+1)
    # 槽位 t 的运动边：组内首槽采用组内第 1 条边（I4），其余采用边 (t-1,t)；
    # 末组 (8,9) 两槽共享组内唯一边 (8,9)；跨组边 (3,4)/(7,8) 不可见（I7）。
    slot_edge_index = torch.zeros(num_steps, dtype=torch.long, device=q_orig.device)
    for g in range(n_groups):
        first = g * quotient_len
        last = min((g + 1) * quotient_len, num_steps)
        slot_edge_index[first] = first
        if last - first > 1:
            slot_edge_index[first + 1 : last] = torch.arange(
                first, last - 1, device=q_orig.device
            )
    motion = k_diff.index_select(2, slot_edge_index)  # [batch, H, num_steps, N]

    # ── 规范融合式（I1：与 D1 逐位同式）──
    numerator = 64.0 * overlap + same_zero + 16.0 * motion
    slot_scores = torch.clamp(_rne16_div_pow2_ste(numerator), max=162.0)  # [batch, H, 10, N]

    # ── 组布局 + wildcard pad 掩码：真实槽排进 [B, H, ng, 4, N] ──
    group_lengths = tuple(
        min(quotient_len, num_steps - g * quotient_len) for g in range(n_groups)
    )
    global_slot = (
        torch.arange(n_groups, device=q_orig.device).view(1, 1, n_groups, 1, 1) * quotient_len
        + torch.arange(quotient_len, device=q_orig.device).view(1, 1, 1, quotient_len, 1)
    )  # [1, 1, ng, 4, 1] 组布局中的全局槽位
    valid = (global_slot >= 0) & (global_slot < num_steps)  # [1, 1, ng, 4, 1]
    src_slot = global_slot.clamp(0, num_steps - 1)
    grouped = slot_scores.unsqueeze(3).expand(
        batch, heads, num_steps, quotient_len, spatial_tokens
    ).gather(
        2, src_slot.expand(batch, heads, n_groups, quotient_len, spatial_tokens)
    )  # [B, H, ng, 4, N]；pad 位填充（值无意义），一律由 valid 掩码排除

    # ── 时间维 run-length 广播账（pad 跳过：仅真实边计入，含 7/9 覆盖）──
    edge_valid = valid[:, :, :, :-1] & valid[:, :, :, 1:]  # [1, 1, ng, 3, 1]
    eq_edge = grouped[:, :, :, :-1].eq(grouped[:, :, :, 1:]) & edge_valid
    valid_edge_count = edge_valid.to(torch.float32).sum(dim=3)  # 各 3/3/1
    runs = 1.0 + valid_edge_count - eq_edge.to(torch.float32).sum(dim=3)  # [B, H, ng, N]
    total_gates = runs.sum(dim=2)  # [B, H, N] 每位置 10 槽序列总独立门数
    mean_runs_per_position = float(total_gates.float().mean().item())
    n_real_edges = int(edge_valid.float().sum().item())
    n_positions = batch * heads * spatial_tokens
    eq_edge_rate = float(eq_edge.float().sum().item()) / (n_real_edges * n_positions)
    rle_stats = {
        "mean_runs_per_position": mean_runs_per_position,  # E[独立门]/位置（10 槽）
        "independent_gate_ratio": mean_runs_per_position / float(num_steps),
        "broadcast_saving": 1.0 - mean_runs_per_position / float(num_steps),
        "eq_edge_rate": eq_edge_rate,  # 仅 7 条真实边
        "num_steps": num_steps,
        "quotient_len": quotient_len,
        "n_groups": n_groups,
        "group_lengths": group_lengths,
        "pad_slots": pad_slots,
        "coverage_edges": n_real_edges,
        "batch_decomposition": (batch_actual, n_pairs, n_sw),
    }

    # ── 写回原 token 布局：token (t_local, n) ← 槽 2·wd(idx) + t_local ──
    tb_of_token = (wd_of_row * 2).unsqueeze(-1) + torch.arange(
        2, device=q_orig.device
    ).view(1, 2)  # [batch, 2]
    gather_index = tb_of_token.unsqueeze(-1).expand(
        batch, 2, spatial_tokens
    ).unsqueeze(1).expand(batch, heads, 2, spatial_tokens)
    scores = slot_scores.gather(2, gather_index).reshape(
        batch, heads, t_steps * spatial_tokens, 1
    )

    pad_mask = valid.logical_not().expand(
        batch, heads, n_groups, quotient_len, spatial_tokens
    )  # [B*, H, 3, 4, N] True = pad（wildcard 掩码）
    slot_views = {
        "scores": slot_scores,  # 仅真实槽 [B*, H, 10, N]
        "overlap": overlap,
        "remainder": slot_scores % 4,
        "motion": motion,
        "same_zero": same_zero,
        "pad_mask": pad_mask,
        "grouped_runs": runs,
        "group_lengths": group_lengths,
    }
    return scores, rle_stats, slot_views


# ─────────────────────────────────────────────────────────────────────────────
# D3 (h88): local5_a3s —— 各向异性 stencil（A3S, Axis-Aligned Anisotropic Stencil）
#
# Contract: CLAUDE_OPERATOR_CONTRACT_DRAFTS_20260818.md 的 D3（Local5 线）。
# Local5 的 5-lane stencil 分数加方向场偏移 ±Δ（Δ=1/16 == 8 个 1/128 网格档，
# 网格精确位移，K2），方向场 = 3×3 时域 XOR 梯度 argmax（2bit/pixel，K3），
# 对齐 lane +Δ、正交 −Δ，把"唯一门"从 1 组 ident-K 分裂为对齐/正交/self 3 偏移类
# （K5；self 折叠进正交 => 2 类权重，gate-plane +1 slot/destination，raw16 广播 ×2）。
#
# Δ=0 锚点恒等（K1）：_binary_axnor_local5_a3s_attention 在 delta_bins==0 时
# **完全不触碰 scores**，其余算术与 _binary_alpha_xnor_stencil_attention
# （temporal_pair=False, spatial_cross=True, motion=0）逐式一致，故与现网
# Local5 门逐位一致（消融与回滚锚点，可注入式训练关键）。
#
# 数学定义（与 check_d3_axis_stencil_20260818.py 的 axis_field / a3s_gate 逐式一致）：
#   m = mean_t popcount(K_t ⊕ K_{t+1})            时域 XOR 梯度（uint8 位异或）
#   grad_axis = |roll(m, (dy,dx)) − m|             3×3 空间差分（E/W/N/S 各向）
#   dirs = argmax(grad_E, grad_W, grad_N, grad_S)  2bit/pixel 方向场码
#   offset[lane] = +Δ if dirs == lane_axis else −Δ（self lane 恒 0）
#
# 新存储对象：方向场位图（2bit/pixel，450bit/窗，<1% 现网存储增量）+ 对齐/正交
# 双权重槽（Δ 固定参数，无需训练，无梯度——方向场 .detach()，argmax 不传梯度）。
# 新执行对象：方向感知唯一门（ident-K 目的地 1 组 -> 3 偏移类；非 ident-K
# 目的地方向场 2bit 查表决定 lane 分组）。
# 诚实生效指标（K4 修正）：门质量再分配受 2^s 门动态范围约束天然有界，诚实
# 指标为**对齐 lane 量化分数 argmax 命中率**（winner class 指向运动轴）。
# Δ 注入式训练：a3s_delta_bins 满档、a3s_delta_warmup_steps > 0 时 Δ 从 0
# 线性渐增至满档（读 module._h9_global_step），起调档位即现网 Local5 恒等。
# ─────────────────────────────────────────────────────────────────────────────

_D3_AXIS_CODES = {"E": 0, "W": 1, "N": 2, "S": 3}  # 方向场码（与 check_d3 一致）


def _d3_axis_field(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
) -> torch.Tensor:
    """3×3 时域 XOR 梯度 argmax 方向场，返回 [B, H, H, W] 码（E=0 W=1 N=2 S=3）。

    与 check_d3_axis_stencil_20260818.py 的 axis_field 逐式一致：时域 XOR
    popcount 平面（uint8 位异或 + 通道求和）先做时间平均，再对 E/W/N/S 四个
    轴向取 |roll(m) − m| 空间差分，最后 argmax。k_orig 布局 [B, H, T*N, D]
    （t-major token 序，与 _binary_temporal_k_xor_popcount 同式），reshape
    为 [B, H, T, side, side, D]。方向场是固定位图（新存储对象，2bit/pixel，
    无梯度——.detach()，argmax 本身亦不可微）。
    """

    if q_orig.ndim != 5 or k_orig.ndim != 4:
        raise ValueError(
            "A3S direction field requires q_orig=[T,B,H,N,D] and k_orig=[B,H,T*N,D]"
        )
    t_steps, batch, heads, spatial_tokens, head_dim = q_orig.shape
    if t_steps < 2:
        raise ValueError("A3S direction field requires the two-slice temporal window (T>=2)")
    if tuple(k_orig.shape) != (batch, heads, t_steps * spatial_tokens, head_dim):
        raise ValueError("k_orig shape is inconsistent with q_orig temporal/spatial layout")
    spatial_side = math.isqrt(spatial_tokens)
    if spatial_side * spatial_side != spatial_tokens:
        raise ValueError("A3S direction field requires a square spatial window")
    k_bin = (k_orig.detach() > 0).to(dtype=torch.uint8)
    k_planes = k_bin.reshape(
        batch, heads, t_steps, spatial_side, spatial_side, head_dim
    )
    motion = (k_planes[:, :, 1:] ^ k_planes[:, :, :-1]).sum(dim=-1).to(torch.float32)
    motion = motion.mean(dim=2)  # 时间平均（与 check_d3 的 mean(dim=0) 同式）
    grads = []
    for dy, dx in ((0, 1), (0, -1), (-1, 0), (1, 0)):  # E, W, N, S
        rolled = torch.roll(motion, shifts=(dy, dx), dims=(-2, -1))
        grads.append((rolled - motion).abs())
    stacked = torch.stack(grads, dim=-1)  # [B, H, H, W, 4]
    return stacked.argmax(dim=-1)


def _d3_effective_delta_bins(
    cfg: ShiftmaxAttentionConfig,
    profile_module: nn.Module | None,
) -> int:
    """Δ 注入式渐增：a3s_delta_warmup_steps 内从 0 线性升至 a3s_delta_bins 满档。

    warmup<=0（或满档 0）时立即返回满档（0 = Δ=0 恒等锚点档）。起调 Δ=0 档
    = 现网 Local5 逐位恒等（K1），随后平滑注入各向异性偏移——loss 不塌的
    结构性保障（同 D1 的"起调即锚点"纪律）。
    """

    target = max(0, int(cfg.a3s_delta_bins))
    warmup = max(0, int(cfg.a3s_delta_warmup_steps))
    if target <= 0 or warmup <= 0:
        return target
    step = max(0, int(getattr(profile_module, "_h9_global_step", 0)))
    return min(target, int(round(target * step / warmup)))


def _d3_a3s_offset(
    scores: torch.Tensor,
    dirs: torch.Tensor,
    delta_bins: int,
) -> torch.Tensor:
    """对齐 lane +Δ / 正交 −Δ / self 0 的分数偏移（[B, H, T*N, 5]）。

    lane 序 = self, N, S, W, E（与 _binary_alpha_xnor_stencil_attention 的
    spatial_cross 循环 (0,0),(-1,0),(1,0),(0,-1),(0,1) 同序）。偏移式与
    check_d3 的 a3s_gate 一致：offset = +Δ·[dirs==axis] − Δ·(1−[dirs==axis])
    （self lane 恒 0）。Δ = delta_bins × 1/128，Q7 网格精确位移（K2：与分数
    量化 commute，clamp 外）。方向场按像素（两时间切片共享同一 2bit 位图）。
    """

    if delta_bins <= 0:
        return torch.zeros_like(scores)
    if scores.shape[-1] != 5:
        raise ValueError("A3S offset expects the Local5 5-lane score layout")
    batch, heads, tokens, _ = scores.shape
    dirs_flat = dirs.reshape(batch, heads, -1)  # [B, H, N]
    spatial_tokens = dirs_flat.shape[-1]
    if tokens % spatial_tokens != 0:
        raise ValueError("A3S offset cannot tile the direction field onto tokens")
    t_steps = tokens // spatial_tokens
    delta = float(delta_bins) * (1.0 / 128.0)
    dirs_tokens = (
        dirs_flat.unsqueeze(2)
        .expand(batch, heads, t_steps, spatial_tokens)
        .reshape(batch, heads, tokens)
    )
    lane_axis = ("self", "N", "S", "W", "E")
    lane_codes = {
        "N": _D3_AXIS_CODES["N"],
        "S": _D3_AXIS_CODES["S"],
        "W": _D3_AXIS_CODES["W"],
        "E": _D3_AXIS_CODES["E"],
    }
    offset = torch.zeros_like(scores)
    for li, axis in enumerate(lane_axis):
        if axis == "self":
            continue
        aligned = (dirs_tokens == lane_codes[axis]).to(dtype=scores.dtype)
        offset[..., li] = delta * aligned - delta * (1.0 - aligned)
    return offset


def _binary_axnor_local5_a3s_attention(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
    *,
    profile_module: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Local5 5-lane stencil + A3S 方向场偏移 ±Δ 的注意力（mode local5_a3s/h88）。

    算术顺序与 _binary_alpha_xnor_stencil_attention（temporal_pair=False,
    spatial_cross=True, motion_xor_alpha=0.0）逐式一致；唯一差异是在
    _apply_hardware_score_quant 之前插入方向场偏移：
      scores += +Δ·[dirs==lane_axis] − Δ·[dirs!=lane_axis]（self lane 恒 0）
    delta_bins==0（K1 锚点）时不触碰 scores——与现网 Local5 逐位一致，
    消融与回滚锚点，可注入式训练关键。motion alpha 与 Local5 分支同纪律
    （静默忽略，H66d 模板继承 binary_motion_xor_alpha 时保持位稳定）。

    返回 (attn, row_sum, gate, a3s_stats)：
      attn/row_sum/gate 与现网 Local5 同布局；
      a3s_stats：direction_field [B,H,H,W] 位图 / delta_bins 实际档 /
      axis_frac_ew（E/W 轴占比，K3 语义账）/ winner_hit_rate（对齐 lane
      argmax 命中率，K4 诚实指标，仅运动承载像素）/ aligned_lane /
      motion_mask（forward 验证挂载用，均 .detach()）。
    """

    if profile_module is not None:
        profile_module._h9_source_gate_cardinality_proxy = None

    q_event = (_qkformer_token_q(q_orig) > 0).to(dtype=q_orig.dtype)
    k_event = (k_orig > 0).to(dtype=q_orig.dtype)
    batch, heads, n_tokens, head_dim = q_event.shape
    t_steps = int(q_orig.shape[0])
    spatial_tokens = n_tokens // t_steps
    spatial_side = math.isqrt(spatial_tokens)
    height, width = spatial_side, spatial_side
    if t_steps * height * width != n_tokens:
        raise ValueError(
            "binary axnor Local5 A3S expects a T x H x H square window, "
            f"got T={t_steps}, tokens={n_tokens}"
        )

    grid = torch.arange(n_tokens, device=q_orig.device).reshape(t_steps, height, width)
    neighbor_indices = [grid]
    valid_masks = [torch.ones_like(grid, dtype=torch.bool)]

    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        yy = torch.arange(height, device=q_orig.device).view(1, height, 1) + dy
        xx = torch.arange(width, device=q_orig.device).view(1, 1, width) + dx
        valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
        yy = yy.clamp(0, height - 1).expand(t_steps, height, width)
        xx = xx.clamp(0, width - 1).expand(t_steps, height, width)
        tt = torch.arange(t_steps, device=q_orig.device).view(t_steps, 1, 1).expand_as(yy)
        neighbor_indices.append(grid[tt, yy, xx])
        valid_masks.append(valid.expand(t_steps, height, width))

    index = torch.stack(neighbor_indices, dim=-1).reshape(n_tokens, -1)
    valid = torch.stack(valid_masks, dim=-1).reshape(n_tokens, -1)
    k_candidates = k_event[:, :, index, :]
    q_candidates = q_event.unsqueeze(-2)
    same_spike = (q_candidates * k_candidates).sum(dim=-1)
    same_silent = ((1.0 - q_candidates) * (1.0 - k_candidates)).sum(dim=-1)
    scores = same_spike + float(cfg.alpha0) * same_silent
    scores = _normalize_consensus_score(scores, head_dim, cfg, active=None)
    if float(cfg.matrix_diag_bias) != 0.0:
        scores[..., 0] = scores[..., 0] + float(cfg.matrix_diag_bias)

    # D3: 方向场偏移（Δ=0 档不触碰 scores —— K1 逐位恒等锚点）
    dirs = _d3_axis_field(q_orig, k_orig)
    delta_bins = _d3_effective_delta_bins(cfg, profile_module)
    if delta_bins > 0:
        offset = _d3_a3s_offset(scores, dirs, delta_bins)
        scores = scores + offset

    # Deploy path：Q7 分数网格，然后掩掉无效候选到最小编码（与现网同式）。
    scores = _apply_hardware_score_quant(scores, cfg)
    invalid_fill = (
        float(cfg.hardware_score_min)
        if cfg.hardware_quant_enabled and cfg.hardware_score_min is not None
        else -1.0e4
    )
    scores = scores.masked_fill(~valid.view(1, 1, n_tokens, -1), invalid_fill)
    valid_for_gate = valid.view(1, 1, n_tokens, -1)

    if cfg.hardware_rtl_shiftmax_enabled:
        gate = _rtl_shiftmax_gate_q17(
            scores,
            dim=-1,
            preserve_mean=bool(cfg.preserve_mean),
            valid_mask=(
                valid_for_gate
                if cfg.hardware_mask_invalid_candidates
                else None
            ),
        )
    else:
        gate_scores = (
            scores.masked_fill(~valid_for_gate, -float("inf"))
            if cfg.hardware_mask_invalid_candidates
            else scores
        )
        gate = shiftmax(gate_scores, dim=-1, eps=cfg.eps)
        if cfg.preserve_mean:
            gate = gate * float(index.shape[-1])
        if cfg.hardware_mask_invalid_candidates:
            gate = gate.masked_fill(~valid_for_gate, 0.0)
    gate = _apply_hardware_gate_quant(gate, cfg)

    value_candidates = k_orig[:, :, index, :]
    attn = (gate.unsqueeze(-1) * value_candidates).sum(dim=-2)
    row_sum = gate.sum(dim=-1)

    # D3 挂载账本（诚实成本 + 方向场语义；forward 验证用，均 .detach()）
    dirs_flat = dirs.reshape(batch, heads, -1)
    n_spatial = dirs_flat.shape[-1]
    axis_frac_ew = float(
        (dirs_flat <= _D3_AXIS_CODES["W"])
        .to(dtype=torch.float32)
        .mean()
        .detach()
        .cpu()
    )
    # 运动承载像素：至少一个时刻、任一通道有事件的像素（K4 只在此子集上度量）
    bar_mask = (
        k_event.detach()
        .reshape(batch, heads, t_steps, n_spatial, head_dim)
        .sum(dim=(2, 4))
        > 0
    )  # [B, H, N]
    aligned_lane = torch.full(
        (batch, heads, n_spatial), -1, dtype=torch.long, device=q_orig.device
    )
    for code, lane in (
        (_D3_AXIS_CODES["E"], 4),
        (_D3_AXIS_CODES["W"], 3),
        (_D3_AXIS_CODES["N"], 1),
        (_D3_AXIS_CODES["S"], 2),
    ):
        aligned_lane = torch.where(dirs_flat == code, lane, aligned_lane)
    gate_argmax = gate.detach().argmax(dim=-1).reshape(
        batch, heads, t_steps, n_spatial
    )
    mask_tokens = (
        bar_mask.unsqueeze(2)
        .expand(batch, heads, t_steps, n_spatial)
        .reshape(batch, heads, -1)
    )
    hit = gate_argmax == aligned_lane.unsqueeze(2)
    winner_hit_rate = (
        float(hit.reshape(batch, heads, -1)[mask_tokens].float().mean().detach().cpu())
        if bool(mask_tokens.any())
        else 0.0
    )
    a3s_stats = {
        "direction_field": dirs.detach(),
        "delta_bins": delta_bins,
        "axis_frac_ew": axis_frac_ew,
        "winner_hit_rate": winner_hit_rate,
        "aligned_lane": aligned_lane.detach(),
        "motion_mask": bar_mask.detach(),
    }
    return attn, row_sum, gate, a3s_stats


# ─────────────────────────────────────────────────────────────────────────────
# D2 (h89): motion_sw12_overlap —— 跨窗语义（stride-12/窗口-15 重叠滑窗 + 滚动分母）
#
# Contract: CLAUDE_OPERATOR_CONTRACT_DRAFTS_20260818.md 的 D2（J1-J6 恒等式，
# CPU 验证脚本 entrypoints/check_d2_overlap_rolling_partition_20260818.py）。
#
# 窗口划分决策（与 check_d2 的 window_partition_overlap 逐式一致）：
#   在注意力算子内部做 stride-12 滑动分区（不动 Swin window_partition_v2 底层）：
#   Swin 把每个时间对窗的 (2,15,15) tile 堆叠进 batch 维，行序
#   row = (b·n_pairs + wd)·n_sw + s（s = ty·n_x + tx 为 tile 行优先索引）。
#   算子把同一 (b, wd) 场的 n_sw 个 tile 还原为 padded 场 (15·n_y, 15·n_x)
#   （“在算子输入上做重叠窗口的重新分区与还原”），再按 1D 链
#   start = 0, stride, 2·stride, …（end 在 total 处 clamp，尾窗部分覆盖）对
#   每个轴做重叠分区，得 n_oy × n_ox 个重叠窗，每窗 450 token（尾窗更少）。
#   token 身份码 = 场坐标 (t, y, x)：相邻窗交叠带（3 宽 × 15 长 × 2 时 = 90
#   token/窗边）中的同一 token 在两侧窗中携带相同身份码与相同 Q7 分数码
#   （J2，按构造成立）；该身份码即跨窗 quotient 目录的机制基板（J4/J5）。
#
# 每 token 分数 = Motion-XOR 规范融合式（D1 同款 canonical，m̄ 在 RNE 内）：
#   q_t=popcount(Q_t), k_t=popcount(K_t), o_t=popcount(Q_t&K_t)
#   sz_t = head_dim − q_t − k_t + o_t
#   m̄ = popcount(K_0 ⊕ K_1)（pair 运动边，逐位置）
#   s = min(RNE16(64·o_t + sz_t + 16·m̄), 162)（Q7 网格 [0,162]；
#       m̄=0 时与 check_d2 的 score_of_token 逐位一致 -> J4/J5 草案实测
#       （J mean 0.948 / 目录贡献 55.0%）在算子分数上原样成立）
#
# 滚动分母（J1 逐位精确，硬约束）：
#   Z_{i+1} = Z_i − Σ_leave + Σ_enter。Z = Σ 2^{s} 是整数幂和（最大
#   s=162 -> 2^162，超出 int64），故用 16bit 块分解（11 块：
#   c = s>>4，v = 1<<(s&15)；每块和 ≤ 450·2^15 < 2^24，int64 逐位精确），
#   增量式用闭环 z_roll = z_full[0] + cumsum(enter − leave)（行优先链，
#   prev(w) = w−1 恒成立，w=0 时 leave=enter=0），与全量重算逐位相等（J1）。
#   leave/enter 项从算子内数据得到：members[w] 由窗口几何（ys/xs 坐标）与
#   dense 布局索引直接算出，leave = members[w−1] \ members[w]（12 宽出带，
#   尾窗 clamp），enter = members[w] \ members[w−1]（12 宽进带，尾窗 clamp）。
#
# 门与还原（J3）：每重叠窗内 shiftmax 归一化（无效 token 掩掉）-> 门 g_w；
#   逐 token 聚合 g_final(t) = Σ_{w∋t} g_w(t)，还原门 = g_final/mult
#   （mult = 重叠重数），故 Σ_t mean(t)·mult(t) == #windows 精确成立（J3
#   等价式）；还原回 Swin dense tile 布局 [B*, H, 450]。
#
# 流量账（J6，check_d2 同式）：450 → 270 exp-term/窗（−40%），窗口数
#   +58.7%（300×390 网格），净 exp-add −4.8%；增量执行器 + 跨窗目录
#   （目录缓存 y 向共享带 90 项，见本文件 _d2_catalog_*）。
# ─────────────────────────────────────────────────────────────────────────────

# D2 全分辨率 w15 族各 stage 的 (n_y, n_x) tile 网格（n_y·n_x = D1 空间窗数：
# 1376/352/88/24/6/2），用于从 batch 分解自动反推 padded 场形状。
_D2_FIELD_GRID_BY_WINDOWS = {
    1376: (32, 43),
    352: (16, 22),
    88: (8, 11),
    24: (4, 6),
    6: (2, 3),
    2: (1, 2),
}
_D2_CHUNK_BITS = 16
_D2_N_CHUNKS = 11  # s ∈ [0, 162] -> c = s>>4 ∈ [0, 10]


def _d2_overlap_chain(
    total: int,
    wsize: int,
    stride: int,
) -> list[tuple[int, int]]:
    """1D 重叠窗链（含尾部 pad clamp），与 check_d2 的 window_partition_overlap 同式。

    返回 (start, end) 列表：start = 0, stride, 2·stride, …；end = min(start+wsize,
    total)；当 end 触及 total 时链终止（尾窗部分覆盖，宽度 < wsize）。
    """

    if wsize <= 0 or stride <= 0:
        raise ValueError("D2 overlap chain requires wsize > 0 and stride > 0")
    win: list[tuple[int, int]] = []
    start = 0
    while True:
        end = min(start + wsize, total)
        win.append((start, end))
        if end >= total:
            break
        start += stride
    return win


def _d2_decompose_field_batch(
    batch_total: int,
    n_pairs: int,
    cfg: ShiftmaxAttentionConfig,
) -> tuple[int, int]:
    """把 attention batch 维分解为 (B, n_sw)：B* = B × n_pairs × n_sw。

    行序 row = (b·n_pairs + wd)·n_sw + s 由 window_partition_v2 固定；候选
    n_sw 与 D1 同一族（各 stage 空间窗数），`sw12_batch` 是偏好而非覆盖
    （评测 bs1 时 batch 变化，自动回退首个整除解）。
    """

    candidates = []
    for n_sw in _D1_SPATIAL_WINDOW_CANDIDATES:
        if n_pairs * n_sw > 0 and batch_total % (n_pairs * n_sw) == 0:
            batch = batch_total // (n_pairs * n_sw)
            if batch * n_pairs * n_sw == batch_total:
                candidates.append((batch, n_sw))
    if not candidates:
        raise ValueError(
            "D2 field-batch decomposition failed: batch_total={} cannot be "
            "factored as B × {} (n_pairs) × n_sw for any D2 spatial window "
            "count {}; set bsa_attention.sw12_window_grid explicitly".format(
                batch_total, n_pairs, _D1_SPATIAL_WINDOW_CANDIDATES
            )
        )
    explicit = int(cfg.sw12_batch or 0)
    if explicit > 0:
        for batch, n_sw in candidates:
            if batch == explicit:
                return batch, n_sw
    return candidates[0]


def _d2_field_grid(
    n_sw: int,
    cfg: ShiftmaxAttentionConfig,
) -> tuple[int, int]:
    """(n_y, n_x) tile 网格：cfg.sw12_window_grid 显式优先，否则按 n_sw 查表。"""

    explicit = tuple(int(v) for v in (cfg.sw12_window_grid or (0, 0)))
    if explicit != (0, 0):
        if explicit[0] <= 0 or explicit[1] <= 0 or explicit[0] * explicit[1] != int(n_sw):
            raise ValueError(
                f"D2 sw12_window_grid {explicit} is inconsistent with n_sw={n_sw}"
            )
        return explicit
    grid = _D2_FIELD_GRID_BY_WINDOWS.get(int(n_sw))
    if grid is None:
        raise ValueError(
            f"D2 auto field grid failed for n_sw={n_sw}; known w15 grids are "
            f"{sorted(_D2_FIELD_GRID_BY_WINDOWS)}; set sw12_window_grid explicitly"
        )
    return grid


def _d2_overlap_window_plan(
    n_y: int,
    n_x: int,
    wsize: int,
    stride: int,
    device: torch.device,
) -> dict[str, Any]:
    """重叠窗几何计划：窗口坐标、成员索引、身份码、重叠重数 mult。

    返回 dict：
      ys / xs     1D 重叠链 [(start, end)]（场坐标）
      n_oy / n_ox / n_ow   各轴链长与重叠窗总数
      field_h / field_w    padded 场尺寸（15·n_y, 15·n_x）
      row_idx / tok_idx   [n_ow, t·225] 每个重叠窗成员的 dense 行内 tile
                          （row_idx）与行内 token 下标（tok_idx，含 t·225 偏移）；
                          成员 = 场上 15×15（尾窗 clamp 后更少）矩形
      valid               [n_ow, t·225] 有效成员掩码（尾窗边界外为 False）
      mult                [n_sw, t·225] 每 dense token 的重叠重数 mult_y·mult_x
      members             [n_ow, t·225] 布尔成员掩码（== valid 展开）
    """

    field_h, field_w = wsize * n_y, wsize * n_x
    ys = _d2_overlap_chain(field_h, wsize, stride)
    xs = _d2_overlap_chain(field_w, wsize, stride)
    n_oy, n_ox = len(ys), len(xs)
    n_ow = n_oy * n_ox
    n_sw = n_y * n_x
    y0 = torch.tensor([s for s, _ in ys], device=device, dtype=torch.long)
    x0 = torch.tensor([s for s, _ in xs], device=device, dtype=torch.long)
    yloc = torch.arange(wsize, device=device, dtype=torch.long)
    xloc = torch.arange(wsize, device=device, dtype=torch.long)
    # 场坐标 fy [n_oy, 15, 1] / fx [1, n_ox, 1, 15]（fx 已是 4D，勿再增维）
    fy = y0[:, None, None] + yloc[None, :, None]
    fx = x0[None, :, None, None] + xloc[None, None, None, :]
    valid_y = (fy < field_h)[:, None, :, :]  # [n_oy, 1, 15, 1]
    valid_x = (fx < field_w)[None, :, :, :]  # [1, 1, n_ox, 15]
    fy = fy[:, None, :, :].expand(n_oy, n_ox, wsize, wsize)
    fx = fx.expand(n_oy, n_ox, wsize, wsize)
    valid = valid_y & valid_x  # 广播到 [n_oy, n_ox, 15, 15]
    # dense 行内下标：行 = tile (y//15, x//15) 行优先，token = (y%15)*15 + (x%15)
    row = (fy // wsize) * n_x + (fx // wsize)
    tok = (fy % wsize) * wsize + (fx % wsize)
    row = row.reshape(n_ow, wsize * wsize)
    tok = tok.reshape(n_ow, wsize * wsize)
    valid = valid.reshape(n_ow, wsize * wsize)
    # 两时间切片展开：[n_ow, 2·225]，token 下标含 t·225 偏移
    t_idx = torch.arange(2, device=device, dtype=torch.long)
    row = row[:, None, :].expand(n_ow, 2, wsize * wsize).reshape(n_ow, 2 * wsize * wsize)
    tok = (t_idx[None, :, None] * (wsize * wsize) + tok[:, None, :]).reshape(
        n_ow, 2 * wsize * wsize
    )
    valid = valid[:, None, :].expand(n_ow, 2, wsize * wsize).reshape(
        n_ow, 2 * wsize * wsize
    )
    # 尾窗越界成员的下标钳制到 0：这些位置只在 gather 后被 valid 掩掉
    # （scores −inf / terms 0 / gates 0），钳制保证 gather 永不出界。
    row = torch.where(valid, row, torch.zeros_like(row))
    tok = torch.where(valid, tok, torch.zeros_like(tok))
    # 滚动链的 entry/exit 条带（J1 的 leave/enter 项，按窗口几何直接得到）：
    #   entry_band[w] = 窗 w 的 stride 宽进带（水平：x_loc >= wsize−stride，
    #     垂直换行：y_loc >= wsize−stride，均按场边界 clamp；w=0 恒 0）
    #   exit_band[w]  = 窗 w 的 stride 宽出带（水平：x_loc < stride，行尾换行：
    #     y_loc < stride；末窗恒 0）
    # 注意：leave/enter 是 (row, tok) 键集差，不能由成员掩码相减得到，
    # 因为相邻窗的 900 个 gather 位置布局不同——故用场坐标条带几何直接给。
    yloc2 = yloc[:, None]
    xloc2 = xloc[None, :]
    entry_band = torch.zeros(n_oy, n_ox, wsize, wsize, dtype=torch.bool, device=device)
    exit_band = torch.zeros(n_oy, n_ox, wsize, wsize, dtype=torch.bool, device=device)
    for wy in range(n_oy):
        for wx in range(n_ox):
            if wx > 0:
                entry_band[wy, wx] = (xloc2 >= wsize - stride) & (
                    x0[wx] + xloc2 < field_w
                )
            elif wy > 0:
                entry_band[wy, wx] = (yloc2 >= wsize - stride) & (
                    y0[wy] + yloc2 < field_h
                )
            if wy == n_oy - 1 and wx == n_ox - 1:
                continue  # 末窗无后继
            if wx < n_ox - 1:
                exit_band[wy, wx] = xloc2 < stride
            else:
                exit_band[wy, wx] = yloc2 < stride
    entry_band = entry_band.reshape(n_ow, wsize * wsize)[:, None, :].expand(
        n_ow, 2, wsize * wsize
    ).reshape(n_ow, 2 * wsize * wsize)
    exit_band = exit_band.reshape(n_ow, wsize * wsize)[:, None, :].expand(
        n_ow, 2, wsize * wsize
    ).reshape(n_ow, 2 * wsize * wsize)
    # 重叠重数 mult（每 dense token 被多少窗覆盖）
    mult_y = (
        (y0[:, None] <= torch.arange(field_h, device=device)[None, :])
        & (torch.arange(field_h, device=device)[None, :] < y0[:, None] + wsize)
    ).sum(dim=0)  # [field_h]
    mult_x = (
        (x0[:, None] <= torch.arange(field_w, device=device)[None, :])
        & (torch.arange(field_w, device=device)[None, :] < x0[:, None] + wsize)
    ).sum(dim=0)  # [field_w]
    ty = torch.arange(n_y, device=device)
    tx = torch.arange(n_x, device=device)
    ypix = torch.arange(wsize, device=device)
    xpix = torch.arange(wsize, device=device)
    # 每 dense token（s, t, yt·15+xt）的场坐标 y = 15·(s//n_x) + yt, x = 15·(s%n_x) + xt
    s_idx = torch.arange(n_sw, device=device)
    fy_dense = (s_idx[:, None, None] // n_x) * wsize + ypix[None, :, None]  # [n_sw, 15, 15]
    fx_dense = (s_idx[:, None, None] % n_x) * wsize + xpix[None, None, :]  # [n_sw, 15, 15]
    mult = (
        mult_y[fy_dense] * mult_x[fx_dense]
    ).reshape(n_sw, wsize * wsize)[:, None, :].expand(
        n_sw, 2, wsize * wsize
    ).reshape(n_sw, 2 * wsize * wsize)
    return {
        "ys": ys,
        "xs": xs,
        "n_y": n_y,
        "n_x": n_x,
        "n_oy": n_oy,
        "n_ox": n_ox,
        "n_ow": n_ow,
        "field_h": field_h,
        "field_w": field_w,
        "wsize": wsize,
        "stride": stride,
        "row_idx": row,
        "tok_idx": tok,
        "valid": valid,
        "members": valid,
        "entry_band": entry_band,
        "exit_band": exit_band,
        "mult": mult,
    }


def _d2_pow2_chunk(
    win_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Q7 分数整数幂和的 16bit 块分解：(块号 c, 块内值 v=1<<(s&15))，int64 精确。

    块级位移是 4bit（c = s>>4 ∈ [0,10]），每块容纳 16 个值位
    （v = 1<<(s&15) ≤ 2^15）；重组 s = 16·c + (s&15) 由 16·c 位移恢复。
    """

    chunk = torch.div(win_scores, 1 << 4, rounding_mode="floor")
    term = torch.bitwise_left_shift(
        torch.ones_like(win_scores), win_scores.bitwise_and(_D2_CHUNK_BITS - 1)
    )
    return chunk, term


def _d2_exp_flow_ledger(
    field_h: int,
    field_w: int,
    wsize: int,
    stride: int,
    t_slices: int = 2,
) -> dict[str, Any]:
    """exp-add 流量账（J6，与 check_d2 的 check_J6 逐式一致）。

    合同口径（check_d2 同式）：每新窗的增量 exp 项 = 450 − 2 条 3 宽共享带
    ×90 = 270/窗（y/x 向共享带分别由滚动链与跨窗目录复用），窗口数按
    stride-12 链计数（尾窗 clamp 不改变计数公式）。
    """

    dense_w = (field_h // wsize) * (field_w // wsize)
    n_y = (
        (field_h - wsize) // stride + 1
        if (field_h - wsize) % stride == 0
        else (field_h - wsize) // stride + 2
    )
    n_x = (
        (field_w - wsize) // stride + 1
        if (field_w - wsize) % stride == 0
        else (field_w - wsize) // stride + 2
    )
    overlap_w = n_y * n_x
    per_win_full = wsize * wsize * t_slices
    per_win_inc = per_win_full - 2 * t_slices * wsize * (wsize - stride)
    dense_terms = dense_w * per_win_full
    overlap_terms = overlap_w * per_win_inc
    return {
        "dense_windows": dense_w,
        "overlap_windows": overlap_w,
        "window_ratio": overlap_w / dense_w,
        "per_window_full": per_win_full,
        "per_window_incremental_formula": per_win_inc,
        "dense_total_terms": dense_terms,
        "overlap_total_terms": overlap_terms,
        "net_delta": 1.0 - overlap_terms / dense_terms,
    }


def _d2_catalog_bands(
    plan: dict[str, Any],
    scores_field: torch.Tensor,
) -> dict[str, Any]:
    """跨窗 quotient 目录：相邻窗共享带的身份码与类码（新存储对象）。

    x 带 = 水平相邻对 (wy, xi)-(wy, xi+1) 的 3 宽列带；y 带 = 垂直相邻对
    (yi, wx)-(yi+1, wx) 的 3 宽行带。身份码 = 场坐标压平下标
    (t·field_h + y)·field_w + x；类码 = 共享 token 的 Q7 分数码（按构造
    两侧窗相同，J2；类码 ⊆ 两侧窗类集交集 -> J4 下界 / J5 目录贡献）。

    scores_field 布局 [n_fields, heads, 2, field_h, field_w]（float，Q7 码）。
    返回 {x_identities, x_classes, y_identities, y_classes, x_pairs, y_pairs}，
    全部 .detach()。
    """

    wsize = int(plan["wsize"])
    n_oy, n_ox = int(plan["n_oy"]), int(plan["n_ox"])
    field_h, field_w = int(plan["field_h"]), int(plan["field_w"])
    n_fields, heads = scores_field.shape[0], scores_field.shape[1]
    yloc = torch.arange(wsize, device=scores_field.device, dtype=torch.long)
    xloc = torch.arange(wsize, device=scores_field.device, dtype=torch.long)
    x_pairs, y_pairs = [], []
    x_idents, x_classes = [], []
    y_identities, y_classes = [], []
    for wy in range(n_oy):
        y_start, y_end = plan["ys"][wy]
        y_len = y_end - y_start
        for xi in range(n_ox - 1):
            s_lo = plan["xs"][xi + 1][0]
            e_hi = plan["xs"][xi][1]
            x_len = e_hi - s_lo  # 恒 = wsize − stride（尾窗 clamp 亦同）
            ys_full = y_start + yloc[:y_len]
            xs_band = s_lo + xloc[:x_len]
            # 身份码跨 (t, y, x) 三维：坐标广播到 t 维后展平
            ident = (
                torch.arange(2, device=scores_field.device, dtype=torch.long)
                .view(2, 1, 1)
                .expand(2, y_len, x_len)
            )
            y_id = ys_full[:, None].expand(y_len, x_len)
            x_id = xs_band[None, :].expand(y_len, x_len)
            flat = ((ident * field_h + y_id[None]) * field_w + x_id[None]).reshape(-1)
            codes = scores_field[:, :, :, ys_full[:, None], xs_band[None, :]].reshape(
                n_fields, heads, 2 * y_len * x_len
            )
            x_pairs.append((wy, xi))
            x_idents.append(flat)
            x_classes.append(codes)
    for yi in range(n_oy - 1):
        s_lo = plan["ys"][yi + 1][0]
        e_hi = plan["ys"][yi][1]
        y_len = e_hi - s_lo
        for wx in range(n_ox):
            x_start, x_end = plan["xs"][wx]
            x_len = x_end - x_start
            ys_band = s_lo + yloc[:y_len]
            xs_full = x_start + xloc[:x_len]
            ident = (
                torch.arange(2, device=scores_field.device, dtype=torch.long)
                .view(2, 1, 1)
                .expand(2, y_len, x_len)
            )
            y_id = ys_band[:, None].expand(y_len, x_len)
            x_id = xs_full[None, :].expand(y_len, x_len)
            flat = ((ident * field_h + y_id[None]) * field_w + x_id[None]).reshape(-1)
            codes = scores_field[:, :, :, ys_band[:, None], xs_full[None, :]].reshape(
                n_fields, heads, 2 * y_len * x_len
            )
            y_pairs.append((yi, wx))
            y_identities.append(flat)
            y_classes.append(codes)
    return {
        "x_identities": [t.detach() for t in x_idents],
        "x_classes": [t.detach() for t in x_classes],
        "y_identities": [t.detach() for t in y_identities],
        "y_classes": [t.detach() for t in y_classes],
        "x_pairs": x_pairs,
        "y_pairs": y_pairs,
    }


def _binary_motion_sw12_overlap_attention(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """D2 (h89): stride-12/窗口-15 重叠滑窗 Motion 注意力 + 滚动分母。

    返回 (attn, row_sum, gate, sw12_stats)：
      attn / row_sum / gate   [B*, H, 450, D] / [B*, H] / [B*, H, 450]
                              （还原到 Swin dense tile 布局）
      sw12_stats              dict：scores（Q7 码）、rolling_z / z_full
                              （16bit 块分解 int64，逐位相等 = J1）、
                              exp_ledger（J6）、catalog（J5）、window_plan、
                              gate_final / gate_mult（J3 账）。
    """

    if q_orig.ndim != 5 or k_orig.ndim != 4:
        raise ValueError("D2 overlap requires q_orig=[T,B,H,N,D] and k_orig=[B,H,T*N,D]")
    t_steps, batch, heads, spatial_tokens, head_dim = q_orig.shape
    if t_steps != 2:
        raise ValueError("D2 overlap requires the two-slice temporal window (2,15,15)")
    if tuple(k_orig.shape) != (batch, heads, t_steps * spatial_tokens, head_dim):
        raise ValueError("k_orig shape is inconsistent with q_orig temporal/spatial layout")
    wsize = int(cfg.sw12_window_size or 0) or 15
    stride = int(cfg.sw12_stride or 0) or 12
    if wsize <= 0 or stride <= 0 or stride > wsize:
        raise ValueError(f"D2 overlap requires 0 < stride <= wsize; got {stride}/{wsize}")
    if spatial_tokens != wsize * wsize:
        raise ValueError(
            f"D2 contract pins a square {wsize}x{wsize} spatial window; got {spatial_tokens} tokens"
        )
    num_steps = int(cfg.sw12_num_steps or 0)
    if num_steps <= 0 or num_steps % t_steps != 0:
        raise ValueError(
            "D2 requires bsa_attention.sw12_num_steps > 0 and divisible by the "
            f"window T=2; got {num_steps}"
        )
    n_pairs = num_steps // t_steps
    if batch % n_pairs != 0:
        raise ValueError(f"D2 batch {batch} not divisible by n_pairs={n_pairs}")

    # ── 每 token 分数：Motion-XOR 规范融合式（Q7 网格 [0,162]）──
    q_event = _binary_event_ste(_qkformer_token_q(q_orig)).reshape(
        batch, heads, t_steps, spatial_tokens, head_dim
    )
    k_event = _binary_event_ste(k_orig).reshape(batch, heads, t_steps, spatial_tokens, head_dim)
    q_count = q_event.sum(dim=-1)
    k_count = k_event.sum(dim=-1)
    overlap = (q_event * k_event).sum(dim=-1)
    same_zero = head_dim - q_count - k_count + overlap
    motion = (k_event[:, :, 0] - k_event[:, :, 1]).abs().sum(dim=-1)  # [B*, H, N]
    numerator = 64.0 * overlap + same_zero + 16.0 * motion.unsqueeze(2)
    scores = torch.clamp(_rne16_div_pow2_ste(numerator), max=162.0)  # [B*, H, 2, N]
    scores = scores.reshape(batch, heads, t_steps * spatial_tokens)

    # ── 场还原与重叠窗分区（窗口坐标：行序 row=(b·n_pairs+wd)·n_sw+s）──
    batch_actual, n_sw = _d2_decompose_field_batch(batch, n_pairs, cfg)
    if batch_actual * n_pairs * n_sw != batch:
        raise ValueError(
            f"D2 field-batch decomposition inconsistent: "
            f"{batch_actual} × {n_pairs} × {n_sw} != {batch}"
        )
    n_y, n_x = _d2_field_grid(n_sw, cfg)
    plan = _d2_overlap_window_plan(n_y, n_x, wsize, stride, q_orig.device)
    n_ow = int(plan["n_ow"])
    # 行块：field f = (b·n_pairs + wd) 覆盖 rows [f·n_sw, (f+1)·n_sw)，
    # 故 n_fields = batch // n_sw（= B × n_pairs）
    n_fields = batch // n_sw
    sc = scores.view(n_fields, n_sw, heads, t_steps * spatial_tokens).permute(
        0, 2, 1, 3
    )  # [n_fields, heads, n_sw, 450]
    # 窗成员一次 gather 取齐：(row, tok) 两维线性化为一个下标。
    # 注意不能用 sc.gather(2, row).gather(3, tok) 两级 gather——两次 gather
    # 各按自己的位置 k 取索引，等价于 (row[tok[k]], tok[k])，成员错配；
    # 必须展平 (row·450+tok) 单次 gather（与 gate 还原的 win_key 同键）。
    sc_flat = sc.reshape(n_fields, heads, n_sw * (t_steps * spatial_tokens))
    win_key3 = (plan["row_idx"] * (t_steps * spatial_tokens) + plan["tok_idx"]).view(
        1, 1, n_ow, -1
    ).expand(n_fields, heads, n_ow, -1)
    win_valid = plan["valid"].view(1, 1, n_ow, -1).expand(n_fields, heads, n_ow, -1)
    win_scores = sc_flat.gather(
        2, win_key3.reshape(n_fields, heads, -1)
    ).view(n_fields, heads, n_ow, -1)
    win_scores_masked = win_scores.masked_fill(~win_valid, -float("inf"))
    gate_w = shiftmax(win_scores_masked, dim=-1, eps=cfg.eps)
    gate_w = gate_w.masked_fill(~win_valid, 0.0)

    # ── 门还原：g_final = scatter_add(g_w)，gate = g_final / mult（J3）──
    # 线性化 (tile, token) -> 行内展开下标后沿 token 维 scatter_add
    # （index_put_ 的多维 index 语义会把 4D index 广播成 6D，不可用；
    #  scatter_add_ 要求 index 与 self 同秩，故展平 (n_ow, 900) 两维）。
    win_key = win_key3.reshape(n_fields, heads, -1)
    g_final_flat = torch.zeros(
        n_fields, heads, n_sw * t_steps * spatial_tokens,
        dtype=gate_w.dtype, device=q_orig.device,
    )
    g_final_flat.scatter_add_(2, win_key, gate_w.reshape(n_fields, heads, -1))
    g_final = g_final_flat.view(
        n_fields, heads, n_sw, t_steps * spatial_tokens
    )
    mult = plan["mult"].view(1, 1, n_sw, -1).to(dtype=gate_w.dtype)
    gate = g_final / mult
    gate = gate.permute(0, 2, 1, 3).reshape(batch, heads, t_steps * spatial_tokens)
    if cfg.preserve_mean:
        gate = gate * float(t_steps * spatial_tokens)
    gate = _apply_hardware_gate_quant(gate, cfg)
    row_sum = gate.sum(dim=2)
    attn = k_orig.mul(gate.unsqueeze(-1))
    attn = _window_context_broadcast(attn, cfg)

    # ── 滚动分母账（J1：16bit 块分解 int64 逐位精确）──
    # 闭环 z_roll[w] = z_full[0] + Σ_{i<=w} enter[i] − Σ_{i<w} exit[i]
    # （enter[i] = 窗 i 进带项，exit[i] = 窗 i 出带项；行优先链 prev(w)=w−1
    #  由 entry_band/exit_band 的场坐标几何直接给出，与键集差逐位一致）。
    sc_i = scores.to(torch.int64).view(n_fields, n_sw, heads, -1).permute(0, 2, 1, 3)
    win_s = sc_i.reshape(
        n_fields, heads, n_sw * (t_steps * spatial_tokens)
    ).gather(2, win_key3.reshape(n_fields, heads, -1)).view(n_fields, heads, n_ow, -1)
    win_s = win_s.masked_fill(~win_valid, 0)
    chunk, term = _d2_pow2_chunk(win_s)
    term = term.masked_fill(~win_valid, 0)
    z_full = torch.zeros(
        n_fields, heads, n_ow, _D2_N_CHUNKS, dtype=torch.int64, device=q_orig.device
    ).scatter_add_(3, chunk, term)
    entry_band = plan["entry_band"].to(device=q_orig.device)
    exit_band = plan["exit_band"].to(device=q_orig.device)
    enter_t = term * entry_band.view(1, 1, n_ow, -1)
    exit_t = term * exit_band.view(1, 1, n_ow, -1)
    enter_chunks = torch.zeros_like(z_full).scatter_add_(3, chunk, enter_t)
    exit_chunks = torch.zeros_like(z_full).scatter_add_(3, chunk, exit_t)
    cum_enter = enter_chunks.cumsum(dim=2)
    cum_exit = exit_chunks.cumsum(dim=2)
    exit_shifted = torch.cat(
        [torch.zeros_like(cum_exit[:, :, :1, :]), cum_exit[:, :, :-1, :]], dim=2
    )
    z_roll = z_full[:, :, :1, :] + cum_enter - exit_shifted

    # ── exp 流量账（J6）与目录（J5）──
    flow = _d2_exp_flow_ledger(
        int(plan["field_h"]), int(plan["field_w"]), wsize, stride, t_slices=t_steps
    )
    # scores_field 布局 [n_fields, heads, 2, field_h, field_w]（目录用）
    sc_field = sc.reshape(n_fields, heads, n_sw, t_steps, wsize, wsize)
    sc_field = sc_field.permute(0, 1, 4, 2, 5, 3).reshape(
        n_fields, heads, n_y, n_x, wsize, wsize, t_steps
    ).permute(0, 1, 6, 2, 4, 3, 5).reshape(
        n_fields, heads, t_steps, n_y * wsize, n_x * wsize
    )
    catalog = _d2_catalog_bands(plan, sc_field)
    sw12_stats = {
        "scores": scores.detach(),
        "rolling_z": z_roll.detach(),
        "z_full": z_full.detach(),
        "exp_ledger": dict(flow),
        "catalog": catalog,
        "window_plan": {
            "n_y": n_y,
            "n_x": n_x,
            "n_oy": int(plan["n_oy"]),
            "n_ox": int(plan["n_ox"]),
            "n_ow": n_ow,
            "field_h": int(plan["field_h"]),
            "field_w": int(plan["field_w"]),
            "wsize": wsize,
            "stride": stride,
            "ys": list(plan["ys"]),
            "xs": list(plan["xs"]),
            "row_idx": plan["row_idx"].detach(),
            "tok_idx": plan["tok_idx"].detach(),
            "valid": plan["valid"].detach(),
            "mult": plan["mult"].detach(),
        },
        "gate_final": g_final.detach(),
        "gate_mult": mult.detach(),
        "batch_decomposition": (batch_actual, n_pairs, n_sw),
        "window_counts": {"dense": n_sw, "overlap": n_ow},
    }
    return attn, row_sum, gate, sw12_stats


def _castling_aux_weight(module: nn.Module, cfg: ShiftmaxAttentionConfig) -> float:
    """Linearly remove the full-matrix auxiliary before deployment."""

    initial = float(cfg.castling_matrix_aux_weight)
    if not module.training or initial <= 0.0:
        return 0.0
    end_step = int(cfg.castling_matrix_aux_end_step)
    if end_step <= 0:
        return initial
    step = max(0, int(getattr(module, "_h9_global_step", 0)))
    return initial * max(0.0, 1.0 - float(step) / float(end_step))


def _castling_binary_matrix_output(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    """Training-only H66a output used to guide the deployed H60 path."""

    scores = _binary_alpha_xnor_matrix_scores(q_orig, k_orig, cfg)
    if cfg.center_scores:
        scores = scores - scores.mean(dim=-1, keepdim=True)
    gate = shiftmax(scores, dim=-1, eps=cfg.eps)
    value = _ternary_sign_ste(k_orig) if cfg.value_mode in {"sign", "event", "ternary"} else k_orig
    return torch.matmul(gate, value)


def _binary_event_ste(x: torch.Tensor) -> torch.Tensor:
    hard = x.gt(0).to(dtype=x.dtype)
    return (hard - x).detach() + x


def _event_selective_temperature(
    scores: torch.Tensor,
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    """Apply a per-token power-of-two inverse-temperature from Q/K activity.

    For union activity count ``a``, the scale is
    ``2**min(ceil(log2(a + 1)), max_shift)``. The default-disabled branch is
    exactly identity. Deployment needs an OR-popcount, leading-one detector,
    and a bounded left shift; it introduces no learned parameter or second
    attention path.
    """

    if not cfg.event_temperature_enabled:
        return scores
    max_shift = int(cfg.event_temperature_max_shift)
    if max_shift < 0:
        raise ValueError("bsa_attention.event_temperature_max_shift must be nonnegative")
    q_event = _binary_event_ste(_qkformer_token_q(q_orig))
    k_event = _binary_event_ste(k_orig)
    active = ((q_event + k_event) > 0).sum(dim=-1, keepdim=True).to(dtype=scores.dtype)
    shift = torch.ceil(torch.log2(active + 1.0)).clamp_(min=0.0, max=float(max_shift))
    scale = torch.pow(torch.full_like(shift, 2.0), shift).detach()
    return scores * scale


def _window_context_broadcast(
    tokens: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    """Broadcast the per-window mean token with the parameter-free CB rule."""

    if not cfg.context_broadcast_enabled:
        return tokens
    if tokens.ndim != 4:
        raise ValueError("window context broadcast expects [B, heads, tokens, channels]")
    return 0.5 * (tokens + tokens.mean(dim=2, keepdim=True))


def _dualrail_binary_tx_token_scores(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
    beta: float | None = None,
) -> torch.Tensor:
    """TX-style score for binary dual-rail events.

    The first half of each head dimension is interpreted as positive rails and
    the second half as negative rails. This restores same/opposite polarity
    evidence for all-binary ATLIF, whose scalar output is otherwise {0,+1}.
    """

    q_event = _binary_event_ste(_qkformer_token_q(q_orig))
    k_event = _binary_event_ste(k_orig)
    d = q_event.shape[-1]
    if d % 2 != 0:
        raise ValueError("dual-rail binary TX requires an even head_dim")
    half = d // 2
    q_pos, q_neg = q_event[..., :half], q_event[..., half:]
    k_pos, k_neg = k_event[..., :half], k_event[..., half:]

    same_nonzero = (q_pos * k_pos + q_neg * k_neg).sum(dim=-1, keepdim=True)
    opposite = (q_pos * k_neg + q_neg * k_pos).sum(dim=-1, keepdim=True)
    q_active = (q_pos + q_neg).gt(0).to(dtype=q_orig.dtype)
    k_active = (k_pos + k_neg).gt(0).to(dtype=q_orig.dtype)
    same_zero = ((1.0 - q_active) * (1.0 - k_active)).sum(dim=-1, keepdim=True)
    single_active = (q_active * (1.0 - k_active) + (1.0 - q_active) * k_active).sum(dim=-1, keepdim=True)

    _mismatch = beta if beta is not None else float(cfg.mismatch_penalty)
    score = (
        same_nonzero
        + float(cfg.alpha0) * same_zero
        - _mismatch * opposite
        - float(cfg.single_active_penalty) * single_active
    )

    active = None
    if cfg.consensus_score_norm == "active":
        active = (q_active + k_active).sum(dim=-1, keepdim=True).clamp_min(1)
    return _normalize_consensus_score(score, half, cfg, active=active)


def _binary_tx_group_scores(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    """Per-group binary TX scores without producing a K/value carrier."""

    evidence = _direct_tx_channel_evidence(q_orig, k_orig, cfg)
    q_event = _qkformer_token_q(q_orig)
    head_dim = q_event.shape[-1]
    groups = int(cfg.direct_shiftmax_groups)
    if groups <= 0 or head_dim % groups != 0:
        raise ValueError(
            f"direct_shiftmax_groups={groups} must be a positive divisor of head_dim={head_dim}"
        )
    group_dim = head_dim // groups
    grouped = evidence.reshape(*evidence.shape[:-1], groups, group_dim)
    return grouped.mean(dim=-1)


def _direct_tx_channel_evidence(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    q_token = _qkformer_token_q(q_orig)
    if cfg.direct_shiftmax_signed_events:
        q_event = _ternary_sign_ste(q_token)
        k_event = _ternary_sign_ste(k_orig)
        same_active = torch.relu(q_event * k_event)
        q_silent = 1.0 - q_event.abs()
        k_silent = 1.0 - k_event.abs()
    else:
        q_event = _binary_event_ste(q_token)
        k_event = _binary_event_ste(k_orig)
        same_active = q_event * k_event
        q_silent = 1.0 - q_event
        k_silent = 1.0 - k_event
    return same_active + float(cfg.alpha0) * q_silent * k_silent


def _direct_group_shiftmax_output(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = _binary_tx_group_scores(q_orig, k_orig, cfg)
    if cfg.center_scores:
        scores = scores - scores.mean(dim=2, keepdim=True)
    scores = _apply_hardware_score_quant(scores, cfg)
    gate = shiftmax(scores, dim=2, eps=cfg.eps)
    row_sum = gate.sum(dim=2)
    if cfg.preserve_mean:
        gate = gate * float(k_orig.shape[2])
    gate = _apply_hardware_gate_quant(gate, cfg)
    repeat = k_orig.shape[-1] // int(cfg.direct_shiftmax_groups)
    direct_gate = gate - 1.0 if cfg.direct_shiftmax_center_output else gate
    attn = direct_gate.repeat_interleave(repeat, dim=-1)
    return attn, row_sum, gate, scores


def _direct_token_channel_shiftmax_output(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Factorized token/channel TX whose normalized scores are the output."""

    evidence = _direct_tx_channel_evidence(q_orig, k_orig, cfg)
    token_scores = evidence.mean(dim=-1, keepdim=True)
    channel_scores = evidence.mean(dim=2, keepdim=True)
    if cfg.center_scores:
        token_scores = token_scores - token_scores.mean(dim=2, keepdim=True)
        channel_scores = channel_scores - channel_scores.mean(dim=3, keepdim=True)
    token_scores = _apply_hardware_score_quant(token_scores, cfg)
    channel_scores = _apply_hardware_score_quant(channel_scores, cfg)
    token_gate = shiftmax(token_scores, dim=2, eps=cfg.eps)
    channel_gate = shiftmax(channel_scores, dim=3, eps=cfg.eps)
    row_sum = token_gate.sum(dim=2)
    if cfg.preserve_mean:
        token_gate = token_gate * float(k_orig.shape[2])
        channel_gate = channel_gate * float(k_orig.shape[3])
    token_gate = _apply_hardware_gate_quant(token_gate, cfg)
    channel_gate = _apply_hardware_gate_quant(channel_gate, cfg)
    if cfg.direct_shiftmax_center_output:
        token_gate = token_gate - 1.0
        channel_gate = channel_gate - 1.0
    attn = (token_gate + channel_gate) * 0.5
    return attn, row_sum, token_gate, token_scores


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

    Defaults absorb TX(alpha0/mismatch/single) + SC(mu=1/8):
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
        float(cfg.faps_same_nonzero_weight) * same_nonzero.to(dtype=q_event.dtype)
        + float(cfg.faps_same_zero_weight) * same_zero.to(dtype=q_event.dtype)
        - float(cfg.faps_opposite_weight) * opposite.to(dtype=q_event.dtype)
        - float(cfg.faps_single_active_weight) * single_active
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


def _binary_alpha_xnor_stencil_attention(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
    *,
    temporal_pair: bool,
    spatial_cross: bool,
    motion_xor_alpha: float = 0.0,
    profile_module: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Local binary alpha-XNOR attention over a square spatial Swin window.

    Unlike the historical H59 score smoothing, every candidate computes the
    actual similarity between q_i and a selected k_j. Invalid spatial-border
    candidates are masked rather than wrapped to the opposite window edge.

    Optional ``motion_xor_alpha`` adds H67-style temporal K XOR-popcount only to
    the self lane. Adding a constant to every candidate would leave Shiftmax
    invariant, so motion must not be broadcast across all stencil lanes.
    """

    if profile_module is not None:
        profile_module._h9_source_gate_cardinality_proxy = None

    q_event = (_qkformer_token_q(q_orig) > 0).to(dtype=q_orig.dtype)
    k_event = (k_orig > 0).to(dtype=q_orig.dtype)
    batch, heads, n_tokens, head_dim = q_event.shape
    t_steps = int(q_orig.shape[0])
    spatial_tokens = n_tokens // t_steps
    spatial_side = math.isqrt(spatial_tokens)
    height, width = spatial_side, spatial_side
    if t_steps * height * width != n_tokens:
        raise ValueError(
            "binary alpha-XNOR stencil expects a T x H x H square window, "
            f"got T={t_steps}, tokens={n_tokens}"
        )

    grid = torch.arange(n_tokens, device=q_orig.device).reshape(t_steps, height, width)
    neighbor_indices = [grid]
    valid_masks = [torch.ones_like(grid, dtype=torch.bool)]

    if temporal_pair:
        if t_steps != 2:
            raise ValueError("temporal-pair alpha-XNOR currently requires T=2")
        neighbor_indices.append(grid.flip(0))
        valid_masks.append(torch.ones_like(grid, dtype=torch.bool))

    if spatial_cross:
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy = torch.arange(height, device=q_orig.device).view(1, height, 1) + dy
            xx = torch.arange(width, device=q_orig.device).view(1, 1, width) + dx
            valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
            yy = yy.clamp(0, height - 1).expand(t_steps, height, width)
            xx = xx.clamp(0, width - 1).expand(t_steps, height, width)
            tt = torch.arange(t_steps, device=q_orig.device).view(t_steps, 1, 1).expand_as(yy)
            neighbor_indices.append(grid[tt, yy, xx])
            valid_masks.append(valid.expand(t_steps, height, width))

    index = torch.stack(neighbor_indices, dim=-1).reshape(n_tokens, -1)
    valid = torch.stack(valid_masks, dim=-1).reshape(n_tokens, -1)
    k_candidates = k_event[:, :, index, :]
    q_candidates = q_event.unsqueeze(-2)
    same_spike = (q_candidates * k_candidates).sum(dim=-1)
    same_silent = ((1.0 - q_candidates) * (1.0 - k_candidates)).sum(dim=-1)
    scores = same_spike + float(cfg.alpha0) * same_silent
    scores = _normalize_consensus_score(scores, head_dim, cfg, active=None)
    if float(cfg.matrix_diag_bias) != 0.0:
        scores[..., 0] = scores[..., 0] + float(cfg.matrix_diag_bias)
    if float(motion_xor_alpha) != 0.0:
        # Self lane only: same arithmetic as H67 token score bias.
        motion = _binary_temporal_k_xor_popcount(q_orig, k_orig)
        scores = scores.clone()
        scores[..., 0:1] = scores[..., 0:1] + float(motion_xor_alpha) * motion
    # Deploy path: Q7 score grid, then mask invalid candidates to the min code.
    # Mask-after-quant keeps the invalid lane at the lowest representable score
    # under the frozen INT8 grid (same discipline as Match-Code deploy).
    scores = _apply_hardware_score_quant(scores, cfg)
    invalid_fill = (
        float(cfg.hardware_score_min)
        if cfg.hardware_quant_enabled and cfg.hardware_score_min is not None
        else -1.0e4
    )
    scores = scores.masked_fill(~valid.view(1, 1, n_tokens, -1), invalid_fill)
    valid_for_gate = valid.view(1, 1, n_tokens, -1)

    if cfg.hardware_rtl_shiftmax_enabled:
        gate = _rtl_shiftmax_gate_q17(
            scores,
            dim=-1,
            preserve_mean=bool(cfg.preserve_mean),
            valid_mask=(
                valid_for_gate
                if cfg.hardware_mask_invalid_candidates
                else None
            ),
        )
    else:
        gate_scores = (
            scores.masked_fill(~valid_for_gate, -float("inf"))
            if cfg.hardware_mask_invalid_candidates
            else scores
        )
        gate = shiftmax(gate_scores, dim=-1, eps=cfg.eps)
        if cfg.preserve_mean:
            gate = gate * float(index.shape[-1])
        if cfg.hardware_mask_invalid_candidates:
            gate = gate.masked_fill(~valid_for_gate, 0.0)
    gate = _apply_hardware_gate_quant(gate, cfg)
    if (
        profile_module is not None
        and cfg.source_gate_cardinality_regularization_weight > 0.0
    ):
        profile_module._h9_source_gate_cardinality_proxy = (
            _source_gate_cardinality_proxy(
                gate,
                source_index=index,
                valid=valid,
                source_k=k_event,
                mode=cfg.source_gate_cardinality_proxy_mode,
            )
        )
    value_candidates = k_orig[:, :, index, :]
    attn = (gate.unsqueeze(-1) * value_candidates).sum(dim=-2)
    row_sum = gate.sum(dim=-1)
    trace_collector = (
        getattr(profile_module, "_h9_local5_trace_collector", None)
        if profile_module is not None
        else None
    )
    if trace_collector is not None:
        trace_collector(
            module=profile_module,
            q_event=q_event.detach(),
            k_event=k_event.detach(),
            k_orig=k_orig.detach(),
            neighbor_index=index.detach(),
            valid=valid.detach(),
            score_q7=scores.detach(),
            gate=gate.detach(),
        )
    return attn, row_sum, gate


def _source_gate_cardinality_proxy(
    gate: torch.Tensor,
    *,
    source_index: torch.Tensor,
    valid: torch.Tensor,
    source_k: torch.Tensor,
    mode: str = "mean_collapse",
) -> torch.Tensor:
    """Differentiable proxy for source-owned exact gate cardinality.

    The hardware cost is proportional to ``popcount(K_s) * C_s``, where ``C_s``
    is the number of distinct nonzero Q1.7 gate codes that consume source ``s``.
    Counting classes is discrete. ``mean_collapse`` minimizes the weighted
    deviation from one source-local mean. ``tail_gap_c2`` is aligned to a
    two-wide issuer: after sorting the nonzero Q1.7 gates, it penalizes all
    adjacent gaps except the largest, and is exactly zero for at most two gate
    classes. The gate tensor already carries STE gradients from hardware
    quantization; neither mode changes the forward value or prunes a relation.
    """

    if gate.ndim != 4:
        raise ValueError("source gate proxy expects gate [B,H,N,R]")
    batch, heads, tokens, roles = gate.shape
    if source_index.shape != (tokens, roles) or valid.shape != (tokens, roles):
        raise ValueError("source index/valid shape does not match gate")
    if source_k.ndim != 4 or source_k.shape[:3] != (batch, heads, tokens):
        raise ValueError("source K must use [B,H,N,D]")

    edge_source = source_index.reshape(1, 1, -1).expand(batch, heads, -1)
    edge_gate = gate.reshape(batch, heads, -1)
    edge_valid = valid.reshape(1, 1, -1).expand(batch, heads, -1)
    # Gate zero is not a source-owned product term. Keep this membership hard
    # while retaining STE gradients through every nonzero gate value.
    edge_live = edge_valid & edge_gate.detach().gt(0)
    edge_live_f = edge_live.to(dtype=gate.dtype)

    source_weight = source_k.detach().ne(0).sum(dim=-1).to(dtype=gate.dtype)
    source_weight = source_weight / float(max(1, source_k.shape[-1]))
    if mode == "mean_collapse":
        source_sum = gate.new_zeros(batch, heads, tokens)
        source_count = gate.new_zeros(batch, heads, tokens)
        source_sum.scatter_add_(2, edge_source, edge_gate * edge_live_f)
        source_count.scatter_add_(2, edge_source, edge_live_f)
        source_mean = source_sum / source_count.clamp_min(1.0)
        edge_mean = source_mean.gather(2, edge_source)
        edge_weight = source_weight.gather(2, edge_source) * edge_live_f
        numerator = ((edge_gate - edge_mean).abs() * edge_weight).sum()
        denominator = edge_weight.sum().clamp_min(1.0)
        return numerator / denominator
    if mode != "tail_gap_c2":
        raise ValueError(f"unsupported source gate cardinality proxy mode: {mode}")

    source_role_gate = gate.new_zeros(batch, heads, tokens, roles)
    source_role_live = gate.new_zeros(batch, heads, tokens, roles)
    for role in range(roles):
        role_source = source_index[:, role].reshape(1, 1, tokens).expand(
            batch, heads, -1
        )
        role_live = (
            valid[:, role].reshape(1, 1, tokens).expand(batch, heads, -1)
            & gate[..., role].detach().gt(0)
        )
        source_role_gate[..., role].scatter_add_(
            2, role_source, gate[..., role] * role_live
        )
        source_role_live[..., role].scatter_add_(
            2, role_source, role_live.to(gate.dtype)
        )
    source_role_gate = source_role_gate * source_role_live.detach().gt(0).to(gate.dtype)
    if roles < 2:
        return source_role_gate.sum() * 0.0
    sorted_gate = source_role_gate.sort(dim=-1).values
    adjacent_live = sorted_gate[..., :-1].detach().gt(0) & sorted_gate[...,
        1:
    ].detach().gt(0)
    gaps = (sorted_gate[..., 1:] - sorted_gate[..., :-1]) * adjacent_live.to(
        gate.dtype
    )
    tail_gap = gaps.sum(dim=-1) - gaps.amax(dim=-1)
    numerator = (tail_gap * source_weight).sum()
    denominator = source_weight.sum().clamp_min(1.0)
    return numerator / denominator


def regularize_source_gate_cardinality(
    model: nn.Module, raw_config: dict | None
) -> torch.Tensor | None:
    """Collect the optional Local5 gate-cardinality proxy from all blocks."""

    cfg = config_from_dict(raw_config)
    weight = float(cfg.source_gate_cardinality_regularization_weight)
    if weight <= 0.0:
        return None
    if cfg.mode not in {"binary_axnor_local5_shiftmax", "lr_ttx", "h66_lr"}:
        raise ValueError(
            "source gate cardinality regularization is only defined for Local5"
        )
    losses = [
        value
        for module in model.modules()
        if torch.is_tensor(
            value := getattr(module, "_h9_source_gate_cardinality_proxy", None)
        )
    ]
    if not losses:
        raise RuntimeError(
            "gate-cardinality regularization enabled but no Local5 proxy was captured"
        )
    return weight * torch.stack(losses).mean()


def _class_major_shiftmax_gate(
    scores: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Shiftmax over unique Q7 classes, then expand gates to tokens.

    This is not a multiplicity-weighted rewrite of token Shiftmax. Each occupied
    Q7 code casts one vote in the partition function; multiplicity is Class File
    metadata used only when expanding K. That is the C8.3 ISA.
    """

    if scores.ndim != 4 or scores.shape[-1] != 1:
        raise ValueError("class-major Shiftmax expects scores [B,H,N,1]")
    step = float(cfg.hardware_score_step) or (1.0 / 128.0)
    lo = -2.0 if cfg.hardware_score_min is None else float(cfg.hardware_score_min)
    hi = 2.0 if cfg.hardware_score_max is None else float(cfg.hardware_score_max)
    if hi <= lo or step <= 0.0:
        raise ValueError("invalid Q7 class grid")
    n_bins = int(round((hi - lo) / step)) + 1
    squeezed = scores.squeeze(-1)
    batch, heads, tokens = squeezed.shape
    codes = torch.round((squeezed.detach() - lo) / step).to(dtype=torch.long)
    codes = codes.clamp(0, n_bins - 1)
    ones = squeezed.new_ones(batch, heads, tokens)
    multiplicity = squeezed.new_zeros(batch, heads, n_bins)
    member_sum = squeezed.new_zeros(batch, heads, n_bins)
    multiplicity.scatter_add_(-1, codes, ones)
    member_sum.scatter_add_(-1, codes, squeezed)
    occupied = multiplicity > 0
    class_mean = member_sum / multiplicity.clamp_min(1.0)
    centers = lo + step * torch.arange(n_bins, device=scores.device, dtype=scores.dtype)
    class_score = class_mean + (centers - class_mean).detach()
    class_score = class_score.masked_fill(~occupied, -1.0e4)
    gate_c = shiftmax(class_score, dim=-1, eps=cfg.eps)
    gate_c = gate_c * occupied.to(dtype=gate_c.dtype)
    gate = gate_c.gather(-1, codes)
    if cfg.preserve_mean:
        gate = gate * float(tokens)
    n_class = occupied.to(dtype=squeezed.dtype).sum(dim=-1)
    pair_equal = squeezed.new_zeros(())
    if tokens % 2 == 0:
        left = codes.reshape(batch, heads, 2, tokens // 2)[:, :, 0]
        right = codes.reshape(batch, heads, 2, tokens // 2)[:, :, 1]
        pair_equal = left.eq(right).to(dtype=squeezed.dtype).mean()
    stats = {
        "n_occupied_classes": n_class.mean(),
        "multiplicity_mean": multiplicity.masked_select(occupied).mean()
        if bool(occupied.any())
        else squeezed.new_zeros(()),
        "pair_class_equal": pair_equal,
        "codes": codes,
        "multiplicity": multiplicity,
        "gate_c": gate_c,
    }
    if cfg.class_stability_regularization_weight > 0.0:
        spatial = int(math.isqrt(tokens // 2)) if tokens % 2 == 0 else int(math.isqrt(tokens))
        if spatial * spatial * (2 if tokens % 2 == 0 else 1) == tokens:
            t_steps = 2 if tokens % 2 == 0 else 1
            grid = squeezed.reshape(batch, heads, t_steps, spatial, spatial)
            horiz = (grid[..., :, 1:] - grid[..., :, :-1]).abs().mean()
            vert = (grid[..., 1:, :] - grid[..., :-1, :]).abs().mean()
            stats["stability_proxy"] = 0.5 * (horiz + vert)
        else:
            stats["stability_proxy"] = (squeezed[..., 1:] - squeezed[..., :-1]).abs().mean()
    return gate.unsqueeze(-1), stats


def regularize_class_stability(
    model: nn.Module, raw_config: dict | None
) -> torch.Tensor | None:
    """C8.1: penalize spatial score TV so Q7 class membership can stay put."""

    cfg = config_from_dict(raw_config)
    weight = float(cfg.class_stability_regularization_weight)
    if weight <= 0.0:
        return None
    if cfg.mode not in {"h82", "class_major_ttx", "cmt_ttx"}:
        return None
    losses = [
        value
        for module in model.modules()
        if torch.is_tensor(value := getattr(module, "_h9_class_stability_proxy", None))
    ]
    if not losses:
        raise RuntimeError("class-stability regularization enabled but no proxy was captured")
    return weight * torch.stack(losses).mean()


def _q7_class_grid(cfg: ShiftmaxAttentionConfig) -> tuple[float, float, float, int]:
    step = float(cfg.hardware_score_step) or (1.0 / 128.0)
    lo = -2.0 if cfg.hardware_score_min is None else float(cfg.hardware_score_min)
    hi = 2.0 if cfg.hardware_score_max is None else float(cfg.hardware_score_max)
    if hi <= lo or step <= 0.0:
        raise ValueError("invalid Q7 class grid")
    n_bins = int(round((hi - lo) / step)) + 1
    return step, lo, hi, n_bins


def _class_file_from_scores(
    scores: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> dict[str, torch.Tensor]:
    """Occupied Class File: the H83 ISA, not a discarded histogram sidecar."""

    if scores.ndim != 4 or scores.shape[-1] != 1:
        raise ValueError("class file expects scores [B,H,N,1]")
    step, lo, _hi, n_bins = _q7_class_grid(cfg)
    squeezed = scores.squeeze(-1)
    batch, heads, tokens = squeezed.shape
    codes = torch.round((squeezed.detach() - lo) / step).to(dtype=torch.long).clamp(
        0, n_bins - 1
    )
    ones = squeezed.new_ones(batch, heads, tokens)
    multiplicity = squeezed.new_zeros(batch, heads, n_bins)
    member_sum = squeezed.new_zeros(batch, heads, n_bins)
    multiplicity.scatter_add_(-1, codes, ones)
    member_sum.scatter_add_(-1, codes, squeezed)
    occupied = multiplicity > 0
    class_mean = member_sum / multiplicity.clamp_min(1.0)
    centers = lo + step * torch.arange(n_bins, device=scores.device, dtype=scores.dtype)
    class_score = class_mean + (centers - class_mean).detach()
    occupied_score = class_score.masked_fill(~occupied, -1.0e4)
    gate_c = shiftmax(occupied_score, dim=-1, eps=cfg.eps) * occupied.to(dtype=scores.dtype)
    class_id = torch.arange(n_bins, device=scores.device).view(1, 1, n_bins).expand_as(
        occupied
    )
    temporal_pair_mask = occupied.new_zeros(batch, heads, n_bins, 2)
    member_jaccard = squeezed.new_zeros(())
    if tokens % 2 == 0:
        pair = tokens // 2
        spatial = int(math.isqrt(pair))
        codes_t = codes.reshape(batch, heads, 2, pair)
        for time_idx in (0, 1):
            presence = squeezed.new_zeros(batch, heads, n_bins)
            presence.scatter_add_(
                -1,
                codes_t[:, :, time_idx],
                squeezed.new_ones(batch, heads, pair),
            )
            temporal_pair_mask[..., time_idx] = presence > 0
        if spatial * spatial == pair:
            one_hot = torch.nn.functional.one_hot(codes, n_bins).to(dtype=scores.dtype)
            one_hot = one_hot.reshape(batch, heads, 2, pair, n_bins)
            inter = (one_hot[:, :, 0] * one_hot[:, :, 1]).sum(dim=2)
            union = (one_hot[:, :, 0] + one_hot[:, :, 1]).gt(0).sum(dim=2).to(
                dtype=scores.dtype
            )
            both = temporal_pair_mask.all(dim=-1)
            member_jaccard = (inter / union.clamp_min(1.0))[both].mean() if bool(both.any()) else squeezed.new_zeros(())
    return {
        "class_id": class_id,
        "occupied": occupied,
        "multiplicity": multiplicity,
        "class_score": class_score,
        "gate_c": gate_c,
        "codes": codes,
        "temporal_pair_mask": temporal_pair_mask,
        "member_jaccard_t0t1": member_jaccard,
        "n_occupied_classes": occupied.to(dtype=scores.dtype).sum(dim=-1),
    }


def _expand_k_from_class_file(
    k_orig: torch.Tensor,
    class_file: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply class gates to member K. Destinations keep their own K rows.

    This is the Class File expand step. It is not token Shiftmax, and it is not
    class-wise K folding that drops the destination.
    """

    codes = class_file["codes"]
    gate_c = class_file["gate_c"]
    token_gate = gate_c.gather(-1, codes).unsqueeze(-1)
    return k_orig.mul(token_gate), token_gate


def regularize_member_jaccard(
    model: nn.Module, raw_config: dict | None
) -> torch.Tensor | None:
    """C8.1 on H83: 1 - T0/T1 member Jaccard of occupied classes."""

    cfg = config_from_dict(raw_config)
    weight = float(cfg.class_stability_regularization_weight)
    if weight <= 0.0:
        return None
    if cfg.mode not in {"h83", "class_file_isa"}:
        return None
    losses = [
        (1.0 - value).clamp_min(0.0)
        for module in model.modules()
        if torch.is_tensor(value := getattr(module, "_h9_member_jaccard", None))
    ]
    if not losses:
        raise RuntimeError("H83 class-stability enabled but no member Jaccard was captured")
    return weight * torch.stack(losses).mean()


def _soft_hard_membership(
    squeezed: torch.Tensor,
    centers: torch.Tensor,
    step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Hard occupied one-hot with STE through a Gaussian-bin soft assignment."""

    logits = -((squeezed.unsqueeze(-1) - centers) ** 2) / (2.0 * step * step + 1.0e-12)
    soft = torch.softmax(logits, dim=-1)
    hard_idx = soft.argmax(dim=-1)
    hard = torch.nn.functional.one_hot(hard_idx, centers.numel()).to(dtype=squeezed.dtype)
    return hard + (soft - soft.detach()), hard


def _pack_occupied_class_file(
    member: torch.Tensor,
    hard: torch.Tensor,
    squeezed: torch.Tensor,
    centers: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> dict[str, torch.Tensor]:
    """Packed occupied records. Expand may only use this dict."""

    batch, heads, tokens, n_bins = member.shape
    occupied = hard.sum(dim=2) > 0
    counts = occupied.to(dtype=torch.long).sum(dim=-1)
    packed_len = int(counts.max().item()) if counts.numel() else 0
    if packed_len <= 0:
        packed_len = 1
    class_id = squeezed.new_zeros(batch, heads, packed_len, dtype=torch.long)
    valid = squeezed.new_zeros(batch, heads, packed_len, dtype=torch.bool)
    member_mask = squeezed.new_zeros(batch, heads, packed_len, tokens)
    class_score = squeezed.new_full((batch, heads, packed_len), -1.0e4)
    for b in range(batch):
        for h in range(heads):
            ids = occupied[b, h].nonzero(as_tuple=False).flatten()
            n_live = int(ids.numel())
            if n_live == 0:
                continue
            valid[b, h, :n_live] = True
            class_id[b, h, :n_live] = ids
            member_mask[b, h, :n_live] = member[b, h, :, ids].transpose(0, 1)
            class_score[b, h, :n_live] = (
                member[b, h, :, ids] * squeezed[b, h].unsqueeze(-1)
            ).sum(0) / member[b, h, :, ids].sum(0).clamp_min(1.0)
    gate_c = shiftmax(class_score.masked_fill(~valid, -1.0e4), dim=-1, eps=cfg.eps)
    gate_c = gate_c * valid.to(dtype=gate_c.dtype)
    multiplicity = member_mask.sum(dim=-1)
    return {
        "class_id": class_id,
        "valid": valid,
        "member_mask": member_mask,
        "class_score": class_score,
        "gate_c": gate_c,
        "multiplicity": multiplicity,
        "n_occupied_classes": counts.to(dtype=squeezed.dtype),
    }


def _expand_k_from_packed_class_file(
    k_orig: torch.Tensor,
    class_file: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply packed class gates through member_mask. No T450 codes.gather."""

    if "codes" in class_file:
        raise RuntimeError("H84 expand forbids a codes tensor in the Class File")
    member_mask = class_file["member_mask"]
    gate_c = class_file["gate_c"]
    token_gate = (member_mask * gate_c.unsqueeze(-1)).sum(dim=2).unsqueeze(-1)
    return k_orig.mul(token_gate), token_gate


def _adjacent_row_class_jaccard(member_mask: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """C8.1: Jaccard of occupied class sets on adjacent spatial rows."""

    batch, heads, packed_len, tokens = member_mask.shape
    if tokens % 2 != 0:
        return member_mask.new_zeros(())
    spatial = int(math.isqrt(tokens // 2))
    if spatial * spatial * 2 != tokens or spatial < 2:
        return member_mask.new_zeros(())
    grid = member_mask.reshape(batch, heads, packed_len, 2, spatial, spatial)
    row_set = grid.sum(dim=-1).clamp(0.0, 1.0)
    a = row_set[..., :, :-1]
    b = row_set[..., :, 1:]
    inter = (a * b).sum(dim=2)
    union = (a + b - a * b).sum(dim=2)
    live = valid.to(dtype=member_mask.dtype).unsqueeze(-1).unsqueeze(-1).expand_as(a)
    weight = live.sum(dim=2).clamp_min(1.0)
    value = (inter / union.clamp_min(1.0)) * (weight > 0).to(dtype=member_mask.dtype)
    return value.sum() / weight.gt(0).to(dtype=member_mask.dtype).sum().clamp_min(1.0)


def regularize_row_jaccard(
    model: nn.Module, raw_config: dict | None
) -> torch.Tensor | None:
    cfg = config_from_dict(raw_config)
    weight = float(cfg.class_stability_regularization_weight)
    if weight <= 0.0 or cfg.mode not in {"h84", "packed_class_file"}:
        return None
    losses = [
        (1.0 - value).clamp_min(0.0)
        for module in model.modules()
        if torch.is_tensor(value := getattr(module, "_h9_row_jaccard", None))
    ]
    if not losses:
        raise RuntimeError("H84 row-Jaccard regularization enabled but no proxy was captured")
    return weight * torch.stack(losses).mean()


def _h85_window_hw(tokens: int) -> tuple[int, int]:
    if tokens % 2 != 0:
        raise ValueError("H85 requires T=2 tokens")
    spatial = int(math.isqrt(tokens // 2))
    if spatial * spatial * 2 != tokens:
        raise ValueError("H85 requires a T x S x S window")
    return 2, spatial


def _build_h85_row_files(
    scores: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> dict[str, torch.Tensor]:
    """Per spatial-row packed Class File plus adjacent-row deltas."""

    if scores.ndim != 4 or scores.shape[-1] != 1:
        raise ValueError("H85 expects scores [B,H,N,1]")
    step, lo, _hi, n_bins = _q7_class_grid(cfg)
    batch, heads, tokens, _ = scores.shape
    t_steps, spatial = _h85_window_hw(tokens)
    grid = scores.squeeze(-1).reshape(batch, heads, t_steps, spatial, spatial)
    centers = lo + step * torch.arange(n_bins, device=scores.device, dtype=scores.dtype)
    max_class = spatial
    class_id = scores.new_zeros(batch, heads, t_steps, spatial, max_class, dtype=torch.long)
    valid = scores.new_zeros(batch, heads, t_steps, spatial, max_class, dtype=torch.bool)
    member_idx = scores.new_zeros(batch, heads, t_steps, spatial, max_class, spatial, dtype=torch.long)
    member_ok = scores.new_zeros(batch, heads, t_steps, spatial, max_class, spatial, dtype=torch.bool)
    class_score = scores.new_full((batch, heads, t_steps, spatial, max_class), -1.0e4)
    multiplicity = scores.new_zeros(batch, heads, t_steps, spatial, max_class)
    n_class = scores.new_zeros(batch, heads, t_steps, spatial, dtype=torch.long)
    for time_idx in range(t_steps):
        for row in range(spatial):
            row_scores = grid[:, :, time_idx, row, :]
            member, hard = _soft_hard_membership(row_scores, centers, step)
            occupied = hard.sum(dim=2) > 0
            for b in range(batch):
                for h in range(heads):
                    ids = occupied[b, h].nonzero(as_tuple=False).flatten()
                    n_live = int(ids.numel())
                    if n_live == 0:
                        continue
                    n_class[b, h, time_idx, row] = n_live
                    valid[b, h, time_idx, row, :n_live] = True
                    class_id[b, h, time_idx, row, :n_live] = ids
                    for c_i in range(n_live):
                        hit = hard[b, h, :, ids[c_i]] > 0
                        idx = hit.nonzero(as_tuple=False).flatten()
                        member_idx[b, h, time_idx, row, c_i, : idx.numel()] = idx
                        member_ok[b, h, time_idx, row, c_i, : idx.numel()] = True
                        multiplicity[b, h, time_idx, row, c_i] = float(idx.numel())
                    soft_mass = member[b, h, :, ids]
                    class_score[b, h, time_idx, row, :n_live] = (
                        (soft_mass * row_scores[b, h].unsqueeze(-1)).sum(0)
                        / soft_mass.sum(0).clamp_min(1.0)
                    )
    gate_c = shiftmax(class_score.masked_fill(~valid, -1.0e4), dim=-1, eps=cfg.eps)
    gate_c = gate_c * valid.to(dtype=gate_c.dtype)
    prev = torch.nn.functional.one_hot(class_id[:, :, :, :-1].clamp_min(0), n_bins)
    prev = prev * valid[:, :, :, :-1].unsqueeze(-1)
    curr = torch.nn.functional.one_hot(class_id[:, :, :, 1:].clamp_min(0), n_bins)
    curr = curr * valid[:, :, :, 1:].unsqueeze(-1)
    prev_set = prev.any(dim=4)
    curr_set = curr.any(dim=4)
    shared = prev_set & curr_set
    insert = curr_set & ~prev_set
    delete = prev_set & ~curr_set
    same_set = shared.any(dim=-1) & ~insert.any(dim=-1) & ~delete.any(dim=-1)
    return {
        "class_id": class_id,
        "valid": valid,
        "member_idx": member_idx,
        "member_ok": member_ok,
        "gate_c": gate_c,
        "multiplicity": multiplicity,
        "n_class": n_class,
        "shared_ids": shared,
        "insert_ids": insert,
        "delete_ids": delete,
        "reuse_set": same_set,
        "n_bins": scores.new_tensor(n_bins, dtype=torch.long),
    }


def _expand_k_from_h85_row_files(
    k_orig: torch.Tensor,
    class_file: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Class-major scatter from per-row member ids. No T450 token gate."""

    forbidden = ("codes", "token_gate", "member_mask")
    if any(key in class_file for key in forbidden):
        raise RuntimeError("H85 expand forbids T450 addressing tensors")
    if k_orig.ndim != 4:
        raise ValueError("H85 expand expects k [B,H,N,D]")
    batch, heads, tokens, dim = k_orig.shape
    t_steps, spatial = _h85_window_hw(tokens)
    k_grid = k_orig.reshape(batch, heads, t_steps, spatial, spatial, dim)
    attn = k_orig.new_zeros(k_grid.shape)
    member_idx = class_file["member_idx"]
    member_ok = class_file["member_ok"]
    gate_c = class_file["gate_c"]
    class_id = class_file["class_id"]
    valid = class_file["valid"]
    shared_ids = class_file["shared_ids"]
    insert_ids = class_file["insert_ids"]
    n_bins = int(class_file["n_bins"].item())
    for time_idx in range(t_steps):
        for row in range(spatial):
            idx = member_idx[:, :, time_idx, row]
            ok = member_ok[:, :, time_idx, row].to(dtype=k_orig.dtype)
            live = valid[:, :, time_idx, row].to(dtype=k_orig.dtype)
            codes = class_id[:, :, time_idx, row].clamp(0, n_bins - 1)
            if row == 0:
                apply = live
            else:
                shared = shared_ids[:, :, time_idx, row - 1].gather(-1, codes)
                inserted = insert_ids[:, :, time_idx, row - 1].gather(-1, codes)
                apply = live * (shared | inserted).to(dtype=k_orig.dtype)
            gather_idx = idx.clamp(0, spatial - 1).unsqueeze(-1).expand(
                batch, heads, idx.shape[2], idx.shape[3], dim
            )
            k_row = k_grid[:, :, time_idx, row]
            k_mem = k_row.unsqueeze(2).expand(batch, heads, idx.shape[2], spatial, dim)
            k_sel = k_mem.gather(3, gather_idx)
            scale = (gate_c[:, :, time_idx, row] * apply).unsqueeze(-1).unsqueeze(-1)
            contrib = k_sel * scale * ok.unsqueeze(-1)
            attn[:, :, time_idx, row].scatter_add_(
                2,
                gather_idx.reshape(batch, heads, -1, dim),
                contrib.reshape(batch, heads, -1, dim),
            )
    return attn.reshape(batch, heads, tokens, dim)


def regularize_h85_delta(
    model: nn.Module, raw_config: dict | None
) -> torch.Tensor | None:
    cfg = config_from_dict(raw_config)
    weight = float(cfg.class_stability_regularization_weight)
    if weight <= 0.0 or cfg.mode not in {"h85", "row_delta_class_file"}:
        return None
    losses = [
        (1.0 - value).clamp_min(0.0)
        for module in model.modules()
        if torch.is_tensor(value := getattr(module, "_h9_reuse_set_rate", None))
    ]
    if not losses:
        raise RuntimeError("H85 delta regularization enabled but no reuse_set rate was captured")
    return weight * torch.stack(losses).mean()


_H86_FORBIDDEN_KEYS = (
    "codes",
    "token_gate",
    "member_mask",
    "member_idx",
    "shared_ids",
    "insert_ids",
    "delete_ids",
    "occupied",
    "n_bins",
    "reuse_set",
)
_H86_FORBIDDEN_LAST_DIMS = {513}


def _h86_assert_expand_operands(class_file: dict[str, torch.Tensor]) -> None:
    """H86 expand may not see T450 addressing tensors or a 513-bin occupancy hist."""

    for key in _H86_FORBIDDEN_KEYS:
        if key in class_file:
            raise RuntimeError(f"H86 expand forbids operand {key!r}")
    for key, value in class_file.items():
        if not torch.is_tensor(value) or value.ndim == 0:
            continue
        if int(value.shape[-1]) in _H86_FORBIDDEN_LAST_DIMS:
            raise RuntimeError(
                f"H86 expand forbids 513-bin occupancy tensor {key!r} shape={tuple(value.shape)}"
            )


def _h86_pack_true_ids(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack True columns on the last axis into (idx, ok) of the same width."""

    idx = mask.to(dtype=torch.float32).argsort(dim=-1, descending=True)
    ok = mask.gather(-1, idx) > 0
    return idx, ok


def _h86_ids_to_col_mask(
    idx: torch.Tensor,
    ok: torch.Tensor,
    spatial: int,
) -> torch.Tensor:
    mask = idx.new_zeros(*idx.shape[:-1], spatial, dtype=torch.float32)
    mask = mask.scatter_add(-1, idx.clamp(0, spatial - 1), ok.to(dtype=mask.dtype))
    return mask.clamp(0.0, 1.0)


def _pack_window_class_major(
    scores: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> dict[str, torch.Tensor]:
    """H82 window class-major, packed to occupied records. No token expand."""

    if scores.ndim != 4 or scores.shape[-1] != 1:
        raise ValueError("H86 expects scores [B,H,N,1]")
    step, lo, _hi, n_bins = _q7_class_grid(cfg)
    squeezed = scores.squeeze(-1)
    batch, heads, tokens = squeezed.shape
    codes = torch.round((squeezed.detach() - lo) / step).to(dtype=torch.long).clamp(
        0, n_bins - 1
    )
    ones = squeezed.new_ones(batch, heads, tokens)
    multiplicity = squeezed.new_zeros(batch, heads, n_bins)
    member_sum = squeezed.new_zeros(batch, heads, n_bins)
    multiplicity.scatter_add_(-1, codes, ones)
    member_sum.scatter_add_(-1, codes, squeezed)
    occupied = multiplicity > 0
    class_mean = member_sum / multiplicity.clamp_min(1.0)
    centers = lo + step * torch.arange(n_bins, device=scores.device, dtype=scores.dtype)
    class_score = class_mean + (centers - class_mean).detach()
    gate_full = shiftmax(class_score.masked_fill(~occupied, -1.0e4), dim=-1, eps=cfg.eps)
    gate_full = gate_full * occupied.to(dtype=gate_full.dtype)
    counts = occupied.to(dtype=torch.long).sum(dim=-1)
    packed_len = int(counts.max().item()) if counts.numel() else 1
    packed_len = max(packed_len, 1)
    class_id = codes.new_zeros(batch, heads, packed_len)
    valid = occupied.new_zeros(batch, heads, packed_len)
    gate_c = squeezed.new_zeros(batch, heads, packed_len)
    for batch_idx in range(batch):
        for head_idx in range(heads):
            ids = occupied[batch_idx, head_idx].nonzero(as_tuple=False).flatten()
            n_live = int(ids.numel())
            if n_live == 0:
                continue
            valid[batch_idx, head_idx, :n_live] = True
            class_id[batch_idx, head_idx, :n_live] = ids
            gate_c[batch_idx, head_idx, :n_live] = gate_full[batch_idx, head_idx, ids]
    return {
        "class_id": class_id,
        "valid": valid,
        "gate_c": gate_c,
        "codes": codes,
        "centers": centers,
        "n_occupied_classes": counts.to(dtype=squeezed.dtype),
    }


def _build_h86_member_delta_file(
    scores: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> dict[str, torch.Tensor]:
    """Window class-major file whose row>0 members exist only as insert/delete."""

    packed = _pack_window_class_major(scores, cfg)
    batch, heads, tokens, _ = scores.shape
    t_steps, spatial = _h85_window_hw(tokens)
    class_id = packed["class_id"]
    valid = packed["valid"]
    packed_len = class_id.shape[-1]
    codes_grid = packed["codes"].reshape(batch, heads, t_steps, spatial, spatial)
    col_of_class = codes_grid.unsqueeze(-1).eq(class_id[:, :, None, None, None, :])
    col_of_class = col_of_class & valid[:, :, None, None, None, :]
    col_of_class = col_of_class.permute(0, 1, 2, 3, 5, 4).to(dtype=scores.dtype)
    row0_idx, row0_ok = _h86_pack_true_ids(col_of_class[:, :, :, 0])
    prev = col_of_class[:, :, :, :-1]
    curr = col_of_class[:, :, :, 1:]
    insert_mask = curr * (1.0 - prev)
    delete_mask = prev * (1.0 - curr)
    insert_idx, insert_ok = _h86_pack_true_ids(insert_mask)
    delete_idx, delete_ok = _h86_pack_true_ids(delete_mask)
    prev_live = prev.sum(dim=-1) > 0
    curr_live = curr.sum(dim=-1) > 0
    class_shared = prev_live & curr_live
    class_insert = (~prev_live) & curr_live
    class_delete = prev_live & (~curr_live)
    centers = packed["centers"]
    member_ste, _hard = _soft_hard_membership(scores.squeeze(-1), centers, float(cfg.hardware_score_step) or (1.0 / 128.0))
    ste_grid = member_ste.reshape(batch, heads, t_steps, spatial, spatial, centers.numel())
    ste_idx = class_id[:, :, None, None, None, :].expand(
        batch, heads, t_steps, spatial, spatial, packed_len
    ).clamp(0, centers.numel() - 1)
    ste_col = ste_grid.gather(-1, ste_idx) * valid[:, :, None, None, None, :].to(dtype=scores.dtype)
    ste_col = ste_col.permute(0, 1, 2, 3, 5, 4)
    prev_s = ste_col[:, :, :, :-1]
    curr_s = ste_col[:, :, :, 1:]
    inter = (prev_s * curr_s).sum(dim=-1)
    union = (prev_s + curr_s - prev_s * curr_s).sum(dim=-1)
    surviving = class_shared.to(dtype=scores.dtype)
    pair_weight = surviving.sum()
    if float(pair_weight.detach()) <= 0.0:
        member_jaccard = scores.new_ones(())
    else:
        member_jaccard = (inter / union.clamp_min(1.0) * surviving).sum() / pair_weight.clamp_min(1.0)
    return {
        "class_id": class_id,
        "valid": valid,
        "gate_c": packed["gate_c"],
        "row0_member_idx": row0_idx,
        "row0_member_ok": row0_ok,
        "member_insert": insert_idx,
        "member_insert_ok": insert_ok,
        "member_delete": delete_idx,
        "member_delete_ok": delete_ok,
        "class_shared": class_shared,
        "class_insert": class_insert,
        "class_delete": class_delete,
        "member_jaccard_surviving": member_jaccard,
        "n_occupied_classes": packed["n_occupied_classes"],
    }


def _expand_k_from_h86_member_delta(
    k_orig: torch.Tensor,
    class_file: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Rebuild row members as prev + insert − delete, then apply window gate_c."""

    _h86_assert_expand_operands(class_file)
    if k_orig.ndim != 4:
        raise ValueError("H86 expand expects k [B,H,N,D]")
    batch, heads, tokens, dim = k_orig.shape
    t_steps, spatial = _h85_window_hw(tokens)
    k_grid = k_orig.reshape(batch, heads, t_steps, spatial, spatial, dim)
    attn = k_orig.new_zeros(k_grid.shape)
    valid = class_file["valid"].to(dtype=k_orig.dtype)
    gate_c = class_file["gate_c"]
    members = _h86_ids_to_col_mask(
        class_file["row0_member_idx"],
        class_file["row0_member_ok"],
        spatial,
    )
    insert_idx = class_file["member_insert"]
    insert_ok = class_file["member_insert_ok"]
    delete_idx = class_file["member_delete"]
    delete_ok = class_file["member_delete_ok"]
    class_shared = class_file["class_shared"].to(dtype=k_orig.dtype)
    class_insert = class_file["class_insert"].to(dtype=k_orig.dtype)
    class_delete = class_file["class_delete"].to(dtype=k_orig.dtype)
    for row in range(spatial):
        if row > 0:
            pair = row - 1
            inserted = _h86_ids_to_col_mask(
                insert_idx[:, :, :, pair],
                insert_ok[:, :, :, pair],
                spatial,
            )
            deleted = _h86_ids_to_col_mask(
                delete_idx[:, :, :, pair],
                delete_ok[:, :, :, pair],
                spatial,
            )
            rebuilt = (members + inserted - deleted).clamp(0.0, 1.0)
            shared = class_shared[:, :, :, pair].unsqueeze(-1)
            allocated = class_insert[:, :, :, pair].unsqueeze(-1)
            dropped = class_delete[:, :, :, pair].unsqueeze(-1)
            members = rebuilt * shared + inserted * allocated + members.new_zeros(()) * dropped
        scale = gate_c[:, :, None, :, None] * members * valid[:, :, None, :, None]
        k_row = k_grid[:, :, :, row]
        attn[:, :, :, row] = (k_row.unsqueeze(3) * scale.unsqueeze(-1)).sum(dim=3)
    return attn.reshape(batch, heads, tokens, dim)


def regularize_h86_member_tv(
    model: nn.Module, raw_config: dict | None
) -> torch.Tensor | None:
    cfg = config_from_dict(raw_config)
    weight = float(cfg.class_stability_regularization_weight)
    if weight <= 0.0 or cfg.mode not in {"h86", "member_delta_class_file"}:
        return None
    losses = [
        (1.0 - value).clamp_min(0.0)
        for module in model.modules()
        if torch.is_tensor(value := getattr(module, "_h9_member_jaccard_surviving", None))
    ]
    if not losses:
        raise RuntimeError("H86 member-TV regularization enabled but no surviving Jaccard was captured")
    return weight * torch.stack(losses).mean()


_EEMFLOW_MC49_CHANNELS = (
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 22, 23, 25, 27, 29, 30,
    31, 32, 33, 35, 37, 38, 39, 40, 41, 42, 43, 45, 47, 48, 49, 50, 51,
    53, 55, 57, 58, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79,
)
MC49_OFFSETS = tuple((index // 9 - 4, index % 9 - 4) for index in _EEMFLOW_MC49_CHANNELS)
DE9_OFFSETS = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
AX17_OFFSETS = tuple((0, dx) for dx in range(-4, 5)) + tuple(
    (dy, 0) for dy in range(-4, 5) if dy != 0
)
PC9_PATCH_WEIGHTS = tuple(
    4 if (dy, dx) == (0, 0) else 2 if dy == 0 or dx == 0 else 1
    for dy, dx in DE9_OFFSETS
)
G4_MATCH_GROUPS = 4
G4_MATCH_GROUP_DIM = 8
BASE_MATCH_CODE_MODES = {
    "binary_de9_match_code", "de9_match_code",
    "binary_mc49_match_code", "mc49_match_code",
    "binary_ax17_match_code", "ax17_match_code",
}
PC9_MATCH_CODE_MODES = {
    "binary_pc9_patch_match_code", "pc9_patch_match_code", "h76_pc9",
}
LC4_MATCH_CODE_MODES = {
    "binary_lc4_match_code", "lc4_match_code", "h77_lc4",
}
G4_MATCH_CODE_MODES = {
    "binary_g4_match_code", "g4_match_code", "h78_g4",
}
CF10_MATCH_CODE_MODES = {
    "binary_cf10_match_code", "cf10_match_code", "h79_cf10",
}
DN9_MATCH_CODE_MODES = {
    "binary_dn9_match_code", "dn9_match_code", "h80_dn9",
}
MATCH_CODE_MODES = (
    BASE_MATCH_CODE_MODES
    | PC9_MATCH_CODE_MODES
    | LC4_MATCH_CODE_MODES
    | G4_MATCH_CODE_MODES
    | CF10_MATCH_CODE_MODES
    | DN9_MATCH_CODE_MODES
)


def _binary_event_ste(value: torch.Tensor) -> torch.Tensor:
    """Binary forward with a bounded identity surrogate for Match-Code Q/K."""

    hard = value.gt(0).to(dtype=value.dtype)
    proxy = value.clamp(min=0.0, max=1.0)
    return hard + proxy - proxy.detach()


def _cross_time_match_events(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    offsets: tuple[tuple[int, int], ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gather binary Q and opposite-time K events for fixed offsets."""

    q_event = _binary_event_ste(_qkformer_token_q(q_orig))
    k_event = _binary_event_ste(k_orig)
    batch, heads, n_tokens, _ = q_event.shape
    t_steps, height, width = int(q_orig.shape[0]), 9, 9
    if t_steps != 2 or t_steps * height * width != n_tokens:
        raise ValueError(
            "cross-time Match-Code expects a 2x9x9 window, "
            f"got T={t_steps}, tokens={n_tokens}"
        )

    grid = torch.arange(n_tokens, device=q_orig.device).reshape(t_steps, height, width)
    indices = []
    masks = []
    tt = (1 - torch.arange(t_steps, device=q_orig.device)).view(t_steps, 1, 1)
    for dy, dx in offsets:
        yy = torch.arange(height, device=q_orig.device).view(1, height, 1) + int(dy)
        xx = torch.arange(width, device=q_orig.device).view(1, 1, width) + int(dx)
        valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
        yy = yy.clamp(0, height - 1).expand(t_steps, height, width)
        xx = xx.clamp(0, width - 1).expand(t_steps, height, width)
        indices.append(grid[tt.expand_as(yy), yy, xx])
        masks.append(valid.expand(t_steps, height, width))

    index = torch.stack(indices, dim=-1).reshape(n_tokens, len(offsets))
    valid = torch.stack(masks, dim=-1).reshape(n_tokens, len(offsets))
    k_candidates = k_event[:, :, index, :]
    q_candidates = q_event.unsqueeze(-2)
    valid = valid.view(1, 1, n_tokens, len(offsets)).expand(batch, heads, -1, -1)
    return q_candidates, k_candidates, valid


def _cross_time_match_counts(
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    offsets: tuple[tuple[int, int], ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return n11, n00 and validity for fixed opposite-time offsets."""

    q_candidates, k_candidates, valid = _cross_time_match_events(q_orig, k_orig, offsets)
    n11 = (q_candidates * k_candidates).sum(dim=-1)
    n00 = ((1.0 - q_candidates) * (1.0 - k_candidates)).sum(dim=-1)
    return n11, n00, valid


def _quantized_match_code_weight(module: nn.Module, cfg: ShiftmaxAttentionConfig) -> torch.Tensor:
    weight = module._h9_match_code_weight
    if not cfg.match_code_weight_quant_enabled:
        return weight
    clipped = weight.clamp(min=cfg.match_code_weight_min, max=cfg.match_code_weight_max)
    return _quantize_ste(clipped, cfg.match_code_weight_step)


def _quantized_lc4_coefficients(module: nn.Module, cfg: ShiftmaxAttentionConfig) -> torch.Tensor:
    coefficients = module._h9_lc4_coefficients
    if not cfg.lc4_coefficient_quant_enabled:
        return coefficients
    clipped = coefficients.clamp(
        min=cfg.lc4_coefficient_min,
        max=cfg.lc4_coefficient_max,
    )
    return _quantize_ste(clipped, cfg.lc4_coefficient_step)


def _quantized_cf10_beta(module: nn.Module, cfg: ShiftmaxAttentionConfig) -> torch.Tensor:
    """Return the per-head CF10 margin/activity coefficients on a dyadic grid."""

    beta = module._h9_cf10_beta.clamp(min=cfg.cf10_beta_min, max=cfg.cf10_beta_max)
    return _quantize_ste(beta, cfg.cf10_beta_step)


def _effective_cf10_match_code_weight(
    module: nn.Module,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    """Append the hard-wired zero null row to CF10's nine stored codewords."""

    weight = _quantized_match_code_weight(module, cfg)
    zero = weight.new_zeros(weight.shape[0], 1, weight.shape[2])
    return torch.cat((weight, zero), dim=1)


def _project_match_descriptor(
    module: nn.Module,
    descriptor: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    """Project a fixed descriptor through the static per-head codebook."""

    weight = _quantized_match_code_weight(module, cfg).to(dtype=descriptor.dtype)
    return torch.einsum("bhnr,hrd->bhnd", descriptor, weight)


def _match_code_attention(
    module: nn.Module,
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
    *,
    dual_evidence: bool,
    offsets: tuple[tuple[int, int], ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Cross-time displacement descriptor projected without a K/V carrier."""

    offsets = offsets or (DE9_OFFSETS if dual_evidence else MC49_OFFSETS)
    n11, n00, valid = _cross_time_match_counts(q_orig, k_orig, offsets)
    head_dim = int(q_orig.shape[-1])
    mask_value = torch.finfo(n11.dtype).min
    if dual_evidence:
        active_scores = _apply_hardware_score_quant(n11 / float(head_dim), cfg)
        silent_scores = _apply_hardware_score_quant(n00 / float(head_dim), cfg)
        active_scores = active_scores.masked_fill(~valid, mask_value)
        silent_scores = silent_scores.masked_fill(~valid, mask_value)
        active_gate = _apply_hardware_gate_quant(shiftmax(active_scores, dim=-1, eps=cfg.eps), cfg)
        silent_gate = _apply_hardware_gate_quant(shiftmax(silent_scores, dim=-1, eps=cfg.eps), cfg)
        descriptor = torch.cat((active_gate, silent_gate), dim=-1)
        scores = torch.cat((active_scores, silent_scores), dim=-1)
        row_sum = active_gate.sum(dim=-1) + silent_gate.sum(dim=-1)
    else:
        scores = _apply_hardware_score_quant(
            (n11 + float(cfg.alpha0) * n00) / float(head_dim), cfg
        ).masked_fill(~valid, mask_value)
        descriptor = _apply_hardware_gate_quant(shiftmax(scores, dim=-1, eps=cfg.eps), cfg)
        row_sum = descriptor.sum(dim=-1)

    attn = _project_match_descriptor(module, descriptor, cfg)
    return attn, row_sum, descriptor, scores


def _same_time_patch_index(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed 3x3 same-time indices and exact in-window validity."""

    t_steps, height, width = 2, 9, 9
    n_tokens = t_steps * height * width
    grid = torch.arange(n_tokens, device=device).reshape(t_steps, height, width)
    indices = []
    masks = []
    for dy, dx in DE9_OFFSETS:
        yy = torch.arange(height, device=device).view(1, height, 1) + int(dy)
        xx = torch.arange(width, device=device).view(1, 1, width) + int(dx)
        valid = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
        yy = yy.clamp(0, height - 1).expand(t_steps, height, width)
        xx = xx.clamp(0, width - 1).expand(t_steps, height, width)
        tt = torch.arange(t_steps, device=device).view(t_steps, 1, 1).expand_as(yy)
        indices.append(grid[tt, yy, xx])
        masks.append(valid.expand(t_steps, height, width))
    return (
        torch.stack(indices, dim=-1).reshape(n_tokens, len(DE9_OFFSETS)),
        torch.stack(masks, dim=-1).reshape(n_tokens, len(DE9_OFFSETS)),
    )


def _pc9_patch_match_code_attention(
    module: nn.Module,
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """H76: fixed dyadic patch consistency over nine Match-Code planes."""

    n11, n00, valid = _cross_time_match_counts(q_orig, k_orig, DE9_OFFSETS)
    head_dim = int(q_orig.shape[-1])
    base_scores = (n11 + float(cfg.alpha0) * n00) / float(head_dim)
    patch_index, patch_valid = _same_time_patch_index(q_orig.device)
    gathered_scores = base_scores[:, :, patch_index, :]
    gathered_valid = valid[:, :, patch_index, :]
    support = gathered_valid & patch_valid.view(1, 1, 162, 9, 1)
    weights = base_scores.new_tensor(PC9_PATCH_WEIGHTS).view(1, 1, 1, 9, 1)
    weighted_support = weights * support.to(dtype=base_scores.dtype)
    normalization = weighted_support.sum(dim=-2).clamp_min(1.0)
    scores = (gathered_scores * weighted_support).sum(dim=-2) / normalization
    scores = _apply_hardware_score_quant(scores, cfg)
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    descriptor = _apply_hardware_gate_quant(shiftmax(scores, dim=-1, eps=cfg.eps), cfg)
    row_sum = descriptor.sum(dim=-1)
    attn = _project_match_descriptor(module, descriptor, cfg)
    return attn, row_sum, descriptor, scores


def _lc4_match_code_attention(
    module: nn.Module,
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """H77: dyadic learned cost over all four binary contingencies."""

    q_candidates, k_candidates, valid = _cross_time_match_events(q_orig, k_orig, DE9_OFFSETS)
    n11 = (q_candidates * k_candidates).sum(dim=-1)
    n10 = (q_candidates * (1.0 - k_candidates)).sum(dim=-1)
    n01 = ((1.0 - q_candidates) * k_candidates).sum(dim=-1)
    n00 = ((1.0 - q_candidates) * (1.0 - k_candidates)).sum(dim=-1)
    contingencies = torch.stack((n11, n10, n01, n00), dim=-1)
    coefficients = _quantized_lc4_coefficients(module, cfg).to(dtype=contingencies.dtype)
    scores = torch.einsum("bhnrc,hc->bhnr", contingencies, coefficients)
    scores = _apply_hardware_score_quant(scores / float(q_orig.shape[-1]), cfg)
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    descriptor = _apply_hardware_gate_quant(shiftmax(scores, dim=-1, eps=cfg.eps), cfg)
    row_sum = descriptor.sum(dim=-1)
    attn = _project_match_descriptor(module, descriptor, cfg)
    return attn, row_sum, descriptor, scores


def _g4_match_code_attention(
    module: nn.Module,
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """H78: four fixed 8-lane displacement distributions and a static codebook."""

    q_candidates, k_candidates, valid = _cross_time_match_events(q_orig, k_orig, DE9_OFFSETS)
    head_dim = int(q_orig.shape[-1])
    expected_dim = G4_MATCH_GROUPS * G4_MATCH_GROUP_DIM
    if head_dim != expected_dim:
        raise ValueError(f"G4 Match-Code expects head_dim={expected_dim}, got {head_dim}")
    q_groups = q_candidates.reshape(*q_candidates.shape[:-1], G4_MATCH_GROUPS, G4_MATCH_GROUP_DIM)
    k_groups = k_candidates.reshape(*k_candidates.shape[:-1], G4_MATCH_GROUPS, G4_MATCH_GROUP_DIM)
    n11 = (q_groups * k_groups).sum(dim=-1)
    n00 = ((1.0 - q_groups) * (1.0 - k_groups)).sum(dim=-1)
    scores = (n11 + float(cfg.alpha0) * n00) / float(G4_MATCH_GROUP_DIM)
    scores = _apply_hardware_score_quant(scores.permute(0, 1, 2, 4, 3), cfg)
    group_valid = valid.unsqueeze(-2).expand(-1, -1, -1, G4_MATCH_GROUPS, -1)
    scores = scores.masked_fill(~group_valid, torch.finfo(scores.dtype).min)
    gates = _apply_hardware_gate_quant(shiftmax(scores, dim=-1, eps=cfg.eps), cfg)
    row_sum = gates.sum(dim=-1).sum(dim=-1)
    descriptor = gates.flatten(start_dim=-2)
    flat_scores = scores.flatten(start_dim=-2)
    attn = _project_match_descriptor(module, descriptor, cfg)
    return attn, row_sum, descriptor, flat_scores


def _cf10_null_score(
    scores: torch.Tensor,
    q_activity: torch.Tensor,
    module: nn.Module,
    cfg: ShiftmaxAttentionConfig,
) -> torch.Tensor:
    """Compute CF10's fixed-bias null evidence from valid, masked local scores."""

    top2 = scores.topk(k=2, dim=-1).values
    beta = _quantized_cf10_beta(module, cfg).to(dtype=scores.dtype)
    beta_m = beta[:, 0].view(1, -1, 1)
    beta_q = beta[:, 1].view(1, -1, 1)
    null_score = (
        top2[..., 0]
        - 1.0
        + beta_m * (top2[..., 0] - top2[..., 1])
        + beta_q * (q_activity - 0.5)
    )
    return _apply_hardware_score_quant(null_score, cfg)


def _cf10_match_code_attention(
    module: nn.Module,
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """H79: row-only local assignment with a fixed-zero null codeword."""

    n11, n00, valid = _cross_time_match_counts(q_orig, k_orig, DE9_OFFSETS)
    head_dim = int(q_orig.shape[-1])
    scores = _apply_hardware_score_quant(
        (n11 + float(cfg.alpha0) * n00) / float(head_dim), cfg
    )
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    q_activity = _binary_event_ste(_qkformer_token_q(q_orig)).mean(dim=-1)
    null_score = _cf10_null_score(scores, q_activity, module, cfg)
    scores10 = torch.cat((scores, null_score.unsqueeze(-1)), dim=-1)
    descriptor = _apply_hardware_gate_quant(shiftmax(scores10, dim=-1, eps=cfg.eps), cfg)
    row_sum = descriptor.sum(dim=-1)
    weight = _effective_cf10_match_code_weight(module, cfg).to(dtype=descriptor.dtype)
    attn = torch.einsum("bhnr,hrd->bhnd", descriptor, weight)
    return attn, row_sum, descriptor, scores10


def _dn9_edge_indices(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map fixed local source edges to exact opposite-time destination sets."""

    t_steps, height, width = 2, 9, 9
    n_tokens = t_steps * height * width
    grid = torch.arange(n_tokens, device=device).reshape(t_steps, height, width)
    incoming_indices = []
    incoming_masks = []
    destination_slots = []
    source_masks = []
    for offset_index, (dy, dx) in enumerate(DE9_OFFSETS):
        destination_y = torch.arange(height, device=device).view(1, height, 1)
        destination_x = torch.arange(width, device=device).view(1, 1, width)
        source_y = destination_y - int(dy)
        source_x = destination_x - int(dx)
        incoming_valid = (
            (source_y >= 0) & (source_y < height) & (source_x >= 0) & (source_x < width)
        )
        source_y = source_y.clamp(0, height - 1).expand(t_steps, height, width)
        source_x = source_x.clamp(0, width - 1).expand(t_steps, height, width)
        source_t = (1 - torch.arange(t_steps, device=device)).view(t_steps, 1, 1)
        source_index = grid[source_t.expand_as(source_y), source_y, source_x]
        incoming_indices.append(source_index * len(DE9_OFFSETS) + offset_index)
        incoming_masks.append(incoming_valid.expand(t_steps, height, width))

        source_y = torch.arange(height, device=device).view(1, height, 1)
        source_x = torch.arange(width, device=device).view(1, 1, width)
        target_y = source_y + int(dy)
        target_x = source_x + int(dx)
        source_valid = (
            (target_y >= 0) & (target_y < height) & (target_x >= 0) & (target_x < width)
        )
        target_y = target_y.clamp(0, height - 1).expand(t_steps, height, width)
        target_x = target_x.clamp(0, width - 1).expand(t_steps, height, width)
        target_t = (1 - torch.arange(t_steps, device=device)).view(t_steps, 1, 1)
        target_index = grid[target_t.expand_as(target_y), target_y, target_x]
        destination_slots.append(target_index * len(DE9_OFFSETS) + offset_index)
        source_masks.append(source_valid.expand(t_steps, height, width))

    return (
        torch.stack(incoming_indices, dim=-1).reshape(n_tokens, len(DE9_OFFSETS)),
        torch.stack(incoming_masks, dim=-1).reshape(n_tokens, len(DE9_OFFSETS)),
        torch.stack(destination_slots, dim=-1).reshape(n_tokens, len(DE9_OFFSETS)),
        torch.stack(source_masks, dim=-1).reshape(n_tokens, len(DE9_OFFSETS)),
    )


def _q17_gate_product(value: torch.Tensor) -> torch.Tensor:
    """Unsigned Q1.7 product with an STE for training."""

    clipped = value.clamp(min=0.0, max=255.0 / 128.0)
    return _quantize_ste(clipped, 1.0 / 128.0)


def _dn9_destination_gate(
    scores: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize each edge against all valid local edges entering its destination."""

    incoming_index, incoming_valid, destination_slot, source_valid = _dn9_edge_indices(scores.device)
    flat_scores = scores.reshape(*scores.shape[:-2], -1)
    incoming_scores = flat_scores[:, :, incoming_index]
    incoming_valid = incoming_valid.view(1, 1, *incoming_valid.shape)
    incoming_scores = incoming_scores.masked_fill(
        ~incoming_valid, torch.finfo(incoming_scores.dtype).min
    )
    incoming_gate = shiftmax(incoming_scores, dim=-1, eps=cfg.eps)
    incoming_gate = incoming_gate.masked_fill(~incoming_valid, 0.0)
    incoming_gate = _apply_hardware_gate_quant(incoming_gate, cfg)
    destination_gate = incoming_gate.reshape(*incoming_gate.shape[:-2], -1)[:, :, destination_slot]
    source_valid = source_valid.view(1, 1, *source_valid.shape)
    return destination_gate.masked_fill(~source_valid, 0.0), source_valid


def _dn9_match_code_attention(
    module: nn.Module,
    q_orig: torch.Tensor,
    k_orig: torch.Tensor,
    cfg: ShiftmaxAttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """H80: product of source-row and local destination Shiftmax assignments."""

    n11, n00, valid = _cross_time_match_counts(q_orig, k_orig, DE9_OFFSETS)
    head_dim = int(q_orig.shape[-1])
    scores = _apply_hardware_score_quant(
        (n11 + float(cfg.alpha0) * n00) / float(head_dim), cfg
    )
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    row_gate = shiftmax(scores, dim=-1, eps=cfg.eps).masked_fill(~valid, 0.0)
    row_gate = _apply_hardware_gate_quant(row_gate, cfg)
    destination_gate, destination_valid = _dn9_destination_gate(scores, cfg)
    descriptor = _q17_gate_product(row_gate * destination_gate)
    descriptor = descriptor.masked_fill(~(valid & destination_valid), 0.0)
    row_sum = descriptor.sum(dim=-1)
    attn = _project_match_descriptor(module, descriptor, cfg)
    return attn, row_sum, descriptor, scores


def _ensure_match_code(module: nn.Module, cfg: ShiftmaxAttentionConfig, module_name: str) -> None:
    if cfg.mode not in MATCH_CODE_MODES:
        return
    if cfg.mode in {"binary_de9_match_code", "de9_match_code"}:
        descriptor_dim = 18
    elif cfg.mode in {"binary_mc49_match_code", "mc49_match_code"}:
        descriptor_dim = 49
    elif cfg.mode in {"binary_ax17_match_code", "ax17_match_code"}:
        descriptor_dim = 17
    elif cfg.mode in (
        PC9_MATCH_CODE_MODES | LC4_MATCH_CODE_MODES | CF10_MATCH_CODE_MODES | DN9_MATCH_CODE_MODES
    ):
        descriptor_dim = 9
    else:
        descriptor_dim = G4_MATCH_GROUPS * len(DE9_OFFSETS)
    head_dim = int(module.linear_q.out_features // module.num_heads)
    shape = (int(module.num_heads), descriptor_dim, head_dim)
    existing = getattr(module, "_h9_match_code_weight", None)
    if existing is not None:
        if tuple(existing.shape) != shape:
            raise ValueError(f"Match-Code weight shape mismatch: {tuple(existing.shape)} != {shape}")
    else:
        reference = module.linear_q.weight
        weight = torch.empty(shape, device=reference.device, dtype=reference.dtype)
        seed = int(cfg.match_code_seed) + sum((index + 1) * ord(char) for index, char in enumerate(module_name))
        generator = torch.Generator(device=reference.device)
        generator.manual_seed(seed)
        nn.init.xavier_uniform_(weight, generator=generator)
        module.register_parameter("_h9_match_code_weight", nn.Parameter(weight))
    if cfg.mode in LC4_MATCH_CODE_MODES:
        coefficient_shape = (int(module.num_heads), 4)
        existing_coefficients = getattr(module, "_h9_lc4_coefficients", None)
        if existing_coefficients is not None:
            if tuple(existing_coefficients.shape) != coefficient_shape:
                raise ValueError(
                    "LC4 coefficient shape mismatch: "
                    f"{tuple(existing_coefficients.shape)} != {coefficient_shape}"
                )
        else:
            reference = module.linear_q.weight
            initial = reference.new_tensor((1.0, 0.0, 0.0, float(cfg.alpha0)))
            coefficients = initial.view(1, 4).expand(coefficient_shape[0], -1).clone()
            module.register_parameter("_h9_lc4_coefficients", nn.Parameter(coefficients))
    if cfg.mode in CF10_MATCH_CODE_MODES:
        beta_shape = (int(module.num_heads), 2)
        existing_beta = getattr(module, "_h9_cf10_beta", None)
        if existing_beta is not None:
            if tuple(existing_beta.shape) != beta_shape:
                raise ValueError(
                    f"CF10 beta shape mismatch: {tuple(existing_beta.shape)} != {beta_shape}"
                )
        else:
            reference = module.linear_q.weight
            module.register_parameter(
                "_h9_cf10_beta",
                nn.Parameter(reference.new_zeros(beta_shape)),
            )


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

    if cfg.mode in {"binary_de9_match_code", "de9_match_code"}:
        attn, row_sum, gate, scores = _match_code_attention(
            self, q_orig, k_orig, cfg, dual_evidence=True
        )
    elif cfg.mode in {"binary_mc49_match_code", "mc49_match_code"}:
        attn, row_sum, gate, scores = _match_code_attention(
            self, q_orig, k_orig, cfg, dual_evidence=False
        )
    elif cfg.mode in {"binary_ax17_match_code", "ax17_match_code"}:
        attn, row_sum, gate, scores = _match_code_attention(
            self, q_orig, k_orig, cfg, dual_evidence=False, offsets=AX17_OFFSETS
        )
    elif cfg.mode in PC9_MATCH_CODE_MODES:
        attn, row_sum, gate, scores = _pc9_patch_match_code_attention(
            self, q_orig, k_orig, cfg
        )
    elif cfg.mode in LC4_MATCH_CODE_MODES:
        attn, row_sum, gate, scores = _lc4_match_code_attention(
            self, q_orig, k_orig, cfg
        )
    elif cfg.mode in G4_MATCH_CODE_MODES:
        attn, row_sum, gate, scores = _g4_match_code_attention(
            self, q_orig, k_orig, cfg
        )
    elif cfg.mode in CF10_MATCH_CODE_MODES:
        attn, row_sum, gate, scores = _cf10_match_code_attention(
            self, q_orig, k_orig, cfg
        )
    elif cfg.mode in DN9_MATCH_CODE_MODES:
        attn, row_sum, gate, scores = _dn9_match_code_attention(
            self, q_orig, k_orig, cfg
        )
    elif cfg.mode in {"binary_axnor_temporal_pair_shiftmax", "tp_ttx", "h66_tp"}:
        attn, row_sum, gate = _binary_alpha_xnor_stencil_attention(
            q_orig,
            k_orig,
            cfg,
            temporal_pair=True,
            spatial_cross=False,
            motion_xor_alpha=float(cfg.binary_motion_xor_alpha or 0.0),
        )
        scores = torch.zeros((), device=q_orig.device, dtype=q_orig.dtype)
    elif cfg.mode in {"binary_axnor_local5_shiftmax", "lr_ttx", "h66_lr"}:
        # Pure Local-5: ignore motion alpha so H66d configs stay bit-stable even
        # if a parent template left binary_motion_xor_alpha set.
        attn, row_sum, gate = _binary_alpha_xnor_stencil_attention(
            q_orig,
            k_orig,
            cfg,
            temporal_pair=False,
            spatial_cross=True,
            motion_xor_alpha=0.0,
            profile_module=self,
        )
        scores = torch.zeros((), device=q_orig.device, dtype=q_orig.dtype)
    elif cfg.mode in {
        "binary_axnor_local5_tp_shiftmax",
        "local5_tp",
        "h66f_local5_tp",
        "h66f",
    }:
        # Scheme A: self + temporal peer + 4-axial spatial neighbors (6 lanes).
        attn, row_sum, gate = _binary_alpha_xnor_stencil_attention(
            q_orig,
            k_orig,
            cfg,
            temporal_pair=True,
            spatial_cross=True,
            motion_xor_alpha=0.0,
            profile_module=self,
        )
        scores = torch.zeros((), device=q_orig.device, dtype=q_orig.dtype)
    elif cfg.mode in {
        "binary_axnor_local5_motion_shiftmax",
        "local5_motion",
        "h66g_local5_motion",
        "h66g",
    }:
        # Local-5 + H67 motion bias on the self lane only.
        motion_alpha = float(cfg.binary_motion_xor_alpha or 0.0)
        if motion_alpha == 0.0:
            motion_alpha = 0.25
        attn, row_sum, gate = _binary_alpha_xnor_stencil_attention(
            q_orig,
            k_orig,
            cfg,
            temporal_pair=False,
            spatial_cross=True,
            motion_xor_alpha=motion_alpha,
            profile_module=self,
        )
        scores = torch.zeros((), device=q_orig.device, dtype=q_orig.dtype)
    elif cfg.mode in {"strict_bsa_qkv_shiftmax", "bsa_qkv_shiftmax", "bsa_true_qkv_shiftmax"}:
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
        scores = _apply_hardware_score_quant(scores, cfg)
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
        gate = _apply_hardware_gate_quant(gate, cfg)
        attn = k_orig.mul(gate)
    elif cfg.mode in {"dualrail_binary_tx_qkselector_shiftmax", "binary_dualrail_tx_shiftmax", "date11_drtx"}:
        # DATE11 dual-rail TX: all-binary ATLIF produces {0,+1} events, so
        # signed evidence is recovered by interpreting each head as positive
        # and negative binary rails before the TX-style score.
        scores = _dualrail_binary_tx_token_scores(q_orig, k_orig, cfg)
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        scores = _apply_hardware_score_quant(scores, cfg)
        gate = shiftmax(scores, dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        gate = _apply_hardware_gate_quant(gate, cfg)
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
    elif cfg.mode in {
        "tx_direct_group_shiftmax",
        "direct_group_shiftmax",
        "h63",
    }:
        attn, row_sum, gate, scores = _direct_group_shiftmax_output(q_orig, k_orig, cfg)
    elif cfg.mode in {
        "tx_direct_token_channel_shiftmax",
        "direct_token_channel_shiftmax",
        "h63_stc",
    }:
        attn, row_sum, gate, scores = _direct_token_channel_shiftmax_output(q_orig, k_orig, cfg)
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
        pre_quant_scores = scores
        if cfg.hardware_rtl_shiftmax_enabled:
            if self.training:
                raise RuntimeError("hardware_rtl_shiftmax_enabled 仅用于部署验证，不能用于训练")
            if not cfg.hardware_quant_enabled:
                raise ValueError("RTL Shiftmax 验证要求 hardware_quant_enabled=true")
            scores = _event_selective_temperature(scores, q_orig, k_orig, cfg)
            pre_quant_scores = scores
            scores = _apply_hardware_score_quant(scores, cfg)
            gate = _rtl_shiftmax_gate_q17(
                scores,
                dim=2,
                preserve_mean=cfg.preserve_mean,
            )
            row_sum = gate.sum(dim=2)
            if cfg.preserve_mean:
                row_sum = row_sum / float(n_tokens)
        else:
            if cfg.center_scores:
                scores = scores - scores.mean(dim=2, keepdim=True)
            scores = _event_selective_temperature(scores, q_orig, k_orig, cfg)
            pre_quant_scores = scores
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
            pre_quant_scores=pre_quant_scores,
            gate=gate,
            cfg=cfg,
        )
        attn = k_orig.mul(gate)
        castling_weight = _castling_aux_weight(self, cfg)
        if castling_weight > 0.0:
            matrix_aux = _castling_binary_matrix_output(q_orig, k_orig, cfg).to(dtype=attn.dtype)
            attn = torch.lerp(attn, matrix_aux, castling_weight)
        attn = _window_context_broadcast(attn, cfg)
        self.h9_castling_aux_weight = float(castling_weight)
    elif cfg.mode in {"h82", "class_major_ttx", "cmt_ttx"}:
        # H82 / C8.3: H81 token scores, but Shiftmax is over unique Q7 classes.
        # Gates expand from the Class File. Motion and Local5 stay off.
        if float(cfg.binary_motion_xor_alpha) != 0.0:
            raise RuntimeError("H82 forbids Motion-XOR; do not mix C8.3 with H67")
        mu = _apply_hardware_mu_quant(_scheduled_bipolar_mu(self, cfg), cfg)
        tx_scores, sc_scores = _tx_sc_fusion_score_pair(q_orig, k_orig, cfg)
        scores = tx_scores + mu * sc_scores
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        scores = _event_selective_temperature(scores, q_orig, k_orig, cfg)
        scores = _apply_hardware_score_quant(scores, cfg)
        gate, class_stats = _class_major_shiftmax_gate(scores, cfg)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            row_sum = row_sum / float(n_tokens)
        gate = _apply_hardware_gate_quant(gate, cfg)
        self._h9_class_file_stats = {
            key: value.detach()
            for key, value in class_stats.items()
            if key not in {"codes", "multiplicity", "gate_c"}
        }
        self._h9_class_stability_proxy = class_stats.get("stability_proxy")
        attn = k_orig.mul(gate)
        attn = _window_context_broadcast(attn, cfg)
    elif cfg.mode in {"h83", "class_file_isa"}:
        if float(cfg.binary_motion_xor_alpha) != 0.0:
            raise RuntimeError("H83 forbids Motion-XOR")
        mu = _apply_hardware_mu_quant(_scheduled_bipolar_mu(self, cfg), cfg)
        tx_scores, sc_scores = _tx_sc_fusion_score_pair(q_orig, k_orig, cfg)
        scores = tx_scores + mu * sc_scores
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        scores = _event_selective_temperature(scores, q_orig, k_orig, cfg)
        scores = _apply_hardware_score_quant(scores, cfg)
        class_file = _class_file_from_scores(scores, cfg)
        class_file["gate_c"] = _apply_hardware_gate_quant(class_file["gate_c"], cfg)
        attn, token_gate = _expand_k_from_class_file(k_orig, class_file)
        gate = token_gate
        row_sum = class_file["gate_c"].sum(dim=-1)
        self._h9_class_file = {
            "n_occupied_classes": class_file["n_occupied_classes"].detach(),
            "multiplicity": class_file["multiplicity"].detach(),
            "occupied": class_file["occupied"].detach(),
            "gate_c": class_file["gate_c"].detach(),
            "temporal_pair_mask": class_file["temporal_pair_mask"].detach(),
            "member_jaccard_t0t1": class_file["member_jaccard_t0t1"].detach(),
        }
        self._h9_member_jaccard = class_file["member_jaccard_t0t1"]
        attn = _window_context_broadcast(attn, cfg)
    elif cfg.mode in {"h84", "packed_class_file"}:
        if float(cfg.binary_motion_xor_alpha) != 0.0:
            raise RuntimeError("H84 forbids Motion-XOR")
        mu = _apply_hardware_mu_quant(_scheduled_bipolar_mu(self, cfg), cfg)
        tx_scores, _sc_scores = _tx_sc_fusion_score_pair(q_orig, k_orig, cfg)
        scores = tx_scores + mu * _sc_scores
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        scores = _event_selective_temperature(scores, q_orig, k_orig, cfg)
        scores = _apply_hardware_score_quant(scores, cfg)
        step, lo, _hi, n_bins = _q7_class_grid(cfg)
        squeezed = scores.squeeze(-1)
        centers = lo + step * torch.arange(n_bins, device=scores.device, dtype=scores.dtype)
        member, hard = _soft_hard_membership(squeezed, centers, step)
        class_file = _pack_occupied_class_file(member, hard, squeezed, centers, cfg)
        class_file["gate_c"] = _apply_hardware_gate_quant(class_file["gate_c"], cfg)
        attn, token_gate = _expand_k_from_packed_class_file(k_orig, class_file)
        gate = token_gate
        row_sum = class_file["gate_c"].sum(dim=-1)
        row_jaccard = _adjacent_row_class_jaccard(
            class_file["member_mask"], class_file["valid"]
        )
        self._h9_class_file = {
            "class_id": class_file["class_id"].detach(),
            "valid": class_file["valid"].detach(),
            "member_mask": class_file["member_mask"].detach(),
            "gate_c": class_file["gate_c"].detach(),
            "multiplicity": class_file["multiplicity"].detach(),
            "n_occupied_classes": class_file["n_occupied_classes"].detach(),
            "row_jaccard": row_jaccard.detach(),
        }
        self._h9_row_jaccard = row_jaccard
        attn = _window_context_broadcast(attn, cfg)
    elif cfg.mode in {"h85", "row_delta_class_file"}:
        if float(cfg.binary_motion_xor_alpha) != 0.0:
            raise RuntimeError("H85 forbids Motion-XOR")
        mu = _apply_hardware_mu_quant(_scheduled_bipolar_mu(self, cfg), cfg)
        tx_scores, _sc_scores = _tx_sc_fusion_score_pair(q_orig, k_orig, cfg)
        scores = tx_scores + mu * _sc_scores
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        scores = _event_selective_temperature(scores, q_orig, k_orig, cfg)
        scores = _apply_hardware_score_quant(scores, cfg)
        class_file = _build_h85_row_files(scores, cfg)
        class_file["gate_c"] = _apply_hardware_gate_quant(class_file["gate_c"], cfg)
        attn = _expand_k_from_h85_row_files(k_orig, class_file)
        gate = class_file["gate_c"]
        row_sum = class_file["gate_c"].sum(dim=-1)
        reuse = class_file["reuse_set"].to(dtype=scores.dtype).mean()
        self._h9_class_file = {
            "class_id": class_file["class_id"].detach(),
            "valid": class_file["valid"].detach(),
            "member_idx": class_file["member_idx"].detach(),
            "member_ok": class_file["member_ok"].detach(),
            "gate_c": class_file["gate_c"].detach(),
            "shared_ids": class_file["shared_ids"].detach(),
            "insert_ids": class_file["insert_ids"].detach(),
            "delete_ids": class_file["delete_ids"].detach(),
            "reuse_set": class_file["reuse_set"].detach(),
            "n_class": class_file["n_class"].detach(),
        }
        self._h9_reuse_set_rate = reuse
        attn = _window_context_broadcast(attn, cfg)
    elif cfg.mode in {"h86", "member_delta_class_file"}:
        if float(cfg.binary_motion_xor_alpha) != 0.0:
            raise RuntimeError("H86 forbids Motion-XOR")
        mu = _apply_hardware_mu_quant(_scheduled_bipolar_mu(self, cfg), cfg)
        tx_scores, _sc_scores = _tx_sc_fusion_score_pair(q_orig, k_orig, cfg)
        scores = tx_scores + mu * _sc_scores
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        scores = _event_selective_temperature(scores, q_orig, k_orig, cfg)
        scores = _apply_hardware_score_quant(scores, cfg)
        class_file = _build_h86_member_delta_file(scores, cfg)
        class_file["gate_c"] = _apply_hardware_gate_quant(class_file["gate_c"], cfg)
        attn = _expand_k_from_h86_member_delta(k_orig, class_file)
        gate = class_file["gate_c"]
        row_sum = class_file["gate_c"].sum(dim=-1)
        self._h9_class_file = {
            "class_id": class_file["class_id"].detach(),
            "valid": class_file["valid"].detach(),
            "gate_c": class_file["gate_c"].detach(),
            "row0_member_idx": class_file["row0_member_idx"].detach(),
            "member_insert": class_file["member_insert"].detach(),
            "member_delete": class_file["member_delete"].detach(),
            "class_shared": class_file["class_shared"].detach(),
            "class_insert": class_file["class_insert"].detach(),
            "class_delete": class_file["class_delete"].detach(),
            "member_jaccard_surviving": class_file["member_jaccard_surviving"].detach(),
            "n_occupied_classes": class_file["n_occupied_classes"].detach(),
        }
        self._h9_member_jaccard_surviving = class_file["member_jaccard_surviving"]
        attn = _window_context_broadcast(attn, cfg)
    elif cfg.mode in {"motion_t5_quotient", "h87"}:
        # D1: Motion T=5 时间商（Motion-XOR 线的扩展，合同草案 D1）。
        # 每槽规范融合式 s_t = min(RNE16(64·o_t + sz_t + 16·m̄_t), 162)；
        # 时间维 run-length 广播执行账（I6：eq=0.979 下独立门 1.084/5，−78.3%）。
        # 保持 Swin 分窗 (2,15,15) 不动，T=5 分组在算子内完成（跨窗时间槽），
        # 全部模型参数与 Motion ep35 锚点 checkpoint 兼容（纯算子消融口径）。
        if float(cfg.binary_motion_xor_alpha) != 0.0:
            raise RuntimeError(
                "h87/motion_t5_quotient embeds the canonical 16·m̄ term; "
                "set binary_motion_xor_alpha=0 (do not double-count motion)"
            )
        scores, rle_stats, slot_views = _binary_t5_quotient_token_scores(
            q_orig, k_orig, cfg
        )
        # F1（2026-08-19）：Q7 整数分数（0..162）进 shiftmax 前 ÷128 —— 定点
        # 指数语义 2^(s/128) 对齐 h67 现网与 RTL q17（D1 漂移诊断 §5 根因）。
        # 分数位账（I1-I7）不变：slot_views/挂载视图仍为整数 Q7。
        scores = scores / 128.0
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        pre_quant_scores = scores
        scores = _event_selective_temperature(scores, q_orig, k_orig, cfg)
        pre_quant_scores = scores
        scores = _apply_hardware_score_quant(scores, cfg)
        gate = shiftmax(scores, dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        gate = _apply_hardware_gate_quant(gate, cfg)
        attn = k_orig.mul(gate)
        attn = _window_context_broadcast(attn, cfg)
        self._h9_d1_rle_stats = rle_stats
        self._h9_d1_slot_scores = slot_views["scores"].detach()
        self._h9_d1_slot_overlap = slot_views["overlap"].detach()
        self._h9_d1_slot_remainder = slot_views["remainder"].detach()
    elif cfg.mode in {"motion_t4_pad_quotient", "h87b"}:
        # B2: Motion T=4 + pad wildcard 时间商（D1 的 plan B 预案，
        # D1_VARIANT_SEARCH_20260819.md §4.1）。真实槽融合式与 D1 逐位一致
        # （s_t = min(RNE16(64·o_t + sz_t + 16·m̄_t), 162)）；末组 2 个 pad
        # 槽以 wildcard 掩码跳过（不参与商组/统计/广播）。运动项嵌入规范
        # 融合式：motion alpha 非 0 抛错（同 D1，运动不双重计数）。
        if float(cfg.binary_motion_xor_alpha) != 0.0:
            raise RuntimeError(
                "h87b/motion_t4_pad_quotient embeds the canonical 16·m̄ term; "
                "set binary_motion_xor_alpha=0 (do not double-count motion)"
            )
        scores, rle_stats, slot_views = _binary_t4_pad_quotient_token_scores(
            q_orig, k_orig, cfg
        )
        # F1（2026-08-19）：与 h87 同 —— Q7 整数分数进 shiftmax 前 ÷128
        # （定点指数语义 2^(s/128) 对齐 RTL q17；真实槽分数位账不变）。
        scores = scores / 128.0
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        scores = _event_selective_temperature(scores, q_orig, k_orig, cfg)
        scores = _apply_hardware_score_quant(scores, cfg)
        gate = shiftmax(scores, dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        gate = _apply_hardware_gate_quant(gate, cfg)
        attn = k_orig.mul(gate)
        attn = _window_context_broadcast(attn, cfg)
        self._h9_b2_rle_stats = rle_stats
        self._h9_b2_slot_scores = slot_views["scores"].detach()
        self._h9_b2_slot_overlap = slot_views["overlap"].detach()
        self._h9_b2_slot_remainder = slot_views["remainder"].detach()
        self._h9_b2_pad_mask = slot_views["pad_mask"].detach()
        self._h9_b2_grouped_runs = slot_views["grouped_runs"].detach()
    elif cfg.mode in {"binary_axnor_local5_a3s_shiftmax", "local5_a3s", "h88"}:
        # D3: Local5 5-lane stencil + 方向场偏移 ±Δ（A3S，合同草案 D3）。
        # 方向场 = 3×3 时域 XOR 梯度 argmax（2bit/pixel，固定位图，无梯度）；
        # 对齐 lane +Δ、正交 −Δ、self 0（Δ 以 Q7 1/128 档计，8 档 = Δ=1/16）。
        # Δ=0 档与现网 Local5 逐位一致（K1 锚点，可注入式训练）；motion alpha
        # 同 Local5 纪律静默忽略（H66d 模板继承时保持位稳定）。
        attn, row_sum, gate, a3s_stats = _binary_axnor_local5_a3s_attention(
            q_orig,
            k_orig,
            cfg,
            profile_module=self,
        )
        scores = torch.zeros((), device=q_orig.device, dtype=q_orig.dtype)
        self._h9_a3s_direction_field = a3s_stats["direction_field"]
        self._h9_a3s_delta_bins = a3s_stats["delta_bins"]
        self._h9_a3s_axis_frac_ew = a3s_stats["axis_frac_ew"]
        self._h9_a3s_winner_hit_rate = a3s_stats["winner_hit_rate"]
    elif cfg.mode in {"motion_sw12_overlap", "h89"}:
        # D2: Motion 跨窗语义 —— stride-12/窗口-15 重叠滑窗 + 滚动分母
        # （合同草案 D2，J1-J6）。窗口划分在算子内部完成（不动 Swin
        # window_partition 与模型参数，checkpoint 兼容，纯算子消融口径）。
        # 运动项嵌入规范融合式（16·m̄，同 D1 纪律）：motion alpha 非 0 抛错。
        if float(cfg.binary_motion_xor_alpha) != 0.0:
            raise RuntimeError(
                "h89/motion_sw12_overlap embeds the canonical 16·m̄ term; "
                "set binary_motion_xor_alpha=0 (do not double-count motion)"
            )
        attn, row_sum, gate, sw12_stats = _binary_motion_sw12_overlap_attention(
            q_orig, k_orig, cfg
        )
        scores = torch.zeros((), device=q_orig.device, dtype=q_orig.dtype)
        self._h9_d2_scores = sw12_stats["scores"].detach()
        self._h9_d2_rolling_z = sw12_stats["rolling_z"].detach()
        self._h9_d2_z_full = sw12_stats["z_full"].detach()
        self._h9_d2_exp_ledger = sw12_stats["exp_ledger"]
        self._h9_d2_catalog = sw12_stats["catalog"]
        self._h9_d2_window_plan = sw12_stats["window_plan"]
        self._h9_d2_gate_final = sw12_stats["gate_final"].detach()
        self._h9_d2_gate_mult = sw12_stats["gate_mult"].detach()
        self._h9_d2_batch_decomposition = sw12_stats["batch_decomposition"]
        self._h9_d2_window_counts = sw12_stats["window_counts"]
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
        scores = _apply_hardware_score_quant(scores, cfg)
        gate = shiftmax(scores, dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        gate = _apply_hardware_gate_quant(gate, cfg)
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
        scores = _apply_hardware_score_quant(scores, cfg)
        gate = shiftmax(scores, dim=2, eps=cfg.eps)
        row_sum = gate.sum(dim=2)
        if cfg.preserve_mean:
            gate = gate * float(n_tokens)
        gate = _apply_hardware_gate_quant(gate, cfg)
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
            "dualrail_binary_tx_qkselector_shiftmax/date11_drtx, "
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
            "tx_direct_group_shiftmax/h63, "
            "tx_direct_token_channel_shiftmax/h63_stc, "
            "h60/tx_sc_k_mag_no_carrier_shiftmax, "
            "h82/class_major_ttx/cmt_ttx, "
            "h83/class_file_isa, "
            "h84/packed_class_file, "
            "h85/row_delta_class_file, "
            "h86/member_delta_class_file, "
            "ternary_alpha_xnor_local_shiftmax/h59_local, "
            "sc_ad_confidence_carrier_blend_shiftmax/h56mc, "
            "ternary_alpha_xnor_shiftmax/h18a, ternary_alpha_xnor_shiftmax_residual/h48, "
            "ternary_alpha_xnor_l1/h18a_l1, "
            "ternary_alpha_xnor_ssa_linear/h42b, ternary_alpha_xnor_ssa_qkv_linear/h42c, "
            "ternary_alpha_xnor_ssa_qkv_shiftmax/h42d, ternary_alpha_xnor_ssa_kreuse_shiftmax/h45, "
            "binary_alpha_xnor_matrix_shiftmax/l1, binary_axnor_temporal_pair_shiftmax/h66_tp, "
            "binary_axnor_local5_shiftmax/h66_lr, "
            "binary_axnor_local5_tp_shiftmax/h66f, binary_axnor_local5_motion_shiftmax/h66g, "
            "binary_de9_match_code/de9_match_code, binary_mc49_match_code/mc49_match_code, "
            "binary_ax17_match_code/ax17_match_code, "
            "binary_pc9_patch_match_code/h76_pc9, binary_lc4_match_code/h77_lc4, "
            "binary_g4_match_code/h78_g4, "
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
        _ensure_match_code(module, cfg, name)
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
    direct_modules = [
        module
        for module in modules
        if getattr(module._h9_shiftmax_cfg, "mode", "")
        in {
            "tx_direct_group_shiftmax",
            "direct_group_shiftmax",
            "h63",
            "tx_direct_token_channel_shiftmax",
            "direct_token_channel_shiftmax",
            "h63_stc",
        }
    ]
    match_code_modules = [
        module for module in modules
        if getattr(module._h9_shiftmax_cfg, "mode", "") in MATCH_CODE_MODES
    ]
    return {
        "num_modules": len(modules),
        "row_sum_mean": sum(row_means) / len(row_means),
        "gate_mean": sum(gate_means) / len(gate_means),
        "score_mean": sum(score_means) / len(score_means),
        "direct_shiftmax_modules": len(direct_modules),
        "direct_shiftmax_groups_mean": (
            sum(float(module._h9_shiftmax_cfg.direct_shiftmax_groups) for module in direct_modules)
            / len(direct_modules)
            if direct_modules
            else 0.0
        ),
        "match_code_modules": len(match_code_modules),
        "match_code_parameters": sum(
            int(module._h9_match_code_weight.numel())
            + int(getattr(module, "_h9_lc4_coefficients", torch.empty(0)).numel())
            + int(getattr(module, "_h9_cf10_beta", torch.empty(0)).numel())
            for module in match_code_modules
        ),
    }


def set_shiftmax_attention_step(model: nn.Module, step: int) -> int:
    count = 0
    for module in model.modules():
        if hasattr(module, "_h9_shiftmax_cfg"):
            module._h9_global_step = int(step)
            count += 1
    return count
