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
    value_mode: str = "threshold"
    value_branch: str = "reuse_k"
    value_init: str = "copy_k"
    alpha0: float = 0.05
    mismatch_penalty: float = 0.5
    relu_k_floor: float = 0.0


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
        value_mode=str(raw.get("value_mode", "threshold")),
        value_branch=str(raw.get("value_branch", "reuse_k")),
        value_init=str(raw.get("value_init", "copy_k")),
        alpha0=float(raw.get("alpha0", 0.05)),
        mismatch_penalty=float(raw.get("mismatch_penalty", 0.5)),
        relu_k_floor=float(raw.get("relu_k_floor", 0.0)),
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


def _qkformer_token_q(q_orig: torch.Tensor) -> torch.Tensor:
    return q_orig.permute(1, 2, 0, 3, 4).reshape(
        q_orig.shape[1],
        q_orig.shape[2],
        q_orig.shape[0] * q_orig.shape[3],
        q_orig.shape[4],
    )


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
    add -1, silent channels add 0. Optional normalization keeps Shiftmax from
    becoming too sharp while remaining power-of-two friendly for head_dim values.
    """

    q_event = _ternary_sign_ste(_qkformer_token_q(q_orig))
    k_event = _ternary_sign_ste(k_orig)
    score = (q_event * k_event).sum(dim=-1, keepdim=True)
    norm = cfg.consensus_score_norm
    if norm in {"head_dim", "dim"}:
        score = score / float(max(1, q_event.shape[-1]))
    elif norm in {"sqrt_head_dim", "sqrt_dim"}:
        score = score / float(max(1, q_event.shape[-1]) ** 0.5)
    elif norm == "active":
        active = (q_event.detach().ne(0) & k_event.detach().ne(0)).sum(dim=-1, keepdim=True).clamp_min(1)
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
    score = (
        same_nonzero.to(dtype=q_orig.dtype)
        + float(cfg.alpha0) * same_zero.to(dtype=q_orig.dtype)
        - float(cfg.mismatch_penalty) * opposite.to(dtype=q_orig.dtype)
    ).sum(dim=-1, keepdim=True)
    active = None
    if cfg.consensus_score_norm == "active":
        active = (q_active | k_active).sum(dim=-1, keepdim=True).clamp_min(1)
    return _normalize_consensus_score(score, q_event.shape[-1], cfg, active=active)


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
    active = None
    if cfg.consensus_score_norm == "active":
        q_active = q_event.ne(0).to(dtype=q_orig.dtype)
        k_active = k_event.ne(0).to(dtype=q_orig.dtype)
        active = torch.matmul(q_active, k_active.transpose(-2, -1))
    return _normalize_consensus_score(score, q_event.shape[-1], cfg, active=active)


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
        # same polarity contributes +1, opposite polarity -1, silence 0. Shiftmax
        # is retained as the BSA-style normalization, but its input is now a
        # sign-consensus score instead of a theta-weighted real-valued product.
        scores = _signed_consensus_token_scores(q_orig, k_orig, cfg)
        if cfg.center_scores:
            scores = scores - scores.mean(dim=2, keepdim=True)
        gate = shiftmax(scores, dim=2, eps=cfg.eps)
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
            "signed_consensus_shiftmax/h13b, signed_consensus_shiftnorm/h13c, "
            "signed_consensus_popcount_l1/h13t, "
            "ternary_alpha_xnor_shiftmax/h18a, ternary_alpha_xnor_l1/h18a_l1, "
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
        if cfg.mode in {"strict_bsa_qkv_shiftmax", "bsa_qkv_shiftmax", "bsa_true_qkv_shiftmax", "a2os2a_qkv_l1", "a2os2a_true_qkv_l1"}:
            _ensure_independent_value_branch(module, cfg)
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
    score_means = [float(getattr(module, "h13_consensus_score_mean", 0.0)) for module in modules]
    return {
        "num_modules": len(modules),
        "row_sum_mean": sum(row_means) / len(row_means),
        "gate_mean": sum(gate_means) / len(gate_means),
        "score_mean": sum(score_means) / len(score_means),
    }
