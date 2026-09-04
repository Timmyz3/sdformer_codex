"""Prototype ternary-native attention gates for discussion only.

This file is intentionally not imported by the H9/H10 training entrypoints.
It sketches hardware-friendly alternatives to softmax/shiftmax for the
SDFormerFlow QKFormer-style attention path:

    attn = K_carrier * token_gate(Q)

The current ATLIF ternary neurons emit threshold-weighted values {-theta, 0,
+theta}, not plain {-1, 0, +1}. These helpers therefore expose both sign-count
and threshold-aware variants.
"""

from __future__ import annotations

import torch


def _pow2_mass_normalize(gate_raw: torch.Tensor, token_dim: int, eps: float = 1.0e-6) -> torch.Tensor:
    """Normalize integer gates with a power-of-two denominator.

    This mirrors the hardware motivation of Shiftmax: division can be replaced
    by a shift when the denominator is rounded up to a power of two.
    """

    mass = gate_raw.sum(dim=token_dim, keepdim=True).clamp_min(eps)
    denom = torch.pow(2.0, torch.ceil(torch.log2(mass)))
    n_tokens = gate_raw.shape[token_dim]
    return gate_raw * (float(n_tokens) / denom)


def _mean_mass_normalize(gate_raw: torch.Tensor, token_dim: int, eps: float = 1.0e-6) -> torch.Tensor:
    """Normalize gates to keep the mean gate close to one."""

    mass = gate_raw.sum(dim=token_dim, keepdim=True).clamp_min(eps)
    n_tokens = gate_raw.shape[token_dim]
    return gate_raw * (float(n_tokens) / mass)


def ternary_native_gate_from_q(
    q: torch.Tensor,
    *,
    theta_q: torch.Tensor | float | None = None,
    token_dim: int = -2,
    channel_dim: int = -1,
    levels: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0),
    beta: float = 4.0,
    use_threshold_strength: bool = False,
    normalize: str = "pow2",
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Build a small-integer token gate from ternary Q events.

    Args:
        q: Threshold-weighted ternary Q, typically shaped [B, heads, tokens, C].
        theta_q: Learned ATLIF threshold. When provided and
            use_threshold_strength=False, q/theta_q recovers approximately
            {-1, 0, +1} before counting spikes.
        token_dim: Token/window dimension normalized over.
        channel_dim: Channel/head-dim dimension reduced into a token score.
        levels: Quantized gate levels. Powers of two are hardware friendly.
        beta: Larger beta makes the quantizer less aggressive.
        use_threshold_strength: If True, use raw q sums, so learned threshold
            magnitude affects the gate. If False, use sign counts only.
        normalize: "pow2", "mean", or "none".

    Returns:
        A non-negative gate with shape q.shape without channel_dim.
    """

    if use_threshold_strength or theta_q is None:
        q_score_source = q
    else:
        theta = torch.as_tensor(theta_q, device=q.device, dtype=q.dtype).clamp_min(eps)
        q_score_source = q / theta

    score = q_score_source.sum(dim=channel_dim, keepdim=False)
    score = score - score.mean(dim=token_dim, keepdim=True)

    max_level = float(max(levels))
    raw_index = torch.round(score / float(beta) + 1.0).clamp(0.0, max_level)

    # Map to the nearest configured level. For the default powers-of-two set,
    # this keeps the runtime representation small and shift-friendly.
    level_tensor = torch.tensor(levels, device=q.device, dtype=q.dtype)
    nearest = (raw_index.unsqueeze(-1) - level_tensor).abs().argmin(dim=-1)
    gate_raw = level_tensor[nearest]

    if normalize == "pow2":
        return _pow2_mass_normalize(gate_raw, token_dim=token_dim, eps=eps)
    if normalize == "mean":
        return _mean_mass_normalize(gate_raw, token_dim=token_dim, eps=eps)
    if normalize == "none":
        return gate_raw
    raise ValueError("normalize must be pow2, mean, or none")


def apply_tngn_qkformer_carrier(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    theta_q: torch.Tensor | float | None = None,
    mode: str = "sign_count",
    token_dim: int = -2,
    channel_dim: int = -1,
    levels: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0),
    beta: float = 4.0,
    normalize: str = "pow2",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply TNGN to the existing QKFormer carrier.

    q and k should already be reshaped to [B, heads, tokens, C]. The output is
    analogous to baseline QKFormer attention carrier, but with a ternary-native
    gate instead of softmax/shiftmax.
    """

    gate = ternary_native_gate_from_q(
        q,
        theta_q=theta_q,
        token_dim=token_dim,
        channel_dim=channel_dim,
        levels=levels,
        beta=beta,
        use_threshold_strength=(mode == "threshold_strength"),
        normalize=normalize,
    )
    return k * gate.unsqueeze(channel_dim), gate
