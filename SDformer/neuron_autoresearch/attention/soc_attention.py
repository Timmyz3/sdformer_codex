"""T5: Sign-Only Consensus (SOC) attention for ternary spiking Q/K.

Replaces the compat_qk_product Shiftmax with pure sign-based popcount attention.
theta handles sparsity (ATLIF threshold); sign handles attention (SOC).
Completely decoupled two-factor design.

Hardware: popcount + L1-norm = comparators + adder tree + 1 divider per token.
Zero exponentiation, zero multiplication beyond the final gate application.
"""

from __future__ import annotations

from types import MethodType
from typing import Any, Iterable

import torch
import torch.nn as nn


def soc_gate(q_sign: torch.Tensor, k_sign: torch.Tensor, eps: float = 1e-6):
    """Compute token consensus gate from ternary Q/K sign tensors.

    Args:
        q_sign: Q signs, shape [B, heads, n_tokens, head_dim], values in {-1,0,+1}
        k_sign: K signs, same shape and semantics
        eps: numerical stability

    Returns:
        gate: per-token scalar gate, shape [B, heads, n_tokens, 1]
    """
    q_active = q_sign.ne(0)
    k_active = k_sign.ne(0)
    both_active = q_active & k_active
    agree = (q_sign.eq(k_sign) & both_active).sum(dim=-1, keepdim=True).float()
    disagree = (q_sign.ne(k_sign) & both_active).sum(dim=-1, keepdim=True).float()
    silent = (~both_active).sum(dim=-1, keepdim=True).float()

    consensus = (agree - disagree) / (agree + disagree + silent + eps)

    pos = torch.clamp(consensus, min=0.0)
    denom = pos.sum(dim=2, keepdim=True) + eps
    gate = pos / denom

    n = q_sign.shape[2]
    gate = gate * float(n)
    return gate


def _soc_attention_forward(self, x, mask=None):
    """Patched forward for Spiking_QK_WindowAttention3D using SOC.

    Replaces the entire QK attention computation with sign-only consensus gating.
    theta thresholds are used ONLY for ATLIF sparsity, NOT for attention.
    """
    del mask
    T, B_, H, W, C = x.shape
    head_dim = C // self.num_heads

    x = self.proj_sn(x.float())
    q = self.linear_q(x)
    if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
        q = self.bn_q(q.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
    q_spike = self.sn_q(q)

    k = self.linear_k(x).float()
    if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
        k = self.bn_k(k.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
    positional_encoding = self.positional_encoding.reshape(T, 1, H, W, C)
    k = self.sn_k(k + positional_encoding)

    q_orig = q_spike.reshape(T, B_, self.num_heads, -1, head_dim)
    k_orig = k.reshape(B_, self.num_heads, -1, head_dim)
    n_tokens = k_orig.shape[2]

    q_sign = q_orig.sign().permute(1, 2, 0, 3, 4).reshape(B_, self.num_heads, n_tokens, head_dim)
    k_sign = k_orig.sign()

    gate = soc_gate(q_sign, k_sign)

    with torch.no_grad():
        self.soc_gate_mean = float(gate.detach().mean().cpu())
        self.soc_agree_mean = float(
            (q_sign.eq(k_sign) & q_sign.ne(0) & k_sign.ne(0)).float().mean().cpu()
        )

    attn = k_orig.mul(gate)

    attn = self.attn_drop(attn)
    x = attn.reshape(B_, self.num_heads, T, H, W, head_dim)
    x = x.permute(2, 0, 3, 4, 1, 5).reshape(T, B_, H, W, C).float()
    attn = self.attn_sn(x)
    x = self.proj(x)
    if self.norm_layer in ["BN", "BNTT", "tdBN", "IN"]:
        x = self.proj_bn(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
    x = x.reshape(B_, n_tokens, C)
    return x, attn


def install_soc_attention(model: nn.Module, enabled: bool = True) -> list[str]:
    """Replace Spiking_QK_WindowAttention3D.forward with SOC version."""
    if not enabled:
        return []

    installed: list[str] = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != "Spiking_QK_WindowAttention3D":
            continue
        module.forward = MethodType(_soc_attention_forward, module)
        installed.append(name)
    return installed


def soc_attention_summary(model: nn.Module) -> dict[str, float | int]:
    modules = [
        m for m in model.modules()
        if hasattr(m, "soc_gate_mean")
    ]
    if not modules:
        return {"num_modules": 0}
    gate_means = [float(m.soc_gate_mean) for m in modules]
    agree_means = [float(m.soc_agree_mean) for m in modules]
    return {
        "num_modules": len(modules),
        "gate_mean": sum(gate_means) / len(gate_means),
        "agree_mean": sum(agree_means) / len(agree_means),
    }
