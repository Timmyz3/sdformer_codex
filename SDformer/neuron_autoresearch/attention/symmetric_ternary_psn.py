"""Symmetric Ternary PSN: S1 + S5 combined.

S1: neg_thre = thre (not thre * 30). Fixes negative firing death.
S5: out = (pos_spike - neg_spike) * thre. Strict magnitude constraint.

Installation: wraps existing H9 ATLIFTernaryPSN modules with sym_forward.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from types import MethodType


def _symmetric_ternary_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
    """Patched forward: neg_thre = thre (S1), output = (pos-neg)*thre (S5).

    Reuses the parent module's PSN weight/bias and threshold update mechanism
    but replaces the ternary surrogate call with symmetric threshold logic.
    """
    T = self.T
    h_seq = torch.addmm(self.bias, self.weight, x_seq.flatten(1))

    # S1: symmetric thresholds
    thre = self.thresh.abs()  # ensure positive
    neg_thre = thre           # NOT thre * neg_scale

    # S5: strict magnitude-constrained output
    pos_spike = (h_seq > thre).float()
    neg_spike = (h_seq < -neg_thre).float()

    # Surrogate gradient via straight-through estimator on threshold boundary
    pos_ste = ((h_seq - thre).abs() < thre).float() * (1.0 / (thre + 1e-8))
    neg_ste = (((-h_seq) - neg_thre).abs() < neg_thre).float() * (1.0 / (neg_thre + 1e-8))

    spike = (pos_spike - neg_spike) * thre
    spike = spike + (pos_ste - neg_ste).detach() * thre - (
        (pos_spike - neg_spike) * thre
    ).detach()

    # ATLIF update tracking (same as original)
    with torch.no_grad():
        pos_active = pos_spike.float()
        neg_active = neg_spike.float()
        total_active = pos_active + neg_active
        thre_update = self.sp * (pos_active - neg_active * 0.5).mean() * (
            total_active.float().mean()
        )

    if isinstance(self.update_value, torch.Tensor):
        self.update_value = self.update_value + thre_update.detach()
    else:
        self.update_value = float(thre_update.detach().cpu().item()
            if thre_update.numel() == 1 else thre_update.detach().mean().cpu().item())

    # Activity tracking
    out = spike.view(x_seq.shape)
    with torch.no_grad():
        active = spike.ne(0).float()
        self.r = active.mean().item()
        self.pos_r = pos_spike.float().mean().item()
        self.neg_r = neg_spike.float().mean().item()
    self.act_value = out.abs().reshape(out.size(0), -1).mean(1).sum()

    return out


def install_symmetric_ternary(model: nn.Module, raw_config: dict | None) -> list[str]:
    """Patch all ATLIFTernaryPSN modules with symmetric ternary forward (S1+S5)."""
    sc = (raw_config or {}).get("symmetric_ternary", {})
    if not sc.get("enabled", False):
        return []

    installed = []
    for name, module in model.named_modules():
        clsname = module.__class__.__name__
        if clsname != "ATLIFTernaryPSN":
            continue
        if hasattr(module, "_sym_original_forward"):
            continue  # already patched
        module._sym_original_forward = module.forward
        module.forward = MethodType(_symmetric_ternary_forward, module)
        installed.append(name)

    return installed
