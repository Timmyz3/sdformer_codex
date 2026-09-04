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
    h_seq = torch.addmm(self.bias, self.weight, x_seq.flatten(1))

    # S1+S5 through the original custom autograd path:
    # neg_scale=1.0 gives symmetric trigger thresholds and the original
    # TernarySurrogate returns exactly {-thresh, 0, +thresh}.
    spike, thre_updates = self.act(h_seq, self.thresh, self.sp, 1.0)
    self.update_value += thre_updates

    # Activity tracking
    out = spike.view(x_seq.shape)
    with torch.no_grad():
        thresh = self.thresh.detach().clamp_min(1e-12)
        ternary = out / thresh
        active = ternary.ne(0).float()
        self.r = active.mean().item()
        self.pos_r = ternary.gt(0).float().mean().item()
        self.neg_r = ternary.lt(0).float().mean().item()
    self.act_value = out.abs().reshape(out.size(0), -1).mean(1).sum()

    return out


def install_symmetric_ternary(model: nn.Module, raw_config: dict | None) -> list[str]:
    """Patch all ATLIFTernaryPSN modules with symmetric ternary forward (S1+S5)."""
    sc = (raw_config or {}).get("symmetric_ternary", {})
    if not sc.get("enabled", False):
        return []

    installed = []
    patch_binary = bool(sc.get("patch_binary", False))
    for name, module in model.named_modules():
        clsname = module.__class__.__name__
        if clsname != "ATLIFTernaryPSN":
            continue
        if getattr(module, "output_mode", "ternary") != "ternary" and not patch_binary:
            continue
        if hasattr(module, "_sym_original_forward"):
            continue  # already patched
        module._sym_original_forward = module.forward
        module.forward = MethodType(_symmetric_ternary_forward, module)
        installed.append(name)

    return installed
