"""PSN temporal mixer with ATLIF threshold updates and ternary output."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def zif_backward(x: torch.Tensor, thre: torch.Tensor) -> torch.Tensor:
    return (1.0 - (x / thre).abs()).clamp_min(0)


class TernarySurrogate(torch.autograd.Function):
    """ATLIF-style threshold update with ternary {-thre, 0, +thre} output."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, thre: torch.Tensor, sp: float, neg_scale: float):
        neg_thre = thre * float(neg_scale)
        pos_active = (input >= thre).float()
        neg_active = (input <= -neg_thre).float()
        ternary = pos_active - neg_active
        pos_updates = zif_backward(input - thre, thre) * pos_active
        neg_updates = zif_backward((-input) - neg_thre, neg_thre) * neg_active
        thre_updates = (sp * (pos_updates + neg_updates)).sum(0).mean().item()
        ctx.save_for_backward(input, thre, neg_thre)
        return ternary * thre, thre_updates

    @staticmethod
    def backward(ctx, grad_input: torch.Tensor, _dummy):
        input, thre, neg_thre = ctx.saved_tensors
        pos_tmp = (1.0 - ((input - thre) / thre).abs()).clamp(min=0)
        neg_tmp = (1.0 - (((-input) - neg_thre) / neg_thre).abs()).clamp(min=0)
        tmp = torch.maximum(pos_tmp, neg_tmp)
        grad_input = grad_input * tmp
        grad_thre = -(grad_input.abs() * tmp).mean()
        return grad_input, grad_thre, None, None


class ATLIFTernaryPSN(nn.Module):
    """PSN-compatible neuron combining PSN, ATLIF threshold growth, and ternary spikes."""

    official_atlif_source = "https://github.com/putshua/Activity-Pruning-SNN"
    official_tsn_source = "https://github.com/yfguo91/Ternary-Spike"

    def __init__(
        self,
        T: int,
        base_psn: nn.Module | None = None,
        thresh: float = 1.0,
        sparsity_eta: float = 0.0,
        negative_threshold_scale: float = 5.0,
        activity_eta: float = 0.0,
        min_threshold: float | None = 1.0e-3,
        max_threshold: float | None = None,
        threshold_lr_scale: float | None = None,
        target_rate: float | None = None,
        target_rate_eta: float = 0.0,
    ) -> None:
        super().__init__()
        self.T = int(T)
        self.act = TernarySurrogate.apply
        self.thresh = nn.Parameter(torch.tensor(float(thresh)), requires_grad=True)
        self.sp = float(sparsity_eta)
        self.negative_threshold_scale = float(negative_threshold_scale)
        self.activity_eta = float(activity_eta)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.threshold_lr_scale = None if threshold_lr_scale is None else float(threshold_lr_scale)
        self.target_rate = target_rate
        self.target_rate_eta = float(target_rate_eta)
        self.r = 0.0
        self.pos_r = 0.0
        self.neg_r = 0.0
        self.act_value = 0.0
        self.update_value = 0.0

        if base_psn is not None and hasattr(base_psn, "weight") and hasattr(base_psn, "bias"):
            self.weight = nn.Parameter(base_psn.weight.detach().clone())
            self.bias = nn.Parameter(base_psn.bias.detach().clone())
            self.surrogate_function = getattr(base_psn, "surrogate_function", None)
        else:
            self.weight = nn.Parameter(torch.zeros([self.T, self.T]))
            self.bias = nn.Parameter(torch.zeros([self.T, 1]))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.constant_(self.bias, -1.0)
            self.surrogate_function = None

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        h_seq = torch.addmm(self.bias, self.weight, x_seq.flatten(1))
        spike, thre_updates = self.act(h_seq, self.thresh, self.sp, self.negative_threshold_scale)
        self.update_value += thre_updates
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

    def extra_repr(self) -> str:
        return (
            f"T={self.T}, thresh={float(self.thresh.detach().cpu()):.4f}, "
            f"sp={self.sp}, neg_scale={self.negative_threshold_scale}"
        )
