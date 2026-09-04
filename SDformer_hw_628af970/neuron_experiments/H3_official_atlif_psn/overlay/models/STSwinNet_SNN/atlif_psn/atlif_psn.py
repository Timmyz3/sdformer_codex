"""PSN temporal mixer with official ATLIF threshold-update mechanics."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def zif_backward(x: torch.Tensor, thre: torch.Tensor) -> torch.Tensor:
    return (1.0 - (x / thre).abs()).clamp_min(0)


class Surrogate(torch.autograd.Function):
    """Official Activity-Pruning-SNN ATLIF surrogate."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, thre: torch.Tensor, sp: float):
        out = (input >= thre).float()
        thre_updates = (sp * zif_backward(input - thre, thre) * out).sum(0).mean().item()
        ctx.save_for_backward(input, thre)
        return out * thre, thre_updates

    @staticmethod
    def backward(ctx, grad_input: torch.Tensor, _dummy):
        input, thre = ctx.saved_tensors
        normalized = (input - thre) / thre
        tmp = (1.0 - normalized.abs()).clamp(min=0)
        grad_input = grad_input * tmp
        grad_thre = -(grad_input * tmp).mean()
        return grad_input, grad_thre, None


class ATLIFPSN(nn.Module):
    """PSN-compatible neuron with official ATLIF threshold update state.

    The PSN matrix keeps SDFormerFlow's temporal mixing. The spike generation,
    output scale, activity statistics, and threshold update accumulator mirror
    Activity-Pruning-SNN ATLIF.
    """

    official_atlif_source = "https://github.com/putshua/Activity-Pruning-SNN"

    def __init__(
        self,
        T: int,
        base_psn: nn.Module | None = None,
        thresh: float = 1.0,
        sparsity_eta: float = 0.0,
    ) -> None:
        super().__init__()
        self.T = int(T)
        self.act = Surrogate.apply
        self.thresh = nn.Parameter(torch.tensor(float(thresh)), requires_grad=True)
        self.sp = float(sparsity_eta)
        self.r = 0.0
        self.s = 0.0
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
        spike, thre_updates = self.act(h_seq, self.thresh, self.sp)
        self.update_value += thre_updates
        out = spike.view(x_seq.shape)
        with torch.no_grad():
            thresh = self.thresh.detach().clamp_min(1e-12)
            normalized = out / thresh
            self.r = normalized.mean().item()
            self.s = normalized.mean(1).sum().item()
        self.act_value = out.reshape(out.size(0), -1).mean(1).sum()
        return out

    def extra_repr(self) -> str:
        return f"T={self.T}, thresh={float(self.thresh.detach().cpu()):.4f}, sp={self.sp}"
