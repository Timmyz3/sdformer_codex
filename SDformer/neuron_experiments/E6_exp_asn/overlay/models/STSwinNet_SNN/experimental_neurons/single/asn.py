from __future__ import annotations

import torch
from torch import nn

from ..base import CandidateNeuron, ensure_time_first


PAPER_TITLE = "Adaptive Spiking Neurons for Vision and Language Modeling"
PAPER_ARXIV = "2604.12365"


class AdaptiveRoundClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membrane: torch.Tensor, alpha: torch.Tensor, D: int, alpha_grad_scale: float):
        rounded = torch.round(membrane)
        lower = alpha
        upper = alpha + float(D)
        ctx.save_for_backward(rounded, lower, upper)
        ctx.alpha_grad_scale = float(alpha_grad_scale)
        return rounded.clamp(min=float(lower.detach()), max=float(upper.detach()))

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        rounded, lower, upper = ctx.saved_tensors
        inside = (rounded >= lower) & (rounded <= upper)
        outside = ~inside

        grad_membrane = grad_output * inside.to(grad_output)
        grad_alpha = grad_output * outside.to(grad_output) * ctx.alpha_grad_scale
        while grad_alpha.ndim > 0:
            grad_alpha = grad_alpha.sum(dim=0)
        return grad_membrane, grad_alpha.view_as(lower), None, None


def adaptive_round_clip(membrane: torch.Tensor, alpha: torch.Tensor, D: int, alpha_grad_scale: float) -> torch.Tensor:
    return AdaptiveRoundClip.apply(membrane, alpha, int(D), float(alpha_grad_scale))


class NASNNode(CandidateNeuron):
    """Normalized Adaptive Spiking Neuron for SDFormerFlow tensors.

    This follows the paper's training-time NASN equations:

    U[t] = H[t-1] + X[t]
    S[t] = clip(round(U[t]), alpha, alpha + D) / N
    H[t] = beta * (U[t] - S[t] * N)
    """

    paper_title = PAPER_TITLE
    paper_arxiv = PAPER_ARXIV

    def __init__(
        self,
        T: int,
        D: int = 4,
        N: int | None = None,
        beta: float = 0.5,
        alpha_init: float = 0.0,
        alpha_grad_scale: float = 1.0,
    ):
        super().__init__()
        if D <= 0:
            raise ValueError("D must be positive")
        if N is None:
            N = D
        if N <= 0:
            raise ValueError("N must be positive")

        self.T = int(T)
        self.D = int(D)
        self.N = int(N)
        self.beta = float(beta)
        self.alpha_grad_scale = float(alpha_grad_scale)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float))
        self.register_memory("H", 0.0)

    def reset_state(self) -> None:
        self.H = 0.0

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(self.H, float):
            self.H = torch.zeros_like(x)
        U = self.H + x
        quantized = adaptive_round_clip(U, self.alpha, self.D, self.alpha_grad_scale)
        S = quantized / float(self.N)
        self.H = self.beta * (U - S * float(self.N))
        return S

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = ensure_time_first(x, self.T)
        self.reset_state()
        outputs = []
        for t in range(T):
            outputs.append(self.single_step_forward(x[t]))
        return torch.stack(outputs, dim=0)


class ASNNode(NASNNode):
    """Unnormalized ASN variant kept for later E6b experiments."""

    def __init__(
        self,
        T: int,
        D: int = 4,
        beta: float = 0.5,
        alpha_init: float = 0.0,
        alpha_grad_scale: float = 1.0,
    ):
        super().__init__(
            T=T,
            D=D,
            N=1,
            beta=beta,
            alpha_init=alpha_init,
            alpha_grad_scale=alpha_grad_scale,
        )
