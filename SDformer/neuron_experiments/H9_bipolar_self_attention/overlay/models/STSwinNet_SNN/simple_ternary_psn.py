"""PSN + Ternary output — no ATLIF threshold adaptation, no gate, no quantile.

Minimalist ternary neuron: keeps PSN parallel temporal mixing, adds
bias-centered ternary output {-theta, 0, +theta} with a learnable theta.
The threshold is updated ONLY by gradient descent — no ATLIF activity-driven
update, no target_rate feedback, no quantile budget.

This is the purest form of PSN + ternary for combinatorial ablation.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class TernarySTE(torch.autograd.Function):
    """Straight-through ternary: sign(h)*θ forward, triangular surrogate backward.

    Forward:  out = sign(h) * theta   ∈ {-θ, 0, +θ}
    Backward: ∂L/∂h ≈ (1 - |h/θ|).clamp(0) * ∂L/∂out   (triangular surrogate)
              ∂L/∂θ = mean(sign(h) * ∂L/∂out)            (sign-weighted)
    """

    @staticmethod
    def forward(ctx, h: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(h, theta)
        return torch.sign(h) * theta

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        h, theta = ctx.saved_tensors
        # Triangular surrogate for h (ATLIF-style)
        grad_h = (1.0 - (h / theta.clamp_min(1e-6)).abs()).clamp_min(0) * grad_output
        # Theta gradient = mean sign * grad_output
        grad_theta = (torch.sign(h) * grad_output).mean()
        return grad_h, grad_theta


def simple_ternary_ste(h_seq: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Apply ternary STE.  See ``TernarySTE`` for gradient formulas."""
    return TernarySTE.apply(h_seq, theta)


class SimpleTernaryPSN(nn.Module):
    """PSN temporal mixer with ternary {-θ, 0, +θ} output.  No ATLIF.

    Parameters
    ----------
    T:
        Number of timesteps (from base_psn or explicit).
    base_psn:
        Baseline PSN module. weight/bias are copied so fine-tuning starts
        from the pre-trained temporal dynamics.
    theta_init:
        Initial threshold magnitude. Output is {-theta, 0, +theta}.
    center_mode:
        ``"bias"`` subtracts the copied PSN bias before thresholding so the
        silent baseline sits at zero.  ``"zero"`` is the raw PSN output.
    """

    def __init__(
        self,
        T: int,
        base_psn: nn.Module | None = None,
        theta_init: float = 1.0,
        center_mode: str = "bias",
    ) -> None:
        super().__init__()
        if center_mode not in {"bias", "zero"}:
            raise ValueError(f"center_mode must be 'bias' or 'zero', got {center_mode!r}")

        self.T = int(T)
        self.center_mode = center_mode

        # --- Copy PSN temporal weights from baseline ---
        if base_psn is not None and hasattr(base_psn, "weight") and hasattr(base_psn, "bias"):
            self.weight = nn.Parameter(base_psn.weight.detach().clone())
            self.bias = nn.Parameter(base_psn.bias.detach().clone())
        else:
            self.weight = nn.Parameter(torch.zeros([self.T, self.T]))
            self.bias = nn.Parameter(torch.zeros([self.T, 1]))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.constant_(self.bias, -1.0)

        # --- Learnable ternary threshold ---
        self.theta = nn.Parameter(torch.tensor(float(theta_init)))

        # --- Bias center (subtract PSN bias so silent baseline = 0) ---
        center = self.bias.detach().clone() if center_mode == "bias" else torch.zeros_like(self.bias.detach())
        self.register_buffer("center", center)

        # --- Activity tracking (read-only, for logging) ---
        self.register_buffer("running_activity", torch.tensor(0.0))
        self.register_buffer("running_pos_rate", torch.tensor(0.0))
        self.register_buffer("running_neg_rate", torch.tensor(0.0))

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # 1. PSN parallel temporal mixing (same as baseline)
        h_seq = torch.addmm(self.bias, self.weight, x_seq.flatten(1))

        # 2. Bias center: remove the learned PSN bias baseline
        if self.center_mode == "bias":
            h_seq = h_seq - self.center.to(device=h_seq.device, dtype=h_seq.dtype)

        # 3. Ternary spike {-theta, 0, +theta} with STE gradient
        out = simple_ternary_ste(h_seq, self.theta)
        out = out.view(x_seq.shape)

        # 4. Logging-only activity tracking (no gradient)
        with torch.no_grad():
            ternary = out / self.theta.clamp_min(1e-12)  # normalized {-1,0,+1}
            activity = (ternary != 0).float().mean()
            pos_rate = (ternary > 0).float().mean()
            neg_rate = (ternary < 0).float().mean()
            self.running_activity.mul_(0.99).add_(activity * 0.01)
            self.running_pos_rate.mul_(0.99).add_(pos_rate * 0.01)
            self.running_neg_rate.mul_(0.99).add_(neg_rate * 0.01)

        return out

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"T={self.T}, theta={float(self.theta.detach().cpu()):.4f}, "
            f"center_mode={self.center_mode}"
        )

    @property
    def activity(self) -> float:
        return float(self.running_activity.item())

    @property
    def pos_rate(self) -> float:
        return float(self.running_pos_rate.item())

    @property
    def neg_rate(self) -> float:
        return float(self.running_neg_rate.item())
