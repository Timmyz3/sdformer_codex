"""Shared utilities for experimental SDFormerFlow neurons."""

from __future__ import annotations

import torch
from torch import nn
from spikingjelly.activation_based import base as sj_base


def ensure_time_first(x: torch.Tensor, T: int | None = None) -> int:
    if x.ndim < 2:
        raise ValueError(f"expected [T, B, ...], got {tuple(x.shape)}")
    if T is not None and x.shape[0] != T:
        raise ValueError(f"expected {T} timesteps, got {x.shape[0]}")
    return x.shape[0]


def reset_like(x: torch.Tensor, value: float = 0.0) -> torch.Tensor:
    return torch.full_like(x, value)


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, threshold: torch.Tensor, lens: float):
        ctx.save_for_backward(input, threshold)
        ctx.lens = lens
        return (input >= threshold).to(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, threshold = ctx.saved_tensors
        lens = ctx.lens
        scaled = (input - threshold) / threshold.clamp_min(1e-6)
        grad = (1.0 - scaled.abs() / lens).clamp_min(0.0)
        grad_input = grad_output * grad
        grad_threshold = -(grad_output * grad).sum().view_as(threshold)
        return grad_input, grad_threshold, None


class TernarySpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, threshold: torch.Tensor):
        ctx.save_for_backward(input, threshold)
        out = torch.zeros_like(input)
        out = torch.where(input >= threshold, torch.ones_like(out), out)
        out = torch.where(input <= -threshold, -torch.ones_like(out), out)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, threshold = ctx.saved_tensors
        scaled = input / threshold.clamp_min(1e-6)
        grad = (1.0 - scaled.abs()).clamp_min(0.0)
        return grad_output * grad, None


class CandidateNeuron(sj_base.MemoryModule):
    backend = "torch"

    @property
    def supported_backends(self):
        return ("torch",)

    def reset_state(self) -> None:
        pass
