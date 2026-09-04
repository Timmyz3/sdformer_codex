from __future__ import annotations

import torch
from torch import nn

from ..base import CandidateNeuron, ensure_time_first, reset_like


class MultiLevelSpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, threshold: torch.Tensor, levels: int):
        ctx.save_for_backward(input, threshold)
        ctx.levels = levels
        level = torch.floor(input / threshold.clamp_min(1e-6)).clamp(0, levels)
        return level * threshold

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, threshold = ctx.saved_tensors
        levels = ctx.levels
        mask = ((input >= 0.5 * threshold) & (input <= (levels + 0.5) * threshold)).to(grad_output)
        return grad_output * mask, None, None


class LMHNode(CandidateNeuron):
    def __init__(self, T: int, v_threshold: float = 1.0, levels: int = 2, initial_mem: float = 0.0):
        super().__init__()
        self.T = T
        self.levels = int(levels)
        self.v_threshold = nn.Parameter(torch.tensor(float(v_threshold)), requires_grad=False)
        self.initial_mem = float(initial_mem)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.temporal_mask = nn.Parameter(torch.zeros(T, T))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = ensure_time_first(x, self.T)
        mem = reset_like(x[0], self.initial_mem * float(self.v_threshold.item()))
        outputs = []
        temporal_weight = 2.0 * self.temporal_mask.sigmoid() / T
        history_decay = self.alpha.sigmoid() + 0.5
        flat_x = x.flatten(1)
        for t in range(T):
            mixed = torch.matmul(temporal_weight[t], flat_x).view_as(x[0])
            mem = history_decay * mem.detach() + mixed
            spike = MultiLevelSpikeFn.apply(mem, self.v_threshold, self.levels)
            mem = mem - spike.detach()
            outputs.append(spike)
        return torch.stack(outputs, dim=0)
