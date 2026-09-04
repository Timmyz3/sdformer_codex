"""LMHT neuron copied from LMHT_SNN with a thin SDFormerFlow wrapper.

Core functions/classes mirror:
`LMHT_SNN/modules.py`

Only `LMHNode` is experiment-local glue that maps SDFormerFlow's neuron factory
arguments to the upstream LMHT constructor.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.autograd import Function

from ..base import CandidateNeuron


class OneLevelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, th):
        out = (input >= 1.0 * th).float() * th
        input = ((input.detach() >= 0.5 * th) * (input.detach() <= 1.5 * th)).float()
        ctx.save_for_backward(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (tmp,) = ctx.saved_tensors
        grad_input = grad_output * tmp
        return grad_input, None


class TwoLevelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, th):
        out2 = (input >= 2.0 * th).float()
        out1 = (input >= 1.0 * th).float() * (1.0 - out2)
        out = out1 * th + out2 * 2.0 * th
        input = ((input.detach() >= 0.5 * th) * (input.detach() <= 2.5 * th)).float()
        ctx.save_for_backward(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (tmp,) = ctx.saved_tensors
        grad_input = grad_output * tmp
        return grad_input, None


class FourLevelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, th):
        out4 = (input >= 4.0 * th).float()
        out3 = (input >= 3.0 * th).float() * (1.0 - out4)
        out2 = (input >= 2.0 * th).float() * (1.0 - out4) * (1.0 - out3)
        out1 = (input >= 1.0 * th).float() * (1.0 - out4) * (1.0 - out3) * (1.0 - out2)
        out = out1 * th + out2 * 2.0 * th + out3 * 3.0 * th + out4 * 4.0 * th
        input = ((input.detach() >= 0.5 * th) * (input.detach() <= 4.5 * th)).float()
        ctx.save_for_backward(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (tmp,) = ctx.saved_tensors
        grad_input = grad_output * tmp
        return grad_input, None


class LMHTNeuron(nn.Module):
    def __init__(self, L: int, T=2, th=1.0, inital_mem=0.0):
        super(LMHTNeuron, self).__init__()
        self.v_threshold = nn.Parameter(torch.tensor([th]), requires_grad=False)
        self.v = None
        self.inital_mem = inital_mem * th

        if L == 2:
            self.act = TwoLevelFunction.apply
        elif L == 4:
            self.act = FourLevelFunction.apply
        else:
            raise ValueError(f"LMHTNeuron follows upstream LMHT_SNN and supports L in {{2, 4}}, got {L}")

        self.alpha = nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.mask = nn.Parameter(torch.zeros((T, T, 1, 1, 1, 1)), requires_grad=True)
        self.mask_linear = nn.Parameter(torch.zeros((T, T, 1, 1, 1)), requires_grad=True)
        self.T = T
        self.scale = 1.0

    def forward(self, x):
        self.v = torch.ones_like(x[0]) * self.inital_mem
        x = x * self.scale

        if len(x.shape) == 5:
            self.core = self.mask
        else:
            self.core = self.mask_linear

        spike_pot = []
        for t in range(self.T):
            self.v = (self.alpha.sigmoid() + 0.5) * self.v.detach() + (
                (2 * self.core[t].sigmoid() / x.shape[0]) * x
            ).sum(dim=0)
            output = self.act(self.v, self.v_threshold)
            self.v -= output.detach()
            spike_pot.append(output)

        return torch.stack(spike_pot, dim=0)


class LMHT_Inference_Neuron(nn.Module):
    def __init__(self, L, alpha, mask, mask_linear, T=4, th=1.0, inital_mem=0.0):
        super(LMHT_Inference_Neuron, self).__init__()
        self.v_threshold = nn.Parameter(torch.tensor([th]), requires_grad=False)
        self.v = None
        self.inital_mem = inital_mem * th

        self.alpha = nn.Parameter(torch.ones(T), requires_grad=False)
        self.mask = nn.Parameter(torch.zeros((T, T, 1, 1, 1, 1)), requires_grad=False)
        self.mask_linear = nn.Parameter(torch.zeros((T, T, 1, 1)), requires_grad=False)
        self.T = T
        self.scale = 1.0

        self.reparameterization(L, alpha, mask, mask_linear)

    def reparameterization(self, L, alpha, mask, mask_linear):
        for t in range(0, self.T, L):
            self.alpha[t] = alpha.sigmoid().item() + 0.5
            for j in range(0, self.T, L):
                self.mask[t : t + L, j : j + L] = 2.0 * mask[t // L, j // L].sigmoid().item() / self.T
                self.mask_linear[t : t + L, j : j + L] = (
                    2.0 * mask_linear[t // L, j // L].sigmoid().item() / self.T
                )

    def forward(self, x):
        self.v = torch.ones_like(x[0]) * self.inital_mem
        x = x * self.scale

        if len(x.shape) == 5:
            self.core = self.mask
        else:
            self.core = self.mask_linear

        spike_pot = []
        for t in range(self.T):
            self.v = self.alpha[t] * self.v + (self.core[t] * x).sum(dim=0)
            output = (self.v >= self.v_threshold) * self.v_threshold
            self.v -= output
            spike_pot.append(output)

        return torch.stack(spike_pot, dim=0)


class FloorLayer(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class IFNeuron(nn.Module):
    def __init__(self, T=2, th=1.0, inital_mem=0.0):
        super(IFNeuron, self).__init__()
        self.v_threshold = nn.Parameter(torch.tensor([th]), requires_grad=False)
        self.v = None
        self.inital_mem = inital_mem * th
        self.T = T
        self.act = OneLevelFunction.apply

    def forward(self, x):
        self.v = torch.ones_like(x[0]) * self.inital_mem
        spike_pot = []
        for t in range(self.T):
            self.v = self.v + x[t]
            output = self.act(self.v, self.v_threshold)
            self.v -= output
            spike_pot.append(output)

        return torch.stack(spike_pot, dim=0)


qcfs = FloorLayer.apply


class QCFS(nn.Module):
    def __init__(self, up=1.0, t=4):
        super().__init__()
        self.thresh = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t

    def forward(self, x):
        x = x / self.thresh
        x = qcfs(x * self.t + 0.5) / self.t
        x = torch.clamp(x, 0, 1)
        x = x * self.thresh
        return x


class LMHNode(CandidateNeuron, LMHTNeuron):
    def __init__(
        self,
        T: int,
        v_threshold: float = 1.0,
        levels: int = 2,
        initial_mem: float = 0.0,
    ):
        LMHTNeuron.__init__(
            self,
            L=int(levels),
            T=T,
            th=float(v_threshold),
            inital_mem=float(initial_mem),
        )
