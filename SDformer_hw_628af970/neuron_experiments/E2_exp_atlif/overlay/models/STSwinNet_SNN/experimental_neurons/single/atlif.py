"""ATLIF copied from Activity-Pruning-SNN with a thin SDFormerFlow wrapper.

Core functions/classes below intentionally mirror:
`Activity-Pruning-SNN/models/submodules/layers.py`

Only `ATLIFNode` is experiment-local glue that maps SDFormerFlow's neuron
factory argument names to the upstream ATLIF constructor.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..base import CandidateNeuron


def zif_backward(x, thre):
    result = (1. - (x / thre).abs()).clamp_min(0)
    return result


class ZIF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, gama):
        out = (input >= 0).float()
        L = torch.tensor([gama])
        ctx.save_for_backward(input, out, L)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (input, out, others) = ctx.saved_tensors
        gama = others[0].item()
        grad_input = grad_output
        tmp = (1 / gama) * (1 / gama) * ((gama - input.abs()).clamp(min=0))
        grad_input = grad_input * tmp
        return grad_input, None


class Surrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thre, sp):
        out = (input >= thre).float()
        thre_updates = (sp * zif_backward(input - thre, thre) * out).sum(0).mean().item()
        ctx.save_for_backward(input, thre)
        return out * thre, thre_updates

    @staticmethod
    def backward(ctx, grad_input, dummy):
        (input, thre) = ctx.saved_tensors
        input = (input - thre) / thre
        tmp = (1.0 - input.abs()).clamp(min=0)
        grad_input = grad_input * tmp
        grad_thre = -(grad_input * tmp).mean()
        return grad_input, grad_thre, None


class ATLIF(nn.Module):
    def __init__(self, T, thresh=1.0, tau=1., gama=1.0, sp=0.):
        super(ATLIF, self).__init__()
        self.act = Surrogate.apply
        self.thresh = nn.Parameter(torch.tensor(thresh), requires_grad=True)
        self.tau = tau
        self.gama = gama
        self.T = T
        self.r = 0
        self.act_value = 0.
        self.sp = sp
        self.update_value = 0.
        self.s = 0.

    def forward(self, x):
        mem = 0.
        spike_pot = []
        for t in range(self.T):
            mem = mem * self.tau + x[t, ...]
            spike, thre_updates = self.act(mem, self.thresh, self.sp)
            mem = (1. - spike / self.thresh.detach()) * mem
            spike_pot.append(spike)
            self.update_value += (thre_updates / self.T)
        x = torch.stack(spike_pot, dim=0)
        self.r = (x / self.thresh.data).mean().item()
        self.s = (x.mean(1) / self.thresh.data).sum().item()
        self.act_value = x.reshape(x.size(0), -1).mean(1).sum()
        return x


class ATLIFNode(CandidateNeuron, ATLIF):
    def __init__(
        self,
        T: int,
        v_threshold: float = 1.0,
        tau: float = 1.0,
        lens: float = 1.0,
        sparsity_eta: float = 0.0,
    ):
        ATLIF.__init__(
            self,
            T=T,
            thresh=v_threshold,
            tau=tau,
            gama=lens,
            sp=sparsity_eta,
        )
        self.firing_rate = 0.0

    def forward(self, x):
        out = ATLIF.forward(self, x)
        self.firing_rate = self.r
        return out

