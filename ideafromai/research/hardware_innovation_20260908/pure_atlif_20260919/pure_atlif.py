"""Isolated recurrent AT-LIF, without PSN's T-by-T weight or time bias.

Recurrence/reset and surrogate follow Activity-Pruning-SNN:
https://github.com/putshua/Activity-Pruning-SNN/blob/main/models/submodules/layers.py
The caller supplies its original official surrogate. Arithmetic is FP32 here;
tau is a retention multiplier (not SpikingJelly's membrane time constant).
"""
import torch
from torch import nn


class RecurrentATLIF(nn.Module):
    def __init__(self, T, threshold, act, retention=1.0):
        super().__init__()
        self.T = int(T)
        self.thresh = nn.Parameter(threshold.detach().float().clone())
        self.act = act
        self.tau = float(retention)
        self.sp = 0.0
        self.output_mode = 'binary'
        self.threshold_mode = 'official_atlif'
        self.r = self.pos_r = self.neg_r = 0.0
        self.act_value = self.update_value = 0.0

    def forward(self, x):
        assert x.shape[0] == self.T
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.float()
            mem = 0.0
            spikes = []
            for t in range(self.T):
                mem = self.tau * mem + x[t]
                spike, update = self.act(mem, self.thresh, self.sp)
                # Forward hard reset. Preserve the upstream reset gradient:
                # only the denominator is detached, not spike or membrane.
                mem = (1.0 - spike / self.thresh.detach()) * mem
                spikes.append(spike)
                self.update_value += update / self.T
            out = torch.stack(spikes)
            self.r = self.pos_r = float((out.detach() / self.thresh.detach()).mean())
            self.act_value = out.reshape(self.T, -1).mean(1).sum()
            return out


def replace(model, scope, retention):
    from models.STSwinNet_SNN.atlif_ternary_psn.atlif_ternary_psn import ATLIFTernaryPSN, OfficialATLIFSurrogate
    replaced = []
    for name, old in list(model.named_modules()):
        if not isinstance(old, ATLIFTernaryPSN):
            continue
        if scope == 't10' and old.T != 10:
            continue
        parent_name, _, leaf = name.rpartition('.')
        new = RecurrentATLIF(old.T, old.thresh, OfficialATLIFSurrogate.apply, retention)
        new.train(old.training)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, new)
        replaced.append(dict(name=name, T=old.T,
                             removed_parameters=old.weight.numel() + old.bias.numel(),
                             consumer_live=not name.endswith(('.sn2_q', '.attn_sn'))))
    return replaced
