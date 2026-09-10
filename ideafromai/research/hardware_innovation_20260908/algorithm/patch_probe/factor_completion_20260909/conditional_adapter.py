"""Dense GPU/CPU numerical adapter for the saved factor completion policy.

Precompiled source-empty radius lookup; decisions read prefix Y only. Full
future Y may be densely produced by the numerical Conv reference and is used
only for rejected decisions. This file does not claim GPU sparse speedup.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from train_factors import FactorConv1


@contextmanager
def fp32_matmul():
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


class ConditionalTemporal(nn.Module):
    def __init__(self, arrays):
        super().__init__()
        a = torch.as_tensor(arrays['a']).float()
        b = torch.as_tensor(arrays['temporal_bias']).float().reshape(10)
        mean = torch.as_tensor(arrays['completion_mean']).float()
        covariance = torch.as_tensor(arrays['completion_covariance']).float()
        prefix = torch.zeros(10,dtype=torch.bool)
        prefix[torch.as_tensor(arrays['prefix']).long()] = True
        tail = (~prefix).nonzero().flatten()
        if len(tail)>7:
            raise ValueError('This saved one-shot policy has at most seven uncomputed columns.')
        gamma = float(arrays['gamma'])
        # code bit i means source-empty at tail[i], not a future Y predicate.
        patterns = torch.arange(1<<len(tail))[:,None]
        empty = torch.zeros(len(patterns),10,dtype=torch.bool)
        empty[:,tail] = ((patterns>>torch.arange(len(tail)))&1).bool()
        known = empty|prefix[None,:]
        remaining = a[None,None,:,:]*(~known)[:,None,None,:]
        with fp32_matmul():
            variance = torch.einsum('gpts,hsu,gptu->gtph',remaining,covariance,remaining).clamp_min(0)
        radius = gamma*variance.sqrt()[:,:,0,:]  # [128,t,H]
        mask = torch.as_tensor(arrays['masks']).bool().repeat_interleave(int(arrays['latent_tile']),1)
        v_support = torch.as_tensor(arrays['v']).ne(0)
        live_h = (mask.float()@v_support.float()).gt(0)
        self.theta = float(arrays['theta_output'])
        bn_bias = torch.as_tensor(arrays['bn_bias']).float()
        constant = a.sum(1)[:,None]*bn_bias[None,:]+b[:,None]-self.theta
        for name,value in dict(a=a,b=b,mean=mean,prefix=prefix,tail=tail,
            radius=radius,live_h=live_h,bn_bias=bn_bias,constant_margin=constant).items():
            self.register_buffer(name,value)
        self.table_metadata = dict(entries=len(patterns),shape=list(radius.shape),
            numerical_reference_fp32_bytes=radius.numel()*4,
            distinct_radius_vectors_per_t=[int(torch.unique(radius[:,t],dim=0).shape[0]) for t in range(10)],
            interpretation='GPU reference lookup; identical rows may compile to much smaller constant tables, no SRAM admission')

    def forward_groups(self,y,empty,regions,return_details=False):
        """Y[G,T10,P4,H96], empty[G,T10,P4] from original Conv source."""
        with fp32_matmul():
            nominal = torch.where(self.prefix[None,:,None,None],y,
                torch.where(empty[...,None],self.bn_bias,self.mean[None,:,None,:]))
            predicted = torch.einsum('ts,gsph->gtph',self.a,nominal)+self.b[None,:,None,None]-self.theta
            tail_empty = empty[:,self.tail,:].permute(0,2,1).long()
            code = (tail_empty*(1<<torch.arange(len(self.tail),device=y.device))).sum(-1)
            radius = self.radius[code].permute(0,2,1,3)
            live = self.live_h[regions][:,None,None,:]
            predicted = torch.where(live,predicted,self.constant_margin[None,:,None,:])
            radius = torch.where(live,radius,torch.zeros_like(radius))
            accept = predicted.abs()>=radius
            # The batched reference computes full PSN arithmetic in groups
            # with any failed gate. Values for accepted gates are discarded;
            # they cannot influence acceptance or the output of accepted gates.
            gate = predicted.ge(0)
            failed_groups = (~accept).any((1,2,3)).nonzero().flatten()
            if len(failed_groups):
                yf = y[failed_groups]
                full = torch.einsum('ts,gsph->gtph',self.a,yf)+self.b[None,:,None,None]-self.theta
                gate[failed_groups] = torch.where(accept[failed_groups],gate[failed_groups],full.ge(0))
        if return_details:
            return gate,dict(accepted=accept,predicted=predicted,radius=radius,empty_code=code)
        return gate


class ConditionalFactorPair:
    """Two hooks bound to the original Conv1 and its full-T10 neuron."""
    def __init__(self,arrays,device):
        self.conv = FactorConv1(arrays).to(device).eval()
        self.temporal = ConditionalTemporal(arrays).to(device).eval()
        self.empty = None
        self.last_counts = None
        self.source_theta = float(arrays['theta_source'])

    def conv_forward(self,x):
        if x.ndim!=5 or x.shape[:3]!=(10,1,96):
            raise ValueError('Expected complete patch source T10,B1,C96.')
        # Ordinary source-empty directory, before either factor. Pooling is
        # only a dense reference for the existing 3x3 source OR operation.
        nonzero = x.ne(0).any(2).float().reshape(10,1,*x.shape[-2:])
        self.empty = F.max_pool2d(nonzero,3,stride=1,padding=1).eq(0)[:,0]
        return self.conv(x)

    def neuron_forward(self,y):
        if self.empty is None:
            raise RuntimeError('Conditional neuron needs this frame\'s Conv1 source directory.')
        t,b,c,height,width = y.shape
        if (t,b,c)!=(10,1,96) or width%4:
            raise ValueError('Expected full T10 and native horizontal P4 groups.')
        out = torch.empty_like(y)
        accepted = total = 0
        for first in range(0,height,8):
            last = min(first+8,height)
            local = y[:,0,:,first:last].reshape(10,96,last-first,width//4,4)
            local = local.permute(2,3,0,4,1).reshape(-1,10,4,96)
            empty = self.empty[:,first:last].reshape(10,last-first,width//4,4)
            empty = empty.permute(1,2,0,3).reshape(-1,10,4)
            region = (torch.arange(first,last,device=y.device)*len(self.temporal.live_h)//height)
            region = region[:,None].expand(-1,width//4).reshape(-1)
            gate,details = self.temporal.forward_groups(local,empty,region,return_details=True)
            value = (gate.to(y.dtype)*self.temporal.theta).reshape(last-first,width//4,10,4,96)
            out[:,0,:,first:last] = value.permute(2,4,0,1,3).reshape(10,96,last-first,width)
            accepted += int(details['accepted'].sum())
            total += gate.numel()
        self.last_counts = dict(gates=total,accepted=accepted,failed=total-accepted,
            scope='network function decisions; dense reference Conv/fallback arithmetic, not cycles')
        self.empty = None
        return out


def install_conditional_factor(conv1,neuron,parameter_file):
    """Return (pair, originals); outer norm1/Conv2/BN2/shortcut are untouched."""
    with np.load(Path(parameter_file)) as data:
        arrays = {k:data[k].copy() for k in data.files}
    if abs(float(neuron.thresh)-float(arrays['theta_output']))>1e-7:
        raise ValueError('The saved theta amplitude must match the original neuron.')
    pair = ConditionalFactorPair(arrays,conv1.weight.device)
    originals = dict(conv1_forward=conv1.forward,neuron_forward=neuron.forward)
    conv1.forward = pair.conv_forward
    neuron.forward = pair.neuron_forward
    return pair,originals
