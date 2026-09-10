"""Dense network evaluator for disjoint-latent full/conditional students.

This preserves the original outer BN and real downstream network. CUDA may
densely compute private Z/Y before deciding; only the separate CPU workload
model represents omitted production. No GPU speed is claimed.
"""
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conditional_adapter import fp32_matmul


class LatentTemporal(nn.Module):
    def __init__(self, arrays):
        super().__init__()
        a = torch.as_tensor(arrays['a']).float()
        b = torch.as_tensor(arrays['temporal_bias']).float().reshape(10)
        mean = torch.as_tensor(arrays['completion_mean']).float()
        covariance = torch.as_tensor(arrays['completion_covariance']).float()
        self.theta = float(arrays['theta_output'])
        self.register_buffer('a', a)
        self.register_buffer('b', b)
        self.register_buffer('bn_scale',torch.as_tensor(arrays['bn_scale']).float())
        self.register_buffer('bn_bias',torch.as_tensor(arrays['bn_bias']).float())
        patterns = torch.arange(1024)[:,None]
        empty = ((patterns>>torch.arange(10))&1).bool()
        remaining = a[None,None]*(~empty)[:,None,None,:]
        with fp32_matmul():
            variance = torch.einsum('gpts,hsu,gptu->gtph',remaining,covariance,remaining).clamp_min(0.)
            radius = float(arrays['gamma'])*variance.sqrt()[:,:,0]
            nominal = mean[None,:,None,:]*(~empty)[:,:,None,None]
            correction = torch.einsum('ts,gsph->gtph',a,nominal)[:,:,0]
        entries = []
        for t in range(10):
            indices = a[t].ne(0).nonzero().flatten()
            compact = torch.arange(1<<len(indices))[:,None]
            bits = (compact>>torch.arange(len(indices)))&1
            selected = (bits*(1<<indices)).sum(-1)
            self.register_buffer('indices_'+str(t),indices)
            self.register_buffer('mean_'+str(t),correction[selected,t])
            self.register_buffer('radius_'+str(t),radius[selected,t])
            entries.append(len(selected))
        self.table_metadata = dict(entries_per_row=entries,FP32_pair_bytes=sum(entries)*96*8,
            scope='train-only private residual mean/radius; indexed by source-empty pattern restricted to each A row; constants and ports still cost resources')

    def forward_groups(self,y,shared_raw,empty,conditional=True,return_details=False,
                       capture_details=False):
        """G,T10,P4,H96; empty[G,T10,P4] from original theta*g source."""
        with fp32_matmul():
            full = torch.einsum('ts,gsph->gtph',self.a,y)+self.b[None,:,None,None]-self.theta
            if not conditional:
                gate = full.ge(0)
                details = dict(accepted=torch.zeros_like(gate),predicted=full)
            else:
                preview_y = shared_raw*self.bn_scale+self.bn_bias
                base = torch.einsum('ts,gsph->gtph',self.a,preview_y)+self.b[None,:,None,None]-self.theta
                offset, radii = [], []
                for t in range(10):
                    indices = getattr(self,'indices_'+str(t))
                    local = empty[:,indices].permute(0,2,1).long()
                    code = (local*(1<<torch.arange(len(indices),device=y.device))).sum(-1)
                    offset.append(getattr(self,'mean_'+str(t))[code])
                    radii.append(getattr(self,'radius_'+str(t))[code])
                predicted = base+torch.stack(offset,1)
                radius = torch.stack(radii,1)
                accept = predicted.abs()>=radius
                gate = torch.where(accept,predicted.ge(0),full.ge(0))
                details = dict(accepted=accept,predicted=predicted,radius=radius)
            if capture_details:
                # References to values already evaluated by this forward;
                # no additional decision or future-dependent permission.
                details['full_margin'] = full
                if conditional:
                    details['shared_margin'] = base
        return (gate,details) if return_details else gate


class LatentPair:
    def __init__(self,arrays,device,conditional=True,capture_callback=None):
        self.temporal = LatentTemporal(arrays).to(device).eval()
        self.u = torch.as_tensor(arrays['u'],device=device).float().T.reshape(-1,96,3,3)
        self.v = torch.as_tensor(arrays['v'],device=device).float().T[:,:,None,None]
        self.shared_rank = int(arrays['shared_rank'])
        self.source_theta = float(arrays['theta_source'])
        self.conditional = conditional
        self.shared_raw = self.empty = None
        self.last_counts = None
        self.capture_callback = capture_callback

    def conv_forward(self,x):
        if x.shape[:3] != (10,1,96):
            raise ValueError('Expected patch source T10,B1,C96.')
        z = F.conv2d(x.flatten(0,1),self.u,padding=1)
        shared = F.conv2d(z[:,:self.shared_rank],self.v[:,:self.shared_rank])
        tail = F.conv2d(z[:,self.shared_rank:],self.v[:,self.shared_rank:])
        self.shared_raw = shared.reshape(10,1,96,*shared.shape[-2:])
        activity = x.ne(0).any(2).float()
        self.empty = F.max_pool2d(activity,3,stride=1,padding=1).eq(0)[:,0]
        callback = self.capture_callback
        if callback is not None and getattr(callback,'active',True):
            callback('conv',z=z,shared_raw=shared,tail_raw=tail,
                     source_empty=self.empty)
        return (shared+tail).reshape_as(self.shared_raw)

    def neuron_forward(self,y):
        if self.shared_raw is None:
            raise RuntimeError('The source/preview belongs to this forward only.')
        t,b,c,height,width = y.shape
        if (t,b,c)!=(10,1,96) or width%4:
            raise ValueError('Expected native horizontal P4 complete-T patch output.')
        out = torch.empty_like(y)
        accepted=total=0
        callback = self.capture_callback
        capturing = callback is not None and getattr(callback,'active',True)
        for first in range(0,height,8):
            last=min(first+8,height)
            def group(values):
                return values[:,0,:,first:last].reshape(10,96,last-first,width//4,4).permute(2,3,0,4,1).reshape(-1,10,4,96)
            empty=self.empty[:,first:last].reshape(10,last-first,width//4,4).permute(1,2,0,3).reshape(-1,10,4)
            gate,details=self.temporal.forward_groups(group(y),group(self.shared_raw),empty,
                conditional=self.conditional,return_details=True,capture_details=capturing)
            if capturing:
                callback('neuron_groups',first=first,last=last,height=height,width=width,
                         empty=empty,details=details,conditional=self.conditional)
            value=(gate.to(y.dtype)*self.temporal.theta).reshape(last-first,width//4,10,4,96)
            out[:,0,:,first:last]=value.permute(2,4,0,1,3).reshape(10,96,last-first,width)
            total+=gate.numel();accepted+=int(details['accepted'].sum())
        self.last_counts=dict(gates=total,accepted=accepted,failed=total-accepted,
            numeric='dense two-factor FP32 network reference, no GPU work saving claimed')
        self.shared_raw=self.empty=None
        return out


def install_latent_factor(conv1,neuron,parameter_file,conditional=True,capture_callback=None):
    """Return pair/original forwards. Call on the real Conv1.0 and sn2 leaf."""
    with np.load(parameter_file) as data:
        arrays={key:data[key].copy() for key in data.files}
    if abs(float(neuron.thresh)-float(arrays['theta_output']))>1e-7:
        raise ValueError('Output theta amplitude differs from the saved student.')
    pair=LatentPair(arrays,conv1.weight.device,conditional=conditional,
                    capture_callback=capture_callback)
    originals=dict(conv1_forward=conv1.forward,neuron_forward=neuron.forward)
    conv1.forward,neuron.forward=pair.conv_forward,pair.neuron_forward
    return pair,originals
