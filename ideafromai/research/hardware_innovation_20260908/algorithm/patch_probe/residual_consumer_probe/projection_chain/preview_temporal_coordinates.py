"""Ordinary full-T10 preview PSN moved before the Conv1 output factor.

Install on the existing frozen TrainableLatentPair/LatentPair in full mode.
The actual source theta*g still enters its saved dequantized U8 producer.
Only exactly dead latent columns are removed. Conv1 returns V(A2 Z), and the
real fixed outer norm1 still runs: D1 V(A2 Z)+beta1. The sn2 replacement adds
(A2*1-1)*beta1, then the existing pair.temporal.b and its real theta gate.
This yields the full noncausal A2(D1 VZ+beta1)+b in real arithmetic.

The current pair owns the effective temporal bias/center convention; do not
substitute the overridden native leaf's old A/b/center. FP32 reassociation,
active-rank compaction, sparse dot order and live-BN offset compilation require
new AEE. This helper claims neither native bit equality nor GPU/hardware speed.
It changes no residual, Conv2, PED, BN callback or training function.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from adapter import fp32_matmul
from flow_backward_probe import HardThetaGate


def sparse_time_mix(matrix, value, supports):
    """All ten source times are available; exact static nonzeros only."""
    flat = value.reshape(10, -1)
    with fp32_matmul():
        rows = []
        for t, indices in enumerate(supports):
            rows.append((matrix[t, indices][None]@flat.index_select(0, indices))[0]
                        if indices.numel() else torch.zeros_like(flat[0]))
        return torch.stack(rows).reshape_as(value)


class PreviewTemporalForward:
    def __init__(self, pair, conv1, neuron, norm1_bn, fp32_channel=False):
        if pair.conditional:
            raise ValueError('This ordinary compilation is for the current full/preview-only function, not statistical tail permission.')
        if norm1_bn.training or not norm1_bn.track_running_stats:
            raise ValueError('Moving A across norm1 requires the actual fixed eval BN; dynamic batch statistics would change under the move.')
        self.pair, self.conv1, self.neuron, self.bn = pair, conv1, neuron, norm1_bn
        self.fp32_channel = fp32_channel
        self.original = dict(conv1=conv1.forward, neuron=neuron.forward)
        self.A = pair.temporal.a.detach().float()
        self.bias = pair.temporal.b.detach().float()
        self.theta = float(pair.temporal.theta)
        u, v = pair.u.detach(), pair.v.detach()
        live = u.flatten(1).ne(0).any(1) & v[:, :, 0, 0].ne(0).any(0)
        self.indices = live.nonzero().flatten()
        self.u = u.index_select(0, self.indices)
        self.v = v.index_select(1, self.indices)
        self.supports = [row.ne(0).nonzero().flatten() for row in self.A]
        with torch.no_grad(), fp32_matmul():
            self.gain = norm1_bn.weight.detach()/torch.sqrt(norm1_bn.running_var+norm1_bn.eps)
            self.beta = norm1_bn.bias.detach()-self.gain*norm1_bn.running_mean
            self.row_sum = self.A.sum(1)
            self.bn_correction = (self.row_sum[:, None]-1)*self.beta[None]
        self.frames, self.frame = [], {}
        nonzero = int(self.A.ne(0).sum())
        self.metadata = dict(mode='preview_time_before_V', active_rank=int(self.indices.numel()),
            saved_latent_slots=int(u.shape[0]), active_indices=self.indices.cpu().tolist(),
            A_nonzero=nonzero, A_rank=int(torch.linalg.matrix_rank(self.A.double())), T=10,
            theta_source=float(pair.source_theta), theta_output=self.theta,
            source='Actual theta*g input and saved dequantized U8 coefficients; no extra theta multiplication, code conversion or source change.',
            bias_center='Use exactly the current adapter\'s effective pair.temporal.b. Its old native leaf A/b/center is overridden and is not applied again.',
            execution='Z=conv(source,U_live); AZ is the full noncausal sparse A2 dot; raw=conv(AZ,V_live); original fixed norm1(raw); sn2 margin=(norm1(raw)+(A2*1-1)*beta1)+pair.b-theta; HardThetaGate.',
            temporal_numeric='A2 sparse rows multiply FP32 Z with TF32 disabled. No causal restriction or approximate zero threshold. Intermediate products/sums and BN correction are FP32.',
            fp32_channel=fp32_channel,
            channel_numeric=('Both Conv1-factor convolutions in this helper disable cuDNN TF32 and preserve other flags; the original BN and helper-external operators retain their policies.'
                if fp32_channel else 'Both Conv1-factor convolutions retain the original cuDNN TF32 policy; the original fixed BN is called unchanged.'),
            temporal_products_per_spatial_position=dict(before=nonzero*int(v.shape[0]),
                after=nonzero*int(self.indices.numel()),
                scope='Both sides grant static A sparsity; counts are arithmetic opportunity, not GPU kernel work or cycles.'),
            correction_cost=dict(FP32_constants_bytes=self.bn_correction.numel()*4,
                extra_additions_per_spatial_position=self.bn_correction.numel(),
                scope='This literal forward retains native BN then adds the compiled10xH correction. Temporal bias addition and theta subtraction also execute; no free correction claim.'),
            external_calls='Original fixed norm1, Conv2, BN2, identity/shortcut, PED and real successors still execute. No intermediate tensor removal is claimed.',
            ranges='Per-frame actual per-T max_abs of Z, AZ, VAZ, fixed norm1 output and sn2 margin; no full tensor trace.',
            claim='Ordinary linear compilation baseline available to every axis. New FP32 function needs its own AEE; no independent innovation or saved-cycle assertion.')
        conv1.forward, neuron.forward = self.conv_forward, self.neuron_forward

    def channel_conv(self, *args, **kwargs):
        if not self.fp32_channel:
            return F.conv2d(*args, **kwargs)
        with torch.backends.cudnn.flags(enabled=None, benchmark=None, benchmark_limit=None,
                                      deterministic=None, allow_tf32=False):
            return F.conv2d(*args, **kwargs)

    def record(self, key, value):
        self.frame[key] = value.detach().reshape(10, -1).abs().amax(1).cpu().tolist()

    def conv_forward(self, x):
        self.frame = {}
        z = self.channel_conv(x.flatten(0, 1), self.u, None,
            self.conv1.stride, self.conv1.padding, self.conv1.dilation)
        az = sparse_time_mix(self.A, z, self.supports)
        raw = self.channel_conv(az, self.v)
        self.record('Z', z)
        self.record('A2_Z', az)
        self.record('V1_A2_Z', raw)
        # Outer norm1 receives this time-mixed raw value and runs normally.
        return raw.reshape(x.shape[0], x.shape[1], raw.shape[1], *raw.shape[-2:])

    def neuron_forward(self, y):
        with fp32_matmul():
            coordinates = y+self.bn_correction[:, None, :, None, None]
            margin = coordinates+self.bias[:, None, None, None, None]-self.theta
            out = HardThetaGate.apply(margin, self.theta)
        self.record('fixed_norm1_of_V1_A2_Z', y)
        self.record('sn2_margin', margin)
        self.frames.append(self.frame)
        self.pair.last_counts = dict(gates=out.numel(), mode='full hard gate with ordinary A2-before-V1 compilation')
        self.pair.shared_raw = self.pair.empty = None
        return out

    def export_constants(self):
        values = dict(U_live=self.u, V_live=self.v, active_indices=self.indices, A2=self.A,
            temporal_effective_bias=self.bias, norm1_gain_fp32=self.gain,
            norm1_offset_fp32=self.beta, A2_row_sum=self.row_sum,
            norm1_temporal_correction=self.bn_correction,
            norm1_gamma=self.bn.weight.detach(), norm1_beta=self.bn.bias.detach(),
            norm1_mean=self.bn.running_mean, norm1_var=self.bn.running_var,
            norm1_eps=torch.as_tensor(self.bn.eps), theta_output=torch.as_tensor(self.theta),
            theta_source=torch.as_tensor(self.pair.source_theta))
        return {name: value.detach().cpu().numpy() for name, value in values.items()}

    def range_report(self, names):
        return dict(scope=self.metadata['ranges'], frames=[dict(file=name, per_T_max_abs=row)
                    for name, row in zip(names, self.frames)])

    def restore(self):
        self.conv1.forward, self.neuron.forward = self.original['conv1'], self.original['neuron']
        self.pair.shared_raw = self.pair.empty = None
