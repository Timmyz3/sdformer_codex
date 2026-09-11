"""Exact fixed forward with STE only for the frozen-grid Conv2 U/F recovery.

Source, preview, thresholds, PED and scales stay fixed. Integer coefficients
and each completed signed24 write have their original hard forward. Only the
backward uses clipped identity through RNE and a triangle at the PED gate.
"""
import torch
import torch.nn.functional as F
from fixed_temporal_coordinates import FixedTemporalForward, STATE_MIN, STATE_MAX, STATE_SCALE


class RoundClip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, low, high):
        ctx.save_for_backward(value)
        ctx.low, ctx.high = low, high
        return value.round().clamp(low, high)

    @staticmethod
    def backward(ctx, gradient):
        value, = ctx.saved_tensors
        return gradient*((value >= ctx.low) & (value <= ctx.high)), None, None


class FixedGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, cutoff, direction, constant, theta):
        margin = (value-cutoff)*direction/STATE_SCALE
        ctx.save_for_backward(margin, direction, constant)
        ctx.theta = abs(float(theta))
        gate = torch.where(direction > 0, value >= cutoff, value <= cutoff)
        return torch.where(constant >= 0, constant.bool(), gate).to(value.dtype)

    @staticmethod
    def backward(ctx, gradient):
        margin, direction, constant = ctx.saved_tensors
        slope = (1-margin.abs()/max(ctx.theta, 1e-12)).clamp_min(0)
        return gradient*slope*direction/STATE_SCALE*(constant < 0), None, None, None, None


class FixedQATForward(FixedTemporalForward):
    def __init__(self, controller, sn2_theta):
        super().__init__(controller, sn2_theta)
        if self.kind != 'diag':
            raise ValueError('This bounded recovery implements the ordinary raw diagonal student only.')
        self.sn2_theta = float(sn2_theta)
        bn = controller.bn
        self.bn_gain = bn.weight.detach().double()/torch.sqrt(bn.running_var.detach().double()+bn.eps)

    def live_q(self, key):
        if key == 'U_conv2_theta':
            value = self.c.conv_u.double().reshape(16, -1)*self.sn2_theta
        elif key == 'F':
            value = self.bn_gain[:, None]*self.c.conv_v.double()[:, :, 0, 0]
        else:
            return self.matrices[key]['q']
        exponent = self.matrices[key]['exponent']
        return RoundClip.apply(value*(2.0**exponent), -32768, 32767)

    def write24(self, name, accumulator, shift=0):
        return RoundClip.apply(torch.ldexp(accumulator, torch.as_tensor(-shift, device=self.device)),
                               STATE_MIN, STATE_MAX)

    def channel_dot(self, key, value, name):
        t, _, height, width = value.shape
        flat = value.permute(1, 0, 2, 3).reshape(value.shape[1], -1)
        q = self.live_q(key)
        accumulator = (q@flat).reshape(q.shape[0], t, height, width).permute(1, 0, 2, 3)
        return self.write24(name, accumulator, self.matrices[key]['exponent'])

    def conv_forward(self, x):
        bits = x.ne(0).flatten(0, 1).double()
        accumulator = F.conv2d(bits, self.live_q('U_conv2_theta').reshape_as(self.c.conv_u),
            None, self.c.rank.stride, self.c.rank.padding, self.c.rank.dilation)
        self.z = self.write24('Z24', accumulator,
            self.matrices['U_conv2_theta']['exponent']-14)
        # The actual fixed consumers read self.z. The outer float branch is
        # still the existing shape/call shadow, with no surrogate gradient.
        with torch.no_grad():
            return self.original['conv'](x)

    def finish(self):
        if self.ready:
            return
        z = self.z[:, :, ::2, ::2]
        branch = self.channel_dot('F', z, 'F_Z_residual24')
        self.updated = self.i.clone()
        self.updated[:, :, ::2, ::2] = self.write24('updated_anchor24',
            self.updated[:, :, ::2, ::2]+branch+self.c_bn[None, :, None, None])
        latent = self.channel_dot('U_ped', self.updated[:, :, ::2, ::2], 'PED_U24')
        output = self.channel_dot('V_ped', latent, 'PED_V24')
        self.continuous = self.write24('PED_bias_output24',
            output+self.projection_bias[None, :, None, None])
        self.ready = True

    def projection_forward(self, ignored_float_shadow):
        self.finish()
        return self.continuous.float()/STATE_SCALE

    def consumer_forward(self, ignored_float_shadow):
        self.finish()
        value = self.updated.index_select(0, self.permutation)
        comp = self.comparisons['consumer']
        shape = (10,)+(1,)*(value.ndim-1)
        gate = FixedGate.apply(value, comp['threshold'].reshape(shape),
            comp['direction'].reshape(shape), comp['constant'].reshape(shape), self.consumer.thresh)
        result = self.emit(self.consumer, gate)
        self.i = self.q = self.z = self.updated = self.continuous = None
        return result
