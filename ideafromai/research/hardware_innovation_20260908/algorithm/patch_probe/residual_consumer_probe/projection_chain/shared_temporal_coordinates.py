"""Three fixed FP32 coordinate functions for the saved two-axis students.

No training or hardware-memory claim. The ordinary software residual path may
still materialize I+branch. q_ui/single_q consume their own I/Z coordinates,
not that already computed branch. All three modes disable TF32 for temporal
matrix operations; spatial convolutions keep the common cuDNN mode unless the
explicit fp32_channel diagnostic disables only their TF32 setting.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from adapter import fp32_matmul


def time_mix(matrix, value):
    with fp32_matmul():
        return (matrix@value.reshape(10, -1)).reshape_as(value)


def theta_gate(module, coordinates):
    """Unbiased T,C,H,W coordinate -> native bias/center/official theta*g."""
    membrane = coordinates.reshape(10, -1)+module.bias
    if module.center_mode != 'zero':
        membrane = membrane-module.center.to(membrane)
    spike, increment = module.act(membrane, module.thresh, module.sp)
    module.update_value += increment/max(1, module.T)
    out = spike.reshape_as(coordinates).unsqueeze(1)
    module.act_value = out.abs().reshape(10, -1).mean(1).sum()
    return out


class CoordinateForward:
    def __init__(self, controller, mode, fp32_channel=False):
        self.c, self.mode = controller, mode
        self.fp32_channel = fp32_channel
        self.source, self.consumer = controller.source, controller.consumer
        self.original = dict(source=self.source.forward, consumer=self.consumer.forward,
            conv=controller.conv.forward, projection=controller.projection.forward)
        self.shared = controller.shared_parameterization is not None
        self.As, self.Ap = self.source.weight.detach(), self.consumer.weight.detach()
        self.U, self.V = controller.u.detach(), controller.v.detach()
        self.frames, self.frame = [], {}
        self.capture_sink = None
        self.q = self.ui = self.z = self.gate_coordinates = self.continuous_latent = None
        self.ready = False
        self.constants = dict(As=self.As, Ap=self.Ap)
        with fp32_matmul(), torch.no_grad():
            if mode != 'native':
                bn = controller.bn
                gain = bn.weight/torch.sqrt(bn.running_var+bn.eps)
                offset = bn.bias-gain*bn.running_mean
                bconv = controller.rank.bias
                self.d_bn = offset if bconv is None else gain*bconv+offset
                self.F = gain[:, None]*controller.conv_v.detach()[:, :, 0, 0]
                self.Agate = self.As if self.shared or mode == 'single_q' else self.Ap
                self.Aones = self.Agate.sum(1)
                self.constants.update(F=self.F, d_BN=self.d_bn, gate_time_matrix=self.Agate)
                if mode == 'q_ui':
                    self.UF = self.U[:, :, 0, 0]@self.F
                    self.Ud = self.U[:, :, 0, 0]@self.d_bn
                    self.constants.update(UF=self.UF, Ud_BN=self.Ud)
                else:
                    inverse64 = torch.linalg.inv(self.As.double())
                    self.inverse = inverse64.float()
                    self.C = None if self.shared else (self.Ap.double()@inverse64).float()
                    self.constants.update(As_inverse=self.inverse)
                    if self.C is not None:
                        self.constants['ordinary_C_Ap_As_inverse'] = self.C
        self.metadata = dict(mode=mode, shared=self.shared,
            temporal_numeric='All time products FP32, TF32 disabled, including explicit native time control; channel policy is stated separately below.',
            fp32_channel=fp32_channel,
            channel_numeric=('Every F.conv2d inside this helper (both Conv2 factors, F, U/UF/V) uses cuDNN with TF32 disabled; all other cuDNN flags and helper-external arithmetic are preserved.'
                if fp32_channel else 'All helper channel convolutions retain the original cuDNN TF32 policy.'),
            constants='BN fold and UF compile in FP32; As inverse and ordinary C compile in Float64 then roundFP32, no entry thresholding.',
            shared_readout='Actual saved d64 multiplies selected FP32 coordinates in Float64 then roundsFP32; no rounded-A correction.',
            source_A_condition=float(torch.linalg.cond(self.As.double())),
            projection_A_condition=float(torch.linalg.cond(self.Ap.double())),
            ranges='Per-frame per-T max_abs from actual intermediate tensors only. Native does not materialize unbiased Q or AZ; native source/consumer membrane and PED latent are recorded instead.',
            claim='New FP32 reassociation requires fresh AEE. No native bit-equivalence, buffer deletion, operator skipping or GPU-speed claim.')
        if mode == 'single_q':
            self.metadata['inverse_max_row_L1'] = float(self.inverse.abs().sum(1).max())
        for module in (self.source, self.consumer):
            if module.output_mode != 'binary' or module.threshold_mode != 'official_atlif':
                raise ValueError('This fixed coordinate probe uses the actual official theta*g neurons.')
        self.source.forward, self.consumer.forward = self.source_forward, self.consumer_forward
        controller.conv.forward, controller.projection.forward = self.conv_forward, self.projection_forward

    def record(self, name, value):
        self.frame[name] = value.detach().reshape(10, -1).abs().amax(1).cpu().tolist()
        if self.capture_sink is not None:
            self.capture_sink(name, value)

    def channel_conv(self, *args, **kwargs):
        if not self.fp32_channel:
            return F.conv2d(*args, **kwargs)
        # flags() otherwise defaults enabled/benchmark/deterministic as well.
        # None preserves each existing setting; only allow_tf32 changes here.
        with torch.backends.cudnn.flags(enabled=None, benchmark=None,
                benchmark_limit=None, deterministic=None, allow_tf32=False):
            return F.conv2d(*args, **kwargs)

    def native_neuron(self, name, module, original, x):
        # Observe native membrane already produced by addmm, not an extra Q.
        previous = getattr(module, '_h9_calibration_observer', None)
        def observe(h, theta):
            self.record(name+'_native_membrane', h)
            if previous is not None:
                previous(h, theta)
        module._h9_calibration_observer = observe
        try:
            with fp32_matmul():
                return original(x)
        finally:
            module._h9_calibration_observer = previous

    def source_forward(self, x):
        self.frame, self.ready = {}, False
        self.q = self.ui = self.z = self.gate_coordinates = self.continuous_latent = None
        identity = x[:, 0]
        self.record('I', identity)
        if self.mode == 'native':
            return self.native_neuron('source', self.source, self.original['source'], x)
        Q = time_mix(self.As, identity)
        self.record('Q_As_I', Q)
        source_spikes = theta_gate(self.source, Q)
        if self.mode == 'q_ui':
            self.q = Q if self.shared else time_mix(self.Ap, identity)
            if not self.shared:
                self.record('Qp_Ap_I', self.q)
            self.ui = self.channel_conv(identity[:, :, ::2, ::2], self.U)
            self.record('UI_anchor', self.ui)
        else:
            self.q = Q
        return source_spikes

    def conv_forward(self, x):
        # Exactly the current R16 producer; retained Z is from this invocation.
        c = self.c
        self.z = self.channel_conv(x.flatten(0, 1), c.conv_u, None,
                         c.rank.stride, c.rank.padding, c.rank.dilation)
        self.record('Z', self.z)
        raw = self.channel_conv(self.z, c.conv_v, c.rank.bias)
        return raw.reshape(x.shape[0], x.shape[1], 96, *raw.shape[-2:])

    def shared_readout(self, coordinates):
        p = self.c.shared_parameterization
        # Negative d and equality at theta retain their actual numeric meaning.
        return (coordinates[p.permutation].double()*p.d[:, None, None, None]).float()

    def finish_coordinates(self):
        if self.ready:
            return
        anchor_z = self.z[:, :, ::2, ::2]
        self.record('Z_anchor', anchor_z)
        az = time_mix(self.Agate, anchor_z)
        self.record('AZ_anchor', az)
        correction = self.channel_conv(az, self.F[:, :, None, None])
        correction = correction+self.Aones[:, None, None, None]*self.d_bn[None, :, None, None]
        # Nonanchor BN branch was deleted, including its offset. No correction
        # is applied outside the exact even/even continuous-consumer anchors.
        self.q[:, :, ::2, ::2] = self.q[:, :, ::2, ::2]+correction
        self.record('updated_gate_coordinates', self.q)
        if self.mode == 'q_ui':
            latent = self.ui+self.channel_conv(anchor_z, self.UF[:, :, None, None])
            self.continuous_latent = latent+self.Ud[None, :, None, None]
            self.gate_coordinates = self.shared_readout(self.q) if self.shared else self.q
        else:
            uq = self.channel_conv(self.q[:, :, ::2, ::2], self.U)
            self.record('UQ_anchor', uq)
            self.continuous_latent = time_mix(self.inverse, uq)
            self.gate_coordinates = self.shared_readout(self.q) if self.shared else time_mix(self.C, self.q)
        self.record('continuous_latent', self.continuous_latent)
        self.ready = True

    def projection_forward(self, x):
        c = self.c
        if self.mode == 'native':
            latent = self.channel_conv(x[:, :, ::2, ::2], c.u)
            self.record('continuous_latent', latent)
        else:
            self.finish_coordinates()
            latent = self.continuous_latent
        out = self.channel_conv(latent, c.v, c.original['projection_bias'])
        result = out+c.projection_bias_delta[None, :, None, None]
        self.record('continuous_output', result)
        return result

    def consumer_forward(self, x):
        if self.mode == 'native':
            out = self.native_neuron('consumer', self.consumer, self.original['consumer'], x)
        else:
            self.finish_coordinates()
            out = theta_gate(self.consumer, self.gate_coordinates)
        self.frames.append(self.frame)
        self.q = self.ui = self.z = self.gate_coordinates = self.continuous_latent = None
        return out

    def export_constants(self):
        return {key: value.detach().cpu().numpy() for key, value in self.constants.items()}

    def range_report(self, names):
        return dict(scope=self.metadata['ranges'], frames=[dict(file=name, per_T_max_abs=row)
                    for name,row in zip(names, self.frames)])

    def restore(self):
        self.source.forward, self.consumer.forward = self.original['source'], self.original['consumer']
        self.c.conv.forward, self.c.projection.forward = self.original['conv'], self.original['projection']
