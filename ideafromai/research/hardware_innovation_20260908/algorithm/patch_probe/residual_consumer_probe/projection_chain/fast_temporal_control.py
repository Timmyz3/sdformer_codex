"""Trainable T10 factor-basis controls on the common R16/R32 residual parent.

The source executes four fixed matching layers, Q=B I, followed by its own
gain/permutation/bias/native theta gate. The ordinary consumer reads raw
I+anchor-BN2-branch through a separate diagonal/permutation readout. The shared
consumer updates canonical Q using the actual Conv2 latent, live V16 and BN2
beta, then uses B's reverse factors for the continuous PED latent. Dense A/B
matrices exported below are coefficient references, never forward operators.

All helper channel convolutions are FP32 with TF32 disabled. Time transforms
execute factor add/subtracts; logits use exact hard signs with identity STE.
This defines two new FP32 functions, not bit-equivalent rewrites of native A.
The caller installs the common nonanchor whole-BN2-branch deletion and frozen
preview surrogate. Software Conv2/BN/add shadows still execute; shared outputs
do not read those shadows. No hardware memory or speed claim follows from this
training implementation. Batch size is the existing one-frame T10 contract.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from fast_temporal_basis import FastTemporalBasis, MATCHINGS
from train_shared_temporal_recovery import SharedTemporalControl, AXES as OLD_AXES
from shared_temporal_coordinates import theta_gate


AXES = ('fast_raw_diagonal', 'fast_shared')


class FastTemporalControl(SharedTemporalControl):
    def __init__(self, modules, conv_arrays, ped_arrays, source_fit, consumer_fits):
        super().__init__(modules, conv_arrays, ped_arrays, fit={})
        self.source_fit, self.consumer_fits = source_fit, consumer_fits
        self.original_source_forward = self.source.forward
        self.original_consumer_forward = self.consumer.forward
        fixed = [self.source.weight, self.consumer.weight, self.source.thresh,
                 self.consumer.thresh, self.source.center, self.consumer.center,
                 self.bn.weight, self.rank.bias, self.original['projection_bias']]
        self.frozen_flags = [(value, value.requires_grad) for value in fixed
                             if isinstance(value, nn.Parameter)]
        self.anchor_indices = {}
        self.clear_cached_graphs()

    def trainable(self, axis):
        if axis not in AXES:
            raise ValueError('Unknown fast temporal control: '+axis)
        # Preserve the common R16 view's strides and all common trainable terms.
        super().trainable(OLD_AXES[0])
        self.axis = axis
        del self.params['source_A'], self.params['consumer_A']
        self.source.weight = self.original['source_A']
        self.consumer.weight = self.original['consumer_A']
        for value, _ in self.frozen_flags:
            value.requires_grad_(False)
        device, dtype = self.source.bias.device, self.source.bias.dtype
        self.basis = FastTemporalBasis(learnable_signs=True, device=device, dtype=dtype)
        with torch.no_grad():
            self.basis.sign_logits.fill_(0.01)
        for name, fit, module in (
                ('source', self.source_fit, self.source),
                ('consumer', self.consumer_fits[axis], self.consumer)):
            gain = nn.Parameter(torch.as_tensor(fit['row_gain'], device=device,
                                                dtype=dtype).reshape(10).clone())
            permutation = torch.as_tensor(fit['row_permutation'], device=device,
                                          dtype=torch.long).reshape(10).clone()
            setattr(self, name+'_row_gain', gain)
            setattr(self, name+'_row_permutation', permutation)
            self.params[name+'_row_gain'] = gain
            with torch.no_grad():
                module.bias.copy_(torch.as_tensor(fit['bias'], device=device,
                                                 dtype=dtype).reshape_as(module.bias))
        self.params['basis_sign_logits'] = self.basis.sign_logits
        self.source.forward, self.consumer.forward = self.source_forward, self.consumer_forward
        self.conv.forward, self.projection.forward = self.conv_forward, self.projection_forward
        self.clear_cached_graphs()
        return self

    @staticmethod
    def channel_conv(value, weight, bias=None, *args, **kwargs):
        # None preserves the caller's other cuDNN settings. The same policy
        # applies to both Conv2 factors, F mixing and PED U/V on both axes.
        with torch.autocast(device_type=value.device.type, enabled=False), \
                torch.backends.cudnn.flags(enabled=None, benchmark=None,
                    benchmark_limit=None, deterministic=None, allow_tf32=False):
            return F.conv2d(value.float(), weight, bias, *args, **kwargs)

    def readout(self, name, value):
        gain = getattr(self, name+'_row_gain')
        permutation = getattr(self, name+'_row_permutation')
        return value.index_select(0, permutation)*gain[:, None, None, None]

    def source_forward(self, x):
        self.clear_cached_graphs()
        identity = x[:, 0].float()
        q = self.basis(identity)
        if self.axis == AXES[1]:
            self.q = q
        else:
            self.identity = identity
        return theta_gate(self.source, self.readout('source', q))

    def conv_forward(self, x):
        self.z = self.channel_conv(x.flatten(0, 1), self.conv_u, None,
                                  self.rank.stride, self.rank.padding, self.rank.dilation)
        # Normal shadow is required by the existing BN/residual call sequence.
        raw = self.channel_conv(self.z, self.conv_v, self.rank.bias)
        return raw.reshape(x.shape[0], x.shape[1], 96, *raw.shape[-2:])

    def finish(self):
        if self.ready:
            return
        gain = self.bn.weight/torch.sqrt(self.bn.running_var+self.bn.eps)
        # Neither live V16 nor live beta is detached or compiled at install.
        mixed_weight = gain[:, None]*self.conv_v[:, :, 0, 0]
        offset = self.bn.bias-gain*self.bn.running_mean
        if self.rank.bias is not None:
            offset = offset+gain*self.rank.bias
        bz = self.basis(self.z[:, :, ::2, ::2])
        correction = self.channel_conv(bz, mixed_weight[:, :, None, None])
        b_one = self.basis(torch.ones(10, 1, device=bz.device, dtype=bz.dtype))[:, 0]
        correction = correction+b_one[:, None, None, None]*offset[None, :, None, None]
        height, width = self.q.shape[-2:]
        key = (height, width, self.q.device)
        if key not in self.anchor_indices:
            self.anchor_indices[key] = torch.arange(height*width, device=self.q.device).reshape(
                height, width)[::2, ::2].reshape(-1)
        # Functional update: the source gate backward still needs the old Q.
        self.updated_q = self.q.flatten(2).index_add(
            2, self.anchor_indices[key], correction.flatten(2)).reshape_as(self.q)
        uq = self.channel_conv(self.updated_q[:, :, ::2, ::2], self.u)
        self.continuous_latent = self.basis.reverse(uq)
        self.ready = True

    def projection_forward(self, x):
        if self.axis == AXES[1]:
            self.finish()
            latent = self.continuous_latent
        else:
            latent = self.channel_conv(x[:, :, ::2, ::2], self.u)
        out = self.channel_conv(latent, self.v, self.original['projection_bias'])
        return out+self.projection_bias_delta[None, :, None, None]

    def consumer_forward(self, x):
        if self.axis == AXES[1]:
            self.finish()
            value = self.updated_q
        else:
            value = x[:, 0].float()
        return theta_gate(self.consumer, self.readout('consumer', value))

    def clear_cached_graphs(self):
        self.identity = self.q = self.z = self.updated_q = self.continuous_latent = None
        self.ready = False
        self.terms.clear()

    @torch.no_grad()
    def constraints(self):
        basis = self.basis.dense_matrix().double()
        signs = self.basis.signs()
        eye = torch.eye(10, device=basis.device, dtype=basis.dtype)
        return dict(axis=self.axis, basis_rank=int(torch.linalg.matrix_rank(basis)),
            basis_transpose_inverse_max_abs=float((basis.T@basis-eye).abs().max()),
            hard_signs=signs.cpu().tolist(), negative_signs_from_initial=int((signs < 0).sum()),
            sign_count_note='Final Hamming distance from all+ initialization; not a count of training flips.',
            source_zero_gains=int((self.source_row_gain == 0).sum()),
            consumer_zero_gains=int((self.consumer_row_gain == 0).sum()),
            source_permutation=self.source_row_permutation.cpu().tolist(),
            consumer_permutation=self.consumer_row_permutation.cpu().tolist(),
            dense_A_is_frozen_reference=not self.source.weight.requires_grad and not self.consumer.weight.requires_grad,
            conv2_factor_strides=[list(self.conv_u.stride()), list(self.conv_v.stride())],
            source_theta=float(self.source.thresh), consumer_theta=float(self.consumer.thresh))

    @property
    def metadata(self):
        return dict(axis=self.axis, T=10, batch=1, sign_parameters=20,
            sign_initial_logit=0.01, sign_forward='where(logit>=0,+1,-1)', sign_backward='Identity STE',
            numeric='FP32 factor add/subtracts and gain readouts; helper channel convolutions disable TF32. No dense time multiply.',
            state=('Canonical Q=B I; update anchors from live Z/V16/BN2 beta; B^T after U for continuous output.'
                   if self.axis == AXES[1] else 'Raw I; source B is temporary, consumer reads actual I+anchor BN2 branch.'),
            gate='Native official theta_gate: own bias and center, >= comparison, actual theta-valued output. Signed/zero gains are kept.',
            branch='Caller deletes the complete BN2 branch at nonanchors. Shared finish adds neither residual nor BN offset there.',
            frozen='Original dense source/consumer weights are references only; theta, center, BN gain/statistics and base biases stay fixed.',
            boundary='Actual factor FP32 functions require fresh network AEE. Shadows remain in software; no claim that their hardware work/state was removed.')

    @torch.no_grad()
    def export(self, parent='', schedule=(), prior_steps=0):
        arrays = super().export(parent, schedule, prior_steps)
        cpu = lambda value: value.detach().cpu().numpy()
        basis = self.basis.dense_matrix()
        consumer_basis = basis if self.axis == AXES[1] else torch.eye(10, device=basis.device, dtype=basis.dtype)
        arrays.update(reference_source_A=arrays['source_A'], reference_consumer_A=arrays['consumer_A'],
            basis_sign_logits=cpu(self.basis.sign_logits), basis_hard_signs=cpu(self.basis.signs()),
            basis_matchings=np.asarray(MATCHINGS, np.int64), basis_B=cpu(basis),
            source_row_gain=cpu(self.source_row_gain), consumer_row_gain=cpu(self.consumer_row_gain),
            source_row_permutation=cpu(self.source_row_permutation),
            consumer_row_permutation=cpu(self.consumer_row_permutation),
            source_A=cpu(self.source_row_gain[:, None]*basis[self.source_row_permutation]),
            consumer_A=cpu(self.consumer_row_gain[:, None]*consumer_basis[self.consumer_row_permutation]),
            factor_numeric=np.array(self.metadata['numeric']),
            definition=np.array('Actual four-factor B forward/reverse with hard STE signs; independent gain/P/bias readouts. Dense source_A/consumer_A are mathematical coefficient references, not bit-equivalent execution paths. '+self.metadata['state']))
        arrays.pop('shared_d')
        arrays.pop('shared_permutation')
        return arrays

    @torch.no_grad()
    def load_saved(self, axis, arrays):
        self.trainable(axis)
        names = dict(r1_BN2_bias='bn2_beta', ped_output_bias_delta='projection_bias_delta')
        for name, parameter in self.params.items():
            parameter.copy_(torch.as_tensor(arrays[names.get(name, name)], device=parameter.device,
                                           dtype=parameter.dtype).reshape_as(parameter))
        for name in ('source', 'consumer'):
            current = getattr(self, name+'_row_permutation')
            current.copy_(torch.as_tensor(arrays[name+'_row_permutation'], device=current.device,
                                          dtype=torch.long).reshape_as(current))
        self.clear_cached_graphs()
        return self

    def restore(self):
        self.clear_cached_graphs()
        self.source.forward, self.consumer.forward = self.original_source_forward, self.original_consumer_forward
        super().restore()
        for value, original_flag in self.frozen_flags:
            value.requires_grad_(original_flag)
