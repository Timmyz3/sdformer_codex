"""Continuous-lifting implementation of the two fast temporal controls.

Only the common raw/shared residual dataflow and lifecycle are inherited from
FastTemporalControl. This class never instantiates sign logits or their STE.
The actual source/shared/inverse passes execute LiftingTemporalBasis. Common
Conv2 R16, PED R32, BN beta and readout biases/gains have the same permissions
as the hard-sign family. Nonanchor branch deletion stays in the caller.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from fast_temporal_basis import MATCHINGS
from fast_temporal_control import FastTemporalControl, AXES
from lifting_temporal_basis import LiftingTemporalBasis
from train_shared_temporal_recovery import SharedTemporalControl, AXES as OLD_AXES


class LiftingTemporalControl(FastTemporalControl):
    def __init__(self, modules, conv_arrays, ped_arrays, source_fit, consumer_fits, basis_lifting):
        super().__init__(modules, conv_arrays, ped_arrays, source_fit, consumer_fits)
        self.initial_lifting = np.asarray(basis_lifting).reshape(4, 5, 2).copy()

    def trainable(self, axis):
        if axis not in AXES:
            raise ValueError('Unknown lifting temporal control: '+axis)
        # Initialize common trainable tensors directly; no temporary sign basis.
        SharedTemporalControl.trainable(self, OLD_AXES[0])
        self.axis = axis
        del self.params['source_A'], self.params['consumer_A']
        self.source.weight = self.original['source_A']
        self.consumer.weight = self.original['consumer_A']
        for value, _ in self.frozen_flags:
            value.requires_grad_(False)
        device, dtype = self.source.bias.device, self.source.bias.dtype
        self.basis = LiftingTemporalBasis(self.initial_lifting, device=device, dtype=dtype)
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
        self.params['basis_lifting'] = self.basis.lifting
        self.source.forward, self.consumer.forward = self.source_forward, self.consumer_forward
        self.conv.forward, self.projection.forward = self.conv_forward, self.projection_forward
        self.clear_cached_graphs()
        return self

    @torch.no_grad()
    def constraints(self):
        return dict(axis=self.axis, basis_kind='lifting40', **self.basis.conditioning(),
            source_zero_gains=int((self.source_row_gain == 0).sum()),
            consumer_zero_gains=int((self.consumer_row_gain == 0).sum()),
            source_permutation=self.source_row_permutation.cpu().tolist(),
            consumer_permutation=self.consumer_row_permutation.cpu().tolist(),
            dense_A_is_frozen_reference=not self.source.weight.requires_grad and not self.consumer.weight.requires_grad,
            conv2_factor_strides=[list(self.conv_u.stride()), list(self.conv_v.stride())],
            source_theta=float(self.source.thresh), consumer_theta=float(self.consumer.thresh))

    @property
    def metadata(self):
        return dict(axis=self.axis, basis_kind='lifting40', T=10, batch=1,
            basis_parameters=40, factor_specification=self.basis.specification(),
            numeric='FP32 lifting multiply/add/subtract steps and gain readouts; all helper channel convolutions disable TF32. No dense temporal forward or runtime inverse.',
            state=('Canonical Q=B I; anchors updated from live Z/V16/BN2 beta; actual inverse lifting after U for continuous output.'
                   if self.axis == AXES[1] else 'Raw I; lifting source B temporary; consumer reads actual I+anchor BN2 branch through independent diagonal/P.'),
            gate='Native theta_gate with each bias/center/actual theta amplitude; signed and zero gains retained.',
            branch='Caller deletes the complete BN2 branch at nonanchors, including offset; shared finish adds neither there.',
            frozen='Dense source/consumer A are references only; theta/center, BN gain/statistics, base biases and preview stay fixed.',
            boundary='A new FP32 student, not cond1 or a fixed-width implementation. Export reports coefficient-domain conditioning/bounds; actual ranges, guard bits, quantization and AEE remain to be measured. Shadows still execute in software.')

    @torch.no_grad()
    def export(self, parent='', schedule=(), prior_steps=0):
        arrays = SharedTemporalControl.export(self, parent, schedule, prior_steps)
        cpu = lambda value: value.detach().cpu().numpy()
        basis = self.basis.dense_matrix(dtype=torch.float64)
        inverse = self.basis.dense_matrix(inverse=True, dtype=torch.float64)
        consumer_basis = basis if self.axis == AXES[1] else torch.eye(10, device=basis.device, dtype=basis.dtype)
        arrays.update(reference_source_A=arrays['source_A'], reference_consumer_A=arrays['consumer_A'],
            basis_kind=np.array('lifting40'), basis_lifting=cpu(self.basis.lifting),
            basis_matchings=np.asarray(MATCHINGS, np.int64),
            basis_B=cpu(basis), basis_inverse_B=cpu(inverse),
            source_row_gain=cpu(self.source_row_gain), consumer_row_gain=cpu(self.consumer_row_gain),
            source_row_permutation=cpu(self.source_row_permutation),
            consumer_row_permutation=cpu(self.consumer_row_permutation),
            source_A=cpu((self.source_row_gain.double()[:, None]*basis[self.source_row_permutation]).float()),
            consumer_A=cpu((self.consumer_row_gain.double()[:, None]*consumer_basis[self.consumer_row_permutation]).float()),
            factor_numeric=np.array(self.metadata['numeric']),
            definition=np.array('Actual four-layer lifting forward/reverse with40 shared live coefficients; independent signed gain/P/bias readouts. basis_B/inverse are F64 real coefficient references built from stored FP32 parameters; source_A/consumer_A are rounded references, not the execution path. '+self.metadata['state']))
        for name, value in self.basis.conditioning().items():
            arrays['lifting_'+name] = np.asarray(value)
        arrays.pop('shared_d')
        arrays.pop('shared_permutation')
        return arrays

    # load_saved(), clear_cached_graphs(), gradients(), forward methods and
    # restore() use the common implementation. load_saved dispatches to this
    # trainable() and copies basis_lifting, never a sign or dense-A parameter.
