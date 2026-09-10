"""Evaluate fixed recovered students with an explicit continuous consumer.

At actual stem output s, compute U*s at PED anchors. Add two source-driven
L_i*z_i and current fixed-BN affine offsets, then V and the original separate
projection bias terms. This function never reads r1out for its continuous
answer. The original gate-producing convolutions still execute in this
accuracy probe. No early stopping, quantization, or runtime speed claim.

For ordinary/W-zero/null, L_i=P_i W_i retains every computed coefficient,
including small nullspace residues. Independent L uses its exported hard-mask
kernels. Dynamic stem BN remains native: this layout pays a full dense U*s,
not a free dynamic-stem projected kernel. Supply --output for a new directory.
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

import evaluate_recovered_consumer as evaluation
from train_consumer_recovery import RecoveryControl, canonical_source
from capture_chain import R0, R1


class ExplicitConsumerControl(RecoveryControl):
    r1_only = False
    evaluation_numeric = (
        'Changed explicit Float32 function: U(actual stem output at anchors) '
        '+ L0*actual z0 + L1*actual z1 + U(current BN affine offsets), '
        'then V with base bias and separate learned delta. No r1out '
        'subtraction/cancellation; no exact equivalence to the recovery forward.'
    )
    evaluation_claim = (
        'Fixed-student numerical reassociation and AEE only. Original gate W '
        'still executes. Full U*s plus both L kernels are charged by this '
        'layout; no dynamic-stem K, early-stop or hardware speed result.'
    )

    def __init__(self, modules, arrays):
        super().__init__(modules, arrays)
        self.stem_latent = None
        self.handles.append(modules[R1 if self.r1_only else R0].register_forward_pre_hook(self.observe_stem))

    def observe_stem(self, module, inputs):
        source = canonical_source(inputs[0])
        self.stem_latent = F.conv2d(source[:, :, ::2, ::2], self.first)

    def observe(self, label, x):
        if self.r1_only and label == 'r0':
            return
        source = canonical_source(x)
        if self.axis == 'independent_sparse_L':
            kernel = self.effective_l(label)
        else:
            weight = self.effective_weight(label)
            kernel = (self.p[label]@weight.flatten(1).double()).to(weight.dtype)
        self.terms[label] = F.conv2d(source, kernel.reshape(32, 96, 3, 3),
                                     stride=2, padding=1)

    def projected(self, unused_r1out):
        # Seed is actual stem output, or actual r1 identity in the partial case.
        value = self.stem_latent
        if not self.r1_only:
            value = value+self.terms.pop('r0')
        value = value+self.terms.pop('r1')
        self.stem_latent = None
        offset = torch.zeros(96, dtype=torch.float64, device=value.device)
        for label in (('r1',) if self.r1_only else ('r0', 'r1')):
            bn, conv = self.bn[label], self.conv[label]
            gain = bn.weight.double()/torch.sqrt(bn.running_var.double()+float(bn.eps))
            offset = offset+bn.bias.double()-gain*bn.running_mean.double()
            if conv.bias is not None:
                offset = offset+gain*conv.bias.double()
        latent_offset = (self.first[:, :, 0, 0].double()@offset).to(value.dtype)
        value = value+latent_offset[None, :, None, None]
        value = F.conv2d(value, self.second, self.projection.bias)
        return value+self.projection_bias_delta[None, :, None, None]

    def restore(self):
        self.stem_latent = None
        super().restore()


if __name__ == '__main__':
    if '--output' not in sys.argv:
        raise SystemExit('Supply --output for the explicit-function evaluation directory.')
    if '--r1-only' in sys.argv:
        sys.argv.remove('--r1-only')
        ExplicitConsumerControl.r1_only = True
        ExplicitConsumerControl.evaluation_numeric = (
            'Changed explicit Float32 function: U(actual r1 identity at anchors) '
            '+ L1*actual z1 + U(current r1 BN affine offset), then V/base bias '
            'and separate delta. This restores the P0W0 continuous contribution '
            'instead of independent L0: a changed student, not a free fold-back.'
        )
        ExplicitConsumerControl.evaluation_claim = (
            'Fixed r1-only function and AEE probe. Original W0/W1 gate chain '
            'still executes; dense U remains and only L1 is added. No inherited '
            'two-L accuracy, early-stop or hardware speed result.'
        )
    evaluation.RecoveryControl = ExplicitConsumerControl
    evaluation.main()
