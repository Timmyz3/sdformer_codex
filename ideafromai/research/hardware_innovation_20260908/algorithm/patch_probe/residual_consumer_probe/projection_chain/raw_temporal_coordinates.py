"""Actual factorized raw-coordinate evaluation of the two ordinary controls.

Source PSN, Conv2R16 and PEDR32 inherit CoordinateForward's native path and
its common channel-precision option. Only the consumer changes: e/L/R each
round to FP32, Rx is actually formed at rank2, then L(Rx) is added to e*x[P].
All temporal products disable TF32. The original local bias/center/theta gate
and real successors remain. This is a new rounded function, not bit-equivalent
to training's roundFP32(diag(e64)P + L64@R64) dense matrix execution.

No GPU execution is launched by this module. Install RawTemporalForward on a
loaded raw-structured controller; restore() removes all four forward hooks.
"""
from __future__ import annotations

import torch

from adapter import fp32_matmul
from shared_temporal_coordinates import CoordinateForward, theta_gate


def factorized_time_mix(e, permutation, value, left=None, right=None):
    """FP32 operands; returns output, diagonal term, actual rank2 latent/tail."""
    flat = value.reshape(10, -1)
    with fp32_matmul():
        diagonal = e[:, None]*flat.index_select(0, permutation)
        if right is None:
            return diagonal.reshape_as(value), diagonal.reshape_as(value), None, None
        latent = right@flat
        tail = left@latent
        output = diagonal+tail
    return (output.reshape_as(value), diagonal.reshape_as(value),
            latent.reshape(right.shape[0], *value.shape[1:]), tail.reshape_as(value))


class RawTemporalForward(CoordinateForward):
    def __init__(self, controller, fp32_channel=False):
        mapping = controller.raw_parameterization
        if mapping is None:
            raise ValueError('Load one of the raw-structured controllers before installing this helper.')
        self.e = mapping.e.detach().float()
        self.permutation = mapping.permutation.detach().clone()
        self.left = mapping.left.detach().float() if mapping.residual else None
        self.right = mapping.right.detach().float() if mapping.residual else None
        self.latent_frames = []
        # Keep self.mode='native': inherited source/Conv2/projection execution
        # must not enter either of the earlier residual-coordinate rewrites.
        super().__init__(controller, mode='native', fp32_channel=fp32_channel)
        self.constants.update(raw_e_fp32=self.e, raw_permutation=self.permutation)
        if self.right is not None:
            self.constants.update(raw_L2_fp32=self.left, raw_R2_fp32=self.right)
        self.metadata.update(mode='raw_factorized', inherited_path='native',
            raw_residual_rank=2 if mapping.residual else 0,
            raw_permutation=self.permutation.cpu().tolist(),
            negative_e_rows=self.e.lt(0).nonzero().flatten().cpu().tolist(),
            temporal_numeric='Source uses the inherited native FP32-time control. Consumer separately rounds e/L/R to FP32; diag=e*x[P], latent=R@x, tail=L@latent, coordinates=diag+tail, then native bias/center/act. All temporal matrix products disable TF32; intermediate arithmetic is FP32.',
            constants='As and Ap are inherited reference matrices. Ap is exported only as the saved dense comparison and is not used by the factorized consumer. raw_e/L2/R2_fp32 and the permutation are the executed coefficients; no small coefficient is zeroed.',
            shared_readout='None: source A and consumer raw-coordinate factors are independent.',
            input='Actual consumer input x=I+branch, [T10,B1,C,H,W]; the existing nonanchor whole-branch deletion remains upstream. No cached membrane or future residual is substituted.',
            consumer_linear_work_per_scalar_T10=dict(
                multiplications=50 if mapping.residual else 10,
                additions=38 if mapping.residual else 0,
                scope='Fixed factorized linear arithmetic only: R2 has20+20 multiplies, plus10 diagonal multiplies;18+10 dot additions plus10 merges. Bias/center/gate and producer/continuous consumer costs remain additional. These are not cycles.'),
            ranges='Per-frame actual10-row max_abs in per_T_max_abs; the real2-row R@x latent is separately stored in per_latent_max_abs. No full tensor trace.',
            claim='New factorized FP32 function needs its own AEE/activity evaluation. Training dense-A AEE does not validate this arithmetic or imply buffer/cycle savings.')

    def consumer_forward(self, x):
        value = x[:, 0].float()
        coordinates, diagonal, latent, tail = factorized_time_mix(
            self.e, self.permutation, value, self.left, self.right)
        self.record('consumer_input_I_plus_branch', value)
        self.record('raw_diagonal_term', diagonal)
        self.record('raw_unbiased_coordinates', coordinates)
        latent_ranges = {}
        if latent is not None:
            self.record('raw_rank2_tail', tail)
            latent_ranges['R2_x'] = latent.detach().reshape(2, -1).abs().amax(1).cpu().tolist()
        out = theta_gate(self.consumer, coordinates)
        self.frames.append(self.frame)
        self.latent_frames.append(latent_ranges)
        self.q = self.ui = self.z = self.gate_coordinates = self.continuous_latent = None
        return out

    def range_report(self, names):
        report = super().range_report(names)
        for frame, latent in zip(report['frames'], self.latent_frames):
            frame['per_latent_max_abs'] = latent
        return report
