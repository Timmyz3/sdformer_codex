"""One fixed-format evaluation of the two recovered lifting40 students.

Every completed state is signed24/f14, RNE then saturate. A lifting half-step
forms (old*4096 +/- q12*other) in signed48 and immediately writes24; the inverse
uses the SAME forty signed16/f12 coefficients, reverse half/layer order. No
dense time multiply, separately quantized inverse, or gain-dependent inverse
is used. Rounding/saturation make this a new function, not an exact round trip.

The other static matrices use the existing per-matrix signed16 dyadic rule.
Integer dots use Float64 only after proving the entire legal input domain fits
signed48. Source and consumer gains compile into inclusive signed thresholds;
theta amplitudes, bias and center are separate. Raw retains I24, shared Q24.
Both PED consumers ignore the software Conv2/BN/residual shadows. The frozen
preview and remaining network keep their existing numeric function. No RTL,
hardware speed/state deletion, whole-network integer or inherited AEE claim.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from fast_temporal_basis import MATCHINGS
from lifting_temporal_basis import LiftingTemporalBasis
from fixed_temporal_coordinates import (
    FixedTemporalForward, STATE_FRAC, STATE_SCALE, STATE_MIN, STATE_MAX,
    as_numpy, tau_fractions,
)

LIFT_FRAC = 12
LIFT_SCALE = 1 << LIFT_FRAC


class FixedLiftingForward(FixedTemporalForward):
    """Install after LiftingTemporalControl.load_saved; restore before reload.

The inherited methods only implement integer matrix compilation, RNE24,
range reporting and signed comparisons. The dense-As constructor/forward
from FixedTemporalForward is deliberately not called.
"""
    def __init__(self, controller, sn2_theta):
        self.c, self.source, self.consumer = controller, controller.source, controller.consumer
        self.device = self.source.bias.device
        if not isinstance(controller.basis, LiftingTemporalBasis):
            raise ValueError('FixedLiftingForward requires the actual lifting40 core.')
        if controller.axis not in ('fast_raw_diagonal', 'fast_shared'):
            raise ValueError('Only the two fixed lifting40 controls are supported.')
        if controller.bn.training or not controller.bn.track_running_stats:
            raise ValueError('This function requires the current fixed eval BN2.')
        self.kind = 'shared' if controller.axis == 'fast_shared' else 'raw'
        self.original = dict(source=self.source.forward, consumer=self.consumer.forward,
            conv=controller.conv.forward, projection=controller.projection.forward)
        self.matrices, self.matrix_metadata, self.constants, self.constant_metadata = {}, {}, {}, {}
        self.comparisons, self.comparison_metadata = {}, {}
        self.frames, self.frame = [], self.new_frame()
        self.i = self.q = self.z = self.updated = self.continuous = None
        self.ready = False
        self.source_permutation = controller.source_row_permutation.detach().clone()
        self.consumer_permutation = controller.consumer_row_permutation.detach().clone()
        self.matchings = torch.as_tensor(MATCHINGS, device=self.device, dtype=torch.long)

        coefficients = as_numpy(controller.basis.lifting)
        q12 = np.rint(coefficients*LIFT_SCALE).astype(np.int64)
        if np.any(q12 < -32768) or np.any(q12 > 32767):
            raise ValueError('The fixed signed16/f12 lifting format does not fit this student.')
        self.lifting_q12 = torch.as_tensor(q12, device=self.device, dtype=torch.float64)
        self.halfstep_bounds = []
        for reverse in (False, True):
            for layer in (range(3, -1, -1) if reverse else range(4)):
                for half in ((1, 0) if reverse else (0, 1)):
                    # Both operands were previously written24. A half-step's
                    # multiplication and aligned identity addition are exact.
                    k = q12[layer, :, half]*(-1 if reverse else 1)
                    lo = LIFT_SCALE*STATE_MIN+np.where(k >= 0, k*STATE_MIN, k*STATE_MAX)
                    hi = LIFT_SCALE*STATE_MAX+np.where(k >= 0, k*STATE_MAX, k*STATE_MIN)
                    bound = self.prove48('lifting half-step', lo, hi)
                    self.halfstep_bounds.append(dict(reverse=reverse, layer=layer,
                        coefficient_half=half, minimum=lo.tolist(), maximum=hi.tolist(),
                        abs_bound=bound, fits_signed48=True))
        diagnostic = LiftingTemporalBasis(q12/LIFT_SCALE, dtype=torch.float64)
        self.constants.update(lifting_q12=q12.astype(np.int16),
            lifting_fraction_bits=np.asarray(LIFT_FRAC), lifting_actual=q12/LIFT_SCALE,
            lifting_matchings=np.asarray(MATCHINGS, np.int64),
            lifting_B_real_reference=diagnostic.dense_matrix().detach().numpy(),
            lifting_inverse_real_reference=diagnostic.dense_matrix(inverse=True).detach().numpy(),
            source_permutation=self.source_permutation, consumer_permutation=self.consumer_permutation)
        self.real_lifting_conditioning = diagnostic.conditioning()

        self.compile_matrix('U_conv2_theta', as_numpy(controller.conv_u).reshape(16, -1)*float(sn2_theta), bits=True)
        bn = controller.bn
        gain = as_numpy(bn.weight)/np.sqrt(as_numpy(bn.running_var)+bn.eps)
        offset = as_numpy(bn.bias)-gain*as_numpy(bn.running_mean)
        conv_bias = np.zeros(96) if controller.rank.bias is None else as_numpy(controller.rank.bias)
        self.compile_matrix('F', gain[:, None]*as_numpy(controller.conv_v)[:, :, 0, 0])
        self.compile_matrix('U_ped', as_numpy(controller.u)[:, :, 0, 0])
        self.compile_matrix('V_ped', as_numpy(controller.v)[:, :, 0, 0])
        self.c_bn = self.constant24('BN2_constant', (gain*conv_bias+offset)*STATE_SCALE)
        base_bias = controller.original['projection_bias']
        output_bias = (np.zeros(96) if base_bias is None else as_numpy(base_bias))+as_numpy(controller.projection_bias_delta)
        self.projection_bias = self.constant24('PED_bias', output_bias*STATE_SCALE)
        self.static_lifting_report = None
        if self.kind == 'shared':
            # Compile the time-constant vector with the same quantized program,
            # including each half-step write. No dense B*constant shortcut.
            self.c_q = self.lift(self.c_bn[None].expand(10, -1), 'static_BN2_lifting')
            self.constants['lifting_BN2_constant_q24'] = self.c_q.to(torch.int32)
            self.static_lifting_report = self.frame
            self.frame = self.new_frame()
        source_tau, consumer_tau = tau_fractions(self.source), tau_fractions(self.consumer)
        lo24, hi24 = [STATE_MIN]*10, [STATE_MAX]*10
        for name, module, tau in (('source', self.source, source_tau), ('consumer', self.consumer, consumer_tau)):
            row_gain = as_numpy(getattr(controller, name+'_row_gain'))
            self.compile_gate(name, tau, row_gain, STATE_FRAC, lo24, hi24)
            self.constants.update({name+'_row_gain': row_gain, name+'_tau_real': np.asarray([float(v) for v in tau]),
                name+'_theta': np.asarray(float(module.thresh.detach())), name+'_bias': as_numpy(module.bias),
                name+'_center': as_numpy(module.center), name+'_center_mode': np.asarray(module.center_mode)})
        self.constants.update(sn2_theta=np.asarray(float(sn2_theta)), state_bits=np.asarray(24),
            state_fraction_bits=np.asarray(STATE_FRAC), accumulator_bits=np.asarray(48))
        self.metadata = dict(mode='fixed_lifting', axis=controller.axis, basis_kind='lifting40', T=10,
            completed_state='signed24 f14; RNE and saturate at I, each lifting half-step, Z, F, anchor merge, PED U, each inverse half-step, PED V and output bias merge',
            lifting_format='All40 coefficients signed16/f12 RNE, no clipping; both directions use these same coefficients. (old<<12)+/-q12*other -> RNE24. No dense inverse.',
            lifting_coefficients=dict(original_max_abs=float(np.max(np.abs(coefficients))),
                quantized_max_abs=float(np.max(np.abs(q12))/LIFT_SCALE),
                max_abs_quantization_error=float(np.max(np.abs(coefficients-q12/LIFT_SCALE))),
                halfstep_bounds=self.halfstep_bounds, real_linear_conditioning=self.real_lifting_conditioning),
            matrices=self.matrix_metadata, static_constants=self.constant_metadata,
            static_lifting=self.static_lifting_report, comparisons=self.comparison_metadata,
            in_flight='Signed48 legal-domain bound for each dot/half-step; Float64 integer products and every partial sum are exact under these bounds. Not RTL.',
            constant_compilation='Live saved FP32 U/V/BN values promoted to F64; BN gain, F and c compiled once. c writes24 then the same q12 lifting program compiles shared Bc24. PED base bias plus separate delta writes24 once.',
            source='I24 -> actual eight lifting half-step writes -> Q24 -> own P/gain compiled threshold -> actual source theta*g. Gain is a readout, not part of the invertible canonical basis.',
            state_order=('Retain Q24; anchor Z24 -> eight lifting writes -> F/residual24; Q+residual+Bc -> anchor24; U -> UQ24; eight inverse writes -> latent24; V -> output24 -> bias merge24'
                if self.kind == 'shared' else
                'Retain I24; source Q is temporary. Anchor Z24 -> F/residual24; I+residual+c -> anchor24; U -> latent24; V -> output24 -> bias merge24'),
            consumer='Own P and saved signed gain compiled to >=ceil or <=floor including equality; exact zero gain/domain constants folded for both axes. Own bias/center/theta kept.',
            residual='Only even/even anchors receive residual and BN2 c. All nonanchor BN2-branch contributions, including bias, are absent.',
            merge_abs_bound=3*(1 << 23),
            shadow='Original FP Conv2, outer BN and identity add still run for software shape compatibility. Neither actual PED consumer uses that float shadow. No extra Q copy/cache saving is claimed.',
            scope='Frozen preview and remaining network stay outside this fixed helper. Fresh full-network AEE and activity required; no native or quantized round-trip equivalence, hardware cycle or memory-removal claim.')
        self.source.forward, self.consumer.forward = self.source_forward, self.consumer_forward
        controller.conv.forward, controller.projection.forward = self.conv_forward, self.projection_forward

    @staticmethod
    def new_frame():
        return dict(clip_counts={}, state_ranges={}, accumulator_ranges={}, lifting_time_indices={})

    def lift(self, value, name, reverse=False):
        """Actual integer factor program; every written five-vector is RNE24.

        x0*4096 and q*x1 are integer Float64, not a Float32 multiply/add.
        The complete aligned half-step is rounded together; rounding only the
        product would change half-way cases. Saturation applies before the
        next dependent half-step. Forward/reverse do not promise cancellation.
        """
        result = value
        shape = (5,)+(1,)*(value.ndim-1)
        for layer in (range(3, -1, -1) if reverse else range(4)):
            first, second = self.matchings[layer, :, 0], self.matchings[layer, :, 1]
            x0, x1 = result.index_select(0, first), result.index_select(0, second)
            a, b = (self.lifting_q12[layer, :, half].reshape(shape) for half in (0, 1))
            key = name+'_L'+str(layer)
            if reverse:
                acc1 = x1*LIFT_SCALE-b*x0
                self.observe('accumulator_ranges', key+'_undo_b', acc1)
                y1 = self.write24(key+'_undo_b', acc1, LIFT_FRAC)
                acc0 = x0*LIFT_SCALE-a*y1
                self.observe('accumulator_ranges', key+'_undo_a', acc0)
                y0 = self.write24(key+'_undo_a', acc0, LIFT_FRAC)
                self.frame['lifting_time_indices'][key+'_undo_b'] = second.cpu().tolist()
                self.frame['lifting_time_indices'][key+'_undo_a'] = first.cpu().tolist()
            else:
                acc0 = x0*LIFT_SCALE+a*x1
                self.observe('accumulator_ranges', key+'_a', acc0)
                y0 = self.write24(key+'_a', acc0, LIFT_FRAC)
                acc1 = x1*LIFT_SCALE+b*y0
                self.observe('accumulator_ranges', key+'_b', acc1)
                y1 = self.write24(key+'_b', acc1, LIFT_FRAC)
                self.frame['lifting_time_indices'][key+'_a'] = first.cpu().tolist()
                self.frame['lifting_time_indices'][key+'_b'] = second.cpu().tolist()
            result = result.index_copy(0, first, y0).index_copy(0, second, y1)
        self.observe('state_ranges', name+'_completed24', result)
        return result

    @torch.no_grad()
    def source_forward(self, x):
        self.c.clear_cached_graphs()
        self.frame = self.new_frame()
        self.ready = False
        identity = self.write24('I24', x[:, 0].double()*STATE_SCALE)
        q = self.lift(identity, 'source_lifting')
        self.i, self.q = (None, q) if self.kind == 'shared' else (identity, None)
        gate = self.compare('source', q.index_select(0, self.source_permutation))
        self.frame['source_gate'] = dict(nonzero=int(gate.sum()), elements=gate.numel())
        return self.emit(self.source, gate)

    @torch.no_grad()
    def conv_forward(self, x):
        m = self.matrices['U_conv2_theta']
        bits = x.ne(0).flatten(0, 1).double()
        accumulator = F.conv2d(bits, m['q'].reshape_as(self.c.conv_u), None,
            self.c.rank.stride, self.c.rank.padding, self.c.rank.dilation)
        self.observe('accumulator_ranges', 'binary_Conv2_U', accumulator)
        self.z = self.write24('Z24', accumulator, m['exponent']-STATE_FRAC)
        return self.original['conv'](x)

    def finish(self):
        if self.ready:
            return
        z = self.z[:, :, ::2, ::2]
        if self.kind == 'shared':
            bz = self.lift(z, 'residual_lifting')
            branch = self.channel_dot('F', bz, 'F_BZ_residual24')
            original = self.q
            addition = branch+self.c_q[:, :, None, None]
        else:
            branch = self.channel_dot('F', z, 'F_Z_residual24')
            original = self.i
            addition = branch+self.c_bn[None, :, None, None]
        accumulator = original[:, :, ::2, ::2]+addition
        self.observe('accumulator_ranges', 'anchor_merge', accumulator)
        anchor = self.write24('updated_anchor24', accumulator)
        # This is software materialization only, not a claim of two physical
        # persistent states. The numeric function retains untouched nonanchors.
        self.updated = original.clone()
        self.updated[:, :, ::2, ::2] = anchor
        self.observe('state_ranges', 'updated_full24', self.updated)
        latent = self.channel_dot('U_ped', anchor, 'PED_U24')
        if self.kind == 'shared':
            latent = self.lift(latent, 'PED_inverse_lifting', reverse=True)
        output = self.channel_dot('V_ped', latent, 'PED_V24')
        accumulator = output+self.projection_bias[None, :, None, None]
        self.observe('accumulator_ranges', 'PED_bias_merge', accumulator)
        self.continuous = self.write24('PED_bias_output24', accumulator)
        self.ready = True

    @torch.no_grad()
    def projection_forward(self, ignored_float_shadow):
        self.finish()
        return self.continuous.float()/STATE_SCALE

    @torch.no_grad()
    def consumer_forward(self, ignored_float_shadow):
        self.finish()
        gate = self.compare('consumer', self.updated.index_select(0, self.consumer_permutation))
        self.frame['consumer_gate'] = dict(nonzero=int(gate.sum()), elements=gate.numel())
        result = self.emit(self.consumer, gate)
        self.frames.append(self.frame)
        self.i = self.q = self.z = self.updated = self.continuous = None
        return result

    def range_report(self, names):
        return dict(scope='All full-frame values, not sampled. Each lifting half-step records the five actually written time rows in lifting_time_indices; completed24/updated_full24 are range-only aliases, not extra writes. All units are integer quanta; real values multiply2^-14.',
            static_constants=self.constant_metadata, static_lifting=self.static_lifting_report,
            frames=[dict(file=name, **row) for name, row in zip(names, self.frames)])

    report = range_report

