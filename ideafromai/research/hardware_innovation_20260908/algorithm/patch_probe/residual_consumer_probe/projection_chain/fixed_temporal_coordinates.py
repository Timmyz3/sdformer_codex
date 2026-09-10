"""Fixed-format evaluation of residual/PSN/PED coordinates, not an RTL model.

Completed values: signed24, f14, RNE then saturate. Each static matrix uses
signed16 and f=floor(log2(32767/maxabs)); zero matrices use f=0. Dot products
use signed48 integer semantics, simulated by Float64, with a compile-time
bound below2^47. Every completed matrix output and residual merge writes24.
Source/consumer theta amplitudes are unchanged; decision thresholds are
separate exact rounded integer cutoffs compiled from bias/center/theta.

Shared: retain Q24=As*I24, update anchors with F*(As*Z24)+As*constant,
then U followed by inverse(quantized As) and V for continuous PED output.
Others: retain I24, update anchors with F*Z24+constant, then U,V.
Independent consumer uses Ap16; diagonal gains become signed comparisons.
R2 uses R16*x -> two24-bit latent values and Lbar16=L/e ->48-bit decision,
including x[P] in that decision. e=0 keeps the corresponding L row directly.
There is no extra diagonal multiplication in this R2 execution (20+20).

The frozen preview remains outside this helper. Original FP Conv2/BN/add
are retained as software shadows to preserve calls/shapes; the two real PED
consumers never use that floating shadow. The rest of the network is FP.
No claim of native equivalence, integerization of the whole net or cycles.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import torch
import torch.nn.functional as F

STATE_BITS, STATE_FRAC = 24, 14
STATE_MIN, STATE_MAX = -(1 << 23), (1 << 23)-1
ACC_LIMIT = 1 << 47
STATE_SCALE = 1 << STATE_FRAC


def as_numpy(value):
    return value.detach().double().cpu().numpy() if torch.is_tensor(value) else np.asarray(value, np.float64)


def tau_fractions(module):
    """Exact binary values of the saved constants; amplitude is not tau."""
    theta = Fraction.from_float(float(module.thresh.detach()))
    bias = as_numpy(module.bias).reshape(10)
    center = (np.broadcast_to(as_numpy(module.center).reshape(-1), (10,))
              if module.center_mode != 'zero' else np.zeros(10))
    return [theta-Fraction.from_float(float(b))+Fraction.from_float(float(c))
            for b,c in zip(bias, center)]


def scaled_fraction(value, exponent):
    return value*(1 << exponent) if exponent >= 0 else value/Fraction(1 << -exponent)


def compile_comparisons(tau, gains, exponent, lower, upper, zero_is_constant=True):
    """Static exact ceil/floor, inclusive comparisons; no near-zero cutoff."""
    thresholds, directions, constants, unbounded = [], [], [], []
    for t, (value, gain) in enumerate(zip(tau, gains)):
        gain = Fraction.from_float(float(gain))
        if gain == 0 and zero_is_constant:
            thresholds.append(0); directions.append(1)
            constants.append(int(value <= 0)); unbounded.append('zero-gain constant')
            continue
        if gain == 0:  # R2 row with no diagonal: its unnormalized L remains.
            gain = Fraction(1)
        direction = 1 if gain > 0 else -1
        fraction = scaled_fraction(value/gain, exponent)
        floor = fraction.numerator//fraction.denominator
        cutoff = -((-fraction.numerator)//fraction.denominator) if direction > 0 else floor
        lo, hi = int(lower[t]), int(upper[t])
        constant = -1
        if direction > 0:
            if cutoff <= lo: constant = 1
            elif cutoff > hi: constant = 0
        else:
            if cutoff >= hi: constant = 1
            elif cutoff < lo: constant = 0
        thresholds.append(cutoff if constant < 0 else 0)
        directions.append(direction); constants.append(constant); unbounded.append(str(cutoff))
    return dict(threshold=np.asarray(thresholds, np.int64),
        direction=np.asarray(directions, np.int8), constant=np.asarray(constants, np.int8),
        mathematical_cutoff=np.asarray(unbounded), scale_exponent=exponent)


class FixedTemporalForward:
    def __init__(self, controller, sn2_theta):
        self.c, self.source, self.consumer = controller, controller.source, controller.consumer
        self.device = self.source.weight.device
        self.original = dict(source=self.source.forward, consumer=self.consumer.forward,
            conv=controller.conv.forward, projection=controller.projection.forward)
        if controller.bn.training or not controller.bn.track_running_stats:
            raise ValueError('The fixed residual coordinate function requires the current fixed eval BN2.')
        self.kind = ('shared' if controller.shared_parameterization is not None else
            'diag_r2' if controller.raw_parameterization is not None and controller.raw_parameterization.residual else
            'diag' if controller.raw_parameterization is not None else 'independent')
        self.matrices, self.matrix_metadata, self.constants, self.constant_metadata = {}, {}, {}, {}
        self.comparisons, self.comparison_metadata = {}, {}
        self.r2_alignment = None
        self.frames, self.frame = [], {}
        self.i = self.q = self.z = self.updated = self.continuous = None
        self.ready = False
        self.compile_matrix('As', self.source.weight)
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
        source_tau, consumer_tau = tau_fractions(self.source), tau_fractions(self.consumer)
        lo24, hi24 = [STATE_MIN]*10, [STATE_MAX]*10
        self.compile_gate('source', source_tau, np.ones(10), STATE_FRAC, lo24, hi24)
        if self.kind == 'shared':
            mapping = controller.shared_parameterization
            self.permutation = mapping.permutation.detach().clone()
            a = self.matrices['As']
            ahat = np.ldexp(a['q_numpy'].astype(np.float64), -a['exponent'])
            # Recompute from the quantized source matrix, not native source A.
            self.compile_matrix('As_inverse', np.linalg.inv(ahat))
            accumulator = a['q_numpy'].sum(1, dtype=np.int64)[:, None]*as_numpy(self.c_bn)[None]
            self.c_q = self.constant24('As_BN2_constant', np.ldexp(accumulator, -a['exponent']))
            self.compile_gate('consumer', consumer_tau, as_numpy(mapping.d), STATE_FRAC, lo24, hi24)
        elif self.kind == 'independent':
            self.compile_matrix('Ap', self.consumer.weight)
            a = self.matrices['Ap']
            self.compile_gate('consumer', consumer_tau, np.ones(10), STATE_FRAC+a['exponent'], a['lower'], a['upper'])
        else:
            mapping = controller.raw_parameterization
            self.permutation = mapping.permutation.detach().clone()
            e = as_numpy(mapping.e)
            if self.kind == 'diag':
                self.compile_gate('consumer', consumer_tau, e, STATE_FRAC, lo24, hi24)
            else:
                self.compile_matrix('R', mapping.right)
                left = as_numpy(mapping.left)
                normalized = np.divide(left, e[:, None], out=left.copy(), where=e[:, None] != 0)
                self.compile_matrix('Lbar', normalized)
                f = self.matrices['Lbar']['exponent']
                self.tail_shift, self.identity_shift = max(-f, 0), max(f, 0)
                self.diagonal_present = torch.as_tensor(e != 0, device=self.device)
                lower, upper = [], []
                for t in range(10):
                    l = int(self.matrices['Lbar']['lower'][t])*(1 << self.tail_shift)
                    h = int(self.matrices['Lbar']['upper'][t])*(1 << self.tail_shift)
                    if e[t] != 0:
                        l += STATE_MIN*(1 << self.identity_shift)
                        h += STATE_MAX*(1 << self.identity_shift)
                    lower.append(l); upper.append(h)
                aligned_bound = self.prove48('R2_aligned_decision', lower, upper)
                self.r2_alignment = dict(tail_shift=self.tail_shift, identity_shift=self.identity_shift,
                    lower=lower, upper=upper, abs_bound=aligned_bound, fits_signed48=True)
                self.compile_gate('consumer', consumer_tau, e, STATE_FRAC+self.identity_shift,
                                  lower, upper, zero_is_constant=False)
        self.constants.update(source_tau_real=np.asarray([float(v) for v in source_tau]),
            consumer_tau_real=np.asarray([float(v) for v in consumer_tau]),
            source_theta=np.asarray(float(self.source.thresh.detach())),
            consumer_theta=np.asarray(float(self.consumer.thresh.detach())), sn2_theta=np.asarray(float(sn2_theta)))
        if self.kind != 'independent':
            self.constants['consumer_permutation'] = self.permutation
        self.metadata = dict(mode='fixed_temporal', axis_kind=self.kind,
            completed_state='signed24 f14; RNE then saturate at every matrix completion and residual merge',
            coefficient_format='Each complete static matrix signed16 with exponent floor(log2(32767/maxabs)); zero matrix exponent0. RNE, no near-zero threshold.',
            in_flight='signed48 integers in Float64; proven whole-domain absolute sum below2^47, hence products/partial sums exact regardless of GEMM summation order. Not RTL.',
            matrices=self.matrix_metadata, static_constants=self.constant_metadata,
            comparisons=self.comparison_metadata, R2_aligned_decision=self.r2_alignment,
            merge_abs_bound=3*(1 << 23),
            compile_numeric='Live FP32 factors/BN constants promoted to Float64 for gain/F/offset compilation; quantized As is inverted in Float64 and independently quantized16.',
            updates='Only even/even anchors receive residual and BN2 constant; nonanchor whole BN branch deletion is preserved.',
            state_order=('I24 -> As -> Q24 -> source gate; Z24 -> As -> AZ24 -> F -> residual24; Q24+residual24+As(constant) -> Qprime24; U -> UQ24 -> As_inverse -> latent24 -> V -> output24 -> bias merge24'
                if self.kind == 'shared' else
                'I24 -> As -> Q24 -> source gate (Q discarded); Z24 -> F -> residual24; I24+residual24+constant -> X24; U -> latent24 -> V -> output24 -> bias merge24'),
            consumer_execution=('Ap16 dot48 directly compares a scaled threshold' if self.kind == 'independent' else
                'R16*x -> rank2 state24; Lbar16*latent plus shifted x[P] ->48-bit signed comparison,20+20 products; e=0 omits x[P] and retains L' if self.kind == 'diag_r2' else
                'Actual saved signed gain is compiled into inclusive >=ceil / <=floor thresholds; no runtime d/e multiplication; exactly zero gain is a constant gate'),
            shadow='Original FP Conv2, outer BN2 and identity add still run for software call/shape compatibility. Both PED consumers ignore their float input and use only cached integer coordinates. Shadow GPU work is not hardware workload.',
            outside_scope='Frozen preview Conv1/sn2 and the rest of the network keep their original numeric function. Source/consumer emit their own actual theta*g, not unit gates.',
            claim='One fixed-format new numeric function; fresh AEE/activity required. No full-network integer, native-equivalence, memory-removal, power or cycle claim.')
        self.source.forward, self.consumer.forward = self.source_forward, self.consumer_forward
        controller.conv.forward, controller.projection.forward = self.conv_forward, self.projection_forward

    def prove48(self, name, lower, upper):
        bound = max(max(abs(int(v)) for v in lower), max(abs(int(v)) for v in upper))
        if bound >= ACC_LIMIT:
            raise ValueError(f'{name} needs more than signed48 for the declared input domain: {bound}')
        return bound

    def compile_matrix(self, name, value, bits=False):
        value = as_numpy(value)
        maximum = float(np.max(np.abs(value)))
        exponent = math.floor(math.log2(32767/maximum)) if maximum else 0
        q = np.clip(np.rint(np.ldexp(value, exponent)), -32768, 32767).astype(np.int64)
        pos = np.maximum(q, 0).sum(1, dtype=np.int64)
        neg = np.minimum(q, 0).sum(1, dtype=np.int64)
        lower, upper = ((neg, pos) if bits else
                        (pos*STATE_MIN+neg*STATE_MAX, pos*STATE_MAX+neg*STATE_MIN))
        bound = self.prove48(name, lower, upper)
        self.matrices[name] = dict(q=torch.as_tensor(q, device=self.device, dtype=torch.float64),
            q_numpy=q, exponent=exponent, lower=lower, upper=upper)
        self.matrix_metadata[name] = dict(shape=list(q.shape), exponent=exponent,
            input_domain='0/1 source bits' if bits else 'signed24 [-8388608,8388607]',
            integer_nonzero=int(np.count_nonzero(q)), original_nonzero=int(np.count_nonzero(value)),
            dot_abs_bound=bound, fits_signed48=True)
        self.constants[name+'_q16'] = q.astype(np.int16)
        self.constants[name+'_exponent'] = np.asarray(exponent)

    def constant24(self, name, value_in_state_units):
        rounded = np.rint(value_in_state_units)
        q = np.clip(rounded, STATE_MIN, STATE_MAX)
        self.constant_metadata[name] = dict(elements=int(q.size),
            clipped_low=int(np.count_nonzero(rounded < STATE_MIN)), clipped_high=int(np.count_nonzero(rounded > STATE_MAX)),
            integer_min=int(q.min()), integer_max=int(q.max()))
        self.constants[name+'_q24'] = q.astype(np.int32)
        return torch.as_tensor(q, device=self.device, dtype=torch.float64)

    def compile_gate(self, name, tau, gains, exponent, lower, upper, zero_is_constant=True):
        compiled = compile_comparisons(tau, gains, exponent, lower, upper, zero_is_constant)
        self.comparisons[name] = {key: torch.as_tensor(compiled[key], device=self.device)
                                 for key in ('threshold', 'direction', 'constant')}
        for key,value in compiled.items():
            self.constants[name+'_'+key] = np.asarray(value)
        self.comparison_metadata[name] = dict(exponent=exponent,
            directions=compiled['direction'].tolist(), constants=compiled['constant'].tolist(),
            exact_cutoffs=compiled['mathematical_cutoff'].tolist(),
            scope='Exact rational binary saved constants; inclusive ceil/floor. Constant rows follow exact gain/domain logic, not a small-gain cutoff.')

    def observe(self, category, name, value):
        self.frame[category][name] = dict(integer_min=float(value.min()), integer_max=float(value.max()),
            per_leading_row_max_abs=value.reshape(value.shape[0], -1).abs().amax(1).cpu().tolist())

    def write24(self, name, accumulator, shift=0):
        scaled = torch.ldexp(accumulator, torch.as_tensor(-shift, device=self.device))
        rounded = torch.round(scaled)
        low, high = rounded < STATE_MIN, rounded > STATE_MAX
        result = rounded.clamp(STATE_MIN, STATE_MAX)
        self.frame['clip_counts'][name] = dict(elements=result.numel(), low=int(low.sum()), high=int(high.sum()))
        self.observe('state_ranges', name, result)
        return result

    def time_dot(self, key, value, name, complete=True):
        m = self.matrices[key]
        accumulator = (m['q']@value.reshape(value.shape[0], -1)).reshape(m['q'].shape[0], *value.shape[1:])
        self.observe('accumulator_ranges', name, accumulator)
        return self.write24(name, accumulator, m['exponent']) if complete else accumulator

    def channel_dot(self, key, value, name):
        m = self.matrices[key]
        t, _, height, width = value.shape
        flat = value.permute(1, 0, 2, 3).reshape(value.shape[1], -1)
        accumulator = (m['q']@flat).reshape(m['q'].shape[0], t, height, width).permute(1, 0, 2, 3)
        self.observe('accumulator_ranges', name, accumulator)
        return self.write24(name, accumulator, m['exponent'])

    def compare(self, name, value):
        compiled = self.comparisons[name]
        shape = (10,)+(1,)*(value.ndim-1)
        threshold = compiled['threshold'].reshape(shape).to(torch.float64)
        positive = compiled['direction'].reshape(shape) > 0
        constant = compiled['constant'].reshape(shape)
        decision = torch.where(positive, value >= threshold, value <= threshold)
        return torch.where(constant >= 0, constant.bool(), decision)

    @staticmethod
    def emit(module, gate):
        result = gate.to(module.thresh.dtype)*module.thresh.detach()
        result = result.unsqueeze(1)
        module.act_value = result.abs().reshape(10, -1).mean(1).sum()
        return result

    @torch.no_grad()
    def source_forward(self, x):
        self.frame = dict(clip_counts={}, state_ranges={}, accumulator_ranges={})
        self.ready = False
        identity = self.write24('I24', x[:, 0].double()*STATE_SCALE)
        q = self.time_dot('As', identity, 'As_I_Q24')
        self.i, self.q = (None, q) if self.kind == 'shared' else (identity, None)
        gate = self.compare('source', q)
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
        # This result is only a software shape/call shadow. No integer consumer
        # below reads it, the outer FP BN result, or the FP residual addition.
        return self.original['conv'](x)

    def finish(self):
        if self.ready:
            return
        z = self.z[:, :, ::2, ::2]
        if self.kind == 'shared':
            az = self.time_dot('As', z, 'As_Z_AZ24')
            branch = self.channel_dot('F', az, 'F_AZ_residual24')
            self.updated = self.q
            addition = branch+self.c_q[:, :, None, None]
        else:
            branch = self.channel_dot('F', z, 'F_Z_residual24')
            self.updated = self.i
            addition = branch+self.c_bn[None, :, None, None]
        self.updated[:, :, ::2, ::2] = self.write24('updated_anchor24',
            self.updated[:, :, ::2, ::2]+addition)
        latent = self.channel_dot('U_ped', self.updated[:, :, ::2, ::2], 'PED_U24')
        if self.kind == 'shared':
            latent = self.time_dot('As_inverse', latent, 'PED_inverse_latent24')
        output = self.channel_dot('V_ped', latent, 'PED_V24')
        self.continuous = self.write24('PED_bias_output24', output+self.projection_bias[None, :, None, None])
        self.ready = True

    @torch.no_grad()
    def projection_forward(self, ignored_float_shadow):
        self.finish()
        return self.continuous.float()/STATE_SCALE

    @torch.no_grad()
    def consumer_forward(self, ignored_float_shadow):
        self.finish()
        if self.kind == 'independent':
            decision_value = self.time_dot('Ap', self.updated, 'consumer_Ap48', complete=False)
        elif self.kind == 'diag_r2':
            latent = self.time_dot('R', self.updated, 'consumer_R_latent24')
            tail = self.time_dot('Lbar', latent, 'consumer_Lbar48', complete=False)
            decision_value = torch.ldexp(tail, torch.as_tensor(self.tail_shift, device=self.device))
            identity = self.updated.index_select(0, self.permutation)
            identity = torch.ldexp(identity, torch.as_tensor(self.identity_shift, device=self.device))
            decision_value = decision_value+identity*self.diagonal_present[:, None, None, None]
            self.observe('accumulator_ranges', 'consumer_R2_aligned48', decision_value)
        else:
            decision_value = self.updated.index_select(0, self.permutation)
        gate = self.compare('consumer', decision_value)
        self.frame['consumer_gate'] = dict(nonzero=int(gate.sum()), elements=gate.numel())
        result = self.emit(self.consumer, gate)
        self.frames.append(self.frame)
        self.i = self.q = self.z = self.updated = self.continuous = None
        return result

    def export_constants(self):
        return {key: value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
                for key,value in self.constants.items()}

    def range_report(self, names):
        return dict(scope='Actual integer ranges and per-write RNE/saturation counts. Leading rows are T except consumer_R_latent24, where they are the2 latent coordinates.',
            static_constants=self.constant_metadata,
            frames=[dict(file=name, **row) for name,row in zip(names, self.frames)])

    def restore(self):
        self.source.forward, self.consumer.forward = self.original['source'], self.original['consumer']
        self.c.conv.forward, self.c.projection.forward = self.original['conv'], self.original['projection']
        self.i = self.q = self.z = self.updated = self.continuous = None
