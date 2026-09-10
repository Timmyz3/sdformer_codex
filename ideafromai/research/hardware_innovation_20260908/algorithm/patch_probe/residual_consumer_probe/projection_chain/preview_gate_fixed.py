"""Gate-only preview reference with AZ24 and an exact integer V dot.

The actual theta*g source and saved dequantized U8 produce FP32 Z. The full
noncausal sparse A2 dot precedes V, then AZ is RNE/saturated signed24 f14.
K=2^15 V must be exactly integral. Float64 evaluates K*AZq with a static
absolute-sum bound below2^53, including every legal AZ24 value and every
reduction order. This is an integer arithmetic reference, not RTL.

The gate's new mathematical function is
  gain[h]*2^-29*dot[t,h] + rowsum(A2)[t]*beta[h] + temporal.b[t] >= theta.
gain, beta and rowsum are the existing helper's actual FP32 constants. Their
subsequent static relationship is evaluated as exact binary rationals, not
rounded intermediate FP32 operations: positive gain uses ceil/>=, negative
gain floor/<=, and zero gain is constant. The output is the actual theta*g;
the compiled dot threshold is not theta. This requires a fresh network AEE.

Conv1 returns a floating shadow dot*2^-29 so the real norm1 call can remain.
sn2 ignores that shadow and decides only from the cached integer dot. No
deletion of norm1, its tensors or executed GPU work is claimed. Each frame
writes only128 deterministic spatial samples shared by all ten times, plus
their integer comparison contract and expected gate. No full tensor capture.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'factor_completion_20260909/latent_stage_train16'))
from adapter import fp32_matmul
from preview_temporal_coordinates import PreviewTemporalForward, sparse_time_mix


AZ_FRAC, V_FRAC = 14, 15
AZ_MIN, AZ_MAX = -(1 << 23), (1 << 23)-1
DOT_FRAC = AZ_FRAC+V_FRAC


def compile_gate_thresholds(gain, beta, row_sum, bias, theta, lower, upper,
                            dot_fractional_bits=DOT_FRAC):
    """Compile exact saved-binary-constant comparisons against integer dot.

    Constants are -1 for a dynamic comparison, or the exact0/1 gate. A cutoff
    outside the proven dot domain is replaced by its nearest domain-equivalent
    finite cutoff: +true=lo, +false=hi+1, -true=hi, -false=lo-1. Thus nonzero
    gains can still use the comparison alone. gain==0 has sense0 and all times
    constant; only those rows require the constant override to define a gate.
    """
    f = lambda value: Fraction.from_float(float(value))
    gain, beta = np.asarray(gain).reshape(-1), np.asarray(beta).reshape(-1)
    row_sum, bias = np.asarray(row_sum).reshape(-1), np.asarray(bias).reshape(-1)
    thresholds = np.zeros((len(row_sum), len(gain)), np.int64)
    sense = np.sign(gain).astype(np.int8)
    constant = np.full(thresholds.shape, -1, np.int8)
    cutoffs = np.empty(thresholds.shape, dtype=object)
    for t in range(len(row_sum)):
        for h in range(len(gain)):
            tau = f(theta)-f(bias[t])-f(row_sum[t])*f(beta[h])
            if gain[h] == 0:
                constant[t, h] = int(tau <= 0)
                cutoffs[t, h] = 'zero-gain constant'
                continue
            boundary = tau*(1 << dot_fractional_bits)/f(gain[h])
            floor = boundary.numerator//boundary.denominator
            cutoff = -((-boundary.numerator)//boundary.denominator) if gain[h] > 0 else floor
            cutoffs[t, h] = str(cutoff)
            lo, hi = int(lower[h]), int(upper[h])
            if gain[h] > 0:
                if cutoff <= lo:
                    constant[t, h] = 1
                elif cutoff > hi:
                    constant[t, h] = 0
            else:
                if cutoff >= hi:
                    constant[t, h] = 1
                elif cutoff < lo:
                    constant[t, h] = 0
            if constant[t, h] < 0:
                thresholds[t, h] = cutoff
            elif gain[h] > 0:
                thresholds[t, h] = lo if constant[t, h] else hi+1
            else:
                thresholds[t, h] = hi if constant[t, h] else lo-1
    return dict(threshold=thresholds, sense=sense, constant_gate=constant,
                mathematical_cutoff=cutoffs.astype(str))


class PreviewGateFixed(PreviewTemporalForward):
    def __init__(self, pair, conv1, neuron, norm1_bn, output_directory, fp32_channel=True,
                 fractional_bits=AZ_FRAC):
        # Reuse precisely the prior helper's live-column selection, U8 source
        # interpretation, A2, effective temporal bias, gain/beta and rowsum.
        super().__init__(pair, conv1, neuron, norm1_bn, fp32_channel=fp32_channel)
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.az_frac = fractional_bits
        self.dot_frac = fractional_bits+V_FRAC
        self.dot = self.sample = None
        self.sample_positions = None
        try:
            k = self.v[:, :, 0, 0].double()*(1 << V_FRAC)
            if not torch.equal(k, k.round()):
                raise ValueError('The fixed preview requires exact integral 2^15*V; V is not rounded again.')
            bound = int(k.abs().sum(1).max())*(1 << 23)
            if bound >= (1 << 53):
                raise ValueError('Legal AZ24 dot/reduction exceeds the exact Float64 integer domain.')
            self.k = k
            self.k_numpy = k.to(torch.int64).cpu().numpy()
            positive = np.maximum(self.k_numpy, 0).sum(1, dtype=np.int64)
            negative = np.minimum(self.k_numpy, 0).sum(1, dtype=np.int64)
            self.dot_lower = positive*AZ_MIN+negative*AZ_MAX
            self.dot_upper = positive*AZ_MAX+negative*AZ_MIN
            self.compiled = compile_gate_thresholds(
                self.gain.cpu().numpy(), self.beta.cpu().numpy(), self.row_sum.cpu().numpy(),
                self.bias.cpu().numpy(), self.theta, self.dot_lower, self.dot_upper,
                dot_fractional_bits=self.dot_frac)
            self.threshold = torch.as_tensor(self.compiled['threshold'], device=k.device, dtype=torch.float64)
            self.sense = torch.as_tensor(self.compiled['sense'], device=k.device)
            self.constant_gate = torch.as_tensor(self.compiled['constant_gate'], device=k.device)
        except Exception:
            super().restore()
            raise
        self.metadata = dict(mode='preview_gate_fixed', T=10,
            active_rank=int(self.indices.numel()), saved_latent_slots=int(pair.u.shape[0]),
            active_indices=self.indices.cpu().tolist(), A_nonzero=int(self.A.ne(0).sum()),
            A_rank=int(torch.linalg.matrix_rank(self.A.double())),
            theta_source=float(pair.source_theta), theta_output=self.theta,
            source='Saved dequantized U8 consumes actual theta*g once; no extra theta multiplication.',
            temporal=f'All T10 available; exact static A2 nonzeros evaluated as FP32 sparse dot with TF32 disabled, then RNE and signed24 f{self.az_frac} saturation.',
            V='Every live coefficient has exact integral K=2^15*V. No second quantization, clipping or approximate zero removal.',
            integer_dot='Float64 K@AZq, with all products and arbitrary partial sums exactly representable on the complete legal AZ24 domain.',
            integer_abs_sum_bound=bound, integer_abs_sum_bound_lt_2p53=True,
            AZ_format=dict(bits=24, fractional_bits=self.az_frac, minimum=AZ_MIN, maximum=AZ_MAX),
            gate=f'gain_h*2^-{self.dot_frac}*dot+(A2*1)_t*beta_h+temporal.b_t >= theta; exact rational static constants compile the integer cutoff, inclusive sense +/- and gain0/domain constants.',
            compile_numeric='gain, beta and row_sum match PreviewTemporalForward FP32 values; Fraction uses those exact saved binary values. Subsequent constant products/additions are rational, not FP32-rounding emulation.',
            bias_center='Only pair.temporal.b, the current adapter effective bias/center convention, is applied. The overridden native sn2 bias/center is not added.',
            fp32_channel=fp32_channel,
            channel_numeric=('U producer uses FP32/cuDNN TF32 off; other cuDNN settings unchanged.' if fp32_channel else
                             'U producer retains existing cuDNN policy. A2 time and integer K dot rules are unchanged.'),
            shadow=f'Conv1 returns roundFP32(dot*2^-{self.dot_frac}); original fixed norm1 runs. sn2 never reads its values. All actual downstream Conv2/BN2/residual/PED calls remain.',
            sampling='Uniform floor-spaced128 flattened spatial indices including endpoints (all positions if smaller); the exact same positions at each T. Rows are time-major, then spatial-index order.',
            capture='sample_0000.npz etc: AZq/time/spatial_index, full comparison contract and this function expected_gate; report maps frame identities.',
            claim='New fixed-AZ gate function needs fresh AEE/activity. No claim of frozen/native bit equivalence, RTL performance or software-shadow deletion.')

    def observe(self, name, value):
        v = value.detach()
        self.frame['ranges'][name] = dict(minimum=float(v.min()), maximum=float(v.max()),
            per_T_max_abs=v.reshape(10, -1).abs().amax(1).cpu().tolist())

    @torch.no_grad()
    def conv_forward(self, x):
        self.frame = dict(ranges={}, source_shape=list(x.shape))
        z = self.channel_conv(x.flatten(0, 1), self.u, None,
            self.conv1.stride, self.conv1.padding, self.conv1.dilation)
        az = sparse_time_mix(self.A, z, self.supports)
        rounded = torch.round(az.double()*(1 << self.az_frac))
        low, high = rounded < AZ_MIN, rounded > AZ_MAX
        self.frame['saturation'] = dict(elements=rounded.numel(), low=int(low.sum()), high=int(high.sum()),
            low_by_T=low.reshape(10, -1).sum(1).cpu().tolist(),
            high_by_T=high.reshape(10, -1).sum(1).cpu().tolist())
        az_q = rounded.clamp(AZ_MIN, AZ_MAX)
        height, width = az_q.shape[-2:]
        positions = np.linspace(0, height*width-1, min(128, height*width), dtype=np.int64)
        self.sample_positions = torch.as_tensor(positions, device=az_q.device)
        count = len(positions)
        sampled = az_q.flatten(2).index_select(2, self.sample_positions).permute(0, 2, 1)
        self.sample = dict(az_q=sampled.reshape(10*count, -1).to(torch.int32).cpu().numpy(),
            time=np.repeat(np.arange(10, dtype=np.int8), count),
            spatial_index=np.tile(positions, 10), spatial_shape=np.array([height, width], np.int64),
            active_indices=self.indices.cpu().numpy(), theta_output=np.array(self.theta),
            threshold=self.compiled['threshold'], sense=self.compiled['sense'],
            constant_gate=self.compiled['constant_gate'])
        # Matrix direction: [H,R] @ [R,T*height*width] -> [T,H,height,width].
        with fp32_matmul():
            dot = self.k@az_q.permute(1, 0, 2, 3).reshape(az_q.shape[1], -1)
        self.dot = dot.reshape(self.k.shape[0], 10, height, width).permute(1, 0, 2, 3)
        raw = (self.dot*(2.0**-self.dot_frac)).float()
        self.observe('Z_fp32', z)
        self.observe('A2_Z_fp32', az)
        self.observe('AZq_integer', az_q)
        self.observe('K_dot_integer', self.dot)
        self.observe('raw_shadow_fp32', raw)
        return raw.reshape(x.shape[0], x.shape[1], raw.shape[1], height, width)

    @torch.no_grad()
    def neuron_forward(self, y):
        threshold = self.threshold[:, :, None, None]
        positive = self.sense[None, :, None, None] > 0
        gate = torch.where(positive, self.dot >= threshold, self.dot <= threshold)
        constants = self.constant_gate[:, :, None, None]
        gate = torch.where(constants >= 0, constants == 1, gate)
        out = (gate.float()*self.theta).unsqueeze(1)
        sampled = gate.flatten(2).index_select(2, self.sample_positions).permute(0, 2, 1)
        self.sample['expected_gate'] = sampled.reshape(-1, gate.shape[1]).cpu().numpy()
        path = self.output_directory/f'sample_{len(self.frames):04d}.npz'
        np.savez_compressed(path, **self.sample)
        self.frame.update(sample_file=path.name, sampled_spatial_points=self.sample['az_q'].shape[0]//10,
            output_gates=out.numel(), nonzero_gates=int(gate.sum()), ignored_norm1_shape=list(y.shape))
        self.frames.append(self.frame)
        self.pair.last_counts = dict(gates=out.numel(), mode='Full integer K dot / fixed AZ24 preview gate')
        self.pair.shared_raw = self.pair.empty = None
        self.dot = self.sample = self.sample_positions = None
        return out

    def export_constants(self):
        values = super().export_constants()
        values.pop('norm1_temporal_correction')  # Not part of this digital gate.
        values.update(K=self.k_numpy, AZ_fractional_bits=np.array(self.az_frac),
            V_fractional_bits=np.array(V_FRAC), dot_fractional_bits=np.array(self.dot_frac),
            dot_lower=self.dot_lower, dot_upper=self.dot_upper, **self.compiled)
        return values

    def range_report(self, names):
        return dict(mode=self.metadata['mode'], capture_directory=str(self.output_directory),
            frames=[dict(file=str(name), **frame) for name, frame in zip(names, self.frames)],
            saturation_totals={key:sum(frame['saturation'][key] for frame in self.frames)
                               for key in ('elements', 'low', 'high')},
            comparison='Thresholds apply directly to K@AZq integers; sense+1 is >=, sense-1 is <=; constant_gate>=0 overrides the comparison.',
            sample_scope='Samples and expected_gate belong only to this new complete integer-dot function; no future answer is supplied to its gate computation.',
            scope=self.metadata['claim'])

    def report(self, names):
        return self.range_report(names)

    def restore(self):
        super().restore()
        self.dot = self.sample = self.sample_positions = None
