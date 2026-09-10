"""Compile a NEW common W8 / Aq14 integer bridge for joint-completion students.

The real captured FP Conv1 and fixed BN1 are folded into one per-H dyadic W8
convolution, following the earlier compiler's ordinary quantization baseline.
This changes the model and therefore needs separate network AEE.  It neither
overwrites nor inherits the old integer student's A, scalar residual moments,
gamma, or threshold tables.

Each saved model keeps its own prefix, Aq14, bias, per-(h,t) residual mean /
covariance and gamma.  Full and prefix predicates use integer Ui and offline
ceil/floor boundaries.  Constants are the exact rational values of the stated
saved/compiled IEEE numbers; there is no implicit runtime FP comparator.
The prefix negative boundary is inclusive (pred <= -radius), matching this
new trainer's abs(pred) >= radius.  Positive wins if both hold at radius zero.
The full predicate is always margin >= 0.  theta_decision and theta_amplitude
are separate fields even though this checkpoint gives them the same value.

CPU replay consumes same-forward source_gate_words and FP norm1 Y.  Its exact
integer convolution/PSN and boundary checks are not RTL or network accuracy.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
import torch

from evaluate_network import CompletionKernel


Q = 1 << 14


def rational(value):
    return Fraction.from_float(float(value))


def signed_bits(low, high):
    return max(int(high).bit_length()+1, (-int(low)-1).bit_length()+1, 1)


def fold_weights(parameter_file, source_theta):
    with np.load(parameter_file, allow_pickle=False) as z:
        weight = z['W1'].astype(np.float64)
        bn_gain = z['bn1_gamma'].astype(np.float64)/np.sqrt(
            z['bn1_var'].astype(np.float64)+float(z['bn1_eps']))
        bn_bias = (z['bn1_beta'].astype(np.float64)
                   + bn_gain*(z['conv1_bias'].astype(np.float64)-z['bn1_mean'].astype(np.float64)))
    folded = weight*float(source_theta)*bn_gain[:, None, None, None]
    peak = np.abs(folded).reshape(96, -1).max(1)
    exponent = np.zeros(96, dtype=np.int32)
    live = peak != 0
    exponent[live] = np.ceil(np.log2(peak[live]/127.)).astype(np.int32)
    scale = np.ldexp(np.ones(96), exponent)
    wq = np.clip(np.rint(folded/scale[:, None, None, None]), -127, 127).astype(np.int8)
    flat = wq.reshape(96, -1).astype(np.int64)
    y_low = np.minimum(flat, 0).sum(1)
    y_high = np.maximum(flat, 0).sum(1)
    return dict(weight_int8=wq, weight_scale=scale, weight_scale_exponent=exponent,
                bn_gain_fp64=bn_gain, bn_bias_fp64=bn_bias,
                Y_lower=y_low, Y_upper=y_high,
                theta_source=np.asarray(source_theta))


def accumulator_bounds(aq, y_low, y_high):
    positive = np.maximum(aq, 0).sum(1)[:, None]
    negative = np.minimum(aq, 0).sum(1)[:, None]
    return (positive*y_low[None]+negative*y_high[None],
            positive*y_high[None]+negative*y_low[None])


def clipped_boundary(value, lo, hi):
    return min(int(hi)+1, max(int(lo)-1, int(value)))


def compile_model(model_file, shared):
    kernel = CompletionKernel(model_file)
    assert kernel.source_theta == float(shared['theta_source'])
    aq = np.rint(kernel.a_cpu.numpy()*Q).astype(np.int64)
    bias = kernel.b_cpu.numpy()
    known = kernel.known_cpu.numpy().astype(bool)
    aq_prefix = aq*known[None]
    full_low, full_high = accumulator_bounds(aq, shared['Y_lower'], shared['Y_upper'])
    prefix_low, prefix_high = accumulator_bounds(aq_prefix, shared['Y_lower'], shared['Y_upper'])
    full_threshold = np.empty((10, 96), dtype=np.int64)
    prefix_positive = np.empty_like(full_threshold)
    prefix_negative = np.empty_like(full_threshold)
    full_center = np.empty((10, 96), dtype=np.float64)
    prefix_center = np.empty_like(full_center)
    residual_offset = kernel.offset.numpy().T
    radius = kernel.radius.numpy().T
    checks = 0
    for t in range(10):
        for h in range(96):
            rate = rational(shared['weight_scale'][h])/Q
            # The affine BN constant belongs only to the produced columns.
            # The unproduced part is already in the saved per-H predictor.
            full_const = (rational(bias[t])-rational(kernel.theta)
                          + rational(shared['bn_bias_fp64'][h])*int(aq[t].sum())/Q)
            prefix_const = (rational(residual_offset[t, h])
                            + rational(shared['bn_bias_fp64'][h])*int(aq_prefix[t].sum())/Q)
            rad = rational(radius[t, h])
            full_center[t, h], prefix_center[t, h] = float(full_const), float(prefix_const)
            full_threshold[t, h] = clipped_boundary(
                math.ceil(-full_const/rate), full_low[t, h], full_high[t, h])
            prefix_positive[t, h] = clipped_boundary(
                math.ceil((rad-prefix_const)/rate), prefix_low[t, h], prefix_high[t, h])
            prefix_negative[t, h] = clipped_boundary(
                math.floor((-rad-prefix_const)/rate), prefix_low[t, h], prefix_high[t, h])
            # Boundary proof is independent of whether these U values happened
            # to occur in the few captured frames.  Both endpoints and equality
            # sides are evaluated as exact rational affine expressions.
            for full in (True, False):
                lo = int((full_low if full else prefix_low)[t, h])
                hi = int((full_high if full else prefix_high)[t, h])
                thresholds = ([full_threshold[t, h]] if full else
                              [prefix_positive[t, h], prefix_negative[t, h]])
                points = {lo, hi, 0}
                for threshold in thresholds:
                    points.update((int(threshold)-1, int(threshold), int(threshold)+1))
                for u in points:
                    if not lo <= u <= hi:
                        continue
                    if full:
                        expect = rate*u+full_const >= 0
                        actual = u >= full_threshold[t, h]
                    else:
                        p = rate*u+prefix_const
                        expect = (p >= rad, p <= -rad)
                        actual = (u >= prefix_positive[t, h], u <= prefix_negative[t, h])
                    if expect != actual:
                        raise ArithmeticError((str(model_file), t, h, full, u, actual, expect))
                    checks += 1
    arrays = dict(**shared, temporal_int16=aq.astype(np.int16),
                  temporal_fractional_bits=np.asarray(14),
                  full_threshold=full_threshold,
                  prefix_threshold_positive=prefix_positive,
                  prefix_threshold_negative=prefix_negative,
                  prefix=np.asarray(kernel.prefix, dtype=np.int32),
                  support=kernel.support_cpu.numpy(), temporal_bias_fp32=bias,
                  theta_decision=np.asarray(kernel.theta), theta_amplitude=np.asarray(kernel.theta),
                  train_mean_fp32=kernel.mean_cpu.numpy(),
                  train_covariance_fp32=kernel.covariance_cpu.numpy(),
                  gamma_fp32=kernel.gamma_cpu.numpy(),
                  predictor_offset_fp32=residual_offset, predictor_radius_fp32=radius,
                  full_center_fp64_diagnostic=full_center,
                  prefix_center_fp64_diagnostic=prefix_center,
                  U_full_lower=full_low, U_full_upper=full_high,
                  U_prefix_lower=prefix_low, U_prefix_upper=prefix_high,
                  negative_comparison_inclusive=np.asarray(True),
                  positive_priority_at_zero_radius=np.asarray(True))
    table_values = np.concatenate((full_threshold.ravel(), prefix_positive.ravel(), prefix_negative.ravel()))
    summary = dict(model_file=str(model_file), prefix=kernel.prefix,
                   theta_amplitude=kernel.theta, theta_decision=kernel.theta,
                   actual_Aq_nonzeros=int(np.count_nonzero(aq)),
                   rational_boundary_cases=checks,
                   Y_lower=int(shared['Y_lower'].min()), Y_upper=int(shared['Y_upper'].max()),
                   Y_signed_bits=signed_bits(shared['Y_lower'].min(), shared['Y_upper'].max()),
                   U_lower=int(full_low.min()), U_upper=int(full_high.max()),
                   U_signed_bits=signed_bits(full_low.min(), full_high.max()),
                   threshold_signed_bits=signed_bits(table_values.min(), table_values.max()),
                   threshold_entries=int(table_values.size),
                   literal_48bit_threshold_bytes=int(table_values.size*6),
                   predicate='full: Ui>=F[t,h]; prefix: Ui>=P[t,h] or Ui<=N[t,h], positive first',
                   interpretation='new common-W8 student; saved FP train moments are retained, not recalibrated on valid')
    assert summary['Y_signed_bits'] <= 24 and summary['U_signed_bits'] <= 48
    return arrays, summary, kernel


def integer_convolution(words, wq):
    """Words G,K,P; output T,H,G,P.  Exact binary/signed-integer sum.

    FP64 BLAS only accelerates the host reference: every term and every sum
    is integral and far below 2^53.  It is not the proposed physical PE.
    """
    bits = ((words[None].astype(np.int32)
             >> np.arange(10, dtype=np.int32)[:, None, None, None]) & 1)
    g, _, p = words.shape
    vectors = bits.transpose(0, 1, 3, 2).reshape(10*g*p, 864)
    products = vectors.astype(np.float64) @ wq.reshape(96, 864).astype(np.float64).T
    assert np.array_equal(products, np.rint(products))
    return products.astype(np.int64).reshape(10, g, p, 96).transpose(0, 3, 1, 2)


def integer_gates(z, arrays):
    aq = arrays['temporal_int16'].astype(np.int64)
    prefix = arrays['prefix']
    full_u = np.tensordot(aq, z, axes=(1, 0))
    prefix_u = np.tensordot(aq[:, prefix], z[prefix], axes=(1, 0))
    full = full_u >= arrays['full_threshold'][:, :, None, None]
    positive = prefix_u >= arrays['prefix_threshold_positive'][:, :, None, None]
    negative = prefix_u <= arrays['prefix_threshold_negative'][:, :, None, None]
    accepted = positive | negative
    conditional = np.where(accepted, positive, full)
    return full, conditional, accepted, full_u, prefix_u


@torch.no_grad()
def replay(paths, shared, compiled):
    totals = {name: dict(frames=0, gates=0, full_float64_mismatches=0,
                        conditional_float64_mismatches=0,
                        full_changes_from_FP_student=0,
                        conditional_changes_from_FP_student=0,
                        accepted=0, conditional_changes_from_own_full=0,
                        source_Y_max_abs_difference_from_FP=0., frames_detail=[])
              for name in compiled}
    for path in paths:
        with np.load(path, allow_pickle=False) as capture:
            words = capture['source_gate_words']
            y_fp = capture['Y']
            frame_name = str(capture['frame_name'])
            assert float(capture['theta_source']) == float(shared['theta_source'])
        z = integer_convolution(words, shared['weight_int8'])
        y_new = (z*shared['weight_scale'][None, :, None, None]
                 + shared['bn_bias_fp64'][None, :, None, None])
        y_error = float(np.max(np.abs(y_new-y_fp)))
        for name, (arrays, _, kernel) in compiled.items():
            full, conditional, accepted, full_u, prefix_u = integer_gates(z, arrays)
            # Direct affine Float64 reference is diagnostic, not the boundary
            # proof.  All required ceil/floor/equality cases were proven above.
            rate = shared['weight_scale'][None, :, None, None]/Q
            full_direct = full_u*rate+arrays['full_center_fp64_diagnostic'][:, :, None, None]
            pred_direct = prefix_u*rate+arrays['prefix_center_fp64_diagnostic'][:, :, None, None]
            radius = arrays['predictor_radius_fp32'][:, :, None, None].astype(np.float64)
            accept_direct = np.abs(pred_direct) >= radius
            conditional_direct = np.where(accept_direct, pred_direct >= 0, full_direct >= 0)
            fp_y = torch.from_numpy(y_fp).permute(2, 1, 3, 0).reshape(-1, 8, 4, 10).contiguous()
            channels = kernel.channel_groups.repeat(words.shape[0], 1)
            fp_full = kernel.context_gates(fp_y, channels, 'exact').reshape(words.shape[0], 96, 4, 10)
            fp_cond = kernel.context_gates(fp_y, channels, 'conditional').reshape(words.shape[0], 96, 4, 10)
            fp_full, fp_cond = fp_full.permute(3, 1, 0, 2).numpy(), fp_cond.permute(3, 1, 0, 2).numpy()
            row = dict(file=frame_name, gates=int(full.size),
                       full_float64_mismatches=int(np.count_nonzero(full != (full_direct >= 0))),
                       conditional_float64_mismatches=int(np.count_nonzero(conditional != conditional_direct)),
                       full_changes_from_FP_student=int(np.count_nonzero(full != fp_full)),
                       conditional_changes_from_FP_student=int(np.count_nonzero(conditional != fp_cond)),
                       accepted=int(accepted.sum()),
                       conditional_changes_from_own_full=int(np.count_nonzero(conditional != full)))
            total = totals[name]
            total['frames'] += 1
            for key, value in row.items():
                if key != 'file':
                    total[key] += value
            total['source_Y_max_abs_difference_from_FP'] = max(
                total['source_Y_max_abs_difference_from_FP'], y_error)
            total['frames_detail'].append(row)
        print('INTEGER_REPLAY', frame_name, 'sampled_norm1_quantization_max_abs', y_error, flush=True)
    for total in totals.values():
        for key in ('full_changes_from_FP_student', 'conditional_changes_from_FP_student',
                    'accepted', 'conditional_changes_from_own_full'):
            total[key+'_fraction'] = total[key]/max(1, total['gates'])
        assert total['full_float64_mismatches'] == 0
        assert total['conditional_float64_mismatches'] == 0
    return totals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parameters', type=Path, required=True,
                   help='new full_capture4/capture/parameters.npz, containing real W1/BN1')
    p.add_argument('--model-files', type=Path, nargs='+', required=True)
    p.add_argument('--source-directory', type=Path, required=True)
    p.add_argument('--max-frames', type=int, default=4,
                   help='first N source capture files for reference replay, 0 for all; never used for fitting')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    first = CompletionKernel(args.model_files[0])
    shared = fold_weights(args.parameters, first.source_theta)
    args.output.mkdir(parents=True, exist_ok=True)
    compiled, reports = {}, {}
    for i, model_file in enumerate(args.model_files):
        name = f'{i:02d}_{model_file.parent.name}_{model_file.stem}'
        arrays, report, kernel = compile_model(model_file, shared)
        np.savez_compressed(args.output/(name+'.npz'), **arrays)
        compiled[name] = (arrays, report, kernel)
        reports[name] = report
    paths = sorted(args.source_directory.rglob('sampled_source.npz'))
    if args.max_frames:
        paths = paths[:args.max_frames]
    assert paths
    replays = replay(paths, shared, compiled)
    report = dict(complete=True, parameters=str(args.parameters),
                  source_files=[str(path) for path in paths],
                  scope='CPU same-capture source reconstruction and new-student integer predicates; no network AEE/RTL/PPA',
                  common_weight_quantization='theta_source and fixed BN1 gain folded into W1; per-H power-of-two scale; round-even, clip -127..127',
                  weight_zero_fraction=float((shared['weight_int8'] == 0).mean()),
                  weight_scale_min=float(shared['weight_scale'].min()),
                  weight_scale_max=float(shared['weight_scale'].max()),
                  constant_definition='exact rational values of FP64 folded BN offset and saved FP32 bias/predictor offset/radius; compiled integer thresholds define runtime',
                  numeric_change='common W8 defines a new student; float bias/moments not fit on this replay; original theta amplitude retained',
                  parent_preservation='original FP parent used for source/Y capture only; old integer contract never changed',
                  cost_boundary='INT24 Zi and INT48 Ui capacity checked; 3xT10xH96 threshold entries; finite ports/reads/comparator cost unmeasured',
                  compiled=reports, replay=replays, wall_seconds=time.monotonic()-started)
    (args.output/'result.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print('DONE', json.dumps(dict(compiled=reports, replay={k:{n:v for n,v in r.items() if n!='frames_detail'}
                                                          for k,r in replays.items()}), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
