"""Train-only prefix calibration and fixed-k physical-group acceptance probe.

This is the ordinary independent-prediction plus group-AND baseline. Full-C
values are used for calibration labels or held-out scoring, never prediction.
No GPU, training, network AEE, RTL or cycle evaluation is performed here.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.dont_write_bytecode = True
from group_completion_opportunity import ROOT, read_torch

KS = [(0, 1), (1, 8), (1, 4), (1, 2), (1, 1), (2, 1)]
PREFIXES = [(96, 4), (192, 2)]


def fraction(n, d):
    return float(n/d) if d else None


def required_signed_bits(lo, hi):
    bits = 1
    while lo < -(1 << (bits-1)) or hi >= 1 << (bits-1):
        bits += 1
    return bits


def main():
    started = time.monotonic()
    alg = ROOT/'algorithm'
    cache = read_torch(alg/'nrv_cost_probe/s2b3_train.pt')
    q = read_torch(alg/'pruning_probe/original_initial.pt')
    codes = cache['start_codes']
    W, B, E, tau = (np.asarray(q[k], dtype=np.int64) for k in
                   ('weight_int8', 'B_int8', 'E_int8', 'threshold_int32'))
    F, K, P, C = codes.shape
    H, T = W.shape[0], B.shape[0]
    assert (F, K, P, C, H, T) == (32, 32, 4, 384, 1536, 10)
    pos = cache['position_indices']
    assert np.array_equal(pos, pos[:, :1]+np.arange(P)) and np.all(pos[:, 0] % P == 0)
    Wf, Bf, Ef = W.astype(np.float64), B.astype(np.float64), E.astype(np.float64)
    B8 = B @ E.T

    def project(source, end):
        S = np.transpose(Ef[source[:, :end]], (0, 2, 1)) @ Wf[:, :end].T
        U = np.transpose(Bf @ S, (0, 2, 1))
        return U.astype(np.int64)  # All operands/partial sums are bounded integers.

    ntrain = 24*K*P
    sums = {end: np.zeros((H, T), dtype=np.int64) for end, _ in PREFIXES}
    squares = {end: np.zeros((H, T), dtype=np.int64) for end, _ in PREFIXES}
    residual_extrema = {end: [0, 0] for end, _ in PREFIXES}
    checked = 0
    for f in range(24):
        source = codes[f].reshape(K*P, C)
        full = project(source, C)
        for end, factor in PREFIXES:
            partial = project(source, end)
            if f == 0:
                chosen = [0, 1, K*P-1]
                class_sums = np.stack([(source[chosen, :end] == j).astype(np.int64)
                                       @ W[:, :end].T for j in range(8)], axis=-1)
                reference = class_sums @ B8.T
                assert np.array_equal(partial[chosen], reference)
                checked += reference.size
            residual = full-factor*partial
            sums[end] += residual.sum(0)
            squares[end] += np.square(residual).sum(0)
            residual_extrema[end][0] = min(residual_extrema[end][0], int(residual.min()))
            residual_extrema[end][1] = max(residual_extrema[end][1], int(residual.max()))

    compiled = {}
    for end, factor in PREFIXES:
        # Exact ceil(tau - mean_residual) without an offline float rounding tie.
        threshold = tau.T-np.floor_divide(sums[end], ntrain)
        mean = sums[end].astype(np.float64)/ntrain
        variance = np.maximum(squares[end].astype(np.float64)/ntrain-mean*mean, 0)
        std = np.sqrt(variance)
        exponent = np.ceil(np.log2(np.maximum(std, 1))).astype(np.int64)
        scale = np.left_shift(np.int64(1), exponent)
        compiled[end] = (threshold, scale)
        # Static full-code-alphabet interval for the deployed scaled margin.
        wp, wn = np.maximum(W[:, :end], 0).sum(1), np.minimum(W[:, :end], 0).sum(1)
        lo = factor*(wp[:, None]*B8.min(1)+wn[:, None]*B8.max(1))-threshold
        hi = factor*(wp[:, None]*B8.max(1)+wn[:, None]*B8.min(1))-threshold
        compiled[end] = {'threshold': threshold, 'scale': scale,
                         'exponent': exponent, 'std_min': float(std.min()),
                         'std_max': float(std.max()),
                         'margin_range_static': [int(lo.min()), int(hi.max())],
                         'margin_signed_bits_static': required_signed_bits(int(lo.min()), int(hi.max()))}

    totals = {}
    for end, _ in PREFIXES:
        totals[end] = [{'k': num/den, 'k_fraction': [num, den],
                        **{name: 0 for name in (
                            'groups', 'true_zero_groups', 'accepted_groups',
                            'accepted_groups_with_any_wrong_gate',
                            'accepted_false_negative_gates', 'accepted_false_positive_gates',
                            'accepted_true_nonzero_gates', 'accepted_predicted_nonzero_gates',
                            'predicted_zero_groups', 'accepted_predicted_zero_groups',
                            'accepted_predicted_zero_true_zero_groups',
                            'accepted_predicted_zero_missed_nonzero_gates',
                            'accepted_predicted_nonzero_groups',
                            'accepted_predicted_nonzero_groups_with_any_wrong_gate')}} for num, den in KS]
    per_frame = []
    for f in range(24, 32):
        source = codes[f].reshape(K*P, C)
        truth = (project(source, C) >= tau.T[None, :, :]).reshape(K, P, H//8, 8, T)
        true_active = truth.sum(axis=(1, 3, 4))
        frame_record = {'frame': cache['frames'][f], 'prefixes': {}}
        for end, factor in PREFIXES:
            config = compiled[end]
            # Predictor inputs below use only prefix codes and training constants.
            margin = factor*project(source, end)-config['threshold'][None, :, :]
            prediction = (margin >= 0).reshape(K, P, H//8, 8, T)
            pred_active = prediction.sum(axis=(1, 3, 4))
            # Truth is accessed only in the scorer after prediction is complete.
            fn = (truth & ~prediction).sum(axis=(1, 3, 4))
            fp = (~truth & prediction).sum(axis=(1, 3, 4))
            wrong = (fn+fp) > 0
            rows = []
            for index, (num, den) in enumerate(KS):
                # Exactly min_320 |margin|/scale >= k, expressed with integer shifts.
                certain = den*np.abs(margin) >= num*config['scale'][None, :, :]
                accept = certain.reshape(K, P, H//8, 8, T).all(axis=(1, 3, 4))
                pred_zero = pred_active == 0
                zero_accept = accept & pred_zero
                record = {
                    'groups': int(accept.size), 'true_zero_groups': int((true_active == 0).sum()),
                    'accepted_groups': int(accept.sum()),
                    'accepted_groups_with_any_wrong_gate': int((accept & wrong).sum()),
                    'accepted_false_negative_gates': int(fn[accept].sum()),
                    'accepted_false_positive_gates': int(fp[accept].sum()),
                    'accepted_true_nonzero_gates': int(true_active[accept].sum()),
                    'accepted_predicted_nonzero_gates': int(pred_active[accept].sum()),
                    'predicted_zero_groups': int(pred_zero.sum()),
                    'accepted_predicted_zero_groups': int(zero_accept.sum()),
                    'accepted_predicted_zero_true_zero_groups': int((zero_accept & (true_active == 0)).sum()),
                    'accepted_predicted_zero_missed_nonzero_gates': int(true_active[zero_accept].sum()),
                    'accepted_predicted_nonzero_groups': int((accept & ~pred_zero).sum()),
                    'accepted_predicted_nonzero_groups_with_any_wrong_gate': int((accept & ~pred_zero & wrong).sum())}
                for key, value in record.items():
                    totals[end][index][key] += value
                rows.append({'k': num/den, **record})
            frame_record['prefixes'][str(end)] = rows
        per_frame.append(frame_record)

    cases = []
    for end, factor in PREFIXES:
        config = compiled[end]
        for record in totals[end]:
            a = record['accepted_groups']
            z = record['accepted_predicted_zero_groups']
            zcorrect = record['accepted_predicted_zero_true_zero_groups']
            record.update(
                acceptance_fraction=fraction(a, record['groups']),
                any_gate_error_given_accept=fraction(record['accepted_groups_with_any_wrong_gate'], a),
                false_negative_fraction_of_accepted_true_nonzero=fraction(record['accepted_false_negative_gates'], record['accepted_true_nonzero_gates']),
                false_positive_fraction_of_accepted_predicted_nonzero=fraction(record['accepted_false_positive_gates'], record['accepted_predicted_nonzero_gates']),
                accepted_zero_prediction_precision=fraction(zcorrect, z),
                accepted_zero_prediction_recall=fraction(zcorrect, record['true_zero_groups']),
                any_gate_error_given_accepted_nonzero_prediction=fraction(record['accepted_predicted_nonzero_groups_with_any_wrong_gate'], record['accepted_predicted_nonzero_groups']))
        threshold, exponent = config['threshold'], config['exponent']
        cases.append({
            'prefix_C': end, 'fixed_factor': factor,
            'calibration': {'positions': ntrain, 'axis': 'one constant per (t,h), shared across positions',
                'residual_range': residual_extrema[end], 'residual_std_range': [config['std_min'], config['std_max']],
                'threshold_range': [int(threshold.min()), int(threshold.max())],
                'threshold_required_signed_bits': required_signed_bits(int(threshold.min()), int(threshold.max())),
                'scale_exponent_range': [int(exponent.min()), int(exponent.max())],
                'threshold_t_h': threshold.T.tolist(), 'scale_log2_t_h': exponent.T.tolist()},
            'deployed_scaled_margin_static_range': config['margin_range_static'],
            'deployed_scaled_margin_required_signed_bits': config['margin_signed_bits_static'],
            'held_out_results': totals[end]})
    result = {
        'kind': 'PREFIX_INDEPENDENT_GATE_PREDICTION_AND_320_GATE_AND_BASELINE',
        'scope': 's2b3 original saved integer student, full C384 labels and 32 captured intact P4 blocks/frame; no validation-dataset AEE',
        'train_frames': cache['frames'][:24], 'held_out_frames': cache['frames'][24:],
        'split_limit': 'Predictor calibration split only: these are existing model-training frames, not a newly independent test set.',
        'source': 'algorithm/nrv_cost_probe/s2b3_train.pt start_codes',
        'parameters': 'algorithm/pruning_probe/original_initial.pt W/B/E/tau',
        'theta': 'Saved source theta is folded into W; output theta is retained separately from decision tau. This probe predicts gate words, including nonzero ones.',
        'calibration_formula': 'r=U_full-f*U_prefix; tau_prime=ceil(tau-mean_train(r))=tau-floor(sum_train(r)/n); scale=2^ceil(log2(max(population_std_train(r),1)))',
        'prediction_formula': 'm=f*U_prefix-tau_prime; predicted_gate=(m>=0); accept_group=AND_320(den_k*abs(m)>=num_k*scale)',
        'deployment_arithmetic': 'f, scales and k denominators are powers of two; no runtime division. Constants are integer. A folded comparison against ceil(tau_prime/f) is possible for the gate alone, but group confidence still needs its exact scaled-margin equivalent.',
        'calibration_numerics': 'Integer residual sums/squared sums; exact integer threshold rounding; float64 population standard deviation rounded upward to integer power-of-two scale.',
        'numerical_check': {'independent_int64_prefix_values': checked,
                            'reference': 'Complete prefix class sums times B8 match rank-six reduction exactly; full-C reference previously checked by group_completion_opportunity.py.'},
        'not_claimed': ['safe/bit-exact early termination', 'AEE or network accuracy', 'cycles/speedup/energy', 'a new mechanism beyond independent prediction plus group AND'],
        'cost_boundary': [
            'Calibration uses full-C labels offline only; predictor runtime cannot read suffix codes, final U, true gates or future confidence.',
            'The check occupies the same PE consumer datapath and must preserve original S for rejected groups. Accepted nonzero words must also be emitted.',
            'Original final tau remains needed on fallback; additional prefix thresholds and scale exponents need storage/reads. They cannot overwrite original tau for free.',
            'Static margin width is reported; the factor-times-prefix implementation is not automatically covered by the existing Acc24 proof.',
            'Fixed all-zero output/whole-MLP deletion and static width controls remain necessary; a high gate-zero fraction alone is not useful accuracy evidence.'],
        'cases': cases, 'held_out_per_frame_counts': per_frame,
        'elapsed_cpu_seconds': time.monotonic()-started}
    target = Path(__file__).with_suffix('.json')
    target.write_text(json.dumps(result, indent=2)+'\n')
    for case in cases:
        print('C', case['prefix_C'], 'margin_bits', case['deployed_scaled_margin_required_signed_bits'], flush=True)
        for row in case['held_out_results']:
            keys = ('k', 'accepted_groups', 'acceptance_fraction', 'any_gate_error_given_accept',
                    'accepted_false_negative_gates', 'accepted_false_positive_gates',
                    'accepted_predicted_zero_groups', 'accepted_zero_prediction_precision',
                    'accepted_zero_prediction_recall', 'accepted_predicted_nonzero_groups')
            print(json.dumps({k: row[k] for k in keys}), flush=True)
    print('seconds', result['elapsed_cpu_seconds'], 'output', target, flush=True)


if __name__ == '__main__':
    main()
