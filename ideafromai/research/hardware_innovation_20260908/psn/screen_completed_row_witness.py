"""One bounded follow-up: a required completed A row witnesses one other row.

Pairs are compiled solely from A: abs(cosine)>=0.8, descending similarity,
greedy disjoint endpoints, at most five. The larger-norm row is the anchor.
Rank-0 static-kappa screening runs first; a witness is attempted only when both
members still require an original-A dot. No test input fits or selects a pair.

For a_j=c*a_i+e, u=a_i*x and d=b_j-center_j-theta, the Cauchy test can be
compiled into F=(alpha*u+beta)*u+gamma-lambda*q, q=||x||^2,
alpha=||a_j||^2/||a_i||^2, beta=2*c*d, gamma=d^2, lambda=||e||^2.
The predicted sign uses a static comparison on u. This is three runtime
multiplies, three adds and two comparisons per attempt, not a free norm update.
A conservative float64 margin is folded into lambda offline. The ordinary
six-multiply literal form is also counted. These are real-arithmetic/float64
opportunities, not certified frozen GPU FP32 or RTL results.
"""
from pathlib import Path
import json
import math
import sys
import numpy as np

sys.dont_write_bytecode = True
from screen_psn_residual_bound import (HERE, CKPT, MODULES, read_checkpoint,
                                      reconstructed_fc1, hook_data, static_norm_constants)


def compile_pairs(a, offset, theta):
    t = len(a)
    norm2 = np.sum(a*a, axis=1)
    possible = []
    for i in range(t):
        for j in range(i+1, t):
            rho = float(a[i]@a[j]/np.sqrt(norm2[i]*norm2[j]))
            if abs(rho) >= .8:
                possible.append((-abs(rho), i, j, rho))
    used, result = set(), []
    for _, i0, j0, rho in sorted(possible):
        if i0 in used or j0 in used:
            continue
        i, j = (i0, j0) if norm2[i0] >= norm2[j0] else (j0, i0)
        c = float(a[j]@a[i]/norm2[i])
        e = a[j]-c*a[i]
        e2 = float(e@e)
        d = float(offset[j, 0]-theta)
        result.append({'anchor': i, 'partner': j, 'cosine': rho, 'c': c,
                       'anchor_norm2': float(norm2[i]), 'partner_norm2': float(norm2[j]),
                       'residual_norm2': e2, 'alpha': float(norm2[j]/norm2[i]),
                       'beta': 2*c*d, 'gamma': d*d, 'lambda_guarded': e2*(1+1e-10),
                       'partner_static_margin': d,
                       'predicted_sign_anchor_threshold': -d/c if c else None})
        used |= {i, j}
        if len(result) == 5:
            break
    return result


def run(key, state, data):
    x, captured_bits, provenance = data
    a = state[MODULES[key]+'.weight'].astype(float)
    offset = (state[MODULES[key]+'.bias']-state[MODULES[key]+'.center']).astype(float)
    theta = float(state[MODULES[key]+'.thresh'])
    t, n = x.shape
    q = np.sum(x*x, axis=0)
    raw = a@x
    truth = raw+offset >= theta
    kappa, _, on, zero = static_norm_constants(a, offset, theta)
    r0_cert = np.where(on[:, None], q <= kappa[:, None], q < kappa[:, None]) | zero[:, None]
    failed0 = ~r0_cert
    pairs = compile_pairs(a, offset, theta)
    attempt_count_per_position = np.zeros(n, dtype=int)
    saves_per_position = np.zeros(n, dtype=int)
    pair_rows = []
    all_errors = 0
    for pair in pairs:
        i, j = pair['anchor'], pair['partner']
        attempt = failed0[i] & failed0[j]
        u = raw[i]
        c, d = pair['c'], pair['partner_static_margin']
        if c > 0:
            predicted = u >= pair['predicted_sign_anchor_threshold']
        elif c < 0:
            predicted = u <= pair['predicted_sign_anchor_threshold']
        else:
            predicted = np.full(n, d >= 0)
        f = ((pair['alpha']*u+pair['beta'])*u+pair['gamma'])-pair['lambda_guarded']*q
        cert = np.where(predicted, f >= 0, f > 0) & attempt
        # Independently evaluate the explicit orthogonal-residual formula.
        remainder = np.maximum(0, q-u*u/pair['anchor_norm2']) + 1e-10*q
        margin = c*u+d
        radius2 = pair['residual_norm2']*remainder
        literal = np.where(margin >= 0, margin*margin >= radius2,
                           margin*margin > radius2) & attempt
        differences = int(np.count_nonzero(cert != literal))
        errors = int(np.count_nonzero(cert & (predicted != truth[j])))
        assert differences == 0 and errors == 0
        attempts, saved = int(attempt.sum()), int(cert.sum())
        attempt_count_per_position += attempt
        saves_per_position += cert
        all_errors += errors
        pair_rows.append({**pair, 'attempted_positions': attempts, 'partner_rows_saved': saved,
                          'certified_fraction_given_attempt': saved/attempts if attempts else 0,
                          'extra_folded_products': 3*attempts,
                          'extra_literal_products': 6*attempts,
                          'saved_original_A_products': t*saved,
                          'net_folded_product_delta_vs_static_rank0': 3*attempts-t*saved,
                          'folded_vs_literal_certificate_mask_differences': differences,
                          'certified_bit_errors': errors})
    attempts = int(attempt_count_per_position.sum())
    saved = int(saves_per_position.sum())
    original_rows = int(failed0.sum())
    baseline_products = t*n + t*original_rows
    candidate_products = baseline_products + 3*attempts - t*saved
    common_slots = fallback0_slots = candidate_lower_slots = 0
    packet_rows = []
    for start in range(0, n, 32):
        f0 = int(failed0[:, start:start+32].sum())
        at = int(attempt_count_per_position[start:start+32].sum())
        sv = int(saves_per_position[start:start+32].sum())
        common = t*math.ceil(32/16) + math.ceil(t*32/16)
        base = (t+2)*math.ceil(f0/16)
        # Optimistic pooled bound: ignores anchor-to-partner critical paths and
        # bank stalls. Three MAC and two comparison slots per witness wave.
        lower = (t+2)*math.ceil((f0-sv)/16) + 5*math.ceil(at/16)
        common_slots += common
        fallback0_slots += base
        candidate_lower_slots += lower
        packet_rows.append({'rank0_failed_rows': f0, 'attempts': at, 'saved_partners': sv,
                            'rank0_fallback_slots': base, 'candidate_optimistic_fallback_plus_witness_slots': lower})
    return {
        'module': MODULES[key], 'provenance': provenance, 'T': t, 'trajectories': n,
        'pair_compile_rule': 'abs(cos)>=0.8; strongest first; disjoint; <=5; larger-norm anchor; A only',
        'pairs': pair_rows,
        'full_A_vs_captured_output_bit_mismatches': int(np.count_nonzero(truth != captured_bits)),
        'witness_output_bit_errors_vs_original_A': all_errors,
        'rank0_original_A_rows': original_rows, 'witness_attempts': attempts,
        'saved_partner_A_rows': saved, 'rank0_paid_products': baseline_products,
        'witness_paid_folded_products': candidate_products,
        'witness_product_ratio_vs_static_rank0': candidate_products/baseline_products,
        'runtime_witness_products_per_attempt': 3,
        'runtime_witness_adds_per_attempt': 3,
        'runtime_witness_comparisons_per_attempt': 2,
        'original_literal_products_per_attempt': 6,
        'anchor_value_extra_RF_write_words': attempts,
        'anchor_value_extra_RF_read_words': attempts,
        'norm_extra_RF_read_words': attempts,
        'extra_external_spill_words': 0,
        'extra_anchor_workspace_words_for_16_lanes': 16 if pairs else 0,
        'rank0_RF_words_for_32_with_masks_descriptors': 32*t+32+math.ceil(64*t/32)+math.ceil(32*(5+(t-1).bit_length())/32),
        'optimistic_service': {
            'label': 'MOST_FAVORABLE_POOLED_BLOCK_ISSUE_BOUND_NOT_RTL_OR_A_CAUSAL_SCHEDULE',
            'rank0_common_plus_fallback_slots': common_slots+fallback0_slots,
            'candidate_common_plus_witness_fallback_lower_slots': common_slots+candidate_lower_slots,
            'ratio': (common_slots+candidate_lower_slots)/(common_slots+fallback0_slots),
            'assumptions': ['Same 16 MAC lanes and 32-column block; ordinary packed masked row service.',
                            'Common rank0 norm production and static comparisons paid.',
                            'Anchor rows were already required by rank0, not extra low-rank projections.',
                            'Saved u RF write overlaps anchor completion; one saved-u and one norm RF read per witness.',
                            'Anchor-partner dependence, bank conflict and queue stalls can only worsen this optimistic estimate.',
                            'No cross-block combination is used to fill witness waves.'],
        },
        'packet_counts': packet_rows,
        'numeric_scope': 'Float64 model of real-arithmetic identity on real FP32 hooks/offline FC1 reconstruction; no frozen FP32 proof.',
    }


def main():
    state = read_checkpoint(CKPT)['model_state_dict']
    datasets = [('fc1_post', reconstructed_fc1(state))]
    root = HERE.parent/'algorithm/probe_r1/capture'
    for key in ['head', 'sn_k']:
        for path in sorted(root.glob('*/psn_'+key+'.npz')):
            datasets.append((key, hook_data(key, state, path)))
    rows = [run(key, state, data) for key, data in datasets]
    result = {'status': 'COMPLETED_ONE_PARAMETER_SELECTED_WITNESS_PROBE', 'layers': rows,
              'no_pair_fitted_to_test_inputs': True, 'no_training_or_RTL': True,
              'interpretation': 'Reuse of a required completed continuous row is distinct from an extra low-rank predictor, but projection inequalities and task compaction remain established methods.'}
    path = HERE/'witness_result.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    for row in rows:
        print(row['provenance'].get('sample','offline_FC1'), row['module'].split('.')[-3:],
              'pairs',[(p['anchor'],p['partner'],round(p['cosine'],4)) for p in row['pairs']],
              'attempts',row['witness_attempts'],'saved',row['saved_partner_A_rows'],
              'products/r0',round(row['witness_product_ratio_vs_static_rank0'],6),
              'optimistic_slots/r0',round(row['optimistic_service']['ratio'],6))
    print('wrote',path)


if __name__ == '__main__':
    main()
