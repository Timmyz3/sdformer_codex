"""Zero-overhead upper bound for exact all-T10 retirement during GP FC1.

Fixed first frame, 32 uniform positions and all hidden outputs; no model fitting.
This is not a cycle model. Future captured codes are used only to verify bounds
and count work that could be skipped, never to form the remaining-work bound.
"""
from collections import Counter
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
from gustavsnn_reference import identities, read_torch


def legal_codes(producer):
    variable = producer['variable_rows']
    constant = producer['constant_gates']
    addresses = [a for a in range(8)
                 if all(variable[r] or bool((a >> r) & 1) == constant[r]
                        for r in range(3))]
    return sorted({producer['mapping'][a] for a in addresses})


def hist(a):
    return {str(k): int(v) for k, v in sorted(Counter(a.ravel().tolist()).items())}


def work_count(finish, W, codes, B):
    """Logical p/h W-use opportunities, before sharing a W across multiple p.

    The H8 control waits for all eight output-row tiles sharing the same p.
    Real P4 execution changes both release dependencies and the shared-request
    denominator. Neither number is an implemented weight SRAM request count.
    """
    P, C = codes.shape
    H = W.shape[0]
    active = np.any(B[:, codes] != 0, axis=0)
    nz = W != 0
    channels = np.arange(C)
    group_finish = finish.reshape(P, H//8, 8).max(2)
    group_nonzero = nz.reshape(H//8, 8, C).any(1)
    baseline = skipped = group_skipped = 0
    broadcast = broadcast_skipped = 0
    for p in range(P):
        uses = nz & active[p, None, :]
        baseline += int(uses.sum())
        skipped += int((uses & (channels >= finish[p, :, None])).sum())
        after_group = channels >= np.repeat(group_finish[p], 8)[:, None]
        group_skipped += int((uses & after_group).sum())
        rows = group_nonzero & active[p, None, :]
        broadcast += int(rows.sum())
        broadcast_skipped += int((rows & (channels >= group_finish[p, :, None])).sum())
    return {
        'nonzero_W_active_source_logical_uses': baseline,
        'individual_ph_skipped_logical_uses': skipped,
        'individual_ph_skipped_fraction': skipped/baseline if baseline else 0,
        'H8_slowest_skipped_logical_uses': group_skipped,
        'H8_slowest_skipped_fraction': group_skipped/baseline if baseline else 0,
        'H8_active_source_rows_with_any_nonzero_W': broadcast,
        'H8_slowest_suppressed_source_rows': broadcast_skipped,
        'H8_slowest_source_row_fraction': broadcast_skipped/broadcast if broadcast else 0,
        'H8_finish_C_histogram': hist(group_finish),
        'H8_early_groups': int(np.count_nonzero(group_finish < C)),
        'H8_total_groups': int(group_finish.size),
    }


def evaluate(name, W, B, tau, all_codes, allowed, metadata):
    W, B, tau = (np.asarray(a, dtype=np.int64) for a in (W, B, tau))
    H, C = W.shape
    positions = np.linspace(0, len(all_codes)-1, 32, dtype=int)
    codes = np.asarray(all_codes[positions], dtype=np.int64)
    P, T = len(positions), B.shape[0]
    assert C == 384 and T == 10 and H % 8 == 0
    assert tau.shape == (T, H) and B.shape == (T, 8)
    assert set(np.unique(codes)).issubset(allowed)

    # Independent complete-C reference: form class sums, then apply saved B.
    S = np.stack([(codes == c).astype(np.int64) @ W.T for c in range(8)], axis=-1)
    complete = S @ B.T - tau.T[None, :, :]
    complete_gates = complete >= 0

    # Known W suffixes and the statically legal full source alphabet only.
    wp = np.maximum(W, 0)
    wn = np.minimum(W, 0)
    pos_suffix = np.pad(np.cumsum(wp[:, ::-1], axis=1)[:, ::-1], ((0, 0), (0, 1)))
    neg_suffix = np.pad(np.cumsum(wn[:, ::-1], axis=1)[:, ::-1], ((0, 0), (0, 1)))
    min_B, max_B = B[:, allowed].min(1), B[:, allowed].max(1)
    partial = np.broadcast_to(-tau.T[None, :, :], (P, H, T)).copy()
    finish = np.full((P, H), C, dtype=np.int16)
    retired = np.zeros((P, H), dtype=bool)
    first_bit_finish = np.full((P, H, T), C, dtype=np.int16)
    bit_retired = np.zeros((P, H, T), dtype=bool)
    checkpoints = []
    # Values of this 16-C prefix are observed; no future code enters min/max.
    for start in range(0, C, 16):
        end = start+16
        seen = B[:, codes[:, start:end]].transpose(1, 0, 2)
        partial += (seen @ W[:, start:end].T).transpose(0, 2, 1)
        lower_remaining = (pos_suffix[:, end, None]*min_B +
                           neg_suffix[:, end, None]*max_B)
        upper_remaining = (pos_suffix[:, end, None]*max_B +
                           neg_suffix[:, end, None]*min_B)
        lower = partial + lower_remaining[None, :, :]
        upper = partial + upper_remaining[None, :, :]
        # Complete values verify safety but never alter the calculated bound.
        assert np.all(lower <= complete) and np.all(complete <= upper)
        one, zero = lower >= 0, upper < 0
        certain = one | zero
        assert np.all(complete_gates[one]) and not np.any(complete_gates[zero])
        new_bits = certain & ~bit_retired
        first_bit_finish[new_bits] = end
        bit_retired |= certain
        done = certain.all(2)
        new = done & ~retired
        finish[new] = end
        retired |= done
        checkpoints.append({'C_seen': end, 'new_all_T10_ph': int(new.sum()),
            'cumulative_all_T10_ph': int(retired.sum()),
            'cumulative_certified_bits': int(bit_retired.sum())})
    assert np.array_equal(partial, complete) and np.all(retired)
    metrics = work_count(finish, W, codes, B)
    result = {'case': name, 'shape': {'P': P, 'C': C, 'H': H, 'T': T},
        'positions': positions.tolist(), 'legal_saved_codes': allowed,
        'B_min_per_t': min_B.tolist(), 'B_max_per_t': max_B.tolist(),
        'B_including_zero_class': B.tolist(),
        'actual_U_min': int(complete.min()), 'actual_U_max': int(complete.max()),
        'all_T10_finish_C_histogram': hist(finish),
        'single_bit_finish_C_histogram_diagnostic_only': hist(first_bit_finish),
        'all_T10_early_ph': int(np.count_nonzero(finish < C)),
        'all_T10_total_ph': int(finish.size),
        'all_T10_early_fraction': float(np.mean(finish < C)),
        'mean_C_seen_for_all_T10': float(finish.mean()),
        'mean_unseen_C_fraction_unweighted': float(np.mean(C-finish)/C),
        'checkpoints': checkpoints, 'work_opportunity': metrics,
        'reference_check': 'complete class-S then B and 16-C direct W*B[code] accumulation agree exactly in int64; every bound contains the true full-C result; every early gate agrees',
        **metadata}
    print(name, 'early', result['all_T10_early_fraction'], 'individual W-use skip',
          metrics['individual_ph_skipped_fraction'], 'H8 skip', metrics['H8_slowest_skipped_fraction'],
          'mean C', result['mean_C_seen_for_all_T10'], flush=True)
    return result


def main():
    started = time.monotonic()
    alg = ROOT/'algorithm'
    params = read_torch(alg/'stage2_temporal_codes/integer_parameters.pt')
    books = np.load(alg/'stage2_temporal_codes/codebooks.npz')
    basis = np.load(alg/'stage2_temporal_codes/signed_basis.npz')
    consumers = json.loads((alg/'stage2_class_shift/consumers.json').read_text())
    producers = json.loads((alg/'direct_code_integer/parameters.json').read_text())
    records = []
    for b in range(6):
        tag = f's2b{b}'
        W, routes, identity = identities(b, params, books, basis, consumers)
        B = routes['packed_class'][1] @ routes['packed_class'][0]
        tau = consumers['power2_trained'][tag]['tau_int32']
        path = alg/f'direct_code_integer/deployment/capture10/v000_zurich_city_09_a_0001_{tag}.npz'
        capture = np.load(path)
        assert np.array_equal(capture['address_to_class'], producers[tag]['mapping'])
        records.append(evaluate(tag, W, B, tau, capture['codes'], legal_codes(producers[tag]),
            {'source_capture': str(path.relative_to(ROOT)), 'identity': identity,
             'output_group_layout': 'contiguous hidden rows h=group*8+tile'}))
    trained_path = alg/'pruning_probe/hidden_H_half_trained.npz'
    trained = np.load(trained_path)
    keep = trained['hidden_indices']
    hidden_path = alg/'pruning_probe/codes/hidden_H_half_trained/zurich_city_09_a_0001/s2b3.npz'
    records.append(evaluate('s2b3_hidden50_trained', trained['weight_int8'][keep],
        trained['B_int8'].astype(np.int64) @ trained['E_int8'].astype(np.int64).T,
        trained['threshold_int32'][:, keep], np.load(hidden_path)['codes'],
        legal_codes(producers['s2b3']),
        {'source_capture': str(hidden_path.relative_to(ROOT)),
         'trained_parameters': str(trained_path.relative_to(ROOT)),
         'output_group_layout': 'compacted surviving hidden rows, eight per broadcast group',
         'output_theta': 'retained by the real consumer; separate from decision tau'}))
    totals = Counter()
    for record in records[:6]:
        work = record['work_opportunity']
        for k in ('nonzero_W_active_source_logical_uses', 'individual_ph_skipped_logical_uses',
                  'H8_slowest_skipped_logical_uses', 'H8_active_source_rows_with_any_nonzero_W',
                  'H8_slowest_suppressed_source_rows'):
            totals[k] += work[k]
        for k in ('all_T10_early_ph', 'all_T10_total_ph'):
            totals[k] += record[k]
    denominator = totals['nonzero_W_active_source_logical_uses']
    aggregate = dict(totals)
    aggregate.update(individual_ph_skipped_fraction=totals['individual_ph_skipped_logical_uses']/denominator,
        H8_slowest_skipped_fraction=totals['H8_slowest_skipped_logical_uses']/denominator,
        H8_slowest_source_row_fraction=totals['H8_slowest_suppressed_source_rows']/totals['H8_active_source_rows_with_any_nonzero_W'],
        all_T10_early_fraction=totals['all_T10_early_ph']/totals['all_T10_total_ph'])
    out = {'kind': 'EXACT_ALL_T10_GP_PREFIX_ZERO_COST_RETIREMENT_UPPER_BOUND',
        'scope': 'integer direct-source student, first diverse frame, six full FC1 hidden dimensions; separate actual hidden-half trained control',
        'fixed_sample': 'zurich_city_09_a/0001; np.linspace(0,1199,32,dtype=int); all H; checkpoints C=16,32,...,384',
        'formula': 'partial=sum_seen W[h,c]*B[t,code[p,c]]-tau[t,h]; remaining bounds use W positive/negative suffix sums times min/max B over the complete statically legal saved source alphabet',
        'criterion': 'all ten gates certain: lower>=0 or upper<0 for every t; no approximation or AEE trade',
        'numerical_contract': 'theta_source already folded into saved W; output theta retained separately from tau; same saved B/tau; this is the new integer student, not frozen ep34 FP equivalence',
        'bound_inputs_exclude': ['future actual source codes', 'actual remaining class frequencies', 'validation-trained masks or thresholds'],
        'costs_set_to_zero_for_this_upper_bound': ['forming ten prefix projections or repeatedly projecting R class sums',
            'bound checks, suffix constants and metadata access', 'per-consumer completion state and scheduling',
            'handling divergent completion times, NR4 compaction and output enqueue'],
        'transaction_boundary': 'per-(p,h,c) nonzero-W uses with active source; H8 waits for the slowest of all eight output tiles at a fixed p. Real P4/NR4 sharing changes both release dependencies and the request denominator; this percentage is not a bound on the complete physical request percentage. No SRAM cycles, energy or area are claimed.',
        'stop_rule': 'If even this zero-cost upper bound saves less than about 3% of remaining source-active nonzero-W work, stop this simple full-alphabet suffix-bound form.',
        'aggregate_six_base_modules': aggregate, 'records': records,
        'elapsed_cpu_seconds': time.monotonic()-started}
    target = ROOT/'psn/gp_all_t10_bound_probe.json'
    target.write_text(json.dumps(out, indent=2)+'\n')
    print('aggregate', aggregate, flush=True)
    print(target, 'seconds', out['elapsed_cpu_seconds'], flush=True)


if __name__ == '__main__':
    main()
