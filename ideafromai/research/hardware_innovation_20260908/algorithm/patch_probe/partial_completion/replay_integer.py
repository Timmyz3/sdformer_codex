"""Integer gate replay and real source/weight/consumer opportunity counts.

Logical weight use is charged after intersecting each real source word, the
quantized weight mask, and unresolved consumers. This is not a bank service or
cycle model. Both temporal structures receive exactly the same ordinary skips.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BITS = 1 << np.arange(10)
POPCOUNT = np.array([i.bit_count() for i in range(1024)])
POP4 = np.array([i.bit_count() for i in range(16)], dtype=np.int32)
ORDERS = {
    'row34': [0, 2, 5, 7, 8, 1, 3, 6, 4, 9],
    'common3_diagonal_34': [2, 3, 7, 0, 5, 8, 1, 6, 4, 9],
}


def source_geometry(sample, weight):
    words = sample['source_gate_words'].astype(np.int64)
    # Original k=((c*3)+kh)*3+kw; P4 is four native adjacent x positions.
    gates = (words[:, None] & BITS[None, :, None, None]) != 0
    pwords = (gates * (1 << np.arange(4))[None, None, None]).sum(-1).astype(np.uint8)
    zero = ((~gates.any(axis=(2, 3))) * BITS).sum(1)
    # Reconstruct all captured Conv1 values directly from source bits and real
    # INT8 W, independent of the GPU convolution and temporal gate evaluator.
    yi = np.matmul(weight.reshape(1, 1, 96, 864).astype(np.int64),
                   gates.astype(np.int64))
    yi = yi.reshape(64, 10, 12, 8, 4).transpose(0, 2, 1, 3, 4).reshape(768, 10, 32)
    captured = sample['Yi']
    difference = np.count_nonzero(yi != captured)
    if difference:
        raise RuntimeError(f'Conv1 source/W reconstruction differs at {difference} values')
    return words, pwords, zero, yi


def replay(sample, params, name, mode, capacity=5):
    aq = params['temporal_int16'].astype(np.int64)
    weight = params['weight_int8'].reshape(96, 864)
    words, source_pwords, zero, y = source_geometry(sample, weight)
    support = aq != 0
    deps = (support * BITS).sum(1)
    seen = np.repeat(zero, 12)
    h = np.broadcast_to(np.arange(96).reshape(12, 8, 1), (12, 8, 4)).reshape(12, 32)
    h = np.tile(h, (64, 1))
    pos, neg = params['threshold_positive'], params['threshold_negative']
    addresses = params['state_to_entry']
    full_u = np.einsum('ts,nsq->ntq', aq, y, dtype=np.int64)
    full_pos = pos[params['full_entry'][None, :, None], h[:, None]]
    full_gate = full_u >= full_pos
    unresolved = np.ones_like(full_gate)
    answer = np.zeros_like(full_gate)
    produced = np.zeros(768, dtype=np.int64)
    unique_weight_vectors = np.zeros((768, 864), dtype=bool)
    issued_batches = np.zeros(768, dtype=np.int64)
    sums = dict(
        conv1_products_after_source_weight_lane_skip=0,
        conv1_source_lane_terms_before_weight_zero_skip=0,
        logical_weight_vector_uses_after_product_filter=0,
        logical_weight_scalar_uses_after_product_filter=0,
        source_only_logical_weight_vector_uses=0,
        SIMD32_Conv_active_time_visits=0,
        PSN_nonzero_Y_coefficient_evaluations=0,
        PSN_coefficient_evaluations_before_Y_zero_skip=0,
        SIMD32_PSN_nonzero_term_visits=0,
        confidence_comparisons=0, exact_gate_comparisons=0,
        factor_compensation_products_shared_across_P4=0,
        row_constant_lookups_shared_across_H8_P4=0,
        base_t_h_scalar_uses_shared_across_P4=0,
        factor_threshold_generations_shared_across_P4=0,
        peak_allocated_Y_columns=0, actual_predicate_checks=0,
    )
    for stage in range(11):
        live = unresolved.any(axis=(1, 2))
        if not live.any():
            break
        for state in np.unique(seen[live]):
            ix = np.flatnonzero(live & (seen == state))
            observed = (int(state) & BITS) != 0
            partial = np.einsum('ts,nsq->ntq', aq[:, observed], y[ix][:, observed], dtype=np.int64)
            entry = addresses[state]
            threshold_pos = pos[entry[None, :, None], h[ix, None]]
            threshold_neg = neg[entry[None, :, None], h[ix, None]]
            positive = partial >= threshold_pos
            negative = partial <= threshold_neg
            # Check the factored hardware predicate on actual network values,
            # including equality and raw-zero BN offsets. Wide arithmetic is
            # necessary: the base/radius are too wide for signed INT32.
            offset = params['bn_offset_int'][h[ix, None]]
            factored = ((partial << params['accumulator_left_shift'][h[ix, None]])
                        - params['base_threshold'][np.arange(10)[None, :, None], h[ix, None]]
                        + (params['entry_mean_int'][entry][None, :, None] << 14)
                        - offset * params['entry_remaining_qsum'][entry][None, :, None])
            radius = params['entry_radius_int'][entry][None, :, None] << 14
            if np.any(positive != (factored >= radius)) or np.any(negative != (factored < -radius)):
                raise RuntimeError('factored integer predicate differs from compiled thresholds')
            sums['actual_predicate_checks'] += int(partial.size)
            exact = (deps & ~int(state)) == 0
            if mode == 'exact':
                evaluated = unresolved[ix] & exact[None, :, None]
                confident = np.broadcast_to(exact[None, :, None], partial.shape)
            else:
                evaluated = unresolved[ix]
                confident = positive | negative
                sums['confidence_comparisons'] += int((evaluated & ~exact[None, :, None]).sum()) * 2
            sums['exact_gate_comparisons'] += int((evaluated & exact[None, :, None]).sum())
            sums['PSN_coefficient_evaluations_before_Y_zero_skip'] += int(
                (evaluated * support[:, observed].sum(1)[None, :, None]).sum())
            nonzero_terms = np.einsum('ts,nsq->ntq', support[:, observed].astype(np.int32),
                                     (y[ix][:, observed] != 0).astype(np.int32))
            sums['PSN_nonzero_Y_coefficient_evaluations'] += int((evaluated * nonzero_terms).sum())
            for s in np.flatnonzero(observed):
                term_lanes = evaluated & support[None, :, s, None] & (y[ix, s, None] != 0)
                sums['SIMD32_PSN_nonzero_term_visits'] += int(term_lanes.any(-1).sum())
            active_h = evaluated.reshape(len(ix), 10, 8, 4).any(-1)
            sums['base_t_h_scalar_uses_shared_across_P4'] += int(active_h.sum())
            # One compensation multiply per h serves all four p. No temporal
            # reuse of this computed constant across contexts is assumed.
            rem = params['entry_remaining_qsum'][entry] != 0
            if mode != 'exact':
                sums['factor_threshold_generations_shared_across_P4'] += int(
                    (active_h & ~exact[None, :, None]).sum())
                sums['factor_compensation_products_shared_across_P4'] += int(
                    (active_h & rem[None, :, None]
                     & (params['bn_offset_int'][h[ix, ::4]][:, None] != 0)).sum())
                sums['row_constant_lookups_shared_across_H8_P4'] += int(
                    (evaluated.any(-1) & ~exact[None]).sum())
            decided = unresolved[ix] & confident
            answer[ix] = np.where(decided, positive, answer[ix])
            unresolved[ix] &= ~decided
        for i in np.flatnonzero(unresolved.any(axis=(1, 2))):
            dependency = int(np.bitwise_or.reduce(deps[unresolved[i].any(1)]))
            retained = dependency & int(seen[i]) & ~int(zero[i // 12])
            needed = dependency & ~int(seen[i])
            free = capacity - int(POPCOUNT[retained])
            if free <= 0:
                raise RuntimeError('selected order exceeds the Y capacity')
            columns = [s for s in ORDERS[name] if needed & (1 << s)][:free]
            batch = sum(1 << s for s in columns)
            if not batch:
                raise RuntimeError('unresolved output has no new temporal dependency')
            wanted = (np.einsum('ts,tq->sq', support.astype(np.int32),
                                unresolved[i].astype(np.int32)) > 0).reshape(10, 8, 4)
            wanted_pwords = (wanted * (1 << np.arange(4))[None, None]).sum(-1).astype(np.uint8)
            wlive = weight[(i % 12) * 8:(i % 12 + 1) * 8].T != 0
            kh = np.zeros((864, 8), dtype=bool)
            for s in columns:
                intersection = source_pwords[i // 12, s, :, None] & wanted_pwords[s, None]
                live_products = POP4[intersection]
                sums['conv1_source_lane_terms_before_weight_zero_skip'] += int(live_products.sum())
                sums['conv1_products_after_source_weight_lane_skip'] += int((live_products * wlive).sum())
                active_kh = (intersection != 0) & wlive
                sums['SIMD32_Conv_active_time_visits'] += int(active_kh.any(1).sum())
                kh |= active_kh
            k = kh.any(1)
            sums['logical_weight_vector_uses_after_product_filter'] += int(k.sum())
            sums['logical_weight_scalar_uses_after_product_filter'] += int(kh.sum())
            sums['source_only_logical_weight_vector_uses'] += int(
                ((np.bitwise_or.reduce(words[i // 12], axis=1) & batch) != 0).sum())
            unique_weight_vectors[i] |= k
            sums['peak_allocated_Y_columns'] = max(sums['peak_allocated_Y_columns'],
                int(POPCOUNT[retained]) + len(columns))
            produced[i] |= batch
            seen[i] |= batch
            issued_batches[i] += 1
    if unresolved.any():
        raise RuntimeError('integer completion did not finish')
    if mode == 'exact' and np.any(answer != full_gate):
        raise RuntimeError('dependency completion changed the complete integer function')
    key = f'gate_{name}_{mode}'
    mismatch = None
    if key in sample:
        mismatch = int(np.count_nonzero(answer != sample[key].reshape(answer.shape)))
        if mismatch:
            raise RuntimeError(f'{key}: CPU/GPU differ at {mismatch} gates')
    wrong = answer != full_gate
    sums.update(contexts=768, gates=int(answer.size), predicted_spikes=int(answer.sum()),
                complete_integer_spikes=int(full_gate.sum()), wrong_gates_vs_own_complete=int(wrong.sum()),
                false_negatives_vs_own_complete=int((full_gate & ~answer).sum()),
                issued_batches=int(issued_batches.sum()),
                produced_time_columns=int(POPCOUNT[produced].sum()),
                unique_weight_vectors_if_no_reload=int(unique_weight_vectors.sum()),
                CPU_GPU_gate_mismatches=mismatch,
                exact_source_weight_reconstruction_values=int(y.size))
    return sums


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-directory', type=Path, default=HERE / 'integer_valid10')
    args = parser.parse_args()
    captures = sorted(args.capture_directory.glob('capture_*.npz'))
    if not captures:
        raise RuntimeError('No integer source/Y capture is available yet')
    results = dict(
        scope='fixed first four validation frames; 64 native P4/frame, all 12 H8 groups; logical operation counts, not cycles',
        format='common INT8 W, Q14 INT16 A, INT24 Yi and INT48 Ui; train32 residual gamma3',
        skip_policy='all controls receive real source-zero, W-zero, per-p/h consumer enables, zero-Y PSN skip and P4-shared compensation',
        state_policy='five retained Yi columns + one working Ui; exact10 is a stronger larger-Y reference, not same-resource admitted',
        product_filter='a logical W[k,H8] use exists only if some requested t,p,h has source=1, W[k,h]!=0 and an unresolved consumer of Yi[t,p,h]',
        SIMD_visits='one k/time visit serves up to 32 P4/H8 lanes; one PSN t/s visit likewise. These are vector work visits, not scheduled cycles, and Conv-add/PSN-multiply resources are different.',
        factor_parameter_uses='one row lookup is a mean/radius/sumR tuple; base t/h uses and nonexact factor generations are counted separately. Parameter caching, shifts/adds and physical table traffic are not closed.',
        weight_nonzero_metadata='pre-request filtering assumes an available static W mask/index; an uncompressed 96x864 bitmap is 10368 bytes shared by all P groups, additional to the 82944-byte INT8 W array',
        exclusions=['finite NRV/metadata generation, bank/port/arbitration/cache costs',
                    'cycles, area, energy, complete-layer representativeness',
                    'equal total storage between five-Y predictor and ten-Y exact',
                    'full-network AEE: use the separately evaluated integer summaries'],
        axes={})
    for name in ORDERS:
        with np.load(HERE / 'integer_deployment' / f'{name}.npz') as archive:
            params = dict(archive)
        for mode, capacity in [('exact', 5), ('individual', 5), ('exact', 10)]:
            rows = []
            for path in captures:
                with np.load(path) as archive:
                    sample = dict(archive)
                row = replay(sample, params, name, mode, capacity)
                rows.append(dict(file=str(sample['file']), **row))
                print(name, mode, capacity, path.name,
                      row['logical_weight_vector_uses_after_product_filter'], flush=True)
            numeric = {key: (max(r[key] for r in rows) if key == 'peak_allocated_Y_columns'
                             else sum(r[key] for r in rows))
                       for key in rows[0] if key != 'file' and rows[0][key] is not None}
            numeric['frames'] = len(rows)
            results['axes'][f'{name}_{mode}_{capacity}Y'] = dict(frames=rows, totals=numeric)
    output = HERE / 'integer_deployment' / 'opportunity_result.json'
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: v['totals'] for k, v in results['axes'].items()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
