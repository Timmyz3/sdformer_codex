"""Private-tail raw-Y execution with exact count-bin retirement.

The three controls share the same student, prefix, original K order, T/P W
broadcast, product-empty control and ordinary threshold-range folding. Work
counts assume zero feedback delay; no memory service or physical cycles are
modelled. The source-count probe and its result are left unchanged.
"""
import json
from pathlib import Path

import numpy as np

from source_count_bound_probe import N_UP, PREFIX, TAIL, capture_geometry, compile_bounds

HERE = Path(__file__).resolve().parent
MODES = ('complete_tail', 'initial_count_only', 'count_bin_retirement')


def prepare(sample, params, geometry):
    y, products, _, counts, _, metadata = geometry
    a = params['temporal_int16'].astype(np.int64)
    tau = params['threshold_positive'][params['full_entry']]
    core = np.einsum('ts,gshp->gthp', a[:, PREFIX], y[:, PREFIX], dtype=np.int64)
    diagonal = a[TAIL, TAIL]
    for j, t in enumerate(TAIL):
        if diagonal[j] == 0 or np.any(np.delete(a[t], list(PREFIX) + [int(t)])):
            raise ValueError('This execution requires one nonzero private tail per output.')
    if np.any(a[np.ix_(PREFIX, TAIL)]):
        raise ValueError('Core-only output rows must not depend on a private tail.')
    delta = tau[TAIL][None, :, :, None] - core[:, TAIL]
    d = diagonal[None, :, None, None]
    # Independent exact integer division. No reciprocal implementation is
    # borrowed into the numerical reference.
    raw_threshold = np.where(d > 0, -np.floor_divide(-delta, d), np.floor_divide(delta, d))
    bound = int(params['Y_abs_bound'].max())
    threshold = np.clip(raw_threshold, -bound - 1, bound + 1)
    full_u = np.einsum('ts,gshp->gthp', a, y, dtype=np.int64)
    full_gate = full_u >= tau[None, :, :, None]
    captured_gate = sample['gate_common3_diagonal_34_exact'].reshape(64, 12, 10, 8, 4)
    captured_gate = captured_gate.transpose(0, 2, 1, 3, 4).reshape(64, 10, 96, 4)
    converted = np.where(d > 0, y[:, TAIL] >= threshold, y[:, TAIL] <= threshold)
    if np.any(converted != full_gate[:, TAIL]) or np.any(full_gate != captured_gate):
        raise RuntimeError('Raw threshold or complete student differs from captured gates.')
    eligible = products[:, TAIL] > 0
    # This ordinary range test can precede reciprocal products. It is given
    # to every control, and does not use n or the twelve-bin table.
    static_positive = delta <= -np.abs(d) * bound
    static_negative = delta > np.abs(d) * bound
    static_done = static_positive | static_negative
    initial_live = eligible & ~static_done
    initial_answer = np.where(eligible, static_positive,
                              core[:, TAIL] >= tau[TAIL][None, :, :, None])
    head = dict(
        threshold_queries_before_product_empty_filter=int(threshold.size),
        product_empty_private_gates=int((~eligible).sum()),
        nonempty_private_threshold_queries=int(eligible.sum()),
        ordinary_range_folded_private_gates=int((eligible & static_done).sum()),
        dynamic_threshold_materializations=int(initial_live.sum()),
        threshold_range_comparisons=int(eligible.sum()) * 2,
        threshold_generation_signed24x16_products=int(initial_live.sum()) * 3,
        literal_complete_tail_active_products=int(products[:, TAIL].sum()),
        range_folded_complete_tail_active_products=int((products[:, TAIL] * initial_live).sum()),
        shared_prefix_active_products=int(products[:, PREFIX].sum()),
        core_A_products_before_Y_zero_skip=int(np.count_nonzero(a[:, PREFIX])) * 64 * 96 * 4,
        core_A_products_after_Y_zero_skip=sum(
            int(np.count_nonzero(a[:, s])) * int(np.count_nonzero(y[:, s])) for s in PREFIX),
        max_abs_dynamic_numerator=int(np.abs(delta).max()),
        max_abs_stored_threshold=int(np.abs(threshold).max()),
        Y_abs_bound=bound,
        full_integer_capture_gate_mismatches=int((full_gate != captured_gate).sum()),
        raw_threshold_gate_mismatches=int((converted != full_gate[:, TAIL]).sum()),
    )
    words = sample['source_gate_words'].astype(np.int64)
    sources = (words[:, None] & (1 << TAIL)[None, :, None, None]) != 0
    return dict(y_tail=y[:, TAIL], sources=sources, counts=counts[:, TAIL],
                initial_live=initial_live, initial_answer=initial_answer,
                full_gate=full_gate, threshold=threshold, positive_diagonal=diagonal > 0,
                core_answer=core[:, PREFIX] >= tau[np.array(PREFIX)][None, :, :, None],
                head=head, metadata=metadata, bound=bound)


def check_and_retire(partial, remaining, live, answer, selected, data, lo_table, hi_table, metrics, initial=False):
    groups = np.flatnonzero(selected.any(axis=(1, 2, 3)))
    if not len(groups):
        return
    selected = selected[groups]
    rem = remaining[groups]
    zero = rem[:, :, None] == 0
    partial_g = partial[groups]
    bins = np.searchsorted(N_UP, rem)
    low = partial_g + lo_table[bins].transpose(0, 1, 3, 2)
    high = partial_g + hi_table[bins].transpose(0, 1, 3, 2)
    low = np.maximum(low, -data['bound'])
    high = np.minimum(high, data['bound'])
    positive_d = data['positive_diagonal'][None, :, None, None]
    eta = data['threshold'][groups]
    positive = np.where(positive_d, low >= eta, high <= eta)
    negative = np.where(positive_d, high < eta, low > eta)
    decided = selected & (positive | negative)
    expected = data['full_gate'][groups][:, TAIL]
    metrics['certified_gate_mismatches'] += int((decided & (positive != expected)).sum())
    metrics['completed_raw_Y_mismatches'] += int(
        (selected & zero & (partial_g != data['y_tail'][groups])).sum())
    metrics['bound_contains_full_Y_violations'] += int(
        (selected & ((data['y_tail'][groups] < low) | (data['y_tail'][groups] > high))).sum())
    nonzero_check = selected & ~zero
    metrics['bound_lane_checks'] += int(nonzero_check.sum())
    metrics['final_raw_Y_comparisons'] += int((selected & zero).sum())
    metrics['bound_comparisons'] += int(nonzero_check.sum()) * 2
    metrics['bound_endpoint_additions'] += int(nonzero_check.sum()) * 2
    metrics['bound_endpoint_additions_after_zero_partial_bypass'] += int(
        (nonzero_check & (partial_g != 0)).sum()) * 2
    metrics['check_P4_H8_t_waves'] += int(selected.reshape(-1, 7, 12, 8, 4).any(axis=(3, 4)).sum())
    # Table identity is (h,n_up), independent of private t and p. Broadcast
    # within the current P4 group, never for free across independently timed G.
    for bin_id in range(1, 12):
        need = nonzero_check & (bins == bin_id)[:, :, None]
        metrics['bound_table_pairs_shared_P_T'] += int(need.any(axis=(1, 3)).sum())
        metrics['bound_table_H8_bin_uses'] += int(need.reshape(-1, 7, 12, 8, 4).any(axis=(1, 3, 4)).sum())
    metrics['initial_count_retired_gates' if initial else 'later_count_retired_gates'] += int(
        (decided & ~zero).sum())
    metrics['final_zero_count_retired_gates'] += int((decided & zero).sum())
    answer[groups] = np.where(decided, positive, answer[groups])
    live[groups] &= ~decided


def replay(data, weight, low_table, high_table, mode):
    live = data['initial_live'].copy()
    answer = data['initial_answer'].copy()
    partial = np.zeros(live.shape, dtype=np.int64)
    remaining = data['counts'].copy()
    metrics = dict(
        tail_active_source_weight_live_products=0,
        tail_W_H8_vector_uses=0,
        tail_W_scalar_uses=0,
        Conv_k_t_P4_H8_visits=0,
        source_directory_P4_descriptors=0,
        source_directory_empty_private_descriptors=0,
        source_directory_no_live_source_descriptors=0,
        source_counter_decrements_shared_H=0,
        source_counter_bin_drop_events=0,
        bound_lane_checks=0, final_raw_Y_comparisons=0,
        bound_comparisons=0, bound_endpoint_additions=0,
        bound_endpoint_additions_after_zero_partial_bypass=0,
        check_P4_H8_t_waves=0,
        bound_table_pairs_shared_P_T=0, bound_table_H8_bin_uses=0,
        initial_count_retired_gates=0, later_count_retired_gates=0,
        final_zero_count_retired_gates=0,
        certified_gate_mismatches=0, completed_raw_Y_mismatches=0,
        bound_contains_full_Y_violations=0, final_gate_mismatches=0,
        peak_abs_partial_Y=0,
    )
    if mode != 'complete_tail':
        check_and_retire(partial, remaining, live, answer, live.copy(), data,
                         low_table, high_table, metrics, initial=True)
    finish = np.full(64, -1, dtype=np.int64)
    finish[~live.any(axis=(1, 2, 3))] = 0
    for k in range(864):
        group_live = live.any(axis=(1, 2, 3))
        if not group_live.any():
            break
        sources = data['sources'][:, :, k]
        counter_live = live.any(2)
        active_source = sources & counter_live
        metrics['source_directory_P4_descriptors'] += int(group_live.sum())
        metrics['source_directory_empty_private_descriptors'] += int(
            (group_live & ~sources.any(axis=(1, 2))).sum())
        metrics['source_directory_no_live_source_descriptors'] += int(
            (group_live & ~active_source.any(axis=(1, 2))).sum())
        update = live & sources[:, :, None]
        update &= (weight[:, k] != 0)[None, None, :, None]
        partial += update * weight[:, k][None, None, :, None]
        metrics['tail_active_source_weight_live_products'] += int(update.sum())
        update_lanes = update.reshape(64, 7, 12, 8, 4)
        metrics['tail_W_H8_vector_uses'] += int(update_lanes.any(axis=(1, 3, 4)).sum())
        metrics['tail_W_scalar_uses'] += int(update.any(axis=(1, 3)).sum())
        metrics['Conv_k_t_P4_H8_visits'] += int(update_lanes.any(axis=(3, 4)).sum())
        metrics['source_counter_decrements_shared_H'] += int(active_source.sum())
        remaining -= active_source
        # A decrement crosses a ceil-power-of-two bin exactly when the new
        # count is zero or a power of two; no future W identity is inspected.
        dropped = active_source & ((remaining & (remaining - 1)) == 0)
        metrics['source_counter_bin_drop_events'] += int(dropped.sum())
        event = dropped if mode == 'count_bin_retirement' else active_source & (remaining == 0)
        selected = live & event[:, :, None]
        check_and_retire(partial, remaining, live, answer, selected, data,
                         low_table, high_table, metrics)
        just_finished = group_live & ~live.any(axis=(1, 2, 3))
        finish[just_finished] = k + 1
        metrics['peak_abs_partial_Y'] = max(metrics['peak_abs_partial_Y'], int(np.abs(partial).max()))
    if live.any():
        raise RuntimeError(f'{mode}: original source order did not finish all private tails')
    full_answer = np.zeros_like(data['full_gate'])
    full_answer[:, PREFIX] = data['core_answer']
    full_answer[:, TAIL] = answer
    metrics['final_gate_mismatches'] = int((full_answer != data['full_gate']).sum())
    metrics['checked_complete_T10_gates'] = int(full_answer.size)
    metrics['source_cursor_sum_at_completion'] = int(finish.sum())
    metrics['source_cursor_max_at_completion'] = int(finish.max())
    metrics['source_directory_nonempty_private_descriptors'] = (
        metrics['source_directory_P4_descriptors'] - metrics['source_directory_empty_private_descriptors'])
    if mode == 'complete_tail' and metrics['tail_active_source_weight_live_products'] != data['head']['range_folded_complete_tail_active_products']:
        raise RuntimeError('Complete-tail issued products do not match the independent full product count.')
    failures = sum(v for k, v in metrics.items() if k.endswith(('mismatches', 'violations')))
    if failures:
        raise RuntimeError(f'{mode}: {failures} numerical failures')
    return metrics


def rates(axis, baseline, prefix_products):
    return dict(
        tail_product_reduction=1 - axis['tail_active_source_weight_live_products'] / baseline['tail_active_source_weight_live_products'],
        prefix_plus_tail_product_reduction=(baseline['tail_active_source_weight_live_products'] - axis['tail_active_source_weight_live_products']) / (prefix_products + baseline['tail_active_source_weight_live_products']),
        tail_W_H8_use_reduction=1 - axis['tail_W_H8_vector_uses'] / baseline['tail_W_H8_vector_uses'],
        tail_Conv_k_t_visit_reduction=1 - axis['Conv_k_t_P4_H8_visits'] / baseline['Conv_k_t_P4_H8_visits'],
        tail_directory_descriptor_reduction=1 - axis['source_directory_P4_descriptors'] / baseline['source_directory_P4_descriptors'],
        nonempty_private_directory_descriptor_reduction=1 - axis['source_directory_nonempty_private_descriptors'] / baseline['source_directory_nonempty_private_descriptors'],
    )


def main():
    params = dict(np.load(HERE / 'integer_deployment/common3_diagonal_34.npz'))
    weight = params['weight_int8'].reshape(96, 864).astype(np.int64)
    low_table, high_table = compile_bounds(weight)
    result = dict(
        scope='same common3 integer student, four real captures, 64 P4/frame and all H96; prefix [2,3,7] already produced',
        source_order='original k=((c*3)+kh)*3+kw, 0..863; all controls share W[k,H8] across every live private time and P4. All 12 H8 groups of a P4 source group advance the same K cursor.',
        checks='initial plus remaining-count ceil-power-of-two bin drops, including zero; zero feedback delay, no cycles',
        raw_predicate='d>0: Yi>=ceil((TUi-Ucore)/d); d<0: Yi<=floor((TUi-Ucore)/d). Saturate to ±(global Y_abs_bound+1).',
        ordinary_controls='product-empty gates and dynamic-threshold static range folding are given to all three controls',
        constants_and_state=dict(
            Y_bits=24, Ui_bits=48, threshold_bits=16,
            prefix_payload_bytes_3Y_1U_plus_thresholds=928,
            tail_payload_bytes_7Y_plus_thresholds=1120,
            source_counter_bytes_per_P4_group=50,
            output_T10_bits_bytes_per_P4_H8=40,
            private_live_bitmap_bytes_per_P4_H8=28,
            optional_private_source_nonempty_K_bitmap_bytes_per_P4=108,
            bound_table_12x96x2x24_bytes=6912,
            full_tau32_bytes=3840,
            threshold_generation='three signed24x16 products per materialized query, fixed shifts/adds and one remainder correction; see compile_tail_division.py. This script independently uses exact division.',
        ),
        exclusions=['zero-latency retirement feedback; no bank, outstanding-read, comparator or pipeline service',
                    'P/T equal-bin table broadcast is charged in pair uses, not assumed single-port completion',
                    'remaining counts are shared across H8 only under this common K cursor; independent H8 backpressure requires private counters/snapshots or a barrier',
                    'logical source descriptor is four T10 fields (40 packed bits), not a physical SRAM transaction',
                    'ordinary skipping of source-empty descriptors is also reported via nonempty payload counts; a 108-byte K bitmap per P4 is one metadata option, with its generation/read/selection still payable',
                    'source counts and ordinary product-empty metadata need production; count generation can overlap the common prefix scan only if its complete T10 descriptor is actually available',
                    'head counts exclude unspecified coefficient/table ports, shifts/adds and division temporary registers',
                    'core_A counts are ordinary zero-skipped scalar products shared by all controls, not an optimally scheduled/CSE-compiled PSN service',
                    'complete-tail may alternatively keep Ui and pay one final diagonal product per private gate instead of materializing raw thresholds; that strong arithmetic option is not timed here',
                    'no inference of whole-layer speed, equal area, SRAM macros or PPA'],
        frames=[], axes={m: dict(frames=[]) for m in MODES})
    for path in sorted((HERE / 'integer_valid10').glob('capture_*.npz')):
        with np.load(path) as sample:
            geometry = capture_geometry(sample, weight)
            data = prepare(sample, params, geometry)
            result['frames'].append(dict(capture=path.name, file=str(sample['file']),
                                         head=data['head'], source_metadata=data['metadata']))
            frame_axes = {}
            for mode in MODES:
                counts = replay(data, weight, low_table, high_table, mode)
                frame_axes[mode] = counts
                result['axes'][mode]['frames'].append(dict(capture=path.name, counts=counts))
                print(path.name, mode, counts['tail_active_source_weight_live_products'],
                      counts['tail_W_H8_vector_uses'], counts['check_P4_H8_t_waves'], flush=True)
            for mode in MODES:
                result['axes'][mode]['frames'][-1]['rates_vs_complete_tail'] = rates(
                    frame_axes[mode], frame_axes['complete_tail'], data['head']['shared_prefix_active_products'])
    head_keys = result['frames'][0]['head']
    result['head_totals'] = {k: (max(f['head'][k] for f in result['frames']) if k.startswith(('max_', 'Y_abs'))
                                 else sum(f['head'][k] for f in result['frames'])) for k in head_keys}
    for mode in MODES:
        frames = result['axes'][mode]['frames']
        result['axes'][mode]['counts'] = {
            k: (max(f['counts'][k] for f in frames) if k.startswith('peak_') or k == 'source_cursor_max_at_completion'
                else sum(f['counts'][k] for f in frames)) for k in frames[0]['counts']}
    base = result['axes']['complete_tail']['counts']
    for mode in MODES:
        result['axes'][mode]['rates_vs_complete_tail'] = rates(
            result['axes'][mode]['counts'], base, result['head_totals']['shared_prefix_active_products'])
    path = HERE / 'private_tail_replay.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(path, flush=True)


if __name__ == '__main__':
    main()
