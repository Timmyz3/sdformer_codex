"""Exact, dependency-filtered interval retirement during one shared K scan.

Only current partial Yi, source counts and static weight bounds drive decisions.
Captured complete Yi/gates are assertion-only labels. Counts are logical work,
not an implementation of ports, pipelines or cycle timing.
"""
import json
from pathlib import Path

import numpy as np

from source_count_bound_probe import N_UP, compile_bounds

HERE = Path(__file__).resolve().parent
BITS = 1 << np.arange(10)


def dependencies(live, support):
    return np.stack([live[:, support[:, s]].any(1) for s in range(10)], axis=1)


def add_work(metrics, prefix, update):
    lanes = update.reshape(64, 10, 12, 8, 4)
    metrics[prefix + '_products'] += int(update.sum())
    metrics[prefix + '_W_H8_uses'] += int(lanes.any(axis=(1, 3, 4)).sum())
    metrics[prefix + '_W_scalar_uses'] += int(update.any(axis=(1, 3)).sum())
    metrics[prefix + '_Conv_k_t_H8_visits'] += int(lanes.any(axis=(3, 4)).sum())


def check(partial, remaining, live, answer, selected, data, m, initial=False):
    groups = np.flatnonzero(selected.any(axis=(1, 2, 3)))
    if not len(groups):
        return
    selected = selected[groups]
    rem = remaining[groups]
    bins = np.searchsorted(N_UP, rem)
    empty = data['empty'][groups]
    # A bin upper-bounds arbitrary remaining source identities. Previously
    # visited weights are not removed from the table: this stays conservative.
    ylo = partial[groups] + data['lo'][bins].transpose(0, 1, 3, 2)
    yhi = partial[groups] + data['hi'][bins].transpose(0, 1, 3, 2)
    ylo = np.maximum(ylo, data['lo'][-1][None, None, :, None])
    yhi = np.minimum(yhi, data['hi'][-1][None, None, :, None])
    ylo = np.where(empty, 0, ylo)
    yhi = np.where(empty, 0, yhi)
    needed = dependencies(selected, data['support'])
    bounded_y = needed & ~empty & (rem[:, :, None] > 0)
    m['Y_bound_pair_uses_before_reuse'] += int(bounded_y.sum())
    m['Y_bound_endpoint_additions_before_bypass'] += 2 * int(bounded_y.sum())
    for table in (data['lo'], data['hi']):
        endpoint = table[bins].transpose(0, 1, 3, 2)
        m['Y_bound_endpoint_additions_after_zero_bypass'] += int(
            (bounded_y & (partial[groups] != 0) & (endpoint != 0)).sum())
    # Clipping to the full-weight range is an ordinary tighter bound, with
    # its comparisons and per-H constants explicitly charged.
    m['Y_static_range_clip_comparisons_before_sign_bypass'] += 2 * int(bounded_y.sum())
    # L[n]>=L[all], U[n]<=U[all]: positive partial cannot need the
    # lower clamp, negative partial cannot need the upper, zero needs neither.
    clamp_use = bounded_y & (partial[groups] != 0)
    m['Y_static_range_clip_comparisons'] += int(clamp_use.sum())
    m['Y_static_range_pair_uses_shared_T_P'] += int(clamp_use.any(axis=(1, 3)).sum())
    for b in range(1, len(N_UP)):
        use = bounded_y & (bins == b)[:, :, None]
        m['Y_bound_table_pairs_shared_T_P'] += int(use.any(axis=(1, 3)).sum())
        m['Y_bound_table_H8_bin_uses'] += int(
            use.reshape(-1, 10, 12, 8, 4).any(axis=(1, 3, 4)).sum())
    m['check_P4_events'] += len(groups)
    m['check_P4_H8_t_waves'] += int(
        selected.reshape(-1, 10, 12, 8, 4).any(axis=(3, 4)).sum())
    m['initial_selected_gates' if initial else 'later_selected_gates'] += int(selected.sum())
    decided_all = np.zeros_like(selected)
    positive_all = np.zeros_like(selected)
    for t in range(10):
        sel = selected[:, t]
        if not sel.any():
            continue
        sl = np.flatnonzero(data['support'][t])
        coeff = data['a'][t, sl][None, :, None, None]
        low_input = np.where(coeff > 0, ylo[:, sl], yhi[:, sl])
        high_input = np.where(coeff > 0, yhi[:, sl], ylo[:, sl])
        u_low = (coeff * low_input).sum(1)
        u_high = (coeff * high_input).sum(1)
        exact = (low_input == high_input).all(1)
        low_nz = low_input != 0
        high_nz = high_input != 0
        unique_high = high_nz & (high_input != low_input)
        m['A_endpoint_products_before_zero_equal_skip'] += 2 * int(sel.sum()) * len(sl)
        m['A_endpoint_products_after_zero_equal_skip'] += int(
            ((low_nz + unique_high.astype(np.int8)) * sel[:, None]).sum())
        m['A_exact_row_products_after_zero_skip'] += int(
            (low_nz * (sel & exact)[:, None]).sum())
        m['A_interval_row_products_after_zero_equal_skip'] += int(
            ((low_nz + unique_high.astype(np.int8)) * (sel & ~exact)[:, None]).sum())
        low_adds = np.maximum(low_nz.sum(1) - 1, 0)
        high_adds = np.maximum(high_nz.sum(1) - 1, 0)
        m['A_endpoint_accumulation_additions'] += int(
            ((low_adds + high_adds * ~exact) * sel).sum())
        m['exact_gate_comparisons'] += int((sel & exact).sum())
        m['interval_gate_comparisons'] += 2 * int((sel & ~exact).sum())
        positive = u_low >= data['tau'][t][None, :, None]
        negative = u_high < data['tau'][t][None, :, None]
        decided = sel & (positive | negative)
        decided_all[:, t] = decided
        positive_all[:, t] = positive
        m['certified_gate_mismatches'] += int(
            (decided & (positive != data['gold_gate'][groups, t])).sum())
        m['propagated_U_bound_violations'] += int(
            (sel & ((data['gold_u'][groups, t] < u_low)
                    | (data['gold_u'][groups, t] > u_high))).sum())
        m['initial_retired_gates' if initial else 'later_retired_gates'] += int(decided.sum())
        m['retired_before_all_Y_exact'] += int((decided & ~exact).sum())
    m['Y_bound_violations'] += int(
        (needed & ((data['gold_y'][groups] < ylo) | (data['gold_y'][groups] > yhi))).sum())
    m['complete_Y_mismatches'] += int(
        (needed & (rem[:, :, None] == 0) & (partial[groups] != data['gold_y'][groups])).sum())
    answer[groups] = np.where(decided_all, positive_all, answer[groups])
    live[groups] &= ~decided_all


def replay(sample, params, weight, lo, hi):
    words = sample['source_gate_words'].astype(np.int64)
    source = (words[:, None] & BITS[None, :, None, None]) != 0  # G,S,K,P
    counts = source.sum(2)
    products = np.matmul(source.astype(np.int16).transpose(0, 1, 3, 2),
                         (weight != 0).astype(np.int16).T).transpose(0, 1, 3, 2)
    a = params['temporal_int16'].astype(np.int64)
    tau = params['threshold_positive'][params['full_entry']].astype(np.int64)
    if not np.all(params['positive_gain'] == 1):
        raise ValueError('The captured student has positive compiled gains.')
    gold_y = sample['Yi'].reshape(64, 12, 10, 8, 4).transpose(0, 2, 1, 3, 4).reshape(64, 10, 96, 4)
    gold_u = np.einsum('ts,gshp->gthp', a, gold_y, dtype=np.int64)
    gold_gate = gold_u >= tau[None, :, :, None]
    captured = sample['gate_common3_diagonal_34_exact'].reshape(64, 12, 10, 8, 4)
    captured = captured.transpose(0, 2, 1, 3, 4).reshape(64, 10, 96, 4)
    if np.any(gold_gate != captured):
        raise RuntimeError('Complete integer function differs from captured gates.')
    data = dict(a=a, support=a != 0, tau=tau, lo=lo, hi=hi, empty=products == 0,
                gold_y=gold_y, gold_u=gold_u, gold_gate=gold_gate)
    names = ('Y_bound_pair_uses_before_reuse', 'Y_bound_endpoint_additions_before_bypass',
             'Y_bound_endpoint_additions_after_zero_bypass', 'Y_static_range_clip_comparisons',
             'Y_static_range_clip_comparisons_before_sign_bypass',
             'Y_static_range_pair_uses_shared_T_P', 'Y_bound_table_pairs_shared_T_P',
             'Y_bound_table_H8_bin_uses', 'check_P4_events', 'check_P4_H8_t_waves',
             'initial_selected_gates', 'later_selected_gates',
             'A_endpoint_products_before_zero_equal_skip', 'A_endpoint_products_after_zero_equal_skip',
             'A_exact_row_products_after_zero_skip', 'A_interval_row_products_after_zero_equal_skip',
             'A_endpoint_accumulation_additions', 'exact_gate_comparisons', 'interval_gate_comparisons',
             'initial_retired_gates', 'later_retired_gates', 'retired_before_all_Y_exact',
             'certified_gate_mismatches', 'propagated_U_bound_violations', 'Y_bound_violations',
             'complete_Y_mismatches', 'source_directory_P4_index_visits',
             'source_nonempty_T10_P4_descriptors', 'source_live_T10_P4_descriptors',
             'source_counter_decrements_shared_H', 'source_count_bin_drop_events')
    m = {k: 0 for k in names}
    for prefix in ('literal_complete', 'initial_only_complete', 'single_scan'):
        for suffix in ('products', 'W_H8_uses', 'W_scalar_uses', 'Conv_k_t_H8_visits'):
            m[prefix + '_' + suffix] = 0
    live = np.ones(gold_y.shape, dtype=bool)
    answer = np.zeros_like(live)
    partial = np.zeros(gold_y.shape, dtype=np.int64)
    remaining = counts.copy()
    check(partial, remaining, live, answer, live.copy(), data, m, initial=True)
    wanted = dependencies(live, data['support']) & ~data['empty']
    initial_wanted = wanted.copy()
    finish = np.full(64, -1, dtype=np.int64)
    finish[~live.any(axis=(1, 2, 3))] = 0
    for k in range(864):
        src = source[:, :, k]
        all_update = src[:, :, None] & (weight[:, k] != 0)[None, None, :, None]
        add_work(m, 'literal_complete', all_update)
        add_work(m, 'initial_only_complete', all_update & initial_wanted)
        group_live = live.any(axis=(1, 2, 3))
        if not group_live.any():
            continue  # Remaining work is only the independent complete baseline count.
        counter_live = wanted.any(2)
        active_source = src & counter_live
        m['source_directory_P4_index_visits'] += int(group_live.sum())
        m['source_nonempty_T10_P4_descriptors'] += int((group_live & src.any(axis=(1, 2))).sum())
        m['source_live_T10_P4_descriptors'] += int(active_source.any(axis=(1, 2)).sum())
        update = all_update & wanted
        partial += update * weight[:, k][None, None, :, None]
        add_work(m, 'single_scan', update)
        remaining -= active_source
        m['source_counter_decrements_shared_H'] += int(active_source.sum())
        dropped = active_source & ((remaining & (remaining - 1)) == 0)
        m['source_count_bin_drop_events'] += int(dropped.sum())
        # Only consumers of the actual (s,p) count event are checked. No
        # unrelated h/p/time row is added merely because its P4 group woke.
        affected = np.einsum('ts,gsp->gtp', data['support'].astype(np.int16),
                             dropped.astype(np.int16)) > 0
        selected = live & affected[:, :, None]
        if selected.any():
            check(partial, remaining, live, answer, selected, data, m)
            wanted = dependencies(live, data['support']) & ~data['empty']
        just_finished = group_live & ~live.any(axis=(1, 2, 3))
        finish[just_finished] = k + 1
    m['final_gate_mismatches'] = int((answer != gold_gate).sum())
    m['unresolved_final_gates'] = int(live.sum())
    m['checked_complete_T10_gates'] = int(gold_gate.size)
    m['source_cursor_sum_at_completion'] = int(finish.sum())
    m['source_cursor_max_at_completion'] = int(finish.max())
    m['peak_abs_partial_Y_final'] = int(np.abs(partial).max())
    m['ordinary_full_PSN_products_after_Y_zero_skip'] = sum(
        int(np.count_nonzero(a[:, s])) * int(np.count_nonzero(gold_y[:, s])) for s in range(10))
    m['ordinary_full_gate_comparisons'] = int(gold_gate.size)
    m['initial_source_T10_fields_observed'] = int(words.size)
    m['initial_source_T10_payload_packed_bytes'] = int(words.size * 10 // 8)
    m['initial_source_count_increments'] = int(counts.sum())
    m['initial_product_empty_Y_fields'] = int(products.size)
    m['initial_source_nonempty_P4_descriptors'] = int(source.any(axis=(1, 3)).sum())
    failures = sum(v for k, v in m.items() if k.endswith(('mismatches', 'violations', 'final_gates')))
    if failures:
        raise RuntimeError(f'Single-scan reference has {failures} numerical failures.')
    return m


def main():
    params = dict(np.load(HERE / 'integer_deployment/common3_diagonal_34.npz'))
    weight = params['weight_int8'].reshape(96, 864).astype(np.int64)
    lo, hi = compile_bounds(weight)
    rows = []
    for path in sorted((HERE / 'integer_valid10').glob('capture_*.npz')):
        with np.load(path) as sample:
            m = replay(sample, params, weight, lo, hi)
            rows.append(dict(capture=path.name, file=str(sample['file']), counts=m))
            print(path.name, m['single_scan_products'], m['single_scan_W_H8_uses'],
                  m['A_endpoint_products_after_zero_equal_skip'], flush=True)
    totals = {k: (max(r['counts'][k] for r in rows) if k.startswith('peak_') or k.endswith('_max_at_completion')
                  else sum(r['counts'][k] for r in rows)) for k in rows[0]['counts']}
    rates = {base: {suffix: 1 - totals['single_scan_' + suffix] / totals[base + '_' + suffix]
                   for suffix in ('products', 'W_H8_uses', 'W_scalar_uses', 'Conv_k_t_H8_visits')}
             for base in ('literal_complete', 'initial_only_complete')}
    result = dict(
        scope='same common3 INT8-W/Q14-A integer student; four captured frames, 64 P4/frame, all H96/T10',
        execution='one original k=0..863 scan per P4 group, common cursor across 12 H8; each W[k,H8] serves all still-needed time/p lanes once',
        trigger='initial all live gates; later only live consumers of each actual (s,p) remaining-count ceil-power-of-two bin drop, including zero',
        exactness='signed A propagates current partial-Y plus source-count weight bounds; certify Ulo>=tau or Uhi<tau. Captured Yi/U/gates are assertion-only labels.',
        ordinary_skips='source/W zero product filter, initially product-empty Yi, equal-endpoint reuse, zero operand bypass, exact-row single comparison, P/T equal-bin table sharing',
        baseline='literal complete single scan plus a stronger fixed-mask full scan retaining the same initial interval decisions; the latter is a count-only baseline, no additional parameter sweep',
        resources=dict(Y24_bytes_per_P4_H8=10*32*3,
                       working_Ulo_Uhi48_bytes_per_P4_H8=2*32*6,
                       optional_Y_endpoint24_temporaries_bytes_per_P4_H8=2*32*3,
                       output_bits_bytes_per_P4_H8=40, live_bits_bytes_per_P4_H8=40,
                       source_count10_bytes_per_P4_shared_H96=50,
                       product_empty_bits_bytes_per_P4_H8=40,
                       bound_table_12x96x2x24_bytes=6912, full_tau32_bytes=3840,
                       W_INT8_bytes=82944, W_nonzero_bitmap_bytes=10368),
        cost_boundary=[
            'Initial counts and product-empty fields are not free: report full source observations/count increments and metadata fields. They may be accumulated by the source producer; otherwise they require an extra source metadata pass before this single W/Conv scan.',
            'Source descriptors are logical P4 40-bit T10 fields. Empty-NRV selection, halo reuse, static W-mask lookups and product-empty reduction need real service.',
            'Table pair reuse across time/p within a P4 group is reported, not assumed to complete through one port in one cycle.',
            'Each check recomputes only the selected PSN rows. A products, interval accumulation, Y endpoint/clamp work, comparisons and P4/H8 row waves are separate counts; they have different datapath costs.',
            'Zero feedback latency and no outstanding W requests: cancellation is an optimistic work opportunity. No bank/port arbitration, control schedule, cycles or PPA is claimed.',
            'All ten Yi partials coexist. This is a larger-state strong reference, not an equal-state replacement for the five-Y predictor.'
        ], count_upper_bins=N_UP.tolist(), frames=rows, totals=totals, reductions=rates)
    out = HERE / 'single_scan_reference.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(out), reductions=rates, totals=totals), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
