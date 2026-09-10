"""One fixed-prefix source-count bound probe; integer opportunities, not cycles.

Both graphs start after Yi columns 2, 3 and 7. Ordinary source/weight product
emptiness is also given to both controls. The twelve-bin bound allows at most
n_up arbitrary active source indices, so it does not use their actual identity.
Actual source identities are used for golden values and work opportunities.
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PREFIX = (2, 3, 7)
TAIL = np.array([s for s in range(10) if s not in PREFIX])
N_UP = np.array([0] + [1 << k for k in range(11)], dtype=np.int64)
TIME_BITS = 1 << np.arange(10)
P_BITS = 1 << np.arange(4)
NAMES = ('row34', 'common3_diagonal_34')


def compile_bounds(weight):
    ordered = np.sort(weight.astype(np.int64), axis=1)
    low_prefix = np.pad(np.cumsum(np.minimum(ordered, 0), axis=1), ((0, 0), (1, 0)))
    high_prefix = np.pad(np.cumsum(np.maximum(ordered[:, ::-1], 0), axis=1), ((0, 0), (1, 0)))
    return low_prefix[:, np.minimum(N_UP, 864)].T, high_prefix[:, np.minimum(N_UP, 864)].T


def capture_geometry(sample, weight):
    words = sample['source_gate_words'].astype(np.int64)  # G,K,P; bit=t.
    gates = ((words[:, None] & TIME_BITS[None, :, None, None]) != 0)
    counts = gates.sum(axis=2)  # G,S,P, independently for each p.
    bins = np.searchsorted(N_UP, counts)
    # G,HG,S,h,p -> G,S,H,p. h and p must not exchange places.
    y = sample['Yi'].reshape(64, 12, 10, 8, 4).transpose(0, 2, 1, 3, 4).reshape(64, 10, 96, 4)
    # Nonzero product count uses W!=0, not the possibly cancelling Yi value.
    products = np.matmul(gates.astype(np.int16).transpose(0, 1, 3, 2),
                         (weight != 0).astype(np.int16).T).transpose(0, 1, 3, 2)
    pwords = (gates * P_BITS[None, None, None]).sum(-1).astype(np.uint8)
    metadata = dict(
        logical_source_T10_field_observations=int(words.size),
        logical_source_T10_payload_packed_bytes=int(words.size * 10 // 8),
        source_bit_observations=int(gates.size),
        active_bit_counter_increments=int(counts.sum()),
        count_fields=int(counts.size),
        max_source_count=int(counts.max()),
        count_bin_population=np.bincount(bins.ravel(), minlength=12).tolist(),
    )
    return y.astype(np.int64), products, pwords, counts, bins, metadata


def logical_work(pwords, wanted, weight, columns=TAIL):
    """Each tail column use and an all-tail ideal reuse lower bound.

    A logical vector is W[k,H8], shared by requested p and h within that use.
    No physical word size, cache hit or feasible five-Y tail batch is assumed.
    """
    masks = (wanted * P_BITS[None, None, None]).sum(-1).astype(np.uint8)
    source = pwords[:, columns]
    vectors = scalars = unique_vectors = unique_scalars = 0
    for hg in range(12):
        h0 = hg * 8
        requested = masks[:, columns, h0:h0 + 8]
        active = ((source[:, :, :, None] & requested[:, :, None, :]) != 0)
        active &= (weight[h0:h0 + 8].T != 0)[None, None]
        vectors += int(active.any(-1).sum())
        scalars += int(active.sum())
        unique_vectors += int(active.any(axis=(1, 3)).sum())
        unique_scalars += int(active.any(1).sum())
    return dict(tail_separate_W_H8_uses=vectors,
                tail_separate_W_scalar_uses=scalars,
                all_tail_unique_W_H8_uses=unique_vectors,
                all_tail_unique_W_scalar_uses=unique_scalars)


def probe(sample, params, geometry, low_table, high_table, name):
    y, products, pwords, counts, bins, _ = geometry
    weight = params['weight_int8'].reshape(96, 864)
    aq = params['temporal_int16'].astype(np.int64)
    support = aq != 0
    if not np.all(params['positive_gain'] == 1):
        raise ValueError('This probe uses the current positive residual scale compilation.')
    tau = params['threshold_positive'][params['full_entry']]
    low = low_table[bins].transpose(0, 1, 3, 2)
    high = high_table[bins].transpose(0, 1, 3, 2)
    table_violations = int(((y < low) | (y > high)).sum())
    known = products == 0  # Optimistic ordinary pre-request product-empty control.
    known[:, PREFIX] = True
    low = np.where(known, y, low)
    high = np.where(known, y, high)
    ap, an = np.maximum(aq, 0), np.minimum(aq, 0)
    ulo = (np.einsum('ts,gshp->gthp', ap, low, dtype=np.int64)
           + np.einsum('ts,gshp->gthp', an, high, dtype=np.int64))
    uhi = (np.einsum('ts,gshp->gthp', ap, high, dtype=np.int64)
           + np.einsum('ts,gshp->gthp', an, low, dtype=np.int64))
    full_u = np.einsum('ts,gshp->gthp', aq, y, dtype=np.int64)
    full_gate = full_u >= tau[None, :, :, None]
    recorded_gate = sample[f'gate_{name}_exact'].reshape(64, 12, 10, 8, 4)
    recorded_gate = recorded_gate.transpose(0, 2, 1, 3, 4).reshape(64, 10, 96, 4)
    positive = ulo >= tau[None, :, :, None]
    negative = uhi < tau[None, :, :, None]
    certified = positive | negative
    unknown_terms = np.einsum('ts,gshp->gthp', support.astype(np.int16),
                              (~known).astype(np.int16))
    ordinary_done = unknown_terms == 0
    before = ((np.einsum('ts,gthp->gshp', support.astype(np.int16),
                        (~ordinary_done).astype(np.int16)) > 0) & ~known)
    after = ((np.einsum('ts,gthp->gshp', support.astype(np.int16),
                       (~certified).astype(np.int16)) > 0) & ~known)
    # Only actual nonzero products are eligible. A cancelling but unproduced
    # Yi==0 is NOT made known by inspecting its golden value.
    before &= products > 0
    after &= products > 0
    cancelled = before & ~after
    before_columns = before.reshape(64, 10, 12, 8, 4).any(axis=(3, 4))
    after_columns = after.reshape(64, 10, 12, 8, 4).any(axis=(3, 4))
    metrics = dict(
        contexts=64 * 12,
        gates=int(full_gate.size),
        ordinary_completed_gates=int(ordinary_done.sum()),
        additional_exact_gates=int((certified & ~ordinary_done).sum()),
        additional_exact_positive_gates=int((positive & ~ordinary_done).sum()),
        additional_exact_negative_gates=int((negative & ~ordinary_done).sum()),
        eligible_tail_columns=int(before_columns.sum()),
        fully_cancelled_tail_columns=int((before_columns & ~after_columns).sum()),
        eligible_tail_lanes=int(before.sum()),
        cancelled_tail_lanes=int(cancelled.sum()),
        tail_active_products_before=int((products * before).sum()),
        tail_active_products_after=int((products * after).sum()),
        cancelled_tail_active_products=int((products * cancelled).sum()),
        bound_gate_comparisons=int((~ordinary_done).sum()) * 2,
        bound_nonzero_A_terms=int((unknown_terms * ~ordinary_done).sum()),
        bound_endpoint_products_before_zero_skip=int((unknown_terms * ~ordinary_done).sum()) * 2,
        bound_table_pairs_before_P4_count_bin_reuse=int(before.sum()),
        known_prefix_active_products=int(products[:, PREFIX].sum()),
        table_Y_bound_violations=table_violations,
        product_empty_Y_violations=int(((products == 0) & (y != 0)).sum()),
        propagated_U_bound_violations=int(((full_u < ulo) | (full_u > uhi)).sum()),
        certified_gate_mismatches=int((certified & (positive != full_gate)).sum()),
        full_integer_capture_gate_mismatches=int((full_gate != recorded_gate).sum()),
        ordinary_done_missing_from_certificate=int((ordinary_done & ~certified).sum()),
    )
    # Equal n_up at the four p positions can broadcast one h table pair.
    pairs = 0
    for bin_id in range(12):
        pairs += int((before & (bins == bin_id)[:, :, None]).any(-1).sum())
    metrics['bound_table_pairs_with_P4_count_bin_reuse'] = pairs
    shared_endpoint_products = 0
    for s in TAIL:
        for t in np.flatnonzero(support[:, s]):
            for bin_id in range(1, 12):
                active = ((~ordinary_done[:, t]) & (~known[:, s])
                          & (bins[:, s] == bin_id)[:, None])
                endpoints = (low_table[bin_id] != 0).astype(np.int16)
                endpoints += (high_table[bin_id] != 0).astype(np.int16)
                shared_endpoint_products += int((active.any(-1) * endpoints[None]).sum())
    metrics['bound_endpoint_products_with_P4_count_bin_reuse'] = shared_endpoint_products
    for label, wanted in [('before', before), ('after', after)]:
        for key, value in logical_work(pwords, wanted, weight).items():
            metrics[key + '_' + label] = value
    for key in ('tail_separate_W_H8_uses', 'tail_separate_W_scalar_uses',
                'all_tail_unique_W_H8_uses', 'all_tail_unique_W_scalar_uses'):
        metrics[key + '_cancelled'] = metrics[key + '_before'] - metrics[key + '_after']
    prefix_work = logical_work(pwords, products > 0, weight, columns=np.array(PREFIX))
    metrics['known_prefix_single_batch_W_H8_uses'] = prefix_work['all_tail_unique_W_H8_uses']
    metrics['prefix_plus_tail_active_products_before'] = (
        metrics['known_prefix_active_products'] + metrics['tail_active_products_before'])
    metrics['prefix_plus_separate_tail_W_H8_uses_before'] = (
        metrics['known_prefix_single_batch_W_H8_uses'] + metrics['tail_separate_W_H8_uses_before'])
    errors = sum(v for k, v in metrics.items()
                 if k.endswith(('violations', 'mismatches', 'missing_from_certificate')))
    if errors:
        raise RuntimeError(f'{name}: {errors} strict-bound or numeric failures')
    return metrics


def rates(m):
    pairs = [('tail_column_cancel_fraction', 'fully_cancelled_tail_columns', 'eligible_tail_columns'),
             ('tail_lane_cancel_fraction', 'cancelled_tail_lanes', 'eligible_tail_lanes'),
             ('tail_active_product_cancel_fraction', 'cancelled_tail_active_products', 'tail_active_products_before'),
             ('tail_separate_W_H8_cancel_fraction', 'tail_separate_W_H8_uses_cancelled', 'tail_separate_W_H8_uses_before'),
             ('all_tail_unique_W_H8_cancel_fraction', 'all_tail_unique_W_H8_uses_cancelled', 'all_tail_unique_W_H8_uses_before'),
             ('prefix_plus_tail_active_product_cancel_fraction', 'cancelled_tail_active_products', 'prefix_plus_tail_active_products_before'),
             ('prefix_plus_separate_tail_W_H8_cancel_fraction', 'tail_separate_W_H8_uses_cancelled', 'prefix_plus_separate_tail_W_H8_uses_before')]
    return {key: (m[n] / m[d] if m[d] else 0.0) for key, n, d in pairs}


def main():
    params = {n: dict(np.load(HERE / 'integer_deployment' / f'{n}.npz')) for n in NAMES}
    weight = params[NAMES[0]]['weight_int8'].reshape(96, 864)
    if not np.array_equal(weight, params[NAMES[1]]['weight_int8'].reshape(96, 864)):
        raise ValueError('The two current structures must share their real quantized Conv1 W.')
    low_table, high_table = compile_bounds(weight)
    result = dict(
        scope='four real captures, 64 native P4 groups/frame, all 96 H; one fixed known prefix, not full-layer representativeness',
        prefix_columns=list(PREFIX), count_upper_bins=N_UP.tolist(),
        bound='L: up to n_up most negative W; U: up to n_up most positive W. Source theta and fixed-BN gain are already folded into INT8 W; no arbitrary source amplitudes are assumed.',
        decision='full Ui >= integer tau; exact positive if Ulo>=tau, exact negative if Uhi<tau. No calibrated mean/radius is used.',
        strong_ordinary_control='both graphs know the prefix and source/weight-product-empty Yi exactly; equality of a nonempty golden Yi to zero does not make it known. Product-empty metadata generation is not free.',
        resources=dict(bound_table_bytes_12x96x2x24=6912,
                       bound_table_plus_full_tau32_bytes=6912 + 10 * 96 * 4,
                       counter_bits_per_P4_group=4 * 10 * 10,
                       counter_bytes_per_P4_group=50,
                       counter_sharing='the 40 source-count fields are shared across all 12 H8 groups',
                       full_W_nonzero_bitmap_bytes=10368,
                       full_INT8_Conv1_W_bytes=82944,
                       table_min=int(low_table.min()), table_max=int(high_table.max())),
        cost_boundary=['table lookups and signed A endpoint propagation are counted as operations, not scheduled service',
                       'bound products may share equal n_up across P4, but the prefix Ui and final comparisons remain per lane',
                       'one working Ui cannot retain two interval endpoints for free; sequential recomputation or extra temporary storage is not modelled',
                       'source directory scan/count reduction, metadata AND/empty reduction, table ports and multiplier sharing need real implementation',
                       'source T10 fields are logical im2col observations; halo reuse and existing line buffers can avoid physical rereads',
                       'per-tail W uses count each temporal column independently; all-tail unique W assumes ideal retention and is not a legal five-Y batch schedule',
                       'no RTL, cycles, SRAM mapping, PPA, full-network AEE, or net benefit follows from these cancellation counts'],
        frames=[], axes={n: dict(frames=[]) for n in NAMES})
    for path in sorted((HERE / 'integer_valid10').glob('capture_*.npz')):
        with np.load(path) as sample:
            geometry = capture_geometry(sample, weight)
            result['frames'].append(dict(capture=path.name, file=str(sample['file']), **geometry[-1]))
            for name in NAMES:
                m = probe(sample, params[name], geometry, low_table, high_table, name)
                result['axes'][name]['frames'].append(dict(capture=path.name, counts=m, rates=rates(m)))
                print(path.name, name, json.dumps(rates(m)), flush=True)
    for name in NAMES:
        frames = result['axes'][name]['frames']
        totals = {key: sum(f['counts'][key] for f in frames) for key in frames[0]['counts']}
        result['axes'][name]['counts'] = totals
        result['axes'][name]['rates'] = rates(totals)
    path = HERE / 'source_count_bound_probe.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(path, flush=True)


if __name__ == '__main__':
    main()
