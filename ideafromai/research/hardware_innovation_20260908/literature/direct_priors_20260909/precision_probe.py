"""One fixed H4/L4 precision-cascade opportunity probe; no cycle claim.

All source bits are available, but low-weight dot products are used only as
labels or for demanded fallback. Decisions use high-weight dots, source counts,
and legal cardinality bounds compiled solely from the actual low weights.
"""
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PATCH = HERE.parents[1] / 'algorithm/patch_probe/partial_completion'
BITS = 1 << np.arange(10)
PBIT = 1 << np.arange(4)
POP4 = np.asarray([x.bit_count() for x in range(16)], dtype=np.int64)
NAMES = ['row34', 'common3_diagonal_34']


def value_matrix(weight, source):
    return np.matmul(weight[None, None].astype(np.int64), source.astype(np.int64))


def packed_words(request, lengths):
    """64-bit payload words touched; vectors are packed in k order.

    One P4/H8 context at a time, no cache across contexts. A vector is at most
    64 bits; an extract crossing a word boundary incurs both word touches.
    """
    ends = np.cumsum(lengths, dtype=np.int64)
    starts = ends - lengths
    total = 0
    for row in request:
        idx = np.flatnonzero(row & (lengths > 0))
        if len(idx):
            total += len(np.unique(np.concatenate((starts[idx] // 64,
                                                   (ends[idx] - 1) // 64))))
    return total


def signed_width(x):
    for b in range(1, 9):
        if x.min() >= -(1 << (b - 1)) and x.max() < (1 << (b - 1)):
            return b
    raise ValueError('Not signed INT8')


def work(source, wanted, weight, mode):
    """Actual source x W-nonzero x needed-lane intersections, across all T."""
    sw = (source * PBIT).sum(-1).astype(np.uint8)  # G,T,K
    nw = (wanted * PBIT).sum(-1).astype(np.uint8)  # G,T,H
    result = dict(products=0, logical_H8_weight_vectors=0,
                  logical_weight_scalars=0, Conv_k_t_H8_visits=0,
                  scalar_source_terms_before_weight_zero_filter=0,
                  metadata_group_lookups=0, packed64_payload_words=0,
                  packed64_metadata_words=0, source_descriptors_per_H8=0,
                  coefficient_payload_bits=0, coefficient_metadata_bits=0,
                  unpack_nonzero_scalars=0)
    source_all_h = np.zeros((64, 864), dtype=bool)
    for hg in range(12):
        w = weight[hg * 8:(hg + 1) * 8].T
        inter = sw[:, :, :, None] & nw[:, :, None, hg * 8:(hg + 1) * 8]
        active = (inter != 0) & (w != 0)[None, None]
        request = active.any(axis=(1, 3))
        metadata = (inter != 0).any(axis=(1, 3))
        result['products'] += int((POP4[inter] * (w != 0)[None, None]).sum())
        result['scalar_source_terms_before_weight_zero_filter'] += int(POP4[inter].sum())
        result['Conv_k_t_H8_visits'] += int(active.any(3).sum())
        result['logical_H8_weight_vectors'] += int(request.sum())
        result['logical_weight_scalars'] += int(active.any(1).sum())
        result['metadata_group_lookups'] += int(metadata.sum())
        result['source_descriptors_per_H8'] += int(metadata.sum())
        source_all_h |= metadata
        nonzero = (w != 0).sum(1)
        if mode == 'full8':
            lengths = np.full(864, 64, dtype=np.int64)
            meta_bits = 8  # shared per-weight nonzero mask for pre-request filtering
        elif mode == 'bwac':
            widths = np.asarray([signed_width(v) for v in w])
            lengths = nonzero * widths
            meta_bits = 12  # 8-bit NZ map plus explicit 4-bit width in this probe
        else:
            lengths = nonzero * 4
            meta_bits = 8  # fixed nibble width; only NZ map is stored
        result['packed64_payload_words'] += packed_words(request, lengths)
        result['packed64_metadata_words'] += packed_words(metadata, np.full(864, meta_bits))
        result['coefficient_payload_bits'] += int(lengths.sum())
        result['coefficient_metadata_bits'] += 864 * meta_bits
        result['unpack_nonzero_scalars'] += int((request * nonzero).sum())
    result['source_descriptors_if_shared_across_all_H8'] = int(source_all_h.sum())
    result['source_packed64_words_if_shared_across_all_H8'] = packed_words(
        source_all_h, np.full(864, 40))  # each source word = P4 x T10 bits
    return result


def psn_terms(a, values, selected):
    result = dict(coefficient_terms=0, nonzero_coefficient_terms=0,
                  SIMD32_nonzero_t_s_visits=0)
    for s in range(10):
        active = selected & (a[:, s] != 0)[None, :, None, None]
        nonzero = active & (values[:, s, None] != 0)
        result['coefficient_terms'] += int(active.sum())
        result['nonzero_coefficient_terms'] += int(nonzero.sum())
        packed = nonzero.reshape(64, 10, 12, 8, 4)
        result['SIMD32_nonzero_t_s_visits'] += int(packed.any(axis=(3, 4)).sum())
    return result


def decided_gate(ulo, uhi, params):
    pos = params['threshold_positive'][params['full_entry']][None, :, :, None]
    neg = params['threshold_negative'][params['full_entry']][None, :, :, None]
    sign = params['positive_gain'][None, None, :, None]
    positive = np.where(sign > 0, ulo >= pos, uhi <= pos)
    negative = np.where(sign > 0, uhi <= neg, ulo >= neg)
    constant = sign == 0
    fixed = params['constant_gate'][None, :, :, None]
    return positive | negative | constant, np.where(constant, fixed, positive)


def run_frame(sample, params, name, high, low, yhigh, ylow, source):
    a = params['temporal_int16'].astype(np.int64)
    n = source.sum(axis=2, dtype=np.int64)  # G,T,P; no per-H future-value oracle
    ascending = np.sort(low, axis=1)
    lower_table = np.pad(np.cumsum(ascending, axis=1), ((0, 0), (1, 0)))
    upper_table = np.pad(np.cumsum(ascending[:, ::-1], axis=1), ((0, 0), (1, 0)))
    yl_lo = lower_table[:, n].transpose(1, 2, 0, 3)
    yl_hi = upper_table[:, n].transpose(1, 2, 0, 3)
    if np.any(ylow < yl_lo) or np.any(ylow > yl_hi):
        raise RuntimeError('Cardinality bound is not valid on captured inputs')
    high_u = np.einsum('ts,gshp->gthp', a, 16 * yhigh, dtype=np.int64)
    ap, an = np.maximum(a, 0), np.minimum(a, 0)
    ulo = high_u + np.einsum('ts,gshp->gthp', ap, yl_lo, dtype=np.int64) \
        + np.einsum('ts,gshp->gthp', an, yl_hi, dtype=np.int64)
    uhi = high_u + np.einsum('ts,gshp->gthp', ap, yl_hi, dtype=np.int64) \
        + np.einsum('ts,gshp->gthp', an, yl_lo, dtype=np.int64)
    done, early_gate = decided_gate(ulo, uhi, params)
    live = ~done
    need_low = np.einsum('ts,gthp->gshp', (a != 0).astype(np.int32),
                         live.astype(np.int32)) > 0
    exact_u = np.einsum('ts,gshp->gthp', a, 16 * yhigh + ylow, dtype=np.int64)
    _, truth = decided_gate(exact_u, exact_u, params)
    fallback = high_u + np.einsum('ts,gshp->gthp', a,
                                  np.where(need_low, ylow, 0), dtype=np.int64)
    _, fallback_gate = decided_gate(fallback, fallback, params)
    answer = np.where(done, early_gate, fallback_gate)
    if np.any(answer != truth):
        raise RuntimeError('Precision cascade changes the complete integer gate')
    captured = sample['gate_' + name + '_exact'].reshape(64, 12, 10, 8, 4)
    captured = captured.transpose(0, 2, 1, 3, 4).reshape(64, 10, 96, 4)
    if np.any(truth != captured):
        raise RuntimeError('Full-entry threshold disagrees with GPU exact gates')
    all_rows = np.ones_like(truth)
    stages = {
        'ordinary_full8': work(source, all_rows, params['weight_int8'].reshape(96, 864), 'full8'),
        'ordinary_bwac': work(source, all_rows, params['weight_int8'].reshape(96, 864), 'bwac'),
        'high4': work(source, all_rows, high, 'high4'),
        'low4_demanded': work(source, need_low, low, 'low4'),
        'low4_all_control': work(source, all_rows, low, 'low4'),
    }
    fees = dict(
        high_psn=psn_terms(a, 16*yhigh, all_rows),
        lower_endpoint=psn_terms(a, yl_lo, all_rows),
        upper_endpoint=psn_terms(a, yl_hi, all_rows),
        low_psn=psn_terms(a, np.where(need_low, ylow, 0), live),
        interval_comparisons=2*int(truth.size),
        fallback_final_comparisons=int(live.sum()),
        source_population_counts_shared_across_H=64*10*4,
        bound_table_pairs_without_count_reuse=64*10*4*96,
        bound_table_pairs_if_shared_across_equal_n=sum(len(np.unique(n[g]))*96 for g in range(64)),
        source_scans=2,
    )
    # A signed coefficient selects opposite low/high Y endpoints. Count actual
    # nonzero operand work, rather than using the same orientation on both.
    for label, side in [('lower_endpoint', 0), ('upper_endpoint', 1)]:
        nnz = 0
        visits = 0
        for t in range(10):
            for s in np.flatnonzero(a[t]):
                value = (yl_lo if ((a[t, s] > 0) == (side == 0)) else yl_hi)[:, s]
                nz = value != 0
                nnz += int(nz.sum())
                visits += int(nz.reshape(64, 12, 8, 4).any(axis=(2, 3)).sum())
        fees[label]['nonzero_coefficient_terms'] = nnz
        fees[label]['SIMD32_nonzero_t_s_visits'] = visits
    source_empty = (n == 0)[:, :, None, :]
    ordinary_empty_done = np.stack([
        np.all(np.broadcast_to(source_empty, yhigh.shape)[:, a[t] != 0], axis=1)
        for t in range(10)], axis=1)
    meaningful = ~ordinary_empty_done
    return dict(file=str(sample['file'].item()), gates=int(truth.size),
                exact_spikes=int(truth.sum()), gate_mismatches=0,
                early_decided_gates=int(done.sum()),
                source_empty_decided_gates=int(ordinary_empty_done.sum()),
                additional_decided_gates_over_source_empty=int((done & meaningful).sum()),
                all_T_per_ph_done=int(done.all(1).sum()),
                all_P4_H8_T_done=int(done.reshape(64,10,12,8,4).all(axis=(1,3,4)).sum()),
                low_Y_elements_cancelled=int((~need_low).sum()),
                nonempty_low_Y_elements_cancelled=int(((~need_low) & ~source_empty).sum()),
                stages=stages, fees=fees)


def main():
    start = time.time()
    parameters = {n: dict(np.load(PATCH/'integer_deployment'/f'{n}.npz')) for n in NAMES}
    weight = parameters[NAMES[0]]['weight_int8'].reshape(96,864).astype(np.int64)
    if not np.array_equal(weight, parameters[NAMES[1]]['weight_int8'].reshape(96,864)):
        raise RuntimeError('The two structures do not share the same W')
    high = np.floor_divide(weight, 16)
    low = weight - 16*high
    if high.min() < -8 or high.max() > 7 or low.min() < 0 or low.max() > 15:
        raise RuntimeError('H4/L4 decomposition range error')
    results = {n: [] for n in NAMES}
    for path in sorted((PATCH/'integer_valid10').glob('capture_*.npz')):
        sample = dict(np.load(path))
        source = (sample['source_gate_words'][:,None] & BITS[None,:,None,None]) != 0
        yh, yl = value_matrix(high, source), value_matrix(low, source)
        captured_y = sample['Yi'].reshape(64,12,10,8,4).transpose(0,2,1,3,4).reshape(64,10,96,4)
        if np.any(16*yh + yl != captured_y):
            raise RuntimeError('High/low reconstruction disagrees with exact captured Yi')
        for name in NAMES:
            row = run_frame(sample, parameters[name], name, high, low, yh, yl, source)
            results[name].append(row)
            print(path.stem, name, row['early_decided_gates'], row['gate_mismatches'], flush=True)
    totals = {}
    for name, rows in results.items():
        sums = {k: sum(r[k] for r in rows) for k in rows[0] if isinstance(rows[0][k], int)}
        sums['stages'] = {
            stage: {k: sum(r['stages'][stage][k] for r in rows)
                    for k in rows[0]['stages'][stage]} for stage in rows[0]['stages']}
        # Static model size is shared, not multiplied by four frames.
        for stage in sums['stages']:
            for k in ['coefficient_payload_bits', 'coefficient_metadata_bits']:
                sums['stages'][stage][k] = rows[0]['stages'][stage][k]
        sums['fees'] = {}
        for k, v in rows[0]['fees'].items():
            sums['fees'][k] = ({q: sum(r['fees'][k][q] for r in rows) for q in v}
                                if isinstance(v, dict) else sum(r['fees'][k] for r in rows))
        st = sums['stages']
        cascade = {k: st['high4'][k]+st['low4_demanded'][k]
                   for k in ['products','logical_H8_weight_vectors','packed64_payload_words',
                             'packed64_metadata_words','metadata_group_lookups',
                             'source_descriptors_per_H8','source_descriptors_if_shared_across_all_H8']}
        sums['cascade_combined'] = cascade
        sums['ratios'] = {
            'conv_products_vs_full': cascade['products']/st['ordinary_full8']['products'],
            'payload64_words_vs_full8': cascade['packed64_payload_words']/st['ordinary_full8']['packed64_payload_words'],
            'payload64_words_vs_bwac': cascade['packed64_payload_words']/st['ordinary_bwac']['packed64_payload_words'],
            'payload_plus_header64_vs_bwac': (cascade['packed64_payload_words']+cascade['packed64_metadata_words']) /
                (st['ordinary_bwac']['packed64_payload_words']+st['ordinary_bwac']['packed64_metadata_words']),
            'early_decided_fraction': sums['early_decided_gates']/sums['gates'],
            'low_products_cancelled_vs_split_always': 1-st['low4_demanded']['products']/st['low4_all_control']['products'],
        }
        a = parameters[name]['temporal_int16'].astype(np.int64)
        high_bound = np.abs(a).sum(1)[:,None] * (16*np.abs(high).sum(1))[None]
        high_plus_low_bound = np.abs(a).sum(1)[:,None] * (16*np.abs(high).sum(1)+low.sum(1))[None]
        sums['static_U_bounds'] = dict(high_abs=int(high_bound.max()),
            high_plus_arbitrary_low_prefix_abs=int(high_plus_low_bound.max()),
            fits_signed32=bool(high_plus_low_bound.max() < 2**31))
        totals[name] = sums
    max_low_sum = int(low.sum(1).max())
    table_bits = max(1,max_low_sum.bit_length())
    output = dict(
        scope='Four fixed validation frames, G64 native P4 per frame, all H96/T10. Fixed H4/L4 split, no training or parameter sweep.',
        function='Exact stored INT8/Q14 integer students and full-entry thresholds. Source theta is already folded in W; emitted theta remains unchanged.',
        decision='Only high-Y, source counts and weight-only cardinality tables drive early decisions; low dot values are labels or selected fallback operands.',
        storage_and_cost=dict(
            bounds='Sharp count-only sorted-L bounds; optimistic full cardinality table, not admitted to the existing small state budget.',
            bound_table_shape=[2,96,865],bound_value_bits=table_bits,
            bound_table_bare_bytes=(2*96*865*table_bits+7)//8,
            high_low_zero_maps_bytes=2*96*864//8,
            raw_full_zero_map_bytes=96*864//8,
            high_or_low_Y_allT_P4H8_bytes_16bit=10*32*2,
            Ucore_allT_P4H8_bytes_32bit=10*32*4,
            gate_and_unresolved_bytes=2*10*32//8,
            lifecycle='All-T high/low state and Ucore are not assumed to coexist for free. Peak state and ports require an explicit schedule; no equal-2KiB service claim.',
            floor_split='Offline exact integer compilation; no online divider. Two sign/zero maps and separate packed payload streams are counted.',
            packing='Each H8 group packs its K vectors consecutively into 64-bit words. Full8 is 64 bits/vector; BWAC uses per-H8 minimum signed width and skips zero scalar payloads; high/low use fixed 4 bits/nonzero. Cross-word extracts touch both words. Metadata is separate.',
            bus='Payload-word touches per P4/H8 context, not DRAM transactions or cycles; cache residency and global coalescing may reduce both controls. Full and cascade share T/P reuse.',
            source='Two stages can rescan sources. Per-H8 and all-H-shared descriptor counts are both exposed; neither is free.',
        ), axes=totals, frames=results,elapsed_seconds=time.time()-start,
        limitations=['No AEE experiment: this probe preserves each existing exact integer function on captured gates.',
                     'No hardware pipeline/latency/energy claim; nibble adds still counted as separate accumulation operations.',
                     'Full bound table, counts, interval MACs, decompression and metadata are paid/identified; table accesses are not free latency.',
                     'This ordinary precision-cascade control does not combine the statistical gamma3 policy.'])
    (HERE/'precision_probe.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({n:v['ratios'] for n,v in totals.items()},indent=2),flush=True)


if __name__ == '__main__':
    main()
