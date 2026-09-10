"""Fixed-frontier probe of exact temporal-pattern residual certificates.

The numerical student never changes. Pattern histograms inspect source words,
not future weight products. Their construction and storage are explicit costs.
These are checkpoint opportunities, not a cancellation schedule or cycles.
"""
import json
from pathlib import Path

import numpy as np

from source_count_bound_probe import N_UP, PREFIX, TAIL, capture_geometry

HERE = Path(__file__).resolve().parent
FRONTIERS = (0, 216, 432, 648, 864)


def bases(a):
    words = np.arange(1024, dtype=np.int64)
    bits = ((words[:, None] >> np.arange(10)) & 1).astype(np.int64)
    out = {'independent_Y': (bits, a.copy())}
    for name, shared in [('pair_2_3', (2, 3)), ('common_word', PREFIX)]:
        outside = [s for s in range(10) if s not in shared]
        local = sum(bits[:, s] << j for j, s in enumerate(shared))
        features = [bits[:, s] for s in outside]
        columns = [a[:, s] for s in outside]
        for mask in range(1, 1 << len(shared)):
            features.append((local == mask).astype(np.int64))
            columns.append(sum(a[:, s] for j, s in enumerate(shared) if mask & (1 << j)))
        out[name] = np.array(features).T, np.array(columns).T
    features, columns = [], []
    for t in range(10):
        support = np.flatnonzero(a[t])
        local = sum(bits[:, s] << j for j, s in enumerate(support))
        for mask in range(1, 1 << len(support)):
            features.append((local == mask).astype(np.int64))
            col = np.zeros(10, dtype=np.int64)
            col[t] = sum(a[t, s] for j, s in enumerate(support) if mask & (1 << j))
            columns.append(col)
    out['row_word'] = np.array(features).T, np.array(columns).T
    for name, (lookup, coefficients) in out.items():
        if not np.array_equal(lookup @ coefficients.T, bits @ a.T):
            raise RuntimeError(f'{name}: temporal coefficient identity is not exact')
    return out


def suffix_tables(weight, k):
    ordered = np.sort(weight[:, k:], axis=1)
    lo = np.pad(np.minimum(ordered, 0).cumsum(1), ((0, 0), (1, 0)))
    hi = np.pad(np.maximum(ordered[:, ::-1], 0).cumsum(1), ((0, 0), (1, 0)))
    sizes = np.minimum(N_UP, ordered.shape[1])
    return lo[:, sizes].T, hi[:, sizes].T


def shared_endpoint_cost(counts, coeff, lo_table, hi_table):
    """Same h/count-bin/coef endpoints may broadcast across the actual P4.

    No broadcast across independent spatial groups is granted. Different
    features with identical coefficient and bin are merged within this check.
    """
    bins = np.searchsorted(N_UP, counts).reshape(64, 4, -1)
    terms = table_pairs = 0
    for b in range(1, len(N_UP)):
        occupied = (bins == b).any(1)  # G,F
        used = occupied.any(1)
        table_pairs += int(used.sum()) * 96
        endpoints = int(np.count_nonzero(lo_table[b])) + int(np.count_nonzero(hi_table[b]))
        for c in np.unique(coeff):
            if c:
                used_coeff = (occupied & (coeff == c).any(0)[None]).any(1)
                terms += int(used_coeff.sum()) * endpoints
    return dict(endpoint_products_with_P4_coefficient_bin_reuse=terms,
                W_bound_table_pairs_with_P4_feature_bin_reuse=table_pairs)


def cancellation_counts(done, support, remaining_products, words, weight, k):
    wanted = np.einsum('ts,gth->gsh', support.astype(np.int16),
                       (~done).astype(np.int16), dtype=np.int16) > 0
    before = remaining_products > 0
    cancelled = before & ~wanted
    products = int((remaining_products * cancelled).sum())
    # Same W[k,H8] shared across every requested T and native P4.
    requested = wanted.reshape(64, 4, 10, 12, 8).transpose(0, 2, 3, 4, 1)
    future = words[:, k:]
    wuses = fulluses = 0
    for hg in range(12):
        wnonzero = (weight[hg * 8:hg * 8 + 8, k:].T != 0)
        live = np.zeros((64, future.shape[1], 8), dtype=bool)
        all_products = np.zeros_like(live)
        for s in range(10):
            source = (future & (1 << s)) != 0
            active = source[:, :, None] & requested[:, s, hg, None]
            live |= active.any(-1)
            all_products |= source.any(-1)[:, :, None]
        live &= wnonzero[None]
        all_products &= wnonzero[None]
        wuses += int(live.any(-1).sum())
        fulluses += int(all_products.any(-1).sum())
    return dict(cancelled_remaining_Conv_products=products,
                remaining_Conv_products_before=int(remaining_products.sum()),
                remaining_W_H8_uses_after=wuses, remaining_W_H8_uses_before=fulluses,
                cancelled_nonempty_Y_lanes=int(cancelled.sum()))


def run_capture(path, params, all_bases):
    weight = params['weight_int8'].reshape(96, 864).astype(np.int64)
    a = params['temporal_int16'].astype(np.int64)
    tau = params['threshold_positive'][params['full_entry']]
    with np.load(path) as sample:
        y, products, _, _, _, _ = capture_geometry(sample, weight)
        words = sample['source_gate_words'].astype(np.int64)
        recorded = sample['gate_common3_diagonal_34_exact']
        file = str(sample['file'])
    y = y.transpose(0, 3, 1, 2).reshape(256, 10, 96)
    remaining_products = products.transpose(0, 3, 1, 2).reshape(256, 10, 96).copy()
    flat_words = words.transpose(0, 2, 1).reshape(256, 864)
    source = ((flat_words[:, :, None] >> np.arange(10)) & 1).astype(np.int16)
    full_u = np.einsum('ts,gsh->gth', a, y, dtype=np.int64)
    expected = full_u >= tau[None]
    recorded = recorded.reshape(64, 12, 10, 8, 4).transpose(0, 4, 2, 1, 3).reshape(256, 10, 96)
    if not np.array_equal(recorded, expected):
        raise RuntimeError('Captured complete gates do not match the exact integer reference')
    partial = np.zeros_like(y)
    hist = {name: lookup[flat_words].sum(1) for name, (lookup, _) in all_bases.items()}
    records = []
    previous = 0
    for k in FRONTIERS:
        if k > previous:
            events = source[:, previous:k].transpose(0, 2, 1).astype(np.int64)
            partial += events @ weight[:, previous:k].T
            remaining_products -= events @ (weight[:, previous:k] != 0).astype(np.int64).T
            for name, (lookup, _) in all_bases.items():
                hist[name] -= lookup[flat_words[:, previous:k]].sum(1)
        u_partial = np.einsum('ts,gsh->gth', a, partial, dtype=np.int64)
        lo_table, hi_table = suffix_tables(weight, k)
        intervals = {}
        for name, (_, coeff) in all_bases.items():
            bins = np.searchsorted(N_UP, hist[name])
            lo, hi = lo_table[bins], hi_table[bins]
            if name == 'independent_Y':
                lo = np.where(remaining_products == 0, 0, lo)
                hi = np.where(remaining_products == 0, 0, hi)
            cp, cn = np.maximum(coeff, 0), np.minimum(coeff, 0)
            lower = u_partial + np.einsum('tf,gfh->gth', cp, lo, dtype=np.int64) + np.einsum('tf,gfh->gth', cn, hi, dtype=np.int64)
            upper = u_partial + np.einsum('tf,gfh->gth', cp, hi, dtype=np.int64) + np.einsum('tf,gfh->gth', cn, lo, dtype=np.int64)
            if np.any(full_u < lower) or np.any(full_u > upper):
                raise RuntimeError(f'{path.name} k={k} {name}: bound excludes the exact final membrane')
            intervals[name] = lower, upper
        base_lo, base_hi = intervals['independent_Y']
        base_done = (base_lo >= tau[None]) | (base_hi < tau[None])
        ordinary_done = np.einsum('ts,gsh->gth', (a != 0).astype(np.int16),
                                  (remaining_products > 0).astype(np.int16), dtype=np.int16) == 0
        for name, (_, coeff) in all_bases.items():
            lower, upper = intervals[name]
            raw_done = (lower >= tau[None]) | (upper < tau[None])
            joint_lo, joint_hi = np.maximum(base_lo, lower), np.minimum(base_hi, upper)
            positive, negative = joint_lo >= tau[None], joint_hi < tau[None]
            done = positive | negative
            if np.any(done & (positive != expected)):
                raise RuntimeError(f'{path.name} k={k} {name}: incorrect gate certificate')
            cost = shared_endpoint_cost(hist[name], coeff, lo_table, hi_table)
            active_endpoints = (lo_table[np.searchsorted(N_UP, hist[name])] != 0).astype(np.int64)
            active_endpoints += (hi_table[np.searchsorted(N_UP, hist[name])] != 0)
            raw_endpoint = int(np.einsum('tf,gfh->', (coeff != 0).astype(np.int64),
                                         active_endpoints))
            record = dict(capture=path.name, file=file, frontier_k=k, method=name,
                          gates=int(expected.size), ordinary_done_gates=int(ordinary_done.sum()),
                          independent_certified_gates=int(base_done.sum()),
                          raw_pattern_certified_gates=int(raw_done.sum()),
                          intersection_certified_gates=int(done.sum()),
                          additional_certificates_vs_independent=int((done & ~base_done).sum()),
                          endpoint_products_before_P4_reuse=raw_endpoint,
                          certificate_comparisons=int((~ordinary_done).sum()) * 2,
                          current_pattern_histogram_nonzero_fields=int((hist[name] > 0).sum()),
                          bound_violations=0, certificate_errors=0, **cost,
                          **cancellation_counts(done, a != 0, remaining_products, words, weight, k))
            records.append(record)
        previous = k
        print(path.name, k, [(r['method'], r['additional_certificates_vs_independent']) for r in records[-4:]], flush=True)
    if not np.array_equal(partial, y):
        raise RuntimeError('Single source pass did not reconstruct complete Yi')
    return records


def main():
    params = dict(np.load(HERE / 'integer_deployment/common3_diagonal_34.npz'))
    all_bases = bases(params['temporal_int16'].astype(np.int64))
    result = dict(
        scope='four captured frames, each 64 native P4 groups/all H96, fixed source-frontier states, no feedback execution or cycles',
        hypothesis='retain exact coincidence of temporal source bits in uncertainty about unfinished common and private producers',
        identity='same complete common3-diagonal34 integer student; no retraining, gamma or change to emitted theta*g',
        bound='n_q source words of projected pattern q imply c_t(q)*sum of an unknown W subset; twelve-bin extreme-weight sums give a conservative range. Different classes may reuse extreme weights in the bound, hence this is not a tight optimization.',
        strong_control='independent Yi bounds with source/W-product-empty knowledge and the SAME static suffix-W tables; report candidate alone and its intersection with independent bounds',
        frontiers=list(FRONTIERS),
        table='each fixed K frontier has 12*96*2*24=6912 bytes of W bounds; first four nonterminal frontiers total27648B. These are compiled from known suffix W, not paid by a free future-weight scan at runtime.',
        cost_limits=['histogram construction scans source words before execution; counters, updates, table ports and generic A propagation are not free',
                     'P4 reuse is a value-use count, not a banked schedule; intersection pays BOTH bound computations plus max/min',
                     'comparison and endpoint counts here cover independent checkpoint queries, not live-gated execution across checkpoints',
                     'future source/W products are used only to evaluate cancelled-work opportunities, not as predictor values; ordinary product-empty metadata is granted to all controls',
                     'all methods use the same student, actual W, source order and exact full gates; no network/GPU rerun is needed for these exact function-preserving identities'],
        representation={}, records=[])
    for name, (lookup, coeff) in all_bases.items():
        result['representation'][name] = dict(
            count_fields_per_spatial_position=int(lookup.shape[1]),
            count_payload_bytes_per_P4=int(lookup.shape[1] * 4 * 10 // 8),
            nonzero_coefficient_entries=int(np.count_nonzero(coeff)),
            coefficient_min=int(coeff.min()), coefficient_max=int(coeff.max()),
            initial_source_histogram_increments_four_frames=sum(
                int(lookup[np.load(path)['source_gate_words']].sum())
                for path in sorted((HERE / 'integer_valid10').glob('capture_*.npz'))))
    for path in sorted((HERE / 'integer_valid10').glob('capture_*.npz')):
        result['records'].extend(run_capture(path, params, all_bases))
    sums = {}
    excluded = {'capture', 'file', 'frontier_k', 'method'}
    for k in FRONTIERS:
        sums[str(k)] = {}
        for name in all_bases:
            selected = [r for r in result['records'] if r['frontier_k'] == k and r['method'] == name]
            sums[str(k)][name] = {key: sum(r[key] for r in selected) for key in selected[0] if key not in excluded}
    result['totals_by_frontier'] = sums
    (HERE / 'correlated_bound_probe.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
