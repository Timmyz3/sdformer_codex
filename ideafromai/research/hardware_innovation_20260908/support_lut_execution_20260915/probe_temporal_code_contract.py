#!/opt/anaconda3/bin/python3.12
"""CPU-only exact temporal/code contraction and operation-capacity bounds.

Reads frozen real post-projection sources; never changes RTL or model data.
Output is an optimistic capacity analysis, not an RTL schedule or AEE result.
"""
from pathlib import Path
import csv
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
P, T, G, K, H = 32, 10, 6, 16, 96


def hist(a):
    values, counts = np.unique(a, return_counts=True)
    return {str(int(v)): int(n) for v, n in zip(values, counts)}


def gate(u, tau, positive, constant, constant_gate):
    compared = np.where(positive[None, None, :], u >= tau[None], u <= tau[None])
    return np.where(constant[None, None, :], constant_gate[None], compared).astype(np.uint8)


def decode(source, dictionary):
    # Actual source bits -> group word -> exact dictionary match, not supplied IDs.
    dwords = (dictionary.astype(np.int64) * (1 << np.arange(16))).sum(-1)
    lookup = [{int(word): k for k, word in enumerate(group)} for group in dwords]
    assert all(len(x) == K for x in lookup)
    assert all(x[0] == 0 for x in lookup)
    words = (source.reshape(P, T, G, 16).astype(np.int64) * (1 << np.arange(16))).sum(-1)
    codes = np.zeros((P, T, G), np.int64)
    for p in range(P):
        for s in range(T):
            for g in range(G):
                assert int(words[p, s, g]) in lookup[g], 'Real source must exactly match D'
                codes[p, s, g] = lookup[g][int(words[p, s, g])]
    return codes


def contract(codes, a):
    membership = codes[:, :, :, None] == np.arange(K)[None, None, None, :]
    membership[:, :, :, 0] = False  # L[g,0,:]=0: no coefficient work for zero code.
    c = np.einsum('psgk,ts->pgkt', membership.astype(np.int64), a)
    term_count = np.einsum('psgk,ts->pgkt', membership.astype(np.int64), (a != 0).astype(np.int64))
    # Independent incremental construction checks the actual reduction ordering.
    second = np.zeros_like(c)
    initialized = np.zeros_like(c, dtype=bool)
    adds = 0
    prefix_lo = prefix_hi = 0
    for p in range(P):
        for s in range(T):
            for g in range(G):
                k = int(codes[p, s, g])
                if k == 0:
                    continue
                for t in range(T):
                    if a[t, s] == 0:
                        continue
                    if initialized[p, g, k, t]:
                        adds += 1
                    else:
                        initialized[p, g, k, t] = True
                    second[p, g, k, t] += a[t, s]
                    prefix_lo = min(prefix_lo, int(second[p, g, k, t]))
                    prefix_hi = max(prefix_hi, int(second[p, g, k, t]))
    assert np.array_equal(c, second)
    assert adds == int(np.maximum(term_count - 1, 0).sum())
    distinct = membership.any(axis=1).sum(axis=-1)  # [p,g], including all six groups.
    stats = {
        'nonzero_code_occurrences': int(np.count_nonzero(codes)),
        'zero_code_occurrences': int((codes == 0).sum()),
        'distinct_nonzero_codes_per_p_g': distinct.tolist(),
        'distinct_nonzero_codes_histogram': hist(distinct),
        'distinct_nonzero_codes_total': int(distinct.sum()),
        'per_group_distinct_code_totals': distinct.sum(axis=0).tolist(),
        'coefficient_input_nonzero_terms': int(term_count.sum()),
        'coefficient_buckets_with_nonzero_inputs': int(np.count_nonzero(term_count)),
        'coefficient_nonzero_after_sum': int(np.count_nonzero(c)),
        'coefficient_exact_cancellations': int(np.count_nonzero(term_count) - np.count_nonzero(c)),
        'coefficient_scalar_reduction_adds': adds,
        'coefficient_first_assignments': int(np.count_nonzero(term_count)),
        'coefficient_logical_writes_incremental': int(term_count.sum()),
        'coefficient_logical_read_modify_writes_after_first': adds,
        'coefficient_range': [int(c.min()), int(c.max())],
        'coefficient_incremental_prefix_range': [prefix_lo, prefix_hi],
        'per_group_nonzero_aggregated_coefficients': np.count_nonzero(c, axis=(0, 2, 3)).tolist(),
        'per_group_scalar_reduction_adds': np.maximum(term_count - 1, 0).sum(axis=(0, 2, 3)).tolist(),
    }
    return c, stats


def main():
    archive = np.load(HERE / 'cases.npz', allow_pickle=False)
    a = archive['A'].astype(np.int64)
    dictionary = archive['D'].astype(np.int64)
    assert a.shape == (T, T) and dictionary.shape == (G, K, 16)
    assert np.all(dictionary[:, 0] == 0)
    real_indices = np.flatnonzero(archive['is_real'])
    assert len(real_indices) == 32
    rtl = list(csv.DictReader((HERE / 'rtl_cycles.csv').open()))
    reference = {(r['case'], int(r['mode']), int(r['bp'])): r for r in rtl}
    sources, source_arrays, cases = {}, {}, []
    sums = {key: 0 for key in (
        'distinct_nonzero_codes', 'coefficient_input_nonzero_terms', 'coefficient_scalar_reduction_adds',
        'coefficient_exact_cancellations', 'candidate_vector_mac', 'candidate_nonzero_scalar_products',
        'candidate_MAC_capacity_floor', 'candidate_shared_ALU_floor_current_accumulation',
        'baseline_vector_fc_updates', 'baseline_vector_psn_mac', 'baseline_ALU_operations_floor',
        'baseline_fc_cycles', 'baseline_psn_cycles', 'baseline_total_cycles',
        'baseline_direct_total_cycles', 'baseline_total_cycles_bp', 'U_values_checked', 'gate_values_checked')}
    product_min = product_max = prefix_min = prefix_max = 0
    for index in real_indices:
        name = str(archive['case_name'][index])
        source_id = f"v{int(archive['validation_index'][index])}_tile{int(archive['tile_index'][index])}"
        s = archive['S'][index].astype(np.int64)
        w = archive['W'][index].astype(np.int64)
        codes = decode(s, dictionary)
        c, stats = contract(codes, a)
        if source_id in sources:
            assert sources[source_id]['statistics'] == stats
            assert np.array_equal(source_arrays[source_id], s)
        else:
            source_arrays[source_id] = s.copy()
            sources[source_id] = {'frame_file': str(archive['frame_file'][index]),
                'validation_index': int(archive['validation_index'][index]),
                'tile_index': int(archive['tile_index'][index]), 'statistics': stats}
        table = np.einsum('gkc,gch->gkh', dictionary, w.reshape(G, 16, H))
        assert np.all(table[:, 0] == 0)
        response_live = np.any(table != 0, axis=-1)
        assert response_live[:, 1:].all()  # No free all-zero response-row shortcut in these real functions.
        u = np.zeros((P, T, H), dtype=np.int64)
        for g in range(G):
            for k in range(1, K):
                product = c[:, g, k, :, None] * table[g, k, None, None, :]
                product_min = min(product_min, int(product.min()))
                product_max = max(product_max, int(product.max()))
                u += product
                prefix_min = min(prefix_min, int(u.min()))
                prefix_max = max(prefix_max, int(u.max()))
        y_dense = s @ w
        u_dense = np.einsum('ts,psh->pth', a, y_dense)
        assert np.array_equal(y_dense, archive['Y'][index])
        assert np.array_equal(u, u_dense) and np.array_equal(u, archive['U'][index])
        out = gate(u, archive['tau'][index], archive['positive_gain'][index],
                   archive['constant_channels'][index], archive['constant_gate'][index])
        assert np.array_equal(out, archive['gold'][index])
        vector_mac = int(np.count_nonzero(c * response_live[None, :, :, None]))
        product_nonzero = int((np.count_nonzero(c, axis=(0, 3)) * np.count_nonzero(table, axis=-1)).sum())
        additions = stats['coefficient_scalar_reduction_adds']
        base = reference[(name, 2, 0)]
        live_y_rows = np.any(s != 0, axis=-1)
        expected_mac = int((live_y_rows * np.count_nonzero(a, axis=0)[None, :]).sum())
        assert expected_mac == int(base['mac'])
        assert stats['nonzero_code_occurrences'] == int(base['updates'])
        # Fixed H96 wiring: each surviving C times a nonzero response row is one issue.
        # ALU bound grants free first C assignment and arbitrarily packed scalar reductions;
        # it otherwise keeps current RTL's U += product use of the shared add_y.
        row = dict(case=name, source=source_id, hblock=int(archive['hblock'][index]),
            distinct_nonzero_codes=stats['distinct_nonzero_codes_total'],
            coefficient_input_nonzero_terms=stats['coefficient_input_nonzero_terms'],
            coefficient_scalar_reduction_adds=additions,
            coefficient_exact_cancellations=stats['coefficient_exact_cancellations'],
            candidate_vector_mac=vector_mac, candidate_nonzero_scalar_products=product_nonzero,
            candidate_MAC_capacity_floor=vector_mac,
            candidate_shared_ALU_floor_current_accumulation=vector_mac + math.ceil(additions / H),
            baseline_vector_fc_updates=int(base['updates']), baseline_vector_psn_mac=int(base['mac']),
            baseline_ALU_operations_floor=int(base['updates']) + int(base['mac']),
            baseline_fc_cycles=int(base['fc_cycles']), baseline_psn_cycles=int(base['psn_cycles']),
            baseline_total_cycles=int(base['cycles']),
            baseline_direct_total_cycles=int(reference[(name, 0, 0)]['cycles']),
            baseline_total_cycles_bp=int(reference[(name, 2, 1)]['cycles']),
            all_nonzero_response_rows=int(response_live[:, 1:].sum()),
            response_lane_zeros=int((table[:, 1:] == 0).sum()),
            U_values_checked=int(u.size), gate_values_checked=int(out.size))
        cases.append(row)
        for key in sums:
            sums[key] += row[key]
    assert len(sources) == 8
    assert sums['baseline_vector_psn_mac'] == 83400
    assert sums['baseline_vector_fc_updates'] == 30128
    assert sums['baseline_total_cycles'] == 184184
    assert sums['baseline_psn_cycles'] == 118344
    assert sums['baseline_fc_cycles'] == 30352
    subset_min = np.minimum(a, 0).sum(axis=1)
    subset_max = np.maximum(a, 0).sum(axis=1)
    assert subset_min.min() >= -(1 << 15) and subset_max.max() < (1 << 15)
    assert prefix_min >= -(1 << 47) and prefix_max < (1 << 47)
    unique_hist = np.asarray([s['statistics']['distinct_nonzero_codes_per_p_g'] for s in sources.values()])
    output = {
        'status': 'PASS', 'scope': 'CPU exact arithmetic and optimistic operation-capacity floors; no RTL/quality experiment',
        'input': str(HERE / 'cases.npz'), 'rtl_reference': str(HERE / 'rtl_cycles.csv'),
        'real_cases': 32, 'independent_source_tiles': 8, 'H96_blocks_per_source': 4,
        'shape': {'P': P, 'T': T, 'groups': G, 'codes_per_group': K, 'H': H},
        'actual_resources': '96 multipliers and the same 96 lanes of 48-bit add_y; 48 is width, not lane count',
        'A': {'matrix': a.tolist(), 'nonzero': int(np.count_nonzero(a)), 'zero': int((a == 0).sum()),
            'nonzero_by_input_s': np.count_nonzero(a, axis=0).tolist(), 'noncausal': bool(np.any(np.triu(a, 1) != 0)),
            'all_subset_coefficient_min_by_t': subset_min.tolist(),
            'all_subset_coefficient_max_by_t': subset_max.tolist(), 'all_subset_C_fits_signed16': True},
        'numerics': {'product_observed_range': [product_min, product_max],
            'ordered_U_prefix_observed_range': [prefix_min, prefix_max], 'intermediate_rounding': 'none',
            'U_mismatches': 0, 'gate_mismatches': 0},
        'distinct_nonzero_codes_histogram_unique_sources': hist(unique_hist),
        'totals_all_32_commands': sums,
        'ratios': {
            'candidate_MAC_vs_baseline_PSN_MAC': sums['candidate_vector_mac'] / sums['baseline_vector_psn_mac'],
            'candidate_MAC_floor_vs_baseline_complete_cycles': sums['candidate_vector_mac'] / sums['baseline_total_cycles'],
            'candidate_shared_ALU_floor_vs_baseline_ALU_floor': sums['candidate_shared_ALU_floor_current_accumulation'] / sums['baseline_ALU_operations_floor'],
            'mean_distinct_nonzero_codes_per_p_g': float(unique_hist.mean()),
            'code_instance_reduction_fraction': 1 - sums['distinct_nonzero_codes'] / sums['baseline_vector_fc_updates']},
        'H4_amortization_not_implemented': {
            'scalar_reduction_adds_if_reused_for_all_four_Hblocks': sums['coefficient_scalar_reduction_adds'] // 4,
            'vector_MAC_count_unchanged': sums['candidate_vector_mac']},
        'potential_C_state': {'one_p_excluding_zero_int16_bytes': G*(K-1)*T*2,
            'all_P_excluding_zero_int16_bytes': P*G*(K-1)*T*2,
            'one_p_time_membership_mask_bits': G*(K-1)*T,
            'not_allocated_or_port_scheduled': True},
        'sources': sources, 'cases': cases,
        'limitations': [
            'Inputs are real post-projection g_prime. Upstream nearest-code projection is not implemented here.',
            'Scalar coefficient reductions charge max(nonzero inputs - 1, 0), granting free first assignment arithmetic.',
            'Capacity floors exclude code matching, memory ports, coefficient table reads/build, state clear/writes, pipeline dependencies, gates, output, config and backpressure.',
            'The stronger MAC-only floor also grants free first-product U initialization and free coefficient aggregation; it is still slower than current complete no-BP service in aggregate.',
            'Per-lane zero product packing would need new routing; counts are provided but no such schedule or bandwidth is assumed.',
            'No Y data consumer exists inside this measured leaf except debug/PSN; a wider graph needing Y would add an obligation.',
            'Ordinary contraction reordering is not a new temporal model and does not invalidate other code/temporal layouts.'
        ]}
    target = HERE / 'probe_temporal_code_contract.json'
    target.write_text(json.dumps(output, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(json.dumps({'status': 'PASS', 'result': str(target), 'A_nonzero': int(np.count_nonzero(a)),
        'U_values': sums['U_values_checked'], 'gate_values': sums['gate_values_checked'],
        'candidate_MAC_floor': sums['candidate_MAC_capacity_floor'],
        'baseline_complete_cycles': sums['baseline_total_cycles'],
        'scalar_reduction_adds': sums['coefficient_scalar_reduction_adds']}))


if __name__ == '__main__':
    main()
