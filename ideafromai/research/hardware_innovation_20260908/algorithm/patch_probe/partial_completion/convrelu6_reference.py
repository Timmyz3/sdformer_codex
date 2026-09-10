"""ConvReLU++ E=6 bound core, adapted to a fixed spatial parent and theta*g.

This reproduces the signed correction and residual-norm bound, not its hash
reference selector or mobile kernel. All rows use the same captured integer
student. No child exact output selects a reference or determines a certificate.
"""
import json
from pathlib import Path

import numpy as np

from source_count_bound_probe import capture_geometry

HERE = Path(__file__).resolve().parent
E = 6


def vector_uses(words, weight, wanted):
    # wanted: G,S,H,P. One W[k,H8] may serve all live S/P for this phase.
    total = 0
    for hg in range(12):
        hit = np.zeros((64, 864, 8), dtype=bool)
        for s in range(10):
            source = ((words & (1 << s)) != 0)
            hit |= (source[:, :, None] & wanted[:, s, hg * 8:hg * 8 + 8, None].transpose(0, 2, 1, 3)).any(-1)
        hit &= (weight[hg * 8:hg * 8 + 8].T != 0)[None]
        total += int(hit.any(-1).sum())
    return total


def compile_top(a, weight):
    indices, values = np.zeros((10, 96, E), dtype=np.int64), np.zeros((10, 96, E), dtype=np.int64)
    for t in range(10):
        for h in range(96):
            fused = (a[t, :, None] * weight[h, None]).ravel()
            idx = np.lexsort((np.arange(fused.size), -np.abs(fused)))[:E]
            indices[t, h], values[t, h] = idx, fused[idx]
    norm2 = (a * a).sum(1)[:, None] * (weight * weight).sum(1)[None]
    if int(norm2.max()) * 864 * 4 >= np.iinfo(np.int64).max:
        raise RuntimeError('Squared-bound reference needs more than INT64')
    return indices // 864, indices % 864, values, norm2


def main():
    params = dict(np.load(HERE / 'integer_deployment/common3_diagonal_34.npz'))
    weight = params['weight_int8'].reshape(96, 864).astype(np.int64)
    a = params['temporal_int16'].astype(np.int64)
    support = a != 0
    tau = params['threshold_positive'][params['full_entry']]
    top_s, top_k, top_v, norm2 = compile_top(a, weight)
    top_sq = top_v * top_v
    norm_bits = int(norm2.max()).bit_length()
    result = dict(
        source='ConvReLU++ MobiSys2023 §3.3 Equations5-7, E=6, https://yuanchun-li.github.io/static/files/MobiSys23_ConvReLU%2B%2B.pdf',
        scope='same four captures, fixed native P4 parent p0 and children p1..3, all H96/T10; signed-correction bound core, not full reference selection or runtime',
        controls='same exact integer common3 student; ordinary source/W-empty row decisions and exact reference support matches granted; all parents paid in full',
        arithmetic='effective row is A[t,:] tensor W[h,:]; delta g is -1/0/+1; norm(delta)^2 is source XOR count on row support; negative top-six products tighten upper bound, positive products tighten lower bound',
        costs=dict(E=E, coefficient_abs_max=int(np.abs(top_v).max()),
                   norm_squared_max=int(norm2.max()),
                   top_index_bits=14, top_coefficient_bits=24, norm_squared_bits=norm_bits,
                   top_index_bytes=10 * 96 * E * 14 // 8,
                   top_coefficient_bytes=10 * 96 * E * 24 // 8,
                   base_and_top_squared_norm_packed_bytes=(1 + E) * 10 * 96 * norm_bits // 8,
                   base_and_top_squared_norm_uint64_bytes=(1 + E) * 10 * 96 * 8,
                   alternative_all_64_subset_norm_packed_bytes=(1 << E) * 10 * 96 * norm_bits // 8,
                   parent_source_bytes=864 * 10 // 8,
                   parent_Ui48_bytes=10 * 96 * 48 // 8),
        exclusions=['no hash reference selection, no cache chain, no stream compaction implementation',
                    'squared predicates are exact reference arithmetic, not free hardware multipliers',
                    'selected top source bits and coefficients require gathers/read ports; subset norm can use a table OR masked sums, not both for free',
                    'logical W uses are not SRAM reads or cycles; parent-first phase may destroy P4 W reuse',
                    'this bound changes no gates or numerical student; four captures are not a full-layer schedule'],
        frames=[])
    for path in sorted((HERE / 'integer_valid10').glob('capture_*.npz')):
        with np.load(path) as sample:
            words = sample['source_gate_words'].astype(np.int64)
            y, products, _, _, _, _ = capture_geometry(sample, weight)
            file = str(sample['file'])
        full_u = np.einsum('ts,gshp->gthp', a, y, dtype=np.int64)
        full_gate = full_u >= tau[None, :, :, None]
        parent = full_u[:, :, :, 0]
        parent_top_bits = (words[:, top_k, 0] >> top_s[None]) & 1
        ordinary = np.einsum('ts,gshp->gthp', support.astype(np.int16),
                             (products > 0).astype(np.int16), dtype=np.int16) == 0
        for p in (1, 2, 3):
            changed = words[:, :, p] ^ words[:, :, 0]
            delta_products = np.stack([
                ((changed & (1 << s)) != 0).astype(np.int16)
                @ (weight != 0).T.astype(np.int16) for s in range(10)], axis=1)
            ordinary[:, :, :, p] |= np.einsum(
                'ts,gsh->gth', support.astype(np.int16),
                (delta_products > 0).astype(np.int16), dtype=np.int16) == 0
        all_axes = {name: ordinary.copy() for name in ('L2_upper_only', 'L2_two_sided', 'E6_upper_only', 'E6_two_sided')}
        checks = 0
        for p in (1, 2, 3):
            delta_word = words[:, :, p] ^ words[:, :, 0]
            per_s = np.array([((delta_word & (1 << s)) != 0).sum(1) for s in range(10)]).T
            n = per_s @ support.T.astype(np.int64)
            child_top = (words[:, top_k, p] >> top_s[None]) & 1
            v = (child_top - parent_top_bits) * top_v[None]
            selected_neg, selected_pos = v <= 0, v >= 0
            correction_neg = np.where(selected_neg, v, 0).sum(-1)
            correction_pos = np.where(selected_pos, v, 0).sum(-1)
            rem_upper = norm2[None] - (selected_neg * top_sq[None]).sum(-1)
            rem_lower = norm2[None] - (selected_pos * top_sq[None]).sum(-1)
            negative_margin = tau[None] - 1 - parent
            positive_margin = parent - tau[None]
            upper_budget = negative_margin - correction_neg
            lower_budget = positive_margin + correction_pos
            squared = n[:, :, None] * norm2[None]
            l2negative = (negative_margin >= 0) & (negative_margin * negative_margin >= squared)
            l2positive = (positive_margin >= 0) & (positive_margin * positive_margin >= squared)
            negative = (upper_budget >= 0) & (upper_budget * upper_budget >= n[:, :, None] * rem_upper)
            positive = (lower_budget >= 0) & (lower_budget * lower_budget >= n[:, :, None] * rem_lower)
            if np.any(negative & full_gate[:, :, :, p]) or np.any(positive & ~full_gate[:, :, :, p]):
                raise RuntimeError('E6 certificate differs from the exact child function')
            if np.any(l2negative & ~negative) or np.any(l2positive & ~positive):
                raise RuntimeError('The E6 bound unexpectedly lost an L2 certificate')
            exact_match = n[:, :, None] == 0
            for name, cert in [('L2_upper_only', l2negative), ('L2_two_sided', l2negative | l2positive),
                               ('E6_upper_only', negative), ('E6_two_sided', negative | positive)]:
                all_axes[name][:, :, :, p] |= cert | exact_match
            checks += 64 * 10 * 96
        all_axes['ordinary_equal_empty'] = ordinary.copy()
        totals = dict(capture=path.name, file=file, full_Conv_products=int(products.sum()),
                      parent_Conv_products=int(products[:, :, :, 0].sum()),
                      child_gate_queries=checks, checked_complete_T10_gates=int(full_gate.size),
                      top_delta_coefficient_products_or_sign_selects=checks * E,
                      both_sides_max_subset_sum_terms=checks * E * 2,
                      full_gate_mismatches=0, axes={})
        parent_wanted = np.zeros_like(products, dtype=bool)
        parent_wanted[:, :, :, 0] = True
        parent_w = vector_uses(words, weight, parent_wanted)
        full_w = vector_uses(words, weight, np.ones_like(products, dtype=bool))
        for name, done in all_axes.items():
            done[:, :, :, 0] = False  # the exact parent is always produced
            wanted = np.einsum('ts,gthp->gshp', support.astype(np.int16),
                               (~done).astype(np.int16), dtype=np.int16) > 0
            needed = products * wanted
            fallback = wanted.copy()
            fallback[:, :, :, 0] = False
            fallback_w = vector_uses(words, weight, fallback)
            totals['axes'][name] = dict(certified_child_gates=int(done[:, :, :, 1:].sum()),
                extra_child_certificates_over_ordinary=int((done & ~ordinary)[:, :, :, 1:].sum()),
                resulting_Conv_products=int(needed.sum()),
                Conv_product_reduction=1 - int(needed.sum()) / int(products.sum()),
                full_single_phase_W_H8_uses=full_w, parent_W_H8_uses=parent_w,
                fallback_W_H8_uses=fallback_w, parent_plus_fallback_W_H8_uses=parent_w + fallback_w,
                ideal_all_phase_W_union_uses=vector_uses(words, weight, wanted))
        ordinary_products = totals['axes']['ordinary_equal_empty']['resulting_Conv_products']
        for axis in totals['axes'].values():
            axis['extra_cancelled_Conv_products_over_ordinary'] = ordinary_products - axis['resulting_Conv_products']
        result['frames'].append(totals)
        print(path.name, {name: (v['resulting_Conv_products'], v['parent_plus_fallback_W_H8_uses']) for name, v in totals['axes'].items()}, flush=True)
    result['totals'] = {key: sum(f[key] for f in result['frames']) for key in result['frames'][0]
                        if key not in ('capture', 'file', 'axes')}
    result['totals']['axes'] = {}
    for name in result['frames'][0]['axes']:
        axis = {key: sum(f['axes'][name][key] for f in result['frames'])
                for key in result['frames'][0]['axes'][name] if key != 'Conv_product_reduction'}
        axis['Conv_product_reduction'] = 1 - axis['resulting_Conv_products'] / result['totals']['full_Conv_products']
        result['totals']['axes'][name] = axis
    (HERE / 'convrelu6_reference.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
