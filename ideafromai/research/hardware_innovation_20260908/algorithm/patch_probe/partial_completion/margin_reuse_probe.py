"""Fixed native-p0 exact decision certificates; four captured integer samples.

Parents are never chosen using a child's answer. Child values are numerical
labels/fallback operands only. This measures logical work, not SRAM service.
"""
import json
from pathlib import Path

import numpy as np

from source_count_bound_probe import N_UP, compile_bounds

HERE = Path(__file__).resolve().parent
BITS = 1 << np.arange(10)
POPCOUNT = np.array([v.bit_count() for v in range(16)], dtype=np.int16)
PAPER = 'https://proceedings.mlsys.org/paper_files/paper/2021/file/b9799a12d683d136cc817f94b73a8938-Paper.pdf'


def down_power2(x):
    return np.fromiter((0 if int(v) == 0 else 1 << (int(v).bit_length()-1)
                        for v in x.flat), dtype=np.int64, count=x.size).reshape(x.shape)


def up_shift(x):
    return np.fromiter(((int(v)-1).bit_length() if int(v) else 0 for v in x.flat),
                       dtype=np.int64, count=x.size).reshape(x.shape)


def encode_radius(budget, bits):
    # Code 0 means radius 0, code e+1 means radius 2**e.
    code = np.fromiter((int(v).bit_length() for v in budget.flat),
                       dtype=np.int64, count=budget.size).reshape(budget.shape)
    decoded = np.where(code == 0, 0, 1 << np.maximum(code-1, 0))
    if int(code.max()) >= (1 << bits) or np.any(decoded != down_power2(budget)):
        raise RuntimeError('Radius encoding exceeds its width or changes rounding.')
    return code, decoded


def required_radius_code(count):
    # H=0 accepts radius 0; otherwise ceil(log2(H))+1 is the smallest code.
    return np.where(count == 0, 0, up_shift(count)+1)


def need_y(rows, support):
    return np.stack([rows[:, support[:, s]].any(1) for s in range(10)], axis=1)


def work(source, wanted, weight):
    """One shared T/P weight scan; return its actual (G,K,H8) request set."""
    pbits = 1 << np.arange(source.shape[-1])
    sw = (source*pbits).sum(-1).astype(np.uint8)
    wanted_words = (wanted*pbits).sum(-1).astype(np.uint8)
    request = np.zeros((64, 864, 12), dtype=bool)
    counts = dict(products=0, W_H8_uses=0, W_scalar_uses=0, Conv_k_t_H8_visits=0)
    for hg in range(12):
        wlive = (weight[hg*8:hg*8+8].T != 0)[None, None]
        intersection = sw[:, :, :, None] & wanted_words[:, :, None, hg*8:hg*8+8]
        active = (intersection != 0) & wlive
        counts['products'] += int((POPCOUNT[intersection]*wlive).sum())
        request[:, :, hg] = active.any(axis=(1, 3))
        counts['W_scalar_uses'] += int(active.any(1).sum())
        counts['Conv_k_t_H8_visits'] += int(active.any(3).sum())
    counts['W_H8_uses'] = int(request.sum())
    need_words = np.bitwise_or.reduce(wanted_words, axis=2)
    counts['needed_nonempty_source_Pword_descriptors'] = int(
        ((sw & need_words[:, :, None]) != 0).any(1).sum())
    return counts, request


def psn_products(values, rows, support):
    return sum(int((rows[:, support[:, s]] & (values[:, s, None] != 0)).sum())
               for s in range(10))


def strict_delta(data, lo, hi):
    addbin = np.searchsorted(N_UP, data['nadd'])
    rembin = np.searchsorted(N_UP, data['nremove'])
    low = (lo[addbin]-hi[rembin]).transpose(0, 1, 3, 2)
    high = (hi[addbin]-lo[rembin]).transpose(0, 1, 3, 2)
    low = np.where(data['delta_products'] == 0, 0, low)
    high = np.where(data['delta_products'] == 0, 0, high)
    a = data['a']; ap, an = np.maximum(a, 0), np.minimum(a, 0)
    dlo = np.einsum('ts,gshp->gthp', ap, low, dtype=np.int64) + np.einsum('ts,gshp->gthp', an, high, dtype=np.int64)
    dhi = np.einsum('ts,gshp->gthp', ap, high, dtype=np.int64) + np.einsum('ts,gshp->gthp', an, low, dtype=np.int64)
    ulo = data['parent_u'][..., None] + dlo
    uhi = data['parent_u'][..., None] + dhi
    positive = ulo >= data['tau'][None, :, :, None]
    negative = uhi < data['tau'][None, :, :, None]
    select = ~data['ordinary_done']
    needed = need_y(select, data['support']) & (data['delta_products'] > 0)
    fees = dict(delta_Y_endpoint_subtractions=2*int(needed.sum()),
                interval_gate_comparisons=2*int(select.sum()),
                A_endpoint_products_after_zero_equal_skip=0, A_endpoint_accumulation_additions=0,
                parent_interval_endpoint_additions_after_zero_bypass=int(
                    ((dlo != 0)*select*(data['parent_u'][..., None] != 0)).sum()) + int(
                    ((dhi != 0)*select*(data['parent_u'][..., None] != 0)).sum()),
                bound_table_pairs_before_reuse=int((needed*(data['nadd'][:, :, None] > 0)).sum())+int((needed*(data['nremove'][:, :, None] > 0)).sum()),
                bound_table_pairs_shared_time_children=0, bound_table_H8_bin_uses=0)
    for b in range(1, len(N_UP)):
        need = needed & ((addbin == b) | (rembin == b))[:, :, None]
        fees['bound_table_pairs_shared_time_children'] += int(need.any(axis=(1, 3)).sum())
        fees['bound_table_H8_bin_uses'] += int(need.reshape(64, 10, 12, 8, 3).any(axis=(1, 3, 4)).sum())
    for t in range(10):
        s = np.flatnonzero(data['support'][t]); coeff = a[t, s][None, :, None, None]
        l = np.where(coeff > 0, low[:, s], high[:, s])
        h = np.where(coeff > 0, high[:, s], low[:, s])
        lnz, hnz = l != 0, h != 0
        fees['A_endpoint_products_after_zero_equal_skip'] += int(
            ((lnz.astype(np.int16)+(hnz & (h != l)))*select[:, t, None]).sum())
        exact = (l == h).all(1)
        fees['A_endpoint_accumulation_additions'] += int(
            ((np.maximum(lnz.sum(1)-1, 0)+np.maximum(hnz.sum(1)-1, 0)*~exact)*select[:, t]).sum())
    if np.any((data['dy'] < low) | (data['dy'] > high)) or np.any((data['child_u'] < ulo) | (data['child_u'] > uhi)):
        raise RuntimeError('Strict source-difference bounds are not conservative.')
    return positive | negative, positive, fees


def probe_frame(sample, params, weight, lo, hi):
    words = sample['source_gate_words'].astype(np.int64)
    source = (words[:, None] & BITS[None, :, None, None]) != 0
    parent, child = source[..., :1], source[..., 1:]
    added, removed = child & ~parent, parent & ~child
    changed = added | removed
    nadd, nremove = added.sum(2), removed.sum(2)
    nflip = nadd + nremove
    wnonzero = (weight != 0).astype(np.int16)
    products = np.matmul(source.astype(np.int16).transpose(0, 1, 3, 2), wnonzero.T).transpose(0, 1, 3, 2)
    dproducts = np.matmul(changed.astype(np.int16).transpose(0, 1, 3, 2), wnonzero.T).transpose(0, 1, 3, 2)
    # Complete numeric references are calculated independently from the source
    # and actual W. They never determine a certificate or a parent selection.
    y = np.einsum('hk,gskp->gshp', weight, source, dtype=np.int64)
    recorded_y = sample['Yi'].reshape(64, 12, 10, 8, 4).transpose(0, 2, 1, 3, 4).reshape(64, 10, 96, 4)
    if np.any(y != recorded_y):
        raise RuntimeError('Actual source/W reconstruction differs from captured Yi.')
    a = params['temporal_int16'].astype(np.int64); support = a != 0
    tau = params['threshold_positive'][params['full_entry']].astype(np.int64)
    if not np.all(params['positive_gain'] == 1):
        raise ValueError('The captured integer student uses positive compiled gains.')
    u = np.einsum('ts,gshp->gthp', a, y, dtype=np.int64)
    full_gate = u >= tau[None, :, :, None]
    recorded = sample['gate_common3_diagonal_34_exact'].reshape(64, 12, 10, 8, 4)
    recorded = recorded.transpose(0, 2, 1, 3, 4).reshape(64, 10, 96, 4)
    if np.any(full_gate != recorded):
        raise RuntimeError('Full integer gates differ from the captured student.')
    ordinary_empty = np.einsum('ts,gshp->gthp', support.astype(np.int16),
                               (products[..., 1:] != 0).astype(np.int16)) == 0
    ordinary_equal = np.einsum('ts,gshp->gthp', support.astype(np.int16),
                               (dproducts != 0).astype(np.int16)) == 0
    ordinary_done = ordinary_empty | ordinary_equal
    parent_u, parent_gate = u[..., 0], full_gate[..., 0]
    ordinary_answer = np.where(ordinary_empty, 0 >= tau[None, :, :, None], parent_gate[..., None])
    margin = np.where(parent_gate, parent_u-tau[None], tau[None]-1-parent_u)
    if np.any(margin < 0):
        raise RuntimeError('Safe integer margin must be nonnegative.')
    dy = y[..., 1:]-y[..., :1]
    data = dict(a=a, support=support, tau=tau, parent_u=parent_u, child_u=u[..., 1:],
                ordinary_done=ordinary_done, nadd=nadd, nremove=nremove,
                delta_products=dproducts, dy=dy)
    parent_rows = np.ones_like(parent, shape=(64, 10, 96, 1))
    child_rows = np.ones_like(child, shape=(64, 10, 96, 3))
    parent_work, parent_req = work(parent, parent_rows, weight)
    full_work, _ = work(source, np.ones_like(y, dtype=bool), weight)
    ordinary_wanted = need_y(~ordinary_done, support)
    ordinary_child_work, ordinary_req = work(child, ordinary_wanted, weight)
    delta_work, delta_req = work(changed, child_rows, weight)
    # Restricted subset-parent control: legal only when no source is removed
    # in this full-K temporal row. It is not the full per-K Prosperity forest.
    subset = nremove == 0
    subset_source = np.where(subset[:, :, None], added, child)
    subset_work, subset_req = work(subset_source, child_rows, weight)
    all_psn = psn_products(y, np.ones_like(full_gate), support)
    parent_psn = psn_products(y[..., :1], np.ones_like(full_gate[..., :1]), support)
    baseline = dict(full_single_scan=full_work,
        ordinary_equal_or_empty_single_scan=dict(
            products=parent_work['products']+ordinary_child_work['products'],
            W_H8_uses=int((parent_req | ordinary_req).sum()),
            PSN_products=parent_psn+psn_products(y[..., 1:], ~ordinary_done, support)),
        signed_delta_single_scan=dict(products=parent_work['products']+delta_work['products'],
            W_H8_uses=int((parent_req | delta_req).sum()),
            PSN_products=parent_psn+psn_products(dy, child_rows, support),
            parent_U_merge_additions=int(((np.einsum('ts,gshp->gthp', a, dy, dtype=np.int64) != 0)
                                         & (parent_u[..., None] != 0)).sum())),
        fixed_parent_subset_only_single_scan=dict(products=parent_work['products']+subset_work['products'],
            W_H8_uses=int((parent_req | subset_req).sum()),
            parent_Y_merge_additions=int((subset[:, :, None] & (dy != 0) & (y[..., :1] != 0)).sum()),
            PSN_products=all_psn), parent=parent_work,
        parent_PSN_products=parent_psn, full_PSN_products=all_psn,
        full_gate_comparisons=int(full_gate.size), parent_gate_comparisons=int(parent_gate.size))
    if baseline['signed_delta_single_scan']['W_H8_uses'] != full_work['W_H8_uses']:
        raise RuntimeError('One-pass parent-plus-XOR W union must equal the original P4 union.')
    row_hamming = np.einsum('ts,gsp->gtp', support.astype(np.int64), nflip)
    max_a_shift = up_shift(np.abs(a).max(1)); wshift = up_shift(np.abs(weight).max(1))
    acap = np.where(support, 1 << up_shift(np.abs(a)), 0)
    weighted_hamming = np.einsum('ts,gsp->gtp', acap, nflip, dtype=np.int64)
    max_hamming = 864*support.sum(1)
    max_weighted = 864*acap.sum(1)
    exact_radius = np.minimum(margin >> (max_a_shift[:, None]+wshift[None])[None], max_hamming[None, :, None])
    radius_code, log_radius = encode_radius(exact_radius, 4)
    exact_weighted = np.minimum(margin >> wshift[None, None], max_weighted[None, :, None])
    weighted_code, log_weighted = encode_radius(exact_weighted, 5)
    hamming_code = required_radius_code(row_hamming)
    weighted_hamming_code = required_radius_code(weighted_hamming)
    log_accept = hamming_code[:, :, None] <= radius_code[..., None]
    weighted_log_accept = weighted_hamming_code[:, :, None] <= weighted_code[..., None]
    code_errors = int((log_accept != (row_hamming[:, :, None] <= log_radius[..., None])).sum())
    code_errors += int((weighted_log_accept != (weighted_hamming[:, :, None] <= log_weighted[..., None])).sum())
    if code_errors or np.any(log_radius[exact_radius == 0]) or np.any(log_weighted[exact_weighted == 0]):
        raise RuntimeError('Encoded comparison or zero-budget semantics changed.')
    anorm2, wnorm2 = (a*a).sum(1), (weight*weight).sum(1)
    rhs_limit = int(max_hamming.max())*int(anorm2.max())*int(wnorm2.max())
    if rhs_limit >= 2**63 or int(margin.max())**2 >= 2**63:
        raise RuntimeError('L2 square comparison requires wider integer arithmetic.')
    l2rhs = row_hamming[:, :, None]*(anorm2[:, None]*wnorm2[None])[None, :, :, None]
    l2accept = margin[..., None]**2 >= l2rhs
    eligible = ~ordinary_done
    parent_queries = eligible.any(-1)
    terms_per_row = np.einsum('ts,gsp->gtp', support.astype(np.int16), (nflip != 0).astype(np.int16))
    frontend = dict(source_T10_parent_child_XORs=int(words[:, :, 1:].size),
        source_T10_add_remove_masks=2*int(words[:, :, 1:].size),
        n_add_counter_increments=int(nadd.sum()), n_remove_counter_increments=int(nremove.sum()),
        source_T10_fields_observed=int(words.size), source_payload_packed_bytes=int(words.size*10//8),
        hamming_row_additions=int(np.maximum(terms_per_row-1, 0).sum()),
        weighted_frontend_nonzero_shifted_terms=int(terms_per_row.sum()),
        weighted_frontend_additions=int(np.maximum(terms_per_row-1, 0).sum()),
        encoded_comparison_mismatches=code_errors,
        max_unweighted_parent_code=int(radius_code.max()),
        max_weighted_parent_code=int(weighted_code.max()),
        source_comparison_scope='fixed p0 vs p1/2/3, 10-bit words; shared across all H96',
        max_weighted_hamming_domain=int(max_weighted.max()),
        weighted_exact_budget_bits=int(max_weighted.max()).bit_length())
    certs = {}
    certs['strict_delta_12bin'] = strict_delta(data, lo, hi)
    for name, acceptance in (
        ('l2_zero_side', l2accept & ~parent_gate[..., None]),
        ('l2_two_sided', l2accept),
        ('exact_radius', row_hamming[:, :, None] <= exact_radius[..., None]),
        ('log_radius', log_accept),
        ('weighted_exact_allowance', weighted_hamming[:, :, None] <= exact_weighted[..., None]),
        ('weighted_log_margin', weighted_log_accept)):
        sel = eligible & (~parent_gate[..., None] if name == 'l2_zero_side' else True)
        query = sel.any(-1)
        fees = dict(parent_safe_margin_subtractions=int(query.sum()), child_bound_comparisons=int(sel.sum()),
                    parent_budget_right_shifts=0, parent_log_encodings=0,
                    parent_margin_square_products=0, child_count_norm_products=0,
                    child_frontend_log_encodings=0,
                    hamming_row_additions=frontend['hamming_row_additions'])
        if name.startswith('l2'):
            fees.update(parent_margin_square_products=int(query.sum()), child_count_norm_products=int(sel.sum()))
        else:
            fees['parent_budget_right_shifts'] = int(query.sum())
            fees['parent_log_encodings'] = int(query.sum()) if 'log' in name else 0
        if name.startswith('weighted'):
            fees.update(weighted_frontend_nonzero_shifted_terms=frontend['weighted_frontend_nonzero_shifted_terms'],
                        weighted_frontend_additions=frontend['weighted_frontend_additions'])
        if 'log' in name:
            fees['child_frontend_log_encodings'] = int(
                ((weighted_hamming if name.startswith('weighted') else row_hamming) != 0).sum())
        certs[name] = acceptance, np.broadcast_to(parent_gate[..., None], acceptance.shape), fees
    axes = {}
    for name, (certificate, predicted, fees) in certs.items():
        additional = certificate & eligible
        known = ordinary_done | additional
        unresolved = ~known
        needed = need_y(unresolved, support)
        raw_work, raw_req = work(child, needed, weight)
        delta_fallback_work, delta_fallback_req = work(changed, needed, weight)
        fallback_y = np.where(needed, y[..., 1:], 0)
        fallback_u = np.einsum('ts,gshp->gthp', a, fallback_y, dtype=np.int64)
        delta_y = np.where(needed, dy, 0)
        delta_u = parent_u[..., None]+np.einsum('ts,gshp->gthp', a, delta_y, dtype=np.int64)
        value = np.where(additional, predicted, ordinary_answer)
        raw_answer = np.where(unresolved, fallback_u >= tau[None, :, :, None], value)
        delta_answer = np.where(unresolved, delta_u >= tau[None, :, :, None], value)
        wrong = int((raw_answer != full_gate[..., 1:]).sum())+int((delta_answer != full_gate[..., 1:]).sum())
        if wrong:
            raise RuntimeError(f'{name}: certificate/fallback changes {wrong} child gates.')
        ordinary_needed = need_y(~ordinary_done, support)
        metrics = dict(child_gates=int(eligible.size), parent_gates=int(parent_gate.size),
            ordinary_empty_child_gates=int(ordinary_empty.sum()), ordinary_equal_delta_child_gates=int(ordinary_equal.sum()),
            ordinary_done_child_gates=int(ordinary_done.sum()),
            additional_certified_nonzero_delta_gates=int(additional.sum()),
            additional_certified_positive_gates=int((additional & predicted).sum()),
            additional_certified_different_from_parent=int((additional & (predicted != parent_gate[..., None])).sum()),
            complete_child_T10_words_certified=int(known.all(1).sum()),
            complete_child_H8_T10_groups_certified=int(known.reshape(64, 10, 12, 8, 3).all(axis=(1, 3)).sum()),
            additional_cancelled_child_original_products=int((products[..., 1:]*(ordinary_needed & ~needed)).sum()),
            additional_cancelled_nonzero_delta_products=int((dproducts*(ordinary_needed & ~needed)).sum()),
            raw_parent_plus_fallback_products=parent_work['products']+raw_work['products'],
            raw_parent_then_fallback_W_H8_uses=parent_work['W_H8_uses']+raw_work['W_H8_uses'],
            raw_parent_fallback_unique_W_H8_lower_bound=int((parent_req | raw_req).sum()),
            raw_parent_then_fallback_Conv_k_t_visits=parent_work['Conv_k_t_H8_visits']+raw_work['Conv_k_t_H8_visits'],
            delta_parent_plus_fallback_products=parent_work['products']+delta_fallback_work['products'],
            delta_parent_then_fallback_W_H8_uses=parent_work['W_H8_uses']+delta_fallback_work['W_H8_uses'],
            delta_parent_fallback_unique_W_H8_lower_bound=int((parent_req | delta_fallback_req).sum()),
            raw_fallback_PSN_products=psn_products(y[..., 1:], unresolved, support),
            delta_fallback_PSN_products=psn_products(dy, unresolved, support),
            fallback_gate_comparisons=int(unresolved.sum()),
            parent_then_raw_fallback_nonempty_source_descriptors=parent_work['needed_nonempty_source_Pword_descriptors']+raw_work['needed_nonempty_source_Pword_descriptors'],
            checked_complete_T10_gates=int(full_gate.size), certificate_or_fallback_gate_mismatches=wrong)
        axes[name] = dict(counts=metrics, certificate_fees=fees)
    return dict(baselines=baseline, frontend=frontend, axes=axes)


def main():
    params = dict(np.load(HERE / 'integer_deployment/common3_diagonal_34.npz'))
    weight = params['weight_int8'].reshape(96, 864).astype(np.int64)
    lo, hi = compile_bounds(weight)
    frames = []
    for path in sorted((HERE / 'integer_valid10').glob('capture_*.npz')):
        with np.load(path) as sample:
            result = probe_frame(sample, params, weight, lo, hi)
            frames.append(dict(file=str(sample['file']), capture=path.name, **result))
            print(path.name, {k:v['counts']['additional_certified_nonzero_delta_gates'] for k,v in result['axes'].items()}, flush=True)
    def sum_dict(items):
        return {k: sum_dict([v[k] for v in items]) if isinstance(items[0][k], dict)
                else sum(v[k] for v in items) for k in items[0]}
    baselines = sum_dict([f['baselines'] for f in frames])
    axes = sum_dict([f['axes'] for f in frames])
    for axis in axes.values():
        c=axis['counts']; full=baselines['full_single_scan']; ordinary=baselines['ordinary_equal_or_empty_single_scan']
        axis['rates'] = dict(raw_product_reduction_vs_full=1-c['raw_parent_plus_fallback_products']/full['products'],
            raw_W_change_vs_full=c['raw_parent_then_fallback_W_H8_uses']/full['W_H8_uses']-1,
            raw_unique_W_lower_bound_reduction=1-c['raw_parent_fallback_unique_W_H8_lower_bound']/full['W_H8_uses'],
            raw_product_reduction_vs_ordinary_equal_empty=1-c['raw_parent_plus_fallback_products']/ordinary['products'],
            delta_product_reduction_vs_signed_delta=1-c['delta_parent_plus_fallback_products']/baselines['signed_delta_single_scan']['products'],
            additional_certificate_fraction=c['additional_certified_nonzero_delta_gates']/(c['child_gates']-c['ordinary_done_child_gates']))
    result = dict(
        scope='same common3 integer student, four real captured frames, 64 native P4/frame, all H96/T10; p0 is the fixed full-computation parent',
        amplitude='theta_source is already folded in actual W; source addition/removal is a signed gate difference. Output remains theta_output*g; integer threshold tau is distinct from theta.',
        source_theta=float(params['theta_source']), output_theta=float(params['theta_output']),
        strict='existing 12-bin L/U W sums, counts rounded upward: deltaY in [L(add)-U(remove),U(add)-L(remove)], followed by signed A interval propagation around exact parent U',
        safe_margin='parent positive: Uparent-tau, allowing equality; parent negative: tau-1-Uparent. Zero margin/radius accepts only zero bound.',
        l2_primary=dict(url=PAPER,section='3.1 and Algorithm 1',
            transfer='fixed-parent range-bound core only; fused row A[t,:] tensor-product W[h,:], squared comparison m^2 >= row_Hamming*||Arow||2^2*||Wh||2^2; original zero side and two-sided threshold extension both reported',
            missing='no multi-frame bound chaining, original cache lifecycle, author runtime or parent-selection reconstruction'),
        radius_controls='exact unweighted radius uses saturated 12-bit count budget; log control rounds down to a power of two. Code 0 means budget 0, code e+1 means 2^e. The shared source frontend emits ceil-log required-radius codes, checked against decoded integer comparisons. Weighted control shifts/adds 34 coefficients once in the source frontend, shared over H96.',
        cache_bytes_per_parent_P4=dict(parent_source_T10_packed_bytes=1080,
            unweighted_exact_12bit_plus_gate_and_source=2640,
            unweighted_log_4bit_plus_gate_and_source=1680,
            weighted_log_5bit_plus_gate_and_source=1800,
            weighted_exact_domain_bits=frames[0]['frontend']['weighted_exact_budget_bits'],
            weighted_exact_packed_plus_gate_and_source=1080+960*(frames[0]['frontend']['weighted_exact_budget_bits']+1)//8,
            full_U48_plus_source=6840, strict_bound_table_shared_layer=6912,
            L2_norm_product_u64_shared_layer=7680),
        controls='Full source/W-zero-skipped compute; source-empty or exact-equal-delta consumer reuse; fixed-p0 subset-only arithmetic (not full Prosperity per-K forest); stronger signed-XOR delta plus linear PSN reuse, also not claimed as a complete Prosperity implementation.',
        cost_boundary=[
            'All parent Conv/PSN/output computation is paid. A quarter of spatial positions are parents; their actual activity is counted rather than assuming a quarter of operations.',
            'Ordinary identity/empty masks are source-derived and can share one W scan with parents. Margin-certificate fallback depends on completed parent U: parent and fallback logical W services are added.',
            'Unique W union is only a retained-weight lower bound, not free removal of the second phase. One H8 full-K strip contains 6912 INT8 coefficient bytes; cache/bank reads and holding costs remain.',
            'Source word comparisons/count reductions are shared across H96 and explicitly listed. Product-empty metadata requires source/W-mask reduction, not a look at a cancelling child Yi.',
            'Failed children have both an original-source exact fallback and a signed-delta exact fallback cost. Both numerical answers are checked; no child truth chooses the certificate or parent.',
            'Logical multiplications, additions, comparisons and descriptor/weight uses are separate. No finite-port schedule, throughput, energy, PPA or whole-frame coverage is claimed.',
            'Parent margin-generation operations can share the original threshold comparison/subtraction datapath; listed operation uses are not all incremental hardware work.',
            'The fixed p0 restriction does not reproduce the complete parent-selection/caching algorithms of Prosperity or MLSys2021.'
        ], frames=frames, baselines=baselines, axes=axes,
        frontend_totals={k: (max(f['frontend'][k] for f in frames) if k.startswith('max_') or k.endswith('_bits')
                             else sum(f['frontend'][k] for f in frames))
                         for k in frames[0]['frontend'] if isinstance(frames[0]['frontend'][k], int)})
    out=HERE/'margin_reuse_probe.json'; out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(out), baselines=baselines,
        axes={k:dict(rates=v['rates'], counts=v['counts'], fees=v['certificate_fees']) for k,v in axes.items()}), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
