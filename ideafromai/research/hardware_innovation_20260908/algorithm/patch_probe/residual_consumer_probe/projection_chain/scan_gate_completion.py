"""Fixed C8x9 scan with source-count bounds at the real T10 PED neuron.

CPU opportunity/requirements only. A source-ready pass supplies exact counts
for all 12 blocks. Initial/block-end permissions use these counts, static
coefficient extrema and already accumulated blocks, never a future true sum.
One (p,h) retires only after all T output gates are certified. Native CUDA
differences are reported separately from the defined Float64 affine function.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def scalar_string(value):
    return str(np.asarray(value).item())


def read_arrays(path):
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def classify(lo, hi, neuron):
    theta = neuron['theta']
    mode, threshold_mode = neuron['mode'], neuron['threshold_mode']
    answer = np.zeros(lo.shape, np.float64)
    if mode == 'binary':
        if threshold_mode == 'symmetric_binary_abs':
            positive = (lo >= theta) | (hi <= -theta)
            zero = (lo > -theta) & (hi < theta)
        else:
            positive, zero = lo >= theta, hi < theta
        answer[positive] = theta
        return positive | zero, answer
    if mode == 'ternary':
        scale = 1. if threshold_mode in ('symmetric_bsa_tsn', 'symmetric_target_rate') else neuron['negative_scale']
        positive, negative = lo >= theta, hi <= -theta*scale
        zero = (lo > -theta*scale) & (hi < theta)
        answer[positive], answer[negative] = theta, -theta
        return positive | negative | zero, answer
    raise ValueError(mode)


def point_project(a, value, bias, center):
    """Defined F64 ordering: ascending s, separate product/add, then b-center."""
    out = np.zeros((a.shape[0], *value.shape[1:]), np.float64)
    for s in range(a.shape[1]):
        out += a[:, s, None, None]*value[s][None]
    return out+bias[:, None, None]-center[:, None, None]


def interval_project(a, lo, hi, bias, center):
    low = np.zeros((a.shape[0], *lo.shape[1:]), np.float64)
    high = np.zeros_like(low)
    for s in range(a.shape[1]):
        c = a[:, s, None, None]
        x, y = c*lo[s][None], c*hi[s][None]
        pl = np.nextafter(np.minimum(x, y), -np.inf)
        ph = np.nextafter(np.maximum(x, y), np.inf)
        pl = np.where(c == 0, 0, pl)
        ph = np.where(c == 0, 0, ph)
        low = np.nextafter(low+pl, -np.inf)
        high = np.nextafter(high+ph, np.inf)
    low = np.nextafter(np.nextafter(low+bias[:, None, None], -np.inf)-center[:, None, None], -np.inf)
    high = np.nextafter(np.nextafter(high+bias[:, None, None], np.inf)-center[:, None, None], np.inf)
    return low, high


def make_tables(coeff, block_k):
    h, k = coeff.shape
    blocks = k//block_k
    by_block = coeff.reshape(h, blocks, block_k).transpose(1, 0, 2)
    ordered = np.sort(by_block, axis=-1)
    lo = np.zeros((blocks, h, block_k+1), np.float64)
    hi = np.zeros_like(lo)
    for n in range(1, block_k+1):
        lo[:, :, n] = np.nextafter(lo[:, :, n-1]+ordered[:, :, n-1], -np.inf)
        hi[:, :, n] = np.nextafter(hi[:, :, n-1]+ordered[:, :, -n], np.inf)
    zero = np.all(by_block == 0, axis=-1)
    lo[zero] = 0
    hi[zero] = 0
    # A conservative absolute-sum bound solely for F64 rounding protection.
    abs_block = np.nextafter(block_k*np.max(np.abs(by_block), axis=-1), np.inf)
    abs_block[zero] = 0
    return lo, hi, abs_block, zero


def count_words(used_kh, word_h=8):
    padded = np.pad(used_kh, ((0, 0), (0, (-used_kh.shape[1]) % word_h)))
    return int(padded.reshape(len(padded), -1, word_h).any(-1).sum())


def source_rows_by_group(need_kj, anchor_groups, groups):
    rows = np.zeros((groups, need_kj.shape[0]), bool)
    for g in np.unique(anchor_groups):
        rows[g] = need_kj[:, anchor_groups == g].any(-1)
    return rows


def issue_block(words_kj, active_hj, weight_hk, anchor_groups, groups):
    pop = np.array([int(i).bit_count() for i in range(1024)], np.int16)
    nz = weight_hk != 0
    consumers = nz.T.astype(np.int16)@active_hj.astype(np.int16)
    terms = int(np.sum(pop[words_kj].astype(np.int64)*consumers))
    need = (words_kj != 0) & (consumers != 0)
    per_group = np.zeros(groups, np.int64)
    for g in np.unique(anchor_groups):
        j = anchor_groups == g
        used = ((words_kj[:, j] != 0).astype(np.int16)@active_hj[:, j].T.astype(np.int16)) > 0
        used &= nz.T
        per_group[g] = count_words(used)
    return terms, need, per_group


def sequential_branch(coeff, source, active=None):
    """Full reference only; no result from here is passed into permissions."""
    out = np.zeros((source.shape[0], coeff.shape[0], source.shape[2]), np.float64)
    for k in range(coeff.shape[1]):
        term = source[:, k, None, :]*coeff[None, :, k, None]
        if active is not None:
            term = np.where(active[None], term, 0)
        out += term
    return out


def scan(coeff, words_kj, identity, offset, neuron, anchor_groups, groups, block_k=72):
    """Controller has no native output, complete Conv2 value, or oracle gate."""
    h, k = coeff.shape
    t, j = neuron['a'].shape[0], words_kj.shape[1]
    blocks = k//block_k
    source = ((words_kj[None] >> np.arange(t)[:, None, None]) & 1).astype(bool)
    counts = source.reshape(t, blocks, block_k, j).sum(2)
    lower, upper, abs_block, zero_block = make_tables(coeff, block_k)
    partial = np.zeros((t, h, j), np.float64)
    pending = np.ones_like(partial, bool)
    answers = np.zeros_like(partial)
    retired_at = np.full((h, j), blocks, np.int16)
    w_need = np.zeros((k, j), bool)
    w_words = np.zeros(groups, np.int64)
    stages = []
    total_terms = 0
    a, bias, center = neuron['a'], neuron['bias'], neuron['center']
    for completed in range(blocks+1):
        live = pending.any(0)
        before = int(live.sum())
        charges = dict(table_pair_uses=0, table_pair_address_union_same_P4=0,
            remaining_range_endpoint_terms=0, interval_PSN_products=0,
            terminal_point_PSN_products=0, threshold_comparisons=int(pending.sum()),
            whole_T_tests=before, input_interval_endpoint_adds=0)
        if completed == blocks:
            point = point_project(a, identity+(partial+offset[None, :, None]), bias, center)
            certified, answer = classify(point, point, neuron)
            charges['terminal_point_PSN_products'] = int(np.einsum('ts,thj->', (a != 0).astype(np.int64), pending.astype(np.int64)))
        else:
            required_s = np.einsum('ts,thj->shj', (a != 0).astype(np.int16), pending.astype(np.int16)) > 0
            remaining_lo = np.zeros_like(partial)
            remaining_hi = np.zeros_like(partial)
            for b in range(completed, blocks):
                n = counts[:, b]
                qlo = lower[b, :, n].transpose(0, 2, 1)
                qhi = upper[b, :, n].transpose(0, 2, 1)
                # Each positive n refers to a static (block,h,n) pair.
                useful = required_s & (n[:, None, :] != 0) & ~zero_block[b][None, :, None]
                uses = int(useful.sum())
                charges['table_pair_uses'] += uses
                charges['remaining_range_endpoint_terms'] += 2*uses
                for g in np.unique(anchor_groups):
                    gj = anchor_groups == g
                    for channel in range(h):
                        selected_n = n[:, gj][useful[:, channel, :][:, gj]]
                        charges['table_pair_address_union_same_P4'] += int(np.unique(selected_n).size)
                # n=0 is the compile-time zero entry; do not inflate it.
                remaining_lo = np.where(qlo == 0, remaining_lo, np.nextafter(remaining_lo+qlo, -np.inf))
                remaining_hi = np.where(qhi == 0, remaining_hi, np.nextafter(remaining_hi+qhi, np.inf))
            # Bound the future ascending-k Float64 additions as well as the
            # real subset sum. No distribution-derived error tolerance is used.
            nfuture = k-completed*block_k
            u = np.finfo(np.float64).eps/2
            gamma = nfuture*u/(1-nfuture*u)
            absolute = np.nextafter(abs_block[completed:].sum(0)*(1+blocks*u), np.inf)
            guard = np.nextafter(gamma*(np.abs(partial)+absolute[None, :, None]), np.inf)
            branch_lo = np.nextafter(np.nextafter(partial+remaining_lo, -np.inf)-guard, -np.inf)
            branch_hi = np.nextafter(np.nextafter(partial+remaining_hi, np.inf)+guard, np.inf)
            xlo = np.nextafter(identity+np.nextafter(branch_lo+offset[None, :, None], -np.inf), -np.inf)
            xhi = np.nextafter(identity+np.nextafter(branch_hi+offset[None, :, None], np.inf), np.inf)
            lo, hi = interval_project(a, xlo, xhi, bias, center)
            certified, answer = classify(lo, hi, neuron)
            charges['interval_PSN_products'] = 2*int(np.einsum('ts,thj->', (a != 0).astype(np.int64), pending.astype(np.int64)))
            charges['input_interval_endpoint_adds'] = 4*int(required_s.sum())
        newly = pending & certified
        answers[newly] = answer[newly]
        pending[newly] = False
        after_live = pending.any(0)
        newly_done = live & ~after_live
        retired_at[newly_done] = completed
        stage = dict(completed_blocks=completed, live_h_before=before,
            newly_certified_scalar_gates=int(newly.sum()), newly_retired_whole_T_h=int(newly_done.sum()),
            live_h_after=int(after_live.sum()), charges=charges)
        if completed < blocks:
            start, end = completed*block_k, (completed+1)*block_k
            terms, need, requests = issue_block(words_kj[start:end], after_live,
                coeff[:, start:end], anchor_groups, groups)
            total_terms += terms
            w_need[start:end] = need
            w_words += requests
            # Natural k ordering. Retired h never receives a future term.
            for c in range(start, end):
                term = source[:, c, None, :]*coeff[None, :, c, None]
                partial += np.where(after_live[None], term, 0)
            stage.update(next_block_W_AAC_terms=terms, next_block_W_H8_words=int(requests.sum()))
        stages.append(stage)
        if not pending.any():
            break
    return dict(answer=answers, retired_at=retired_at, W_AAC_terms=total_terms,
        W_need_kj=w_need, W_H8_words_per_group=w_words, stages=stages,
        counts=counts, table_entries=int(lower.size+upper.size))


def l_requirements(matrix, words, anchor_groups, groups):
    active = np.ones((matrix.shape[0], words.shape[1]), bool)
    terms, need, requests = issue_block(words, active, matrix, anchor_groups, groups)
    return dict(AAC_terms=terms, coefficient_nonzeros=int(np.count_nonzero(matrix)),
        H8_coefficient_words=int(requests.sum()), need_kj=need,
        source_NRV_rows=int(source_rows_by_group(need, anchor_groups, groups).sum()))


def union_report(wneed, lneed, anchor_groups, groups):
    w = source_rows_by_group(wneed, anchor_groups, groups)
    l = source_rows_by_group(lneed, anchor_groups, groups)
    return dict(W_NRV_rows=int(w.sum()), L_NRV_rows=int(l.sum()),
        source_union_NRV_rows=int((w | l).sum()), both_NRV_rows=int((w & l).sum()),
        per_native_P4=[dict(group=g, W=int(w[g].sum()), L=int(l[g].sum()),
            union=int((w[g] | l[g]).sum()), W_only=int((w[g] & ~l[g]).sum()),
            L_only=int((l[g] & ~w[g]).sum()), both=int((w[g] & l[g]).sum())) for g in range(groups)])


def neuron_from_npz(p):
    a = p['proj_sn_A'].astype(np.float64)
    center = np.asarray(p['proj_sn_center'], np.float64).reshape(-1)
    if scalar_string(p['proj_sn_center_mode']) == 'zero':
        center = np.zeros(a.shape[0])
    return dict(a=a, bias=np.broadcast_to(p['proj_sn_b'].astype(np.float64).reshape(-1), (len(a),)),
        center=np.broadcast_to(center, (len(a),)), theta=float(p['proj_sn_theta']),
        mode=scalar_string(p['proj_sn_output_mode']), threshold_mode=scalar_string(p['proj_sn_threshold_mode']),
        negative_scale=float(p['proj_sn_negative_threshold_scale']))


def frame_probe(data, bound, neuron, projection, block_k=72):
    if 'conv2_source_theta_g_max_abs' in data and float(data['conv2_source_theta_g_max_abs']) != 0:
        raise ValueError('Actual source does not match the exported theta*g domain')
    words0 = data['conv2_source_gate_words'].astype(np.uint16)
    anchor = data['anchor_mask'].astype(bool)
    groups, k, p = words0.shape
    gg, pp = np.nonzero(anchor)
    words = words0.transpose(1, 0, 2)[:, gg, pp]
    t = len(neuron['a'])
    weight = bound['W2'].astype(np.float64).reshape(-1, k)
    h = len(weight)
    gain = bound['bn2_gain'].astype(np.float64)
    offset = bound['bn2_offset'].astype(np.float64)+gain*bound['conv2_bias'].astype(np.float64)
    coeff = weight*gain[:, None]*float(bound['sn2_theta'])
    identity = data['identity'].astype(np.float64)[:, :, gg, pp]
    if k % block_k or (words >> t).any():
        raise ValueError('Expected complete C8x9 blocks and legal T bits')
    result = scan(coeff, words, identity, offset, neuron, gg, groups, block_k)
    # The complete answer is built only after the controller has returned.
    source = ((words[None] >> np.arange(t)[:, None, None]) & 1).astype(bool)
    complete_branch = sequential_branch(coeff, source)
    complete_u = point_project(neuron['a'], identity+(complete_branch+offset[None, :, None]), neuron['bias'], neuron['center'])
    _, complete_gate = classify(complete_u, complete_u, neuron)
    mismatches = int(np.count_nonzero(result['answer'] != complete_gate))
    full_terms, full_need, full_words = issue_block(words, np.ones((h, len(gg)), bool), coeff, gg, groups)
    empty_l = np.zeros_like(words, bool)
    full_source_rows = source_rows_by_group(words != 0, gg, groups)
    records = dict(sampled_native_P4=groups, sampled_anchor_positions=int(anchor.sum()),
        sampled_anchor_T_positions=int(anchor.sum()*t), nonanchor_positions_baseline_deleted=int((~anchor).sum()),
        gate_comparison_domain='Actual PED anchors only; the common parent has already deleted the whole normalized branch elsewhere.',
        source_theta=float(bound['sn2_theta']), source_occurrences=int(source.sum()),
        W_full_AAC_terms=full_terms, W_gate_stop_AAC_terms=result['W_AAC_terms'],
        W_cancelled_AAC_terms=full_terms-result['W_AAC_terms'],
        W_cancelled_AAC_fraction=1-result['W_AAC_terms']/max(1, full_terms),
        W_full_H8_coefficient_words=int(full_words.sum()),
        W_gate_stop_H8_coefficient_words=int(result['W_H8_words_per_group'].sum()),
        W_cancelled_H8_coefficient_words=int(full_words.sum()-result['W_H8_words_per_group'].sum()),
        retired_h_by_completed_blocks=np.bincount(result['retired_at'].reshape(-1), minlength=k//block_k+1).tolist(),
        certified_gate_errors_vs_complete_F64=mismatches, stages=result['stages'],
        full_W_source=union_report(full_need, empty_l, gg, groups),
        gate_stop_W_source=union_report(result['W_need_kj'], empty_l, gg, groups),
        source_count_metadata=dict(nonempty_NRV_rows=int(full_source_rows.sum()),
            dense_row_slots_in_anchor_containing_P4=int(len(np.unique(gg))*k),
            active_time_bit_counter_increments=int(source.sum()), count_values=int(result['counts'].size),
            alternative_T_bitplane_popcounts=int(result['counts'].size),
            note='Exact per-T counts require bit counters or a block transpose plus popcounts; a scalar popcount of a T word is insufficient. Metadata pass is not removed by the later W/L union. Residency/re-read and transpose ports are unscheduled.'),
        baseline_anchor_PSN_products=int(np.count_nonzero(neuron['a'])*h*len(gg)),
        baseline_anchor_gate_comparisons=int(t*h*len(gg)),
        explicit_controls={})
    if 'proj_sn_output' in data:
        native = data['proj_sn_output'].astype(np.float64)[:, :, gg, pp]
        records.update(complete_F64_gate_errors_vs_native=int(np.count_nonzero(complete_gate != native)),
            certified_gate_errors_vs_native=int(np.count_nonzero(result['answer'] != native)))
    if 'proj_native_membrane' in data and complete_u.size:
        records['complete_F64_membrane_max_abs_vs_native'] = float(np.max(np.abs(complete_u-data['proj_native_membrane'][:, :, gg, pp])))
    if 'norm2_before_spatial_delete' in data and complete_branch.size:
        records['complete_F64_BN_branch_max_abs_vs_native'] = float(np.max(np.abs(
            complete_branch+offset[None, :, None]-data['norm2_before_spatial_delete'][:, :, gg, pp])))
    pmat = projection.get('r1_P')
    if pmat is None:
        pmat = projection['U'].astype(np.float64)*gain[None, :]
    variants = {'ordinary_explicit_dense_L1': pmat.astype(np.float64)@weight}
    if 'r1_L' in projection:
        variants['independent_explicit_actual_L1'] = projection['r1_L'].astype(np.float64).reshape(-1, k)
    for name, matrix in variants.items():
        lr = l_requirements(matrix, words, gg, groups)
        need = lr.pop('need_kj')
        before = union_report(full_need, need, gg, groups)
        after = union_report(result['W_need_kj'], need, gg, groups)
        records['explicit_controls'][name] = dict(L1=lr, no_stop_source=before, stopped_source=after,
            source_union_cancelled_NRV_rows=before['source_union_NRV_rows']-after['source_union_NRV_rows'],
            W_cancelled_minus_extra_L1_AAC_terms=full_terms-result['W_AAC_terms']-lr['AAC_terms'],
            eligibility='Requires the separately evaluated explicit value path and its constants. W gate permissions alone do not remove the ordinary direct-U continuous dependency; independent L0 cannot be freely folded into identity.')
    records['ordinary_direct_U_permission'] = dict(W_AAC_terms=full_terms,
        W_H8_coefficient_words=int(full_words.sum()), W_early_cancelled_terms=0,
        reason='Original anchor continuous value still consumes W1; the gate-only opportunity above is not a legal early stop for unchanged direct U.')
    if mismatches:
        raise RuntimeError('A certified whole-T word differs from the complete defined Float64 gate: '+str(mismatches))
    return records


def self_test():
    rng = np.random.default_rng(960912)
    checks, early_cases = 0, 0
    for mode in ('binary', 'ternary'):
        for sign in (1., -1.):
            for scale in (0., 1., 1000.):
                t, h, k, groups, p = 3, 4, 8, 3, 4
                a = np.eye(t)+rng.uniform(-.1, .1, (t, t))
                neuron = dict(a=a, bias=np.array([0., .1, -.2]), center=np.array([.03, 0., -.01]),
                    theta=.875, mode=mode, threshold_mode='ordinary', negative_scale=.75)
                gate = rng.random((t, groups, k, p)) < .35
                if scale == 0:
                    gate[:] = False
                words = np.sum(gate.astype(np.uint16) << np.arange(t)[:, None, None, None], axis=0)
                anchor = np.array([[1, 0, 1, 0], [0, 0, 0, 0], [1, 0, 1, 0]], bool)
                identity = rng.normal(0, .2, (t, h, groups, p))+scale
                w = rng.uniform(-1, 1, (h, k)); w[0, :2] = 0
                gain = np.array([1., -.5, .75, -1.25])
                bound = dict(W2=w, bn2_gain=gain, bn2_offset=np.array([.25, -.2, .1, 0.]),
                    conv2_bias=np.array([.1, 0., -.1, 0.]), sn2_theta=sign*1.375)
                projection = dict(U=rng.normal(size=(2, h)))
                l = (projection['U']*gain[None])@w
                l[:, :4] = 0
                projection['r1_L'] = l
                data = dict(conv2_source_gate_words=words, anchor_mask=anchor, identity=identity)
                r = frame_probe(data, bound, neuron, projection, block_k=2)
                assert r['certified_gate_errors_vs_complete_F64'] == 0
                assert r['W_full_AAC_terms'] >= r['W_gate_stop_AAC_terms']
                dense = r['explicit_controls']['ordinary_explicit_dense_L1']
                assert dense['no_stop_source']['source_union_NRV_rows'] == dense['stopped_source']['source_union_NRV_rows']
                early_cases += int(r['retired_h_by_completed_blocks'][0] > 0)
                checks += 1
    # Exact threshold endpoints exercise strict negative/zero inequalities.
    for threshold_mode in ('ordinary', 'symmetric_binary_abs'):
        n = dict(theta=.875, mode='binary', threshold_mode=threshold_mode, negative_scale=1.)
        x = np.array([-.875, np.nextafter(-.875, 0), .875, np.nextafter(.875, 0)])
        known, y = classify(x, x, n)
        assert known.all() and y[2] == .875 and y[3] == 0
        if threshold_mode == 'symmetric_binary_abs':
            assert y[0] == .875 and y[1] == 0
    # Actual 72-column count endpoints, checked using independent chosen sets.
    c = rng.integers(-64, 65, (5, 864)).astype(np.float64)/32
    lo, hi, _, _ = make_tables(c, 72)
    for b in range(12):
        for n in (0, 1, 35, 71, 72):
            chosen = rng.choice(72, n, replace=False)
            sums = c[:, b*72+chosen].sum(-1)
            assert np.all(sums >= lo[b, :, n]) and np.all(sums <= hi[b, :, n])
    # Independent H8 address and product oracle, including a partial last H8.
    words = rng.integers(0, 8, (6, 6), dtype=np.uint16)
    active = rng.random((10, 6)) < .55
    w = rng.integers(-1, 2, (10, 6))
    gg = np.array([0, 0, 2, 2, 3, 3])
    terms, need, issued = issue_block(words, active, w, gg, 4)
    expected_terms = 0; expected_need = np.zeros_like(need); addresses = [set() for _ in range(4)]
    for k in range(6):
        for j in range(6):
            for h in range(10):
                if active[h, j] and w[h, k] != 0:
                    expected_terms += int(words[k, j]).bit_count()
                    if words[k, j]:
                        expected_need[k, j] = True
                        addresses[gg[j]].add((k, h//8))
    assert terms == expected_terms and np.array_equal(need, expected_need)
    assert issued.tolist() == [len(x) for x in addresses]
    print(json.dumps(dict(cpu_self_test='PASS', cases=checks, initially_retired_cases=early_cases,
        table_72_endpoints_cases=60, H8_address_oracle='PASS',
        checks='No future reference enters controller; signed A/gain/source theta, offset/conv bias, zero W, nonanchor deletion, complete-T retirement, dense-L source union and exact threshold endpoints.')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.capture is None:
        parser.error('--capture is required')
    directories = ([args.capture] if (args.capture/'neuron_parameters.npz').exists() else
                   sorted(p.parent for p in args.capture.glob('*/neuron_parameters.npz')))
    if not directories:
        raise ValueError('No axis contains neuron_parameters.npz')
    output = args.output or args.capture/'scan_gate_completion.json'
    report = dict(complete=False, capture=str(args.capture), axes={},
        scope='Sampled native G64/P4, anchor-only W1 opportunity; no whole-frame extrapolation, CUDA equality, cycle simulation or PPA.',
        method='Natural12 C8x9 blocks, initial and block-end checks, exact per-block source counts and static count extrema; all T gates for p,h must be certified before W retirement.',
        numeric='Coefficients/gain/theta folded in Float64, ascending-k additions and ascending-s PSN. Directed bounds plus a gamma_n guard cover future Float64 addition rounding. Native FP32/TF32 is a separate reference; these bounds are not an admitted hardware format.',
        omitted_physical_cost='Coefficient/metadata SRAM layouts, real word width, table sharing/caches, count generation ports, PSN/control arithmetic issue/latency, identity/partial lifetime, output backpressure and complete projection value path.')
    for directory in directories:
        bound = read_arrays(directory/'bound_parameters.npz')
        neuron = neuron_from_npz(read_arrays(directory/'neuron_parameters.npz'))
        if neuron['a'].shape != (10, 10):
            raise ValueError('The real probe is fixed to the actual T10 consumer')
        projection = read_arrays(directory/'projection_parameters.npz')
        files = sorted(directory.glob('[0-9][0-9]_*.npz'))
        if not files:
            raise ValueError('Axis parameters exist but its real frame captures have not arrived: '+str(directory))
        h = bound['W2'].shape[0]; k = int(bound['W2'].size//h); blocks = k//72
        entries = 2*blocks*h*73
        axis = dict(frames=[], charges=dict(count_bound_table_entries=entries,
            evaluation_Float64_table_payload_bytes=entries*8,
            hypothetical_Float32_table_payload_bytes=entries*4,
            Float32_table_note='Capacity scenario only; outward bounds and gate safety not validated in FP32.',
            per_native_P4_count_bits=blocks*10*4*7,
            possible_raw_partial_Float64_payload_per_P4_bytes=10*h*4*8,
            pending_gate_bits_per_P4=10*h*4, whole_T_live_bits_per_P4=h*4,
            note='Payload examples are not SRAM macros or mandatory live state. Identity/default storage and two interval streams need a real schedule; streaming bounds can avoid materializing all endpoints. Table address-union counts are not free ports.'),
            A_shape=list(neuron['a'].shape), A_nonzero=int(np.count_nonzero(neuron['a'])),
            A_rank=int(np.linalg.matrix_rank(neuron['a'])))
        for file in files:
            data = read_arrays(file)
            r = frame_probe(data, bound, neuron, projection)
            r['file'] = file.name
            r['frame_name'] = scalar_string(data.get('frame_name', np.array(file.stem)))
            r['split'] = scalar_string(data.get('split', np.array('not_recorded')))
            axis['frames'].append(r)
            print(directory.name, file.name, json.dumps({key:r[key] for key in
                ('W_full_AAC_terms', 'W_gate_stop_AAC_terms', 'W_cancelled_AAC_fraction', 'certified_gate_errors_vs_complete_F64')}), flush=True)
        keys = ('W_full_AAC_terms', 'W_gate_stop_AAC_terms', 'W_cancelled_AAC_terms',
            'W_full_H8_coefficient_words', 'W_gate_stop_H8_coefficient_words', 'W_cancelled_H8_coefficient_words',
            'sampled_anchor_positions', 'sampled_anchor_T_positions', 'source_occurrences',
            'certified_gate_errors_vs_complete_F64', 'complete_F64_gate_errors_vs_native')
        axis['totals'] = {key:sum(row.get(key,0) for row in axis['frames']) for key in keys}
        axis['totals']['charges'] = {key:sum(stage['charges'][key] for row in axis['frames'] for stage in row['stages'])
            for key in ('table_pair_uses', 'table_pair_address_union_same_P4', 'remaining_range_endpoint_terms',
                'interval_PSN_products', 'terminal_point_PSN_products', 'threshold_comparisons', 'whole_T_tests', 'input_interval_endpoint_adds')}
        axis['totals']['explicit_controls'] = {}
        if axis['frames']:
            for name in axis['frames'][0]['explicit_controls']:
                controls = [row['explicit_controls'][name] for row in axis['frames']]
                axis['totals']['explicit_controls'][name] = dict(
                    L1_AAC_terms=sum(row['L1']['AAC_terms'] for row in controls),
                    L1_H8_coefficient_words=sum(row['L1']['H8_coefficient_words'] for row in controls),
                    full_source_union_NRV_rows=sum(row['no_stop_source']['source_union_NRV_rows'] for row in controls),
                    stopped_source_union_NRV_rows=sum(row['stopped_source']['source_union_NRV_rows'] for row in controls),
                    W_cancelled_minus_extra_L1_AAC_terms=sum(row['W_cancelled_minus_extra_L1_AAC_terms'] for row in controls))
        report['axes'][directory.name] = axis
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    report['complete'] = True
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print('DONE', str(output), flush=True)


if __name__ == '__main__':
    main()
