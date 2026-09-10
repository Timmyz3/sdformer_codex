"""One fixed CPU comparison: complete temporal-word rearrangement bounds.

Same two recovered students, original64 P4/actual PED anchors, natural12 C8x9
blocks, H8 coefficient requests and explicit-L source unions as
scan_gate_completion.py. No file in that implementation is changed. The
controller receives source words, fixed coefficients and completed partials;
it never receives future true Conv sums or the actual answer.

For a block, sort the72 values d_t(word)=sum_s A[t,s]*bit_s(word), INCLUDING
the zero-word slots, and pair with the72 sorted effective coefficients. This
is an exact real-arithmetic extremum over assignments consistent with the
word histogram, not the actual k-to-word assignment. CPU precomputation of
all dynamic block intervals is explicitly charged, not an offline table.
F64 enclosure covers d/dot roundoff and the original ascending-k then
ascending-s function. Native FP32/TF32 remains a separate reference.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np

import scan_gate_completion as base


def gamma(n):
    unit = np.finfo(np.float64).eps/2
    return n*unit/(1-n*unit)


def upward_sum_abs(values, axis):
    n = values.shape[axis]
    return np.nextafter(np.abs(values).sum(axis=axis)*(1+gamma(2*n+2)), np.inf)


def prepare_word_bounds(coeff, words_kj, a, block_k):
    """Dynamic block extrema; no pairing with the actual source-k positions."""
    h, k = coeff.shape
    t, j = len(a), words_kj.shape[1]
    blocks = k//block_k
    labels = np.arange(1 << t, dtype=np.uint16)
    bits = ((labels[None] >> np.arange(t)[:, None]) & 1).astype(bool)
    # A is fixed: this table can be compiled once, so runtime is charged for
    # table requests, not for needlessly rebuilding every d with T additions.
    d_table = np.zeros((t, len(labels)), np.float64)
    for s in range(t):
        d_table += a[:, s, None]*bits[s][None]
    a_abs = upward_sum_abs(a, 1)
    by_block = coeff.reshape(h, blocks, block_k).transpose(1, 0, 2)
    ordered_w = np.sort(by_block, axis=-1)
    abs_block = upward_sum_abs(by_block, 2)
    lo = np.zeros((blocks, t, h, j), np.float64)
    hi = np.zeros_like(lo)
    distinct_words, distinct_d = [], []
    logical_products = logical_adds = nonzero_products = nonzero_adds = 0
    value_lookups = 0
    for b in range(blocks):
        words = words_kj[b*block_k:(b+1)*block_k]
        # source-ready data, shared across all output channels h.
        d = d_table[:, words]
        sorted_d = np.sort(d, axis=1)
        dw = np.asarray([np.unique(words[:, p]).size for p in range(j)], np.int16)
        dd = np.asarray([[np.unique(d[out, :, p]).size for p in range(j)]
                         for out in range(t)], np.int16)
        distinct_words.extend(dw.tolist())
        distinct_d.extend(dd.reshape(-1).tolist())
        value_lookups += t*int(np.count_nonzero(words))
        # One dense CPU reference product for each bound, with static W order.
        # Sorting order and endpoint reduction are not the target FP32 order.
        for reverse, target in ((False, hi), (True, lo)):
            chosen = sorted_d[:, ::-1, :] if reverse else sorted_d
            flat = chosen.transpose(1, 0, 2).reshape(block_k, t*j)
            dot = (ordered_w[b]@flat).reshape(h, t, j).transpose(1, 0, 2)
            # |d_round-d_real|<=gamma_T sum|A|. Sorting rounded d cannot
            # break safety: either extremum is Lipschitz in each d with
            # constant |w|. Dot product rounding adds gamma_(2*block_k).
            # The larger combined gamma below encloses both errors.
            guard = np.nextafter(gamma(2*block_k+2*t+8)*
                a_abs[:, None]*abs_block[b][None], np.inf)[:, :, None]
            lower = dot-guard if reverse else dot+guard
            target[b] = np.nextafter(lower, -np.inf if reverse else np.inf)
            all_zero = np.all(words == 0, axis=0)
            target[b, :, :, all_zero] = 0
            target[b, :, abs_block[b] == 0] = 0
            logical_products += block_k*t*h*j
            logical_adds += (block_k-1)*t*h*j
            counts = (ordered_w[b] != 0).astype(np.int16)@(
                flat != 0).astype(np.int16)
            nonzero_products += int(counts.sum())
            nonzero_adds += int(np.maximum(counts-1, 0).sum())
    summary = dict(
        source_prepass_word_slots=int(k*j), source_prepass_nonzero_words=int(np.count_nonzero(words_kj)),
        source_prepass_logical_bits=int(k*j*t),
        source_prepass_note='All source words must already exist for all12 blocks; omitted NRV zero words still occupy positions in each72-slot histogram. Pre-read, residency or re-read is not free.',
        static_d_entries=int(d_table.size), static_d_Float64_payload_bytes=int(d_table.nbytes),
        static_sorted_weight_entries=int(ordered_w.size), static_sorted_weight_Float64_payload_bytes=int(ordered_w.nbytes),
        dynamic_d_table_value_requests_nonzero_words=int(value_lookups),
        dynamic_d_zero_word_constant_bypasses=int(t*(words_kj.size-np.count_nonzero(words_kj))),
        dynamic_sort_jobs=int(blocks*t*j), dynamic_sort_key_items=int(blocks*t*j*block_k),
        dynamic_sort_comparisons=None,
        dynamic_sort_note='Actual NumPy sorts of72 d values per(t,block,p); comparison/port/latency count is not instrumented. Shared across h, not free across t.',
        dynamic_bound_endpoint_products=int(logical_products), dynamic_bound_endpoint_additions=int(logical_adds),
        hypothetical_zero_skipping_endpoint_products=int(nonzero_products),
        hypothetical_zero_skipping_endpoint_additions_first_assignment_free=int(nonzero_adds),
        zero_skipping_note='Logical arithmetic control only; no mapped selector/ports/schedule. Dense CPU computation above really computes both full72-term endpoints.',
        dynamic_interval_Float64_payload_bytes=int(lo.nbytes+hi.nbytes),
        per_full_P4_H8_dynamic_intervals_Float64_bytes=int(blocks*t*8*4*2*8),
        interval_storage_note='CPU materializes all dynamic block endpoints. Streaming, recomputation or smaller windows require a separate charged schedule; not a static ROM.',
        distinct_words_per_block_context=dict(mean=float(np.mean(distinct_words)), maximum=int(max(distinct_words))),
        distinct_d_per_t_block_context=dict(mean=float(np.mean(distinct_d)), maximum=int(max(distinct_d))),
        unimplemented_histogram_prefix_control='Static d-order plus runtime histogram counts can query static sorted-W prefix sums to reduce endpoint arithmetic; not implemented/measured here. Zero d still advances cumulative slots. Count generation, ordering, table reads and endpoint products remain due.',
        numerical_guard='Per-block gamma_(2*72+2*T+8)*sum|W|*sum|A| plus directed rounding; current projection-order guard is charged separately.')
    return lo, hi, abs_block, summary


def scan_temporal(coeff, words_kj, identity, offset, neuron, anchor_groups, groups, block_k=72):
    h, k = coeff.shape
    t, j = len(neuron['a']), words_kj.shape[1]
    blocks = k//block_k
    source = ((words_kj[None] >> np.arange(t)[:, None, None]) & 1).astype(bool)
    a, bias, center = neuron['a'], neuron['bias'], neuron['center']
    lower, upper, abs_block, preparation = prepare_word_bounds(coeff, words_kj, a, block_k)
    # Suffixes are real dynamic ranges, not future exact contributions. Build
    # once to avoid the old unoptimized repeated sum of all remaining blocks.
    suffix_lo = np.zeros((blocks+1, t, h, j), np.float64)
    suffix_hi = np.zeros_like(suffix_lo)
    for b in range(blocks-1, -1, -1):
        suffix_lo[b] = np.nextafter(lower[b]+suffix_lo[b+1], -np.inf)
        suffix_hi[b] = np.nextafter(upper[b]+suffix_hi[b+1], np.inf)
    preparation.update(dynamic_suffix_endpoint_adds=int(2*blocks*t*h*j),
        CPU_suffix_Float64_payload_bytes=int(suffix_lo.nbytes+suffix_hi.nbytes),
        suffix_note='Same ordinary count-bound control may precompute/incrementally maintain suffixes; this loop optimization is not credited as a new mechanism.')
    partial = np.zeros((t, h, j), np.float64)
    pending = np.ones_like(partial, bool)
    answers = np.zeros_like(partial)
    retired_at = np.full((h, j), blocks, np.int16)
    need_all = np.zeros((k, j), bool)
    word_requests = np.zeros(groups, np.int64)
    stages = []
    terms_all = 0
    a_abs = np.abs(a)
    for completed in range(blocks+1):
        live = pending.any(0)
        before = int(live.sum())
        requested_products = int(np.einsum('ts,thj->', (a != 0).astype(np.int64), pending.astype(np.int64)))
        charges = dict(known_point_PSN_products=0, terminal_point_PSN_products=0,
            threshold_comparisons=int(pending.sum()), whole_T_tests=before,
            dynamic_interval_endpoint_reads=0, known_plus_range_adds=0,
            F64_guard_products=0, F64_guard_additions=0,
            literal_known_point_PSN_products=int(np.count_nonzero(a)*h*j))
        known = base.point_project(a, identity+(partial+offset[None, :, None]), bias, center)
        if completed == blocks:
            certified, answer = base.classify(known, known, neuron)
            charges['terminal_point_PSN_products'] = requested_products
        else:
            nfuture = k-completed*block_k
            absolute = upward_sum_abs(abs_block[completed:], 0)
            # Compare the reference F64 result with known_F64+future_REAL.
            # Both known_F64 and the complete reference differ from their
            # real expressions. A conservative sum of both error envelopes:
            # 2*gamma_(nfuture+2*T+8)*(sum_s |A|*(|identity|+|partial|
            # +|offset|+sum_remaining|w|)+|b|+|center|).
            # This includes original ascending-k addition, BN offset/identity
            # additions, the ascending-s product/reduction and b-center.
            magnitude = np.abs(identity)+np.abs(partial)+np.abs(offset)[None, :, None]+absolute[None, :, None]
            scale = np.zeros_like(partial)
            for s in range(t):
                scale += a_abs[:, s, None, None]*magnitude[s][None]
            scale += np.abs(bias)[:, None, None]+np.abs(center)[:, None, None]
            scale = np.nextafter(scale*(1+gamma(2*t+8)), np.inf)
            guard = np.nextafter(2*gamma(nfuture+2*t+8)*scale, np.inf)
            low = np.nextafter(np.nextafter(known+suffix_lo[completed], -np.inf)-guard, -np.inf)
            high = np.nextafter(np.nextafter(known+suffix_hi[completed], np.inf)+guard, np.inf)
            certified, answer = base.classify(low, high, neuron)
            charges.update(known_point_PSN_products=requested_products,
                dynamic_interval_endpoint_reads=2*int(pending.sum()),
                known_plus_range_adds=2*int(pending.sum()),
                F64_guard_products=requested_products+int(pending.sum()),
                F64_guard_additions=requested_products+int(pending.sum()))
        newly = pending & certified
        answers[newly] = answer[newly]
        pending[newly] = False
        remaining = pending.any(0)
        retired_at[live & ~remaining] = completed
        stage = dict(completed_blocks=completed, live_h_before=before,
            newly_certified_scalar_gates=int(newly.sum()),
            newly_retired_whole_T_h=int((live & ~remaining).sum()),
            live_h_after=int(remaining.sum()), charges=charges)
        if completed < blocks:
            start, end = completed*block_k, (completed+1)*block_k
            terms, need, words = base.issue_block(words_kj[start:end], remaining,
                coeff[:, start:end], anchor_groups, groups)
            terms_all += terms
            need_all[start:end] = need
            word_requests += words
            # Actual coefficients meet their actual words only here, after the
            # controller has requested this completed block.
            for column in range(start, end):
                partial += np.where(remaining[None],
                    source[:, column, None, :]*coeff[None, :, column, None], 0)
            stage.update(next_block_W_AAC_terms=terms, next_block_W_H8_words=int(words.sum()))
        stages.append(stage)
        if not pending.any():
            break
    return dict(answer=answers, retired_at=retired_at, W_AAC_terms=terms_all,
        W_need_kj=need_all, W_H8_words_per_group=word_requests, stages=stages,
        preparation=preparation)


def frame_probe(data, bound, neuron, projection, block_k=72):
    words0 = data['conv2_source_gate_words'].astype(np.uint16)
    anchor = data['anchor_mask'].astype(bool)
    groups, k, _ = words0.shape
    gg, pp = np.nonzero(anchor)
    words = words0.transpose(1, 0, 2)[:, gg, pp]
    weight = bound['W2'].astype(np.float64).reshape(-1, k)
    h, t = len(weight), len(neuron['a'])
    gain = bound['bn2_gain'].astype(np.float64)
    offset = bound['bn2_offset'].astype(np.float64)+gain*bound['conv2_bias'].astype(np.float64)
    coeff = weight*gain[:, None]*float(bound['sn2_theta'])
    identity = data['identity'].astype(np.float64)[:, :, gg, pp]
    result = scan_temporal(coeff, words, identity, offset, neuron, gg, groups, block_k)
    # Reference and native answers are deliberately consulted only after scan.
    source = ((words[None] >> np.arange(t)[:, None, None]) & 1).astype(bool)
    complete_branch = base.sequential_branch(coeff, source)
    complete_u = base.point_project(neuron['a'], identity+(complete_branch+offset[None, :, None]),
        neuron['bias'], neuron['center'])
    _, complete_gate = base.classify(complete_u, complete_u, neuron)
    mismatches = int(np.count_nonzero(result['answer'] != complete_gate))
    full_terms, full_need, full_words = base.issue_block(words, np.ones((h, len(gg)), bool), coeff, gg, groups)
    record = dict(sampled_native_P4=groups, sampled_anchor_positions=int(anchor.sum()),
        sampled_anchor_T_positions=int(anchor.sum()*t), nonanchor_positions_baseline_deleted=int((~anchor).sum()),
        gate_comparison_domain='Only actual PED anchors; whole normalized r1 branch outside anchors is already deleted in both parents.',
        source_theta=float(bound['sn2_theta']), source_occurrences=int(source.sum()),
        W_full_AAC_terms=full_terms, W_gate_stop_AAC_terms=result['W_AAC_terms'],
        W_cancelled_AAC_terms=full_terms-result['W_AAC_terms'],
        W_cancelled_AAC_fraction=1-result['W_AAC_terms']/max(1, full_terms),
        W_full_H8_coefficient_words=int(full_words.sum()), W_gate_stop_H8_coefficient_words=int(result['W_H8_words_per_group'].sum()),
        W_cancelled_H8_coefficient_words=int(full_words.sum()-result['W_H8_words_per_group'].sum()),
        retired_h_by_completed_blocks=np.bincount(result['retired_at'].reshape(-1), minlength=k//block_k+1).tolist(),
        certified_gate_errors_vs_complete_F64=mismatches, stages=result['stages'],
        preparation=result['preparation'], explicit_controls={})
    if 'conv2_source_theta_g_max_abs' in data:
        record['source_theta_g_max_abs'] = float(data['conv2_source_theta_g_max_abs'])
        if record['source_theta_g_max_abs']:
            raise ValueError('The recorded source has a nonzero theta*g reconstruction error')
    if 'proj_sn_output' in data:
        native = data['proj_sn_output'].astype(np.float64)[:, :, gg, pp]
        record.update(complete_F64_gate_errors_vs_native=int(np.count_nonzero(complete_gate != native)),
            certified_gate_errors_vs_native=int(np.count_nonzero(result['answer'] != native)))
    if 'proj_native_membrane' in data:
        record['complete_F64_membrane_max_abs_vs_native'] = float(np.abs(complete_u-data['proj_native_membrane'][:, :, gg, pp]).max())
    pmat = projection.get('r1_P', projection['U'].astype(np.float64)*gain[None])
    matrices = {'ordinary_explicit_dense_L1': pmat.astype(np.float64)@weight}
    if 'r1_L' in projection:
        matrices['independent_explicit_actual_L1'] = projection['r1_L'].astype(np.float64).reshape(-1, k)
    for name, matrix in matrices.items():
        charge = base.l_requirements(matrix, words, gg, groups)
        need = charge.pop('need_kj')
        before = base.union_report(full_need, need, gg, groups)
        after = base.union_report(result['W_need_kj'], need, gg, groups)
        record['explicit_controls'][name] = dict(L1=charge, no_stop_source=before, stopped_source=after,
            source_union_cancelled_NRV_rows=before['source_union_NRV_rows']-after['source_union_NRV_rows'],
            W_cancelled_minus_extra_L1_AAC_terms=full_terms-result['W_AAC_terms']-charge['AAC_terms'],
            eligibility='Requires its separately evaluated explicit value-path function. Unchanged direct-U still consumes full W1; L0 and its residual contribution are not free.')
    record['ordinary_direct_U_permission'] = dict(W_AAC_terms=full_terms,
        W_H8_coefficient_words=int(full_words.sum()), W_early_cancelled_terms=0,
        reason='Actual continuous anchor value still needs W1 in the original direct-U formulation.')
    if mismatches:
        raise RuntimeError('Whole-word bound disagrees with complete defined F64: '+str(mismatches))
    return record


def self_test():
    # Exhaust all word-to-weight assignments: the new interval must enclose
    # them all even though the actual source/weight pairing is not supplied.
    a = np.array([[1., -1.], [1., 1.]])
    w = np.array([[-2., 3.]])
    words = np.array([[0], [3]], np.uint16)
    lo, hi, _, _ = prepare_word_bounds(w, words, a, 2)
    assert lo[0, 0, 0, 0] < 0 < hi[0, 0, 0, 0]
    assert abs(lo[0, 0, 0, 0]) < 1e-12 and abs(hi[0, 0, 0, 0]) < 1e-12
    rng = np.random.default_rng(961010)
    a = rng.uniform(-2, 2, (3, 3)); w = rng.uniform(-5, 5, (4, 4))
    words = np.array([[0], [3], [5], [7]], np.uint16)
    lo, hi, _, _ = prepare_word_bounds(w, words, a, 4)
    for permutation in itertools.permutations(words[:, 0]):
        bits = ((np.array(permutation)[None] >> np.arange(3)[:, None]) & 1).astype(np.longdouble)
        real = a.astype(np.longdouble)@bits@w.astype(np.longdouble).T
        assert np.all(real >= lo[0, :, :, 0]) and np.all(real <= hi[0, :, :, 0])
    cases = 0
    for mode in ('binary', 'ternary'):
        for sign in (-1., 1.):
            for large in (0., 1e6):
                t, h, k, j = 3, 5, 12, 3
                a = rng.uniform(-2, 2, (t, t))
                coeff = rng.uniform(-1, 1, (h, k))*sign
                words = rng.integers(0, 1 << t, (k, j), dtype=np.uint16)
                identity = rng.normal(size=(t, h, j))+large
                offset = rng.normal(size=h)
                neuron = dict(a=a, bias=np.array([0., .1, -.2]), center=np.array([.03, 0., -.01]),
                    theta=.875, mode=mode, threshold_mode='ordinary', negative_scale=.75)
                result = scan_temporal(coeff, words, identity, offset, neuron, np.arange(j), j, 4)
                source = ((words[None] >> np.arange(t)[:, None, None]) & 1).astype(bool)
                branch = base.sequential_branch(coeff, source)
                u = base.point_project(a, identity+(branch+offset[None, :, None]), neuron['bias'], neuron['center'])
                _, gates = base.classify(u, u, neuron)
                assert np.array_equal(gates, result['answer'])
                cases += 1
    print(json.dumps(dict(self_test='PASS', exhaustive_assignments=24,
        signed_F64_scan_cases=cases, cancellation_example='[1,-1],words00/11 narrows [-5,5] to outward-rounded zero')))


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
    directories = [args.capture/name for name in ('ordinary_rank32', 'independent_sparse_L')]
    output = args.output or args.capture/'scan_temporal_word_completion.json'
    baseline_path = args.capture/'scan_gate_completion.json'
    baseline = json.loads(baseline_path.read_text())
    report = dict(complete=False, capture=str(args.capture), baseline=str(baseline_path),
        fixed=dict(block_columns=72, blocks=12, order='natural C8 then3x3', frames_per_axis=4,
            retirement='All10 output gates of p,h must be certified', required_r1_only_AAC_fraction=.1709),
        method='Complete-word histogram extrema via sorted72 d values and sorted fixed coefficients; zero-word slots included; same source/W/H8/L union bookkeeping.',
        numeric='Same folded F64 coefficient, original ascending-k and ascending-s reference as ordinary scanner. Directed rounding and conservative gamma guards cover reassociation. No nativeFP32/TF32 exactness or admitted hardware format claim.',
        scope='Fixed8 sampled training captures, no new network capture/training/order/block/threshold sweep. Literal bound cost and optimistic simple-zero arithmetic are separate from cycles.',
        omitted='Histogram/prefix optimized endpoint implementation, finite SRAM ports/wordwidth, state scheduling, bound latency, control/merge costs and complete value-path execution.',
        axes={})
    started = time.monotonic()
    for directory in directories:
        bound = base.read_arrays(directory/'bound_parameters.npz')
        neuron = base.neuron_from_npz(base.read_arrays(directory/'neuron_parameters.npz'))
        projection = base.read_arrays(directory/'projection_parameters.npz')
        files = sorted(directory.glob('[0-9][0-9]_*.npz'))
        if len(files) != 4 or neuron['a'].shape != (10, 10):
            raise ValueError('This fixed comparison requires4 actualT10 captures per axis')
        axis = dict(frames=[], A_nonzero=int(np.count_nonzero(neuron['a'])), A_rank=int(np.linalg.matrix_rank(neuron['a'])))
        old_by_frame = {row['file']: row for row in baseline['axes'][directory.name]['frames']}
        for file in files:
            data = base.read_arrays(file)
            row = frame_probe(data, bound, neuron, projection)
            row.update(file=file.name, frame_name=base.scalar_string(data['frame_name']), split=base.scalar_string(data['split']))
            old = old_by_frame[file.name]
            row['ordinary_count_bound_comparison'] = dict(
                W_full_AAC_denominator_difference=row['W_full_AAC_terms']-old['W_full_AAC_terms'],
                additional_cancelled_W_AAC_terms=row['W_cancelled_AAC_terms']-old['W_cancelled_AAC_terms'],
                additional_cancelled_H8_words=row['W_cancelled_H8_coefficient_words']-old['W_cancelled_H8_coefficient_words'],
                ordinary_cancelled_AAC_fraction=old['W_cancelled_AAC_fraction'])
            axis['frames'].append(row)
            report['axes'][directory.name] = axis
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
            print(directory.name, file.name, json.dumps({key:row[key] for key in
                ('W_cancelled_AAC_fraction', 'W_cancelled_H8_coefficient_words', 'certified_gate_errors_vs_complete_F64', 'ordinary_count_bound_comparison')}), flush=True)
        sum_keys = ('W_full_AAC_terms', 'W_gate_stop_AAC_terms', 'W_cancelled_AAC_terms',
            'W_full_H8_coefficient_words', 'W_gate_stop_H8_coefficient_words', 'W_cancelled_H8_coefficient_words',
            'certified_gate_errors_vs_complete_F64', 'complete_F64_gate_errors_vs_native')
        total = {key:sum(row[key] for row in axis['frames']) for key in sum_keys}
        total['W_cancelled_AAC_fraction'] = total['W_cancelled_AAC_terms']/total['W_full_AAC_terms']
        total['W_cancelled_H8_fraction'] = total['W_cancelled_H8_coefficient_words']/total['W_full_H8_coefficient_words']
        total['necessary_r1_only_17_09_percent_AAC_gate'] = total['W_cancelled_AAC_fraction'] > .1709
        total['necessary_gate_is_not_net_gain'] = 'This excludes the entire dynamic bound/source-prepass/control/state cost and does not substitute for explicit value-path AEE.'
        prep_keys = ('source_prepass_word_slots', 'source_prepass_nonzero_words', 'dynamic_d_table_value_requests_nonzero_words',
            'dynamic_sort_jobs', 'dynamic_sort_key_items', 'dynamic_bound_endpoint_products', 'dynamic_bound_endpoint_additions',
            'hypothetical_zero_skipping_endpoint_products', 'hypothetical_zero_skipping_endpoint_additions_first_assignment_free',
            'dynamic_suffix_endpoint_adds')
        total['preparation'] = {key:sum(row['preparation'][key] for row in axis['frames']) for key in prep_keys}
        charge_keys = axis['frames'][0]['stages'][0]['charges']
        total['check_charges'] = {key:sum(stage['charges'][key] for row in axis['frames'] for stage in row['stages']) for key in charge_keys}
        total['additional_cancelled_W_AAC_terms_vs_count'] = sum(row['ordinary_count_bound_comparison']['additional_cancelled_W_AAC_terms'] for row in axis['frames'])
        total['additional_cancelled_H8_words_vs_count'] = sum(row['ordinary_count_bound_comparison']['additional_cancelled_H8_words'] for row in axis['frames'])
        total['explicit_controls'] = {}
        for name in axis['frames'][0]['explicit_controls']:
            controls = [row['explicit_controls'][name] for row in axis['frames']]
            total['explicit_controls'][name] = dict(L1_AAC_terms=sum(row['L1']['AAC_terms'] for row in controls),
                L1_H8_coefficient_words=sum(row['L1']['H8_coefficient_words'] for row in controls),
                full_source_union_NRV_rows=sum(row['no_stop_source']['source_union_NRV_rows'] for row in controls),
                stopped_source_union_NRV_rows=sum(row['stopped_source']['source_union_NRV_rows'] for row in controls),
                W_cancelled_minus_extra_L1_AAC_terms=sum(row['W_cancelled_minus_extra_L1_AAC_terms'] for row in controls))
        axis['totals'] = total
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    report.update(complete=True, wall_seconds=time.monotonic()-started)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print('DONE', str(output), flush=True)


if __name__ == '__main__':
    main()
