"""One fixed source-count bound at the real PED consumer; CPU opportunity model.

Known inputs are identity, the complete sn2 support and static W2/BN2/PSN.
Full Conv2 values are used only to check the answer. This is a Float64 affine
range experiment, not a bit-exact CUDA replacement or a cycle simulation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def project(a, value):
    return np.einsum('ts,shgp->thgp', a, value, optimize=True)


def classify(lo, hi, theta, mode, threshold_mode, negative_scale):
    known = np.zeros(lo.shape, bool)
    answer = np.zeros(lo.shape, np.float64)
    if mode == 'binary':
        if threshold_mode == 'symmetric_binary_abs':
            positive = (lo >= theta) | (hi <= -theta)
            zero = (lo > -theta) & (hi < theta)
        else:
            positive, zero = lo >= theta, hi < theta
        known = positive | zero
        answer[positive] = theta
    elif mode == 'ternary':
        scale = 1. if threshold_mode in ('symmetric_bsa_tsn', 'symmetric_target_rate') else negative_scale
        positive, negative = lo >= theta, hi <= -theta*scale
        zero = (lo > -theta*scale) & (hi < theta)
        known = positive | negative | zero
        answer[positive], answer[negative] = theta, -theta
    else:
        raise ValueError(mode)
    return known, answer


def request_counts(source, needed, weight_nonzero):
    """Same declared K-major/H-contiguous FP32 storage for both executions.

    One coefficient word is fetched at most once per native P4 across T10.
    No extra inter-P4 weight cache is silently given to either execution.
    """
    result = {str(width): 0 for width in (2, 8)}
    for group in range(source.shape[2]):
        activity = np.einsum('skp,shp->kh',
            source[:, :, group].astype(np.int32),
            needed[:, :, group].astype(np.int32), optimize=True) > 0
        activity &= weight_nonzero.T
        for width in (2, 8):
            result[str(width)] += int(activity.reshape(activity.shape[0], -1, width).any(-1).sum())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    args.output = args.output or args.capture.parent/'count_bound'
    args.output.mkdir(parents=True, exist_ok=True)
    params = np.load(args.capture/'neuron_parameters.npz')
    bound = np.load(args.capture/'bound_parameters.npz')
    a = params['proj_sn_A'].astype(np.float64)
    b = params['proj_sn_b'].astype(np.float64).reshape(10, 1, 1, 1)
    center = params['proj_sn_center'].astype(np.float64)
    if str(params['proj_sn_center_mode']) == 'zero':
        center = np.zeros_like(center)
    center = np.broadcast_to(center.reshape(-1, 1, 1, 1), (10, 1, 1, 1))
    theta = float(params['proj_sn_theta'])
    mode, threshold_mode = str(params['proj_sn_output_mode']), str(params['proj_sn_threshold_mode'])
    negative_scale = float(params['proj_sn_negative_threshold_scale'])
    w = bound['W2'].astype(np.float64).reshape(96, 864)
    gain = bound['bn2_gain'].astype(np.float64)
    offset = bound['bn2_offset'].astype(np.float64)+gain*bound['conv2_bias'].astype(np.float64)
    coeff = w*gain[:, None]*float(bound['sn2_theta'])
    sorted_coeff = np.sort(coeff, axis=1)
    lower_table = np.concatenate((np.zeros((96, 1)), np.cumsum(sorted_coeff, axis=1)), axis=1)
    upper_table = np.concatenate((np.zeros((96, 1)), np.cumsum(sorted_coeff[:, ::-1], axis=1)), axis=1)
    pos, neg = np.maximum(a, 0), np.minimum(a, 0)
    rows = []
    captures = sorted(args.capture.glob('[0-9][0-9]_*.npz'))
    for file in captures:
        data = np.load(file)
        if float(data['conv2_source_theta_g_max_abs']) != 0:
            raise ValueError('Captured Conv2 source is not the exported theta*g domain.')
        words = data['conv2_source_gate_words']
        # T,K,G,P, preserving the native G64/P4 and original spatial padding.
        source = ((words[None].transpose(0, 2, 1, 3) >> np.arange(10)[:, None, None, None]) & 1).astype(bool)
        counts = source.sum(1)
        lower_s = lower_table[:, counts].transpose(1, 0, 2, 3)
        upper_s = upper_table[:, counts].transpose(1, 0, 2, 3)
        identity = data['identity'].astype(np.float64)
        default = project(a, identity+offset[None, :, None, None])+b-center
        lo = default+project(pos, lower_s)+project(neg, upper_s)
        hi = default+project(pos, upper_s)+project(neg, lower_s)
        known, answer = classify(lo, hi, theta, mode, threshold_mode, negative_scale)
        # Direct real-affine Conv2 only checks the envelope; it never selects
        # an input, a bound, a row, or a candidate threshold.
        branch = np.einsum('hk,skgp->shgp', coeff, source.astype(np.float64), optimize=True)
        reconstructed = default+project(a, branch)
        native = data['proj_native_membrane'].astype(np.float64)
        actual = data['proj_sn_output'].astype(np.float64)
        anchor = data['anchor_mask'].astype(bool)
        nonanchor = ~anchor[None, None]
        usable = known & nonanchor
        # A Conv2 output at source time s remains live if any unresolved t
        # depends on it, or if the continuous PED projection needs it.
        # Backward demand uses A.T: one live output t keeps every source s
        # for which A[t,s] is nonzero.
        needed = project((a != 0).T.astype(np.int32), (~known).astype(np.int32)) > 0
        needed |= anchor[None, None]
        complete = np.ones(needed.shape, bool)
        baseline_requests = request_counts(source, complete, w != 0)
        candidate_requests = request_counts(source, needed, w != 0)
        no_anchor_groups = ~anchor.any(-1)
        grouped = known.reshape(10, 12, 8, len(anchor), 4).all((0, 2, 4))
        source_empty = counts == 0
        whole_T_empty = source_empty.all(0)
        baseline_adds = int(counts.sum()*96)
        required_adds = int((counts[:, None]*needed).sum())
        row = dict(file=file.name, gates=known.size,
            source_count_min=int(counts.min()), source_count_mean=float(counts.mean()), source_count_max=int(counts.max()),
            source_empty_time_positions=int(source_empty.sum()), source_time_positions=counts.size,
            certified_gates=int(known.sum()), certified_gate_fraction=float(known.mean()),
            certified_gates_on_nonempty_T10_positions=int((known & ~whole_T_empty[None, None]).sum()),
            whole_T_empty_positions=int(whole_T_empty.sum()),
            usable_nonanchor_gates=int(usable.sum()), usable_nonanchor_fraction_of_all=float(usable.mean()),
            nonanchor_gates=int(np.broadcast_to(nonanchor, known.shape).sum()),
            no_anchor_native_P4=int(no_anchor_groups.sum()),
            certified_whole_native_P4_H8_T10=int(grouped.sum()),
            certified_whole_nonanchor_native_P4_H8_T10=int(grouped[:, no_anchor_groups].sum()),
            whole_native_P4_H8_T10=grouped.size,
            reconstructed_branch_max_abs_vs_native=float(np.max(np.abs(branch+offset[None, :, None, None]-data['norm2_branch']))),
            reconstructed_membrane_max_abs_vs_native=float(np.max(np.abs(reconstructed-native))),
            real_affine_envelope_violations=int(((reconstructed < lo-1e-10) | (reconstructed > hi+1e-10)).sum()),
            native_envelope_violations=int(((native < lo) | (native > hi)).sum()),
            certified_gate_errors_vs_native=int((known & (answer != actual)).sum()),
            minimum_certified_distance_from_threshold=float(np.min(np.maximum(lo-theta, theta-hi)[known])) if known.any() and mode == 'binary' and threshold_mode != 'symmetric_binary_abs' else None,
            baseline_FP32_coefficient_word_requests=baseline_requests,
            bound_FP32_coefficient_word_requests=candidate_requests,
            baseline_scalar_Conv2_adds=baseline_adds,
            required_scalar_Conv2_adds=required_adds,
            scalar_add_reduction_before_all_extra_cost=1-required_adds/max(1, baseline_adds),
            request_reduction={str(width): 1-candidate_requests[str(width)]/max(1, baseline_requests[str(width)]) for width in (2, 8)})
        rows.append(row)
        np.savez_compressed(args.output/file.name, certified=known, answer=answer,
            lower=lo, upper=hi, default_membrane=default, source_counts=counts,
            needed_conv2_time_outputs=needed, anchor_mask=anchor, positions=data['positions'])
        print(json.dumps(row, ensure_ascii=False), flush=True)
    scalar_keys = ('gates', 'certified_gates', 'certified_gates_on_nonempty_T10_positions',
        'whole_T_empty_positions', 'baseline_scalar_Conv2_adds', 'required_scalar_Conv2_adds',
        'usable_nonanchor_gates', 'nonanchor_gates',
        'source_empty_time_positions', 'source_time_positions', 'real_affine_envelope_violations',
        'native_envelope_violations', 'certified_gate_errors_vs_native',
        'certified_whole_native_P4_H8_T10', 'certified_whole_nonanchor_native_P4_H8_T10',
        'whole_native_P4_H8_T10')
    aggregate = {key: sum(row[key] for row in rows) for key in scalar_keys}
    aggregate['certified_fraction'] = aggregate['certified_gates']/aggregate['gates']
    aggregate['usable_nonanchor_fraction_of_all'] = aggregate['usable_nonanchor_gates']/aggregate['gates']
    aggregate['scalar_add_reduction_before_all_extra_cost'] = 1-aggregate['required_scalar_Conv2_adds']/aggregate['baseline_scalar_Conv2_adds']
    aggregate['baseline_requests'] = {str(w): sum(row['baseline_FP32_coefficient_word_requests'][str(w)] for row in rows) for w in (2, 8)}
    aggregate['candidate_requests'] = {str(w): sum(row['bound_FP32_coefficient_word_requests'][str(w)] for row in rows) for w in (2, 8)}
    aggregate['request_reduction'] = {str(w): 1-aggregate['candidate_requests'][str(w)]/aggregate['baseline_requests'][str(w)] for w in (2, 8)}
    result = dict(complete=True, capture=str(args.capture), rows=rows, aggregate=aggregate,
        method='One fixed whole-K864 top-count envelope: n known active theta*g sources imply sums of the n smallest/largest static BN2-scaled W2 coefficients; then signed Aproj interval propagation. No block-size/threshold/order sweep.',
        control='Ordinary source-count interval early completion; same baseline gets source/weight zero skip and full T10/P4 coefficient reuse. M reuse is not included in these claimed counts.',
        scope='Four train frames, original64 native P4 each; real-affine Float64 opportunity and explicitly declared FP32 coefficient words. Not whole-frame requests, native CUDA equivalence, RTL cycles or PPA.',
        charges=dict(bound_table_entries=int(lower_table.size+upper_table.size),
            bound_table_FP32_payload_bytes=int((lower_table.size+upper_table.size)*4),
            note='Evaluation uses FP64 tables. FP32 payload is only a capacity scenario, not a verified outward-rounded hardware bound.',
            per_P4_input_count_state_bits=10*4*10,
            per_P4_count_table_reads=2*10*96*4,
            per_P4_interval_A_products=2*int(np.count_nonzero(a))*96*4,
            per_P4_default_A_products=int(np.count_nonzero(a))*96*4,
            coefficient_word_layout='byte=(k*96+h)*4; 64-bit word H2 / 256-bit word H8; retained for all T10/P4 uses, reset between sampled native P4; no bank stalls or inter-P4 cache modeled',
            extra='Count production, table traffic, default membrane lifetime, two bound streams, comparisons and actual ports remain to be paid; request savings are not net service savings.'))
    (args.output/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
