"""A bounded CPU PSN probe; no training, RTL, PPA, or frozen-FP claim.

The original dense A and continuous X are retained. A rank-r SVD factorization
is only a cheap first classifier. Cauchy-Schwarz bounds the omitted matrix E:
    |(E X)[t,p]|^2 <= ||E[t,:]||_2^2 * ||X[:,p]||_2^2.
The input norm is produced by reading ALL T values, never a free future bound.
Uncertified outputs are evaluated with the ORIGINAL A row, not a lossy E row.
We separately report a lossy low-rank-only model, an explicitly uncharged
actual-residual oracle, and the paid norm-bound classifier plus dense fallback.
All arithmetic counts are scalar operations, NOT cycles or energy.

Fixed selections before measuring: sample0; front head.sn T10, stage0/block0
mlp.sn2 T10, stage0/block0 attn.sn_k T2. T10 ranks 1/2; T2 rank 1 is a negative
control. FC1 reconstruction uses all rows in the current BN domain for 32
uniformly spaced hidden channels; its probe keeps sixteen uniform spatial
packets of 32 positions, preserving complete T. Real FP32 hook captures, if
available, are accepted only as complete-T arrays with their recorded indices.
The requested rank-0 norm-only classifier is a strong control. One additional
bound uses SVD orthogonality: ||E_t||^2 (||X||^2-||V_r X||^2). Computing r
squares, the subtraction, and a conservative float64 margin is charged; squared
comparisons avoid a square root. Neither projection geometry nor rank-0 norm
prediction is a new primitive. No basis is fit to these test inputs.

The paid storage model compares the strong input-stationary dense A service
(one source read, T partial sums) with two realizable fallback organizations:
one reread broadcasting X to all failed rows, or one failed row at a time.
Candidate X is always saved before the decision; latent/norm/decision state and
both reads and writes are charged. No saved PSN product is an upstream FC1 save.
"""
from pathlib import Path
import argparse
import json
import math
import struct
import sys
import zlib

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07')
sys.path.insert(0, str(HERE.parents[1] / 'mechanism_rebuild_gh_20260906/scripts'))
from checkpoint_numpy import read_checkpoint

CKPT = HW / 'system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth'
CAP = HW / 'results/m1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831'
FC = HW / 'results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901'
PRE = 'sttmultires_unet.encoders.swin3d.'
MODULES = {
    'head': PRE + 'patch_embed.head.sn.spiking_neuron',
    'fc1_post': PRE + 'layers.0.swin_blocks.0.mlp.sn2.spiking_neuron',
    'sn_k': PRE + 'layers.0.swin_blocks.0.attn.sn_k.spiking_neuron',
}
HEADER = struct.Struct('<8sHH11I')
QUANTILES = [0, .01, .05, .25, .5, .75, .95, .99, 1]


def quantiles(x):
    return {str(q): float(v) for q, v in zip(QUANTILES, np.quantile(x, QUANTILES))}


def static_norm_constants(a, offset, theta):
    """Compile the rank-0 comparison; never pay two multiplies per output.

    The FP32 sensitivity constants additionally allow sequential norm rounding
    and a conservative standard dot-product error envelope, assuming normal
    finite IEEE FP32 arithmetic (no TF32, underflow, or overflow). This is a
    separate deployment sensitivity, not a proof about arbitrary GPU kernels.
    """
    t = len(a)
    a2 = np.sum(a * a, axis=1)
    margin = offset[:, 0] - theta
    zero_rows = a2 == 0
    kappa = np.divide(margin * margin, a2, out=np.full(t, np.inf), where=~zero_rows)
    unit = np.finfo(np.float32).eps / 2
    gamma_norm = (2*t*unit) / (1-2*t*unit)
    gamma_dot = ((2*t+4)*unit) / (1-(2*t+4)*unit)
    gap = np.maximum(0, np.abs(margin) - gamma_dot*np.abs(offset[:, 0]))
    guarded = np.divide(gap * gap * (1-gamma_norm), a2 * (1+gamma_dot)**2,
                        out=np.full(t, np.inf), where=~zero_rows)
    guarded32 = np.nextafter(guarded.astype(np.float32), np.float32(-np.inf)).astype(np.float64)
    return kappa, guarded32, margin >= 0, zero_rows


def read_fc_sample0():
    specs = {r['layer_id']: r for r in json.loads((FC / 'layers.json').read_text())['layers']}
    chosen = {k: v for k, v in specs.items()
              if v['module_name'] in {PRE + 'layers.0.swin_blocks.0.mlp.fc1',
                                      PRE + 'layers.0.swin_blocks.0.mlp.fc2'}}
    rows = {k: [] for k in chosen}
    with (FC / 'fc_frames.bin').open('rb') as f:
        while raw := f.read(HEADER.size):
            magic, version, hs, lid, sid, fi, start, n, c, br, nnz, rb, cb, crc = HEADER.unpack(raw)
            if sid > 0:
                break
            if lid not in chosen:
                f.seek(cb, 1)
                continue
            payload = zlib.decompress(f.read(cb))
            bits = np.unpackbits(np.frombuffer(payload[:n * br], np.uint8).reshape(n, br),
                                 axis=1, bitorder='little')[:, :c]
            rows[lid].append(bits)
    return {chosen[k]['module_name']: (chosen[k], np.concatenate(v)) for k, v in rows.items()}


def reconstructed_fc1(state):
    data = read_fc_sample0()
    prefix = PRE + 'layers.0.swin_blocks.0.mlp.'
    spec, source = data[prefix + 'fc1']
    _, expected = data[prefix + 'fc2']
    t = 10
    p = len(source) // t
    h = spec['output_channels']
    channels = np.rint(np.linspace(0, h - 1, 32)).astype(int)
    starts = np.rint(np.linspace(0, p - 32, 16)).astype(int)
    positions = (starts[:, None] + np.arange(32)).reshape(-1)
    w = state[prefix + 'fc1.weight'][channels].astype(np.float64)
    theta_source = float(state[prefix + 'sn1.spiking_neuron.thresh'])
    y = (source.astype(np.float64) @ w.T * theta_source).reshape(t, p, len(channels))
    mu = y.mean((0, 1))
    var = ((y - mu) ** 2).mean((0, 1))
    gamma = state[prefix + 'bn1.norm_layer.weight'][channels].astype(np.float64)
    beta = state[prefix + 'bn1.norm_layer.bias'][channels].astype(np.float64)
    x = gamma * (y - mu) / np.sqrt(var + 1e-5) + beta
    a = state[MODULES['fc1_post'] + '.weight'].astype(np.float64)
    offset = (state[MODULES['fc1_post'] + '.bias'] - state[MODULES['fc1_post'] + '.center']).astype(np.float64)
    theta = float(state[MODULES['fc1_post'] + '.thresh'])
    full_h = np.einsum('ts,sph->tph', a, x, optimize=True) + offset[:, None, :]
    original_bits = expected.reshape(t, p, h)[:, :, channels].astype(bool)
    all_domain_mismatches = int(np.count_nonzero((full_h >= theta) != original_bits))
    chosen_x = x[:, positions, :].transpose(0, 2, 1).reshape(t, -1)
    chosen_bits = original_bits[:, positions, :].transpose(0, 2, 1).reshape(t, -1)
    info = {
        'kind': 'FLOAT64_RECONSTRUCTION_FROM_REAL_FC1_SOURCE_AND_FROZEN_WEIGHTS',
        'original_input_shape': [t, 1, 120, 160, h],
        'BN_domain_rows_including_T': int(t * p), 'BN_eps_assumption': 1e-5,
        'selected_hidden_channels': channels.tolist(), 'selected_spatial_packet_starts': starts.tolist(),
        'spatial_packet_size': 32, 'full_T': True,
        'complete_selected_channel_BN_domain_output_bits_checked': int(full_h.size),
        'complete_selected_channel_BN_domain_bit_mismatches': all_domain_mismatches,
        'source_theta': theta_source,
        'numeric_scope': 'Float64 operator reconstruction; original GPU FP32 input is not retained in M1707.',
    }
    return chosen_x, chosen_bits, info


def hook_data(key, state, path):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        x = z['x'].astype(np.float64)
        out = z['output']
        info = {'kind': 'REAL_FP32_PSN_HOOK_INPUT', 'path': str(path),
                'sample': path.parent.name,
                'original_input_shape': z['original_shape'].tolist(),
                'selected_columns': int(z['indices'].size), 'full_T': True,
                'original_dtype': str(z['x'].dtype), 'center_mode': str(z['center_mode']),
                'capture_A_matches_checkpoint': bool(np.array_equal(z['A'], state[MODULES[key]+'.weight'])),
                'capture_theta_amplitude': float(z['theta']),
                'captured_output_values': np.unique(out).tolist()}
        assert info['capture_A_matches_checkpoint']
        return x, out != 0, info


def matrix_inventory(state, activity):
    result = []
    for key, name in MODULES.items():
        a = state[name + '.weight'].astype(np.float64)
        u, s, vh = np.linalg.svd(a, full_matrices=False)
        ar = activity[name]
        rows = []
        for rank in ([1, 2] if len(a) == 10 else [1]):
            e = a - (u[:, :rank] * s[:rank]) @ vh[:rank]
            rows.append({'rank': rank, 'residual_row_L2': np.linalg.norm(e, axis=1).tolist(),
                         'matrix_relative_Frobenius_error': float(np.linalg.norm(e) / np.linalg.norm(a)),
                         'low_rank_products_per_trajectory': 2 * len(a) * rank})
        result.append({
            'key': key, 'module': name, 'T': len(a), 'matrix': a.tolist(),
            'rank': int(np.linalg.matrix_rank(a)), 'singular_values': s.tolist(),
            'nonzero_weights': int(np.count_nonzero(a)),
            'future_weights_nonzero': int(np.count_nonzero(np.triu(a, 1))),
            'past_weights_nonzero': int(np.count_nonzero(np.tril(a, -1))),
            'negative_weights': int((a < 0).sum()), 'bias': state[name + '.bias'].reshape(-1).tolist(),
            'center': state[name + '.center'].reshape(-1).tolist(),
            'theta_amplitude': float(state[name + '.thresh']),
            'input_first_shape': ar['input_first_shape'],
            'input_first_density': ar['input_first_density'],
            'input_first_binary01_ratio': ar['input_first_binary01_ratio'],
            'input_first_min': ar['input_first_value_min'], 'input_first_max': ar['input_first_value_max'],
            'full_PSN_products_per_sample': ar['input_first_elements'] * len(a),
            'input_FP32_payload_bytes_per_sample_not_required_SRAM': ar['input_first_bytes_fp32'],
            'factorizations': rows,
        })
    return result


def counted_cost(t, n, rank, certified, bound_mode):
    failed = ~certified
    f = failed.sum(0)
    total_failed = int(f.sum())
    any_failed = int(np.count_nonzero(f))
    source_values = t * n
    base_products = t * t * n
    product_counts = {
        'latent_projection': rank * t * n, 'low_rank_output_projection': t * rank * n,
        'input_squared_norm': t * n, 'margin_square': t * n,
        'residual_bound_times_norm': t * n, 'original_A_fallback': total_failed * t,
    }
    if bound_mode == 'static_norm':
        product_counts['margin_square'] = 0
        product_counts['residual_bound_times_norm'] = 0
    addition_counts = {
        'latent_projection': rank * (t - 1) * n,
        'low_rank_output_with_folded_bias_threshold': t * rank * n,
        'input_squared_norm': (t - 1) * n,
        'original_A_fallback_with_folded_bias_threshold': total_failed * t,
    }
    if bound_mode == 'orthogonal_norm':
        product_counts['latent_squared_norm'] = rank * n
        product_counts['conservative_norm_margin'] = n
        addition_counts['latent_norm_reduce_subtract_margin'] = (rank + 1) * n
    packet = 32
    return {
        'dense_baseline': {
            'scalar_products': base_products, 'accumulator_adds_including_folded_bias_threshold': base_products,
            'final_comparisons': t * n, 'source_values_read_once': source_values,
            'source_bytes_at_assumed_FP32': 4 * source_values,
            'minimum_partial_sum_words_for_32_trajectories': packet * t,
        },
        'candidate_scalar_products': product_counts,
        'candidate_scalar_adds': addition_counts,
        'candidate_total_products': sum(product_counts.values()),
        'candidate_total_adds': sum(addition_counts.values()),
        'paid_product_ratio_vs_dense': sum(product_counts.values()) / base_products,
        'paid_add_ratio_vs_dense': sum(addition_counts.values()) / base_products,
        'uncertified_output_count': total_failed, 'trajectories_with_any_fallback': any_failed,
        'failed_rows_per_trajectory_quantiles': quantiles(f),
        'bound_squared_comparisons': t * n, 'margin_sign_tests': 0 if bound_mode == 'static_norm' else t*n,
        'fallback_final_comparisons': total_failed,
        'static_coefficient_words_dense_A_plus_folded_offset': t * t + t,
        'static_coefficient_words_candidate_including_original_A': t * t + 2 * t * rank + 2 * t,
        'multiplication_ratio_formula': 'static rank0: 1+1/T-cert; otherwise 1+(2*r+3)/T-cert + ((r+1)/T^2 only for guarded orthogonal norm)',
        'minimum_product_ratio_even_if_all_certified': 1/t if bound_mode == 'static_norm' else (2 * rank + 3) / t + ((rank+1)/t**2 if bound_mode == 'orthogonal_norm' else 0),
        'norm_remainder_clamp_comparisons': n if bound_mode == 'orthogonal_norm' else 0,
        'square_roots_required': 0,
        'source_values_first_read': source_values,
        'candidate_X_saved_values_before_certification': source_values,
        'candidate_X_save_bytes_FP32': 4 * source_values,
        'parallel_failed_rows_X_reread_values': any_failed * t,
        'parallel_failed_rows_X_reread_bytes_FP32': 4 * any_failed * t,
        'serial_failed_row_X_reread_values': total_failed * t,
        'serial_failed_row_X_reread_bytes_FP32': 4 * total_failed * t,
        'parallel_extra_X_write_plus_read_bytes_FP32': 4 * t * (n + any_failed),
        'serial_extra_X_write_plus_read_bytes_FP32': 4 * t * (n + total_failed),
        'candidate_input_store_words_for_32_trajectories': packet * t,
        'candidate_first_pass_latent_norm_words_for_32': packet * (rank + 1),
        'candidate_serial_fallback_peak_words_for_32': packet * (t + rank + 2),
        'candidate_parallel_fallback_worst_peak_words_for_32': packet * max(t + rank + 2, 2 * t),
        'candidate_parallel_fallback_observed_peak_words_for_32': packet * max(t + rank + 2, t + int(f.max())),
        'candidate_decision_and_failed_mask_bits_for_32': 2 * packet * t,
        'dense_and_candidate_original_output_bits': t * n,
        'assumptions': [
            'A, factors, residual row norm coefficients and folded bias/threshold are static local coefficients.',
            'A one-read full-T input-stationary dense service is the baseline, not T redundant source rereads.',
            'Candidate writes all X before full-T certification; replay cannot be free future knowledge.',
            'Parallel fallback rereads X once for the failed-row set and holds its partial sums.',
            'Serial fallback uses one row accumulator and rereads X for each failed row.',
            'State numbers are allocated logical words for the selected 32-trajectory packet, not mapped SRAM.',
            'Classifier control, bounds comparisons, packing and source/cache scheduling have no cycle model here.',
            'No upstream operator, BN statistics, or first-pass FC1 product is removed by these counts.',
        ],
    }


def block_rf_probe(x, a, left, right, offset, theta, expected_certified, bound_mode):
    """Execute bounded 32-column blocks in one shared, banked 2-KiB RF.

    A full original X block stays until the failed-row mask drains. Only 16
    output tasks are live in MAC registers, so T additional full rows of PSN
    partial sums are NOT simultaneously allocated. Both the dense input-
    resident baseline and fallback use the same p-major compactable row-task
    engine. This is ordinary masked task compaction, not a claimed invention.

    Slot counts assume 16 one-cycle scalar MAC lanes, 16 RF banks each 1R1W,
    same-address broadcast, local static coefficient selection for every lane,
    and a separate comparison/compaction step. They are an explicit idealized
    block service, not RTL timing or a claim that such ports have zero area.
    RF words use float64 numerics in this CPU test; byte capacity assumes a
    future FP32 payload contract and does not establish IEEE-754 equivalence.
    """
    t, n = x.shape
    b, width, banks, capacity = 32, 16, 16, 512
    rank = right.shape[0]
    assert n % b == 0
    resid_sq = np.sum((a - left @ right) ** 2, axis=1)
    kappa, _, static_on, zero_rows = static_norm_constants(a, offset, theta)
    totals = {k: 0 for k in [
        'blocks', 'external_X_input_words_common', 'external_output_flag_bits_common',
        'external_X_spill_words_candidate', 'candidate_X_RF_writes', 'dense_X_RF_writes',
        'dense_X_RF_reads', 'dense_MAC_issue_slots', 'dense_compare_slots',
        'projection_X_RF_reads', 'projection_result_RF_writes', 'projection_MAC_issue_slots',
        'classifier_RF_reads', 'classifier_MAC_issue_slots', 'classifier_compare_compact_slots',
        'orthogonal_norm_RF_reads', 'orthogonal_norm_RF_writes',
        'orthogonal_norm_MAC_add_issue_slots', 'orthogonal_norm_clamp_slots',
        'fallback_X_RF_reads', 'fallback_MAC_issue_slots', 'fallback_compare_slots',
        'fallback_dispatch_slots_if_unoverlapped', 'fallback_task_count',
        'fallback_unused_MAC_lanes_over_T_steps', 'fallback_extra_read_slots_from_bank_conflicts',
        'numeric_output_bit_mismatches', 'certification_mask_mismatches',
    ]}
    packet_slot_ratios = []

    def read_cost(addresses):
        unique = set(int(v) for v in addresses)
        loads = [0] * banks
        for addr in unique:
            assert 0 <= addr < capacity
            loads[addr % banks] += 1
        return len(unique), max(loads, default=0)

    def original_rows(rf, jobs):
        values, reads, slots, extra, unused = [], 0, 0, 0, 0
        for start in range(0, len(jobs), width):
            batch = jobs[start:start + width]
            pp = np.array([j[0] for j in batch], dtype=int)
            rr = np.array([j[1] for j in batch], dtype=int)
            acc = offset[rr, 0] - theta
            for s in range(t):
                addresses = s * b + pp
                nr, beats = read_cost(addresses)
                reads += nr
                slots += beats
                extra += beats - 1
                acc += a[rr, s] * rf[addresses]
            values.extend((acc >= 0).tolist())
            unused += (width - len(batch)) * t
        return values, reads, slots, extra, unused

    for start in range(0, n, b):
        xp = x[:, start:start+b]
        rf = np.zeros(capacity, dtype=np.float64)
        input_words = t * b
        rf[:input_words] = xp.reshape(-1)
        all_jobs = [(p, out) for p in range(b) for out in range(t)]
        _, br, bs, _, _ = original_rows(rf, all_jobs)
        dense_compare = math.ceil(len(all_jobs) / width)
        dense_slots = bs + dense_compare
        totals['dense_X_RF_reads'] += br
        totals['dense_MAC_issue_slots'] += bs
        totals['dense_compare_slots'] += dense_compare

        # Factors and norm are computed in the same bounded dot-task engine.
        projection_jobs = [(p, j) for p in range(b) for j in range(rank + 1)]
        ps = pr = 0
        for first in range(0, len(projection_jobs), width):
            batch = projection_jobs[first:first+width]
            pp = np.array([j[0] for j in batch], dtype=int)
            jj = np.array([j[1] for j in batch], dtype=int)
            acc = np.zeros(len(batch))
            for s in range(t):
                addresses = s * b + pp
                nr, beats = read_cost(addresses)
                pr += nr
                ps += beats
                xx = rf[addresses]
                coefficients = np.array([xx[k] if j == rank else right[j, s]
                                         for k, j in enumerate(jj)])
                acc += coefficients * xx
            destinations = input_words + pp * (rank + 1) + jj
            rf[destinations] = acc
        totals['projection_X_RF_reads'] += pr
        totals['projection_result_RF_writes'] += len(projection_jobs)
        totals['projection_MAC_issue_slots'] += ps

        oslots = oclamp = 0
        if bound_mode == 'orthogonal_norm':
            for first in range(0, b, width):
                pp = np.arange(first, min(first + width, b))
                energy = np.zeros(len(pp))
                for j in range(rank):
                    addresses = input_words + pp * (rank + 1) + j
                    nr, beats = read_cost(addresses)
                    totals['orthogonal_norm_RF_reads'] += nr
                    oslots += beats
                    energy += rf[addresses] ** 2
                addresses = input_words + pp * (rank + 1) + rank
                nr, beats = read_cost(addresses)
                totals['orthogonal_norm_RF_reads'] += nr
                oslots += beats
                original_norm = rf[addresses].copy()
                guard = 1e-10 * original_norm
                rf[addresses] = np.maximum(original_norm - energy, 0) + guard
                # Two explicit add/subtract steps, one clamp, and any extra
                # bank write beat. The guard is a paid multiply above.
                oslots += 2 + max(0, beats - 1)
                oclamp += 1
                totals['orthogonal_norm_RF_writes'] += len(pp)
            totals['orthogonal_norm_MAC_add_issue_slots'] += oslots
            totals['orthogonal_norm_clamp_slots'] += oclamp

        bits = np.zeros((t, b), dtype=bool)
        certified = np.zeros((t, b), dtype=bool)
        cs = cr = cc = 0
        for first in range(0, len(all_jobs), width):
            batch = all_jobs[first:first+width]
            pp = np.array([j[0] for j in batch], dtype=int)
            rr = np.array([j[1] for j in batch], dtype=int)
            if bound_mode == 'static_norm':
                addresses = input_words + pp
                nr, beats = read_cost(addresses)
                cr += nr
                # Static sign and kappa selection; RF read feeds a comparator,
                # not a margin-square or row-norm multiplier.
                cc += beats
                decision = np.where(static_on[rr], rf[addresses] <= kappa[rr],
                                    rf[addresses] < kappa[rr]) | zero_rows[rr]
                bits[rr, pp] = static_on[rr]
                certified[rr, pp] = decision
                continue
            acc = offset[rr, 0] - theta
            for j in range(rank):
                addresses = input_words + pp * (rank + 1) + j
                nr, beats = read_cost(addresses)
                cr += nr
                cs += beats
                acc += left[rr, j] * rf[addresses]
            margin_squared = acc * acc
            cs += 1
            addresses = input_words + pp * (rank + 1) + rank
            nr, beats = read_cost(addresses)
            cr += nr
            cs += beats
            bound_squared = resid_sq[rr] * rf[addresses]
            decision = np.where(acc >= 0, margin_squared >= bound_squared,
                                margin_squared > bound_squared)
            cc += 1
            bits[rr, pp] = acc >= 0
            certified[rr, pp] = decision
        totals['classifier_RF_reads'] += cr
        totals['classifier_MAC_issue_slots'] += cs
        totals['classifier_compare_compact_slots'] += cc

        # Latent and norm lifetimes end here. X remains at its original address.
        failed_jobs = [(p, out) for p in range(b) for out in range(t) if not certified[out, p]]
        values, fr, fs, extra, unused = original_rows(rf, failed_jobs)
        for (p, out), value in zip(failed_jobs, values):
            bits[out, p] = value
        fc = math.ceil(len(failed_jobs) / width)
        totals['fallback_X_RF_reads'] += fr
        totals['fallback_MAC_issue_slots'] += fs
        totals['fallback_compare_slots'] += fc
        totals['fallback_dispatch_slots_if_unoverlapped'] += fc
        totals['fallback_task_count'] += len(failed_jobs)
        totals['fallback_unused_MAC_lanes_over_T_steps'] += unused
        totals['fallback_extra_read_slots_from_bank_conflicts'] += extra
        reference = a @ xp + offset >= theta
        totals['numeric_output_bit_mismatches'] += int(np.count_nonzero(reference != bits))
        totals['certification_mask_mismatches'] += int(np.count_nonzero(certified != expected_certified[:, start:start+b]))
        candidate_slots = ps + oslots + oclamp + cs + cc + fs + fc + fc
        packet_slot_ratios.append(candidate_slots / dense_slots)
        totals['blocks'] += 1
        totals['external_X_input_words_common'] += input_words
        totals['external_output_flag_bits_common'] += t * b
        totals['candidate_X_RF_writes'] += input_words
        totals['dense_X_RF_writes'] += input_words
    assert totals['numeric_output_bit_mismatches'] == 0
    assert totals['certification_mask_mismatches'] == 0
    base_slots = totals['dense_MAC_issue_slots'] + totals['dense_compare_slots']
    candidate_slots = sum(totals[k] for k in [
        'projection_MAC_issue_slots', 'classifier_MAC_issue_slots', 'classifier_compare_compact_slots',
        'orthogonal_norm_MAC_add_issue_slots', 'orthogonal_norm_clamp_slots',
        'fallback_MAC_issue_slots', 'fallback_compare_slots', 'fallback_dispatch_slots_if_unoverlapped'])
    mask_words = math.ceil(2 * b * t / 32)
    descriptor_words = math.ceil(2 * width * (5 + (t-1).bit_length()) / 32)
    candidate_rf_peak = b * t + b * (rank + 1) + mask_words + descriptor_words
    assert candidate_rf_peak <= capacity
    dense_stream_arithmetic_floor = math.ceil(n * t * t / width)
    return {
        'scope': 'EXPLICIT_FINITE_BLOCK_RF_SERVICE_MODEL_NOT_RTL_CYCLES',
        'resources': {'RF_capacity_words': capacity, 'assumed_word_bits': 32,
                      'RF_banks': banks, 'ports_per_bank': '1R1W', 'scalar_MAC_lanes': width,
                      'same_address_multicast': True,
                      'static_coefficients': 'Local indexed coefficients available to all lanes; physical fanout/mux cost unresolved.',
                      'MAC_register_words_common': width},
        'RF_lifecycle': {
            'X_words': b*t, 'latent_and_norm_words': b*(rank+1),
            'mask_words': mask_words, 'two_wave_descriptor_words': descriptor_words,
            'candidate_peak_reserved_RF_words': candidate_rf_peak,
            'candidate_peak_reserved_RF_bytes': 4*candidate_rf_peak,
            'candidate_bytes_if_X32_work64': 4*b*t + 8*b*(rank+1) + 4*(mask_words+descriptor_words),
            'candidate_bytes_if_all_numeric_words64': 8*(b*t+b*(rank+1)) + 4*(mask_words+descriptor_words),
            'dense_input_resident_RF_words_plus_output_flags': b*t + math.ceil(b*t/32),
            'dense_input_streaming_partial_words_plus_output_flags': b*t + math.ceil(b*t/32),
            'no_full_T_fallback_partial_sum_array': True,
            'external_X_spill_words': 0,
        },
        'totals': totals,
        'candidate_serial_phase_slots': candidate_slots,
        'dense_input_resident_serial_phase_slots': base_slots,
        'serial_phase_slot_ratio': candidate_slots / base_slots,
        'packet_serial_phase_ratio_quantiles': quantiles(packet_slot_ratios),
        'strong_dense_input_streaming_MAC_slot_lower_bound': dense_stream_arithmetic_floor,
        'candidate_MAC_issue_slots_only': sum(totals[k] for k in ['projection_MAC_issue_slots',
                                          'classifier_MAC_issue_slots', 'fallback_MAC_issue_slots',
                                          'orthogonal_norm_MAC_add_issue_slots']),
        'common_external_input_bytes_FP32': 4*n*t,
        'external_input_extra_bytes_candidate_vs_dense_input_resident': 0,
        'limits': [
            'Complete T input arrives before this block service; this does not remove BN-domain waiting.',
            'External input means the producer-to-PSN interface, not necessarily DRAM; no block spill is needed.',
            'The strong streaming A baseline can avoid the X RF save by keeping T partial sums instead.',
            'Input-resident dense A is measured here; streaming A is retained as a separate lower-bound comparator.',
            'Original row tasks and failed row tasks share the same ordinary p-major maskable engine.',
            'The service includes one comparison step per task wave and one unoverlapped failed-wave dispatch step.',
            'Cross-packet phase overlap can change latency and needs additional contexts; none is silently assumed.',
            'Word-level numeric model is float64; FP32 register capacity is an explicitly unvalidated deployment-width assumption.',
            'RF topology, broadcast and coefficient selection still require implementation costs; no PPA inference.',
        ],
    }


def run_layer(key, state, data):
    x, captured_bits, provenance = data
    name = MODULES[key]
    a = state[name + '.weight'].astype(np.float64)
    t, n = x.shape
    assert a.shape == (t, t) and captured_bits.shape == x.shape
    offset = (state[name + '.bias'] - state[name + '.center']).astype(np.float64)
    theta = float(state[name + '.thresh'])
    exact = a @ x + offset
    truth = exact >= theta
    u, singular, vh = np.linalg.svd(a, full_matrices=False)
    norm_sq = np.sum(x * x, axis=0)
    margin_abs = np.abs(exact - theta)
    rows = []
    settings = [(0, 'static_norm')]
    settings += [(r, mode) for r in ([1, 2] if t == 10 else [1])
                 for mode in ['full_norm', 'orthogonal_norm']]
    for rank, bound_mode in settings:
        left = u[:, :rank] * singular[:rank]
        right = vh[:rank]
        e = a - left @ right
        residual_sq = np.sum(e * e, axis=1)
        latent = right @ x
        approx = left @ latent + offset
        margin = approx - theta
        effective_norm = norm_sq
        if bound_mode == 'orthogonal_norm':
            effective_norm = np.maximum(norm_sq - np.sum(latent * latent, axis=0), 0) + 1e-10 * norm_sq
        bound_sq = residual_sq[:, None] * effective_norm
        # This is a float64 opportunity calculation of a real-arithmetic bound.
        # Strict exclusion on the inactive side preserves the >= firing rule.
        certified = np.where(margin >= 0, margin * margin >= bound_sq,
                             margin * margin > bound_sq)
        static_details = None
        if bound_mode == 'static_norm':
            kappa, kappa_fp32, static_on, zero_rows = static_norm_constants(a, offset, theta)
            certified = np.where(static_on[:, None], norm_sq <= kappa[:, None],
                                 norm_sq < kappa[:, None]) | zero_rows[:, None]
            x32 = x.astype(np.float32)
            norm32 = np.zeros(n, dtype=np.float32)
            for s in range(t):
                norm32 = (norm32 + (x32[s] * x32[s]).astype(np.float32)).astype(np.float32)
            fp32_cert = np.where(static_on[:, None], norm32 <= kappa_fp32[:, None],
                                 norm32 < kappa_fp32[:, None]) | zero_rows[:, None]
            static_details = {
                'kappa_float64': kappa.tolist(), 'kappa_guarded_downward_FP32': kappa_fp32.tolist(),
                'static_predicted_on': static_on.tolist(), 'zero_A_rows': zero_rows.tolist(),
                'runtime_output_multipliers': 0,
                'all_margins_negative': bool(np.all(~static_on)),
                'sorted_kappa_order': np.argsort(kappa).tolist(),
                'nested_certification_masks_when_all_margins_negative': t+1 if np.all(~static_on) else None,
                'ROM_mask_decode_not_implemented_as_new_mechanism': True,
                'guarded_FP32_sequential_norm_certified_fraction': float(fp32_cert.mean()),
                'guarded_FP32_certification_mask_changes_vs_float64': int(np.count_nonzero(fp32_cert != certified)),
                'guarded_FP32_certified_bit_errors_vs_dense_float64': int(np.count_nonzero(fp32_cert & (static_on[:, None] != truth))),
                'guarded_FP32_paid_cost': counted_cost(t, n, 0, fp32_cert, 'static_norm'),
                'FP32_scope': 'Conservative normal IEEE FP32 norm/dot sensitivity; not a TF32 or all GPU reduction-order proof.',
            }
        approx_bits = margin >= 0
        decoded = np.where(certified, approx_bits, truth)
        actual_residual = exact - approx
        oracle_radius = np.abs(actual_residual)
        oracle = np.where(margin >= 0, margin >= oracle_radius, -margin > oracle_radius)
        assert np.count_nonzero(decoded != truth) == 0
        assert np.count_nonzero(certified & (approx_bits != truth)) == 0
        row_stats = []
        for out in range(t):
            row_stats.append({
                'output_time': out, 'residual_row_L2': float(np.sqrt(residual_sq[out])),
                'certified_fraction': float(certified[out].mean()),
                'certified_off_fraction': float(np.mean(certified[out] & ~approx_bits[out])),
                'certified_on_fraction': float(np.mean(certified[out] & approx_bits[out])),
                'low_rank_only_bit_mismatch_fraction': float(np.mean(approx_bits[out] != truth[out])),
                'actual_margin_abs_quantiles': quantiles(margin_abs[out]),
                'approx_margin_abs_quantiles': quantiles(np.abs(margin[out])),
                'bound_radius_quantiles': quantiles(np.sqrt(bound_sq[out])),
            })
        cost = counted_cost(t, n, rank, certified, bound_mode)
        rows.append({
            'rank': rank, 'bound_mode': bound_mode, 'matrix_is_still_original_fullrank_A': True,
            'static_norm_control': static_details,
            'residual_times_top_right_basis_absmax': float(np.max(np.abs(e @ right.T))) if rank else 0.0,
            'orthogonal_norm_float64_guard': 'max(norm2-latent2,0)+1e-10*norm2; guard multiply/add paid' if bound_mode == 'orthogonal_norm' else None,
            'projected_energy_fraction_quantiles': quantiles(np.sum(latent * latent, axis=0) / np.maximum(norm_sq, np.finfo(float).tiny)),
            'certified_fraction': float(certified.mean()),
            'certified_output_bit_errors_vs_dense_float64': int(np.count_nonzero(decoded != truth)),
            'lossy_low_rank_only_new_model': {
                'products': 2 * rank * t * n,
                'product_ratio': 2 * rank / t,
                'bit_mismatch_count': int(np.count_nonzero(approx_bits != truth)),
                'bit_mismatch_fraction': float(np.mean(approx_bits != truth)),
                'not_frozen_model': True, 'AEE_unknown': True,
            },
            'optimistic_uncharged_actual_residual_magnitude_oracle': {
                'certified_fraction': float(oracle.mean()),
                'products_if_only_factor_compute_and_uncertified_A_rows_were_charged':
                    int(2 * rank * t * n + np.count_nonzero(~oracle) * t),
                'not_implementable_as_free_metadata': True,
                'explanation': 'Exact abs(E X) is obtained retrospectively from dense outputs; computing it is not free.',
            },
            'norm_bound_with_all_metadata_cost_omitted_diagnostic': {
                'products': int(2 * rank * t * n + np.count_nonzero(~certified) * t),
                'ratio': (2 * rank * t * n + np.count_nonzero(~certified) * t) / (t * t * n),
                'not_a_costed_result': True,
            },
            'paid_cost': cost, 'per_output_time': row_stats,
            'block_RF_model': block_rf_probe(x, a, left, right, offset, theta, certified, bound_mode),
        })
    return {
        'module': name, 'provenance': provenance, 'T': t, 'trajectories': n,
        'continuous_input_min': float(x.min()), 'continuous_input_max': float(x.max()),
        'source_values_exactly_representable_in_FP32': bool(np.array_equal(x, x.astype(np.float32).astype(np.float64))),
        'input_binary01_fraction': float(np.mean((x == 0) | (x == 1))),
        'input_exact_zero_fraction': float(np.mean(x == 0)),
        'threshold_amplitude': theta, 'reference_firing_fraction': float(truth.mean()),
        'dense_float64_vs_captured_output_bit_mismatches': int(np.count_nonzero(truth != captured_bits)),
        'actual_margin_abs_quantiles': quantiles(margin_abs),
        'actual_margin_abs_near_threshold_counts': {str(v): int(np.count_nonzero(margin_abs <= v))
                                                    for v in [1/128, 1/64, 1/32, 1/16, .25, .5, 1]},
        'variants': rows,
    }


def future_counterexample(state):
    name = MODULES['fc1_post']
    a = state[name + '.weight'].astype(float)
    offset = (state[name + '.bias'] - state[name + '.center']).astype(float).reshape(-1)
    theta = float(state[name + '.thresh'])
    row = int(np.argmax(np.abs(a[:, -1])))
    boundary = (theta - offset[row]) / a[row, -1]
    xs = [np.zeros(len(a)), np.zeros(len(a))]
    xs[0][-1] = boundary - 1
    xs[1][-1] = boundary + 1
    return {'module': name, 'row': row, 'identical_prefix_length': len(a) - 1,
            'x_last': [float(x[-1]) for x in xs],
            'output_value': [float(a[row] @ x + offset[row]) for x in xs],
            'output_bit': [bool(a[row] @ x + offset[row] >= theta) for x in xs],
            'meaning': 'Without a valid bound on unproduced continuous inputs, the shared prefix cannot certify this output.'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default=str(HERE / 'result.json'))
    ap.add_argument('--captures-only', action='store_true', help='Reuse prior FC1 result and add available FP32 hooks.')
    args = ap.parse_args()
    state = read_checkpoint(CKPT)['model_state_dict']
    activity = {r['name']: r for r in json.loads((CAP / 'atlif_activity.json').read_text())}
    output = Path(args.output)
    layers = []
    if args.captures_only and output.exists():
        layers = [r for r in json.loads(output.read_text())['layers']
                  if r['module'] == MODULES['fc1_post']]
    else:
        print('Reconstructing selected FC1 hidden channels over the complete current BN domain.', flush=True)
        layers.append(run_layer('fc1_post', state, reconstructed_fc1(state)))
    missing = []
    capture_root = HERE.parent / 'algorithm/probe_r1/capture'
    for key in ['head', 'sn_k']:
        paths = sorted(capture_root.glob('*/psn_' + key + '.npz'))
        if not paths:
            missing.append({'module': MODULES[key], 'expected_file': str(HERE / ('capture_' + key + '_sample0.npz')),
                            'requirement': 'sample0, 128 uniformly spaced contiguous packets of 32 flattened columns; complete T; x, output, indices, original_shape.'})
        for path in paths:
            layers.append(run_layer(key, state, hook_data(key, state, path)))
    result = {
        'status': 'COMPLETED_BOUNDED_FLOAT64_OPERATION_AND_STATE_PROBE',
        'matrix_inventory': matrix_inventory(state, activity), 'layers': layers,
        'missing_numerical_inputs': missing, 'future_input_counterexample': future_counterexample(state),
        'claim_boundary': [
            'No network training, AEE result, production RTL, EDA, energy, or cycle speedup.',
            'The paid Cauchy classifier uses full-T input norm; it is not an online leak/reset neuron.',
            'SVD, low-rank multiplication and norm inequalities are established methods, not claimed new.',
            'Any remaining architecture idea concerns exact threshold-consumer fallback and its state/traffic conflict.',
            'Float64 agreement on observed inputs is not a guarantee of the original GPU FP32 reduction order.',
            'Input/accumulator word counts assume FP32 payload width only for a capacity comparison.',
        ],
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    for layer in layers:
        print(layer['module'], 'N=', layer['trajectories'],
              'archive_bit_mismatch=', layer['dense_float64_vs_captured_output_bit_mismatches'])
        for variant in layer['variants']:
            cost = variant['paid_cost']
            print(' rank', variant['rank'], variant['bound_mode'], 'cert=', round(variant['certified_fraction'], 6),
                  'paid_products/dense=', round(cost['paid_product_ratio_vs_dense'], 6),
                  'any_fallback=', cost['trajectories_with_any_fallback'],
                  'extra_X_bytes_parallel=', cost['parallel_extra_X_write_plus_read_bytes_FP32'])
    print('missing_hooks=', len(missing), 'wrote', output, flush=True)


if __name__ == '__main__':
    main()
