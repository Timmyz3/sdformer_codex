"""Frozen FC2 flags, full-T reduction reference; no RTL latency or PPA claim."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import numpy as np

BASE = Path(__file__).resolve().parent
PARSER = BASE.parent / 'mechanism_rebuild_gh_20260906/scripts'
sys.path.insert(0, str(PARSER))
from screen_threshold_packets import sources, EXPECTED, CKPT, digest
from checkpoint_numpy import read_checkpoint


def occupancy_metrics(occ, T):
    """Valid flags, not value-zero tests. Vectorized over all spatial positions."""
    work = occ.copy()
    need = work.astype(np.uint8)
    P = len(occ)
    emit = np.zeros(P, dtype=np.int64)
    fold = np.zeros(P, dtype=np.int64)
    peak = np.zeros(P, dtype=np.uint8)
    events = np.zeros(P, dtype=np.int64)
    for t in range(T):
        # Root q=1 at every height belongs to a disjoint nonzero-signature forest.
        peak = np.maximum(peak, need[:, 1])
        odd = work[:, 1::2].sum(1, dtype=np.int64)
        emit += np.maximum(odd-1, 0)
        events += odd
        if t == T-1:
            break
        left, right = work[:, ::2], work[:, 1::2]
        fold += (left[:, 1:] & right[:, 1:]).sum(1, dtype=np.int64)
        next_need = np.maximum(need[:, ::2], left.astype(np.uint8) + need[:, 1::2])
        work = left | right
        # Drop the unused parent value, never the lower-output work already performed.
        work[:, 0] = False
        next_need[:, 0] = 0
        need = next_need
    assert np.all(peak <= T)
    return emit, fold, peak, events


def materialized_values(groups, phi, T, prune):
    x = {q: phi[ids].sum(0) for q, ids in groups.items()}
    Y = np.zeros((T, phi.shape[1]), dtype=np.int64)
    emit = fold = 0
    for t in range(T):
        odd = [v for q, v in x.items() if q & 1]
        if odd:
            Y[t] = np.sum(odd, axis=0)
            emit += len(odd)-1
        if t == T-1:
            break
        new = {}
        for q, value in x.items():
            parent = q >> 1
            if prune and parent == 0:
                continue
            if parent in new:
                new[parent] += value
                fold += 1
            else:
                new[parent] = value.copy()
        x = new
    return Y, emit, fold


def streaming_values(groups, phi, T):
    """T disjoint postorder trees, exactly one source coefficient read per member.

    Working value ownership is counted explicitly. Output registers are separate.
    No materialized V_sigma array is used by this implementation.
    """
    present = [{q >> h for q in groups if q >> h} for h in range(T)]
    Y = np.zeros((T, phi.shape[1]), dtype=np.int64)
    out_valid = np.zeros(T, dtype=bool)
    c = Counter()
    live = 0

    def node(level, index):
        nonlocal live
        assert index > 0
        c['node_presence_queries'] += 1
        if index not in present[level]:
            return None
        c['nonempty_nodes_visited'] += 1
        if level == 0:
            ids = groups[index]
            assert len(ids)
            # Prepend-list producer: members are returned in reverse arrival order.
            value = phi[ids[0]].copy()
            live += 1
            c['working_vector_peak'] = max(c['working_vector_peak'], live)
            c['source_coefficient_reads'] += 1
            for cid in ids[1:]:
                value += phi[cid]
                c['source_coefficient_reads'] += 1
                c['build_adds'] += 1
        else:
            left = node(level-1, index*2)
            right = node(level-1, index*2+1)
            if left is None:
                value = right
            elif right is None:
                value = left
            else:
                left += right
                value = left
                live -= 1
                c['fold_adds'] += 1
        assert value is not None
        if index & 1:
            if out_valid[level]:
                Y[level] += value
                c['emit_adds'] += 1
            else:
                Y[level] = value
                out_valid[level] = True
            c['output_value_accepts'] += 1
        return value

    for level in range(T):
        value = node(level, 1)
        if value is not None:
            live -= 1
        assert live == 0
    assert c['working_vector_peak'] <= T
    c['destination_commits'] = T
    return Y, c


def groups_for(sig):
    groups = {}
    for cid, q in enumerate(sig):
        if q:
            groups.setdefault(int(q), []).append(cid)
    return {q: ids[::-1] for q, ids in groups.items()}


def diagnostic_phi(C, lanes=8):
    c, h = np.arange(C)[:, None], np.arange(lanes)[None, :]
    W = (((c+3)*(h+5)*13) % 256 - 128).astype(np.int64)
    W[:, 0], W[:, 1] = -128, 127
    theta = (999883 + 17*np.arange(C)).astype(np.int64)
    return W * theta[:, None]


def verify_values(sig, T, phi):
    groups = groups_for(sig)
    bits = ((np.asarray(sig)[None, :] >> np.arange(T)[:, None]) & 1).astype(np.int64)
    expected = bits @ phi
    live = np.asarray(sig) != 0
    pseudo = phi[live].sum(0)
    loas = np.zeros_like(expected)
    for t in range(T):
        if bits[t].any():
            loas[t] = pseudo - phi[live & (bits[t] == 0)].sum(0)
    flat = np.zeros_like(expected)
    for q, ids in groups.items():
        value = phi[ids].sum(0)
        for t in range(T):
            if q >> t & 1:
                flat[t] += value
    materialized, _, _ = materialized_values(groups, phi, T, False)
    pruned, e, f = materialized_values(groups, phi, T, True)
    streamed, counters = streaming_values(groups, phi, T)
    for other in (loas, flat, materialized, pruned, streamed):
        assert np.array_equal(other, expected)
    occ = np.zeros((1, 1 << T), dtype=bool)
    occ[0, list(groups)] = True
    emits, folds, peak, accepts = occupancy_metrics(occ, T)
    assert (int(emits[0]), int(folds[0])) == (e, f)
    assert counters['emit_adds'] == e and counters['fold_adds'] == f
    assert counters['working_vector_peak'] == int(peak[0])
    assert counters['output_value_accepts'] == int(accepts[0])
    assert counters['source_coefficient_reads'] == int(live.sum())
    counters['lane_outputs_verified_all_five_alternatives'] = 5*T*phi.shape[1]
    return dict(counters)


def directed():
    cases = {
        'all_silent': (3, [0]*16), 'one_active_source': (3, [0, 5, 0, 0]),
        'alternating_AB_members': (4, [10, 5]*8),
        'shared_110_111': (3, [6, 7]),
        'signed_cancellation': (3, [3, 3, 6, 7]),
        'same_bank_class_members': (3, [3 if c % 8 == 0 else 5 for c in range(40)]),
        'all_1023_nonzero_signatures': (10, list(range(1, 1024))),
    }
    result = {}
    for label, (T, sig) in cases.items():
        phi = diagnostic_phi(len(sig))
        if label == 'signed_cancellation':
            phi[1] = -phi[0]  # Structurally nonempty group has zero numeric value.
        result[label] = verify_values(np.array(sig, dtype=np.int64), T, phi)
    return result


def stats_layer(sig, spec, T, theta):
    P, C = sig.shape
    H = spec['output_channels']
    pop = np.array([q.bit_count() for q in range(1 << T)], dtype=np.int64)
    counts = np.zeros((P, 1 << T), dtype=np.uint16)
    # Full counts retain singleton and rare classes; no top-class cherry-picking.
    np.add.at(counts, (np.repeat(np.arange(P), C), sig.reshape(-1)), 1)
    counts[:, 0] = 0
    occ = counts != 0
    R = occ.sum(1, dtype=np.int64)
    U = counts.sum(1, dtype=np.int64)
    activity = np.stack([((sig >> t) & 1).sum(1, dtype=np.int64) for t in range(T)], axis=1)
    live_outputs = (activity != 0).sum(1, dtype=np.int64)
    direct = activity.sum(1) - live_outputs
    build = U-R
    scatter = occ @ pop - live_outputs
    emit, fold, peak, accepts = occupancy_metrics(occ, T)
    # Complete zero-aware RSR++ before pruning dead index-zero aggregate values.
    work = occ.copy()
    unpruned_emit = np.zeros(P, dtype=np.int64)
    unpruned_fold = np.zeros(P, dtype=np.int64)
    for t in range(T):
        unpruned_emit += np.maximum(work[:, 1::2].sum(1, dtype=np.int64)-1, 0)
        if t != T-1:
            left, right = work[:, ::2], work[:, 1::2]
            unpruned_fold += (left & right).sum(1, dtype=np.int64)
            work = left | right
    # Direct FTP is always present as a stronger denominator when correction is costly.
    correction_terms = np.where(activity > 0, U[:, None] - activity, 0)
    loas_pseudo = np.maximum(U-1, 0)
    loas_corr_build = np.maximum(correction_terms-1, 0).sum(1)
    loas_corr_join = (correction_terms > 0).sum(1)
    loas = loas_pseudo + loas_corr_build + loas_corr_join
    assert np.all(emit+fold <= scatter)
    # Source c modulo8 bank mapping: request-only optimistic bounds, not service cycles.
    bank = np.zeros((P, 1 << T, 8), dtype=np.uint16)
    for b in range(8):
        selected = sig[:, b::8]
        np.add.at(bank[:, :, b], (np.repeat(np.arange(P), selected.shape[1]), selected.reshape(-1)), 1)
    bank[:, 0, :] = 0
    direct_rounds = bank.sum(1).max(1)
    grouped_rounds = bank.max(2).sum(1)
    assert np.all(grouped_rounds >= direct_rounds)
    ptr = math.ceil(math.log2(C+1))
    slices = math.ceil(H/96)
    theta_array = np.asarray(theta)
    nonempty = R > 0
    aggregate = {
        'direct_FTP_binary_adds_per_output_channel': int(direct.sum()),
        'LoAS_adapted_binary_adds_per_output_channel': int(loas.sum()),
        'flat_build_binary_adds_per_output_channel': int(build.sum()),
        'flat_scatter_binary_adds_per_output_channel': int(scatter.sum()),
        'flat_total_binary_adds_per_output_channel': int((build+scatter).sum()),
        'RSRpp_materialized_emit_adds_per_output_channel': int(unpruned_emit.sum()),
        'RSRpp_materialized_fold_adds_per_output_channel': int(unpruned_fold.sum()),
        'RSRpp_materialized_total_adds_per_output_channel': int((build+unpruned_emit+unpruned_fold).sum()),
        'RSRpp_pruned_emit_adds_per_output_channel': int(emit.sum()),
        'RSRpp_pruned_fold_adds_per_output_channel': int(fold.sum()),
        'RSRpp_pruned_total_adds_per_output_channel': int((build+emit+fold).sum()),
        'source_coefficient_vector_reads_per_output_slice_all_variants': int(U.sum()),
        'original_firing_contribution_occurrences_per_output_channel': int(activity.sum()),
        'source_signature_inspections': int(P*C),
        'head_valid_bit_clears': int(P*(1 << T)),
        'head_valid_tests': int(U.sum()), 'head_valid_sets': int(R.sum()),
        'head_reads_during_prepend': int((U-R).sum()),
        'head_writes_during_prepend': int(U.sum()),
        'next_writes_during_prepend': int(U.sum()),
        'head_reads_during_consume_all_output_slices': int(R.sum()*slices),
        'next_reads_during_consume_all_output_slices': int(U.sum()*slices),
        'ascending_class_bitmap_positions_examined_all_output_slices': int(P*(1 << T)*slices),
        'direct_bank_request_round_lower_bound_per_output_slice': int(direct_rounds.sum()),
        'class_serial_bank_request_round_lower_bound_per_output_slice': int(grouped_rounds.sum()),
        'destination_commits_per_output_channel': int(P*T),
        'output_value_accepts_pruned_per_output_channel': int(accepts.sum()),
    }
    return {
        'module': spec['module_name'], 'input_shape': spec['input_shape'], 'P': P, 'C': C, 'H': H, 'T': T,
        'output_slices_of_96': slices,
        'source_threshold_shape': list(theta_array.shape),
        'source_threshold_min_max': [float(theta_array.min()), float(theta_array.max())],
        'threshold_note': 'Actual checkpoint sn2 threshold; normalization in stored flags does not erase its amplitude.',
        'R_mean': float(R.mean()), 'R_percentiles_0_25_50_75_90_99_100': np.percentile(R, [0,25,50,75,90,99,100]).tolist(),
        'R_histogram': {str(int(r)): int(n) for r,n in zip(*np.unique(R, return_counts=True))},
        'flat_extra_working_vectors': 1,
        'materialized_compact_R_vector_peak': int(R.max()),
        'streamed_extra_working_vector_peak': int(peak.max()),
        'streamed_extra_working_vector_mean_nonempty': float(peak[nonempty].mean()) if nonempty.any() else 0,
        'persistent_output_vectors_each_variant': T,
        'head_next_and_valid_minimum_bits_excluding_control_and_signatures': (1024+C)*ptr + 1024,
        'head_next_pointer_bits': ptr,
        'direct_full_signature_bits_without_compression': P*C*T,
        'member_table_capacity_bits_per_position': {'head': 1024*ptr, 'next': C*ptr, 'valid': 1024},
        'aggregate': aggregate,
        'ratios': {
            'flat_over_direct_adds': float((build+scatter).sum()/direct.sum()),
            'RSRpp_pruned_over_direct_adds': float((build+emit+fold).sum()/direct.sum()),
            'RSRpp_pruned_over_flat_adds': float((build+emit+fold).sum()/(build+scatter).sum()),
            'LoAS_adapted_over_direct_adds': float(loas.sum()/direct.sum()),
            'class_serial_over_direct_bank_request_round_lower_bounds': float(grouped_rounds.sum()/direct_rounds.sum()),
        },
        'signed_numeric_cancellation_is_not_skipped': True,
    }


def main():
    out = BASE / 'fc2_signatures_r1.json'
    assert not out.exists(), 'Do not overwrite the first FC2 receipt.'
    plan_path = BASE / 'fc2_signature_plan.json'
    plan = json.loads(plan_path.read_text())
    plan_sha = digest(plan_path)
    conformance = directed()
    assert digest(CKPT) == '4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48'
    state = read_checkpoint(CKPT)['model_state_dict']
    arrays = sources()
    layers = []
    print('Capture/checkpoint SHA and directed arithmetic checks passed.', flush=True)
    for stage in plan['stages']:
        pre = f'sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.0.mlp.'
        spec, S = arrays[pre+'fc2']
        T = plan['T']
        P, C = len(S)//T, S.shape[1]
        H = spec['output_channels']
        assert [P, C, H] == plan['expected_shapes_P_C_H'][str(stage)]
        assert spec['input_shape'][0] == T
        sig = np.tensordot(1 << np.arange(T, dtype=np.int64), S.reshape(T, P, C), axes=(0, 0))
        for t in range(T):
            assert np.array_equal(((sig >> t)&1).astype(np.uint8), S.reshape(T,P,C)[t])
        theta = state[pre+'sn2.spiking_neuron.thresh']
        W = state[pre+'fc2.weight']
        assert W.shape == (H, C) and np.isfinite(W).all() and np.isfinite(theta).all()
        result = stats_layer(sig, spec, T, theta)
        positions = np.unique(np.linspace(0, P-1, 64, dtype=np.int64))
        phi = diagnostic_phi(C)
        checks = [verify_values(sig[p], T, phi) for p in positions]
        result['diagnostics'] = {
            'spatial_positions': positions.tolist(), 'lanes': 8,
            'value_reference': 'Deterministic INT8 weights and non-1 rational theta, denominator1000000; not checkpoint weights.',
            'lane_outputs_verified_all_five_alternatives': sum(c['lane_outputs_verified_all_five_alternatives'] for c in checks),
            'mismatches': 0, 'per_position_counters': checks}
        result['source_flag_tensor_sha256'] = hashlib.sha256(S.tobytes(order='C')).hexdigest()
        result['complete_temporal_bit_reconstruction_mismatches'] = 0
        layers.append(result)
        print(json.dumps({k: result[k] for k in ('module','R_mean','materialized_compact_R_vector_peak','streamed_extra_working_vector_peak','ratios')}, ensure_ascii=False), flush=True)
    assert digest(plan_path) == plan_sha
    report = {'date': '2026-09-07', 'status': 'FIRST_TRUE_FC2_SIGNATURE_REFERENCE_EXECUTED',
              'plan': plan, 'plan_sha256': plan_sha, 'script_sha256': digest(Path(__file__)),
              'parser_sha256': digest(PARSER/'screen_threshold_packets.py'), 'capture_sha256': EXPECTED,
              'checkpoint_sha256': digest(CKPT), 'python_version': sys.version, 'numpy_version': np.__version__,
              'directed_checks': conformance, 'layers': layers,
              'claim_boundary': {'frozen_FP32_equivalence': False, 'new_AEE': False, 'rtl_cycles': False, 'ppa': False, 'whole_network': False},
              'limits': ['Inherited RSR++ computation and ordinary postorder scheduling; this receipt does not establish novelty.',
                         'The source producer order allowing classification overlap is not implemented; preprocessing is not credited as free.',
                         'Logical additions, narrow operations, and optimistic bank request rounds are different units and not summed as cycles.',
                         'The classifier layout is explicit and relatively large; SRAM word packing, ports, reset implementation and area remain open.',
                         'DFS value checks cover64 positions per stage; occupancy/arithmetic opportunity counts cover every selected spatial position.',
                         'Output slice replication/weight tiling, pipeline hazards and final dynamic BN service are not timed.',
                         'GustavSNN and Comperity full-text access gaps remain in the novelty review.']}
    with out.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print('Completed', str(out), flush=True)


if __name__ == '__main__':
    main()
