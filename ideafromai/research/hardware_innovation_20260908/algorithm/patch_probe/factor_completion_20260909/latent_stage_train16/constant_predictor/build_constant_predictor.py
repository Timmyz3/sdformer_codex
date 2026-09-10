"""Ordinary code0 predictor plus exact all-empty special case; no training.

The full integer U8/VQ5/Aq14 function is unchanged. Expanded tables exist
only for the existing adapter. Physical predictor payload is 960 fixed
positive/negative INT48 pairs, not 116*96 repeated pairs. Full thresholds
already belong to both students; they also produce the all-empty constants.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent
sys.path.insert(0, str(STAGE))
from compile_integer_factors import NAMES, predict, signed_width


def read_arrays(path):
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def convert(original):
    arrays = {key: value.copy() for key, value in original.items()}
    changed = []
    for t in range(10):
        for key in (f'integer_threshold_pos_t{t}', f'integer_threshold_neg_t{t}',
                    f'integer_predictor_mean_t{t}', f'integer_predictor_radius_t{t}'):
            value = arrays[key]
            value[:-1] = original[key][0]
            # Last code is all A-related source columns empty, not merely
            # some empty time columns. Preserve its original exact table.
            assert np.array_equal(value[-1], original[key][-1])
            changed.append(key)
        assert np.array_equal(arrays[f'integer_threshold_pos_t{t}'][-1],
                              arrays['integer_full_threshold'][t])
    assert all(np.array_equal(value, original[key]) for key, value in arrays.items() if key not in changed)
    pos = np.stack([arrays[f'integer_threshold_pos_t{t}'][0] for t in range(10)])
    neg = np.stack([arrays[f'integer_threshold_neg_t{t}'][0] for t in range(10)])
    threshold_bits = signed_width(min(pos.min(), neg.min()), max(pos.max(), neg.max()))
    expanded = [len(arrays[f'integer_threshold_pos_t{t}']) for t in range(10)]
    metadata = dict(
        kind='code0_constants_plus_exact_all_empty', entries_per_row=[1]*10,
        expanded_reference_entries_per_row=expanded,
        compiled_fixed_threshold_pairs=10*96,
        conservative_INT48_pair_bytes=10*96*2*6,
        actual_signed_threshold_bits=threshold_bits,
        bitpacked_actual_width_payload_bytes=(10*96*2*threshold_bits+7)//8,
        expanded_reference_INT48_bytes=sum(expanded)*96*2*6,
        expanded_reference_INT64_bytes=sum(expanded)*96*2*8,
        H8_pair_row_bytes=8*2*6, H8_pair_rows=10*12,
        physical_extra_all_empty_threshold_bytes=0,
        optional_all_empty_gate_bitmap_bytes=10*96//8,
        all_empty='AND source-empty over nonzero A row; emit (0>=existing_full_threshold); no new threshold pair',
        scope='Bare 48-bit constant threshold payload:11520B. H8 pair rows96B align to both64/256bit words. Ports, muxes, directory, tags and full thresholds are separate; expanded adapter rows are not physical storage.')
    arrays.update(integer_constant_predictor=np.asarray(True),
        integer_constant_threshold_pos=pos, integer_constant_threshold_neg=neg,
        integer_constant_all_empty_output_bits=np.packbits(
            (0 >= arrays['integer_full_threshold']).reshape(-1), bitorder='little'),
        integer_constant_all_empty_output_shape=np.asarray((10, 96), np.int32),
        integer_predictor_compact_metadata_json=np.asarray(json.dumps(metadata)),
        integer_predictor_policy=np.asarray(
            'No training: all partial-empty codes use original code0; all-empty retains the original exact constants. Positive wins ties. Full integer function unchanged; conditional decisions remain statistical.'),
        numeric_scope=np.asarray(str(original['numeric_scope'])+
            ' Ordinary constant predictor control: partial-empty patterns use original code0; exact all-empty retained. No recalibration, clipping or training.'))
    return arrays, metadata, changed


def compact_predict(arrays, shared_u, empty):
    pos = np.zeros_like(shared_u, dtype=bool)
    neg = np.zeros_like(pos)
    all_empty_rows = np.zeros(shared_u.shape[:3], dtype=bool)
    for t in range(10):
        indices = arrays[f'integer_empty_indices_t{t}']
        all_empty = empty[:, indices].all(1)
        all_empty_rows[:, t] = all_empty
        positive = np.where(all_empty[..., None], arrays[f'integer_threshold_pos_t{t}'][-1],
                            arrays['integer_constant_threshold_pos'][t])
        negative = np.where(all_empty[..., None], arrays[f'integer_threshold_neg_t{t}'][-1],
                            arrays['integer_constant_threshold_neg'][t])
        pos[:, t] = shared_u[:, t] >= positive
        neg[:, t] = shared_u[:, t] <= negative
    return pos, pos | neg, all_empty_rows


def need_z_from_accept(accepted, empty, a, v, shared):
    need_y = np.einsum('gtph,ts->gsph', (~accepted).astype(np.int32), (a != 0).astype(np.int32)) > 0
    need_y &= (~empty)[..., None]
    support = (v[shared:] != 0).reshape(-1, 2, 96).any(1)
    need_z = (need_y.astype(np.uint8) @ support.T.astype(np.uint8)) > 0
    return need_y, need_z


def requested(source, need, coefficient_nonzero):
    # Forty T/P positions: unsigned8 OR-count never exceeds40.
    src = source.reshape(len(source), 40, 864).astype(np.uint8)
    mask = need.reshape(len(source), 40, -1).transpose(0, 2, 1).astype(np.uint8)
    return ((mask @ src) != 0) & coefficient_nonzero[None]


def counters():
    return dict(gates=0, full_positive=0, accepted=0, false_positive_vs_full=0,
        false_negative_vs_full=0, wrong_gates_vs_full=0, U_shared_J2_requests=0,
        U_tail_J2_requests=0, U_full_J2_requests=0, needed_tail_Z_scalar_values=0,
        source_shared_live_words=0, source_tail_scan_live_words=0,
        source_tail_used_descriptors=0, all_empty_row_contexts=0,
        all_empty_gate_differences=0, no_tail_native_P4_groups=0,
        sparse_fallback_gate_differences=0)


def measure(original, arrays, source_files, chunk_groups=8):
    import torch
    from integer_adapter import IntegerLatentPair
    torch.set_num_threads(4)
    u = arrays['u_int8'].astype(np.int64)
    v = arrays['integer_v_align_coeff'].astype(np.int64)
    a = arrays['integer_a_q14'].astype(np.int64)
    shared = int(arrays['shared_rank'])
    unz = u.T.reshape(-1, 2, 864).any(1)
    policies = dict(original_pattern_table=counters(), constant_code0=counters())
    checks = dict(full_function_field_differences=0, full_gate_differences=0,
        compact_expanded_gate_differences=0, compact_expanded_accept_differences=0,
        integer_adapter_full_gate_differences=0, integer_adapter_conditional_gate_differences=0,
        integer_adapter_accept_differences=0)
    comparisons = dict(constant_vs_original_conditional_gate_differences=0,
        constant_vs_original_accept_differences=0)
    original_full = IntegerLatentPair(original, 'cpu', conditional=False)
    constant_full = IntegerLatentPair(arrays, 'cpu', conditional=False)
    constant_conditional = IntegerLatentPair(arrays, 'cpu', conditional=True)
    provenance = []
    with torch.no_grad():
        for source_file in source_files:
            with np.load(source_file) as data:
                words = data['source_gate_words'].astype(np.uint16)
                name, group_ids = str(data['file']), data['group_ids'].tolist()
            provenance.append(dict(path=str(source_file), frame=name, group_ids=group_ids))
            for first in range(0, len(words), chunk_groups):
                source = ((words[first:first+chunk_groups, :, :, None] >> np.arange(10)) & 1).transpose(0, 3, 2, 1).astype(bool)
                empty = ~source.any(-1)
                z = source.astype(np.int64) @ u
                shared_y = z[..., :shared] @ v[:shared]
                full_y = shared_y+z[..., shared:] @ v[shared:]
                full_u = np.einsum('ts,gsph->gtph', a, full_y)
                shared_u = np.einsum('ts,gsph->gtph', a, shared_y)
                full_gate = full_u >= arrays['integer_full_threshold'][None, :, None, :]
                old_full_gate = full_u >= original['integer_full_threshold'][None, :, None, :]
                checks['full_gate_differences'] += int(np.count_nonzero(full_gate != old_full_gate))
                live_source = source.any((1, 2))
                full_requests = live_source[:, None, :] & unz[None]
                shared_requests = full_requests[:, :shared//2]
                full_tensor = torch.from_numpy(full_y).double()
                shared_tensor = torch.from_numpy(shared_y).double()
                empty_tensor = torch.from_numpy(empty)
                for pair in (original_full, constant_full):
                    observed, _ = pair.forward_groups(full_tensor, shared_tensor, empty_tensor)
                    checks['integer_adapter_full_gate_differences'] += int(np.count_nonzero(observed.numpy() != full_gate))
                for policy, model in [('original_pattern_table', original), ('constant_code0', arrays)]:
                    positive, accepted = predict(model, shared_u, empty)
                    mixed = positive | (~accepted & full_gate)
                    cp, ca, all_empty = compact_predict(arrays, shared_u, empty)
                    if policy == 'original_pattern_table':
                        original_mixed, original_accepted = mixed, accepted
                    if policy == 'constant_code0':
                        comparisons['constant_vs_original_conditional_gate_differences'] += int(np.count_nonzero(mixed != original_mixed))
                        comparisons['constant_vs_original_accept_differences'] += int(np.count_nonzero(accepted != original_accepted))
                        checks['compact_expanded_gate_differences'] += int(np.count_nonzero(cp != positive))
                        checks['compact_expanded_accept_differences'] += int(np.count_nonzero(ca != accepted))
                        observed, actual_accept = constant_conditional.forward_groups(full_tensor, shared_tensor, empty_tensor)
                        checks['integer_adapter_conditional_gate_differences'] += int(np.count_nonzero(observed.numpy() != mixed))
                        checks['integer_adapter_accept_differences'] += int(np.count_nonzero(actual_accept.numpy() != accepted))
                    need_y, need_z = need_z_from_accept(accepted, empty, a, v, shared)
                    requests = requested(source, need_z, unz[shared//2:])
                    sparse_tail = z[..., shared:]*np.repeat(need_z, 2, -1)
                    sparse_y = shared_y+sparse_tail @ v[shared:]
                    sparse_u = np.einsum('ts,gsph->gtph', a, sparse_y)
                    sparse_gate = positive | (~accepted & (sparse_u >= arrays['integer_full_threshold'][None, :, None, :]))
                    tail_groups = need_z.any((1, 2, 3))
                    used_source = (source & need_z.any(-1)[..., None]).any((1, 2))
                    out = policies[policy]
                    out['gates'] += full_gate.size
                    out['full_positive'] += int(full_gate.sum())
                    out['accepted'] += int(accepted.sum())
                    out['false_positive_vs_full'] += int((mixed & ~full_gate).sum())
                    out['false_negative_vs_full'] += int((~mixed & full_gate).sum())
                    out['wrong_gates_vs_full'] += int((mixed != full_gate).sum())
                    out['U_shared_J2_requests'] += int(shared_requests.sum())
                    out['U_tail_J2_requests'] += int(requests.sum())
                    out['U_full_J2_requests'] += int(full_requests.sum())
                    out['needed_tail_Z_scalar_values'] += int(need_z.sum())*2
                    out['source_shared_live_words'] += int(live_source.sum())
                    out['source_tail_scan_live_words'] += int((live_source & tail_groups[:, None]).sum())
                    out['source_tail_used_descriptors'] += int(used_source.sum())
                    out['all_empty_row_contexts'] += int(all_empty.sum())
                    out['all_empty_gate_differences'] += int(((mixed != full_gate) & all_empty[..., None]).sum())
                    out['no_tail_native_P4_groups'] += int((~tail_groups).sum())
                    out['sparse_fallback_gate_differences'] += int((sparse_gate != mixed).sum())
            print('SOURCE', name, 'constant_wrong', policies['constant_code0']['wrong_gates_vs_full'], flush=True)
        # Actual captures need not contain an all-empty PSN row. Exercise that
        # specified interface separately, without adding it to workload counts.
        empty = np.ones((1, 10, 4), dtype=bool)
        zero_y = np.zeros((1, 10, 4, 96), dtype=np.int64)
        expected = np.broadcast_to(0 >= arrays['integer_full_threshold'][None, :, None, :], zero_y.shape)
        cp, ca, all_empty = compact_predict(arrays, zero_y, empty)
        ep, ea = predict(arrays, zero_y, empty)
        observed, actual_accept = constant_conditional.forward_groups(
            torch.from_numpy(zero_y).double(), torch.from_numpy(zero_y).double(), torch.from_numpy(empty))
        directed = dict(scope='Synthetic all-empty interface check; excluded from actual-source counts.',
            gates=zero_y.size, all_empty_row_contexts=int(all_empty.sum()),
            compact_gate_differences=int(np.count_nonzero(cp != expected)),
            expanded_gate_differences=int(np.count_nonzero(ep != expected)),
            integer_adapter_gate_differences=int(np.count_nonzero(observed.numpy() != expected)),
            not_accepted=int((~ca).sum()+(~ea).sum()+(~actual_accept.numpy()).sum()))
        if any(directed[key] for key in ('compact_gate_differences', 'expanded_gate_differences',
                                         'integer_adapter_gate_differences', 'not_accepted')):
            raise AssertionError(directed)
    for values in policies.values():
        values['gate_error_rate_vs_own_full'] = values['wrong_gates_vs_full']/values['gates']
        values['FN_rate_vs_own_full_positive'] = values['false_negative_vs_full']/max(values['full_positive'], 1)
        values['acceptance_rate'] = values['accepted']/values['gates']
        values['U_total_J2_requests'] = values['U_shared_J2_requests']+values['U_tail_J2_requests']
        values['U_request_ratio_vs_own_full'] = values['U_total_J2_requests']/values['U_full_J2_requests']
        values['U_requested_2xINT8_payload_bytes'] = values['U_total_J2_requests']*2
        if values['all_empty_gate_differences'] or values['sparse_fallback_gate_differences']:
            raise AssertionError(values)
    if any(checks.values()):
        raise AssertionError(checks)
    ratio = policies['constant_code0']['U_total_J2_requests']/policies['original_pattern_table']['U_total_J2_requests']
    return dict(source=provenance, policies=policies, checks=checks, comparisons=comparisons,
        directed_all_empty=directed,
        constant_vs_original_U_request_ratio=ratio,
        constant_minus_original_wrong_gates=policies['constant_code0']['wrong_gates_vs_full']-policies['original_pattern_table']['wrong_gates_vs_full'],
        counter_scope='Existing actual-source valid4, every saved64 P4 per frame, fullT10/K864/H96. U request=J2 coefficient pair, OR over actualT/P and U nonzero, no cycles. Tail used descriptors are distinct from full source-directory scan reads. No old Yi/gate capture used as this student input.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=STAGE/'integer_factors')
    parser.add_argument('--source', type=Path, default=STAGE.parent.parent/'partial_completion/integer_valid10')
    parser.add_argument('--output', type=Path, default=HERE)
    parser.add_argument('--generate-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(complete=False, training=False,
        scope='Ordinary fixed predictor control; full integer student unchanged. No GPU, AEE, bounded-tail model or hardware speed claim.', models={})
    models = {}
    for name in NAMES:
        original = read_arrays(args.input/(name+'.npz'))
        arrays, metadata, changed = convert(original)
        destination = args.output/(name+'.npz')
        np.savez_compressed(destination, **arrays)
        result['models'][name] = dict(input=str(args.input/(name+'.npz')), output=str(destination),
            compact_metadata=metadata, changed_predictor_fields=changed,
            full_function_parameters_unchanged=True)
        models[name] = (original, arrays)
        print('PARAMETERS', destination, flush=True)
    outfile = args.output/'result.json'
    outfile.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    if args.generate_only:
        return
    source_files = sorted(args.source.glob('capture_*.npz'))
    if not source_files:
        raise ValueError('No existing real source captures.')
    for name in NAMES:
        started = time.monotonic()
        result['models'][name]['local'] = measure(*models[name], source_files)
        result['models'][name]['CPU_wall_seconds'] = time.monotonic()-started
        outfile.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        print('RESULT', name, json.dumps(result['models'][name]['local']['policies']), flush=True)
    result['complete'] = True
    outfile.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
