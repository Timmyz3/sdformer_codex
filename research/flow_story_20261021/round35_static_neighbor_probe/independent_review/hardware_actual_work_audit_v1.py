#!/usr/bin/env python3
"""Independent R35 receipt audit. CPU only; no model or implementation validator.

The pixel/row golden comes from the independent prelaunch mask unpacking.
Kernel calls below are independently derived, not copied from validate_work.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(value, why):
    if not value:
        raise AssertionError(why)


def expected_calls(high, low, cout, reference=False):
    limit = (max(24, min(65536, (8 * 1024 * 1024) // cout)) // 8) * 8
    result = []
    for scope, count, bits in [('high8', high, range(8)), ('low4', low, range(4, 8))]:
        for start in range(0, count, limit):
            rows = min(limit, count - start)
            pad = max(24, ((rows + 7) // 8) * 8) - rows
            for bit in ([None] if reference else bits):
                result.append(dict(kind='dense_integer_reference' if reference else 'binary_weight_plane',
                    bit=bit, rows=rows, padding_rows=pad, scope=scope,
                    coefficient=None if reference else (-128 if bit == 7 else 1 << bit)))
    return result


def audit_work(work, golden, reference=False):
    check(set(work) == set(golden), 'all 28 target keys')
    summaries = {}
    for name, g in golden.items():
        w = work[name]
        for key in ('rows', 'tail_rows', 'restored_complete_rows', 'high8_rows', 'low4_rows',
                    'selected_complete_tiles', 'selected_spatial_mask_sha256', 'high8_rows_sha256'):
            check(w[key] == g[key], name + ': ' + key)
        check((w['Cin'], w['Cout'], w['geometric_shape']) == (g['K'], g['Cout'], [10, 1, g['H'], g['W']]), name + ': shape')
        check(w['complete_rows'] == g['rows'] - g['tail_rows'], name + ': complete rows')
        check(w['row_order'] == 'T,B,output_y,output_x' and w['mode'] == 'M', name + ': mode / canonical rows')
        check(w['reference'] == reference, name + ': reference scope')
        for key in ('binary_verified', 'int32_accumulation', 'spatial_rows_retained', 'full_T_and_channels',
                    'canonical_scatter_before_BN', 'independent_domain_execution'):
            check(w[key] is True, name + ': ' + key)
        check(w['cycle_claim'] is False and w['software_plane_dtype'] == 'int8', name + ': software scope')
        check(w['scale_variants'] == 2 and w['scale_applications'] == w['scale_applications_per_output'] == 1,
              name + ': separate scales, once per output')
        rep = w['representation']
        expected_rep = dict(variant='rne4', q8_shape=[g['Cout'], g['K']], quantization_device='cpu',
            rounding='ties_to_even', high_scale='per_output_maxabs/127', low_representation_generated=True,
            low_scale='per_output_maxabs/7/16', low_encoding='signed_q4_left_shift4',
            low_rule='independent_rne4_maxabs7_clamp_minus8_plus7', low_offset_per_active_input=0)
        for key, value in expected_rep.items():
            check(rep[key] == value, name + ': representation ' + key)
        for key in ('q8_sha256', 's8_sha256', 'q4_sha256', 's4_sha256', 'low_q_sha256', 'low_scale_sha256'):
            check(len(rep[key]) == 64 and all(c in '0123456789abcdef' for c in rep[key]), name + ': representation digest')
        check(w['midpoint_correction']['rows'] == w['midpoint_correction']['offset'] == 0, name + ': no midpoint correction')
        calls = expected_calls(g['high8_rows'], g['low4_rows'], g['Cout'], reference)
        check(w['kernel_calls'] == calls, name + ': exact domain/chunk/plane/coeff/padding sequence')
        check(w['actual_integer_kernel_calls'] == len(calls), name + ': actual call count')
        check(w['explicit_padded_rows'] == sum(k['padding_rows'] for k in calls), name + ': explicit padding total')
        check(w['logical_row_plane_work'] == (None if reference else g['logical_row_plane_work']), name + ': row planes')
        if not reference:
            for bit in range(8):
                check(sum(k['rows'] for k in calls if k['bit'] == bit) == g['row_planes'][str(bit)], name + ': per-plane accepted rows')
        summaries[name] = dict(actual_kernel_calls=len(calls), explicit_CUDA_padding_rows=sum(k['padding_rows'] for k in calls),
            high8_rows=g['high8_rows'], low4_rows=g['low4_rows'], tail_rows=g['tail_rows'],
            restored_complete_rows=g['restored_complete_rows'], accepted_bitplane128_core_beats=g['accepted_bitplane128_core_beats'])
    return summaries


def main():
    p = argparse.ArgumentParser()
    p.add_argument('actual', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    here = Path(__file__).resolve().parent
    r35 = here.parent
    gold = read(here / 'hardware_geometry_audit_v1.json')
    contract = read(r35 / 'ACTION_CONTRACT_V1.json')
    freeze = read(r35 / 'SOURCE_FREEZE_V1.json')
    for name, digest in freeze['files'].items():
        check(sha(r35 / name) == digest, 'unchanged frozen ' + name)
    result = read(a.actual / 'results.json')
    check(result['status'] == 'R35_STATIC360_COMPLETE', 'complete collector')
    check(result['runner_sha256'] == freeze['files']['collect_static.py'], 'runner binding')
    check(result['source_freeze_sha256'] == sha(r35 / 'SOURCE_FREEZE_V1.json'), 'source freeze binding')
    check(result['action_contract_sha256'] == sha(r35 / 'ACTION_CONTRACT_V1.json'), 'contract binding')
    check(result['membership_sha256'] == contract['membership_sha256'], 'membership binding')
    check(result['endpoint'] == contract['endpoint'] and result['final_state_sha256'] == contract['endpoint']['state_sha256'], 'endpoint retained')
    check((result['forward_attempts'], result['model_forwards'], result['formal_model_forwards'], result['engineering_model_forwards']) == (124, 124, 120, 4), 'actual 120 + 4')
    check(result['completed_optimizer_steps'] == 0 and result['training_performed'] is False and result['GT_arrays_read'] is False, 'no training/GT')
    check(result['wall_seconds'] < 1200, 'execution below 20 min')
    ev = result['evaluations']['static360']
    check(ev['GT_arrays_read'] is False and ev['state_sha256'] == contract['endpoint']['state_sha256'], 'evaluation scope')
    check(ev['mask_layout']['tile_ids_selection_order'] == contract['tile_ids_selection_order'], 'exact selection order')
    expected = gold['expected_affines']
    summaries = audit_work(ev['first_frame_full_work'], expected)
    first_sha = hashlib.sha256(json.dumps(ev['first_frame_full_work'], sort_keys=True).encode()).hexdigest()
    dense = result['parity']['static360_dense']
    check(dense['comparison']['pass'], 'full captured dense parity')
    check(dense['actual_record']['work'] == ev['first_frame_full_work'], 'first actual captured work')
    audit_work(dense['dense_record']['work'], expected, reference=True)
    for name, w in ev['first_frame_full_work'].items():
        check(w['representation'] == dense['dense_record']['work'][name]['representation'], name + ': dense same representations')
    check(len(ev['frames']) == ev['model_forwards'] == 120, '120 receipts')
    check([f['target_id'] for f in ev['frames']] == result['selected_ids'] == contract['target_ids'], 'exact identity order')
    roles = []
    prediction_files = []
    for frame in ev['frames']:
        tid = frame['target_id']
        check(frame['action'] == 'static360' and frame['work_sha256'] == first_sha, tid + ': immutable per-frame work digest')
        check(read(a.actual / 'records' / (tid + '.json')) == frame, tid + ': record bytes agree with result')
        check(set(frame['actual_affine_call_counts']) == set(expected) and set(frame['actual_affine_call_counts'].values()) == {1}, tid + ': 28 actual affine calls')
        for key, count in [('actual_BN_calls', 78), ('actual_PSN_calls', 105)]:
            check(len(frame[key]) == count and set(frame[key].values()) == {1}, tid + ': ' + key)
        check(frame['work_checks']['all28_logical_core_ticks'] == gold['all28_core_beats'], tid + ': implementation core work agrees with independent arithmetic')
        check(frame['work_checks']['added_four_patch_core_ticks_vs_Q4'] == gold['added_four_patch_core_beats'], tid + ': added work')
        path = a.actual / frame['prediction_file']
        check(sha(path) == frame['prediction_file_sha256'], tid + ': prediction file digest')
        z = np.load(path, allow_pickle=False)
        check(z.files == ['head3'], tid + ': one primary payload')
        pred = z['head3']
        check(pred.dtype == np.float32 and pred.shape == (2, 480, 640) and np.isfinite(pred).all(), tid + ': payload shape/finiteness')
        check(hashlib.sha256(pred.tobytes()).hexdigest() == frame['prediction_sha256'] == frame['heads']['head3']['sha256'], tid + ': actual captured output payload')
        roles.append(frame['role']); prediction_files.append(frame['prediction_file_sha256'])
    check(roles == result['roles'] and {role: roles.count(role) for role in set(roles)} == dict(fit=64, tune=16, confirm=40), 'roles unchanged')
    for role in ('fit', 'tune', 'confirm'):
        q = result['parity']['Q4_' + role]
        check(q['all_four_head_hashes_exact'] is True and all(q['heads'][n]['sha256'] == q['reference']['heads'][n]['sha256'] for n in q['heads']), role + ': Q4 four-head parity')
        check(q['work_checks']['all28_logical_core_ticks'] == gold['background_all28_core_beats'], role + ': Q4 original work')
    out = dict(schema='r35_independent_actual_hardware_work_v1', status='PASS',
        bindings={str(p): sha(p) for p in [a.actual / 'results.json', r35 / 'SOURCE_FREEZE_V1.json',
                   r35 / 'ACTION_CONTRACT_V1.json', here / 'hardware_geometry_audit_v1.json', Path(__file__)]},
        identities=120, actual_affine_receipts=3360, actual_formal_forwards=120, actual_engineering_forwards=4,
        first_actual_work_sha256=first_sha, per_target=summaries,
        all28_accepted_bitplane128_core_beats=gold['all28_core_beats'],
        unchanged_Q4_background_core_beats=gold['background_all28_core_beats'],
        added_four_patch_core_beats=gold['added_four_patch_core_beats'],
        four_isolated_conv_service_ticks=1032538860,
        exact_first_actual_dense_domain_calls=True, all120_matching_work_digests=True,
        all120_actual_prediction_payloads_verified=True, wall_seconds=result['wall_seconds'],
        limitations=['Repeated full work is independently retained only for the first formal identity; all subsequent identities bind the runtime-audited work SHA.',
                    'CUDA kernel padding is explicit software overhead, not a change to accepted hardware work.',
                    'Four isolated convolution services exclude shared stem and inter-operator BN/PSN, not network latency.',
                    'No independent numerical AEE evaluation is performed by this hardware audit.',
                    'No new RTL, synthesis, simulation grid, GPU forward, or model training performed by this auditor.'])
    path = a.output or here / 'hardware_actual_work_audit_v1.json'
    check(not path.exists(), 'new audit output only')
    path.write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k not in ('bindings', 'per_target')}, indent=2))


if __name__ == '__main__':
    main()
