"""CPU valid4 workload after real-flow recovery; saved training moments only.

Each supplied NPZ is measured against its own complete factor function.
No training, statistics fitting, GPU execution, or cycle conversion occurs.
U8/VQ5 files are dequantized FP32 functions, not integer datapaths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from latent_stage import (HERE, FACTOR, PARTIAL, SHARED, J, GAMMA,
    load_data, batch, read_model, measure, storage, completion)
from adapter import LatentTemporal


def arrays_from(path):
    with np.load(path) as data:
        return {key: data[key].copy() for key in data.files}


def scalar(arrays, key, default=None):
    return arrays[key].item() if key in arrays else default


@torch.no_grad()
def check_saved_function(model, data, constants, moments, arrays):
    """Same saved compact-table predicates and actual omitted-tail replay."""
    temporal = LatentTemporal(arrays)
    counts = dict(gates=0, compact_table_gate_differences=0,
        compact_table_accept_differences=0, missing_tail_gate_differences=0,
        predicted_margin_max_difference=0.)
    for first in range(0, len(data['y']), 8):
        item = batch(data, torch.arange(first, min(first+8, len(data['y']))),
            'cpu', float(arrays['theta_source']))
        state = completion(model, item, constants, moments)
        gate, details = temporal.forward_groups(state['y'], state['shared'],
            state['empty'], return_details=True)
        counts['gates'] += gate.numel()
        counts['compact_table_gate_differences'] += int((gate != state['gate']).sum())
        counts['compact_table_accept_differences'] += int((details['accepted'] != state['accept']).sum())
        counts['predicted_margin_max_difference'] = max(counts['predicted_margin_max_difference'],
            float((details['predicted']-state['predicted']).abs().max()))
        tail_z = state['z'][..., SHARED:]*state['need_z'].repeat_interleave(J, -1)
        raw = state['shared']+tail_z@(model.v*model.connectivity)[SHARED:]
        y = raw*constants['bn_scale']+constants['bn_bias']
        margin = torch.einsum('ts,gsph->gtph', constants['a'], y)
        margin += constants['b'][None, :, None, None]-constants['theta']
        actual = torch.where(state['accept'], state['predicted'].ge(0), margin.ge(0))
        counts['missing_tail_gate_differences'] += int((actual != state['gate']).sum())
    counts['scope'] = 'CPU FP32 all valid4 sampled P4, not CUDA/frozen FP32 bit equivalence or integer proof'
    return counts


def physical_format_notes(model, arrays, metrics, state):
    ubits, vbits = int(scalar(arrays, 'u_weight_bits', 32)), int(scalar(arrays, 'v_weight_bits', 32))
    # Keep original FP32-reference byte counts intact. Narrow fields below
    # are coefficient payload estimates, not port cycles or whole state.
    requested = metrics['U_shared_requests']+metrics['U_tail_requests']
    latent_count = model.u.shape[1]
    notes = dict(U_weight_bits=ubits, V_nonzero_weight_bits=vbits,
        U_J2_vector_payload_bits=J*ubits,
        U_requested_payload_bits=requested*J*ubits,
        U_full_requested_payload_bits=metrics['U_full_requests']*J*ubits,
        U_all_coefficients_payload_bits=model.u.numel()*ubits,
        V_nonzero_payload_bits=int((model.v*model.connectivity).ne(0).sum())*vbits,
        source_theta=float(arrays['theta_source']), output_theta=float(arrays['theta_output']),
        state_scope='Z, Y, margin and residual tables retain the existing FP32-reference widths; U8/VQ5 does not establish an integer accumulator format',
        payload_scope='Bare weight bits only. Packing, source/weight delivery, scales, structural V indices and register/SRAM ports are not converted to cycles.',
        illustrative_state_bytes=state['A_then_V_P4_H96_execution']['total_without_coefficients_tables_or_source_cache_bytes'])
    if ubits == 8:
        decoded = arrays['u_int8'].astype(np.float32)*arrays['u_dyadic_scale'][None, :]/float(arrays['theta_source'])
        notes.update(U_export_reconstruction_equal=bool(np.array_equal(decoded, arrays['u'])),
            U_per_latent_scale_entries=latent_count,
            U_export_scale_exponent_storage_bytes=int(arrays['u_scale_exponent'].nbytes),
            U_export_scale_float32_storage_bytes=int(arrays['u_dyadic_scale'].nbytes))
    if vbits == 5:
        decoded = np.ldexp(arrays['v_sign'].astype(np.float32), arrays['v_shift'])
        notes.update(V_export_reconstruction_equal=bool(np.array_equal(decoded, arrays['v'])),
            V_zero_encoding='Five-bit sign/exponent describes the nonzero Q5 values; exact zeros additionally require a bitmap/index or a wider fixed-slot representation.')
    return notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-files', nargs='+', type=Path, required=True,
        help='Final recovered NPZ files; one or all six, each keeps its own statistics.')
    parser.add_argument('--output', type=Path, default=HERE/'flow_recovery64/cpu_valid4')
    parser.add_argument('--source', type=Path, default=FACTOR.parent/'joint_completion_20260909/full_capture4/capture')
    parser.add_argument('--capture', type=Path, default=PARTIAL/'capture.pt')
    parser.add_argument('--valid-source', type=Path, default=PARTIAL/'integer_valid10')
    parser.add_argument('--operator', type=Path, default=PARTIAL/'shared_column_deployment_source.npz')
    args = parser.parse_args()
    args.temporal = 'common3'
    torch.set_num_threads(4)
    data, op, a, b, theta, yscale, mscale, rate, groups = load_data(args)
    constants = dict(a=a, b=b, theta=theta,
        bn_scale=torch.tensor(op['bn_scale'], dtype=torch.float32),
        bn_bias=torch.tensor(op['bn_bias'], dtype=torch.float32),
        y_scale=yscale, margin_scale=mscale, rate=rate)
    args.output.mkdir(parents=True, exist_ok=True)
    result = dict(complete=False,
        scope='Real-flow-recovered exported FP32/QDQ students; CPU local valid4 workloads only, no network AEE or cycle measurement',
        comparison='Every conditional path has its own complete factor-function denominator; all supplied axes reported',
        statistics='Read completion_mean/covariance from each NPZ. No local, validation or sampled-P4 recalibration.',
        capture=dict(files=data['valid']['files'], native_P4_groups_per_frame=len(groups),
            groups_total=len(data['valid']['y']), shape='G,T10,P4,H96',
            source='theta*g; K=((c*3)+kh)*3+kw, zero padding, complete T10 words',
            teacher='Same common3 A/b/theta on original unfactored norm1 Y; own-full and mixed errors remain separate',
            pairing=data['valid']['pairing']),
        arithmetic='U source-gated adds, continuous V and A operations, source words and J2 coefficient requests are separate units; no component ratios multiplied',
        axes={})
    output_file = args.output/'result.json'
    with torch.no_grad():
        for filename in args.model_files:
            started = time.monotonic()
            arrays = arrays_from(filename)
            if not bool(scalar(arrays, 'completion_statistics_valid', False)):
                raise ValueError(str(filename)+': saved completion statistics are not valid')
            if not np.array_equal(arrays['a'].astype(np.float32), a.numpy()):
                raise ValueError('Expected the same common3 A, not a different temporal student')
            if not np.array_equal(arrays['temporal_bias'].reshape(10).astype(np.float32), b.numpy()):
                raise ValueError('Temporal bias differs from the paired full-function teacher')
            if (float(arrays['theta_source']) != float(op['source_theta']) or
                    float(arrays['theta_output']) != theta):
                raise ValueError('The captured source/output theta amplitude must be retained')
            if int(arrays['shared_rank']) != SHARED or int(arrays['latent_tile']) != J or float(arrays['gamma']) != GAMMA:
                raise ValueError('This fixed reference expects shared32, native P4/J2 and gamma3')
            for key in ('bn_scale', 'bn_bias'):
                if not np.array_equal(arrays[key].astype(np.float32), constants[key].numpy()):
                    raise ValueError('The saved fixed BN affine differs: '+key)
            train_files = arrays.get('completion_train_files', np.array([])).tolist()
            if set(train_files) & set(data['valid']['files']):
                raise ValueError('The saved completion statistics include these validation filenames')
            moments = dict(mean=torch.from_numpy(arrays['completion_mean']).float(),
                covariance=torch.from_numpy(arrays['completion_covariance']).float())
            model = read_model(filename)
            metrics, snapshots = measure(model, data['valid'], constants, moments, float(arrays['theta_source']))
            metrics.update(own_full_positive=int(snapshots['full'].sum()),
                conditional_false_negatives_vs_own_full=int((snapshots['full'] & ~snapshots['mixed']).sum()),
                conditional_false_positives_vs_own_full=int((~snapshots['full'] & snapshots['mixed']).sum()))
            metrics['conditional_FN_rate_vs_own_full'] = metrics['conditional_false_negatives_vs_own_full']/max(metrics['own_full_positive'], 1)
            state = storage(model, a)
            checks = check_saved_function(model, data['valid'], constants, moments, arrays)
            name = filename.stem
            np.savez_compressed(args.output/(name+'_gates.npz'), **snapshots,
                files=np.array(data['valid']['files']), group_ids=groups)
            result['axes'][name] = dict(parameters=str(filename), structure=str(arrays['structure'].item()),
                initial_local_updates=scalar(arrays, 'initial_local_updates'),
                recovery_updates=scalar(arrays, 'recovery_updates'),
                total_updates_after_recovery=scalar(arrays, 'total_updates_after_recovery'),
                statistics=dict(train_files=train_files,
                    spatial_observations_per_time_channel=scalar(arrays, 'completion_spatial_observations'),
                    calibration_layout=scalar(arrays, 'completion_calibration_layout', 'trainer all-R U convolution; confirm against run metadata')),
                valid=metrics, storage=state, checks=checks,
                numeric_format=physical_format_notes(model, arrays, metrics, state),
                CPU_wall_seconds=time.monotonic()-started)
            output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
            print('MEASURE', name, json.dumps({key: metrics[key] for key in (
                'gates', 'full_gate_error', 'mixed_gate_error', 'mixed_FN_rate',
                'mixed_vs_own_full_rate', 'U_request_ratio', 'U_add_ratio', 'source_word_ratio')}), flush=True)
            if checks['compact_table_gate_differences'] or checks['compact_table_accept_differences'] or checks['missing_tail_gate_differences']:
                raise RuntimeError(name+': the saved reference and compact-table/omitted-tail checks differ; see result.json')
    result['complete'] = True
    output_file.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print('DONE', str(output_file), flush=True)


if __name__ == '__main__':
    main()
