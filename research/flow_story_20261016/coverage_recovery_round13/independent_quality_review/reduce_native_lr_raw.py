#!/usr/bin/env python3
"""Narrow post-result I/O adapter using the unchanged pre-result independent R13 metric math."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
R13 = HERE.parent
LR = R13.parent / 'native_lr_stability'
PINS = {
    HERE / 'reduce_raw_quality.py': '26a2e5c97b988f10685e1c440a11506354d9591cb9b36a567292abb6d0348088',
    HERE / 'actual_reduction_v1/independent_reduction.json': 'aaf5fae835475903b01f7ca4656884b1c1d3ba51de0c8fc03d72ddabd8f98a14',
    LR / 'EXECUTION_FREEZE_V1.json': '44b0101857918d3b655bf30fdf8f7d8735958e229d77196f1bc414c2d857a572',
    LR / 'PROTOCOL.md': '7f1b2a30c00584af4458a0bec00e0dc84b70cd6ec694c8cc0627e4ba0c368f34',
    LR / 'recover_v1/results.json': 'ca9892cde1ec23b6e45d9240b88fac983b3dd1e7dea0bbfba7a1f3a5c098607f',
    LR / 'smoke_v1/results.json': 'cafd496116f583695a9a9d49382377e8336cf96e6a633e8b392a06f6ab7498fa',
    R13 / 'input_snapshot/receipt.json': '2188cce3a09235dc5c96b793e94c025433b4154bd09ffd8ee15c2dcd9a3df8b6',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok:
        raise ValueError(message)


def reduce():
    for path, digest in PINS.items():
        require(sha(path) == digest, 'Pinned source changed: ' + str(path))
    for path, digest in read(LR / 'EXECUTION_FREEZE_V1.json')['source_sha256'].items():
        require(sha(path) == digest, 'LR execution freeze source changed: ' + path)
    spec = importlib.util.spec_from_file_location('frozen_independent_r13_math', HERE / 'reduce_raw_quality.py')
    math = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(math)
    member_path, region_path = R13 / 'data_protocol/manifest_v1.json', R13 / 'preparation/regions_v1.json'
    require(sha(member_path) == math.MANIFEST_SHA and sha(region_path) == math.REGION_SHA, 'Membership/region changed')
    member, region = read(member_path), read(region_path)
    ids = [r['target_id'] for r in member['check_members']]
    sequences = sorted(member['split_sequences']['check'])
    require(len(ids) == 8 and len(set(ids)) == 8 and len(sequences) == 2, 'Population changed')
    order = [r['target_id'] for r in member['train_order_diversity']]
    z = read(LR / 'recover_v1/results.json')
    smoke = read(LR / 'smoke_v1/results.json')
    old_path = R13 / 'raw/experiments_v1/diversity/recover/native/results.json'
    old = read(old_path)
    audit = read(HERE / 'actual_reduction_v1/independent_reduction.json')
    require(audit['status'] == 'PASS' and audit['input_results']['diversity/native']['result_sha256'] == sha(old_path), 'Old independent baseline not bound')
    require(z['binding']['baseline_result']['sha256'] == sha(old_path), 'Old result not bound by LR run')
    require(z['schema'] == 'flow_native_lr_stability_v1' and z['status'] == 'NATIVE_LR_RECOVERY_COMPLETE', 'Incomplete formal')
    require(z['arm'] == 'native' and z['condition'] == 'diversity' and z['mode'] == 'recover', 'Wrong formal arm')
    require(z['completed_optimizer_steps'] == len(z['steps']) == 200 and z['full_forward_count'] == len(z['forward_records']) == 228, 'Formal counters')
    require(smoke['completed_optimizer_steps'] == 5 and smoke['full_forward_count'] == 9 and smoke['smoke_weights_discarded'], 'Smoke counters/discard')
    require([s['step'] for s in z['steps']] == list(range(1, 201)) and [s['target_id'] for s in z['steps']] == order, 'Training order')
    require(z['binding']['fit_ids'] == order and z['binding']['check_ids'] == ids, 'Binding order')
    changes = {k for k in old['binding']['recipe'] if old['binding']['recipe'][k] != z['binding']['recipe'][k]}
    require(set(old['binding']['recipe']) == set(z['binding']['recipe']) and changes == {'lr'}, 'More than LR changed')
    require(old['binding']['recipe']['lr'] == 2.5e-4 and z['binding']['recipe']['lr'] == 2.5e-5, 'Wrong LR levels')
    for field in ('checkpoint_sha256', 'membership_sha256', 'region_sha256', 'runtime_manifest_sha256',
                  'BN_buffers_sha256', 'BN_receipt_sha256', 'loss_sha256', 'allowed_parameter_count'):
        require(z['binding'][field] == old['binding'][field], 'Shared source differs: ' + field)
    for field in ('frozen_state_unchanged', 'BN_buffers_unchanged', 'original_modules_restored', 'all_hooks_removed',
                  'paired_initial_state_exact', 'paired_initial_all_eight_native_predictions_exact'):
        require(z[field] is True, 'Execution closure: ' + field)
    require(z['initial_model_state_sha256'] == old['initial_model_state_sha256'], 'Initial state changed')
    require(z['initial_model_state_sha256'] == smoke['initial_model_state_sha256'], 'Formal is not fresh W0')
    require(z['final_state_sha256'] != z['initial_model_state_sha256'], 'No state update')
    cache, evidence = math.load_snapshot(R13 / 'input_snapshot', R13 / 'input_snapshot/receipt.json', ids)
    masks = {tid: math.masks_from_original(v['gt'], v['mask'], v['events'], region['boundary_threshold_pixels'],
              region['large_displacement_threshold_pixels']) for tid, v in cache.items()}
    numeric = math.NumericAudit()
    cells, predictions, npz_files = {}, {}, []
    for phase, modes in (('initial', ('native',)), ('endpoint', ('native', 'FF'))):
        frames = z['evaluations'][phase]['frames']
        require([f['target_id'] for f in frames] == ids, 'Evaluation frame order')
        for mode in modes:
            cells[phase + '/' + mode] = []
        for i, frame in enumerate(frames):
            tid = frame['target_id']
            require(frame['sequence'] == tid.rsplit('_', 1)[0] and set(frame['arms']) == set(modes), 'Evaluation identity/layouts')
            require(frame['input_evidence'] == evidence[tid], 'Source byte binding')
            name = frame['prediction_file']
            require(Path(name).name == name and name.endswith('.npz'), 'Unsafe prediction path')
            path = LR / 'recover_v1' / name
            require(sha(path) == frame['prediction_file_sha256'], 'Prediction archive bytes changed')
            npz_files.append({'file': str(path), 'sha256': sha(path)})
            with np.load(path, allow_pickle=False) as payload:
                require(set(payload.files) == set(modes), 'NPZ keys differ')
                for mode in modes:
                    pred = payload[mode]
                    require(pred.shape == (2, 480, 640) and pred.dtype == np.float32 and np.isfinite(pred).all(), 'Prediction contract')
                    psha = math.payload_sha(pred)
                    require(psha == frame['arms'][mode]['prediction_sha256'], 'Prediction payload bytes changed')
                    if phase == 'initial':
                        require(psha == old['evaluations']['initial']['frames'][i]['arms']['native']['prediction_sha256'], 'Initial payload differs from original R13')
                    key = phase + '/' + mode
                    predictions.setdefault(key, {})[tid] = psha
                    measured = math.measure(pred, cache[tid]['gt'], masks[tid])
                    reported = frame['arms'][mode]['metrics']
                    require(reported['target_id'] == tid and reported['sequence'] == frame['sequence'] and set(reported['regions']) == set(math.REGIONS), 'Reported identity/support')
                    for reg in math.REGIONS:
                        for field in ('valid_pixels', 'error_sum', 'AEE'):
                            numeric.check(measured[reg][field], reported['regions'][reg][field], '/'.join((key, tid, reg, field)))
                    cells[key].append({'target_id': tid, 'sequence': frame['sequence'], 'regions': measured})
    require(len(npz_files) == 16 and len(cells) == 3 and sum(len(x) for x in cells.values()) == 24, 'Final prediction counts')
    summaries = {key: math.summarize_frames(frames, sequences) for key, frames in cells.items()}
    for key, scopes in summaries.items():
        phase, mode = key.split('/')
        for scope, values in scopes.items():
            report = z['evaluations'][phase]['summaries'][mode]
            if scope != 'overall':
                report = report['per_sequence'][scope]
            require(report['frame_count'] == (8 if scope == 'overall' else 4), 'Scope count differs')
            for reg in math.REGIONS:
                for field, value in values[reg].items():
                    numeric.check(value, report['regions'][reg][field], '/'.join((key, scope, reg, field)))
    baseline = audit['conditions']['diversity']
    references = {key: baseline['summaries'][key] for key in ('initial/native', 'initial/FF', 'native/native', 'native/FF')}
    comparisons = {'native_vs_W0_native': ('endpoint/native', 'initial/native'),
                   'native_vs_highLR_native': ('endpoint/native', 'native/native'),
                   'FF_at_nativeW_vs_W0_FF': ('endpoint/FF', 'initial/FF'),
                   'FF_at_nativeW_vs_highLR_nativeW_FF': ('endpoint/FF', 'native/FF')}
    gates, all_region_deltas, frame_deltas = {}, {}, {}
    for name, (candidate, reference) in comparisons.items():
        gates[name] = math.compare(summaries[candidate], references[reference])
        all_region_deltas[name] = {}
        for scope in summaries[candidate]:
            all_region_deltas[name][scope] = {}
            for reg in math.REGIONS:
                x = summaries[candidate][scope][reg]['frame_mean_AEE']
                y = references[reference][scope][reg]['frame_mean_AEE']
                require((x is None) == (y is None), 'Paired region support differs')
                all_region_deltas[name][scope][reg] = {'candidate': x, 'reference': y,
                    'delta': x-y if x is not None else None, 'relative_change': x/y-1 if y not in (None, 0) else None}
        old_frames = baseline['all_endpoints'][reference]
        frame_deltas[name] = []
        for cur, previous in zip(cells[candidate], old_frames):
            require(cur['target_id'] == previous['target_id'], 'Baseline member order differs')
            x, y = cur['regions']['all_valid']['AEE'], previous['regions']['all_valid']['AEE']
            frame_deltas[name].append({'target_id': cur['target_id'], 'candidate': x, 'reference': y, 'delta': x-y})
    return {'schema': 'flow_native_lr_independent_raw_reduction_v1', 'status': 'PASS',
        'auditor_sha256': sha(__file__), 'pinned_sources': {str(k): v for k, v in PINS.items()},
        'result_sha256': sha(LR / 'recover_v1/results.json'), 'old_R13_result_sha256': sha(old_path),
        'shared_initial_state_and_eight_payloads_exact': True,
        'only_recipe_change': {'lr': [2.5e-4, 2.5e-5]}, 'source_array_sha256': evidence,
        'prediction_archives': npz_files, 'prediction_payload_sha256': predictions,
        'counts': {'source_arrays': 24, 'prediction_npz': 16, 'prediction_payloads': 24, 'state_layouts': 3,
                   'regions_per_frame': 16, 'frame_region_metrics': 384, 'scope_region_summaries': 144,
                   'formal_full_forwards': 228, 'formal_optimizer_updates': 200,
                   'smoke_plus_formal_full_forwards': 237, 'smoke_plus_formal_optimizer_updates': 205},
        'metric_reproduction': {'atol': math.ATOL, 'rtol': math.RTOL, 'numeric_comparisons': numeric.count,
            'maximum_absolute_difference': numeric.maximum_absolute, 'maximum_relative_difference': numeric.maximum_relative},
        'all_endpoints': cells, 'summaries': summaries, 'references_from_prior_independent_reduction': references,
        'comparisons': gates, 'all_region_deltas': all_region_deltas, 'all_frame_allGT_deltas': frame_deltas,
        'scope': 'Independent raw-array metric reproduction under the unchanged frozen R13 formula; I/O adaptation created after result completion. One native training arm only; terminal FF is the same native-trained weights under FF execution. Check8 is development. Tensor-update arithmetic and full execution audit are separate. No compressed-base, flow-specific increment or hardware-benefit claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refuse overwrite')
    result = reduce()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'counts': result['counts'],
        'metric_reproduction': result['metric_reproduction'],
        'comparisons': {k: v['pass'] for k, v in result['comparisons'].items()}}, indent=2))


if __name__ == '__main__':
    main()
