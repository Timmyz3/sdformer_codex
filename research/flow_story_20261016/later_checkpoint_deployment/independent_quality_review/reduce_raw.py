#!/usr/bin/env python3
"""Independent CPU raw-array adapter for the fixed 193-forward deployment diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis_core import METRIC, analyze, require

HERE = Path(__file__).resolve().parent
DATE = HERE.parents[1]
R13 = DATE / 'coverage_recovery_round13'
PURE_FREEZE_SHA = '74552c7c9a705975087605d5608d19e2bacd0e57e466dc4fdad708bd320c1a20'
CHECKPOINT_SHA = '1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977'
OLD_REDUCTION_SHA = 'aaf5fae835475903b01f7ca4656884b1c1d3ba51de0c8fc03d72ddabd8f98a14'
LR_REDUCTION_SHA = '7e259f3ee38dceb6dde7f7d796336d705181935b069d783caf6a7dfe222635cf'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe_local_file(parent, name):
    require(isinstance(name, str) and Path(name).name == name and name not in ('.', '..'), 'Unsafe artifact filename')
    return parent / name


def source_preflight():
    require(sha(HERE / 'freeze_v1.json') == PURE_FREEZE_SHA, 'Pure pre-result freeze changed')
    for name, digest in read(HERE / 'freeze_v1.json')['files'].items():
        require(sha(HERE / name) == digest, 'Pure analysis source changed: ' + name)
    for name, digest in read(HERE / 'io_freeze_v1.json')['files'].items():
        require(sha(HERE / name) == digest, 'I/O source changed: ' + name)
    pins = read(HERE / 'io_source_pins_v1.json')
    require(pins['schema'] == 'flow_later_checkpoint_quality_io_pins_v1', 'Unknown source pins')
    for relative, digest in pins['sources'].items():
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe source pin')
        require(sha(DATE / path) == digest, 'Source pin changed: ' + relative)
    require(pins['checkpoint_sha256'] == CHECKPOINT_SHA, 'Wrong checkpoint source pin')
    return pins


def bind_execution(result, audit, result_sha, pins):
    require(audit['schema'] == 'flow_later_checkpoint_independent_execution_audit_v1'
            and audit['status'] == 'PASS', 'Independent execution audit incomplete')
    require(audit['source_result_sha256'] == result_sha, 'Execution audit is for another result')
    require(audit['auditor_sha256'] == pins['execution_auditor_sha256'], 'Unknown execution auditor')
    require(audit['pre_result_freeze_sha256'] == pins['execution_audit_freeze_sha256'], 'Execution audit source freeze differs')
    require(audit['checkpoint_sha256'] == result['binding']['checkpoint_sha256'] == CHECKPOINT_SHA, 'Checkpoint binding')
    expected = {'full_forwards': 193, 'optimizer_updates': 0, 'calibration_forwards': 176,
                'quality_prediction_count': 16, 'quality_archive_count': 8, 'dual_forward_count': 1}
    require(all(audit[k] == v for k, v in expected.items()), 'Independent execution counts')
    for name in ('source_state_independently_reconstructed', 'non_bn_weights_unchanged',
                 'BN_state_closure', 'actual_dual_raw_arrays_exact'):
        require(audit[name] is True, 'Independent execution closure: ' + name)
    for name, field in (('own_BN_buffers', 'BN_buffers_sha256'), ('own_BN_receipt', 'BN_receipt_sha256')):
        require(result[name]['sha256'] == audit[field], 'New BN is not bound to independent execution audit')


def reduce(inputs, pins):
    require(inputs['schema'] == 'flow_later_checkpoint_quality_inputs_v1', 'Unknown inputs')
    result_path = Path(inputs['result']['path'])
    audit_path = Path(inputs['execution_audit']['path'])
    require(sha(result_path) == inputs['result']['sha256'], 'Result bytes changed')
    require(sha(audit_path) == inputs['execution_audit']['sha256'], 'Audit bytes changed')
    result, audit = read(result_path), read(audit_path)
    require(result['schema'] == 'flow_later_checkpoint_deployment_v1'
            and result['status'] == 'LATER_CHECKPOINT_193_FORWARD_COMPLETE', 'Wait for complete actual result')
    bind_execution(result, audit, sha(result_path), pins)
    require(result['training_performed'] is False and result['optimizer_steps'] == 0, 'Deployment cannot train')
    require(result['full_forward_count'] == len(result['forward_records']) == 193, 'Wrong forward count')
    require(result['calibration_completed'] == 176, 'Incomplete calibration')
    require(result['deployment_initial_state_sha256'] == result['deployment_final_state_sha256'], 'Deployment state changed')
    for name in ('non_bn_state_unchanged', 'BN_buffers_unchanged', 'original_modules_restored', 'all_hooks_removed'):
        require(result[name] is True, 'Deployment closure: ' + name)
    member_path, region_path = R13 / 'data_protocol/manifest_v1.json', R13 / 'preparation/regions_v1.json'
    require(sha(member_path) == METRIC.MANIFEST_SHA and sha(region_path) == METRIC.REGION_SHA, 'Data/region protocol changed')
    member, region = read(member_path), read(region_path)
    ids = [r['target_id'] for r in member['check_members']]
    bn_ids = [r['target_id'] for r in member['bn_members']]
    sequences = sorted(member['split_sequences']['check'])
    require(len(ids) == len(set(ids)) == 8 and len(bn_ids) == len(set(bn_ids)) == 176, 'Population counts')
    require(len(sequences) == 2 and all(sum(r['sequence'] == s for r in member['check_members']) == 4 for s in sequences), 'Two sequences/four frames each')
    require(not set(ids) & set(bn_ids), 'Calibration/check identity overlap')
    binding = result['binding']
    require(binding['membership_sha256'] == METRIC.MANIFEST_SHA and binding['region_sha256'] == METRIC.REGION_SHA, 'Result input protocol binding')
    require(binding['check_ids'] == ids and binding['bn_ids'] == bn_ids, 'Result member order')
    require(binding['checkpoint_step'] == 271482 and binding['old_BN_loaded'] is False and binding['calibration_GT_used'] is False, 'Wrong calibration state contract')
    for name, digest in binding['source_sha256'].items():
        require(sha(safe_local_file(HERE.parent, name)) == digest, 'Runner bound source differs: ' + name)
    expected_records = [{'phase': 'calibration', 'target_id': tid, 'layout': 'native', 'mode': 'actual'} for tid in bn_ids]
    for i, tid in enumerate(ids):
        for layout in ('native', 'FF'):
            item = {'phase': 'evaluation', 'target_id': tid, 'layout': layout, 'mode': 'actual'}
            expected_records.append(item)
            if i == 0 and layout == 'FF':
                expected_records.append(dict(item, mode='dual_reference'))
    for idx, (actual, expected) in enumerate(zip(result['forward_records'], expected_records), 1):
        require(actual['forward_id'] == idx and all(actual[k] == v for k, v in expected.items()), 'Actual forward population/order')
    bn_path = safe_local_file(result_path.parent, result['own_BN_buffers']['path'])
    bn_receipt_path = safe_local_file(result_path.parent, result['own_BN_receipt']['path'])
    require(sha(bn_path) == result['own_BN_buffers']['sha256'] and sha(bn_receipt_path) == result['own_BN_receipt']['sha256'], 'Actual BN artifact bytes differ')
    bn = read(bn_receipt_path)
    require(bn['schema'] == 'flow_later_checkpoint_own_bn_v1' and bn['status'] == 'OWN_BN_COMPLETE', 'New BN receipt incomplete')
    require(bn['checkpoint_sha256'] == CHECKPOINT_SHA and bn['bn_target_ids'] == bn_ids and bn['buffers_sha256'] == sha(bn_path), 'New BN source identity')
    require(bn['calibration_forwards'] == 176 and bn['old_CMA_loaded'] is False and bn['GT_used'] is False and bn['optimizer_steps'] == 0, 'New BN provenance')
    require(bn['non_bn_state_before_sha256'] == bn['non_bn_state_after_sha256'], 'Calibration changed weights')
    with np.load(bn_path, allow_pickle=False) as arrays:
        require(len(arrays.files) == 72, 'Exactly72 private BN buffers')
        for key in arrays.files:
            value = arrays[key]
            require(np.isfinite(value).all(), 'Nonfinite BN')
            if key.endswith('.num_batches_tracked'):
                require(value.shape == () and value.dtype == np.int64 and int(value) == 176, 'BN counter')
            else:
                require(value.ndim == 1 and value.dtype == np.float32, 'BN array geometry')
                require(not key.endswith('.running_var') or (value >= 0).all(), 'Negative BN variance')
    cache, evidence = METRIC.load_snapshot(R13 / 'input_snapshot', R13 / 'input_snapshot/receipt.json', ids)
    masks = {tid: METRIC.masks_from_original(v['gt'], v['mask'], v['events'],
             region['boundary_threshold_pixels'], region['large_displacement_threshold_pixels']) for tid, v in cache.items()}
    frames = result['evaluations']['endpoint']['frames']
    require([f['target_id'] for f in frames] == ids, 'Prediction member order')
    cells, archives, payloads = {'native': [], 'FF': []}, [], {}
    numeric = METRIC.NumericAudit()
    for frame in frames:
        tid = frame['target_id']
        require(frame['sequence'] == tid.rsplit('_', 1)[0] and frame['input_evidence'] == evidence[tid], 'Frame/source binding')
        require(set(frame['arms']) == {'native', 'FF'}, 'Exactly two candidate layouts')
        npz = safe_local_file(result_path.parent, frame['prediction_file'])
        require(npz.suffix == '.npz' and sha(npz) == frame['prediction_file_sha256'], 'Prediction artifact bytes differ')
        archives.append({'path': str(npz.resolve()), 'sha256': sha(npz)})
        with np.load(npz, allow_pickle=False) as arrays:
            require(set(arrays.files) == {'native', 'FF'}, 'Quality NPZ keys')
            for layout in ('native', 'FF'):
                pred = arrays[layout]
                require(pred.shape == (2, 480, 640) and pred.dtype == np.float32 and np.isfinite(pred).all(), 'Prediction contract')
                pred_sha = METRIC.payload_sha(pred)
                require(pred_sha == frame['arms'][layout]['prediction_sha256'], 'Prediction payload differs')
                payloads.setdefault(layout, {})[tid] = pred_sha
                metrics = METRIC.measure(pred, cache[tid]['gt'], masks[tid])
                reported = frame['arms'][layout]['metrics']
                require(reported['target_id'] == tid and reported['sequence'] == frame['sequence']
                        and set(reported['regions']) == set(METRIC.REGIONS), 'Reported metric identity/support')
                for region_name in METRIC.REGIONS:
                    for field in ('valid_pixels', 'error_sum', 'AEE'):
                        numeric.check(metrics[region_name][field], reported['regions'][region_name][field],
                                      '/'.join((layout, tid, region_name, field)))
                cells[layout].append({'target_id': tid, 'sequence': frame['sequence'], 'regions': metrics})
    for layout, rows in cells.items():
        for scope, regions in METRIC.summarize_frames(rows, sequences).items():
            reported = result['evaluations']['endpoint']['summaries'][layout]
            if scope != 'overall':
                reported = reported['per_sequence'][scope]
            require(reported['frame_count'] == (8 if scope == 'overall' else 4), 'Summary frame count')
            for region_name, values in regions.items():
                for field, value in values.items():
                    numeric.check(value, reported['regions'][region_name][field], '/'.join((layout, scope, region_name, field)))
    old_path = R13 / 'independent_quality_review/actual_reduction_v1/independent_reduction.json'
    lr_path = R13 / 'independent_quality_review/native_lr_actual_v1/independent_reduction.json'
    require(sha(old_path) == OLD_REDUCTION_SHA and sha(lr_path) == LR_REDUCTION_SHA, 'Historical independent results changed')
    old, lr = read(old_path), read(lr_path)
    require(old['status'] == lr['status'] == 'PASS', 'Historical reductions incomplete')
    require(old['source_array_sha256'] == lr['source_array_sha256'] == evidence, 'Historical source bytes differ')
    references = {'original/native': old['conditions']['diversity']['all_endpoints']['initial/native'],
                  'original/FF': old['conditions']['diversity']['all_endpoints']['initial/FF'],
                  'smallLR/native': lr['all_endpoints']['endpoint/native'],
                  'smallLR/nativeW_FF': lr['all_endpoints']['endpoint/FF']}
    out = analyze(cells, references, member['check_members'], True)
    out.update(status='PASS', raw_reduction_schema='flow_later_checkpoint_independent_raw_quality_v1',
        source_result_sha256=sha(result_path), source_execution_audit_sha256=sha(audit_path),
        source_array_sha256=evidence, prediction_archives=archives, prediction_payload_sha256=payloads,
        checkpoint_sha256=CHECKPOINT_SHA, BN_buffers_sha256=sha(bn_path), BN_receipt_sha256=sha(bn_receipt_path),
        auditor_sha256=sha(__file__), pure_freeze_sha256=PURE_FREEZE_SHA,
        io_freeze_sha256=sha(HERE / 'io_freeze_v1.json'), io_source_pins_sha256=sha(HERE / 'io_source_pins_v1.json'),
        counts={'source_arrays': 24, 'prediction_payloads': 16, 'quality_prediction_npz': 8,
                'fresh_frame_region_metrics': 256, 'fresh_scope_region_summaries': 96,
                'all_state_layouts_including_previous': 6, 'formal_full_forwards': 193, 'optimizer_updates': 0},
        metric_reproduction={'atol': METRIC.ATOL, 'rtol': METRIC.RTOL, 'numeric_comparisons': numeric.count,
                            'maximum_absolute_difference': numeric.maximum_absolute,
                            'maximum_relative_difference': numeric.maximum_relative})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('source-preflight', 'reduce'), required=True)
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--inputs-sha256')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    pins = source_preflight()
    if args.stage == 'source-preflight':
        print(json.dumps({'status': 'PASS', 'scope': 'Source SHA only; no actual quality/array read', 'source_pins': len(pins['sources'])}))
        return
    require(args.inputs is not None and args.inputs_sha256 and args.output is not None, 'Explicit inputs/output required')
    require(sha(args.inputs) == args.inputs_sha256, 'Actual input manifest changed')
    require(not args.output.exists(), 'Refuse output overwrite')
    out = reduce(read(args.inputs), pins)
    out['input_manifest_sha256'] = args.inputs_sha256
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(out, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': out['status'], 'counts': out['counts'],
                     'metric_reproduction': out['metric_reproduction'], 'resource_decision': out['resource_decision']}, indent=2))


if __name__ == '__main__':
    main()
