#!/usr/bin/env python3
"""Independent CPU raw quality for the fixed selected16 BN native8 diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from analysis_core import METRIC, analyze, require

HERE = Path(__file__).resolve().parent
DATE = HERE.parents[1]
R13 = DATE / 'coverage_recovery_round13'
R14 = DATE / 'bn_conversion_round14'
K = DATE / 'later_checkpoint_deployment'
PURE_FREEZE_SHA = '2c0f71283d45ca0ccd29885c7a37ec8006037d1d2405f15dfc5d977487b4c915'
CHECKPOINT_SHA = '1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977'
SOURCE_SHA = 'ad14a7d45091aa75d64df83e1a71bd89b35dfbbf49c9ab84861d5ae0aa270560'
NON_BN_SHA = 'b721031729a9a5f69d03037be53e62ebf8c1536dad0ad8a8f8e1601fe9fc2798'
SELECTED_SHA = '6b4be9b8f136c4f978c892fff5532723828654f203c2d468e837a70ad0fd4451'
OWN_BN_SHA = '4d887a8fce408e3a33cb95bd120d2ed3d865541a39c034205f4172ab7bdd30dc'
OWN_RECEIPT_SHA = '24d33ba100150775a733fe6ecfabd4e88a1c152ca080adbf1f387a30ce574b9c'
BN_SCOPE = 'SELECTED16_FIXED_62_CURRENT'
CACHE_PATH = {'current78': R14 / 'run_v2/results.json', 'private24': K / 'run_v1/results.json'}
CACHE_SHA = {'current78': 'c9ead13e411512f570fc6ca926fcfaaa67c2f93e04c700bdd7799c85fcb98404',
             'private24': '8e0dafe172a27b6d56b484bada005ec18d15e62ae18afdc7b0bff7870ba1a841'}
CACHE_AUDIT_SHA = {'current78': '6bdcd565ccff753447f9becb7066f6670c7216841d644b174998a0a77960da71',
                   'private24': '7455e073fcfb1ff981e228807cf917a4e4657a7ef988d072d7a16cdcbb77e9bd'}
REDUCTION_PATH = {'current78': R14 / 'independent_quality_review/actual_reduction_v1.json',
                  'private24': K / 'independent_quality_review/actual_reduction_v1.json'}
REDUCTION_SHA = {'current78': 'd97d9ed7b5ea84c5c002d432aed6b5e50377291bba3033127df98ae28b5a2f41',
                 'private24': '44a0aea59db0fe52bd67eefd1eb2f89afa9e6111e7cd36d642c0d10493be6f23'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sibling(parent, name):
    require(isinstance(name, str) and name and Path(name).name == name and name not in ('.', '..'), 'Sibling file required')
    return parent / name


def source_preflight():
    require(sha(HERE / 'freeze_v1.json') == PURE_FREEZE_SHA, 'Pure source freeze changed')
    for manifest in ('freeze_v1.json', 'io_freeze_v1.json'):
        for name, digest in read(HERE / manifest)['files'].items():
            require(sha(HERE / name) == digest, 'Quality source changed: ' + name)
    pins = read(HERE / 'io_source_pins_v1.json')
    require(pins['schema'] == 'flow_selected16_quality_io_pins_v1', 'Unknown source pins')
    for relative, digest in pins['sources'].items():
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe source pin')
        require(sha(DATE / path) == digest, 'Source pin changed: ' + relative)
    return pins


def bind_execution(result, audit, result_sha, pins, ids):
    require(audit['schema'] == 'flow_selected16_round15_independent_execution_audit_v1' and audit['status'] == 'PASS', 'Actual execution audit required')
    require(audit['source_result_sha256'] == result_sha and audit['auditor_sha256'] == pins['execution_auditor_sha256']
            and audit['audit_freeze_sha256'] == pins['execution_audit_freeze_sha256'], 'Actual source/auditor freeze')
    counts = {'full_forwards': 8, 'optimizer_updates': 0, 'calibration_forwards': 0, 'dual_forwards': 0, 'FF_forwards': 0,
              'quality_prediction_count': 8, 'quality_archive_count': 8, 'retained_BN_archive_count': 1,
              'state_tensors': 525, 'state_elements': 54944214, 'fixed_BN_count': 16, 'current_BN_count': 62,
              'retained_BN_buffer_count': 48, 'executed_FC1_rows': 4464000, 'executed_FC2_rows': 4464000}
    for key, value in counts.items():
        require(audit[key] == value, 'Actual execution count: ' + key)
    require(set(result['checkpoints']) == {'271482'}, 'Only preselected original271482')
    item = result['checkpoints']['271482']
    require(audit['BN_scope'] == result['binding']['BN_scope'] == BN_SCOPE, 'Fixed16/current62 scope')
    require(audit['checkpoint_sha256'] == result['binding']['checkpoint_sha256'] == item['checkpoint_sha256'] == CHECKPOINT_SHA, 'Original checkpoint bytes')
    require(audit['source_state_sha256'] == item['source_state_sha256'] == SOURCE_SHA
            and audit['non_BN_state_sha256'] == NON_BN_SHA, 'Same original nonBN source')
    require(audit['selected16_state_sha256'] == item['initial_selected16_state_sha256']
            == item['final_selected16_state_sha256'] == SELECTED_SHA, 'Unchanged selected16 state')
    require(audit['retained_BN_values_match_own_source'] is True, 'Actual ownBN subset bytes')
    for key in ('all_states_unchanged', 'BN_buffers_unchanged', 'all_hooks_removed', 'original_modules_restored'):
        require(audit[key] is True and result[key] is True, 'Execution closure: ' + key)
    for key in ('state_unchanged', 'BN_buffers_unchanged', 'original_modules_restored', 'all_hooks_removed'):
        require(item[key] is True, 'Source state closure')
    require(item['checkpoint_step'] == 271482 and item['full_forward_count'] == 8, 'One native8 population')
    require(set(audit['references']) == {'private24', 'current78'}, 'Both fixed references')
    rows = audit['quality_predictions']
    require(len(rows) == 8 and [(p['checkpoint_step'], p['target_id'], p['layout']) for p in rows]
            == [(271482, tid, 'native') for tid in ids], 'Audited native8 identities/order')
    return {p['target_id']: p['array_sha256'] for p in rows}


def load_cached(label, binding_ref, audit_ref, ids, evidence, common):
    path = CACHE_PATH[label]
    require(sha(path) == CACHE_SHA[label] and sha(REDUCTION_PATH[label]) == REDUCTION_SHA[label], 'Historical source result bytes')
    source, reduced = read(path), read(REDUCTION_PATH[label])
    require(reduced['status'] == 'PASS' and reduced['source_array_sha256'] == evidence, 'Historical raw population')
    for key in ('membership_sha256', 'runtime_manifest_sha256', 'region_sha256'):
        require(source['binding'][key] == common[key], 'Paired input protocol: ' + key)
    if label == 'current78':
        case = source['checkpoints']['271482']
        state, bn_sha = case['final_allcurrent_state_sha256'], None
        frames = case['evaluations']['endpoint']['frames']
        rows = reduced['all_endpoints']['271482/current']
    else:
        state, bn_sha = source['deployment_final_state_sha256'], source['own_BN_buffers']['sha256']
        frames = source['evaluations']['endpoint']['frames']
        rows = reduced['all_endpoints']['later/native']
    expected = {'cache_result_sha256': CACHE_SHA[label], 'execution_audit_sha256': CACHE_AUDIT_SHA[label],
                'cache_phase': 'endpoint', 'checkpoint_sha256': CHECKPOINT_SHA,
                'cached_deployment_state_sha256': state, 'cached_BN_buffers_sha256': bn_sha}
    for key, value in expected.items():
        require(binding_ref[key] == audit_ref[key] == value, 'Same-W cache binding: ' + key)
    require(binding_ref['cache_not_loaded_as_model'] is True, 'Cache is comparison only')
    require([f['target_id'] for f in frames] == ids, 'Cached population/order')
    descriptors, audit_payloads, archives = [], [], []
    for frame in frames:
        tid = frame['target_id']
        require(frame['input_evidence'] == evidence[tid], 'Cached source input')
        descriptor = {k: frame[k] for k in ('target_id', 'sequence', 'input_evidence', 'prediction_file', 'prediction_file_sha256')}
        descriptor['prediction_sha256'] = frame['arms']['native']['prediction_sha256']
        archive = sibling(path.parent, frame['prediction_file'])
        require(sha(archive) == frame['prediction_file_sha256'], 'Cached archive bytes')
        with np.load(archive, allow_pickle=False) as arrays:
            pred = arrays['native']
            require(pred.shape == (2, 480, 640) and pred.dtype == np.float32 and np.isfinite(pred).all(), 'Cached prediction geometry')
            require(METRIC.payload_sha(pred) == descriptor['prediction_sha256'], 'Cached native payload')
        descriptors.append(descriptor)
        audit_payloads.append({'target_id': tid, 'sequence': frame['sequence'], 'array_sha256': descriptor['prediction_sha256']})
        archives.append({'path': str(archive.resolve()), 'sha256': sha(archive)})
    require(binding_ref['cached_native_predictions'] == descriptors and audit_ref['cached_native_predictions'] == audit_payloads, 'Cache independently bound')
    return rows, descriptors, archives


def verify_retained_BN(result, result_dir):
    chosen = result['retained_BN_buffers']
    path = sibling(result_dir, chosen['file'])
    require(sha(path) == chosen['sha256'] and chosen['source_own72_sha256'] == OWN_BN_SHA
            and chosen['source_own_receipt_sha256'] == OWN_RECEIPT_SHA, 'Retained BN actual artifact/source')
    require(result['binding']['own_BN_buffers_sha256'] == OWN_BN_SHA
            and result['binding']['own_BN_receipt_sha256'] == OWN_RECEIPT_SHA, 'Retained BN binding')
    original = K / 'run_v1/bn_buffers.npz'
    require(sha(original) == OWN_BN_SHA and sha(K / 'run_v1/bn_receipt.json') == OWN_RECEIPT_SHA, 'Actual original ownBN bytes')
    names = {f'sttmultires_unet.encoders.swin3d.layers.{s}.swin_blocks.{b}.mlp.bn{bn}.norm_layer'
             for s, depth in ((0, 2), (2, 6)) for b in range(depth) for bn in (1, 2)}
    keys = {n + '.' + suffix for n in names for suffix in ('running_mean', 'running_var', 'num_batches_tracked')}
    require(len(names) == 16 and len(keys) == 48 and chosen['keys'] == sorted(keys), 'Predetermined stage0two/stage2six BN subset')
    with np.load(path, allow_pickle=False) as selected, np.load(original, allow_pickle=False) as own:
        require(set(selected.files) == keys and len(own.files) == 72, '48 selected from own72')
        for key in keys:
            require(selected[key].dtype == own[key].dtype and selected[key].shape == own[key].shape
                    and np.array_equal(selected[key], own[key]) and np.isfinite(selected[key]).all(), 'Actual retained BN value differs')
    return {'path': str(path.resolve()), 'sha256': sha(path), 'source_own72_sha256': OWN_BN_SHA, 'retained_buffers': 48}


def reduce(inputs, pins):
    require(inputs['schema'] == 'flow_selected16_quality_inputs_v1', 'Unknown quality inputs')
    rp, ap = Path(inputs['result']['path']), Path(inputs['execution_audit']['path'])
    require(sha(rp) == inputs['result']['sha256'] and sha(ap) == inputs['execution_audit']['sha256'], 'Actual result/audit bytes')
    result, audit = read(rp), read(ap)
    require(result['schema'] == 'flow_selected16_bn_round15_v1' and result['status'] == 'ROUND15_SELECTED16_S8_COMPLETE', 'Complete fixedS8 required')
    require(result['training_performed'] is False and result['optimizer_steps'] == result['calibration_forwards'] == result['dual_forwards'] == 0, 'Zero training/calibration/dual')
    require(result['full_forward_count'] == len(result['forward_records']) == 8, 'Exactly8 full forwards')
    mp, regp = R13 / 'data_protocol/manifest_v1.json', R13 / 'preparation/regions_v1.json'
    require(sha(mp) == METRIC.MANIFEST_SHA and sha(regp) == METRIC.REGION_SHA, 'Fixed membership/regions')
    members, region = read(mp), read(regp)
    ids = [r['target_id'] for r in members['check_members']]
    sequences = sorted(members['split_sequences']['check'])
    require(len(ids) == len(set(ids)) == 8 and len(sequences) == 2
            and all(sum(r['sequence'] == s for r in members['check_members']) == 4 for s in sequences), 'Same two sequences/four each')
    binding = result['binding']
    require(binding['membership_sha256'] == METRIC.MANIFEST_SHA and binding['region_sha256'] == METRIC.REGION_SHA
            and binding['check_ids'] == ids and binding['checkpoint_step'] == 271482, 'Actual fixed input/W binding')
    expected = [dict(forward_id=i+1, checkpoint_step=271482, target_id=tid, layout='native', BN_scope=BN_SCOPE, mode='actual') for i, tid in enumerate(ids)]
    require(binding['forward_plan'] == expected and binding['total_full_forwards'] == 8, 'Fixed forward plan')
    for record, plan in zip(result['forward_records'], expected):
        require(all(record[k] == v for k, v in plan.items()), 'Actual native8 order')
        ex = record['execution']
        require(ex['layers'] == {} and ex['summed_fc1_rows'] == ex['summed_fc2_rows'] == 558000, 'Full native FC rows')
    for name, digest in binding['source_sha256'].items():
        require(sha(sibling(HERE.parent, name)) == digest, 'Runner source changed')
    audited = bind_execution(result, audit, sha(rp), pins, ids)
    bn_evidence = verify_retained_BN(result, rp.parent)
    cache, evidence = METRIC.load_snapshot(R13 / 'input_snapshot', R13 / 'input_snapshot/receipt.json', ids)
    masks = {tid: METRIC.masks_from_original(v['gt'], v['mask'], v['events'], region['boundary_threshold_pixels'],
                                             region['large_displacement_threshold_pixels']) for tid, v in cache.items()}
    references, descriptors, cached_archives = {}, {}, {}
    for label in ('current78', 'private24'):
        references[label], descriptors[label], cached_archives[label] = load_cached(label, binding['references'][label], audit['references'][label], ids, evidence, binding)
    phase = result['checkpoints']['271482']['evaluations']['endpoint']
    require([f['target_id'] for f in phase['frames']] == ids, 'New prediction order')
    cells, archives, payloads = [], [], {}
    numeric = METRIC.NumericAudit()
    for index, frame in enumerate(phase['frames']):
        tid = frame['target_id']
        require(frame['sequence'] == tid.rsplit('_', 1)[0] and frame['input_evidence'] == evidence[tid], 'Same raw input')
        require(frame['paired_cached_native'] == {label: descriptors[label][index] for label in descriptors}
                and set(frame['arms']) == {'native'}, 'Same-W paired cache native only')
        archive = sibling(rp.parent, frame['prediction_file'])
        require(archive.suffix == '.npz' and sha(archive) == frame['prediction_file_sha256'], 'New quality archive bytes')
        archives.append({'path': str(archive.resolve()), 'sha256': sha(archive)})
        with np.load(archive, allow_pickle=False) as arrays:
            require(arrays.files == ['native'], 'One native quality payload')
            pred = arrays['native']
            require(pred.shape == (2, 480, 640) and pred.dtype == np.float32 and np.isfinite(pred).all(), 'New prediction contract')
            digest = METRIC.payload_sha(pred)
            require(digest == audited[tid] == frame['arms']['native']['prediction_sha256'], 'Actual independently audited payload')
            payloads[tid] = digest
            measured = METRIC.measure(pred, cache[tid]['gt'], masks[tid])
        reported = frame['arms']['native']['metrics']
        require(reported['target_id'] == tid and reported['sequence'] == frame['sequence']
                and set(reported['regions']) == set(METRIC.REGIONS), 'Reported metric identity/regions')
        for region_name in METRIC.REGIONS:
            for field in ('valid_pixels', 'error_sum', 'AEE'):
                numeric.check(measured[region_name][field], reported['regions'][region_name][field], '/'.join((tid, region_name, field)))
        cells.append({'target_id': tid, 'sequence': frame['sequence'], 'regions': measured})
    require(len({a['path'] for a in archives}) == 8, 'Eight distinct native archives')
    for scope, regions in METRIC.summarize_frames(cells, sequences).items():
        reported = phase['summaries']['native']
        if scope != 'overall':
            reported = reported['per_sequence'][scope]
        require(reported['frame_count'] == (8 if scope == 'overall' else 4), 'Summary frame count')
        for region_name, values in regions.items():
            for field, value in values.items():
                numeric.check(value, reported['regions'][region_name][field], '/'.join((scope, region_name, field)))
    out = analyze(cells, references['current78'], references['private24'], members['check_members'], True)
    out.update(status='PASS', raw_reduction_schema='flow_selected16_independent_raw_quality_v1',
        source_result_sha256=sha(rp), source_execution_audit_sha256=sha(ap), source_array_sha256=evidence,
        prediction_archives=archives, prediction_payload_sha256=payloads, cached_prediction_archives=cached_archives,
        retained_BN_actual=bn_evidence, historical_independent_reduction_sha256=REDUCTION_SHA,
        auditor_sha256=sha(__file__), pure_freeze_sha256=PURE_FREEZE_SHA,
        io_freeze_sha256=sha(HERE / 'io_freeze_v1.json'), io_source_pins_sha256=sha(HERE / 'io_source_pins_v1.json'),
        counts={'source_arrays': 24, 'new_prediction_payloads': 8, 'new_quality_npz': 8, 'cached_native_payloads_verified': 16,
                'fresh_frame_region_metrics': 128, 'fresh_scope_region_summaries': 48, 'combined_states': 3,
                'combined_frame_region_metrics': 384, 'combined_scope_region_summaries': 144,
                'formal_full_forwards': 8, 'optimizer_updates': 0, 'calibration_forwards': 0, 'dual_forwards': 0},
        metric_reproduction={'atol': METRIC.ATOL, 'rtol': METRIC.RTOL, 'numeric_comparisons': numeric.count,
                            'maximum_absolute_difference': numeric.maximum_absolute, 'maximum_relative_difference': numeric.maximum_relative})
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
        print(json.dumps({'status': 'PASS', 'scope': 'Only source SHA; no new actual quality read', 'source_pins': len(pins['sources'])}))
        return
    require(args.inputs is not None and args.inputs_sha256 and args.output is not None, 'Explicit inputs/output')
    require(sha(args.inputs) == args.inputs_sha256 and not args.output.exists(), 'Input hash/fresh output')
    out = reduce(read(args.inputs), pins)
    out['input_manifest_sha256'] = args.inputs_sha256
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(out, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': out['status'], 'counts': out['counts'], 'metric_reproduction': out['metric_reproduction'], 'next_step_scope': out['next_step_scope']}, indent=2))


if __name__ == '__main__':
    main()
