#!/usr/bin/env python3
"""Independent raw-array accounting for fixed same-W BN conversion, 16 forwards."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from analysis_core import METRIC, STEPS, analyze, require

HERE = Path(__file__).resolve().parent
DATE = HERE.parents[1]
R13 = DATE / 'coverage_recovery_round13'
K = DATE / 'later_checkpoint_deployment'
PURE_FREEZE_SHA = '706380812119252264e591cf8c9cde53ed207a936a843fe1786fdb40f540feeb'
CHECKPOINT_SHA = {'135930': 'd5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30',
                  '271482': '1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977'}
SOURCE_SHA = {'135930': '2bd43221e02d376df89e2b90ddce5a14f05301f45dd5ce871304b2a71b8c2831',
              '271482': 'ad14a7d45091aa75d64df83e1a71bd89b35dfbbf49c9ab84861d5ae0aa270560'}
CURRENT_SHA = {'135930': '8d4b5141731b0bed7bd12da67217ee5f68ad72288d03fd6c56726d8b6acd9ef5',
               '271482': 'b721031729a9a5f69d03037be53e62ebf8c1536dad0ad8a8f8e1601fe9fc2798'}
CACHE_SHA = {'135930': '254fb4ec5f18d30e4245b4c277ee1c5fb53beaf7fb3cf42cf60d0a4ee4390d0f',
             '271482': '8e0dafe172a27b6d56b484bada005ec18d15e62ae18afdc7b0bff7870ba1a841'}
CACHE_AUDIT_SHA = {'135930': 'eb77f69e87d80192157b1b3fd458289cdff56ea07a134bb1738076f9325d9a61',
                   '271482': '7455e073fcfb1ff981e228807cf917a4e4657a7ef988d072d7a16cdcbb77e9bd'}
CACHE_PATH = {'135930': R13 / 'raw/experiments_v1/diversity/recover/native/results.json',
              '271482': K / 'run_v1/results.json'}
REDUCTION_PATH = {'135930': R13 / 'independent_quality_review/actual_reduction_v1/independent_reduction.json',
                  '271482': K / 'independent_quality_review/actual_reduction_v1.json'}
REDUCTION_SHA = {'135930': 'aaf5fae835475903b01f7ca4656884b1c1d3ba51de0c8fc03d72ddabd8f98a14',
                 '271482': '44a0aea59db0fe52bd67eefd1eb2f89afa9e6111e7cd36d642c0d10493be6f23'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def safe_local_file(parent, name):
    require(isinstance(name, str) and bool(name) and Path(name).name == name and name not in ('.', '..'), 'Unsafe artifact name')
    return parent / name


def source_preflight():
    require(sha(HERE / 'freeze_v1.json') == PURE_FREEZE_SHA, 'Pure pre-result freeze changed')
    for manifest in ('freeze_v1.json', 'io_freeze_v1.json'):
        for name, digest in read(HERE / manifest)['files'].items():
            require(sha(HERE / name) == digest, 'Quality source changed: ' + name)
    pins = read(HERE / 'io_source_pins_v1.json')
    require(pins['schema'] == 'flow_round14_quality_io_pins_v1', 'Unknown source pins')
    for relative, digest in pins['sources'].items():
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts, 'Unsafe source pin')
        require(sha(DATE / path) == digest, 'Source pin changed: ' + relative)
    return pins


def bind_execution(result, audit, result_sha, pins, ids):
    require(audit['schema'] == 'flow_bn_conversion_round14_independent_execution_audit_v1'
            and audit['status'] == 'PASS', 'Actual independent execution audit required')
    require(audit['source_result_sha256'] == result_sha, 'Audit for another result')
    require(audit['auditor_sha256'] == pins['execution_auditor_sha256']
            and audit['audit_freeze_sha256'] == pins['execution_audit_freeze_sha256'], 'Unknown audit source/freeze')
    for key, value in {'full_forwards': 16, 'optimizer_updates': 0, 'calibration_forwards': 0,
                       'dual_forwards': 0, 'quality_prediction_count': 16, 'quality_archive_count': 16,
                       'executed_FC1_rows': 8928000, 'executed_FC2_rows': 8928000}.items():
        require(audit[key] == value, 'Execution audit count: ' + key)
    require(audit['BN_scope'] == result['binding']['BN_scope'] == 'ALL78_CURRENT', 'Wrong BN scope')
    for key in ('all_states_unchanged', 'all_BN_buffers_removed', 'all_hooks_removed', 'original_modules_restored'):
        require(audit[key] is True and result[key] is True, 'Execution closure: ' + key)
    require(set(audit['source_states']) == set(audit['references']) == set(result['checkpoints']) == set(STEPS), 'Exactly two fixed sources')
    for step in STEPS:
        actual, state = result['checkpoints'][step], audit['source_states'][step]
        require(state['checkpoint_sha256'] == actual['checkpoint_sha256'] == CHECKPOINT_SHA[step], 'Checkpoint bytes')
        require(state['source_state_sha256'] == actual['source_state_sha256'] == SOURCE_SHA[step], 'Original state')
        require(state['allcurrent_state_sha256'] == actual['initial_allcurrent_state_sha256']
                == actual['final_allcurrent_state_sha256'] == CURRENT_SHA[step], 'Same unchanged all-current state')
        require(state['state_tensors'] == 477 and state['state_elements'] == 54919238
                and state['removed_BN_buffers'] == 234 and state['state_unchanged'] is True
                and state['all78_current_verified'] is True, 'Actual all78current source inventory')
        require(actual['checkpoint_step'] == int(step) and actual['full_forward_count'] == 8, 'Per-source population')
        for key in ('state_unchanged', 'BN_buffers_unchanged', 'original_modules_restored', 'all_hooks_removed'):
            require(actual[key] is True, 'Per-source closure')
    predictions = audit['quality_predictions']
    expected = [(int(step), tid, 'native') for step in STEPS for tid in ids]
    require(len(predictions) == 16 and [(p['checkpoint_step'], p['target_id'], p['layout']) for p in predictions] == expected,
            'Audited exact ordered16 unique native predictions')
    return {(str(p['checkpoint_step']), p['target_id']): p['array_sha256'] for p in predictions}


def load_cached(step, binding_ref, audit_ref, ids, evidence, current_binding):
    path = CACHE_PATH[step]
    require(sha(path) == CACHE_SHA[step] and sha(REDUCTION_PATH[step]) == REDUCTION_SHA[step], 'Frozen historical result bytes')
    source, reduced = read(path), read(REDUCTION_PATH[step])
    require(reduced['status'] == 'PASS' and reduced['source_array_sha256'] == evidence, 'Historical independent raw population')
    for key in ('membership_sha256', 'runtime_manifest_sha256', 'region_sha256'):
        require(source['binding'][key] == current_binding[key], 'Paired input protocol: ' + key)
    phase = 'initial' if step == '135930' else 'endpoint'
    state = source['initial_model_state_sha256'] if step == '135930' else source['deployment_final_state_sha256']
    bn_sha = source['binding']['BN_buffers_sha256'] if step == '135930' else source['own_BN_buffers']['sha256']
    for key, value in {'cache_phase': phase, 'cache_result_sha256': CACHE_SHA[step],
                        'execution_audit_sha256': CACHE_AUDIT_SHA[step], 'cached_deployment_state_sha256': state,
                        'cached_BN_buffers_sha256': bn_sha, 'checkpoint_sha256': CHECKPOINT_SHA[step]}.items():
        require(binding_ref[key] == audit_ref[key] == value, 'Same-W cache binding: ' + key)
    require(binding_ref['allcurrent_state_sha256'] == CURRENT_SHA[step]
            and binding_ref['cache_not_loaded_as_model'] is True and binding_ref['no_quality_based_admission'] is True,
            'Cache is comparison only')
    frames = source['evaluations'][phase]['frames']
    require([f['target_id'] for f in frames] == ids, 'Cache member order')
    descriptors, payloads, artifacts = [], [], []
    for frame in frames:
        tid = frame['target_id']
        require(frame['input_evidence'] == evidence[tid], 'Cached input differs')
        descriptor = {k: frame[k] for k in ('target_id', 'sequence', 'input_evidence', 'prediction_file', 'prediction_file_sha256')}
        descriptor['prediction_sha256'] = frame['arms']['native']['prediction_sha256']
        descriptors.append(descriptor)
        archive = safe_local_file(path.parent, frame['prediction_file'])
        require(sha(archive) == frame['prediction_file_sha256'], 'Cached actual archive differs')
        with np.load(archive, allow_pickle=False) as values:
            pred = values['native']
            require(pred.shape == (2, 480, 640) and pred.dtype == np.float32 and np.isfinite(pred).all(), 'Cache shape')
            require(METRIC.payload_sha(pred) == descriptor['prediction_sha256'], 'Cached payload differs')
        payloads.append({'target_id': tid, 'sequence': frame['sequence'], 'array_sha256': descriptor['prediction_sha256']})
        artifacts.append({'path': str(archive.resolve()), 'sha256': sha(archive)})
    require(binding_ref['cached_native_predictions'] == descriptors and audit_ref['cached_native_predictions'] == payloads,
            'Actual cached native descriptors not independently bound')
    rows = reduced['conditions']['diversity']['all_endpoints']['initial/native'] if step == '135930' else reduced['all_endpoints']['later/native']
    return rows, descriptors, artifacts


def reduce(inputs, pins):
    require(inputs['schema'] == 'flow_round14_quality_inputs_v1', 'Unknown quality inputs')
    rp, ap = Path(inputs['result']['path']), Path(inputs['execution_audit']['path'])
    require(sha(rp) == inputs['result']['sha256'] and sha(ap) == inputs['execution_audit']['sha256'], 'Actual result/audit bytes')
    result, audit = read(rp), read(ap)
    require(result['schema'] == 'flow_bn_conversion_round14_v1' and result['status'] == 'ROUND14_BN_CONVERSION_COMPLETE', 'Complete fixed16 required')
    require(result['training_performed'] is False and result['optimizer_steps'] == result['calibration_forwards'] == result['dual_forwards'] == 0, 'Zero training/calibration/dual')
    require(result['full_forward_count'] == len(result['forward_records']) == 16, 'Wrong total forwards')
    mp, regp = R13 / 'data_protocol/manifest_v1.json', R13 / 'preparation/regions_v1.json'
    require(sha(mp) == METRIC.MANIFEST_SHA and sha(regp) == METRIC.REGION_SHA, 'Fixed population/regions')
    members, region = read(mp), read(regp)
    ids = [r['target_id'] for r in members['check_members']]
    seqs = sorted(members['split_sequences']['check'])
    require(len(ids) == len(set(ids)) == 8 and len(seqs) == 2
            and all(sum(r['sequence'] == s for r in members['check_members']) == 4 for s in seqs), 'Fixed two sequences/four each')
    binding = result['binding']
    require(binding['membership_sha256'] == METRIC.MANIFEST_SHA and binding['region_sha256'] == METRIC.REGION_SHA
            and binding['check_ids'] == ids, 'Runtime data binding')
    expected = [dict(forward_id=i*8+j+1, checkpoint_step=int(step), target_id=tid, layout='native', BN_scope='ALL78_CURRENT', mode='actual')
                for i, step in enumerate(STEPS) for j, tid in enumerate(ids)]
    require(binding['forward_plan'] == expected and binding['total_full_forwards'] == 16, 'Declared forward order')
    for item, plan in zip(result['forward_records'], expected):
        require(all(item[k] == v for k, v in plan.items()), 'Actual forward order')
        ex = item['execution']
        require(ex['layers'] == {} and ex['summed_fc1_rows'] == ex['summed_fc2_rows'] == 558000, 'Actual native full rows')
    for name, digest in binding['source_sha256'].items():
        require(sha(safe_local_file(HERE.parent, name)) == digest, 'Bound runner source differs')
    audited_payload = bind_execution(result, audit, sha(rp), pins, ids)
    cache, evidence = METRIC.load_snapshot(R13 / 'input_snapshot', R13 / 'input_snapshot/receipt.json', ids)
    masks = {tid: METRIC.masks_from_original(v['gt'], v['mask'], v['events'], region['boundary_threshold_pixels'],
                                             region['large_displacement_threshold_pixels']) for tid, v in cache.items()}
    cells, old_cells, archives, cached_archives, payloads = {}, {}, [], {}, {}
    numeric = METRIC.NumericAudit()
    for step in STEPS:
        old_cells[step], descriptors, cached_archives[step] = load_cached(step, binding['references'][step], audit['references'][step], ids, evidence, binding)
        phase = result['checkpoints'][step]['evaluations']['endpoint']
        require([f['target_id'] for f in phase['frames']] == ids, 'Current prediction member order')
        cells[step], payloads[step] = [], {}
        for frame, descriptor in zip(phase['frames'], descriptors):
            tid = frame['target_id']
            require(frame['sequence'] == tid.rsplit('_', 1)[0] and frame['input_evidence'] == evidence[tid], 'Actual frame/input binding')
            require(frame['paired_cached_native'] == descriptor and set(frame['arms']) == {'native'}, 'Same-W native pairing')
            archive = safe_local_file(rp.parent, frame['prediction_file'])
            require(archive.suffix == '.npz' and sha(archive) == frame['prediction_file_sha256'], 'Current artifact bytes')
            archives.append({'path': str(archive.resolve()), 'sha256': sha(archive)})
            with np.load(archive, allow_pickle=False) as arrays:
                require(arrays.files == ['native'], 'One native quality payload')
                pred = arrays['native']
                require(pred.shape == (2, 480, 640) and pred.dtype == np.float32 and np.isfinite(pred).all(), 'Current prediction geometry')
                digest = METRIC.payload_sha(pred)
                require(digest == frame['arms']['native']['prediction_sha256'] == audited_payload[(step, tid)], 'Actual audited payload')
                payloads[step][tid] = digest
                measured = METRIC.measure(pred, cache[tid]['gt'], masks[tid])
            reported = frame['arms']['native']['metrics']
            require(reported['target_id'] == tid and reported['sequence'] == frame['sequence']
                    and set(reported['regions']) == set(METRIC.REGIONS), 'Reported metric population')
            for region_name in METRIC.REGIONS:
                for field in ('valid_pixels', 'error_sum', 'AEE'):
                    numeric.check(measured[region_name][field], reported['regions'][region_name][field], '/'.join((step, tid, region_name, field)))
            cells[step].append({'target_id': tid, 'sequence': frame['sequence'], 'regions': measured})
        for scope, regions in METRIC.summarize_frames(cells[step], seqs).items():
            reported = phase['summaries']['native']
            if scope != 'overall':
                reported = reported['per_sequence'][scope]
            require(reported['frame_count'] == (8 if scope == 'overall' else 4), 'Summary member count')
            for region_name, values in regions.items():
                for field, value in values.items():
                    numeric.check(value, reported['regions'][region_name][field], '/'.join((step, scope, region_name, field)))
    require(len(set(a['path'] for a in archives)) == 16, 'Sixteen distinct actual quality archives')
    out = analyze(cells, old_cells, members['check_members'], True)
    out.update(status='PASS', raw_reduction_schema='flow_round14_independent_raw_quality_v1',
        source_result_sha256=sha(rp), source_execution_audit_sha256=sha(ap),
        source_array_sha256=evidence, prediction_archives=archives, cached_prediction_archives=cached_archives,
        prediction_payload_sha256=payloads, checkpoint_sha256=CHECKPOINT_SHA, historical_independent_reduction_sha256=REDUCTION_SHA,
        auditor_sha256=sha(__file__), pure_freeze_sha256=PURE_FREEZE_SHA,
        io_freeze_sha256=sha(HERE / 'io_freeze_v1.json'), io_source_pins_sha256=sha(HERE / 'io_source_pins_v1.json'),
        counts={'source_arrays': 24, 'new_prediction_payloads': 16, 'new_quality_npz': 16, 'cached_native_payloads_verified': 16,
                'fresh_frame_region_metrics': 256, 'fresh_scope_region_summaries': 96,
                'combined_states': 4, 'combined_frame_region_metrics': 512, 'combined_scope_region_summaries': 192,
                'formal_full_forwards': 16, 'optimizer_updates': 0, 'calibration_forwards': 0, 'dual_forwards': 0},
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
        print(json.dumps({'status': 'PASS', 'scope': 'Source SHA only; no new actual quality/array read', 'source_pins': len(pins['sources'])}))
        return
    require(args.inputs is not None and args.inputs_sha256 and args.output is not None, 'Explicit inputs/output required')
    require(sha(args.inputs) == args.inputs_sha256 and not args.output.exists(), 'Input SHA/fresh output required')
    out = reduce(read(args.inputs), pins)
    out['input_manifest_sha256'] = args.inputs_sha256
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(out, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': out['status'], 'counts': out['counts'], 'metric_reproduction': out['metric_reproduction'],
                      'next_step_scope': out['next_step_scope']}, indent=2))


if __name__ == '__main__':
    main()
