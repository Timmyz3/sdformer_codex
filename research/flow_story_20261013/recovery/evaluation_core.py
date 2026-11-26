"""Candidate-specific BN calibration and declared train-side endpoint evaluation.

The caller supplies a source-verified factory that reconstructs one saved graph
and strictly loads its weights. This module never chooses a checkpoint, scores
channels, changes weights, or uses final-valid data.
"""
import copy
import hashlib
import io
from pathlib import Path
import time

BN_COLLECTOR_SHA = 'a364c6a08816a7df1c1eee4ce249f44ea9cf0631c172053c3e40eaf2df74b00f'
BN_SCHEDULE_SHA = '2f2b374eddf68d13dce6d6aead93218f9568f3c5143a59a25cd779a806289659'


def run_evaluation(a, context, factory, identity, populations, e, torch, functional):
    """Evaluate declared populations; calibrate only with the fixed BN176 inputs."""
    import numpy as np
    native, trainer, metrics, old_dev_rows, hashes, config, _, _, regions, plan = context
    e.require(regions is not None and a.geometry == 'full9', 'Complete fixed regions/full9 required')
    e.require(e.sha(a.breadth_root/'collect_bn_cma.py') == BN_COLLECTOR_SHA,
              'Frozen BN calibration primitives changed')
    cal = e.import_file('recovery_pinned_bn_collection', a.breadth_root/'collect_bn_cma.py')
    schedule = e.read_pin(a.breadth_root/'bn_calibration_schedule.json', BN_SCHEDULE_SHA)
    fit_ids = schedule['stages']['initial']['target_ids']
    all_rows = metrics.canonical_rows(a.canonical)
    roles = metrics.auxiliary_roles(a.roles, all_rows)
    rows = {r['target_id']: r for r in all_rows}
    old_ids = [r['target_id'] for r in old_dev_rows]
    e.require(len(fit_ids) == len(set(fit_ids)) == 176 and
              all(roles.get(t) == 'bn_calibration' for t in fit_ids), 'Wrong BN176 fit population')
    e.require(populations.get('gate_dev32') == old_ids and
              set(populations) <= {'gate_dev32', 'check64'}, 'Only explicit dev32/check64 may be evaluated')
    all_eval_ids = [t for values in populations.values() for t in values]
    e.require(len(all_eval_ids) == len(set(all_eval_ids)) and not set(all_eval_ids) & set(fit_ids),
              'Evaluation duplicates or BN fit overlap')
    for name, ids in populations.items():
        e.require(len(ids) == (32 if name == 'gate_dev32' else 64) and
                  all(roles.get(t) == 'gate_development' and rows[t]['protocol_split'] == 'train' for t in ids),
                  'Wrong train-side evaluation population')
    e.require(a.stage == 'endpoint' or 'check64' not in populations, 'Check64 may only be read at the frozen endpoint')
    device = torch.device(a.device)
    e.require(device.type == 'cuda' and torch.cuda.is_available(), 'Explicit GPU evaluator required')
    torch.set_num_threads(a.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.cuda.set_device(device)
    started = time.monotonic()
    result = {'schema': 'flow_recovery_candidate_evaluation_v1', 'status': 'RUNNING',
              'identity': identity, 'stage': a.stage, 'populations': populations,
              'native_preflight': plan, 'source_sha256': {'evaluation_core.py': e.sha(__file__),
                  'collect_bn_cma.py': BN_COLLECTOR_SHA},
              'geometry': 'full9', 'precision': 'FP32 TF32off noautocast',
              'scope': 'Train-side interim-backbone exploration, no final-valid, optimizer, checkpoint selection or hardware gains.',
              'calibration': {}, 'evaluations': {}, 'evaluation_input_evidence': {},
              'runtime': {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                          'device': str(device), 'GPU': torch.cuda.get_device_name(device)}}

    def deadline():
        e.require(time.monotonic()-started < a.wall_minutes*60, 'Bounded evaluation wall budget exhausted')

    def model_new():
        deadline()
        model = factory()
        model.to(device)
        functional.set_step_mode(model, 'm')
        model.eval()
        return model

    def input_only(target):
        rel = native.target_files(rows[target])['input']
        path = native.checked_path(a.data_root, rel)
        raw = path.read_bytes()
        e.require(len(raw) == hashes[rel]['bytes'] and hashlib.sha256(raw).hexdigest() == hashes[rel]['sha256'],
                  'Calibration input differs from accepted native data')
        events = np.load(io.BytesIO(raw), allow_pickle=False)
        e.require(events.shape == (10, 480, 640) and events.dtype == np.float32 and np.isfinite(events).all(),
                  'Invalid native calibration input')
        return e.prepare_input(events, device, torch), {'target_id': target, 'relative': rel, **hashes[rel]}

    model = model_new()
    bns = cal.reset_bn_cma(model, torch)
    before = cal.non_bn_digest(model, bns)
    evidence = []
    with torch.inference_mode():
        for index, target in enumerate(fit_ids):
            deadline(); chunk, record = input_only(target)
            functional.reset_net(model)
            output = model(chunk)['flow'][-1]
            e.require(bool(torch.isfinite(output).all()), 'Nonfinite candidate calibration output')
            evidence.append(record)
            del output, chunk
            if (index+1) % 32 == 0:
                print(__import__('json').dumps({'candidate_BN_calibration': index+1, 'of': 176}), flush=True)
    e.require(cal.non_bn_digest(model, bns) == before, 'Calibration changed non-BN state')
    buffers = {}
    for name, layer in bns.items():
        e.require(int(layer.num_batches_tracked) == 176 and bool(torch.isfinite(layer.running_mean).all()) and
                  bool(torch.isfinite(layer.running_var).all()) and bool((layer.running_var >= 0).all()),
                  'Incomplete/invalid candidate BN calibration')
        buffers[name] = {field: getattr(layer, field).detach().cpu().clone()
                         for field in ('running_mean', 'running_var', 'num_batches_tracked')}
    result['calibration'] = {'method': 'candidate_native_batch_CMA_reset', 'fit_ids': fit_ids,
        'fit_inputs': evidence, 'fit_uses_GT_values': False, 'non_BN_state_unchanged': True,
        'non_BN_state_sha256': before, 'count_per_BN': 176, 'BN_count': len(bns),
        'schedule_sha256': BN_SCHEDULE_SHA,
        'buffers': {n: {k: v.tolist() for k, v in d.items()} for n, d in buffers.items()},
        'scope': 'Candidate-own current-stat fitting pass; averages batch statistics, not pooled moments or fixed-point convergence.'}
    functional.reset_net(model); del model, bns
    torch.cuda.empty_cache(); e.write_result(a.output, result)

    for policy in ('P_paper', 'D_candidate_CMA176'):
        model = model_new()
        layers = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)}
        e.require(set(layers) == set(buffers) and cal.non_bn_digest(model, layers) == before,
                  'Fresh evaluation model differs from calibrated candidate graph/weights')
        if policy == 'D_candidate_CMA176':
            with torch.no_grad():
                for name, layer in layers.items():
                    for field, value in buffers[name].items():
                        current = getattr(layer, field)
                        e.require(current is not None and current.shape == value.shape and current.dtype == value.dtype,
                                  'Candidate BN buffer layout differs')
                        current.copy_(value.to(current.device))
        e.apply_bn(model, 'P_paper' if policy == 'P_paper' else 'D', torch)
        signature = e.bn_signature(model, torch)
        with torch.inference_mode():
            for population, ids in populations.items():
                label = policy+'__'+population
                meter = metrics.MetricAccumulator()
                result['evaluations'][label] = {'BN_policy': policy, 'population': population, 'status': 'RUNNING'}
                for index, target in enumerate(ids):
                    deadline(); row = rows[target]; expected = {}; file_evidence = {}
                    for kind, rel in native.target_files(row).items():
                        native.checked_path(a.data_root, rel)
                        expected['events' if kind == 'input' else kind] = hashes[rel]['sha256']
                    arrays, _ = metrics.load_arrays(a.data_root/'saved_flow_data', row,
                        evidence=file_evidence, expected_hashes=expected)
                    old = result['evaluation_input_evidence'].setdefault(target, file_evidence)
                    e.require(old == file_evidence, 'Evaluation input changed between BN policies')
                    masks = metrics.partitions(arrays['gt'], arrays['mask'], arrays['events'], regions)
                    chunk = e.prepare_input(arrays['events'], device, torch)
                    functional.reset_net(model); output = model(chunk)['flow'][-1]
                    e.require(tuple(output.shape) == (1, 2, 480, 640) and bool(torch.isfinite(output).all()),
                              'Invalid candidate flow prediction')
                    meter.update(target, row['sequence'], output[0].detach().cpu().numpy(), arrays['gt'], masks)
                    del output, chunk
                    if (index+1) % 16 == 0:
                        print(__import__('json').dumps({'evaluation': label, 'completed': index+1, 'of': len(ids)}), flush=True)
                result['evaluations'][label].update(status='COMPLETE', frames=meter.frames, summary=meter.summary())
                e.require(signature == e.bn_signature(model, torch), 'BN state mutated during fixed evaluation')
                e.write_result(a.output, result)
        functional.reset_net(model); del model, layers
        torch.cuda.empty_cache()
    result.update(status='CANDIDATE_EVALUATION_COMPLETE', wall_seconds=time.monotonic()-started,
                  candidate_BN_calibration_completed=True, all_evaluation_BN_states_unchanged=True)
    e.write_result(a.output, result)
    return result
