#!/usr/bin/env python3
"""Exploratory full-frame BN recalibration; no optimizer or final-valid input.

Resets BN running buffers and uses native PyTorch cumulative averages of each
batch's mean and unbiased variance. This is NOT pooled population variance or a
fixed-point calibration guarantee. All non-BN modules remain in eval mode.
"""
import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import evaluate_breadth as e

SCHEDULE_SHA = '2f2b374eddf68d13dce6d6aead93218f9568f3c5143a59a25cd779a806289659'


def reset_bn_cma(model, torch):
    model.eval()
    bns = {}
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
            e.require(layer.track_running_stats and layer.running_mean is not None,
                      'Calibration requires original tracked BN')
            layer.reset_running_stats()
            layer.momentum = None
            layer.train()
            bns[name] = layer
    e.require(bool(bns), 'No BN discovered')
    e.require(all(not m.training for m in model.modules()
                  if not isinstance(m, torch.nn.modules.batchnorm._BatchNorm)),
              'Non-BN training mode is prohibited')
    return bns


def non_bn_digest(model, bns):
    excluded = {name+'.'+field for name in bns
                for field in ('running_mean', 'running_var', 'num_batches_tracked')}
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if name in excluded:
            continue
        value = tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
    return h.hexdigest()


def collect(a, context, schedule_path):
    native, trainer, metrics, eval_rows, hashes, config, run, side, _, preflight = context
    schedule = e.read_pin(schedule_path, SCHEDULE_SHA)
    fit_ids = schedule['stages']['initial']['target_ids']
    canonical = metrics.canonical_rows(a.canonical)
    roles = metrics.auxiliary_roles(a.roles, canonical)
    e.require(len(fit_ids) == len(set(fit_ids)) == 176 and
              all(roles.get(x) == 'bn_calibration' for x in fit_ids) and
              not set(fit_ids) & {r['target_id'] for r in eval_rows},
              'Calibration schedule/role/disjointness differs')
    rows = {r['target_id']: r for r in canonical}
    e.require(all(rows[x]['protocol_split'] == 'train' for x in fit_ids),
              'Final-valid prohibited')
    result = {'schema': 'flow_breadth_bn_calibration_v1', 'status': 'PREFLIGHT_ONLY',
              'method': 'cma_native_batch_stats_reset', 'checkpoint_sha256': a.checkpoint_sha256,
              'data_contract_sha256': e.digest(config['data_contract']),
              'source_sha256': preflight['source_sha256'], 'geometry': a.geometry,
              'fit_target_ids': fit_ids, 'fit_target_ids_sha256': e.digest(fit_ids),
              'fit_role': 'bn_calibration', 'schedule_sha256': SCHEDULE_SHA, 'fit_stage': 'initial',
              'fit_uses_final_valid': False, 'optimizer_steps': 0, 'script_sha256': e.sha(__file__),
              'evaluator_sha256': e.sha(e.__file__), 'buffers': {}, 'completed_ids': [],
              'scope': 'Train-side BN diagnostic; no weight training, pooled-moment claim, or calibration convergence claim'}
    if not a.run_inference:
        e.write_result(a.output, result);return result
    import torch
    torch.set_num_threads(a.cpu_threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device(a.device)
    e.require(device.type == 'cuda' and torch.cuda.is_available(), 'Explicit CUDA required')
    torch.cuda.set_device(device)
    payload = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    e.require(payload.get('schema_version') == e.SCHEMA and payload.get('contract') == run['contract'] and
              payload.get('contract_sha256') == side['contract_sha256'] and
              payload.get('provenance') == run['provenance'] and
              payload.get('effective_config') == run['effective_config'] and
              payload['training']['completed_optimizer_steps'] == side['completed_optimizer_steps'],
              'Checkpoint payload binding differs')
    sys.path.insert(0, str(a.source_root));os.environ['SDFORMER_SNN_BACKEND'] = 'torch'
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
    from models.STSwinNet_SNN.Spiking_submodules import PSN
    from utils.runtime_backend import configure_snn_backend
    from spikingjelly.activation_based import functional
    for obj in (MS_SpikingformerFlowNet_en4, PSN, configure_snn_backend):
        e.require(Path(inspect.getfile(obj)).resolve().is_relative_to(a.source_root), 'Unpinned import')
    config = copy.deepcopy(config)
    config['swin_transformer']['input_size'] = [480,640] if a.geometry == 'full9' else [288,384]
    model = MS_SpikingformerFlowNet_en4(copy.deepcopy(config['model']), copy.deepcopy(config['swin_transformer']))
    model.load_state_dict(payload['model_state_dict'], strict=True);del payload
    model.to(device);functional.set_step_mode(model, 'm');configure_snn_backend(model, device, config, PSN)
    bns = reset_bn_cma(model, torch);before = non_bn_digest(model, bns)
    started = time.perf_counter();ys,xs = metrics.crop_slices(a.geometry)
    result.update(status='RUNNING', BN_count=len(bns), source_non_bn_state_sha256=before, input_evidence=[])
    try:
        with torch.inference_mode():
            for i, target in enumerate(fit_ids):
                e.require(time.perf_counter()-started < a.wall_minutes*60, 'BN diagnostic wall budget exhausted')
                relative = native.target_files(rows[target])['input']
                path = native.checked_path(a.data_root, relative)
                expected = hashes[relative]['sha256']
                e.require(e.sha(path) == expected, 'Calibration event bytes differ')
                events = np.load(path, allow_pickle=False)
                e.require(events.shape == (10,480,640) and np.isfinite(events).all(), 'Native event shape/finite differs')
                chunk = e.prepare_input(events[:,ys,xs], device, torch)
                functional.reset_net(model)
                output = model(chunk)['flow'][-1]
                e.require(torch.isfinite(output).all().item(), 'Nonfinite calibration forward')
                del chunk, output
                result['completed_ids'].append(target)
                result['input_evidence'].append({'target_id': target, 'relative': relative, 'sha256': expected})
                if (i+1)%16 == 0:
                    print(json.dumps({'BN_calibration_completed':i+1, 'of':len(fit_ids)}), flush=True)
        after = non_bn_digest(model, bns)
        e.require(after == before, 'Non-BN model state mutated')
        result.update(non_bn_state_unchanged=True, after_non_bn_state_sha256=after)
        for name, bn in bns.items():
            e.require(bn.num_batches_tracked.item() == len(fit_ids) and
                      torch.isfinite(bn.running_mean).all().item() and
                      torch.isfinite(bn.running_var).all().item() and not (bn.running_var<0).any().item(),
                      'Incomplete/nonfinite BN statistics')
            result['buffers'][name] = {'running_mean':bn.running_mean.detach().cpu().tolist(),
                                      'running_var':bn.running_var.detach().cpu().tolist(),
                                      'num_batches_tracked':int(bn.num_batches_tracked.item())}
        result['status'] = 'COMPLETE'
    except BaseException as exc:
        result.update(status='FAILED_NOT_VALID_CALIBRATION', error=type(exc).__name__+': '+str(exc));raise
    finally:
        result['elapsed_seconds'] = time.perf_counter()-started
        result['GPU_executed'] = True
        e.write_result(a.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--schedule', type=Path, required=True)
    extra, argv = parser.parse_known_args()
    args = e.parse_args(argv)
    result = collect(args, e.preflight(args), extra.schedule)
    print(json.dumps({'status':result['status'], 'output':str(args.output)}))


if __name__ == '__main__':main()
