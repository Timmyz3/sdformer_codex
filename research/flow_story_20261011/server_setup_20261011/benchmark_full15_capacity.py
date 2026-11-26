#!/usr/bin/env python3
"""Independent bounded native full-frame W15 FP32 training-capacity probe.

Random initialization, canonical TRAIN inputs, no checkpoint or AEE. The sibling
W9 benchmark is reused only for read-only input/provenance validation and external
process-memory sampling. Without --run-gpu this script never imports torch.
"""
import argparse
import copy
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

sys.dont_write_bytecode = True
import benchmark_native_v2 as native_probe


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] /
                   'preflight/data_protocol/artifacts/common_targets.json')
    p.add_argument('--manifest-sha256', default=native_probe.PINNED_MANIFEST_SHA)
    p.add_argument('--config', type=Path)
    p.add_argument('--target-id', action='append', default=[])
    p.add_argument('--sample-count', type=int, default=4)
    p.add_argument('--warmup-steps', type=int, default=2)
    p.add_argument('--timed-steps', type=int, default=10)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--process-memory-interval', type=float, default=0.5)
    p.add_argument('--run-gpu', action='store_true')
    p.add_argument('--output', type=Path)
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    if not (0 <= a.warmup_steps <= 2 and 1 <= a.timed_steps <= 10 and
            1 <= a.sample_count <= 4 and len(a.target_id) <= 4 and
            1 <= a.cpu_threads <= 16 and a.process_memory_interval >= 0.2):
        p.error('Bounded probe: warmup 0-2, timed 1-10, at most 4 train targets')
    if a.run_gpu and a.output is None:
        p.error('--run-gpu requires --output')
    if a.output and a.output.exists() and not a.overwrite:
        p.error('Output exists; select a new path or explicitly --overwrite')
    a.source_root, a.data_root, a.manifest = a.source_root.resolve(), a.data_root.resolve(), a.manifest.resolve()
    a.config = (a.config or a.source_root / 'configs/train_DSEC_supervised_SDformerFlow_en4.yml').resolve()
    # Compatibility fields for input/config audit only; no call to the W9 runner.
    a.precision, a.eval_warmup, a.eval_steps = 'fp32', 0, 0
    return a


def load_plan(a):
    raw, samples, result = native_probe.load_plan(a)
    result.update({
        'scope': 'Native full-frame W15 FP32 training capacity/time only; random initialization plus at most 12 probe updates. NOT completed 9-to-15 fine-tuning, AEE, or reproduction.',
        'script_sha256': native_probe.sha(__file__),
        'shared_helper_sha256': native_probe.sha(native_probe.__file__),
        'shared_helper_usage': 'Read-only canonical train NPY/config/provenance validator and nvidia-smi sampler; W9 run_gpu is never called or modified.',
        'training_geometry': [1, 10, 2, 480, 640],
        'attention_window': [2, 15, 15],
        'steps': {'warmup_train_updates': a.warmup_steps, 'timed_train_updates': a.timed_steps},
        'training_recipe_scope': 'Full frame without crop; native horizontal/vertical flips, signed voxel encoding/minmax, imported all-scale supervised loss and original AdamW/clip. No epoch scheduler event.',
        'explicit_capacity_controls': {'input_size': [480, 640], 'crop': None,
                                       'window_size': [2, 15, 15], 'use_amp': False},
        'precision_scope': 'FP32 capacity control; raw YAML AMP=True is retained as provenance, not silently presented as the original precision protocol.',
        'timed_input_transfer': 'Cached full-frame pinned host input to device, native flip/encoding; no crop.',
        'checkpoint_loaded': False, 'checkpoint_saved': False,
        'validation_samples_used': False, 'AEE_computed': False,
        'geometry_reasoning': 'Native MS_PED stem downsamples by four; attention stages have spatial sizes 120x160,60x80,30x40,15x20. W15 requires right padding to 165,90,45,30 respectively. Padding is native attention behavior; successful GPU forward/backward remains the execution test.'})
    result.pop('full_eval_geometry', None)
    return raw, samples, result


def run_gpu(a, raw, samples, result):
    result['gpu_execution_attempted'] = True
    import numpy as np
    import torch
    import torchvision
    sys.path.insert(0, str(a.source_root))
    from configs.parser import YAMLParser
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
    from models.STSwinNet_SNN.Spiking_submodules import PSN
    from loss.flow_supervised import flow_loss_supervised
    from DSEC_dataloader.data_augmentation import Compose, Random_horizontal_flip, Random_vertical_flip
    from utils.runtime_backend import configure_snn_backend
    from spikingjelly.activation_based import functional
    for item in (YAMLParser, MS_SpikingformerFlowNet_en4, flow_loss_supervised, Compose):
        if not Path(inspect.getfile(item)).resolve().is_relative_to(a.source_root):
            raise RuntimeError('Native import from unexpected source tree')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; CPU substitution is prohibited')
    torch.set_num_threads(a.cpu_threads)
    device = torch.device('cuda:%d' % a.device)
    torch.cuda.set_device(device)
    result['gpu_executed'] = True
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    os.environ['SDFORMER_SNN_BACKEND'] = 'torch'
    parser = YAMLParser.__new__(YAMLParser)
    parser.reset_config(); parser.parse_dict(copy.deepcopy(raw))
    config = YAMLParser.combine_entries(parser.config)
    config['data']['path'] = str(a.data_root)
    config['loader']['crop'] = None
    config['loader']['seed'] = a.seed
    config['swin_transformer']['input_size'] = [480, 640]
    config['swin_transformer']['window_size'] = [2, 15, 15]
    config['optimizer']['use_amp'] = False
    config['runtime'] = {'snn_backend': 'torch', 'allow_tf32': False, 'cudnn_benchmark': False}
    result['effective_config'] = copy.deepcopy(config)
    result['runtime'] = {
        'python': sys.version, 'torch': torch.__version__, 'torchvision': torchvision.__version__,
        'cuda_runtime': torch.version.cuda, 'device_logical_index': a.device,
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'device_name': torch.cuda.get_device_name(device),
        'device_total_bytes': torch.cuda.get_device_properties(device).total_memory,
        'tf32': False, 'cudnn_benchmark': False, 'cpu_threads': a.cpu_threads,
        'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'spikingjelly', 'timm', 'einops', 'PyYAML']}}

    host = [{k: torch.from_numpy(s[k]).pin_memory() for k in ('events', 'gt', 'mask')} for s in samples]
    transform = Compose([Random_horizontal_flip(config['loader']['augment_prob'][0]),
                         Random_vertical_flip(config['loader']['augment_prob'][1])])
    result['construction_started'] = True
    model = MS_SpikingformerFlowNet_en4(copy.deepcopy(config['model']),
                                       copy.deepcopy(config['swin_transformer'])).to(device)
    model.init_weights()
    functional.reset_net(model); functional.set_step_mode(model, 'm')
    configure_snn_backend(model, device, config, PSN)
    model.train()
    result['parameter_count'] = sum(p.numel() for p in model.parameters())
    result['training_BN_policy'] = 'Native train() tracked BN buffers; unchanged PSN/BN, no freezing or calibration'
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['wd'])
    optimizer.zero_grad(set_to_none=True)
    loss_fn = flow_loss_supervised(config, device)

    def memory():
        return {'allocated_current_bytes': torch.cuda.memory_allocated(device),
                'reserved_current_bytes': torch.cuda.memory_reserved(device),
                'max_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'max_reserved_bytes': torch.cuda.max_memory_reserved(device)}

    def step(i):
        source = host[i % len(host)]
        functional.reset_net(model); functional.set_step_mode(model, 'm')
        chunk = source['events'].unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=False)
        label = source['gt'].unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=False)
        mask = source['mask'].unsqueeze(0).unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=False)
        chunk, label, mask = transform((chunk, label, mask))
        valid = int(mask.sum().item())
        if valid == 0:
            raise RuntimeError('Full frame has no valid GT; no silent resampling')
        neg, pos = torch.nn.functional.relu(-chunk), torch.nn.functional.relu(chunk)
        chunk = torch.cat((pos.unsqueeze(2), neg.unsqueeze(2)), dim=2)
        nonzero = chunk[chunk != 0]
        if nonzero.numel() == 0:
            raise RuntimeError('Native nonzero minmax undefined for zero event input')
        min_val, max_val = torch.min(nonzero), torch.max(nonzero)
        if not min_val == max_val:
            chunk[chunk != 0] = (chunk[chunk != 0] - min_val) / (max_val - min_val)
        if tuple(chunk.shape) != (1, 10, 2, 480, 640):
            raise RuntimeError('Unexpected full-frame input shape')
        output = model(chunk)
        flows = output['flow']
        if not flows or any(tuple(flow.shape) != (1, 2, 480, 640) for flow in flows):
            raise RuntimeError('Unexpected native full-frame prediction shapes')
        if not torch.stack([torch.isfinite(flow).all() for flow in flows]).all().item():
            raise FloatingPointError('Nonfinite full-frame prediction')
        loss = loss_fn(flows, label, mask, gamma=config['loss']['gamma'])
        scalar = float(loss.detach().item())
        if not math.isfinite(scalar):
            raise FloatingPointError('Nonfinite full-frame training loss')
        loss.backward()
        if not any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Backward produced no gradients')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['loss']['clip_grad'], error_if_nonfinite=True)
        grad_norm = float(norm.item())
        if not math.isfinite(grad_norm):
            raise FloatingPointError('Nonfinite aggregate gradient norm')
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        return {'target_id': samples[i % len(samples)]['target_id'], 'valid_full_GT_pixels': valid,
                'prediction_shapes': [list(flow.shape) for flow in flows],
                'loss_finite': True, 'all_gradients_finite_via_global_norm': True,
                'loss_scalar_numerical_health_only': scalar,
                'gradient_norm_before_native_clip': grad_norm, 'AEE_computed': False}

    def phase(name, count, offset):
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        before = memory(); timings, health = [], []
        monitor = native_probe.ProcessMemoryMonitor(a.process_memory_interval); monitor.start()
        started = time.perf_counter(); failed = False
        try:
            for i in range(count):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                wall = time.perf_counter(); start.record()
                entry = step(i + offset)
                end.record(); torch.cuda.synchronize(device)
                timings.append({'step': i, 'cuda_elapsed_ms': start.elapsed_time(end),
                                'synchronized_wall_seconds': time.perf_counter() - wall})
                health.append(entry)
        except BaseException:
            failed = True
            raise
        finally:
            total = time.perf_counter() - started
            try:
                after = memory()
            except Exception as exc:
                after = {'unavailable': str(exc)}
            process = monitor.finish()
            cuda_sum = sum(t['cuda_elapsed_ms'] for t in timings)
            measured = {
                'status': 'FAILED_PARTIAL' if failed else 'PASS',
                'requested_steps': count, 'completed_steps': len(timings),
                'total_synchronized_wall_seconds': total,
                'wall_steps_per_second': len(timings) / total if not failed else None,
                'cuda_elapsed_sum_ms': cuda_sum,
                'cuda_event_steps_per_second': len(timings) * 1000 / cuda_sum if cuda_sum and not failed else None,
                'median_cuda_step_ms': statistics.median(t['cuda_elapsed_ms'] for t in timings) if timings else None,
                'memory_before': before, 'memory_after_and_peak': after,
                'process_memory': process, 'timings': timings, 'numerical_health': health,
                'timing_scope': 'Cached pinned inputs; H2D, flip/encoding, reset, native forward/loss/backward/AdamW and finite guards, synchronized every step. Dataset IO excluded. CUDA event elapsed is not kernel-only; external process memory is sampled, not an exact peak.'}
            result.setdefault('phases', {})[name] = measured
        print(json.dumps({'phase': name, 'status': measured['status'],
                          'wall_steps_per_second': measured['wall_steps_per_second'],
                          'max_allocated_bytes': after.get('max_allocated_bytes')}), flush=True)

    if a.warmup_steps:
        phase('full15_train_warmup_excluded_from_steady_rate', a.warmup_steps, 0)
    phase('full15_train_timed_fp32', a.timed_steps, a.warmup_steps)
    functional.reset_net(model)
    result['status'] = 'BOUNDED_FULL15_TRAINING_CAPACITY_PASS_NOT_FINETUNING_OR_ACCURACY'


def main():
    a = parse_args()
    raw, samples, result = load_plan(a)
    if not a.run_gpu:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result['status'] = 'GPU_FULL15_CAPACITY_PROBE_STARTED'
    try:
        run_gpu(a, raw, samples, result)
    except BaseException as exc:
        result['status'] = 'FAILED_NO_FINETUNING_OR_ACCURACY_CONCLUSION'
        result['exception'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        raise
    finally:
        a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
