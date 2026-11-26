#!/usr/bin/env python3
"""Bounded random-initialized native-v2 throughput/capacity probe; never AEE evidence.

Without --run-gpu, only validate selected real TRAIN NPYs/config and print a plan.
No checkpoint load/save, native-source writes, validation samples, or long training.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time
import traceback

sys.dont_write_bytecode = True
PINNED_MANIFEST_SHA = '6829f9ac8181f8eb5e7ce606c23915c2109c28447a0342bde8696cda045a7b22'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True, help='Native SDformerFlow source directory')
    p.add_argument('--data-root', type=Path, required=True, help='saved_flow_data containing event/GT/mask NPYs')
    p.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'preflight/data_protocol/artifacts/common_targets.json')
    p.add_argument('--manifest-sha256', default=PINNED_MANIFEST_SHA)
    p.add_argument('--config', type=Path, help='Default: source-root/configs/train_DSEC_supervised_SDformerFlow_en4.yml')
    p.add_argument('--target-id', action='append', default=[], help='Optional repeatable canonical TRAIN target ID, no extension')
    p.add_argument('--sample-count', type=int, default=4, help='Otherwise select first complete train targets in canonical order')
    p.add_argument('--warmup-steps', type=int, default=10)
    p.add_argument('--timed-steps', type=int, default=100)
    p.add_argument('--eval-warmup', type=int, default=2)
    p.add_argument('--eval-steps', type=int, default=5, help='Timed full-frame forwards for EACH P_code/P_paper')
    p.add_argument('--precision', choices=['fp32', 'amp_fp16_native'], default='fp32')
    p.add_argument('--device', type=int, default=0, help='Logical CUDA index; record CUDA_VISIBLE_DEVICES separately')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--process-memory-interval', type=float, default=0.5)
    p.add_argument('--run-gpu', action='store_true', help='Explicitly execute bounded GPU benchmark; default is CPU data/config plan only')
    p.add_argument('--output', type=Path, help='Required with --run-gpu; writes JSON only, never model state')
    p.add_argument('--overwrite', action='store_true')
    a = p.parse_args()
    if not (1 <= a.timed_steps <= 1000 and 0 <= a.warmup_steps <= 100 and
            1 <= a.eval_steps <= 100 and 0 <= a.eval_warmup <= 20 and
            1 <= a.sample_count <= 16 and 1 <= a.cpu_threads <= 16 and
            a.process_memory_interval >= 0.2):
        p.error('Step/sample/thread/monitor settings outside bounded benchmark limits')
    if a.run_gpu and a.output is None:
        p.error('--run-gpu requires --output')
    if a.output and a.output.exists() and not a.overwrite:
        p.error('Output exists; select a new path or explicitly --overwrite')
    a.source_root, a.data_root, a.manifest = a.source_root.resolve(), a.data_root.resolve(), a.manifest.resolve()
    a.config = (a.config or a.source_root / 'configs/train_DSEC_supervised_SDformerFlow_en4.yml').resolve()
    return a


def load_plan(a):
    import numpy as np
    import yaml
    config = yaml.safe_load(a.config.read_text())
    expected = [
        (config['model']['name'], 'MS_SpikingformerFlowNet_en4', 'model'),
        (config['data']['num_frames'], 10, 'num_frames'),
        (config['data']['num_chunks'], 1, 'num_chunks'),
        (config['data']['step_mode'], 'm', 'step_mode'),
        (config['model']['encoding'], 'voxel', 'encoding'),
        (config['model']['norm_input'], 'minmax', 'norm_input'),
        (config['loader']['batch_size'], 1, 'batch_size'),
        (config['loader']['crop'], [288, 384], 'crop'),
        (config['loader']['resolution'], [480, 640], 'resolution'),
        (config['swin_transformer']['window_size'], [2, 9, 9], 'window'),
        (config['spiking_neuron']['num_steps'], 10, 'T'),
        (config['spiking_neuron']['neuron_type'], 'psn', 'neuron_type'),
        (config['spiking_neuron']['spike_norm'], 'BN', 'spike_norm'),
        (config['optimizer']['name'], 'AdamW', 'optimizer'),
        (config['optimizer']['lr'], 0.0001, 'lr'),
        (config['optimizer']['wd'], 0.01, 'weight_decay'),
        (config['optimizer']['num_acc'], 1, 'accumulation'),
        (config['loss']['clip_grad'], 100.0, 'clip_grad'),
        (config['loss']['gamma'], None, 'loss_gamma'),
        (config['metrics']['mask_events'], False, 'mask_events'),
        (config['loader']['polarity'], True, 'signed_voxel_polarity'),
        (config['data']['spike_th'], None, 'spike_threshold'),
    ]
    for observed, wanted, key in expected:
        if observed != wanted:
            raise ValueError('Unsupported contract change for %s: %r != %r' % (key, observed, wanted))
    if sha(a.manifest) != a.manifest_sha256:
        raise ValueError('Canonical manifest SHA mismatch')
    canonical = json.loads(a.manifest.read_text())
    rows = canonical['records']
    train = {r['target_id']: r for r in rows if r['protocol_split'] == 'train'}
    valid = {r['target_id'] for r in rows if r['protocol_split'] == 'valid'}
    if len(train) != 6000 or len(valid) != 2152 or set(train) & valid:
        raise ValueError('Expected disjoint fixed 6000 train / 2152 valid manifest')

    def paths(row):
        name = row['v2_target_filename']
        if Path(name).name != name or not name.endswith('.npy'):
            raise ValueError('Unsafe filename in manifest')
        return {'events': a.data_root / 'event_tensors/10bins/left' / row['sequence'] / name,
                'gt': a.data_root / 'gt_tensors' / name,
                'mask': a.data_root / 'mask_tensors' / name}

    if a.target_id:
        ids = a.target_id
        if len(ids) > 16 or len(set(ids)) != len(ids) or any(t not in train or t in valid for t in ids):
            raise ValueError('Explicit targets must be 1-16 unique canonical TRAIN IDs')
    else:
        ids = [t for t, row in train.items() if all(p.is_file() for p in paths(row).values())][:a.sample_count]
        if len(ids) != a.sample_count:
            raise ValueError('Only %d complete train targets found; request fewer explicitly or supply data' % len(ids))
    samples, evidence = [], []
    started = time.perf_counter()
    for target in ids:
        ps = paths(train[target])
        arrays = {k: np.load(str(path), allow_pickle=False) for k, path in ps.items()}
        shapes = {'events': (10, 480, 640), 'gt': (2, 480, 640), 'mask': (480, 640)}
        for key, array in arrays.items():
            if array.shape != shapes[key] or array.dtype.hasobject or not np.isfinite(array).all():
                raise ValueError('Bad real NPY shape/type/finite check: ' + str(ps[key]))
        mask = arrays['mask']
        if not np.logical_or(mask == 0, mask == 1).all() or int(np.count_nonzero(mask)) == 0:
            raise ValueError('Mask must be nonempty binary valid-GT mask: ' + target)
        if not np.any(arrays['events'] != 0):
            raise ValueError('Native nonzero-minmax is undefined on an all-zero event tensor: ' + target)
        samples.append({'target_id': target, 'events': np.asarray(arrays['events'], dtype=np.float32),
                        'gt': np.asarray(arrays['gt'], dtype=np.float32), 'mask': mask.astype(bool)})
        evidence.append({'target_id': target, 'protocol_split': 'train',
                         'valid_GT_pixels': int(np.count_nonzero(mask)),
                         'files': {k: {'path': str(ps[k]), 'sha256': sha(ps[k]),
                                      'shape': list(arrays[k].shape), 'dtype': str(arrays[k].dtype)} for k in ps}})
    source_names = [
        'models/STSwinNet_SNN/Spiking_STSwinNet.py',
        'models/STSwinNet_SNN/Spiking_swin_transformer3D.py',
        'models/STSwinNet_SNN/Spiking_modules.py', 'models/STSwinNet_SNN/Spiking_submodules.py',
        'loss/flow_supervised.py', 'DSEC_dataloader/DSEC_dataset_lite.py',
        'DSEC_dataloader/data_augmentation.py', 'configs/parser.py',
        'utils/runtime_backend.py', 'train_flow_parallel_supervised_SNN.py']
    return config, samples, {
        'scope': 'Bounded capacity/throughput probe using random initialization plus probe updates; NOT AEE/reproduction evidence',
        'status': 'CPU_DATA_CONFIG_PLAN_VALIDATED', 'gpu_executed': False,
        'manifest': {'path': str(a.manifest), 'sha256': sha(a.manifest)},
        'samples': evidence, 'sample_selection': 'Explicit IDs or first complete canonical TRAIN targets; repeated cached host inputs, not a scientific sample',
        'data_load_and_file_hash_wall_seconds': time.perf_counter() - started,
        'dataset_IO_in_timed_steps': False, 'timed_input_transfer': 'Full-frame pinned host-to-device transfer, then native on-device crop/flip/encoding',
        'raw_yaml': {'path': str(a.config), 'sha256': sha(a.config), 'config': config},
        'source_sha256': {name: sha(a.source_root / name) for name in source_names},
        'script_sha256': sha(__file__), 'seed': a.seed,
        'precision': a.precision, 'source_yaml_use_amp': config['optimizer']['use_amp'],
        'precision_scope': 'FP32 is an explicitly different precision control; original YAML has AMP enabled. Architecture/loss/optimizer retained, not a paper-protocol reproduction.',
        'steps': {'warmup_train_updates': a.warmup_steps, 'timed_train_updates': a.timed_steps,
                  'full_eval_warmup_per_policy': a.eval_warmup, 'full_eval_timed_per_policy': a.eval_steps},
        'checkpoint_loaded': False, 'checkpoint_saved': False, 'validation_samples_used': False,
        'training_geometry': [1, 10, 2, 288, 384], 'full_eval_geometry': [1, 10, 2, 480, 640],
        'attention_window': [2, 9, 9], 'backend': 'torch',
        'training_recipe_scope': 'RandomCrop plus native H/V flips; all multi-scale predictions use imported loss with valid-GT mask. Constant original LR within bounded updates; no epoch scheduler event.'}


class ProcessMemoryMonitor:
    """External NVML samples are observed maxima, not exact process-memory peaks."""
    def __init__(self, interval):
        self.interval, self.samples, self.errors = interval, [], []
        self.stop_event = threading.Event()
        self.thread = None
    def sample(self):
        try:
            r = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5)
            if r.returncode:
                raise RuntimeError(r.stderr.strip())
            hits = []
            for line in r.stdout.splitlines():
                parts = [part.strip() for part in line.split(',')]
                if len(parts) == 3 and parts[1] == str(os.getpid()):
                    try: used = float(parts[2])
                    except ValueError: continue
                    hits.append({'gpu_uuid': parts[0], 'used_MiB': used})
            self.samples.append({'monotonic_seconds': time.perf_counter(), 'current_pid_gpu_records': hits})
        except Exception as exc:
            if not self.errors:
                self.errors.append(type(exc).__name__ + ': ' + str(exc))
    def start(self):
        self.sample()
        def loop():
            while not self.stop_event.wait(self.interval):
                self.sample()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
    def finish(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=6)
        self.sample()
        values = [sum(r['used_MiB'] for r in s['current_pid_gpu_records']) for s in self.samples if s['current_pid_gpu_records']]
        return {'sample_interval_seconds': self.interval, 'sample_count': len(self.samples),
                'max_observed_current_process_MiB': max(values) if values else None,
                'meaning': 'Observed nvidia-smi samples summed over this PID GPU records; not exact peak. Missing PID can reflect namespace visibility, never inferred as zero.',
                'errors': self.errors, 'samples': self.samples}


def run_gpu(a, raw, samples, result):
    result['gpu_execution_attempted'] = True
    import numpy as np
    import torch
    import torchvision  # Native source imports torchvision; initialize before any optional CUDA backends.
    sys.path.insert(0, str(a.source_root))
    from configs.parser import YAMLParser
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
    from models.STSwinNet_SNN.Spiking_submodules import PSN
    from loss.flow_supervised import flow_loss_supervised
    from DSEC_dataloader.data_augmentation import Compose, RandomCrop, Random_horizontal_flip, Random_vertical_flip
    from utils.runtime_backend import configure_snn_backend
    from spikingjelly.activation_based import functional
    import inspect
    for item in (MS_SpikingformerFlowNet_en4, flow_loss_supervised, Compose, YAMLParser):
        if not Path(inspect.getfile(item)).resolve().is_relative_to(a.source_root):
            raise RuntimeError('Native import came from an unexpected source tree')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; do not silently substitute CPU')
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
    use_amp = a.precision == 'amp_fp16_native'
    config['optimizer']['use_amp'] = use_amp
    config['data']['path'] = str(a.data_root)
    config['loader']['seed'] = a.seed
    config['runtime'] = {'snn_backend': 'torch', 'allow_tf32': False, 'cudnn_benchmark': False}
    result['effective_config'] = copy.deepcopy(config)
    result['runtime'] = {'python': sys.version, 'torch': torch.__version__, 'torchvision': torchvision.__version__,
                         'cuda_runtime': torch.version.cuda, 'device_logical_index': a.device,
                         'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                         'device_name': torch.cuda.get_device_name(device),
                         'device_total_bytes': torch.cuda.get_device_properties(device).total_memory,
                         'tf32': False, 'cudnn_benchmark': False, 'cpu_threads': a.cpu_threads,
                         'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'spikingjelly', 'timm', 'einops', 'PyYAML']}}
    host = [{k: torch.from_numpy(s[k]).pin_memory() for k in ('events', 'gt', 'mask')} for s in samples]
    transform = Compose([RandomCrop((288, 384)), Random_horizontal_flip(config['loader']['augment_prob'][0]),
                         Random_vertical_flip(config['loader']['augment_prob'][1])])

    def build(resolution):
        swin = copy.deepcopy(config['swin_transformer']); swin['input_size'] = list(resolution)
        model = MS_SpikingformerFlowNet_en4(copy.deepcopy(config['model']), swin).to(device)
        model.init_weights()
        functional.reset_net(model); functional.set_step_mode(model, 'm')
        configure_snn_backend(model, device, config, PSN)
        return model

    def prepare(index, training):
        source = host[index % len(host)]
        chunk = source['events'].unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=False)
        label = source['gt'].unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=False)
        mask = source['mask'].unsqueeze(0).unsqueeze(0).to(device=device, non_blocking=False)
        if training:
            chunk, label, mask = transform((chunk, label, mask.float()))
        valid = int(mask.sum().item())
        if valid == 0:
            raise RuntimeError('Native randomly cropped sample has no valid GT; stop instead of silently resampling')
        # Literal native signed-voxel split and nonzero minmax, including native operation order.
        neg, pos = torch.nn.functional.relu(-chunk), torch.nn.functional.relu(chunk)
        chunk = torch.cat((pos.unsqueeze(2), neg.unsqueeze(2)), dim=2)
        nonzero = chunk[chunk != 0]
        if nonzero.numel() == 0:
            raise RuntimeError('Native nonzero-minmax undefined on empty event crop; no silent recipe change')
        min_val, max_val = torch.min(nonzero), torch.max(nonzero)
        if not min_val == max_val:
            chunk[chunk != 0] = (chunk[chunk != 0] - min_val) / (max_val - min_val)
        return chunk, label, mask, valid

    def memory():
        return {'allocated_current_bytes': torch.cuda.memory_allocated(device),
                'reserved_current_bytes': torch.cuda.memory_reserved(device),
                'max_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'max_reserved_bytes': torch.cuda.max_memory_reserved(device)}

    def phase(name, count, step):
        torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        before = memory(); timings, health = [], []
        monitor = ProcessMemoryMonitor(a.process_memory_interval); monitor.start()
        started = time.perf_counter()
        failed = False
        try:
            for i in range(count):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                wall = time.perf_counter(); start.record()
                h = step(i)
                end.record(); torch.cuda.synchronize(device)
                timings.append({'step': i, 'cuda_elapsed_ms': start.elapsed_time(end),
                                'synchronized_wall_seconds': time.perf_counter() - wall})
                health.append(h)
        except BaseException:
            failed = True
            raise
        finally:
            total_wall = time.perf_counter() - started
            try:
                after = memory()
            except Exception as exc:
                after = {'unavailable': str(exc)}
            process_memory = monitor.finish()
            cuda_sum = sum(t['cuda_elapsed_ms'] for t in timings)
            completed = len(timings)
            measured = {'status': 'FAILED_PARTIAL' if failed else 'PASS',
                        'requested_steps': count, 'completed_steps': completed,
                        'total_synchronized_wall_seconds': total_wall,
                        'wall_steps_per_second': None if failed else completed / total_wall,
                        'cuda_elapsed_sum_ms': cuda_sum,
                        'cuda_event_steps_per_second': completed * 1000 / cuda_sum if cuda_sum and not failed else None,
                        'median_cuda_step_ms': statistics.median(t['cuda_elapsed_ms'] for t in timings) if timings else None,
                        'memory_before': before, 'memory_after_and_peak': after, 'process_memory': process_memory,
                        'timings': timings, 'numerical_health': health,
                        'timing_scope': 'H2D, native augmentation/encoding, reset, forward, native loss/backward/optimizer for train, and numerical guards; explicit per-step synchronization. CUDA elapsed is stream event elapsed, not kernel-only time. Cached host data excludes dataset/dataloader IO; nvidia-smi sampling runs concurrently.'}
            result.setdefault('phases', {})[name] = measured
        print(json.dumps({'phase': name, 'steps': count, 'wall_steps_per_second': measured['wall_steps_per_second'],
                          'max_allocated_bytes': after['max_allocated_bytes'], 'numerical_checks': 'PASS'}), flush=True)
        return measured

    model = build((288, 384)); model.train()
    result['parameter_count'] = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optimizer']['lr'], weight_decay=config['optimizer']['wd'])
    loss_fn = flow_loss_supervised(config, device)
    # The legacy namespace matches upstream and also supports Torch 2.2.x.
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)
    result['training_BN_policy'] = 'Native train() with original tracked BN buffers; no freezing, calibration, or PSN replacement'
    result['AMP_clip_contract'] = 'If selected: preserve native clip-on-scaled-gradients BEFORE scaler.step/unscale. This is a separately labeled native-AMP timing arm, not corrected AMP training.'

    def train_step(i):
        functional.reset_net(model); functional.set_step_mode(model, 'm')
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            chunk, label, mask, valid = prepare(i, True)
            output = model(chunk)
            loss = loss_fn(output['flow'], label, mask, gamma=config['loss']['gamma'])
        scalar = float(loss.detach().item())
        if not math.isfinite(scalar):
            result['numerical_failure'] = {
                'stage': 'native_forward_loss', 'precision': a.precision,
                'meaning': 'Stop this bounded configuration; this alone does not establish AMP unavailable or a persistent model failure.'}
            raise FloatingPointError('Non-finite native training loss')
        scaler.scale(loss).backward() if use_amp else loss.backward()
        if not any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('Backward produced no parameter gradients')
        try:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['loss']['clip_grad'], error_if_nonfinite=True)
        except RuntimeError as exc:
            result['numerical_failure'] = {
                'stage': 'gradient_norm_before_native_clip', 'precision': a.precision,
                'gradients_are_scaled': use_amp, 'scaler_scale': scaler.get_scale() if use_amp else None,
                'message': str(exc),
                'meaning': 'Probe stops rather than silently retries. In AMP this checks SCALED gradients; transient scaler overflow is not evidence that AMP is unavailable or that unscaled training cannot work.'}
            raise
        grad_norm = float(norm.item())
        if not math.isfinite(grad_norm):
            raise FloatingPointError('Non-finite aggregate gradient norm')
        if use_amp:
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        return {'target_id': samples[i % len(samples)]['target_id'], 'valid_crop_GT_pixels': valid,
                'loss_finite': True, 'all_gradients_finite_via_global_norm': True,
                'loss_scalar_numerical_health_only': scalar, 'gradient_norm_before_native_clip': grad_norm,
                'loss_is_not_reported_as_AEE': True}

    if a.warmup_steps:
        phase('train_warmup_excluded_from_steady_rate', a.warmup_steps, train_step)
    phase('train_timed_' + a.precision, a.timed_steps, lambda i: train_step(i + a.warmup_steps))
    functional.reset_net(model)
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    del optimizer, scaler, model
    torch.cuda.empty_cache(); torch.cuda.synchronize(device)
    # Same probe weights, rebuilt using native full-frame constructor, W9 unchanged.
    model = build((480, 640))
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    result['full_eval_weights'] = 'In-memory state after random-init bounded probe updates; strict load into native full-frame W9 constructor. No old checkpoint. No AEE computed.'
    result['full_eval_precision'] = 'FP32 for both policies, regardless of training precision arm'
    result['full_eval_memory_scope'] = 'Optimizer state freed and allocator cache released before full-frame eval; model/input/forward only'

    def bn_signature():
        entries = {}
        for name, layer in model.named_modules():
            if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                entries[name] = {'training': layer.training, 'track_running_stats': layer.track_running_stats,
                    'buffers': {key: None if getattr(layer, key) is None else hashlib.sha256(
                        getattr(layer, key).detach().cpu().numpy().tobytes()).hexdigest()
                        for key in ('running_mean', 'running_var', 'num_batches_tracked')}}
        return entries

    def eval_step(i):
        functional.reset_net(model); functional.set_step_mode(model, 'm')
        with torch.inference_mode():
            chunk, label, mask, valid = prepare(i, False)
            flows = model(chunk)['flow']
            if not flows or any(tuple(flow.shape) != (1, 2, 480, 640) for flow in flows):
                raise RuntimeError('Unexpected native full-frame prediction shapes')
            if not torch.stack([torch.isfinite(flow).all() for flow in flows]).all().item():
                raise FloatingPointError('Non-finite full-frame prediction')
        return {'target_id': samples[i % len(samples)]['target_id'], 'input_split': 'train',
                'valid_full_GT_pixels': valid, 'prediction_shapes': [list(flow.shape) for flow in flows],
                'outputs_finite': True, 'AEE_computed': False}

    for policy in ('P_code', 'P_paper'):
        if policy == 'P_paper':
            for layer in model.modules():
                if isinstance(layer, torch.nn.modules.batchnorm._BatchNorm):
                    layer.track_running_stats = False
                    layer.running_mean = layer.running_var = layer.num_batches_tracked = None
        before = bn_signature()
        if a.eval_warmup:
            phase('eval_' + policy + '_warmup', a.eval_warmup, eval_step)
        entry = phase('eval_' + policy + '_timed', a.eval_steps, eval_step)
        entry['BN_policy'] = ('eval() with tracked probe-update running buffers' if policy == 'P_code' else
                              'eval() but track_running_stats=False and running buffers=None: current full T/B/H/W statistics')
        entry['BN_buffers_unchanged_during_eval'] = before == bn_signature()
        if not entry['BN_buffers_unchanged_during_eval']:
            raise RuntimeError('BN buffers changed during evaluation')
        entry['BN_layer_count'] = len(before)
    result['gpu_executed'] = True
    result['status'] = 'BOUNDED_CAPACITY_THROUGHPUT_PASS_NOT_ACCURACY'
    result['cuda_initialized'] = torch.cuda.is_initialized()


def main():
    a = parse_args()
    raw, samples, result = load_plan(a)
    if not a.run_gpu:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    a.output.parent.mkdir(parents=True, exist_ok=True)
    result['status'] = 'GPU_PROBE_STARTED'
    try:
        run_gpu(a, raw, samples, result)
    except BaseException as exc:
        result['status'] = 'FAILED_NO_ACCURACY_CONCLUSION'
        result['exception'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        raise
    finally:
        a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
