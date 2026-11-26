#!/usr/bin/env python3
"""Pinned native v2 crop baseline, new weights only; default is CPU preflight.

FP32 B1/T10, native augmentation/loss/PSN reset, paper-described crop schedule.
No validation, MLflow, automatic full15 stage, AMP, or legacy checkpoint import.
Engineering subsets are a separate, non-promotable training contract.
"""
import argparse
import copy
import csv
import datetime
import fcntl
import hashlib
import importlib.metadata
import inspect
import io
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

sys.dont_write_bytecode = True
SCHEMA = 'flow_a_native_v2_checkpoint_v1'
CREATOR = 'flow_a_20261012_native_baseline'
CANONICAL_SHA = '6829f9ac8181f8eb5e7ce606c23915c2109c28447a0342bde8696cda045a7b22'
SOURCE_MANIFEST_SHA = '8007d0fa80db95aaff3ff93d34c386a0d6e396df9fd5ae5ecef8a2b1e407594e'
DEPLOYMENT_SHA = 'a85f1c81773982871b9549a6c98b4dc38f8bf1969a0b154efd82f3e280ece3be'
DEPLOYMENT_CSV_SHA = '188cb6f44758b5e10d9b92837af1cb63b8af42a4eca70cc39d890edba788e1ba'
SHAPES = {'input': (10, 480, 640), 'gt': (2, 480, 640), 'mask': (480, 640)}
DTYPES = {'input': 'float32', 'gt': 'float32', 'mask': 'bool'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def checked_json(path, expected_sha):
    if sha(path) != expected_sha:
        raise ValueError('Pinned SHA mismatch: ' + str(path))
    return json.loads(Path(path).read_text())


def safe_relative(value):
    p = Path(value)
    if p.is_absolute() or not p.parts or '..' in p.parts or str(p) != value:
        raise ValueError('Unsafe relative path: ' + value)
    return p


def under_root(root, relative):
    path = root / safe_relative(relative)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Path escapes root: ' + relative)
    return path


def canonical_rows(path):
    data = checked_json(path, CANONICAL_SHA)
    records = data['records']
    train = [r for r in records if r['protocol_split'] == 'train']
    valid = [r for r in records if r['protocol_split'] == 'valid']
    tids, vids = [r['target_id'] for r in train], [r['target_id'] for r in valid]
    if len(records) != 8152 or len(train) != 6000 or len(valid) != 2152 or \
            len(set(tids)) != 6000 or len(set(vids)) != 2152 or set(tids) & set(vids):
        raise ValueError('Canonical membership must be disjoint 6000 train / 2152 valid')
    for r in records:
        for key in ('v2_target_filename', 'gt_filename', 'mask_filename', 'sequence'):
            if Path(r[key]).name != r[key] or r[key] in ('.', '..'):
                raise ValueError('Unsafe canonical member')
    return train


def target_files(row):
    return {
        'input': 'saved_flow_data/event_tensors/10bins/left/' + row['sequence'] + '/' + row['v2_target_filename'],
        'gt': 'saved_flow_data/gt_tensors/' + row['gt_filename'],
        'mask': 'saved_flow_data/mask_tensors/' + row['mask_filename'],
    }


def read_hashes(path):
    result = {}
    with Path(path).open(newline='') as f:
        for r in csv.DictReader(f):
            rel = r['relative_path']; safe_relative(rel)
            if rel in result or len(r['sha256']) != 64 or \
                    any(c not in '0123456789abcdef' for c in r['sha256']):
                raise ValueError('Duplicate/bad hash inventory member: ' + rel)
            result[rel] = {'bytes': int(r['bytes']), 'sha256': r['sha256']}
    return result


def validate_acceptance(report, root, deployment, hashes):
    """Check full source/destination acceptance, not merely an existence marker."""
    if report.get('schema_version') != 1 or report.get('mode') != 'fullhash' or \
            report.get('passed_requested_checks') is not True or \
            report.get('complete_source_destination_hash_equivalence_verified') is not True or \
            report.get('all_file_hash_inventory_generated') is not True:
        raise ValueError('Formal training requires full source/destination hash acceptance')
    if Path(report['root']).resolve() != root.resolve():
        raise ValueError('Acceptance root differs from requested dataset root')
    if report['manifest_sha256'] != DEPLOYMENT_SHA or report['manifest_csv_sha256'] != DEPLOYMENT_CSV_SHA:
        raise ValueError('Acceptance belongs to a different deployment manifest')
    c = report['counts']
    if any(c.get(k) != v for k, v in {
        'expected_files': 24496, 'passed_files': 24496, 'failed_files': 0,
        'existing_files': 24496, 'expected_bytes': 122759202642,
        'observed_bytes': 122759202642, 'hashes_computed': 24496,
        'source_hashes_verified': 24496, 'npy_headers_checked': 24456,
        'previously_known_hashes_verified': 34}.items()):
        raise ValueError('Incomplete data acceptance counters')
    if set(hashes) != set(deployment) or report.get('reference_hashes_sha256') is None:
        raise ValueError('Full accepted inventory membership/source reference missing')
    files = {r['relative_path']: r for r in report['files']}
    if len(report['files']) != 24496 or set(files) != set(deployment):
        raise ValueError('Acceptance file records differ from frozen membership')
    for rel, row in deployment.items():
        h, accepted = hashes[rel], files[rel]
        if h['bytes'] != int(row['bytes']) or accepted.get('passed') is not True or \
                accepted.get('source_hash_matches') is not True or accepted.get('sha256') != h['sha256']:
            raise ValueError('Unverified acceptance member: ' + rel)
        if row['known_sha256'] and h['sha256'] != row['known_sha256']:
            raise ValueError('Known audited SHA differs: ' + rel)
        p = under_root(root, rel)
        if not p.is_file() or p.stat().st_size != h['bytes']:
            raise ValueError('Accepted file now missing/wrong size: ' + rel)


def preflight(a):
    source = checked_json(a.source_manifest, SOURCE_MANIFEST_SHA)
    if len(source) != 63:
        raise ValueError('Expected the pinned 63-file source snapshot')
    for rel, expected in source.items():
        if sha(under_root(a.source_root, rel)) != expected:
            raise ValueError('Native source changed: ' + rel)
    train = canonical_rows(a.manifest)
    checked_json(a.deployment_manifest, DEPLOYMENT_SHA)
    if sha(a.deployment_csv) != DEPLOYMENT_CSV_SHA:
        raise ValueError('Deployment CSV SHA mismatch')
    with a.deployment_csv.open(newline='') as f:
        rows = list(csv.DictReader(f))
    deployment = {r['relative_path']: r for r in rows}
    if len(rows) != 24496 or len(deployment) != 24496:
        raise ValueError('Deployment membership changed')
    hashes = read_hashes(a.reference_hashes)
    acceptance = None
    if a.scope == 'canonical6000':
        if a.target_id or a.target_list or not a.data_acceptance:
            raise ValueError('Canonical scope needs full acceptance; target subsets are forbidden')
        acceptance = json.loads(a.data_acceptance.read_text())
        inventory = acceptance['hash_inventory']
        if inventory['files'] != 24496 or inventory['sha256'] != sha(a.reference_hashes):
            raise ValueError('Use the exact accepted full hash inventory CSV')
        validate_acceptance(acceptance, a.data_root, deployment, hashes)
        selected = train
    else:
        ids = list(a.target_id)
        if a.target_list:
            ids += [s.strip() for s in a.target_list.read_text().splitlines() if s.strip()]
        by_id = {r['target_id']: r for r in train}
        if not 2 <= len(ids) <= 6000 or len(set(ids)) != len(ids) or any(t not in by_id for t in ids):
            raise ValueError('Engineering scope needs 2-6000 explicit unique canonical TRAIN IDs')
        # Canonical order is authoritative; caller order cannot silently alter the training contract.
        wanted = set(ids); selected = [r for r in train if r['target_id'] in wanted]
    file_hashes = {}
    for row in selected:
        for category, rel in target_files(row).items():
            if rel not in hashes or rel not in deployment or deployment[rel]['split'] != 'train' or \
                    deployment[rel]['target'] != row['target_id'] or deployment[rel]['category'] != category:
                raise ValueError('Missing/mismatched TRAIN hash member: ' + rel)
            h = hashes[rel]
            if h['bytes'] != int(deployment[rel]['bytes']):
                raise ValueError('Reference size differs from sealed manifest: ' + rel)
            known = deployment[rel]['known_sha256']
            if known and known != h['sha256']:
                raise ValueError('Reference differs from previously audited SHA: ' + rel)
            p = under_root(a.data_root, rel)
            if not p.is_file() or p.stat().st_size != h['bytes']:
                raise ValueError('Missing/wrong-size train member: ' + rel)
            if a.scope == 'engineering_subset' and sha(p) != h['sha256']:
                raise ValueError('Engineering sample content not source-hash verified: ' + rel)
            file_hashes[rel] = h
    ids = [r['target_id'] for r in selected]
    return selected, file_hashes, source, {
        'scope': a.scope, 'targets': len(selected), 'target_ids': ids,
        'training_target_ids_sha256': digest(ids),
        'training_files_sha256': digest(file_hashes),
        'canonical_manifest_sha256': CANONICAL_SHA,
        'source_manifest_sha256': SOURCE_MANIFEST_SHA,
        'deployment_manifest_sha256': DEPLOYMENT_SHA,
        'deployment_csv_sha256': DEPLOYMENT_CSV_SHA,
        'reference_hashes_sha256': sha(a.reference_hashes),
        'data_acceptance_sha256': sha(a.data_acceptance) if acceptance else None,
        'source_verified_files': len(source), 'sample_SHA_checked_on_every_Dataset_read': True,
        'validation_used': False, 'GPU_executed': False,
    }


class VerifiedTrainDataset:
    """Map dataset: real per-item IO and SHA; no cached sample cycling or RNG."""
    def __init__(self, root, rows, hashes):
        self.root, self.rows, self.hashes = root, rows, hashes

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import numpy as np
        import torch
        row, output = self.rows[index], {}
        for category, rel in target_files(row).items():
            data = under_root(self.root, rel).read_bytes()
            expected = self.hashes[rel]
            if len(data) != expected['bytes'] or hashlib.sha256(data).hexdigest() != expected['sha256']:
                raise ValueError('Training sample changed after acceptance: ' + rel)
            array = np.load(io.BytesIO(data), allow_pickle=False)
            if array.shape != SHAPES[category] or str(array.dtype) != DTYPES[category] or not np.isfinite(array).all():
                raise ValueError('Training NPY shape/dtype/finite mismatch: ' + rel)
            output[category] = torch.from_numpy(array)
        return output['input'], output['mask'], output['gt'], row['target_id']


def epoch_order(torch, count, seed, epoch):
    generator = torch.Generator().manual_seed(seed + 1000003 * epoch)
    return torch.randperm(count, generator=generator).tolist()


def capture_rng(torch, np):
    n = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': {'name': n[0], 'keys': n[1].tolist(), 'pos': int(n[2]),
                      'has_gauss': int(n[3]), 'cached_gaussian': float(n[4])},
            'torch_cpu': torch.get_rng_state(),
            'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(torch, np, value):
    random.setstate(value['python'])
    n = value['numpy']
    np.random.set_state((n['name'], np.asarray(n['keys'], dtype=np.uint32), n['pos'], n['has_gauss'], n['cached_gaussian']))
    torch.set_rng_state(value['torch_cpu'])
    if value['torch_cuda']:
        if len(value['torch_cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count changed across resume')
        torch.cuda.set_rng_state_all(value['torch_cuda'])


def save_checkpoint(torch, directory, payload, keep_last=2):
    """Publish immutable step file, digest sidecar, then atomic latest pointer."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    step = payload['training']['completed_optimizer_steps']
    path = directory / ('checkpoint_step_%09d.pt' % step)
    if path.exists() or path.with_suffix('.json').exists():
        raise ValueError('Checkpoint step is immutable; refuse to overwrite an existing step')
    tmp = path.with_suffix('.pt.tmp')
    with tmp.open('wb') as f:
        torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    side = {'schema_version': SCHEMA, 'created_by': CREATOR,
            'checkpoint_file': path.name, 'checkpoint_sha256': sha(path),
            'contract_sha256': payload['contract_sha256'],
            'training_scope': payload['provenance']['training_scope'],
            'completed_optimizer_steps': step,
            'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    atomic_json(path.with_suffix('.json'), side)
    atomic_json(directory / 'latest.json', side)
    # Only this new run's recognized schema files are eligible for retention.
    recognized = []
    for candidate in sorted(directory.glob('checkpoint_step_*.pt'), reverse=True):
        sidecar = candidate.with_suffix('.json')
        if sidecar.exists():
            metadata = json.loads(sidecar.read_text())
            if metadata.get('schema_version') == SCHEMA and metadata.get('contract_sha256') == payload['contract_sha256']:
                recognized.append(candidate)
    for candidate in recognized[keep_last:]:
        candidate.unlink(); candidate.with_suffix('.json').unlink()
    return side


def load_checkpoint(torch, path, expected_contract):
    path = Path(path)
    if path.name == 'latest.json':
        pointer = json.loads(path.read_text())
        name = pointer['checkpoint_file']
        if Path(name).name != name:
            raise ValueError('Unsafe checkpoint pointer')
        path = path.parent / name
    side = json.loads(path.with_suffix('.json').read_text())
    expected = digest(expected_contract)
    if side.get('schema_version') != SCHEMA or side.get('created_by') != CREATOR or \
            side.get('contract_sha256') != expected or side.get('checkpoint_file') != path.name:
        raise ValueError('Checkpoint provenance/contract mismatch; legacy imports forbidden')
    if sha(path) != side['checkpoint_sha256']:
        raise ValueError('Checkpoint file SHA mismatch')
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value.get('schema_version') != SCHEMA or value.get('contract_sha256') != expected or \
            value.get('contract') != expected_contract or value['provenance']['created_by'] != CREATOR or \
            value['provenance']['random_init'] is not True or value['provenance']['old_checkpoint_loaded'] is not False:
        raise ValueError('Checkpoint internal provenance mismatch')
    if value['training']['completed_optimizer_steps'] != side['completed_optimizer_steps']:
        raise ValueError('Checkpoint step/sidecar mismatch')
    return value, side


def effective_config(raw, a, parser_class):
    parser = parser_class.__new__(parser_class)
    parser.reset_config(); parser.parse_dict(copy.deepcopy(raw))
    config = parser_class.combine_entries(parser.config)
    config['data']['path'] = str(a.data_root / 'saved_flow_data')
    config['loader'].update(batch_size=1, n_epochs=80, seed=a.seed, n_workers=a.workers,
                            crop=[288, 384], parallel=False, gpu=a.device)
    config['swin_transformer'].update(input_size=[288, 384], window_size=[2, 9, 9])
    config['optimizer'].update(name='AdamW', lr=1e-3, wd=0.01, use_amp=False, num_acc=1,
                               scheduler='StepLR', step_size=10, gamma=0.5)
    config['optimizer'].pop('milestones', None)
    config['runtime'] = {'snn_backend': 'torch', 'allow_tf32': False, 'cudnn_benchmark': False,
                         'cudnn_deterministic': True, 'precision': 'fp32'}
    return config


def run_training(a, rows, hashes, sources, plan, started):
    import numpy as np
    import torch
    import torchvision
    import yaml
    if not torch.cuda.is_available():
        raise RuntimeError('--run-gpu requires CUDA; no silent CPU training fallback')
    torch.set_num_threads(a.cpu_threads)
    torch.set_num_interop_threads(1)
    device = torch.device('cuda', a.device); torch.cuda.set_device(device)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    os.environ['SDFORMER_SNN_BACKEND'] = 'torch'
    sys.path.insert(0, str(a.source_root))
    from configs.parser import YAMLParser
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
    from models.STSwinNet_SNN.Spiking_submodules import PSN
    from loss.flow_supervised import flow_loss_supervised
    from DSEC_dataloader.data_augmentation import Compose, RandomCrop, Random_horizontal_flip, Random_vertical_flip
    from utils.runtime_backend import configure_snn_backend
    from spikingjelly.activation_based import functional
    for item in (YAMLParser, MS_SpikingformerFlowNet_en4, PSN, flow_loss_supervised,
                 Compose, RandomCrop, Random_horizontal_flip, Random_vertical_flip, configure_snn_backend):
        if not Path(inspect.getfile(item)).resolve().is_relative_to(a.source_root):
            raise ValueError('Imported native module from outside pinned source root')
    raw = yaml.safe_load((a.source_root / 'configs/train_DSEC_supervised_SDformerFlow_en4.yml').read_text())
    config = effective_config(raw, a, YAMLParser)
    runtime = {'python': sys.version.split()[0], 'torch': str(torch.__version__),
               'torchvision': str(torchvision.__version__), 'cuda': torch.version.cuda,
               'cudnn': torch.backends.cudnn.version(), 'device_name': torch.cuda.get_device_name(device),
               'device_count': torch.cuda.device_count(),
               'packages': {name: importlib.metadata.version(name) for name in ('numpy', 'spikingjelly', 'timm', 'einops', 'PyYAML')}}
    semantic_config = copy.deepcopy(config)
    semantic_config['data']['path'] = '<verified_DSEC>/saved_flow_data'
    semantic_config['loader']['gpu'] = '<logical-device>'
    contract = {'schema_version': SCHEMA, 'scope': a.scope,
                'training_target_ids_sha256': plan['training_target_ids_sha256'],
                'training_files_sha256': plan['training_files_sha256'],
                'canonical_manifest_sha256': CANONICAL_SHA, 'source_manifest_sha256': SOURCE_MANIFEST_SHA,
                'source_sha256': sources, 'trainer_sha256': sha(__file__),
                'effective_config': semantic_config, 'seed': a.seed, 'epochs': 80,
                'cpu_threads': a.cpu_threads, 'runtime': runtime,
                'order': 'torch.randperm(count, local Generator(seed+1000003*epoch)); committed cursor',
                'precision_choice': 'FP32 explicitly chosen; different from native YAML AMP',
                'paper_recipe': 'crop80 AdamW lr.001 wd.01 halve every10 epochs; crop B1 not established by paper',
                'validation': 'none; no selection, early stopping, validation BN update, or full15 stage'}
    provenance = {'created_by': CREATOR, 'random_init': True, 'old_checkpoint_loaded': False,
                  'source_sha256': sources, 'canonical_manifest_sha256': CANONICAL_SHA,
                  'training_manifest_sha256': plan['training_files_sha256'],
                  'training_target_ids_sha256': plan['training_target_ids_sha256'],
                  'training_scope': a.scope, 'training_protocol_split': 'train',
                  'training_geometry': [288, 384], 'window_size': [2, 9, 9],
                  'BN_policy': 'native_train_tracked', 'precision': 'fp32'}
    model = MS_SpikingformerFlowNet_en4(copy.deepcopy(config['model']), copy.deepcopy(config['swin_transformer'])).to(device)
    model.init_weights()
    functional.reset_net(model); functional.set_step_mode(model, 'm')
    configure_snn_backend(model, device, config, PSN)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    loss_fn = flow_loss_supervised(config, device)
    transform = Compose([RandomCrop((288, 384)), Random_horizontal_flip(0.5), Random_vertical_flip(0.5)])
    training = {'completed_optimizer_steps': 0, 'epoch': 0, 'next_index': 0,
                'zero_valid_crops': 0, 'training_seconds': 0.0}
    resumed = None
    if a.resume:
        saved, resumed = load_checkpoint(torch, a.resume, contract)
        model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        scheduler.load_state_dict(saved['scheduler_state_dict'])
        training = saved['training']
        if not (0 <= training['epoch'] <= 80 and 0 <= training['next_index'] < len(rows)) or \
                training['completed_optimizer_steps'] != training['epoch'] * len(rows) + training['next_index'] or \
                scheduler.last_epoch != training['epoch']:
            raise ValueError('Inconsistent resume epoch/cursor/scheduler')
        expected_order = epoch_order(torch, len(rows), a.seed, training['epoch']) if training['epoch'] < 80 else []
        if saved['epoch_order'] != expected_order:
            raise ValueError('Resume order differs from the recorded epoch permutation')
        restore_rng(torch, np, saved['rng_state'])
    elif (a.run_dir / 'latest.json').exists():
        raise ValueError('Run has a checkpoint; use explicit --resume, never silently restart')
    optimizer.zero_grad(set_to_none=True)
    dataset = VerifiedTrainDataset(a.data_root, rows, hashes)
    stop_requested = []
    def request_stop(signum, frame):
        stop_requested.append(signum)
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    atomic_json(a.run_dir / 'run_contract.json', {'contract': contract, 'contract_sha256': digest(contract),
                'provenance': provenance, 'effective_config': config, 'preflight': plan,
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'resume_from': resumed, 'segment_limits': {'max_steps_total': a.max_steps,
                                                          'wall_hours': a.wall_hours}})
    log = (a.run_dir / 'train_steps.jsonl').open('a', buffering=1)
    segment_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '_' + str(os.getpid())
    log.write(json.dumps({'event': 'segment_start', 'segment_id': segment_id,
                         'resume_from': resumed, 'starting_training': training}) + '\n')
    last_checkpoint = None
    if resumed and (a.run_dir / resumed['checkpoint_file']).is_file() and \
            sha(a.run_dir / resumed['checkpoint_file']) == resumed['checkpoint_sha256']:
        last_checkpoint = resumed
    segment_start_steps = training['completed_optimizer_steps']
    compute_started = time.monotonic()
    stop_reason = 'epochs_complete'
    def save(reason):
        nonlocal last_checkpoint
        if last_checkpoint and last_checkpoint['completed_optimizer_steps'] == training['completed_optimizer_steps']:
            print(json.dumps({'checkpoint_reused': last_checkpoint['checkpoint_file'], 'reason': reason}), flush=True)
            return
        functional.reset_net(model)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        payload = {'schema_version': SCHEMA, 'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                   'rng_state': capture_rng(torch, np), 'effective_config': copy.deepcopy(config),
                   'contract': contract, 'contract_sha256': digest(contract), 'provenance': provenance,
                   'training': copy.deepcopy(training), 'stop_reason': reason,
                   'epoch_order': epoch_order(torch, len(rows), a.seed, training['epoch']) if training['epoch'] < 80 else []}
        last_checkpoint = save_checkpoint(torch, a.run_dir, payload, a.keep_last)
        print(json.dumps({'checkpoint_saved': last_checkpoint['checkpoint_file'], 'reason': reason}), flush=True)
    try:
        while training['epoch'] < 80:
            order = epoch_order(torch, len(rows), a.seed, training['epoch'])
            # Separate generator: constructing/reconstructing the DataLoader cannot consume model RNG.
            loader_generator = torch.Generator().manual_seed(a.seed + 2000003 + training['epoch'])
            loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                    sampler=order[training['next_index']:], num_workers=a.workers,
                    pin_memory=True, drop_last=False, generator=loader_generator,
                    **({'prefetch_factor': 2} if a.workers else {}))
            for chunk, mask, label, target_ids in loader:
                if stop_requested or time.monotonic() - started >= a.wall_hours * 3600 or \
                        (a.max_steps is not None and training['completed_optimizer_steps'] >= a.max_steps):
                    stop_reason = 'signal' if stop_requested else 'wall_budget' if time.monotonic() - started >= a.wall_hours * 3600 else 'max_steps'
                    break
                step_started = time.monotonic()
                expected_id = rows[order[training['next_index']]]['target_id']
                if target_ids != [expected_id] and tuple(target_ids) != (expected_id,):
                    raise ValueError('DataLoader target order/cursor mismatch')
                functional.reset_net(model); functional.set_step_mode(model, 'm')
                chunk = chunk.to(device=device, dtype=torch.float32, non_blocking=False)
                label = label.to(device=device, dtype=torch.float32, non_blocking=False)
                mask = mask.unsqueeze(1).to(device=device, non_blocking=False)
                chunk, label, mask = transform((chunk, label, mask.float()))
                valid = int(mask.sum().item())
                neg, pos = torch.nn.functional.relu(-chunk), torch.nn.functional.relu(chunk)
                chunk = torch.cat((pos.unsqueeze(2), neg.unsqueeze(2)), dim=2)
                nz = chunk[chunk != 0]
                if nz.numel() == 0:
                    raise RuntimeError('Native nonzero-minmax undefined on empty event crop; no resampling')
                low, high = torch.min(nz), torch.max(nz)
                if not low == high:
                    chunk[chunk != 0] = (chunk[chunk != 0] - low) / (high - low)
                output = model(chunk)
                if not all(torch.isfinite(flow).all().item() for flow in output['flow']):
                    raise FloatingPointError('Nonfinite native flow output')
                loss = loss_fn(output['flow'], label, mask, gamma=None)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError('Nonfinite native loss')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0, error_if_nonfinite=True)
                lr_used = optimizer.param_groups[0]['lr']
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                functional.reset_net(model)
                torch.cuda.synchronize(device)
                training['completed_optimizer_steps'] += 1; training['next_index'] += 1
                training['zero_valid_crops'] += int(valid == 0)
                elapsed = time.monotonic() - step_started
                training['training_seconds'] += elapsed
                record = {'segment_id': segment_id, 'step': training['completed_optimizer_steps'], 'epoch_0based': training['epoch'],
                          'index_0based': training['next_index'] - 1, 'target_id': expected_id,
                          'loss': float(loss.item()), 'grad_norm_preclip': float(grad_norm.item()),
                          'valid_GT_pixels_after_crop': valid, 'lr': lr_used,
                          'step_seconds_excluding_loader_wait': elapsed}
                log.write(json.dumps(record, allow_nan=False) + '\n')
                if training['next_index'] == len(rows):
                    scheduler.step(); training['epoch'] += 1; training['next_index'] = 0
                if training['completed_optimizer_steps'] % a.checkpoint_steps == 0:
                    save('periodic')
                if training['completed_optimizer_steps'] % a.log_every == 0:
                    print(json.dumps(record), flush=True)
            else:
                continue
            break
        save(stop_reason)
        summary = {'status': 'SEGMENT_COMPLETE', 'segment_id': segment_id, 'scope': a.scope, 'stop_reason': stop_reason,
                   'training': training, 'segment_completed_steps': training['completed_optimizer_steps'] - segment_start_steps,
                   'segment_training_wall_seconds_including_loader_and_checkpoints': time.monotonic() - compute_started,
                   'invocation_wall_seconds': time.monotonic() - started,
                   'checkpoint': last_checkpoint, 'validation_used': False, 'old_checkpoint_loaded': False,
                   'GPU_executed': True, 'precision': 'fp32',
                   'max_allocated_bytes': torch.cuda.max_memory_allocated(device),
                   'max_reserved_bytes': torch.cuda.max_memory_reserved(device)}
        atomic_json(a.run_dir / 'segment_result.json', summary)
        atomic_json(a.run_dir / 'segments' / (segment_id + '.json'), summary)
        print(json.dumps(summary), flush=True)
    finally:
        log.close()


def parse_args(argv=None):
    previous = Path(__file__).resolve().parents[2] / 'flow_story_20261011'
    setup = previous / 'server_setup_20261011'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True, help='DSEC root containing saved_flow_data')
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=previous / 'preflight/data_protocol/artifacts/common_targets.json')
    p.add_argument('--source-manifest', type=Path, default=setup / 'native_source_sha256.json')
    p.add_argument('--deployment-manifest', type=Path, default=setup / 'phase1_deployment_manifest.json')
    p.add_argument('--deployment-csv', type=Path, default=setup / 'phase1_deployment_manifest.csv')
    p.add_argument('--scope', choices=['canonical6000', 'engineering_subset'], default='canonical6000')
    p.add_argument('--reference-hashes', type=Path, required=True,
                   help='Formal: exact accepted hash inventory; engineering: trusted source transfer hashes')
    p.add_argument('--data-acceptance', type=Path, help='Independent fullhash validator report; formal scope only')
    p.add_argument('--target-list', type=Path, help='Engineering only: one canonical TRAIN target ID per line')
    p.add_argument('--target-id', action='append', default=[])
    p.add_argument('--resume', type=Path, help='This entry own .pt or latest.json, exact contract only')
    p.add_argument('--max-steps', type=int, help='Cumulative optimizer-step stopping limit, not additional steps')
    p.add_argument('--wall-hours', type=float, default=12, help='One invocation budget, maximum12h plus safe checkpoint flush')
    p.add_argument('--checkpoint-steps', type=int, default=250)
    p.add_argument('--keep-last', type=int, default=2)
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--run-gpu', action='store_true')
    a = p.parse_args(argv)
    if not (0 < a.wall_hours <= 12 and 0 <= a.workers <= 2 and 1 <= a.cpu_threads <= 2 and
            1 <= a.checkpoint_steps <= 1000 and 2 <= a.keep_last <= 5 and 1 <= a.log_every <= 100 and
            0 <= a.seed < 2**31 and a.device >= 0):
        p.error('Arguments outside bounded resource/checkpoint limits')
    if a.max_steps is not None and not 1 <= a.max_steps <= 480000:
        p.error('--max-steps must be1..480000')
    if a.scope == 'engineering_subset' and (a.max_steps is None or a.max_steps > 500):
        p.error('Engineering subset needs cumulative --max-steps <=500')
    for key, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, key, value.resolve())
    return a


def main():
    a = parse_args(); started = time.monotonic()
    a.run_dir.mkdir(parents=True, exist_ok=True)
    with (a.run_dir / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            rows, hashes, source, plan = preflight(a)
            atomic_json(a.run_dir / 'preflight.json', plan)
            if not a.run_gpu:
                print(json.dumps(plan, indent=2)); return 0
            run_training(a, rows, hashes, source, plan, started)
            return 0
        except BaseException as exc:
            atomic_json(a.run_dir / 'failure.json', {'status': 'FAILED', 'exception': type(exc).__name__,
                        'message': str(exc), 'traceback': traceback.format_exc(),
                        'wall_seconds': time.monotonic() - started,
                        'recovery': 'No checkpoint is published from a partial/failed update. Resume last verified checkpoint explicitly.'})
            raise


if __name__ == '__main__':
    raise SystemExit(main())
