#!/usr/bin/env python3
"""One bounded static-recovery arm from the frozen135930 native checkpoint."""
import argparse
import contextlib
import copy
import datetime
import fcntl
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback
sys.dont_write_bytecode = True

import model_api as api

BASE_SHA = '1cacbaeea4d9eaf19a304c3faaa13f5278f0212b6cdc92210c4619447d571628'
START_SHA = 'd5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30'
ORDER_SHA = '44976c1e0e145b6ca374fa8697c43e06aaeea5dd0fd2f0c1d36f370ae601b15d'
ORDER_SEED = 'flow-recovery-order-v1-20261013:'
AUGMENT_PREFIX = 'flow-recovery-augmentation-v1:'
MODEL_SEED = 174713
FIT20_SHA = '29ec7ba9c9faca74c62c3e170c9d3c70855a4605a7eab31f9331c519f56f8915'
FIT2_SHA = '9a908ea2da02812c771ff42e9a04f96e14388f27526a73660e4e6abb489e4f6f'
FIT20_COLLECTOR_SHA = '8f951e4714ef58648a381c4b5272b89d2cafe5d3c301042d2b8e536e151fcc2b'
FIT2_COLLECTOR_SHA = '49b0668626d4be21f861efc587a397e13b17a17bf4fe0d382478656560fac831'
DATA_MANIFEST_CONTRACT_SHA = 'c4421befb3f2f9779b259ed0f0db114d70a06ff9d19723a98949e5ad281cfd03'
HERE = Path(__file__).resolve().parent


def seed_for_target(target_id):
    return int(api.hashlib.sha256((AUGMENT_PREFIX+target_id).encode()).hexdigest()[:16], 16) % 2**63


@contextlib.contextmanager
def augmentation_rng(torch, np, target_id):
    """Native crop/flip RNG isolated from dropout, worker and resume RNG."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    seed = seed_for_target(target_id)
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.random.default_generator.manual_seed(seed)  # CPU only: never reseed CUDA dropout here.
            yield seed
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def validate_order(manifest, all_rows, roles, canonical_sha, roles_sha):
    expected = {r['target_id'] for r in all_rows
                if r['protocol_split'] == 'train' and roles.get(r['target_id']) == 'gate_fit'}
    ids = manifest.get('target_ids', [])
    api.require(manifest.get('schema') == 'flow_recovery_gatefit4573_v1' and len(ids) == len(set(ids)) == 4573
                and set(ids) == expected and manifest.get('target_ids_sha256') == api.digest(ids), 'Recovery population/order differs')
    expected_order = sorted(expected, key=lambda t: (api.hashlib.sha256((ORDER_SEED+t).encode()).hexdigest(), t))
    api.require(ids == expected_order and manifest.get('order_seed') == ORDER_SEED, 'Recovery order seed differs')
    api.require(manifest.get('canonical_sha256') == canonical_sha and manifest.get('roles_sha256') == roles_sha
                and manifest.get('checkpoint_sha256') == START_SHA and manifest.get('checkpoint_optimizer_steps') == 135930
                and manifest.get('final_valid_used') is False, 'Recovery source binding differs')
    api.require(manifest.get('augmentation_seed_policy', {}).get('seed_prefix') == AUGMENT_PREFIX
                and manifest.get('repeat_to_replace_empty_crop') is False, 'Augmentation/empty-crop contract differs')
    by_id = {r['target_id']: r for r in all_rows}
    return [by_id[t] for t in ids]


def preflight(a):
    sys.path.insert(0, str(a.breadth_root))
    base = api.import_pinned('recovery_frozen_evaluator', a.breadth_root/'evaluate_breadth.py', BASE_SHA)
    native, trainer, metrics, pins = base.support_modules(a.prior_root)
    core = trainer.load_core()
    # Reuse original strict canonical6000/new-native acceptance + checkpoint validation.
    dummy = HERE/'preflight_dense.json'
    args = argparse.Namespace(**vars(a), cases=dummy, cases_sha256=api.sha(dummy), geometry='full9',
                              regions=None, regions_sha256=None, actual_pair_count=0)
    context = base.preflight(args)
    _, _, _, _, hashes, config, run, side, _, plan = context
    api.require(a.checkpoint_sha256 == START_SHA and side['completed_optimizer_steps'] == 135930,
                'All arms must start from the frozen135930 snapshot')
    manifest = base.read_pin(a.recovery_manifest, ORDER_SHA)
    all_rows = metrics.canonical_rows(a.canonical)
    roles = metrics.auxiliary_roles(a.roles, all_rows)
    rows = validate_order(manifest, all_rows, roles, api.sha(a.canonical), api.sha(a.roles))
    config = copy.deepcopy(config)
    config['data']['path'] = str(a.data_root/'saved_flow_data')
    config['loader'].update(batch_size=1, n_epochs=1, crop=[288, 384], parallel=False, seed=MODEL_SEED, n_workers=a.workers)
    config['swin_transformer'].update(input_size=[288, 384], window_size=[2, 9, 9])
    config['optimizer'].update(name='AdamW', lr=.00025, wd=.01, use_amp=False, num_acc=1, scheduler=None)
    for key in ('step_size', 'gamma', 'milestones'):
        config['optimizer'].pop(key, None)
    svs = None
    svs_binding = None
    if a.arm == 'svs_hidden25':
        api.require(a.svs_artifact and a.svs_artifact_sha256 and a.svs_fit_manifest and a.svs_collector_sha256,
                    'SVS needs explicit score artifact, fixed fit manifest and collector SHA')
        fit_sha = FIT2_SHA if a.svs_fit_scope == 'fit2_engineering' else FIT20_SHA
        fit = base.read_pin(a.svs_fit_manifest, fit_sha)
        svs = base.read_pin(a.svs_artifact, a.svs_artifact_sha256)
        ids = fit['target_ids']
        api.require(len(ids) == (2 if a.svs_fit_scope == 'fit2_engineering' else 20)
                    and set(ids) <= {r['target_id'] for r in rows} and svs.get('fit_ids') == ids,
                    'SVS fit membership outside fixed gate-fit set')
        ev = svs.get('evidence', {})
        api.require(a.svs_collector_sha256 == (FIT2_COLLECTOR_SHA if a.svs_fit_scope == 'fit2_engineering' else FIT20_COLLECTOR_SHA),
                    'Unreviewed SVS collector source')
        api.require(ev.get('collector_sha256') == a.svs_collector_sha256 and ev.get('evaluator_sha256') == BASE_SHA
                    and ev.get('static_methods_sha256') == api.STATIC_SHA and ev.get('sample_count') == len(ids)
                    and ev.get('CPU_only_SVD') is True and ev.get('cuda_initialized') is False
                    and ev.get('source_site') == 'sn2 binary output before drop2/fc2'
                    and ev.get('score_input') == 'lossless uint8 counts; float64/10; no spatial approximation', 'SVS evidence differs')
        if a.svs_fit_scope == 'fit20':
            api.require(ev.get('fit_manifest_sha256') == FIT20_SHA
                        and ev.get('data_manifest_contract_sha256') == DATA_MANIFEST_CONTRACT_SHA
                        and ev.get('upstream_collector_sha256') == FIT2_COLLECTOR_SHA, 'SVS20 lacks exact fit/source binding')
        else:
            api.require(a.max_targets <= 32, 'fit2_engineering is limited to32 visits and cannot become a full recovery arm')
        svs_binding = {'artifact_sha256': a.svs_artifact_sha256, 'fit_manifest_sha256': fit_sha,
                       'collector_sha256': a.svs_collector_sha256, 'fit_scope': a.svs_fit_scope,
                       'fit_ids': ids, 'fixed_through_recovery': True}
    else:
        api.require(not any((a.svs_artifact, a.svs_artifact_sha256, a.svs_fit_manifest, a.svs_collector_sha256)),
                    'Non-SVS arm must not silently bind extra score artifacts')
    recipe = {'start_checkpoint_sha256': START_SHA, 'start_step': 135930, 'arm': a.arm,
              'canonical_manifest_sha256': api.sha(a.canonical), 'roles_sha256': api.sha(a.roles),
              'recovery_manifest_sha256': ORDER_SHA, 'training_target_ids': manifest['target_ids'],
              'training_target_ids_sha256': manifest['target_ids_sha256'], 'data_contract': plan['data_contract'],
              'source_sha256': plan['source_sha256'], 'source_manifest_sha256': api.sha(a.source_manifest),
              'implementation_sha256': {'train_recovery.py': api.sha(__file__), 'model_api.py': api.sha(api.__file__),
                                       'training_core': trainer.CORE_SHA, 'evaluator': BASE_SHA, 'static': api.STATIC_SHA, 'svs': api.SVS_SHA},
              'svs_binding': svs_binding, 'optimizer': {'name': 'AdamW', 'fresh_for_all_arms': True, 'lr': .00025,
                                                      'weight_decay': .01, 'clip_grad_norm': 100., 'scheduler': None},
              'one_pass': True, 'max_target_visits': 4573, 'max_updates': 4573, 'cumulative_wall_cap_seconds': 2700,
              'zero_valid_GT': 'Record and advance cursor without forward/BN/optimizer update; never repeat or replace',
              'augmentation_seed_prefix': AUGMENT_PREFIX, 'model_rng_seed': MODEL_SEED,
              'precision': 'FP32 TF32off noautocast', 'training_BN': 'native train tracked',
              'geometry': {'crop': [288, 384], 'T': 10, 'B': 1, 'window': [2, 9, 9]},
              'no_validation_or_best_checkpoint_selection': True, 'cpu_threads': a.cpu_threads}
    resumed = None
    if a.resume:
        resumed, _ = api.read_checkpoint(a.resume, a.resume_sha256)
        api.require(all(resumed['contract'].get(k) == v for k, v in recipe.items()), 'Resume source/data/arm/recipe differs')
        validate_cursor(resumed['training'], recipe['training_target_ids'])
        api.require(resumed['training']['visited_targets'] < a.max_targets <= 4573,
                    'Resume requires a greater cumulative target bound within the same one-pass budget')
        api.require(resumed['architecture']['arm'] == a.arm and resumed['architecture']['source_sha256'] == recipe['source_sha256'],
                    'Resume architecture source/arm differs')
    return base, core, native, rows, hashes, config, run, side, recipe, svs, resumed


def validate_cursor(training, order):
    visited, updates, empty = (training[k] for k in ('visited_targets', 'completed_optimizer_steps', 'zero_valid_crops'))
    api.require(all(type(v) is int for v in (visited, updates, empty)) and 0 <= visited <= len(order)
                and updates >= 0 and empty >= 0 and updates+empty == visited, 'Recovery cursor/update accounting differs')
    api.require(math.isfinite(training['charged_wall_seconds']) and training['charged_wall_seconds'] >= 0,
                'Invalid persisted wall budget')


def begin_budget(core, directory, recipe, *, resume_sha256, checkpoint_charge=0.):
    """Reserve the remaining cap before CUDA work; hard-killed segments fail closed."""
    path = directory/'wall_budget.json'
    key = api.digest(recipe)
    previous = json.loads(path.read_text()) if path.exists() else None
    if resume_sha256:
        api.require(previous is not None and previous.get('schema') == 'flow_recovery_wall_budget_v1'
                    and previous.get('recipe_sha256') == key, 'Resume needs this arm cumulative wall ledger')
        api.require(previous.get('status') == 'SETTLED',
                    'Unsettled interrupted segment reserves all remaining wall budget; no automatic retry')
        latest = json.loads((directory/'latest.json').read_text())
        api.require(latest.get('checkpoint_sha256') == resume_sha256,
                    'Resume must use latest committed cursor, never replay an older portion of the pass')
        charge = previous['charged_wall_seconds']
        api.require(math.isfinite(charge) and charge >= checkpoint_charge >= 0, 'Wall ledger regresses checkpoint charge')
    else:
        api.require(previous is None, 'Fresh arm cannot reuse an existing wall ledger')
        charge = 0.
    api.require(charge < 2700, 'This arm exhausted its cumulative45minute budget')
    budget = {'schema': 'flow_recovery_wall_budget_v1', 'recipe_sha256': key, 'status': 'IN_PROGRESS',
              'starting_charged_wall_seconds': charge, 'charged_wall_seconds': charge,
              'reserved_remaining_seconds': 2700-charge, 'cap_seconds': 2700,
              'resume_sha256': resume_sha256, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'accounting': 'Each GPU-requested invocation includes preflight, model construction, loader waits, updates and checkpoint flush. Hard kill keeps remaining cap reserved.'}
    core.atomic_json(path, budget)
    return budget


def settle_budget(core, directory, budget, elapsed, outcome):
    api.require(math.isfinite(elapsed) and elapsed >= 0, 'Invalid segment duration')
    settled = dict(budget, status='SETTLED', charged_wall_seconds=budget['starting_charged_wall_seconds']+elapsed,
                   last_invocation_seconds=elapsed, last_outcome=outcome, reserved_remaining_seconds=0,
                   settled_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    core.atomic_json(directory/'wall_budget.json', settled)
    return settled


def same_tree(torch, left, right):
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(same_tree(torch, left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(same_tree(torch, x, y) for x, y in zip(left, right))
    return type(left) is type(right) and left == right


def save_checkpoint(torch, core, directory, payload, keep_last=2):
    visited = payload['training']['visited_targets']
    path = directory/('checkpoint_visit_%06d.pt' % visited)
    if path.exists() or path.with_suffix('.json').exists():
        side = json.loads(path.with_suffix('.json').read_text())
        previous, old = api.read_checkpoint(path, side['checkpoint_sha256'], expected_contract=payload['contract'])
        api.require(old['visited_targets'] == visited and old['completed_optimizer_steps'] == payload['training']['completed_optimizer_steps']
                    and all(same_tree(torch, previous[key], payload[key]) for key in
                            ('model_state_dict', 'optimizer_state_dict', 'rng_state', 'training_order', 'architecture')),
                    'Existing immutable checkpoint cursor differs')
        return side
    temp = path.with_suffix('.pt.tmp')
    with temp.open('wb') as f:
        torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)
    side = {'schema_version': api.SCHEMA, 'checkpoint_file': path.name, 'checkpoint_sha256': api.sha(path),
            'contract_sha256': payload['contract_sha256'], 'arm': payload['contract']['arm'],
            'visited_targets': visited, 'completed_optimizer_steps': payload['training']['completed_optimizer_steps'],
            'architecture_sha256': payload['architecture_sha256']}
    core.atomic_json(path.with_suffix('.json'), side)
    core.atomic_json(directory/'latest.json', side)
    candidates = sorted(directory.glob('checkpoint_visit_*.pt'), reverse=True)
    for old in [p for p in candidates if p.name != 'checkpoint_visit_000000.pt'][keep_last:]:
        old_side = json.loads(old.with_suffix('.json').read_text())
        api.require(old_side.get('schema_version') == api.SCHEMA and old_side.get('contract_sha256') == payload['contract_sha256'],
                    'Retention refuses foreign checkpoint')
        old.unlink(); old.with_suffix('.json').unlink()
    return side


def run_training(a, context, invocation_started, budget):
    base, core, native, rows, hashes, config, source_run, source_side, recipe, svs, restored = context
    import numpy as np
    import torch
    import torchvision
    api.require(torch.cuda.is_available(), 'Explicit --run-gpu requires CUDA')
    torch.set_num_threads(a.cpu_threads)
    torch.set_num_interop_threads(1)
    device = torch.device('cuda', a.device); torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    os.environ['SDFORMER_SNN_BACKEND'] = 'torch'
    sys.path.insert(0, str(a.source_root))
    from models.STSwinNet_SNN.Spiking_submodules import PSN
    from loss.flow_supervised import flow_loss_supervised
    from DSEC_dataloader.data_augmentation import Compose, RandomCrop, Random_horizontal_flip, Random_vertical_flip
    from utils.runtime_backend import configure_snn_backend
    from spikingjelly.activation_based import functional
    for item in (PSN, flow_loss_supervised, Compose, RandomCrop, Random_horizontal_flip, Random_vertical_flip, configure_snn_backend):
        api.require(Path(inspect.getfile(item)).resolve().is_relative_to(a.source_root), 'Unpinned native training import')
    runtime = {'python': sys.version.split()[0], 'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
               'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'device_name': torch.cuda.get_device_name(device),
               'device_count': torch.cuda.device_count()}
    if a.resume:
        side = json.loads(a.resume.with_suffix('.json').read_text())
        architecture = restored['architecture']
        model = api.construct_model(config, architecture, source_root=a.source_root, prior_root=a.prior_root, breadth_root=a.breadth_root)
        model.load_state_dict(restored['model_state_dict'], strict=True)
        projection = restored['initial_projection']
    else:
        api.require(not (a.run_dir/'run_contract.json').exists() and not list(a.run_dir.glob('checkpoint_visit_*')),
                    'Fresh arm requires unused run directory; use explicit resume')
        initial = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
        api.require(initial.get('schema_version') == base.SCHEMA and initial.get('contract') == source_run['contract']
                    and initial.get('contract_sha256') == source_side['contract_sha256']
                    and initial.get('effective_config') == source_run['effective_config']
                    and initial.get('provenance') == source_run['provenance']
                    and initial['training']['completed_optimizer_steps'] == 135930, 'Initial payload/source metadata differs')
        api.require(all(isinstance(t, torch.Tensor) and (not t.is_floating_point() or torch.isfinite(t).all())
                        for t in initial['model_state_dict'].values()), 'Nonfinite initial checkpoint state')
        svs_expected = None
        if a.arm == 'svs_hidden25':
            temporary = api.instantiate_native(config, a.source_root, recipe['source_sha256'])
            temporary.load_state_dict(initial['model_state_dict'], strict=True)
            base.apply_bn(temporary, 'P_paper', torch)
            geometry = {'name': 'full9', 'input_hw': [480, 640], 'window': [2, 9, 9], 'T': 10, 'B': 1,
                        'precision': 'FP32 TF32off noautocast', 'input_preprocess': 'pinned evaluate_breadth.prepare_input'}
            svs_expected = {'checkpoint_sha256': START_SHA, 'data_contract_sha256': api.digest(source_run['effective_config']['data_contract']),
                            'source_sha256': api.digest(recipe['source_sha256']),
                            'bn_policy_sha256': api.digest({'policy': 'P_paper', 'eval_dropout_off': True,
                                'original_dense_bn_signature': base.bn_signature(temporary, torch), 'recalibration_performed': False}),
                            'geometry_sha256': api.digest(geometry), 'fit_ids_sha256': api.digest(svs['fit_ids']), 'split': 'train'}
            del temporary
        model, architecture, projection = api.build_initial_model(config, initial['model_state_dict'], a.arm,
            source_root=a.source_root, prior_root=a.prior_root, breadth_root=a.breadth_root,
            source_sha256=recipe['source_sha256'], svs_artifact=svs, svs_expected=svs_expected)
        del initial
    contract = {'schema_version': api.SCHEMA, **recipe, 'architecture_sha256': api.digest(architecture),
                'effective_config': api.semantic_config(config), 'runtime': runtime}
    if restored:
        api.require(restored['contract'] == contract and restored.get('scheduler_state_dict') is None,
                    'Resume recipe/architecture/runtime differs')
    model.to(device); functional.set_step_mode(model, 'm'); configure_snn_backend(model, device, config, PSN); model.train()
    bn = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    api.require(len(bn) == 78 and all(m.training and m.track_running_stats and m.running_mean is not None and m.running_var is not None for m in bn),
                'Recovery must retain all78 native tracked training BN')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.00025, weight_decay=.01)
    training = {'visited_targets': 0, 'completed_optimizer_steps': 0, 'zero_valid_crops': 0, 'charged_wall_seconds': 0.}
    if restored:
        optimizer.load_state_dict(restored['optimizer_state_dict'])
        training = copy.deepcopy(restored['training'])
        validate_cursor(training, recipe['training_target_ids'])
        api.require(restored['training_order'] == recipe['training_target_ids'], 'Resume sampler order differs')
        core.restore_rng(torch, np, restored['rng_state'])
    else:
        random.seed(MODEL_SEED); np.random.seed(MODEL_SEED); torch.manual_seed(MODEL_SEED); torch.cuda.manual_seed_all(MODEL_SEED)
    api.require(all(g['lr'] == .00025 and g['weight_decay'] == .01 for g in optimizer.param_groups), 'Optimizer recipe drift')
    functional.reset_net(model); optimizer.zero_grad(set_to_none=True)
    loss_fn = flow_loss_supervised(config, device)
    transform = Compose([RandomCrop((288, 384)), Random_horizontal_flip(.5), Random_vertical_flip(.5)])
    core.under_root = native.checked_path
    dataset = core.VerifiedTrainDataset(a.data_root, rows, hashes)
    loader_generator = torch.Generator().manual_seed(MODEL_SEED+2000003)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, sampler=list(range(training['visited_targets'], len(rows))),
        num_workers=a.workers, pin_memory=True, drop_last=False, generator=loader_generator,
        **({'prefetch_factor': 2} if a.workers else {}))
    run_record = {'contract': contract, 'contract_sha256': api.digest(contract), 'architecture': architecture,
                  'effective_config': config, 'initial_projection': projection}
    if (a.run_dir/'run_contract.json').exists():
        api.require(json.loads((a.run_dir/'run_contract.json').read_text()) == run_record, 'Run directory contract differs')
    else:
        core.atomic_json(a.run_dir/'run_contract.json', run_record)
    segment_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_'+str(os.getpid())
    start_charge = budget['starting_charged_wall_seconds']; segment_started = invocation_started; last_checkpoint = None
    stop_requested = []
    def signal_handler(signum, frame):
        stop_requested.append(signum)
    signal.signal(signal.SIGTERM, signal_handler); signal.signal(signal.SIGINT, signal_handler)
    def save(reason):
        nonlocal last_checkpoint
        functional.reset_net(model); optimizer.zero_grad(set_to_none=True); torch.cuda.synchronize(device)
        training['charged_wall_seconds'] = start_charge+time.monotonic()-segment_started
        payload = {'schema_version': api.SCHEMA, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': None, 'rng_state': core.capture_rng(torch, np), 'training': copy.deepcopy(training),
            'training_order': recipe['training_target_ids'], 'architecture': architecture, 'architecture_sha256': api.digest(architecture),
            'initial_projection': projection, 'effective_config': config, 'contract': contract, 'contract_sha256': api.digest(contract),
            'provenance': {'scope': 'bounded_gate_fit_recovery', 'initial_checkpoint_sha256': START_SHA, 'random_init': False,
                           'optimizer_fresh_at_recovery_start': True, 'validation_used': False}, 'stop_reason': reason}
        last_checkpoint = save_checkpoint(torch, core, a.run_dir, payload, a.keep_last)
        return last_checkpoint
    if not restored:
        save('initial_projection_step0')
    else:
        last_checkpoint = side
    log = (a.run_dir/'train_steps.jsonl').open('a', buffering=1)
    log.write(json.dumps({'event': 'segment_start', 'segment_id': segment_id, 'resume_sha256': a.resume_sha256,
                          'starting_training': training})+'\n')
    stop_reason = 'one_pass_complete'
    try:
        for events, mask, label, target_ids in loader:
            charged = start_charge+time.monotonic()-segment_started
            if stop_requested or charged >= min(2700, a.wall_minutes*60) or training['visited_targets'] >= a.max_targets:
                stop_reason = 'signal' if stop_requested else 'wall_cap' if charged >= min(2700, a.wall_minutes*60) else 'max_targets'
                break
            index = training['visited_targets']; target = rows[index]['target_id']
            api.require(list(target_ids) == [target], 'DataLoader order/cursor differs')
            tick = time.monotonic(); functional.reset_net(model); functional.set_step_mode(model, 'm')
            events = events.to(device=device, dtype=torch.float32); label = label.to(device=device, dtype=torch.float32)
            mask = mask.unsqueeze(1).to(device=device)
            with augmentation_rng(torch, np, target) as augment_seed:
                events, label, mask = transform((events, label, mask.float()))
            valid = int(mask.sum().item())
            record = {'segment_id': segment_id, 'index_0based': index, 'target_id': target, 'augmentation_seed': augment_seed,
                      'valid_GT_pixels_after_crop': valid, 'lr': .00025, 'loss': None, 'grad_norm_preclip': None}
            if valid == 0:
                training['zero_valid_crops'] += 1
                record['status'] = 'SKIPPED_ZERO_VALID_GT_NO_UPDATE'
            else:
                neg, pos = torch.nn.functional.relu(-events), torch.nn.functional.relu(events)
                events = torch.cat((pos.unsqueeze(2), neg.unsqueeze(2)), dim=2)
                nz = events[events != 0]
                api.require(nz.numel() > 0, 'Native nonzero-minmax undefined; no resampling')
                low, high = torch.min(nz), torch.max(nz)
                if not low == high:
                    events[events != 0] = (events[events != 0]-low)/(high-low)
                output = model(events)
                api.require(all(torch.isfinite(f).all().item() for f in output['flow']), 'Nonfinite native flow')
                loss = loss_fn(output['flow'], label, mask, gamma=None)
                api.require(torch.isfinite(loss).item(), 'Nonfinite native loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 100., error_if_nonfinite=True)
                if index < 8:
                    psn_parameters = [p for m in model.modules() if m.__class__.__name__ == 'PSN' for p in m.parameters(recurse=False)]
                    record['gradient_audit'] = {'PSN_parameters': len(psn_parameters),
                        'PSN_grad_present': sum(p.grad is not None for p in psn_parameters),
                        'PSN_grad_nonzero': sum(p.grad is not None and bool(torch.count_nonzero(p.grad).item()) for p in psn_parameters),
                        'all_present_gradients_finite': True, 'scope': 'Checked by clip_grad_norm(error_if_nonfinite=True); no inference-only construction'}
                optimizer.step(); optimizer.zero_grad(set_to_none=True); functional.reset_net(model)
                training['completed_optimizer_steps'] += 1
                record.update(status='UPDATED', loss=float(loss.item()), grad_norm_preclip=float(norm.item()))
                del output, loss
            training['visited_targets'] += 1
            torch.cuda.synchronize(device)
            record.update(visited_targets=training['visited_targets'], optimizer_step=training['completed_optimizer_steps'],
                          step_seconds_excluding_loader_wait=time.monotonic()-tick)
            log.write(json.dumps(record, allow_nan=False)+'\n')
            del events, mask, label
            if training['visited_targets'] % a.checkpoint_targets == 0:
                save('periodic')
            if training['visited_targets'] % a.log_every == 0:
                print(json.dumps(record), flush=True)
        validate_cursor(training, recipe['training_target_ids'])
        save(stop_reason)
        result = {'schema': 'flow_recovery_segment_v1', 'status': 'SEGMENT_COMPLETE', 'arm': a.arm, 'stop_reason': stop_reason,
                  'training': training, 'checkpoint': last_checkpoint, 'GPU_executed': True, 'validation_used': False,
                  'full_pass_complete': training['visited_targets'] == 4573,
                  'segment_seconds': time.monotonic()-segment_started, 'invocation_seconds': time.monotonic()-invocation_started,
                  'max_allocated_bytes': torch.cuda.max_memory_allocated(device), 'max_reserved_bytes': torch.cuda.max_memory_reserved(device)}
        core.atomic_json(a.run_dir/'segments'/(segment_id+'.json'), result)
        core.atomic_json(a.run_dir/'segment_result.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        core.atomic_json(a.run_dir/'segments'/(segment_id+'.failed.json'), {'status': 'FAILED_NO_PARTIAL_UPDATE_CHECKPOINT',
            'error': type(exc).__name__+': '+str(exc), 'traceback': traceback.format_exc(), 'last_verified_checkpoint': last_checkpoint})
        raise
    finally:
        log.close()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('breadth-root', 'prior-root', 'source-root', 'data-root', 'source-manifest', 'canonical', 'roles',
                 'archive-audit', 'acceptance', 'checkpoint', 'recovery-manifest', 'run-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--acceptance-sha256', required=True)
    p.add_argument('--checkpoint-sha256', default=START_SHA)
    p.add_argument('--arm', choices=api.ARMS, required=True)
    p.add_argument('--resume', type=Path); p.add_argument('--resume-sha256')
    p.add_argument('--svs-artifact', type=Path); p.add_argument('--svs-artifact-sha256')
    p.add_argument('--svs-fit-manifest', type=Path); p.add_argument('--svs-collector-sha256')
    p.add_argument('--svs-fit-scope', choices=('fit20', 'fit2_engineering'), default='fit20')
    p.add_argument('--max-targets', type=int, default=4573)
    p.add_argument('--wall-minutes', type=float, default=45)
    p.add_argument('--workers', type=int, default=2); p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--device', type=int, default=0); p.add_argument('--checkpoint-targets', type=int, default=250)
    p.add_argument('--keep-last', type=int, default=2); p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--run-gpu', action='store_true')
    a = p.parse_args(argv)
    api.require(bool(a.resume) == bool(a.resume_sha256), 'Explicit resume path and SHA must be paired')
    api.require(0 <= a.max_targets <= 4573 and 0 < a.wall_minutes <= 45 and 0 <= a.workers <= 2 and 1 <= a.cpu_threads <= 4
                and 1 <= a.checkpoint_targets <= 1000 and 2 <= a.keep_last <= 5 and 1 <= a.log_every <= 100, 'Invalid bounded resource limits')
    for name, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, name, value.resolve())
    if a.resume:
        api.require(a.resume.parent == a.run_dir, 'Resume must stay in the same arm directory')
    return a


def main():
    a = parse_args(); started = time.monotonic()
    a.run_dir.mkdir(parents=True, exist_ok=True)
    with (a.run_dir/'.recovery.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        context = preflight(a)
        core = context[1]
        preflight_result = {'schema': 'flow_recovery_preflight_v1', 'status': 'PREFLIGHT_PASS_NO_TRAINING',
                            'recipe': context[8], 'GPU_executed': False, 'max_targets': a.max_targets, 'wall_minutes': a.wall_minutes}
        if not (a.run_dir/'preflight.json').exists():
            core.atomic_json(a.run_dir/'preflight.json', preflight_result)
        if a.run_gpu:
            restored = context[-1]
            budget = begin_budget(core, a.run_dir, context[8], resume_sha256=a.resume_sha256,
                                  checkpoint_charge=restored['training']['charged_wall_seconds'] if restored else 0.)
            outcome = 'FAILED'
            try:
                run_training(a, context, started, budget)
                outcome = 'SEGMENT_COMPLETE'
            finally:
                settle_budget(core, a.run_dir, budget, time.monotonic()-started, outcome)
        else:
            print(json.dumps({'status': 'PREFLIGHT_PASS_NO_TRAINING', 'arm': a.arm, 'targets': 4573}))


if __name__ == '__main__':
    main()
