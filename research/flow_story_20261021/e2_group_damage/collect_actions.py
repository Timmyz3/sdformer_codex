#!/usr/bin/env python3
"""E2: same recovered Q6-W, 16 identities x 22 complete-group actions.

352 formal inference calls plus two dense-reference engineering calls.
No optimizer, selector fitting, GT-array access, or autonomous GPU launch.
"""
import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import signal
import sys
import time
import traceback
from pathlib import Path
sys.dont_write_bytecode = True
import group_precision_core as core
import group_contract as groups
HERE = Path(__file__).resolve().parent
CORE_SHA = 'a66496b73c2d5d97bd0995af93b4fd04d1f0a464c1cc449750a86a750eb1f96e'
GROUP_SHA = '33a5866002d91f4bacfded17fa58a8878e25efb73f00e7d7efba50fab9878f56'
SMOKE_SHA = '48563d36f1b448a66d18278e0448deee02a7650583182a628dcb3865923c6c5c'
HELPER_SHA = '495f7dbfdc9d54f74ed2f40ef5cfb9c6fcc58da5a03998efca733845d0b6753a'
OLD_CORE_SHA = '19623636229b4dc5a85132e4acc8bf217f9f62a5e51fcab0fdd7394ee186f788'
WEIGHT_SHA = '7fd8cb4ab172ca7f21bc786c8b0eae6f508cb2e5a6f5fb34653ac8607552e0aa'
STATE_SHA = 'cffd154767fc6b01de1a88c24f8eebf06c32f097703df61b0cf1e54af2782747'
OLD_RESULT_SHA = '4629f944a0411bae89d67f8809ac642aadeb3bb26abc243d6e2aa04a18d1b763'
SELECTION_SHA = '60b7d6d525d4130f5f0efecaff95511808b2186c3aaba9e9f37a957db32ef31f'
require = core.require


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def action_name(action):
    return 'background_Q6' if not action else ('single_' if len(action) == 1 else 'pair_') + '_'.join(map(str, action))


def prepare(a):
    import torch
    require(not torch.cuda.is_initialized() and not a.output.exists(), 'Fresh CPU-only preflight')
    require(sha(__file__) == a.runner_sha256 and sha(HERE/'group_precision_core.py') == CORE_SHA
            and sha(HERE/'group_contract.py') == GROUP_SHA, 'Frozen E2 implementation sources')
    require(Path(core.__file__).resolve() == HERE/'group_precision_core.py'
            and groups.core is core, 'Actual imported core identity')
    base = a.r27_root/'implementation'
    for name, expected in (('run_smoke.py', SMOKE_SHA), ('run_recovery.py', HELPER_SHA),
                           ('recovery_integer_core.py', OLD_CORE_SHA)):
        require(sha(base/name) == expected, 'Frozen original R27 helper '+name)
    require(sha(a.weights) == WEIGHT_SHA and sha(a.old_result) == OLD_RESULT_SHA
            and sha(a.selections) == SELECTION_SHA, 'Accepted recovered Q6-W/result and root roles')
    require(sha(a.contract) == a.contract_sha256, 'Root-chosen action contract bytes')
    contract = groups.validate(read(a.contract))
    require(contract['source_state_sha256'] == STATE_SHA, 'Same recovered Q6 state, not a new endpoint')
    require(contract.get('status') != 'UNFROZEN_CALLER_REQUESTED_METADATA_CANDIDATE',
            'Root must freeze actual action contract before collection')
    smoke = load(base/'run_smoke.py', '_e2_original_smoke')
    helper = load(base/'run_recovery.py', '_e2_original_helper')
    old = copy.copy(a); old.runner_sha256 = SMOKE_SHA
    p, members, _, _ = smoke.prepare(old)
    helper.validate_order(members)
    selection = read(a.selections); ids = selection['e2_reserved_ids']
    rows = {r['target_id']: r for r in members['dev_members']}
    require(len(ids) == len(set(ids)) == 16 and all(t in rows for t in ids), 'Exact16 reserved development identities')
    require(not set(ids) & set(selection['localization_ids']), 'E2 roles distinct from E1 localization')
    require(contract.get('weights_sha256') == WEIGHT_SHA
            and contract.get('selection_sha256') == SELECTION_SHA
            and contract.get('target_ids') == ids, 'Root contract binds the exact W and E2 identity order')
    require(contract.get('formal_quality_forwards') == 352
            and contract.get('engineering_dense_reference_forwards') == 2,
            'Root-frozen352+2 execution count')
    old_result = read(a.old_result)
    require(old_result['arm'] == 'Q6' and old_result['status'] == 'FIXED512_AND128_DEVELOPMENT_PREDICTIONS_COMPLETE'
            and old_result['weights_sha256'] == WEIGHT_SHA
            and old_result['final_state_sha256'] == STATE_SHA, 'Accepted original Q6 endpoint metadata')
    prior = {r['target_id']: r for r in old_result['evaluations']['endpoint']['frames']}
    require(all(t in prior for t in ids), 'Every background has prior four-head byte references')
    if a.capture_prefix:
        require(contract.get('prefix_target') in core.TARGETS, 'Explicit caller-frozen prefix affine')
        controlled = set().union(*(set(g['tiles']) for g in contract['groups']))
        require(contract['prefix_target'] in controlled, 'Prefix is input of a controlled affine')
    return smoke, helper, p, members, [rows[t] for t in ids], contract, prior, selection


def execute(a, smoke, helper, p, members, rows, contract, prior, result):
    import torch
    import numpy as np
    require(not torch.cuda.is_initialized(), 'Global lock precedes CUDA')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.manual_seed(20261019)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda:0'); torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(20261019); torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    model, functional, bn = smoke.make_model(p, 'full9', device, torch)
    cp = torch.load(a.weights, map_location='cpu', weights_only=False)
    model.load_state_dict(cp['model_state_dict'], strict=True); del cp
    model.eval().requires_grad_(False)
    require(p['modules'][0].state_digest(model) == STATE_SHA, 'Actual recovered Q6-W')
    bn_signature = p['base'].bn_signature(model, torch)
    result.update(status='RUNNING', BN_scope=bn, model_forwards=0, formal_model_forwards=0,
                  engineering_model_forwards=0, evaluations={}, parity={},
                  runtime=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                               device=torch.cuda.get_device_name(0), AMP=False, TF32=False))
    write(a.output/'results.json', result)
    for action in groups.actions():
        label = action_name(action); bound = groups.BoundCore(contract, action)
        folder = a.output/'predictions'/label; folder.mkdir(parents=True, exist_ok=False)
        frames = []; tic = time.monotonic()
        for index, row in enumerate(rows):
            tid = row['target_id']; evidence = helper.source_evidence(members, tid)
            events = helper.event_only(p, row, evidence, np)
            chunk = p['base'].prepare_input(events, device, torch); del events
            engineering = index == 0 and action in ((0,), (0, 1))
            # Optional prefix is collected only during background forward.
            # New action-independent prefix capture must be caller frozen.
            prefix = None; prefix_handles = []; call_order = []
            if a.capture_prefix and not action:
                controlled = set().union(*(set(g['tiles']) for g in contract['groups']))
                prefix_target = contract['prefix_target']
                def see_prefix(module, inputs, *, name):
                    nonlocal prefix
                    call_order.append(name)
                    if name == prefix_target:
                        x = inputs[0].detach()
                        require(bool(((x == 0) | (x == 1)).all()), 'Selected affine prefix is binary')
                        arr = x.to(torch.uint8).cpu().contiguous().numpy()
                        prefix = dict(shape=list(arr.shape), logical_dtype='binary_uint8',
                                      input_sha256=core.digest(x), packed=np.packbits(arr.reshape(-1), bitorder='little'))
                # Attach after installation below so hooks observe actual wrappers.
            if engineering:
                actual, rec = smoke.capture_full(model, functional, chunk, bound.mode, bound, torch, ste=False)
                result['model_forwards'] += 1
                dense, dense_rec = smoke.capture_full(model, functional, chunk, bound.mode, bound, torch,
                                                      ste=False, reference=True)
                result['model_forwards'] += 1; result['engineering_model_forwards'] += 1
                comparison = smoke.compare_full(actual, dense, torch)
                require(comparison['pass'], 'Full integer/PSN/head dense parity '+label)
                pf = a.output/(label+'_parity_heads.npz')
                np.savez_compressed(pf, **{k: v.numpy() for k, v in actual.items() if k.startswith('head')})
                result['parity'][label] = dict(target_id=tid, comparison=comparison,
                    actual_record=rec, dense_record=dense_rec, heads_file=pf.name, heads_sha256=sha(pf))
                heads = {f'head{i}': helper.tensor_record(actual[f'head{i}'][0]) for i in range(4)}
                prediction = actual['head3'][0].numpy()
                work = rec['work']; bns = rec['actual_BN_calls']; calls = rec['actual_target_calls']
                del actual, dense
            else:
                functional.reset_net(model)
                with bound.installed(model, bound.mode, ste=False) as wraps:
                    if a.capture_prefix and not action:
                        for name in controlled:
                            def callback(module, inputs, name=name):
                                return see_prefix(module, inputs, name=name)
                            prefix_handles.append(wraps[name].register_forward_pre_hook(callback))
                    try:
                        # observed() removes its own hooks and checks for leftovers.
                        # Prefix hooks must therefore be removed inside this context.
                        with helper.observed(model, bound, torch) as (bns, measured):
                            try:
                                with torch.no_grad():
                                    flows = model(chunk)['flow']
                            finally:
                                for handle in prefix_handles:
                                    handle.remove()
                                prefix_handles.clear()
                        work = {n: copy.deepcopy(w.last_audit) for n, w in wraps.items()}
                    finally:
                        for handle in prefix_handles:
                            handle.remove()
                    calls = {n: len(v) for n, v in measured.items()}
                result['model_forwards'] += 1
                require(len(flows) == 4, 'Four actual heads')
                heads = {f'head{i}': helper.tensor_record(v[0]) for i, v in enumerate(flows)}
                prediction = flows[3][0].detach().cpu().numpy(); del flows
            result['formal_model_forwards'] += 1
            require(prediction.shape == (2, 480, 640) and prediction.dtype == np.float32
                    and np.isfinite(prediction).all(), 'Finite complete final flow')
            require(set(work) == set(core.TARGETS) and all(v == 1 for v in calls.values()),
                    'Exactly28 actual target calls')
            if not action:
                require(heads == prior[tid]['heads'], 'All four old Q6 heads replay exactly')
            prefix_record = None
            if a.capture_prefix and not action:
                require(prefix is not None and call_order and call_order[0] == contract['prefix_target'],
                        'Recorded prefix precedes every controlled affine')
                pp = a.output/'prefix'/f'{tid}.npz'; pp.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(pp, bits=prefix.pop('packed'))
                prefix_record = dict(**prefix, target=contract['prefix_target'], file=str(pp.relative_to(a.output)),
                                     file_sha256=sha(pp), controlled_call_order=call_order,
                                     role='offline_candidate_feature_no_runtime_predictor_or_free_cost_claim')
            file = folder/f'{tid}.npz'; np.savez_compressed(file, head3=prediction)
            frames.append(dict(target_id=tid, sequence=row['sequence'], input_evidence=evidence,
                action=list(action), prediction_file=str(file.relative_to(a.output)),
                prediction_file_sha256=sha(file), prediction_sha256=heads['head3']['sha256'],
                heads=heads, actual_BN_calls=bns, actual_affine_call_counts=calls, work=work,
                prefix=prefix_record))
            del chunk, prediction
            write(a.output/'progress.json', dict(action=label, completed_identity_in_action=index+1,
                formal_model_forwards=result['formal_model_forwards'],
                engineering_model_forwards=result['engineering_model_forwards']))
            if (index+1) % 4 == 0:
                print(json.dumps(dict(action=label, identities=index+1)), flush=True)
        require(p['modules'][0].state_digest(model) == STATE_SHA
                and p['base'].bn_signature(model, torch) == bn_signature, 'Unchanged W and current-BN after action')
        result['evaluations'][label] = dict(action=list(action), mask_layout=bound.mask_record,
            frames=frames, state_sha256=STATE_SHA, model_forwards=16, GT_arrays_read=False,
            wall_seconds=time.monotonic()-tic)
        write(a.output/'results.json', result)
    require(result['model_forwards'] == 354 and result['formal_model_forwards'] == 352
            and result['engineering_model_forwards'] == 2, 'Exact352+2 fixed inference budget')
    result.update(status='E2_352_ACTION_LABELS_AND2_DENSE_REFERENCES_COMPLETE',
        background_replay_bitwise_equal_all16=True, final_state_sha256=STATE_SHA,
        wall_seconds=time.monotonic()-start, max_allocated_bytes=torch.cuda.max_memory_allocated(),
        max_reserved_bytes=torch.cuda.max_memory_reserved(), completed_optimizer_steps=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('r27-root', 'r26-root', 'members', 'weights', 'old-result', 'selections', 'contract', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--runner-sha256', required=True)
    parser.add_argument('--contract-sha256', required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--capture-prefix', action='store_true')
    parser.add_argument('--wall-minutes', type=int, default=60)
    a = parser.parse_args()
    require(a.wall_minutes == 60, 'One bounded60min action collection')
    smoke, helper, support, members, rows, contract, prior, selection = prepare(a)
    result = dict(schema='flow_e2_action_collection_v1', status='CPU_PREFLIGHT_PASS',
        sources=dict(runner=sha(__file__), core=CORE_SHA, group_contract=GROUP_SHA,
                     old_smoke=SMOKE_SHA, old_helper=HELPER_SHA, old_Q6_core=OLD_CORE_SHA,
                     weights=WEIGHT_SHA, old_result=OLD_RESULT_SHA, selections=SELECTION_SHA,
                     members=sha(a.members), action_contract=a.contract_sha256),
        source_state_sha256=STATE_SHA, action_contract=contract, selected_ids=[r['target_id'] for r in rows],
        selection_scope=selection['scope'], expected_formal_forwards=352, expected_engineering_forwards=2,
        expected_quality_predictions=352, training_performed=False, development_GT_arrays_read=False,
        quality_computed=False, quality_selection=False, capture_prefix=a.capture_prefix,
        resume_allowed=False, wall_cap_minutes=60, loader_source_receipt=support['binding'])
    if not a.execute:
        require(a.preflight and not a.preflight.exists(), 'Fresh CPU preflight receipt')
        write(a.preflight, result)
        return
    lock = support['modules'][3].GPU_LOCK.open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.output.mkdir(parents=True, exist_ok=False)
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('E2 fixed60min cap')))
        signal.alarm(60*60)
        try:
            execute(a, smoke, helper, support, members, rows, contract, prior, result)
        except BaseException as error:
            result.update(status='E2_PARTIAL_FAILED', error=str(error), traceback=traceback.format_exc())
            write(a.output/'results.json', result)
            raise
        finally:
            signal.alarm(0)
        write(a.output/'results.json', result)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
