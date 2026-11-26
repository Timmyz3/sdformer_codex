#!/usr/bin/env python3
"""R19: independently evaluated frozen compact graph, fixed122 bounded forwards.

R16 compact != expanded remains a failure; no third kernel or equivalence claim.
120 new quality endpoints plus two exact CC replays, no old quality/GT selection.
"""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
TARGET = 'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.mlp'
CANDIDATES = tuple((r, c) for r in range(3) for c in range(4))
ARMS = ('native', 'FF', 'CC') + tuple('fine_%d_%d' % p for p in CANDIDATES)
ATOL = RTOL = 2e-5
PRIMITIVE_SHA = 'e874e976ce4833a0638092ca78612e8224139eb1cc3b49d9ce0098488a3bb731'
SHARED_MATH_SHA = 'ad85fab7c0de64bad5b1fca15baca1af3347e32e226fb17ec18487485cc837ec'
GRAPH_ID = 'r19_W271482_all78current_s2b2_R16V2_compact'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def checked(spec):
    require(set(spec) == {'path', 'sha256'}, 'Path/SHA asset required')
    p = Path(spec['path'])
    require(p.is_file() and sha(p) == spec['sha256'], 'Actual asset differs: '+str(p))
    return p


def plan(rows, mode):
    require(mode == 'compact122' and len(rows) == 8, 'Fixed H8 and plan')
    result = []
    for i, row in enumerate(rows):
        for arm in ARMS:
            result.append(dict(target_id=row['target_id'], arm=arm, execution='compact' if arm != 'native' else 'native',
                               quality_endpoint=True, endpoint_arm=arm))
            if i == 0 and arm == 'CC':
                for repeat in (1, 2):
                    result.append(dict(target_id=row['target_id'], arm='CC_repeat%d'%repeat, execution='compact',
                                       quality_endpoint=False, endpoint_arm='CC'))
    require(len(result) == 122, 'Exactly122 actual forwards including two CC repeats')
    return [dict(p, forward_id=i+1) for i, p in enumerate(result)]


def mask_for(arm, torch):
    mask = torch.zeros((4, 5), dtype=torch.bool)
    if arm == 'FF':
        mask[:] = True
    elif arm.startswith('fine_'):
        r, c = map(int, arm.split('_')[1:])
        require((r, c) in CANDIDATES, 'Complete spatial candidate only')
        mask[r, c] = True
    else:
        require(arm in ('CC','CC_repeat1','CC_repeat2'), 'Unknown mask')
    return mask


def expected_rows(item):
    require(item['execution'] in ('native', 'compact'), 'No expanded path in R19')
    if item['execution'] == 'native' or item['arm'] == 'FF': return 558000
    if item['arm'].startswith('fine_'): return 548120
    require(item['arm'] in ('CC','CC_repeat1','CC_repeat2'), 'Expected actual compact arm')
    return 547400


def compare(a, b, torch):
    require(a.shape == b.shape and a.dtype == b.dtype, 'Numerical reference shape/dtype')
    require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), 'Finite numerical reference tensors')
    d = (a.double()-b.double()).abs()
    bad = d > ATOL+RTOL*b.double().abs()
    return {'pass': not bool(bad.any()), 'max_abs': float(d.max()), 'mean_abs': float(d.mean()),
            'outside_tolerance': int(bad.sum()), 'different_elements': int((d != 0).sum()),
            'elements': a.numel(), 'atol': ATOL, 'rtol': RTOL}


def prepare(a):
    import torch
    require(not torch.cuda.is_initialized(), 'CPU preflight only')
    require(not a.output.exists() and 0 < a.wall_minutes <= 20, 'Fresh output, bounded inference')
    require(sha(a.inputs) == a.inputs_sha256 and sha(a.freeze) == a.freeze_sha256, 'Explicit manifest/freeze bytes')
    inp, freeze = read(a.inputs), read(a.freeze)
    require(inp['schema'] == 'flow_r19_compact_graph_inputs_v1', 'Explicit R19 input schema')
    require(freeze['schema'] == 'flow_r19_compact_graph_source_freeze_v1', 'Frozen source schema')
    for spec in freeze['files']:
        checked(spec)
    require(any(Path(s['path']).resolve() == Path(__file__).resolve() for s in freeze['files']), 'Actual entrypoint must be frozen')
    admission_path = checked(inp['independent_admission'])
    primitive_path = checked(inp['primitive'])
    require(inp['BN_mode'] == 'NATIVE_DENSE_CURRENT', 'Explicit frozen V2 BN contract')
    require(sha(primitive_path) == PRIMITIVE_SHA and
            sha(primitive_path.parent/'compact_current_mlp.py') == SHARED_MATH_SHA, 'Unmodified R16 V2 and V1 mathematical dependency')
    admission = read(admission_path)
    require(admission['schema'] == 'flow_r19_independent_admission_v1' and admission['status'] == 'ADMIT'
            and a.plan in admission['permitted_plans'], 'Independent numerical admission required')
    require(admission['runner_sha256'] == sha(__file__) and admission['core_sha256'] == sha(primitive_path)
            and admission['BN_mode'] == inp['BN_mode'], 'Admission binds actual math/runner')
    require(any(Path(s['path']).resolve() == primitive_path.resolve() for s in freeze['files']), 'Actual primitive must be frozen')
    require(any(Path(s['path']).resolve() == (primitive_path.parent/'compact_current_mlp.py').resolve() for s in freeze['files']), 'Frozen shared V1 math dependency')
    old = Path(inp['old_root'])
    conversion = old/'bn_conversion_round14'
    sys.path.insert(0, str(conversion))
    r14 = load(conversion/'run_conversion_v2.py', '_r19_original_source_helpers')
    k, h0, modules = r14.runtime()
    rtpath, memberpath = checked(inp['runtime']), checked(inp['membership'])
    h8path, regionpath, cp = checked(inp['h8']), checked(inp['regions']), checked(inp['checkpoint'])
    rt, members, h8, regions = map(read, (rtpath, memberpath, h8path, regionpath))
    require(sha(memberpath) == k.MEMBER_SHA and sha(regionpath) == k.REGION_SHA, 'Fixed membership/regions')
    require(h8 == h0.h8_manifest(members, sha(memberpath)), 'Exact identity-only H8 selector')
    require(sha(cp) == k.CHECKPOINT_SHA, 'Only original W271482')
    require(sha(cp.with_suffix('.json')) == k.SIDECAR_SHA and sha(cp.parent/'run_contract.json') == k.RUN_CONTRACT_SHA, 'Checkpoint sidecar/run source')
    for path, digest in rt['source_pins'].items():
        require(sha(path) == digest, 'Frozen native dependency '+path)
    sys.path[:0] = [rt['breadth_root'], rt['motion_root']]
    base = load(Path(rt['breadth_root'])/'evaluate_breadth.py', '_r19_native_preflight')
    args = list(rt['base_args'])
    require(not any(x.startswith('--bn-') or x.startswith('--svs-') or x.startswith('--regions') or x == '--run-inference' for x in args), 'No implicit legacy experiment')
    for flag, value in (('--checkpoint', str(cp)), ('--checkpoint-sha256', k.CHECKPOINT_SHA), ('--output', str(a.output/'results.json'))):
        require(args.count(flag) == 1, 'Single native argument '+flag)
        args[args.index(flag)+1] = value
    b = base.parse_args(args)
    b.device = 'cuda:0'
    context = list(base.preflight(b))
    native, trainer, metrics, _, hashes, config, run, side, _, _ = context
    canonical = metrics.canonical_rows(b.canonical)
    roles = metrics.auxiliary_roles(b.roles, canonical)
    modules[0].validate_members(members, canonical, roles)
    r14.validate_manifest_sources(members, b.canonical, b.roles)
    metrics.validate_region_contract(regions, canonical, roles, require_complete_fit=False)
    require(regions['fit_target_ids'] == [r['target_id'] for r in members['train_members_diversity']], 'Fixed training region fit')
    context[8] = regions
    cp_audit = r14.validate_checkpoint(cp, '271482', tuple(context), h0, torch)
    evidence = []
    for row in h8['members']:
        files = native.target_files(row)
        entry = {}
        for kind, relative in files.items():
            require(sha(native.checked_path(b.data_root, relative)) == hashes[relative]['sha256'], 'Actual H8 input bytes')
            entry['events' if kind == 'input' else kind] = hashes[relative]['sha256']
        evidence.append(dict(target_id=row['target_id'], input_evidence=entry))
    binding = {'graph_id': GRAPH_ID, 'plan': a.plan, 'forward_plan': plan(h8['members'], a.plan), 'checkpoint_cpu_audit': cp_audit,
               'h8_sha256': sha(h8path), 'membership_sha256': sha(memberpath), 'region_sha256': sha(regionpath),
               'inputs_sha256': a.inputs_sha256, 'source_freeze_sha256': a.freeze_sha256,
               'independent_admission_sha256': sha(admission_path), 'other_11_MLPs': 'native',
               'hardware_mapping_status': 'Independent compact algorithm graph; R16 compact-to-expanded equivalence remains failed; no hardware claim',
               'compact_forwards': 114, 'native_forwards': 8, 'quality_endpoints': 120, 'determinism_replays': 2, 'target': TARGET,
               'BN': 'all78_current', 'quality_interpretation': a.plan == 'compact122', 'optimizer_steps': 0,
               'primitive_sha256': sha(primitive_path), 'BN_mode': inp['BN_mode'],
               'shared_math_sha256': SHARED_MATH_SHA, 'expected_FC_rows_each': 67021520, 'expected_BN_calls': 9516,
               'calibration_forwards': 0, 'precision': 'FP32; no AMP/TF32; T10/B1/full9', 'wall_minutes': a.wall_minutes}
    return dict(binding=binding, data_evidence=evidence, h8=h8, context=tuple(context), base=base, b=b,
                rt=rt, r14=r14, k=k, h0=h0, modules=modules, primitive_path=primitive_path)


def forward(model, functional, targets, shapes, chunk, item, primitive, BN_mode, torch):
    require(item['execution'] in ('native', 'compact'), 'R19 never calls expanded math')
    require(BN_mode == 'NATIVE_DENSE_CURRENT', 'Fixed original V2 BN contract')
    functional.reset_net(model)
    handles, captured = [], {}
    observed = {name: {'fc1_rows': 0, 'fc2_rows': 0} for name in targets}
    bn_calls, bn_shapes = {}, {}
    target = targets[TARGET]
    parent = model.get_submodule(TARGET.rsplit('.', 1)[0])
    attr = TARGET.rsplit('.', 1)[1]
    audit = {}
    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.native = target
        def forward(self, u):
            mask = mask_for(item['arm'], torch)
            require(item['execution'] == 'compact', 'Compact wrapper only; FF has no shortcut')
            branch, local = primitive.compact_mlp(self.native, u, mask)
            require(local['BN_mode'] == BN_mode and local['each_cell_BN1_and_BN2_bitwise_equal'], 'Original within-cell gather contract')
            for key in ('representatives','statistics_barriers','FC1_retained_elements','FC2_retained_elements',
                        'full_T_and_all_channels','native_MLP_shortcut','schedule','BN_mode','native_BN_calls',
                        'full_domain_BN_elements','expand_written_elements','BN_output_dense_elements',
                        'production_BN1_gather_read_elements','production_BN1_gather_written_elements',
                        'diagnostic_BN2_gather_read_elements','cell_exact_audit_read_elements',
                        'each_cell_BN1_and_BN2_bitwise_equal'):
                audit[key] = local[key]
            audit['cells'] = local['cells']
            audit['areas'] = local['areas'].detach().cpu().tolist()
            audit['packed_input_shape'] = list(local['packed_input'].shape)
            audit['packed_input_sha256'] = tensor_sha(local['packed_input'])
            audit['diagnostic_moments_full_domain_elements_per_scan'] = local['full_domain_BN_elements']
            audit['diagnostic_moments_scans'] = 2
            return branch
    active = target if item['execution'] == 'native' else Wrapper().eval()
    def store_target(module, ins, out):
        captured['u'] = ins[0].detach().cpu().clone()
        captured['branch'] = out.detach().cpu().clone()
    try:
        for name, module in targets.items():
            for fc in ('fc1', 'fc2'):
                def count(module, ins, name=name, fc=fc):
                    observed[name][fc+'_rows'] += ins[0].numel()//ins[0].shape[-1]
                handles.append(getattr(module, fc).register_forward_pre_hook(count))
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                bn_calls[name] = 0
                bn_shapes[name] = []
                def bn_count(module, ins, name=name):
                    bn_calls[name] += 1
                    bn_shapes[name].append({'shape': list(ins[0].shape), 'dtype': str(ins[0].dtype), 'stride': list(ins[0].stride())})
                handles.append(module.register_forward_pre_hook(bn_count))
        for sn in ('sn1', 'sn2'):
            def store_sn(module, ins, out, sn=sn):
                require(sn not in captured, 'One actual target PSN invocation')
                captured[sn] = out.detach().cpu().clone()
            handles.append(getattr(target, sn).register_forward_hook(store_sn))
        handles.append(active.register_forward_hook(store_target))
        setattr(parent, attr, active)
        with torch.inference_mode():
            output = model(chunk)
        flows = [x.detach().cpu().clone() for x in output['flow']]
        require(len(flows) == 4 and all(bool(torch.isfinite(x).all()) for x in flows), 'Four finite native flow heads')
        require(all(bool(torch.isfinite(x).all()) for x in captured.values()), 'Finite target inputs/branches/spikes')
        if item['execution'] == 'compact':
            for sn in ('sn1','sn2'):
                captured[sn+'_packed'] = captured[sn]
                captured[sn] = primitive.expand(captured[sn], audit['cells'], 30, 40)
        for name, counts in observed.items():
            expected = expected_rows(item)-546000 if name == TARGET else math.prod(shapes[name][:-1])
            require(counts['fc1_rows'] == counts['fc2_rows'] == expected, 'Actual module FC rows differ '+name)
        require(sum(bn_calls.values()) == 78, 'Actual native BN calls differ')
        for name, count in bn_calls.items():
            require(count == 1, 'Per-module BN execution '+name)
        record = dict(item, all12_observed_fc_rows=observed, native_BN_calls=bn_calls, native_BN_inputs=bn_shapes, target_execution=audit,
                      summed_fc1_rows=sum(v['fc1_rows'] for v in observed.values()),
                      summed_fc2_rows=sum(v['fc2_rows'] for v in observed.values()),
                      tensor_records={k: {'shape': list(v.shape), 'dtype':str(v.dtype), 'sha256': tensor_sha(v)} for k, v in captured.items()},
                      head_records=[{'shape': list(x.shape), 'sha256': tensor_sha(x)} for x in flows])
        return flows, captured, record
    finally:
        setattr(parent, attr, target)
        for handle in handles:
            handle.remove()
        require(all(model.get_submodule(n) is m for n, m in targets.items()), 'Original modules restored')
        require(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()), 'All hooks removed')


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def numerical_pair(actual, reference, torch):
    af, ac, ar = actual
    rf, rc, rr = reference
    continuous = {'branch': compare(ac['branch'], rc['branch'], torch)}
    continuous.update({'head%d' % i: compare(x, y, torch) for i, (x, y) in enumerate(zip(af, rf))})
    spikes = {sn: int((ac[sn] != rc[sn]).sum()) for sn in ('sn1', 'sn2')}
    same_u = torch.equal(ac['u'], rc['u'])
    return {'pass': same_u and all(v['pass'] for v in continuous.values()) and not any(spikes.values()),
            'same_u_exact': same_u, 'continuous': continuous, 'PSN_flips': spikes}


def deterministic_pair(actual, reference, torch):
    """All captured native arrays, including packed spikes, compared by bytes."""
    af,ac,_=actual;rf,rc,_=reference
    require(len(af)==len(rf)==4 and set(ac)==set(rc)=={'u','branch','sn1','sn2','sn1_packed','sn2_packed'}, 'Complete compact replay arrays')
    aa=dict(ac,**{'head%d'%i:x for i,x in enumerate(af)})
    rr=dict(rc,**{'head%d'%i:x for i,x in enumerate(rf)})
    checks={}
    for name,x in aa.items():
        y=rr[name]
        require(x.shape==y.shape and x.dtype==y.dtype, 'Replay shape/dtype '+name)
        xb=x.contiguous().view(torch.uint8);yb=y.contiguous().view(torch.uint8)
        checks[name]={'bitwise_equal':bool(torch.equal(xb,yb)),
                      'different_bytes':int((xb!=yb).sum()),'shape':list(x.shape),'dtype':str(x.dtype),
                      'actual_sha256':tensor_sha(x),'reference_sha256':tensor_sha(y)}
    return {'pass':all(v['bitwise_equal'] for v in checks.values()),'array_checks':checks}


def execute(a, p, result):
    import numpy as np
    import torch
    require(not torch.cuda.is_initialized(), 'CUDA starts only inside shared lock')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261017)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(20261017)
    begin = time.monotonic()
    rt, base, b, context = p['rt'], p['base'], p['b'], p['context']
    native, trainer, metrics, _, hashes, config, run, side, regions, _ = context
    r13, collector, probe, parent, r9, r8 = p['modules']
    loader = load(Path(rt['breadth_root'])/'run_same_shape_reference.py', '_r19_native_loader')
    scope = load(Path(rt['breadth_root'])/'run_bn_scope.py', '_r19_inventory')
    sys.path.insert(0, str(p['primitive_path'].parent))
    primitive = load(p['primitive_path'], '_r19_frozen_v2_compact_graph')
    model, functional = loader.load_model(b, context, base, torch, device)
    require(r13.state_digest(model) == p['k'].SOURCE_STATE_SHA, 'Actual original model state')
    bn = p['r14'].core.switch_allcurrent(model, base, scope, r13.state_digest, torch, p['k'].SOURCE_NON_BN_SHA)
    targets, shapes = parent.target_layers(model)
    require(tuple(shapes[TARGET]) == (10, 1, 30, 40, 384), 'One fixed target geometry')
    before, signature = r13.state_digest(model), base.bn_signature(model, torch)
    result.update(status='RUNNING', inference_executed=True, runtime={'torch': str(torch.__version__), 'numpy': np.__version__,
                  'device': torch.cuda.get_device_name(device), 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                  'TF32': False, 'AMP': False}, BN_scope=bn, initial_allcurrent_state_sha256=before,
                  forward_records=[], numerical_checks=[])
    rows = {r['target_id']: r for r in p['h8']['members']}
    last_id = None
    native_output = None
    first_cc_output = None
    for item in p['binding']['forward_plan']:
        require(time.monotonic()-begin < a.wall_minutes*60, 'Fixed wall cap')
        tid = item['target_id']
        if tid != last_id:
            row = rows[tid]
            files = native.target_files(row)
            evidence = {'events' if q == 'input' else q: hashes[v]['sha256'] for q, v in files.items()}
            eventpath = native.checked_path(b.data_root, files['input'])
            require(sha(eventpath) == evidence['events'], 'Actual event bytes at inference')
            events = np.load(eventpath, allow_pickle=False)
            require(events.shape == (10, 480, 640) and events.dtype == np.float32 and np.isfinite(events).all(), 'Native event tensor')
            chunk = base.prepare_input(events, device, torch)
            del events
            last_id = tid
            native_output = None
        result['full_forward_attempts'] += 1
        write(a.output/'results.json', result)
        output = forward(model, functional, targets, shapes, chunk, item, primitive, p['binding']['BN_mode'], torch)
        flows, captures, record = output
        record['input_evidence'] = evidence
        name = '%03d_%s_%s_%s.npz' % (item['forward_id'], tid, item['arm'], item['execution'])
        save = {'head%d' % i: x.numpy() for i, x in enumerate(flows)}
        # Every action saves all heads; every native/FF pair also saves actual
        # input, branch, and both PSN outputs for independent rechecking.
        if item['arm'] in ('native', 'FF') or (tid == p['h8']['members'][0]['target_id'] and item['arm'].startswith('CC')):
            save.update({k: v.numpy() for k, v in captures.items()})
        np.savez_compressed(a.output/name, **save)
        record.update(prediction_file=name, prediction_file_sha256=sha(a.output/name))
        result['forward_records'].append(record)
        result['full_forward_count'] += 1
        if item['execution'] == 'native':
            native_output = output
        else:
            require(native_output is not None and tensor_sha(captures['u']) == tensor_sha(native_output[1]['u']), 'All actions share exact predecision input bytes')
        check = None
        if item['arm'] == 'FF':
            check = dict(kind='allfine_vs_native', arm='FF', **numerical_pair(output, native_output, torch))
        elif item['arm'] == 'CC' and tid == p['h8']['members'][0]['target_id']:
            first_cc_output = output
        elif item['arm'].startswith('CC_repeat'):
            require(first_cc_output is not None, 'Actual first CC reference must exist')
            check = dict(kind='CC_exact_replay', arm=item['arm'], reference_forward_id=first_cc_output[2]['forward_id'],
                         **deterministic_pair(output, first_cc_output, torch))
        if check:
            result['numerical_checks'].append(dict(target_id=tid, forward_id=item['forward_id'], **check))
        p['k'].core.check_closure(model, targets, before, r13.state_digest, signature, base.bn_signature, torch)
        record['state_and_BN_unchanged'] = True
        record['original_modules_restored_and_hooks_removed'] = True
        write(a.output/'results.json', result)
        print(json.dumps({'forward': item['forward_id'], 'plan': a.plan, 'arm': item['arm'], 'execution': item['execution'],
                          'numeric_pass': check['pass'] if check else None}), flush=True)
        require(check is None or check['pass'], 'Predeclared numerical contract failed; no action-value interpretation')
    require(result['full_forward_count'] == len(p['binding']['forward_plan']), 'Complete fixed plan')
    require(sum(r['quality_endpoint'] for r in result['forward_records']) == 120 and len(result['numerical_checks']) == 10, 'Full quality/numerical closure')
    result.update(status='COMPACT_GRAPH122_COMPLETE',
                  final_state_sha256=r13.state_digest(model), all_states_unchanged=True, all_hooks_removed=True,
                  original_modules_restored=True, wall_seconds=time.monotonic()-begin,
                  max_allocated_bytes=torch.cuda.max_memory_allocated(), max_reserved_bytes=torch.cuda.max_memory_reserved())
    # Quality is intentionally reduced separately after execution/numeric audit.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--inputs-sha256', required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--freeze-sha256', required=True)
    parser.add_argument('--plan', choices=('compact122',), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--wall-minutes', type=float, default=20)
    parser.add_argument('--preflight-output', type=Path)
    parser.add_argument('--execute', action='store_true')
    a = parser.parse_args()
    p = prepare(a)
    result = dict(schema='flow_r19_compact_graph_v1', graph_id=GRAPH_ID, status='PREFLIGHT_PASS_NO_INFERENCE', binding=p['binding'],
                  data_evidence=p['data_evidence'], inference_executed=False, full_forward_count=0, full_forward_attempts=0,
                  training_performed=False, optimizer_steps=0, calibration_forwards=0,
                  flow_specific_increment_claim=False, hardware_benefit_claim=False)
    if not a.execute:
        require(a.preflight_output and not a.preflight_output.exists(), 'Fresh preflight receipt')
        write(a.preflight_output, result)
        print(result['status'])
        return
    lock = p['modules'][3].GPU_LOCK.open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.output.mkdir(parents=True, exist_ok=False)
        write(a.output/'results.json', result)
        def alarm(signum, frame):
            raise TimeoutError('R19 hard wall cap')
        signal.signal(signal.SIGALRM, alarm)
        signal.alarm(int(a.wall_minutes*60)+60)
        try:
            execute(a, p, result)
        except BaseException as exc:
            result.update(status='INCOMPLETE_OR_NUMERICAL_FAILURE', error=str(exc), traceback=traceback.format_exc())
            write(a.output/'results.json', result)
            raise
        finally:
            signal.alarm(0)
        write(a.output/'results.json', result)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
