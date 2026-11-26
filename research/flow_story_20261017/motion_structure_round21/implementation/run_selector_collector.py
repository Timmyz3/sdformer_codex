#!/usr/bin/env python3
"""R21: fixed fresh fit/development action-label collector, 362 bounded forwards.

Unchanged R16 V2 compact graph; no coarse-expanded equivalence claim.
360 new quality endpoints plus two exact background replays; no training/GT selection.
"""
import argparse
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
ROLE_SHA = 'e6f216f6b017fc2cbba262ec51c70474f342e593e1549058a2200970949e10a1'
BACKGROUND_ROWS = {'CC': 547400, 'T8': 549360}
TOTAL_ROWS = {'CC': 198874960, 'T8': 199490400}
ATOL = RTOL = 2e-5
PRIMITIVE_SHA = 'e874e976ce4833a0638092ca78612e8224139eb1cc3b49d9ce0098488a3bb731'
SHARED_MATH_SHA = 'ad85fab7c0de64bad5b1fca15baca1af3347e32e226fb17ec18487485cc837ec'
def graph_id(background):
    require(background in BACKGROUND_ROWS, 'Explicit supported background')
    return 'r21_W271482_all78current_s2b2_R16V2_'+background+'_fit16dev8'


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


def load_frozen_primitive(path):
    path=Path(path)
    shared_path=path.parent/'compact_current_mlp.py'
    require(sha(path)==PRIMITIVE_SHA and sha(shared_path)==SHARED_MATH_SHA, 'Frozen primitive bytes')
    sys.path.insert(0,str(path.parent))
    primitive=load(path,'_r21_frozen_v2_compact_graph')
    shared=sys.modules.get('compact_current_mlp')
    require(shared is not None and hasattr(shared,'__file__') and
            Path(shared.__file__).resolve()==shared_path.resolve() and sha(shared.__file__)==SHARED_MATH_SHA,
            'Actually imported shared mathematical module path/SHA')
    require(all(getattr(primitive,key) is getattr(shared,key) for key in ('validate','pack','expand')),
            'Actual V2 imported symbols bind frozen shared module')
    return primitive


def checked(spec):
    require(set(spec) == {'path', 'sha256'}, 'Path/SHA asset required')
    p = Path(spec['path'])
    require(p.is_file() and sha(p) == spec['sha256'], 'Actual asset differs: '+str(p))
    return p


def arms(background):
    require(background in BACKGROUND_ROWS, 'Explicit supported background')
    return ('native', 'FF', background) + tuple('fine_%d_%d'%p for p in CANDIDATES)


def plan(rows, mode, background, role_map):
    require(mode == 'selector362' and len(rows) == 24, 'Fixed fit16/development8 and plan')
    require(len({r['target_id'] for r in rows}) == 24 and set(role_map) == {r['target_id'] for r in rows}, 'Exact distinct selector membership')
    require(list(role_map[r['target_id']] for r in rows) == ['fit']*16+['development']*8, 'Fixed fit then development order')
    result = []
    for i, row in enumerate(rows):
        for arm in arms(background):
            result.append(dict(target_id=row['target_id'], selector_role=role_map[row['target_id']], arm=arm,
                               background=background, execution='native' if arm == 'native' else 'compact',
                               quality_endpoint=True, endpoint_arm=arm))
            if i == 0 and arm == background:
                for repeat in (1, 2):
                    result.append(dict(target_id=row['target_id'], selector_role=role_map[row['target_id']],
                                       arm=background+'_repeat%d'%repeat, background=background, execution='compact',
                                       quality_endpoint=False, endpoint_arm=background))
    require(len(result) == 362, 'Exactly362 actual forwards including two background repeats')
    return [dict(p, forward_id=i+1) for i, p in enumerate(result)]


def mask_for(arm, torch, background):
    require(background in BACKGROUND_ROWS, 'Explicit supported background')
    mask = torch.zeros((4, 5), dtype=torch.bool)
    if arm == 'FF':
        mask[:] = True
        return mask
    if background == 'T8':
        mask[:] = True
        mask[:3, :4] = False
    if arm.startswith('fine_'):
        r,c=map(int,arm.split('_')[1:])
        require((r,c) in CANDIDATES, 'Only one complete spatial tile')
        mask[r,c]=True
    else:
        require(arm in (background, background+'_repeat1', background+'_repeat2'), 'Mask outside frozen background family')
    return mask


def expected_rows(item):
    require(item['execution'] in ('native', 'compact') and item['background'] in BACKGROUND_ROWS, 'Frozen collector execution/background')
    if item['execution'] == 'native':
        require(item['arm'] == 'native', 'Native execution only for native arm')
        return 558000
    if item['arm'] == 'FF': return 558000
    if item['arm'].startswith('fine_'):
        require(tuple(map(int,item['arm'].split('_')[1:])) in CANDIDATES, 'Complete tile geometry')
        return BACKGROUND_ROWS[item['background']]+720
    require(item['arm'] in (item['background'], item['background']+'_repeat1', item['background']+'_repeat2'), 'Frozen compact background')
    return BACKGROUND_ROWS[item['background']]


def save_captures(item, first_target_id):
    return item['arm'] in ('native', 'FF') or (
        item['target_id'] == first_target_id and item['endpoint_arm'] == item['background'])


def payload_names(item, first_target_id):
    if save_captures(item,first_target_id):
        return {'head%d'%i for i in range(4)} | {'u','branch','sn1','sn2'} | ({'sn1_packed','sn2_packed'} if item['execution']=='compact' else set())
    return {'head3'}


def background_contract(background, inp, freeze, admission):
    require(background in BACKGROUND_ROWS and inp['background'] == freeze['background'] == background,
            'One background bound to CLI/input/sourcefreeze')
    require(admission['permitted_backgrounds'] == [background], 'One independently admitted background')


def compare(a, b, torch):
    require(a.shape == b.shape and a.dtype == b.dtype, 'Numerical reference shape/dtype')
    require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()), 'Finite numerical reference tensors')
    d = (a.double()-b.double()).abs()
    bad = d > ATOL+RTOL*b.double().abs()
    return {'pass': not bool(bad.any()), 'max_abs': float(d.max()), 'mean_abs': float(d.mean()),
            'outside_tolerance': int(bad.sum()), 'different_elements': int((d != 0).sum()),
            'elements': a.numel(), 'atol': ATOL, 'rtol': RTOL}


def selector_members(selector, canonical, hashes, native):
    """Only inspect frozen identity/acceptance metadata; never visit array paths."""
    require(selector['schema'] == 'flow_r20_next_selector_identity_roles_v1' and
            selector['status'] == 'FROZEN_METADATA_ONLY', 'Frozen selector metadata schema')
    groups = selector['groups']
    require(set(groups) == {'fit', 'development', 'sealed_confirmation'} and
            selector['group_counts'] == {'fit':16, 'development':8, 'sealed_confirmation':8}, 'Exact role population')
    require(all(len(groups[k]) == selector['group_counts'][k] for k in groups), 'Actual role counts')
    all_ids = [r['target_id'] for group in groups.values() for r in group]
    require(len(set(all_ids)) == len(all_ids) == 32, 'Roles must be pairwise disjoint')
    selected = groups['fit'] + groups['development']
    role_map = {r['target_id']:role for role in ('fit','development') for r in groups[role]}
    records = {r['target_id']:r for r in selector['records']}
    require(len(records) == len(selector['records']) == 32 and set(records) == set(all_ids), 'Complete unique role records')
    by_id = {r['target_id']:r for r in canonical}
    for row in selected:
        tid = row['target_id']; record = records[tid]
        require(row == by_id[tid] and row['protocol_split'] == 'train' and
                record['canonical'] == row and record['role'] == role_map[tid], 'Canonical selected identity/role')
        proof = {v['category']:v for v in record['accepted_array_metadata']}
        require(len(record['accepted_array_metadata']) == len(proof) == 3 and set(proof) == {'input','gt','mask'}, 'Three accepted array metadata entries')
        for kind, relative in native.target_files(row).items():
            item = proof[kind]
            require(item['relative_path'] == relative and item['target'] == tid and item['split'] == 'train' and
                    item['sequence'] == row['sequence'] and item['sha256'] == hashes[relative]['sha256'] and
                    int(item['bytes']) == hashes[relative]['bytes'], 'Selected acceptance identity/path/SHA/size')
    return selected, role_map


def source_preflight(b, base):
    """Reuse frozen source/encoding/checkpoint contracts without stat-ing sealed arrays.

    Full accepted dataset metadata stays validated. Only the allowed24 actual
    files are visited separately by verify_selected_files; this is not training.
    """
    require(b.geometry == 'full9', 'Only full9 original geometry')
    native, trainer, metrics, pins = base.support_modules(b.prior_root)
    train, hashes, binding, audit = native.validate_native_bundle(
        b.data_root, b.canonical, b.acceptance, b.acceptance_sha256, b.archive_audit,
        check_current_files=False)
    binding.update(entrypoint_sha256=pins['baseline/train_native_baseline.py'],
                   adapter_sha256=pins['baseline/native_data_acceptance.py'],
                   training_core_sha256=trainer.CORE_SHA, checkpoint_schema=base.SCHEMA,
                   checkpoint_creator=base.CREATOR)
    core = trainer.load_core()
    sources = native.pinned_json(b.source_manifest, core.SOURCE_MANIFEST_SHA)
    require(len(sources) == 63, 'Expected63 native source files')
    for relative, digest in sources.items():
        require(sha(native.checked_path(b.source_root, relative)) == digest, 'Native source differs '+relative)
    require(sha(b.checkpoint) == b.checkpoint_sha256, 'Explicit original checkpoint bytes')
    side = read(b.checkpoint.with_suffix('.json'))
    require(side['checkpoint_sha256'] == b.checkpoint_sha256 and side['checkpoint_file'] == b.checkpoint.name, 'Checkpoint sidecar binding')
    run = read(b.checkpoint.parent/'run_contract.json')
    config = base.validate_contract(run, side, binding, sources, trainer, train, hashes)
    receipt = {'full_accepted_metadata_validated':True, 'all_dataset_current_file_stat_checked':False,
               'accepted_metadata_audit':audit, 'allowed24_current_files_checked_separately':True,
               'sealed_confirmation_array_paths_visited':0, 'GT_array_loads':0}
    return native, trainer, metrics, [], hashes, config, run, side, None, receipt


def verify_selected_files(selected, native, root, hashes, role_map):
    require(len(selected) == len(role_map) == 24 and set(role_map) == {r['target_id'] for r in selected}, 'Only allowed24 file scope')
    evidence = []
    for row in selected:
        entry = {}
        for kind, relative in native.target_files(row).items():
            path = native.checked_path(root, relative)
            require(path.is_file() and path.stat().st_size == hashes[relative]['bytes'] and
                    sha(path) == hashes[relative]['sha256'], 'Actual allowed24 input/GT/mask size/SHA')
            entry['events' if kind == 'input' else kind] = hashes[relative]['sha256']
        require(set(entry) == {'events','gt','mask'}, 'All three selected source files')
        evidence.append(dict(target_id=row['target_id'], selector_role=role_map[row['target_id']], input_evidence=entry))
    return evidence


def prepare(a):
    import torch
    require(not torch.cuda.is_initialized(), 'CPU preflight only')
    require(not a.output.exists() and 0 < a.wall_minutes <= 45, 'Fresh output, bounded inference')
    require(sha(a.inputs) == a.inputs_sha256 and sha(a.freeze) == a.freeze_sha256, 'Explicit manifest/freeze bytes')
    inp, freeze = read(a.inputs), read(a.freeze)
    require(inp['schema'] == 'flow_r21_selector_collector_inputs_v1', 'Explicit R21 input schema')
    require(freeze['schema'] == 'flow_r21_selector_collector_source_freeze_v1', 'Frozen source schema')
    for spec in freeze['files']:
        checked(spec)
    require(any(Path(s['path']).resolve() == Path(__file__).resolve() for s in freeze['files']), 'Actual entrypoint must be frozen')
    admission_path = checked(inp['independent_admission'])
    primitive_path = checked(inp['primitive'])
    require(inp['BN_mode'] == 'NATIVE_DENSE_CURRENT', 'Explicit frozen V2 BN contract')
    require(sha(primitive_path) == PRIMITIVE_SHA and
            sha(primitive_path.parent/'compact_current_mlp.py') == SHARED_MATH_SHA, 'Unmodified R16 V2 and V1 mathematical dependency')
    load_frozen_primitive(primitive_path)  # CPU import identity check; no model/forward/CUDA.
    admission = read(admission_path)
    background_contract(a.background, inp, freeze, admission)
    require(admission['schema'] == 'flow_r21_independent_admission_v1' and admission['status'] == 'ADMIT'
            and a.plan in admission['permitted_plans'] and admission['permitted_backgrounds'] == [a.background], 'Independent numerical admission required')
    require(admission['runner_sha256'] == sha(__file__) and admission['core_sha256'] == sha(primitive_path)
            and admission['BN_mode'] == inp['BN_mode'], 'Admission binds actual math/runner')
    require(any(Path(s['path']).resolve() == primitive_path.resolve() for s in freeze['files']), 'Actual primitive must be frozen')
    require(any(Path(s['path']).resolve() == (primitive_path.parent/'compact_current_mlp.py').resolve() for s in freeze['files']), 'Frozen shared V1 math dependency')
    old = Path(inp['old_root'])
    conversion = old/'bn_conversion_round14'
    sys.path.insert(0, str(conversion))
    r14 = load(conversion/'run_conversion_v2.py', '_r21_original_source_helpers')
    k, h0, modules = r14.runtime()
    rtpath, memberpath = checked(inp['runtime']), checked(inp['membership'])
    rolepath, regionpath, cp = checked(inp['selector_roles']), checked(inp['regions']), checked(inp['checkpoint'])
    require(sha(rolepath) == ROLE_SHA, 'Frozen identity-only fit/development/confirmation roles')
    rt, members, selector, regions = map(read, (rtpath, memberpath, rolepath, regionpath))
    require(sha(memberpath) == k.MEMBER_SHA and sha(regionpath) == k.REGION_SHA, 'Fixed membership/regions')
    require(sha(cp) == k.CHECKPOINT_SHA, 'Only original W271482')
    require(sha(cp.with_suffix('.json')) == k.SIDECAR_SHA and sha(cp.parent/'run_contract.json') == k.RUN_CONTRACT_SHA, 'Checkpoint sidecar/run source')
    for path, digest in rt['source_pins'].items():
        require(sha(path) == digest, 'Frozen native dependency '+path)
    sys.path[:0] = [rt['breadth_root'], rt['motion_root']]
    base = load(Path(rt['breadth_root'])/'evaluate_breadth.py', '_r21_native_preflight')
    args = list(rt['base_args'])
    require(not any(x.startswith('--bn-') or x.startswith('--svs-') or x.startswith('--regions') or x == '--run-inference' for x in args), 'No implicit legacy experiment')
    for flag, value in (('--checkpoint', str(cp)), ('--checkpoint-sha256', k.CHECKPOINT_SHA), ('--output', str(a.output/'results.json'))):
        require(args.count(flag) == 1, 'Single native argument '+flag)
        args[args.index(flag)+1] = value
    b = base.parse_args(args)
    b.device = 'cuda:0'
    context = list(source_preflight(b, base))
    native, trainer, metrics, _, hashes, config, run, side, _, _ = context
    canonical = metrics.canonical_rows(b.canonical)
    roles = metrics.auxiliary_roles(b.roles, canonical)
    modules[0].validate_members(members, canonical, roles)
    r14.validate_manifest_sources(members, b.canonical, b.roles)
    metrics.validate_region_contract(regions, canonical, roles, require_complete_fit=False)
    require(regions['fit_target_ids'] == [r['target_id'] for r in members['train_members_diversity']], 'Fixed training region fit')
    context[8] = regions
    selected, role_map = selector_members(selector, canonical, hashes, native)
    context[3] = selected
    cp_audit = r14.validate_checkpoint(cp, '271482', tuple(context), h0, torch)
    evidence = verify_selected_files(selected, native, b.data_root, hashes, role_map)
    context[9]['allowed24_current_files_SHA_checked'] = 72
    binding = {'graph_id': graph_id(a.background), 'plan': a.plan, 'background': a.background, 'forward_plan': plan(selected, a.plan, a.background, role_map), 'checkpoint_cpu_audit': cp_audit,
               'selector_roles_sha256': sha(rolepath), 'selected_target_ids': [r['target_id'] for r in selected], 'selector_role_map': role_map, 'confirmation_array_accesses': 0, 'membership_sha256': sha(memberpath), 'region_sha256': sha(regionpath),
               'inputs_sha256': a.inputs_sha256, 'source_freeze_sha256': a.freeze_sha256,
               'independent_admission_sha256': sha(admission_path), 'other_11_MLPs': 'native',
               'hardware_mapping_status': 'Independent compact algorithm graph; R16 compact-to-expanded equivalence remains failed; no hardware claim',
               'compact_forwards': 338, 'native_forwards': 24, 'quality_endpoints': 360, 'determinism_replays': 2, 'target': TARGET,
               'BN': 'all78_current', 'quality_interpretation': a.plan == 'selector362', 'optimizer_steps': 0,
               'primitive_sha256': sha(primitive_path), 'BN_mode': inp['BN_mode'],
               'shared_math_sha256': SHARED_MATH_SHA, 'expected_FC_rows_each': TOTAL_ROWS[a.background], 'expected_BN_calls': 28236,
               'data_preflight': context[9], 'calibration_forwards': 0, 'precision': 'FP32; no AMP/TF32; T10/B1/full9', 'wall_minutes': a.wall_minutes}
    return dict(binding=binding, data_evidence=evidence, selected=selected, role_map=role_map, context=tuple(context), base=base, b=b,
                rt=rt, r14=r14, k=k, h0=h0, modules=modules, primitive_path=primitive_path)


def forward(model, functional, targets, shapes, chunk, item, primitive, BN_mode, torch):
    require(item['execution'] in ('native', 'compact'), 'R21 never calls expanded math')
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
            mask = mask_for(item['arm'], torch, item['background'])
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
                      head_records=[{'shape': list(x.shape), 'dtype':str(x.dtype), 'sha256': tensor_sha(x)} for x in flows])
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


def numerical_check(item, output, native_output, replay_references, first_target_id, torch):
    """One matching first-frame reference per replay family; no cross-family reuse."""
    if item['arm'] == 'FF':
        return dict(kind='allfine_vs_native', arm='FF', **numerical_pair(output, native_output, torch))
    if item['arm'] == item['background'] and item['target_id'] == first_target_id:
        require(item['arm'] not in replay_references, 'Single original replay reference')
        replay_references[item['arm']] = output
    elif '_repeat' in item['arm']:
        endpoint = item['endpoint_arm']
        require(item['target_id'] == first_target_id and endpoint in replay_references,
                'Actual first matching background reference must exist')
        reference = replay_references[endpoint]
        require(reference[2]['arm'] == endpoint and reference[2]['target_id'] == item['target_id'],
                'Replay reference arm/frame binding')
        return dict(kind=endpoint+'_exact_replay', arm=item['arm'], reference_forward_id=reference[2]['forward_id'],
                    **deterministic_pair(output, reference, torch))
    return None


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
    loader = load(Path(rt['breadth_root'])/'run_same_shape_reference.py', '_r21_native_loader')
    scope = load(Path(rt['breadth_root'])/'run_bn_scope.py', '_r21_inventory')
    primitive = load_frozen_primitive(p['primitive_path'])
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
    rows = {r['target_id']: r for r in p['selected']}
    last_id = None
    native_output = None
    first_replay_outputs = {}
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
        saved_keys = payload_names(item, p['selected'][0]['target_id'])
        save = {'head%d' % i: x.numpy() for i, x in enumerate(flows) if 'head%d'%i in saved_keys}
        # All four heads are captured/hashed. Native/FF and first background
        # replays save every raw array; the remaining actions save head3 only.
        if save_captures(item, p['selected'][0]['target_id']):
            save.update({k: v.numpy() for k, v in captures.items()})
        np.savez_compressed(a.output/name, **save)
        require(set(save) == saved_keys, 'Exact frozen raw-payload set')
        record.update(prediction_file=name, prediction_file_sha256=sha(a.output/name), saved_payload_keys=sorted(saved_keys))
        result['forward_records'].append(record)
        result['full_forward_count'] += 1
        if item['execution'] == 'native':
            native_output = output
        else:
            require(native_output is not None and tensor_sha(captures['u']) == tensor_sha(native_output[1]['u']), 'All actions share exact predecision input bytes')
        check = numerical_check(item, output, native_output, first_replay_outputs,
                                p['selected'][0]['target_id'], torch)
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
    require(sum(r['quality_endpoint'] for r in result['forward_records']) == 360 and len(result['numerical_checks']) == 26, 'Full quality/numerical closure')
    require(all(sum(r['summed_'+fc+'_rows'] for r in result['forward_records']) == TOTAL_ROWS[a.background] for fc in ('fc1', 'fc2')),
            'Complete actual FC row closure')
    result.update(status='SELECTOR_COLLECTOR362_COMPLETE',
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
    parser.add_argument('--plan', choices=('selector362',), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--background', choices=('CC','T8'), required=True)
    parser.add_argument('--wall-minutes', type=float, default=45)
    parser.add_argument('--preflight-output', type=Path)
    parser.add_argument('--execute', action='store_true')
    a = parser.parse_args()
    p = prepare(a)
    result = dict(schema='flow_r21_selector_collector_v1', graph_id=graph_id(a.background), status='PREFLIGHT_PASS_NO_INFERENCE', binding=p['binding'],
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
            raise TimeoutError('R21 hard wall cap')
        signal.signal(signal.SIGALRM, alarm)
        signal.alarm(int(a.wall_minutes*60))
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
