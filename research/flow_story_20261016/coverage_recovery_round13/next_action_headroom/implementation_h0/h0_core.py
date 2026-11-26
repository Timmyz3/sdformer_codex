"""H0 identities, state loading and tail-aware adapter; no entrypoint side effects.

The frozen probe's forward code is reused with a private wrapper namespace.
No legacy prepare/execute, checkpoint selection, training, or quality reader here.
"""
import contextlib
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
R13_PATH = HERE.parents[1]/'implementation/run_coverage_recovery.py'
R13_SHA = '9aefdaae4e203816f24415eb73c6c480be1529baba28f5c74bc10e9ca1764ef5'
MEMBER_SHA = 'eba4544fe99c082d7d099f90d03470d7abad22b01b68d3eb4d022e61e78151d1'
TARGET = 'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.mlp'
ANCHORS = ('native', 'FF', 'CC', 'P', 'F12', 'T8', 'matched_N', 'matched_F')
DUALS = ('CC', 'FF', 'P', 'F12', 'T8')
BUFFERS = ('running_mean', 'running_var', 'num_batches_tracked')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_file(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_frozen():
    pins = json.loads((HERE/'source_pins.json').read_text())
    for relative, expected in pins['sources'].items():
        require(sha(dependency_path(relative)) == expected, 'Frozen H0 dependency differs: '+relative)
    require(sha(R13_PATH) == R13_SHA, 'Frozen R13 entry differs')
    r13 = load_file(R13_PATH, '_h0_r13_helpers')
    collector, probe, parent, r9, r8 = r13.load_runtime()
    return r13, collector, probe, parent, r9, r8


def dependency_path(relative):
    relative = Path(relative)
    if HERE.parents[2].name == 'research_20261016':
        require(relative.parts[0] == 'research' and relative.parts[1].startswith('flow_story_'), 'Unsupported deployed source mapping')
        if relative.name == 'common_targets.json':return HERE.parents[4]/'setup/common_targets.json'
        if relative.name == 'train_auxiliary_roles.csv':return HERE.parents[3]/'research_20261013/breadth/train_auxiliary_roles.csv'
        return (HERE.parents[3]/relative.parts[1].replace('flow_story_','research_',1)).joinpath(*relative.parts[2:])
    return REPO/relative


def h8_manifest(membership, membership_sha):
    require(membership_sha == MEMBER_SHA, 'H8 requires fixed R13 identity manifest')
    require(membership['schema'] == 'flow_round13_coverage_identity_v1', 'Wrong identity schema')
    seqs = membership['split_sequences']['train']
    require(len(seqs) == len(set(seqs)) == 8, 'Exactly eight training sequences')
    selected = []
    for seq in seqs:
        rows = [r for r in membership['selector_members'] if r['sequence'] == seq]
        require(len(rows) == 5, 'Five unused selector members per sequence')
        selected.append(min(rows, key=lambda r: (hashlib.sha256(('flow-headroom-v1:'+r['target_id']).encode()).hexdigest(), r['target_id'])))
    excluded = {r['target_id'] for k in ('train_members_diversity', 'bn_members', 'check_members') for r in membership[k]}
    require(not excluded & {r['target_id'] for r in selected}, 'H8 overlaps excluded identity role')
    return {'schema': 'flow_h0_h8_identity_v1', 'membership_sha256': membership_sha,
            'selection': 'per train sequence min (SHA256(flow-headroom-v1:+target_id), target_id); manifest train sequence order',
            'members': selected, 'quality_used_for_selection': False,
            'scope': 'Eight training-side exploratory frames, one per sequence; not independent confirmation or unseen-backbone sequences.'}


def forward_plan(rows):
    require(len(rows) == 8 and len({r['target_id'] for r in rows}) == 8, 'Exactly H8 required')
    plan = []
    # CC precedes P/F12/T8 so their unchanged causal prefix can be checked.
    for index, row in enumerate(rows):
        for anchor in ('native', 'CC', 'FF', 'P', 'F12', 'T8'):
            plan.append({'state': 'W', 'anchor': anchor, 'mode': 'actual', 'target_id': row['target_id']})
            if index == 0 and anchor in DUALS:
                plan.append({'state': 'W', 'anchor': anchor, 'mode': 'dual_reference', 'target_id': row['target_id']})
    for state, anchor in (('N', 'matched_N'), ('F', 'matched_F')):
        plan.extend({'state': state, 'anchor': anchor, 'mode': 'actual', 'target_id': r['target_id']} for r in rows)
    require(len(plan) == 69 and sum(p['mode'] == 'dual_reference' for p in plan) == 5, 'H0 budget differs')
    return [dict(item, forward_id=i+1) for i, item in enumerate(plan)]


def selected_layer(name):
    return any('.layers.%d.' % stage in name for stage in (0, 2))


def mask_for(shape, name, anchor, torch):
    require(anchor in ('CC', 'FF', 'P', 'F12', 'T8'), 'Unknown static H0 mask')
    _, _, height, width, _ = shape
    mask = torch.zeros((math.ceil(height/9), math.ceil(width/9)), dtype=torch.bool)
    if anchor == 'FF':
        mask[:] = True
    elif name == TARGET:
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                complete = min(9, height-9*y) == min(9, width-9*x) == 9
                mask[y, x] = anchor == 'P' or anchor == 'F12' and complete or anchor == 'T8' and not complete
    return mask


def tile_geometry(shape):
    t, batch, height, width, _ = shape
    result = []
    for y in range(math.ceil(height/9)):
        for x in range(math.ceil(width/9)):
            h, w = min(9, height-9*y), min(9, width-9*x)
            points, reps = h*w, math.ceil(h/3)*math.ceil(w/3)
            result.append({'tile': [y, x], 'real_hw': [h, w], 'complete': h == w == 9,
                           'fine_fc_rows': t*batch*points, 'coarse_fc_rows': t*batch*reps,
                           'fine_extra_fc_rows': t*batch*(points-reps)})
    return result


def expected_rows(shapes, anchor, torch, dual=False):
    rows = 0
    for name, shape in shapes.items():
        if anchor == 'native' or not selected_layer(name):
            rows += math.prod(shape[:-1])
            continue
        mask = mask_for(shape, name, anchor, torch)
        for tile in tile_geometry(shape):
            y, x = tile['tile']
            rows += tile['fine_fc_rows']+tile['coarse_fc_rows'] if dual else tile['fine_fc_rows'] if bool(mask[y,x]) else tile['coarse_fc_rows']
    return rows


def adapted_probe(probe, anchor):
    @contextlib.contextmanager
    def wrap(model, targets, shapes, unused, mode, merge, torch):
        require(unused == (), 'No arbitrary action input in H0')
        wrappers = {}
        with contextlib.ExitStack() as stack:
            for name in targets:
                if selected_layer(name):
                    wrappers[name] = stack.enter_context(merge.replace_mlp(model, name, mask_for(shapes[name],name,anchor,torch),
                                                                         3, mode, expected_shape=shapes[name]))
            require(TARGET in wrappers and len(wrappers) == 8, 'Exactly eight selected MLPs')
            yield wrappers
    namespace = dict(probe.forward_probe.__globals__)
    namespace.update(wrap=wrap, TARGET=TARGET)
    # Isolated globals avoid altering historical imported modules.
    return types.FunctionType(probe.forward_probe.__code__, namespace, '_h0_frozen_forward', probe.forward_probe.__defaults__, probe.forward_probe.__closure__)


def forward(runtime, model, functional, targets, shapes, chunk, anchor, merge, torch, mode='actual', deadline=lambda:None):
    r13, collector, probe, parent, r9, r8 = runtime
    if anchor in ('native', 'matched_N'):
        require(mode == 'actual', 'Native has no dual branch')
        flows, execution, branches, inputs = r13.reference_forward(r9,r8,parent,model,functional,targets,shapes,chunk,'native',merge,torch,deadline=deadline)
    else:
        actual_anchor = 'FF' if anchor == 'matched_F' else anchor
        flows, execution, branches, _ = adapted_probe(probe,actual_anchor)(model,functional,targets,shapes,chunk,(),merge,torch,
                                                                         mode=mode,capture=True,deadline=deadline)
        inputs = {n:r['input_sha256'] for n,r in execution['layers'].items()}
    expected = expected_rows(shapes,'native' if anchor == 'matched_N' else 'FF' if anchor == 'matched_F' else anchor,torch,mode=='dual_reference')
    require(execution['summed_fc1_rows'] == execution['summed_fc2_rows'] == expected, 'H0 independent row formula differs')
    require(len(branches) == len(inputs) == 8 and set(branches) == set(inputs), 'Eight branch/input records required')
    execution.update(anchor=anchor,mode=mode,head_records=[{'shape':list(x.shape),'dtype':str(x.dtype),'sha256':merge.tensor_sha(x),
                                                          'finite':bool(torch.isfinite(x).all())} for x in flows],
                     branch_records={n:{'input_sha256':inputs[n],'output_sha256':merge.tensor_sha(v),'shape':list(v.shape)} for n,v in branches.items()},
                     target_geometry=tile_geometry(shapes[TARGET]))
    require(all(m is model.get_submodule(n) for n,m in targets.items()), 'Original MLP not restored')
    require(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()), 'Unremoved forward hooks')
    return flows, execution, branches, inputs


def check_pair(actual, dual, torch):
    a, ax, ab, ai = actual
    d, dx, db, di = dual
    require(len(a) == len(d) and all(torch.equal(x,y) for x,y in zip(a,d)), 'Dual full flow heads differ')
    require(set(ab) == set(db) and len(ab) == 8 and ai == di, 'Dual input/branch keysets differ')
    require(all(torch.equal(ab[n],db[n]) for n in ab), 'Dual complete branches differ')
    return {'all_flow_heads_exact':True,'head_count':len(a),'all8_branch_inputs_exact':True,'all8_branches_exact':True}


def check_prefix(execution, coarse):
    names = ['sttmultires_unet.encoders.swin3d.layers.%d.swin_blocks.%d.mlp' % (s,b) for s,b in ((0,0),(0,1),(2,0),(2,1),(2,2))]
    for name in names:
        require(execution['branch_records'][name]['input_sha256'] == coarse['branch_records'][name]['input_sha256'], 'Changed same-frame coarse prefix input')
        if name != TARGET:
            require(execution['branch_records'][name]['output_sha256'] == coarse['branch_records'][name]['output_sha256'], 'Changed coarse prefix branch')
    return {'same_target_input':True,'preceding4_selected_MLP_branches_exact':True,'checked_input_names':names,
            'FF_excluded_because_upstream_actions_differ':True}


def state_dict_digest(state):
    digest = hashlib.sha256()
    for name,value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode());digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def drive_plan(plan, switch_state, infer, save_actual, pair_checker, expected_heads=4):
    """Production orchestration also exercised with mock inference on CPU."""
    require(len(plan) == 69, 'Exactly69 forward calls, no hidden sampling')
    state = None; previous_actual = None; coarse = {}; records = []; pairs = {}
    for item in plan:
        if item['state'] != state:
            switch_state(item['state']);state = item['state']
        output = infer(item)
        require(len(output[0]) == expected_heads, 'All native output heads required')
        if item['mode'] == 'dual_reference':
            require(previous_actual is not None and previous_actual[0]['anchor'] == item['anchor'] and
                    previous_actual[0]['target_id'] == item['target_id'], 'Dual must immediately match its actual')
            pairs[item['anchor']] = pair_checker(previous_actual[1],output)
            previous_actual = None
        else:
            if item['state'] == 'W' and item['anchor'] == 'CC':
                coarse[item['target_id']] = output[1]
            elif item['state'] == 'W' and item['anchor'] in ('P','F12','T8'):
                output[1]['prefix_audit'] = check_prefix(output[1],coarse[item['target_id']])
            save_actual(item,output)
            previous_actual = (item,output)
        records.append(dict(item,execution=output[1]))
    require(len(records) == 69 and set(pairs) == set(DUALS), 'Missing H0 calls/audits')
    return records,pairs


def install_recovered(model, payload, descriptor, common_buffers, torch):
    """Fully validate before mutation; original checkpoint cannot substitute for W."""
    require(payload.get('schema') == 'flow_round13_recovered_shared_native_state_v1', 'Named R13 recovered artifact required')
    require(payload.get('completed_optimizer_steps') == 200 and payload.get('BN_scope') == 'MLP_R13_COMMON_CMA176'
            and payload.get('BN_buffers_unchanged') is True, 'Incomplete recovered state/BN')
    require(payload.get('binding') == descriptor['binding'] and payload.get('initial_state_sha256') == descriptor['initial_state_sha256']
            and payload.get('final_state_sha256') == descriptor['final_state_sha256'], 'Recovered artifact provenance differs')
    current, state = model.state_dict(), payload['model_state_dict']
    require(set(current) == set(state), 'Recovered state keyset differs from fresh legal BN architecture')
    require(set(common_buffers) == {n for n in state if n.endswith(BUFFERS)}, 'Recovered BN keyset differs')
    for name,value in state.items():
        require(isinstance(value,torch.Tensor) and tuple(value.shape) == tuple(current[name].shape) and value.dtype == current[name].dtype,
                'Recovered tensor shape/dtype differs: '+name)
        require(bool(torch.isfinite(value).all()), 'Nonfinite recovered tensor: '+name)
        if name in common_buffers:
            require(torch.equal(value.cpu(),common_buffers[name].cpu()), 'Recovered common BN value differs: '+name)
    require(state_dict_digest(state) == descriptor['final_state_sha256'], 'Recovered complete state digest differs')
    require(len(payload['trainable_names']) == len(set(payload['trainable_names'])) and
            set(payload['trainable_names']) == set(descriptor['trainable_names']), 'Recovery parameter names differ')
    allowed = {n for n,_ in model.named_parameters() if '.mlp.' in n and selected_layer(n)}
    require(set(payload['trainable_names']) == allowed, 'Recovered trainable names differ from selected eight native MLPs')
    model.load_state_dict(state,strict=True);model.eval().requires_grad_(False)
    require(state_dict_digest(model.state_dict()) == descriptor['final_state_sha256'], 'Loaded recovered state differs')
    return {'state_sha256':descriptor['final_state_sha256'],'strict_full_state_load':True,'common_BN_exact':True,
            'recovered_arm':descriptor['binding']['arm'],'condition':descriptor['binding']['condition']}
