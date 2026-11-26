"""RNE4 background and six whole-tile Q8 upgrades; no GPU or training entry."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

CORE_SHA = 'ce07ae04c9f3d58738ae134cdc1c8276542a41a04584720b5d69359e220d1f9f'
E0_RUNNER_SHA = '417ac2fa91a08048e77a6729fb4210b784bc7f472bffddd8fe13b3f50e27695a'
INVENTORY_SHA = 'f1aca17e9a6f7e3afc6f57260fce313ae8197d61eedb5c5c9c0d65603ac6ad14'
SELECTION_SHA = '60b7d6d525d4130f5f0efecaff95511808b2186c3aaba9e9f37a957db32ef31f'
MEMBERS_SHA = '40eba2514b56fd33a910fda4e37d4f86e80431ffe8149efc5f79c18827991e7b'
SMOKE_SHA = '6dadb497bf97fc33929209a3feed0acedcac241036d27bbcda502d8f35244d83'
HELPER_SHA = '5a5bf2abaca8fe23fc269e4dc830acbd3ee6537e52958c29c1520543756a4f95'
R28_CORE_SHA = 'f4e4dae18a7e59fab286f2e59f23f72d20583a2b8de22dee4c78204bbfd2fe9a'


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_core(path):
    require(sha(path) == CORE_SHA, 'Exact unmodified frozen E0 RNE core')
    return load(path, '_e3_frozen_e0_core')


def shape_contract(core, inventory, selections):
    require(sha(inventory) == INVENTORY_SHA and sha(selections) == SELECTION_SHA,
            'Frozen geometry and identity-role sources')
    geometries = {}
    for row in read(inventory)['inventory']['affines']:
        name = row['weight'].removesuffix('.weight')
        if name in core.TARGETS:
            geometries[name] = {k: row[k] for k in ('T', 'H', 'W', 'K', 'Cout')}
    require(set(geometries) == set(core.TARGETS), 'All28 inventory geometries')
    groups = []
    for i in range(6):
        tiles, ticks = {}, 0
        for name in core.CONVS:
            g = geometries[name]
            require(g == dict(T=10, H=240, W=320, K=864, Cout=96),
                    'Actual four patch geometries')
            cells = [[y, x] for y in range(g['H'] // 9) for x in range(g['W'] // 9)]
            require(len(cells) == 910, '910 complete patch tiles')
            tiles[name] = cells[i * 151:(i + 1) * 151]
            ticks += 4 * 151 * 81 * g['T'] * math.ceil(g['K']/128) * math.ceil(g['Cout']/16)
        groups.append(dict(id=i, tiles=tiles, added_core_ticks_vs_Q4=ticks))
    selection = read(selections)
    return dict(schema='flow_e3_rne_patch6_probe_contract_v1',
        status='PREPARED_NOT_ENDPOINT_BOUND_NOT_AUTHORIZATION',
        quantizer='frozen_E0_independent_RNE4_and_Q8_scales',
        core_sha256=CORE_SHA, inventory_sha256=INVENTORY_SHA,
        selection_sha256=SELECTION_SHA, members_sha256=MEMBERS_SHA,
        geometries=geometries, family_targets=list(core.CONVS), groups=groups,
        target_ids=selection['e2_reserved_ids'], selection_scope=selection['scope'],
        background='all28_complete_RNE4_true_tails_Q8',
        action='one_group_direct_Q8_other_complete_RNE4_not_Q4_then_refine',
        remaining_complete_tiles_per_patch=4, full_T=10,
        current_BN_count=78, target_affine_count=28,
        family_reason='preselected_compute_coverage_not_old_truncation_E1_sensitivity',
        runtime_information='current_raw_T10_voxel_before_first_controlled_patch_conv',
        predictor_implemented=False, development_GT_arrays_read=False,
        formal_forwards=112, new_action_predictions=96, background_replay_forwards=16,
        engineering_dense_reference_forwards=6, total_forwards=118,
        endpoint=None, net_cycle_claim=False, resume_allowed=False)


def endpoint_binding(result_path, weights_path, ids):
    result = read(result_path)
    require(result['status'] == 'RNE4_JOINT512_AND320_DEVELOPMENT_PREDICTIONS_COMPLETE'
            and result['completed_optimizer_steps'] == 512
            and result['model_forwards'] == 1353 and result['task'] == 'recovery',
            'Complete fixed512 formal E0 endpoint only')
    require(result['sources']['runner'] == E0_RUNNER_SHA
            and result['sources']['core'] == CORE_SHA
            and result['sources']['members'] == MEMBERS_SHA, 'Accepted original E0 source contract')
    require(sha(weights_path) == result['weights_sha256'], 'Actual endpoint checkpoint bytes')
    ev = result['evaluations']['endpoint_Q4']
    require(ev['model_forwards'] == 64 and ev['state_sha256'] == result['final_state_sha256'],
            'Complete same-W Q4 reference')
    prior = {r['target_id']: r for r in ev['frames']}
    require(len(prior) == 64 and all(t in prior for t in ids), 'All16 prior four-head references')
    require(result['final_parity']['Q4']['dense_integer']['pass']
            and result['final_parity']['Q4']['training_deployment']['pass'],
            'Original endpoint Q4 full-graph numerical parity')
    return dict(role='formal_step512_only', result_sha256=sha(result_path),
                weights_sha256=sha(weights_path), state_sha256=result['final_state_sha256'],
                completed_optimizer_steps=512), prior


def validate(contract, core, *, require_endpoint=True):
    require(contract['schema'] == 'flow_e3_rne_patch6_probe_contract_v1'
            and contract['core_sha256'] == CORE_SHA
            and contract['inventory_sha256'] == INVENTORY_SHA
            and contract['selection_sha256'] == SELECTION_SHA
            and contract['members_sha256'] == MEMBERS_SHA, 'Explicit pinned RNE probe contract')
    require(set(contract['geometries']) == set(core.TARGETS)
            and contract['family_targets'] == list(core.CONVS), 'Only four patch targets are upgraded')
    ids = contract['target_ids']
    require(len(ids) == len(set(ids)) == 16, 'Exact16 identities')
    require((contract['formal_forwards'], contract['engineering_dense_reference_forwards'],
             contract['total_forwards']) == (112, 6, 118), 'Fixed112+6 budget')
    require(len(contract['groups']) == 6, 'Six ordered groups')
    for i, group in enumerate(contract['groups']):
        require(group['id'] == i and set(group['tiles']) == set(core.CONVS), 'One common macro per patch')
        for name in core.CONVS:
            g = contract['geometries'][name]
            require(g == dict(T=10, H=240, W=320, K=864, Cout=96), 'Patch shape fixed')
            cells = [[y, x] for y in range(26) for x in range(35)]
            require(group['tiles'][name] == cells[i*151:(i+1)*151], 'Exact fixed row-major151')
        require(group['added_core_ticks_vs_Q4'] == 82192320, 'Four-plane replacement arithmetic')
    for name in core.LINEARS:
        g = contract['geometries'][name]; stage = int(name.split('.layers.')[1].split('.')[0])
        k = 96 * 2**stage if name.endswith('.fc1') else 384 * 2**stage
        co = 384 * 2**stage if name.endswith('.fc1') else 96 * 2**stage
        require(g == dict(T=10, H=480//(4*2**stage), W=640//(4*2**stage), K=k, Cout=co),
                'Original MLP full geometry')
    if require_endpoint:
        require(contract['status'] == 'FROZEN_ENDPOINT_STEP512_PENDING_ROOT_EXECUTION_DECISION'
                and isinstance(contract['endpoint'], dict), 'Endpoint binding required before preflight')
        ep = contract['endpoint']
        require(ep['role'] == 'formal_step512_only' and ep['completed_optimizer_steps'] == 512
                and all(len(ep[k]) == 64 for k in ('result_sha256', 'weights_sha256', 'state_sha256')),
                'Bound immutable result/state/checkpoint hashes')
    return contract


def materialize(contract, action, core):
    import torch
    validate(contract, core, require_endpoint=False)
    require(type(action) is int and -1 <= action < 6, 'Background -1 or one of six single upgrades')
    masks = {}
    for name in core.TARGETS:
        g = contract['geometries'][name]; group = core.group_name(name)
        masks.setdefault(group, torch.zeros(g['H'], g['W'], dtype=torch.bool))
    if action >= 0:
        for name, cells in contract['groups'][action]['tiles'].items():
            for y, x in cells:
                masks[name][y*9:y*9+9, x*9:x*9+9] = True
    require(set(masks) == set(core.GROUPS), 'Explicit all16 E0 mask groups')
    return masks, dict(action=action, mode='Q4' if action == -1 else 'M',
        mask_sha256={n: core.digest(m) for n, m in masks.items()},
        selected_tiles={n: int(m.sum())//81 for n, m in masks.items()},
        added_core_ticks_vs_Q4=0 if action == -1 else 82192320, net_cycle_claim=False)


class BoundCore:
    def __init__(self, core, contract, action):
        self.engine = core
        self.masks, self.mask_record = materialize(contract, action, core)
        self.mode = self.mask_record['mode']

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def installed(self, model, mode, *, ste=False, reference=False):
        require(not ste and mode == self.mode, 'Inference-only fixed action context')
        return self.engine.installed(model, mode, masks=self.masks if mode == 'M' else None,
                                     ste=False, reference=reference, variant='rne4')


def main():
    parser = argparse.ArgumentParser()
    for name in ('core', 'inventory', 'selections', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--endpoint-result', type=Path)
    parser.add_argument('--weights', type=Path)
    a = parser.parse_args()
    require(not a.output.exists(), 'Fresh contract file')
    core = load_core(a.core)
    c = shape_contract(core, a.inventory, a.selections)
    require(bool(a.endpoint_result) == bool(a.weights), 'Endpoint result and weights together')
    if a.endpoint_result:
        c['endpoint'], _ = endpoint_binding(a.endpoint_result, a.weights, c['target_ids'])
        c['status'] = 'FROZEN_ENDPOINT_STEP512_PENDING_ROOT_EXECUTION_DECISION'
    validate(c, core, require_endpoint=bool(a.endpoint_result))
    write(a.output, c)
    print(json.dumps(dict(status=c['status'], sha256=sha(a.output), GPU_used=False)))


if __name__ == '__main__':
    main()
