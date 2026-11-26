#!/usr/bin/env python3
"""CPU-only dense EPE of frozen coarse heads, not full-model flow or a speedup."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
os.environ['CUDA_VISIBLE_DEVICES'] = ''

BASE_SHA = '1cacbaeea4d9eaf19a304c3faaa13f5278f0212b6cdc92210c4619447d571628'
FIT_SHA = 'd93869c693faa507923e8b85a8f2cff89cc0a4b502645ec52afa7608bf8f37a3'
HEAD_SHA = 'eb0b5e3c3e675c4466a42b73c77237735460913fd79452b9eb9a662226f42ef5'
GROUP_SHA = '7931fb2b6016b18551871c4d8d4e643451607623d6598b3d605fca2f8935f216'
CORE_SHA = '6178520f2e841eebd39ac97b22d716944771c656d7a76ce26ed7a1c57d0d6dcb'
FIT_MEMBERS_SHA = '29ec7ba9c9faca74c62c3e170c9d3c70855a4605a7eab31f9331c519f56f8915'
RAW_INPUT = 'raw descriptor; normalization already folded; do NOT normalize again'
METHODS = ('constant_component_median', 'linear', 'C16_GELU2')
REFERENCE = 'GT_tile_mean_diagnostic_reference'
GEOMETRY = {'0': ((120, 160), 4, 96), '2': ((30, 40), 16, 384)}
HERE = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(4194304), b''):
            h.update(part)
    return h.hexdigest()


def pinned_module(name, path, expected):
    require(sha(path) == expected, 'Source SHA differs: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def validate_binding(cache, weights, cache_sha, contract_sha):
    require(cache.get('schema') == 'flow_coarse_feature_cache_v1' and
            cache.get('contract_sha256') == contract_sha and cache.get('BN_scope') == 'MLP_CMA176'
            and cache.get('label_geometry') == 'logical_cell_support_origin0_v1', 'Cache contract/geometry differs')
    require(weights.get('schema') == 'flow_coarse_folded_head_weights_v1'
            and weights.get('provenance', {}).get('cache_sha256') == cache_sha,
            'Weights do not belong to this cache')
    require(weights.get('flow_contract') == cache['flow_contract'] and weights['provenance'].get('GPU_used') is False,
            'Head provenance/flow contract differs')
    expected = {'fit_coarse_cache.py': FIT_SHA, 'coarse_head.py': HEAD_SHA, 'group_policy.py': GROUP_SHA}
    require(weights['provenance'].get('source_sha256') == expected, 'Head training source differs')
    recipe = weights.get('recipe', {})
    require(recipe.get('development_selection') is False and recipe.get('device') == 'cpu'
            and recipe.get('steps_per_trainable_head') == 1000 and recipe.get('seed') == 20261013,
            'Not the fixed1000step fit-only head artifact')
    require(set(weights.get('stages', {})) == {'0', '2'}, 'Weights stage coverage differs')


def predict_tiles(descriptor, stage_weights, method):
    """Folded weights consume RAW descriptor. There is deliberately no normalizer argument."""
    import torch
    import torch.nn.functional as F
    require(method in METHODS and descriptor.device.type == 'cpu' and descriptor.ndim == 4
            and descriptor.shape[0] == 1 and descriptor.shape[-1] == stage_weights['channels']
            and bool(torch.isfinite(descriptor).all()), 'Invalid raw descriptor')
    item = stage_weights['methods'][method]; channels = stage_weights['channels']
    with torch.no_grad():
        if method == 'constant_component_median':
            constant = item.get('constant_dxdy')
            require(isinstance(constant, torch.Tensor) and constant.device.type == 'cpu'
                    and tuple(constant.shape) == (2,) and bool(torch.isfinite(constant).all()), 'Invalid constant head')
            result = constant.float().reshape(1, 1, 1, 2).expand(*descriptor.shape[:-1], 2)
        else:
            require(item.get('input_contract') == RAW_INPUT, 'Folded head requires raw input; reject repeat normalization')
            state = item.get('state_dict', {})
            shapes = {'weight': (2, channels), 'bias': (2,)} if method == 'linear' else {
                'fc1.weight': (16, channels), 'fc1.bias': (16,), 'fc2.weight': (2, 16), 'fc2.bias': (2,)}
            require(set(state) == set(shapes), 'Unexpected head state keys')
            require(all(isinstance(state[k], torch.Tensor) and state[k].device.type == 'cpu'
                        and tuple(state[k].shape) == shape and state[k].dtype == torch.float32
                        and bool(torch.isfinite(state[k]).all()) for k, shape in shapes.items()), 'Invalid folded weights')
            raw = descriptor.float()
            result = F.linear(raw, state['weight'], state['bias']) if method == 'linear' else F.linear(
                F.gelu(F.linear(raw, state['fc1.weight'], state['fc1.bias']), approximate='none'), state['fc2.weight'], state['fc2.bias'])
        require(bool(torch.isfinite(result).all()), 'Nonfinite head prediction')
        return result


def expand_tiles(prediction, targets, *, gt_valid=None, diagnostic_reference=False):
    """Origin0 logical support; smaller real tails cover the full480x640 image."""
    import torch
    h, w = targets['feature_hw']; sx, sy = targets['feature_stride_pixels_xy']
    require(targets.get('image_origin_xy') == [0., 0.] and type(sx) is int and type(sy) is int
            and sx == sy and (h, w, sx) in ((120, 160, 4), (30, 40, 16)), 'Only declared full-frame origin0 geometry accepted')
    ny, nx = (h+8)//9, (w+8)//9
    require(tuple(prediction.shape) == (1, ny, nx, 2) and prediction.device.type == 'cpu', 'Tile output geometry differs')
    value = prediction.detach().float().clone()
    if diagnostic_reference:
        valid = targets['valid']
        require(gt_valid is not None and valid.dtype == torch.bool and tuple(valid.shape) == (1, ny, nx), 'Reference needs GT validity')
        # MetricAccumulator requires finite predictions everywhere. Placeholder0 is permitted
        # only on tiles with no valid GT; it cannot contribute a single scored pixel.
        expanded_valid = valid[0].repeat_interleave(9*sy, 0).repeat_interleave(9*sx, 1)[:480, :640]
        require(not bool((gt_valid.bool() & ~expanded_valid).any()), 'Empty-tile placeholder would cover valid GT')
        value[~valid] = 0.
    require(bool(torch.isfinite(value).all()), 'Nonfinite expanded prediction')
    dense = value[0].repeat_interleave(9*sy, 0).repeat_interleave(9*sx, 1)[:480, :640].permute(2, 0, 1).contiguous()
    require(tuple(dense.shape) == (2, 480, 640), 'Logical support did not cover full frame')
    return dense.numpy()


def read_options(contract):
    parser = argparse.ArgumentParser(add_help=False)
    for key in ('data-root', 'canonical', 'roles', 'prior-root', 'regions'):
        parser.add_argument('--'+key, type=Path, required=True)
    for key in ('regions-sha256', 'checkpoint-sha256'):
        parser.add_argument('--'+key, required=True)
    a, _ = parser.parse_known_args(contract['base_args'])
    return a


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'weights', 'cache-contract', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ('cache-sha256', 'weights-sha256', 'contract-sha256'):
        parser.add_argument('--'+name, required=True)
    a = parser.parse_args(argv)
    require(not a.output.exists(), 'Output exists; preserve previous result')
    require(sha(a.cache_contract) == a.contract_sha256, 'Cache contract SHA differs')
    contract = json.loads(a.cache_contract.read_text())
    require(contract.get('schema') == 'flow_coarse_cache_contract_v1' and contract.get('CPU_only') is True
            and contract.get('stages') == [0, 2], 'Foreign cache collection contract')
    for name, expected in (('coarse_head.py', HEAD_SHA), ('group_policy.py', GROUP_SHA)):
        require(sha(HERE/name) == expected, 'Local source differs: '+name)
    fit = pinned_module('coarse_dense_fit_helper', HERE/'fit_coarse_cache.py', FIT_SHA)
    from coarse_head import make_tile_targets
    import torch
    torch.set_num_threads(2)
    require(not torch.cuda.is_initialized(), 'CPU-only evaluator found initialized CUDA')
    cache, membership = fit.load_verified_cache(a.cache, a.cache_sha256, strict_population=True)
    require(sha(a.weights) == a.weights_sha256, 'Weights SHA differs')
    weights = torch.load(a.weights, map_location='cpu', weights_only=True)
    validate_binding(cache, weights, a.cache_sha256, a.contract_sha256)
    options = read_options(contract)
    breadth = Path(contract['breadth_root'])
    base = pinned_module('coarse_dense_base', breadth/'evaluate_breadth.py', BASE_SHA)
    native, trainer, metrics, pins = base.support_modules(options.prior_root)
    dev32, records, roles, _ = base.selected_rows(options.canonical, options.roles, metrics)
    core = pinned_module('coarse_dense_core', HERE/'intervention_core.py', CORE_SHA)
    dev = core.choose_targets(dev32, 8)
    fit_manifest = Path(contract['fit_manifest'])
    require(contract['fit_manifest_sha256'] == FIT_MEMBERS_SHA and sha(fit_manifest) == FIT_MEMBERS_SHA, 'Fit20 membership differs')
    fit_ids = json.loads(fit_manifest.read_text())['target_ids']
    require(membership['target_ids'] == {'fit': fit_ids, 'dev': [r['target_id'] for r in dev]}
            and all(r['protocol_split'] == 'train' and roles.get(r['target_id']) == 'gate_development' for r in dev), 'Cache membership differs from fixed fit20/dev8')
    require(cache['checkpoint_sha256'] == options.checkpoint_sha256 and cache['flow_contract'] == contract['flow_contract'], 'Backbone/flow provenance differs')
    regions = base.read_pin(options.regions, options.regions_sha256)
    metrics.validate_region_contract(regions, records, roles, require_complete_fit=True)
    frames = {f['target_id']: f for f in cache['frames'] if f['role'] == 'dev'}
    meters = {s: {m: metrics.MetricAccumulator() for m in (*METHODS, REFERENCE)} for s in GEOMETRY}
    evidence_rows = []
    with torch.no_grad():
        for row in dev:
            frame = frames[row['target_id']]
            expected = frame['input_evidence']
            require(set(expected) == {'events', 'gt', 'mask'}, 'Missing cached raw input evidence')
            evidence = {}
            arrays, _ = metrics.load_arrays(options.data_root/'saved_flow_data', row, evidence=evidence, expected_hashes=expected)
            masks = metrics.partitions(arrays['gt'], arrays['mask'], arrays['events'], regions)
            gt, valid = torch.from_numpy(arrays['gt']), torch.from_numpy(arrays['mask'])
            check = {'target_id': row['target_id'], 'sequence': row['sequence'], 'input_sha256': evidence, 'stages': {}}
            for stage, (hw, stride, channels) in GEOMETRY.items():
                item = frame['stages'][stage]; targets = item['targets']
                require(targets['feature_hw'] == list(hw) and targets['feature_stride_pixels_xy'] == [stride, stride]
                        and targets['image_origin_xy'] == [0., 0.] and weights['stages'][stage]['channels'] == channels
                        and item['descriptor'].shape[-1] == channels, 'Stage geometry/channels differ')
                reference = make_tile_targets(gt, valid, feature_hw=hw, feature_stride_pixels=(stride, stride),
                    flow_contract=cache['flow_contract'], image_origin_xy=(0., 0.))
                require(torch.equal(reference['valid'], targets['valid']) and
                        torch.equal(reference['valid_pixel_count'], targets['valid_pixel_count']) and
                        torch.equal(reference['support_pixel_count'], targets['support_pixel_count']) and
                        torch.allclose(reference['flow'], targets['flow'], atol=1e-6, rtol=1e-6, equal_nan=True),
                        'Cached GT tile labels do not match SHA-verified raw GT/support')
                require(int(reference['valid_pixel_count'].sum()) == int(valid.sum())
                        and int(reference['support_pixel_count'].sum()) == 480*640, 'Incomplete full-frame support')
                for method in METHODS:
                    prediction = predict_tiles(item['descriptor'], weights['stages'][stage], method)
                    dense = expand_tiles(prediction, targets)
                    meters[stage][method].update(row['target_id'], row['sequence'], dense, arrays['gt'], masks)
                dense = expand_tiles(reference['flow'], reference, gt_valid=valid, diagnostic_reference=True)
                meters[stage][REFERENCE].update(row['target_id'], row['sequence'], dense, arrays['gt'], masks)
                check['stages'][stage] = {'valid_tiles': int(reference['valid'].sum()), 'empty_tiles': int((~reference['valid']).sum()),
                    'valid_GT_pixels': int(valid.sum()), 'support_pixels': 480*640,
                    'cached_label_recheck_atol': 1e-6, 'cached_label_recheck_rtol': 1e-6}
            evidence_rows.append(check)
    require(not torch.cuda.is_initialized(), 'CUDA initialized')
    output = {'schema': 'flow_coarse_dense_evaluation_v1', 'status': 'COARSE_DENSE_EVALUATION_COMPLETE',
        'scope': 'Coarse-head-only piecewise constant full-frame EPE on fixed backbone-seen development8; not full-model AEE or a compression delta',
        'reference_scope': 'GT tile arithmetic-mean expansion: mixed-motion diagnostic, not a strict EPE-optimal lower bound. Empty-GT tiles get finite placeholders only outside all scored GT.',
        'geometry': 'logical_cell_support_origin0_v1; feature strides4/16; displacement stays original-image pixels',
        'normalization': RAW_INPUT, 'hardware_benefit_claim': False, 'training_performed': False,
        'GPU_initialized': False, 'native_backbone_forward_performed': False, 'final_valid_used': False,
        'region_note': 'All existing masks retained; voxel_zero means encoded-voxel inactivity, not independently measured raw-event absence.',
        'provenance': {'cache_sha256': a.cache_sha256, 'weights_sha256': a.weights_sha256,
            'cache_contract_sha256': a.contract_sha256, 'regions_sha256': options.regions_sha256,
            'checkpoint_sha256': cache['checkpoint_sha256'], 'source_sha256': {'evaluate_coarse_dense.py': sha(__file__),
            'fit_coarse_cache.py': FIT_SHA, 'coarse_head.py': HEAD_SHA, 'group_policy.py': GROUP_SHA,
            'intervention_core.py': CORE_SHA, 'evaluate_breadth.py': BASE_SHA}, 'support_source_pins': pins},
        'dev_target_ids': [r['target_id'] for r in dev], 'input_evidence': evidence_rows,
        'stages': {s: {'methods': {m: {'summary': meter.summary(), 'frames': meter.frames} for m, meter in methods.items()}}
                   for s, methods in meters.items()}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        stream.write(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'status': output['status'], 'dev_frames': len(dev), 'output': str(a.output), 'GPU_used': False}))


if __name__ == '__main__':
    main()
