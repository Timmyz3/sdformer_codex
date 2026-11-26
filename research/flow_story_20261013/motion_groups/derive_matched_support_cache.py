#!/usr/bin/env python3
"""CPU cache-only same-support diagnostic; no backbone/head training or GPU.

Merge stage0 9x9 descriptors in 4x4 groups, weighted by REAL feature points.
The resulting stage0 descriptor describes 36x36 feature cells (144px image
support), with clipped tails, and uses a clone of the existing stage2 targets.
This is not the original native stage0 runtime head or its cost contract.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import torch

HERE = Path(__file__).resolve().parent
PINS = {
    'fit_coarse_cache.py': 'd93869c693faa507923e8b85a8f2cff89cc0a4b502645ec52afa7608bf8f37a3',
    'coarse_head.py': 'eb0b5e3c3e675c4466a42b73c77237735460913fd79452b9eb9a662226f42ef5',
    'group_policy.py': '7931fb2b6016b18551871c4d8d4e643451607623d6598b3d605fca2f8935f216',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4194304), b''):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def real_tile_counts(feature_hw, tile_size=9):
    require(len(feature_hw) == 2 and all(type(v) is int and v > 0 for v in feature_hw), 'Positive integer feature shape required')
    require(type(tile_size) is int and tile_size > 0, 'Positive tile size required')
    h, w = feature_hw
    return torch.tensor([[min(tile_size, h-y)*min(tile_size, w-x)
                          for x in range(0, w, tile_size)]
                         for y in range(0, h, tile_size)], dtype=torch.int64)


def coarsen_descriptor(descriptor, feature_hw=(120, 160), tile_size=9, group_size=4):
    """No GT argument: geometry-only weights, FP64 sums then source-dtype cast.

    This preserves the weighted mean of the ALREADY ROUNDED source descriptors;
    it does not claim bitwise equality to a fresh larger native reduction.
    """
    counts = real_tile_counts(feature_hw, tile_size)
    require(type(group_size) is int and group_size > 0, 'Positive group size required')
    require(isinstance(descriptor, torch.Tensor) and descriptor.device.type == 'cpu'
            and descriptor.is_floating_point() and descriptor.ndim == 4
            and tuple(descriptor.shape[:3]) == (1, *counts.shape)
            and descriptor.shape[-1] > 0 and bool(torch.isfinite(descriptor).all()), 'Descriptor geometry/type differs')
    ny, nx = counts.shape
    output = descriptor.new_empty((1, (ny+group_size-1)//group_size,
                                   (nx+group_size-1)//group_size, descriptor.shape[-1]))
    group_counts = torch.zeros(output.shape[1:3], dtype=torch.int64)
    visits = torch.zeros_like(counts)
    for oy, y in enumerate(range(0, ny, group_size)):
        for ox, x in enumerate(range(0, nx, group_size)):
            n = counts[y:y+group_size, x:x+group_size]
            d = descriptor[0, y:y+group_size, x:x+group_size].double()
            output[0, oy, ox] = (d*n.double().unsqueeze(-1)).sum((0, 1))/int(n.sum())
            group_counts[oy, ox] = n.sum()
            visits[y:y+group_size, x:x+group_size] += 1
    require(bool((visits == 1).all()) and int(group_counts.sum()) == feature_hw[0]*feature_hw[1], 'Grouping must cover every real feature point exactly once')
    require(bool(torch.isfinite(output).all()), 'Nonfinite coarsened descriptor')
    return output.contiguous(), group_counts


def validate_source(cache):
    for name, expected in PINS.items():
        require(sha(HERE/name) == expected, 'Pinned cache validator changed: '+name)
    from fit_coarse_cache import validate_cache
    membership = validate_cache(cache, strict_population=True)
    require(cache.get('BN_scope') == 'MLP_CMA176' and cache.get('backbone_seen') is True
            and cache.get('label_geometry') == 'logical_cell_support_origin0_v1'
            and 'derivation' not in cache, 'Require original native CPU cache and BN scope')
    for key in ('contract_sha256', 'checkpoint_sha256'):
        value = cache.get(key)
        require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Missing original '+key)
    for frame in cache['frames']:
        for stage, hw, stride, channels in (('0', (120,160), 4, 96), ('2', (30,40), 16, 384)):
            item = frame['stages'][stage]; target = item['targets']
            count = real_tile_counts(hw)
            require(tuple(item['descriptor'].shape) == (1, *count.shape, channels)
                    and item['descriptor'].dtype == torch.float32, 'Original stage descriptor shape/type differs')
            require(target.get('feature_hw') == list(hw)
                    and target.get('feature_stride_pixels_xy') == [stride, stride]
                    and target.get('image_origin_xy') == [0.,0.], 'Original target geometry differs')
            require(torch.equal(target['support_pixel_count'], (count*stride*stride).unsqueeze(0)), 'Target support is not the declared full-image partition')
            expected_bounds = [[[x*9*stride, y*9*stride, min((x+1)*9,hw[1])*stride,
                                 min((y+1)*9,hw[0])*stride] for x in range(count.shape[1])]
                               for y in range(count.shape[0])]
            require(target['physical_bounds_xyxy'] == expected_bounds, 'Target physical bounds differ')
        require(torch.equal(frame['stages']['0']['targets']['valid_pixel_count'].sum(),
                            frame['stages']['2']['targets']['valid_pixel_count'].sum()), 'Stages do not cover equal valid GT counts')
    return membership


def derive_cache(cache, source_sha, script_sha):
    membership = validate_source(cache)
    counts = real_tile_counts((120,160))
    groups = coarsen_descriptor(cache['frames'][0]['stages']['0']['descriptor'])[1]
    derivation = {
        'schema': 'flow_same_support_cache_derivation_v1',
        'source_cache_sha256': source_sha, 'source_collection_contract_sha256': cache['contract_sha256'],
        'derivation_script_sha256': script_sha,
        'formula': 'd_G = sum_{i in G}(n_i*d_i) / sum_{i in G} n_i; n_i=real stage0 feature points in source 9x9 tile',
        'grouping': '4x4 source tiles; nonoverlapping origin0 groups; clip source-grid and feature tails',
        'arithmetic': 'FP64 multiply/sum/divide over cached FP32 means, cast once to FP32; not a fresh native reduction equivalence claim',
        'source_stage0_feature_hw': [120,160], 'source_stage0_descriptor_shape': [1,14,18,96],
        'derived_stage0_descriptor_shape': [1,4,5,96],
        'source_stage0_tile_feature_yx': [9,9], 'derived_stage0_effective_tile_feature_yx': [36,36],
        'nominal_image_support_yx': [144,144], 'image_origin_xy': [0.,0.],
        'source_stage0_real_feature_points': counts.tolist(), 'derived_real_feature_points': groups.tolist(),
        'weight_source': 'geometry only; no GT, validity counts, labels, roles or head outputs select/reweight features',
        'targets': 'deep clone of each same-frame original stage2 targets, with original stage2 logical support metadata',
        'stage2': 'retained unchanged as paired same-support control',
        'scope': 'same-support information diagnostic; not native stage0 runtime head, causal depth isolation, compression, hardware timing or speedup',
        'original_stage0_head_cost_applicable': False, 'training_performed': False, 'GPU_used': False,
    }
    contract = {'schema': 'flow_matched_support_derivation_contract_v1', 'CPU_only': True,
        'checkpoint_sha256': cache['checkpoint_sha256'], 'BN_scope': cache['BN_scope'],
        'backbone_seen': cache['backbone_seen'], 'flow_contract': copy.deepcopy(cache['flow_contract']),
        'source_validator_sha256': dict(PINS), 'membership': membership, 'derivation': derivation,
        'constraints': ['No raw input/GT reads or backbone forward', 'Do not modify source cache or membership',
                        'Do not change frozen head fitting recipe or preexisting weights',
                        'Use a new derivation contract SHA; never reuse the collection contract SHA',
                        'Channel counts remain96 vs384; same support alone does not isolate network depth']}
    contract_raw = json_bytes(contract)
    new_contract_sha = hashlib.sha256(contract_raw).hexdigest()
    require(new_contract_sha != cache['contract_sha256'], 'Derived contract must differ from collection contract')
    derived = copy.deepcopy(cache)
    derived['contract_sha256'] = new_contract_sha
    derived['derivation'] = copy.deepcopy(derivation)
    for frame in derived['frames']:
        item = frame['stages']['0']
        item['descriptor'], _ = coarsen_descriptor(item['descriptor'])
        item['targets'] = copy.deepcopy(frame['stages']['2']['targets'])
    from fit_coarse_cache import validate_cache
    require(validate_cache(derived, strict_population=True) == membership, 'Derivation altered membership/channel contract')
    return derived, contract, contract_raw


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--cache-sha256', required=True)
    parser.add_argument('--output-dir', required=True, type=Path)
    a = parser.parse_args(argv)
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), 'CPU-only derivation cannot use initialized CUDA')
    require(not a.output_dir.exists(), 'Output directory exists; refuse overwrite')
    require(sha(a.cache) == a.cache_sha256, 'Source cache SHA differs')
    cache = torch.load(a.cache, map_location='cpu', weights_only=True)
    derived, contract, contract_raw = derive_cache(cache, a.cache_sha256, sha(__file__))
    a.output_dir.mkdir(parents=True, exist_ok=False)
    contract_path = a.output_dir/'derivation_contract.json'
    with contract_path.open('xb') as stream:
        stream.write(contract_raw)
    output = a.output_dir/'features.pt'
    with output.open('xb') as stream:
        torch.save(derived, stream)
    require(sha(contract_path) == derived['contract_sha256'] and not torch.cuda.is_initialized(), 'Final contract/device check failed')
    report = {'schema': 'flow_matched_support_derivation_report_v1', 'status': 'MATCHED_SUPPORT_CACHE_COMPLETE',
        'source_cache_sha256': a.cache_sha256, 'derived_cache_sha256': sha(output),
        'derivation_contract_sha256': sha(contract_path), 'derivation_script_sha256': sha(__file__),
        'complete_frames': len(derived['frames']), 'membership': contract['membership'],
        'derivation': contract['derivation'], 'checkpoint_sha256': derived['checkpoint_sha256'],
        'BN_scope': derived['BN_scope'], 'CPU_threads': 2, 'GPU_initialized': False,
        'head_training_performed': False, 'source_cache_unchanged': sha(a.cache) == a.cache_sha256}
    with (a.output_dir/'report.json').open('xb') as stream:
        stream.write(json_bytes(report))
    print(json.dumps({'status': report['status'], 'output_dir': str(a.output_dir),
                      'cache_sha256': report['derived_cache_sha256'], 'contract_sha256': derived['contract_sha256']}))


if __name__ == '__main__':
    main()
