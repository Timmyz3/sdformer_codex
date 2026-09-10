"""Trade strict early-completion reach for a small bound interface.

Uses the existing exact integer private-tail execution, source order and checks.
All bounds are conservative. Bit packing is a capacity count, not circuit area.
"""
import json
from pathlib import Path

import numpy as np

from private_tail_replay import prepare, replay
from source_count_bound_probe import N_UP, capture_geometry, compile_bounds

HERE = Path(__file__).resolve().parent


def upper_power2(x):
    return np.fromiter((0 if int(v) == 0 else 1 << (int(v)-1).bit_length()
                       for v in x.flat), dtype=np.int64, count=x.size).reshape(x.shape)


def main():
    params = dict(np.load(HERE/'integer_deployment/common3_diagonal_34.npz'))
    weight = params['weight_int8'].reshape(96, 864).astype(np.int64)
    lo, hi = compile_bounds(weight)
    group_lo = np.repeat(lo.reshape(12, 12, 8).min(-1), 8, axis=1)
    group_hi = np.repeat(hi.reshape(12, 12, 8).max(-1), 8, axis=1)
    maxpos = upper_power2(np.maximum(weight.max(1), 0))
    maxneg = upper_power2(np.maximum(-weight.min(1), 0))
    bounds = {
        'exact_channel': (lo, hi),
        'dyadic_channel': (-upper_power2(-lo), upper_power2(hi)),
        'exact_H8_shared': (group_lo, group_hi),
        'dyadic_H8_shared': (-upper_power2(-group_lo), upper_power2(group_hi)),
        'count_times_dyadic_max': (-N_UP[:, None]*maxneg, N_UP[:, None]*maxpos),
    }
    for name, (l, h) in bounds.items():
        if np.any(l > lo) or np.any(h < hi):
            raise RuntimeError(f'{name} is not an outer bound of the exact table')
    result = dict(
        scope='same common3 integer student and four native P4 captures; same paid three-column prefix and private-tail K-order execution',
        purpose='test the accuracy of a smaller strict bound interface before adding any circuit; rounding and channel sharing are known techniques, not a novelty claim',
        packing=dict(exact_per_channel_INT16_bytes=int(lo.size*2*2),
                     exact_H8_INT16_bytes=int(lo.size//8*2*2),
                     dyadic_per_channel_5bit_bytes=int(lo.size*2*5//8),
                     dyadic_H8_5bit_bytes=int(lo.size//8*2*5//8),
                     count_times_dyadic_max_5bit_constants_bytes=96*2*5//8,
                     exponent_code='0 means zero bound; code e+1 means 2**e, outward rounded; low endpoint is the negative magnitude'),
        exclusions=['no physical cycle, ports or energy inference',
                    'all variants still pay the original prefix and raw threshold generation',
                    'dyadic bounds can use threshold-distance/sign-extension checks but those are not implemented here',
                    'logical table-hit counts from the old generic evaluator are not transactions for the packed formats',
                    'count-times-maximum is a strong simple control with no count-indexed table; it is usually looser'],
        frames=[], totals={})
    if np.abs(lo).max() > 32767 or hi.max() > 32767:
        raise RuntimeError('Exact signed16 table does not cover these weights')
    for path in sorted((HERE/'integer_valid10').glob('capture_*.npz')):
        with np.load(path) as sample:
            geom = capture_geometry(sample, weight)
            data = prepare(sample, params, geom)
        frame = dict(capture=path.name, prefix_products=data['head']['shared_prefix_active_products'],
                     full_Conv_products=data['head']['shared_prefix_active_products'] + data['head']['literal_complete_tail_active_products'], axes={})
        for name, (l, h) in bounds.items():
            metrics = replay(data, weight, l, h, 'count_bin_retirement')
            frame['axes'][name] = metrics
        result['frames'].append(frame)
        print(path.name, {k:v['tail_active_source_weight_live_products'] for k,v in frame['axes'].items()}, flush=True)
    result['prefix_products'] = sum(f['prefix_products'] for f in result['frames'])
    result['full_Conv_products'] = sum(f['full_Conv_products'] for f in result['frames'])
    for name in bounds:
        result['totals'][name] = {k: sum(f['axes'][name][k] for f in result['frames'])
                                 for k in result['frames'][0]['axes'][name]
                                 if k != 'peak_abs_partial_Y'}
        axis = result['totals'][name]
        axis['prefix_plus_tail_products'] = result['prefix_products'] + axis['tail_active_source_weight_live_products']
        axis['reduction_vs_same_full_Conv'] = 1-axis['prefix_plus_tail_products']/result['full_Conv_products']
        axis['peak_abs_partial_Y'] = max(f['axes'][name]['peak_abs_partial_Y'] for f in result['frames'])
    (HERE/'compact_bound_probe.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
