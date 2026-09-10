"""Ordinary partial-lane inheritance control, fixed common DAG order.

Counts logical H8 vector accesses, not physical SRAM words or clocks. A parent
vector and coefficient vector each cost one in the selection proxy; their
different widths, packing, ports, caching, writes and scheduling are not modeled.
"""
import json
import time
import numpy as np
from probe_weight_support import ALG, OUT, FRAMES, M, K, POP, subset_order, add


def pick(cost, used, direct, permitted=None):
    if permitted is not None:
        cost = np.where(permitted[None, :, :], cost, 1000)
    parent = cost.argmin(-1)
    selected = np.take_along_axis(cost, parent[..., None], -1)[..., 0]
    take = selected < direct
    parent = np.where(take, parent, -1)
    reads = np.take_along_axis(used, np.maximum(parent, 0)[..., None], -1)[..., 0] & take
    total = np.where(take, selected, direct)
    return total, reads, parent


def tile(raw, wm, result):
    group_w = wm.reshape(-1, 8)
    effective = raw[None, :, None] & group_w[:, None, :]
    child = effective[:, :, None, :]
    parent = effective[:, None, :, :]
    pc = POP[raw]
    order = ((pc[None, :] < pc[:, None]) |
             ((pc[None, :] == pc[:, None]) &
              (np.arange(M)[None, :] < np.arange(M)[:, None])))
    legal = ((parent & child) == parent) & order[None, :, :, None] & (POP[parent] >= 2)
    residual = child & ~np.where(legal, parent, 0).astype(np.uint16)
    coeff = POP[np.bitwise_or.reduce(residual, axis=-1)]
    used = legal.any(-1)
    cost = coeff + used
    direct = POP[np.bitwise_or.reduce(effective, axis=-1)]
    add(result, 'direct_coefficient_vector_uses', direct.sum())
    for axis, permitted in [('raw_parent_with_lane_skip', subset_order(raw)),
                            ('weight_masked_parent', None)]:
        total, reads, selected = pick(cost, used, direct, permitted)
        add(result, axis + '_coefficient_vector_uses', (total - reads).sum())
        add(result, axis + '_parent_vector_reads', reads.sum())
        add(result, axis + '_access_proxy', total.sum())
        gg, rr = np.nonzero(reads)
        pp = selected[gg, rr]
        histogram = np.bincount(legal[gg, rr, pp].sum(-1).astype(int), minlength=9)
        for lanes, count in enumerate(histogram):
            add(result, axis + f'_inheriting_lanes_{lanes}', count)
        referenced = np.zeros((len(group_w), M), dtype=bool)
        referenced[gg, pp] = True
        add(result, axis + '_distinct_referenced_parent_vectors', referenced.sum())
    add(result, 'tiles', 1)


def main():
    started = time.time()
    w = np.load(ALG / 'group_pruning_probe/controls/row_2of4_trained.npz')['weight_int8']
    dictionary = np.load(ALG / 'stage2_temporal_codes/codebooks.npz')['s2b3_dictionary']
    all_stats, per_frame = {}, []
    for frame in FRAMES:
        code = np.load(ALG / 'group_pruning_probe/controls/codes/row_2of4_trained' / frame / 's2b3.npz')['codes']
        positions = np.linspace(0, len(code) - 1, 32, dtype=int)
        gates = dictionary[code[positions]].transpose(0, 2, 1).reshape(-1, 384)
        masks = (gates.reshape(-1, 24, K).astype(np.uint32) *
                 (1 << np.arange(K, dtype=np.uint32))).sum(-1).astype(np.uint16)
        stats = {}
        for c in range(24):
            wm = ((w[:, c*K:(c+1)*K] != 0).astype(np.uint32) *
                  (1 << np.arange(K, dtype=np.uint32))).sum(-1).astype(np.uint16)
            for m in range(0, len(masks), M):
                tile(masks[m:m+M, c], wm, stats)
        for name, value in stats.items():
            add(all_stats, name, value)
        per_frame.append({'frame': frame, **stats})
        print(json.dumps({'frame': frame,
                          'raw_access_proxy': stats['raw_parent_with_lane_skip_access_proxy'],
                          'masked_access_proxy': stats['weight_masked_parent_access_proxy'],
                          'elapsed_s': round(time.time() - started, 1)}), flush=True)
    r = {'scope': 'Same fixed four-frame real 2:4 source/weight sample as weight_support_result.json; local M64/K16, H8, one optional parent per child, common raw-popcount DAG order.',
         'ordinary_control': 'A selected parent can serve only legal lanes; other lanes compute directly. This is a strengthened ordinary control, not a new mechanism claim.',
         'selection_proxy': 'Minimize number of logically addressed nonzero H8 coefficient vectors plus one parent-vector read when used. A parent requires at least two nonzero effective coefficients on an inheriting lane.',
         'limits': ['Logical H8 requests, not physical packed memory requests, SRAM bandwidth, cycles or full-layer execution.',
                    'No cost for graph detection, output/parent writes, source decode, full-K state or PSN; actual W and parent widths differ.',
                    'Whole-H8 parent legality is unnecessary: partial-lane fallback is available to all future baselines.',
                    'The same result may be obtainable by common-node CSE and zero-residual aliasing; not yet compared.'],
         'aggregate': all_stats, 'per_frame': per_frame, 'elapsed_s': time.time() - started}
    (OUT / 'masked_parent_result.json').write_text(json.dumps(r, indent=2) + '\n')


if __name__ == '__main__':
    main()
