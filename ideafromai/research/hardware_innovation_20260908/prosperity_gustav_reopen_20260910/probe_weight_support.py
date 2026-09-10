"""Bounded, real-data structural probe; no clock, bandwidth, or AEE model."""
from pathlib import Path
import json
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
ALG = ROOT / 'algorithm'
FRAMES = ['zurich_city_02_c_0001', 'zurich_city_07_a_0001',
          'zurich_city_09_a_0001', 'zurich_city_11_b_0001']
POP = np.array([int(x).bit_count() for x in range(65536)], dtype=np.int16)
M, K = 64, 16
IDX = np.arange(M)
BEFORE_INDEX = IDX[None, :] < IDX[:, None]


def subset_order(masks):
    """Rows are children, columns parents; tie on original row index."""
    pc = POP[masks]
    return ((masks[..., None, :] & masks[..., :, None]) == masks[..., None, :]) & (
        (pc[..., None, :] < pc[..., :, None]) |
        ((pc[..., None, :] == pc[..., :, None]) & BEFORE_INDEX))


def best(legal, gain):
    scores = np.where(legal, gain[..., None, :], -1)
    parent = np.argmax(scores, axis=-1)
    value = np.take_along_axis(scores, parent[..., None], axis=-1)[..., 0]
    return np.maximum(value, 0), np.where(value > 0, parent, -1)


def add(total, name, value):
    total[name] = total.get(name, 0) + int(value)


def one_tile(raw, w_masks, weights, stats, check_numeric=False):
    h_count = len(w_masks)
    raw_pc = POP[raw]
    raw_legal = subset_order(raw)
    raw_parent = np.argmax(np.where(raw_legal, raw_pc[None, :], -1), axis=-1)
    raw_has_parent = raw_legal.any(-1)
    unique_w, inverse = np.unique(w_masks, return_inverse=True)
    effective = unique_w[:, None] & raw[None, :]
    pc = POP[effective]
    gain = np.maximum(pc - 1, 0)
    legal = subset_order(effective)
    ideal_gain, ideal_parent = best(legal, gain)
    common_order = ((raw_pc[None, :] < raw_pc[:, None]) |
                    ((raw_pc[None, :] == raw_pc[:, None]) & BEFORE_INDEX))
    # Effective subset with a shared original-popcount topological order.
    fixed_legal = ((effective[:, None, :] & effective[:, :, None]) ==
                   effective[:, None, :]) & common_order
    fixed_gain, _ = best(fixed_legal, gain)
    raw_lane_gain, _ = best(raw_legal, gain)
    all_pc = pc[inverse]
    all_gain = gain[inverse]
    all_parent = ideal_parent[inverse]
    original_gain = all_gain[:, raw_parent] * raw_has_parent[None, :]
    add(stats, 'effective_coefficient_terms', all_pc.sum())
    add(stats, 'original_raw_forest_gain_proxy', original_gain.sum())
    add(stats, 'best_raw_legal_per_lane_gain_proxy', raw_lane_gain[inverse].sum())
    add(stats, 'weight_aware_per_lane_gain_proxy', ideal_gain[inverse].sum())
    add(stats, 'weight_aware_shared_order_gain_proxy', fixed_gain[inverse].sum())
    add(stats, 'nontrivial_parent_uses', (all_parent >= 0).sum())
    child = np.arange(M)[None, :]
    parents_nonnegative = np.maximum(all_parent, 0)
    # Whether the selected parent is outside the raw activation subset graph.
    is_new = ~raw_legal[child, parents_nonnegative] & (all_parent >= 0)
    add(stats, 'selected_parent_outside_raw_graph', is_new.sum())

    for size in (8, 4, 2, 1):
        if size == 1:
            group_gain = ideal_gain[inverse]
            raw_group_gain = raw_lane_gain[inverse]
        else:
            grouped = np.bitwise_or.reduce(w_masks.reshape(-1, size), axis=1)
            group_masks = grouped[:, None] & raw[None, :]
            group_values = all_gain.reshape(-1, size, M).sum(axis=1)
            group_gain, _ = best(subset_order(group_masks), group_values)
            raw_group_gain, _ = best(raw_legal, group_values)
        add(stats, f'h{size}_one_plan_gain_proxy', group_gain.sum())
        add(stats, f'h{size}_raw_one_plan_gain_proxy', raw_group_gain.sum())
        add(stats, f'h{size}_zero_columns',
            np.count_nonzero(~np.any(weights.reshape(-1, size, K) != 0, axis=1)))
        add(stats, f'h{size}_columns', h_count // size * K)

    parents_h8 = all_parent.reshape(-1, 8, M).transpose(0, 2, 1)
    sorted_parents = np.sort(parents_h8, axis=-1)
    unique_flags = np.concatenate([np.ones_like(sorted_parents[..., :1], dtype=bool),
                                   sorted_parents[..., 1:] != sorted_parents[..., :-1]], axis=-1)
    distinct = (unique_flags & (sorted_parents >= 0)).sum(-1)
    hist = np.bincount(distinct.reshape(-1), minlength=9)
    for count, number in enumerate(hist):
        add(stats, f'h8_independent_parent_count_{count}', number)
    # Opposite selected edges in the same H8 group: each scalar graph is still a DAG.
    adjacency = np.zeros((h_count // 8, M, M), dtype=bool)
    group, row, lane = np.nonzero(parents_h8 >= 0)
    adjacency[group, row, parents_h8[group, row, lane]] = True
    reciprocal = adjacency & adjacency.transpose(0, 2, 1)
    add(stats, 'h8_opposite_selected_edge_pairs', np.triu(reciprocal, 1).sum())
    add(stats, 'tiles', 1)

    if check_numeric:
        gates = ((raw[:, None].astype(np.uint32) >> np.arange(K)) & 1).astype(np.int64)
        reference = weights.astype(np.int64) @ gates.T
        hh, rr = np.nonzero(all_parent >= 0)
        pp = all_parent[hh, rr]
        residual = gates[rr] * (1 - gates[pp])
        rebuilt = reference[hh, pp] + (residual * weights[hh].astype(np.int64)).sum(-1)
        errors = np.count_nonzero(rebuilt != reference[hh, rr])
        add(stats, 'integer_parent_identity_checks', len(rr))
        add(stats, 'integer_parent_identity_errors', errors)


def enrich(d):
    n = d['effective_coefficient_terms']
    for key in list(d):
        if key.endswith('_gain_proxy'):
            d[key + '_fraction_of_terms'] = d[key] / n
    d['new_parent_fraction'] = d['selected_parent_outside_raw_graph'] / max(1, d['nontrivial_parent_uses'])
    for size in (8, 4, 2, 1):
        d[f'h{size}_zero_column_fraction'] = d[f'h{size}_zero_columns'] / d[f'h{size}_columns']
    return d


def main():
    started = time.time()
    dictionary = np.load(ALG / 'stage2_temporal_codes/codebooks.npz')['s2b3_dictionary']
    report = {
        'scope': 'Structural probe, 4 fixed frames, 32 uniformly spaced positions each, full T10/C384/H1536; local M64/K16 relations only.',
        'identity': 'Existing trained integer bits3 s2b3 student, not native ep34 or lifting student; source static theta is already folded into saved W. tau remains distinct.',
        'source_decode': 'dictionary[codes], not B times E; one common source gate capture for both weight controls (captures were previously checked equal).',
        'metric': 'gain_proxy = effective parent coefficients - 1 for parents with at least 2 coefficients; pays one scalar parent-add but not parent storage, traffic, graph detection, issue, full-K accumulation or PSN.',
        'limits': ['Not cycles, full-layer schedule, bandwidth, RTL or AEE.',
                   'Does not compare complete common-node CSE/APEC with zero-residual aliasing; that control is necessary before interpreting novelty.',
                   'Independent lane optima are optimistic and use more parent plans; same-address W broadcast is not an additional saving.',
                   'This block is a diagnostic source of real pruned weights, not a claim it is the best publication target.'],
        'frames': FRAMES, 'axes': {}}
    code_base = ALG / 'group_pruning_probe/controls/codes/row_2of4_trained'
    all_gates = {}
    for frame in FRAMES:
        codes = np.load(code_base / frame / 's2b3.npz')['codes']
        positions = np.linspace(0, len(codes) - 1, 32, dtype=int)
        gates = dictionary[codes[positions]].transpose(0, 2, 1).reshape(-1, 384)
        all_gates[frame] = gates
    for axis in ('original_trained', 'row_2of4_trained'):
        w = np.load(ALG / 'group_pruning_probe/controls' / (axis + '.npz'))['weight_int8']
        total = {}
        frames = []
        for frame, gates in all_gates.items():
            stats = {}
            masks = (gates.reshape(-1, 24, K).astype(np.uint32) *
                     (1 << np.arange(K, dtype=np.uint32))).sum(-1).astype(np.uint16)
            for c in range(24):
                local_w = w[:, c*K:(c+1)*K]
                wm = ((local_w != 0).astype(np.uint32) *
                      (1 << np.arange(K, dtype=np.uint32))).sum(-1).astype(np.uint16)
                for m in range(0, len(masks), M):
                    one_tile(masks[m:m+M, c], wm, local_w, stats,
                             check_numeric=(c == 0 and m == 0))
            for key, value in stats.items():
                add(total, key, value)
            frames.append({'frame': frame, **enrich(stats)})
            print(json.dumps({'axis': axis, 'frame': frame,
                'weight_parent_proxy': stats['weight_aware_per_lane_gain_proxy'],
                'raw_parent_proxy': stats['best_raw_legal_per_lane_gain_proxy'],
                'elapsed_s': round(time.time() - started, 1)}), flush=True)
        report['axes'][axis] = {'weight_nonzeros': int(np.count_nonzero(w)),
                                'aggregate': enrich(total), 'per_frame': frames}
        (OUT / 'weight_support_result.json').write_text(json.dumps(report, indent=2) + '\n')
    report['elapsed_s'] = time.time() - started
    (OUT / 'weight_support_result.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
