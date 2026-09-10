"""Exploratory graph change, after fixed basis routing failed. Not hardware time."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from collections import Counter
import itertools
import json
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import screen_materialization as ref
old, digest, LEDGER, LEDGER_SHA = ref.old, ref.digest, ref.LEDGER, ref.LEDGER_SHA


def forest(original, virtual):
    masks = original + virtual
    n = len(masks)
    arr = np.array(masks, dtype=np.uint16)
    pc = old.POP[arr].astype(np.int32)
    idx = np.arange(n)
    legal = ((arr[:, None] & arr[None, :]) == arr[None, :])
    legal &= ~((arr[:, None] == arr[None, :]) & (idx[None, :] >= idx[:, None]))
    legal &= pc[None, :] > 0
    scores = np.where(legal, pc[None, :], 0)
    pidx = scores.argmax(1)
    parents = np.where((scores.max(1) > 0) & (pc >= 2), pidx, -1).astype(np.int32)
    needed = set(range(len(original)))
    for node in sorted(range(n), key=lambda i: (-pc[i], -i)):
        if node in needed and parents[node] >= 0:
            needed.add(int(parents[node]))
    kept = [v for i, v in enumerate(virtual, len(original)) if i in needed]
    if len(kept) != len(virtual):
        return forest(original, kept)
    remaining = Counter(int(p) for p in parents if p >= 0)
    c, live = Counter(), set()
    order = sorted(range(n), key=lambda i: (pc[i], i))
    for node in order:
        p = int(parents[node])
        residual_pc = int(pc[node] - (pc[p] if p >= 0 else 0))
        c['source_coefficient_reads'] += residual_pc
        c['binary_vector_adds'] += residual_pc if p >= 0 else max(0, residual_pc-1)
        if p >= 0:
            assert p in live
            c['wide_reads'] += 1
            remaining[p] -= 1
            if remaining[p] == 0:
                live.remove(p)
        if remaining[node]:
            live.add(node)
            c['wide_writes'] += 1
        c['peak_live_parent_vectors'] = max(c['peak_live_parent_vectors'], len(live))
    assert not live
    c['wide_read_write_events'] = c['wide_reads'] + c['wide_writes']
    c['original_destination_commits'] = len(original)
    c['live_virtual_nodes'] = len(kept)
    return kept, masks, parents, order, c


def verify(graph, original_count):
    _, masks, parents, order, _ = graph
    values = {}
    for node in order:
        p = int(parents[node])
        if p >= 0:
            values[node] = values[p] + old.vector_value(masks[node] ^ masks[p])
        else:
            values[node] = old.vector_value(masks[node])
        assert np.array_equal(values[node], old.vector_value(masks[node]))
    return original_count * old.LANES


def optimize(original, pool, budget, mode):
    current = forest(original, [])
    steps, planner = [], Counter()
    for _ in range(budget):
        best, best_key, chosen = None, None, None
        for v in pool:
            if v in current[0]:
                continue
            planner['exact_graph_trials'] += 1
            planner['trial_relation_pair_upper_bound_before_dead_node_pruning'] += (len(original) + len(current[0]) + 1)**2
            trial = forest(original, current[0] + [v])
            now, after = current[-1], trial[-1]
            if after['binary_vector_adds'] >= now['binary_vector_adds']:
                continue
            if mode == 'resource_dominance' and any(after[k] > now[k] for k in (
                    'source_coefficient_reads', 'wide_read_write_events', 'peak_live_parent_vectors')):
                continue
            key = (after['binary_vector_adds'], after['wide_read_write_events'], v)
            if best_key is None or key < best_key:
                best, best_key, chosen = trial, key, v
        if best is None:
            break
        verified = verify(best, len(original))
        steps.append({'new_virtual_mask': chosen, 'retained_virtual_masks': best[0],
                      'counts': dict(best[-1]), 'diagnostic_lane_values_verified': verified})
        current = best
    return {'steps': steps, 'final': dict(current[-1]), 'virtual_masks': current[0], 'planner': dict(planner)}


def main():
    out = BASE / 'virtual_parents_r1.json'
    assert not out.exists()
    plan_path = BASE / 'virtual_parent_plan.json'
    plan = json.loads(plan_path.read_text())
    plan_sha = digest(plan_path)
    assert digest(LEDGER) == LEDGER_SHA
    main_plan = json.loads((BASE / 'materialization_plan.json').read_text())
    sealed = json.loads((BASE / 'materialization_r1.json').read_text())
    totals, baseline, tiles = {}, Counter(), []
    # A genuinely absent shared parent: 0111 and1011 share0011.
    directed = optimize([7, 11], [3], 4, 'arithmetic_only')
    assert directed['final']['binary_vector_adds'] == 3
    # Dominance gate correctly refuses this tiny case if it adds port/state costs.
    assert not optimize([7, 11], [3], 4, 'resource_dominance')['steps']
    with LEDGER.open('rb') as stream:
        for sample, op, part, chunk in itertools.product(main_plan['evaluation_samples'], main_plan['operators'], main_plan['k_partitions'], main_plan['evaluation_chunks']):
            rows, _ = old.read_rows(stream, sample, op, part, chunk*64, min(64, 3000-chunk*64))
            original_set = set(rows)
            pair_count = len(rows)*(len(rows)-1)//2
            pool = sorted({a & b for i,a in enumerate(rows) for b in rows[i+1:]
                           if (a & b).bit_count() >= 2 and (a & b) not in original_set})
            base_graph = forest(rows, [])
            base = base_graph[-1]
            verified = verify(base_graph, len(rows))
            ref.add(baseline, base)
            variants = {}
            for mode in plan['variants']:
                point = optimize(rows, pool, plan['virtual_node_budget'], mode)
                variants[mode] = point
                total = totals.setdefault(mode, Counter())
                ref.add(total, point['final'])
                total['tiles'] += 1
                total['tiles_admitted_at_least_one_virtual'] += bool(point['steps'])
                total['accepted_steps'] += len(point['steps'])
                total['diagnostic_lane_values_verified'] += verified + sum(s['diagnostic_lane_values_verified'] for s in point['steps'])
                total['original_pairwise_intersections_considered'] += pair_count
                total['distinct_absent_nontrivial_intersections'] += len(pool)
                total.update(point['planner'])
            tiles.append({'sample': sample, 'operator': op, 'k_partition': part, 'chunk': chunk,
                          'candidate_pool_size': len(pool), 'baseline': dict(base), 'variants': variants})
            if len(tiles) % 32 == 0:
                print('Graph-change tiles evaluated:', len(tiles), flush=True)
    assert len(tiles) == 192
    for k in ('binary_vector_adds', 'source_coefficient_reads', 'wide_reads', 'wide_writes', 'wide_read_write_events', 'peak_live_parent_vectors'):
        assert baseline[k] == sealed['strong_baseline'][k], k
    assert digest(plan_path) == plan_sha
    report = {'date': '2026-09-07', 'status': 'EXPLORATORY_GRAPH_AUGMENTATION_REFERENCE',
              'plan': plan, 'plan_sha256': plan_sha, 'script_sha256': digest(Path(__file__)),
              'imported_reference_sha256': digest(BASE/'screen_materialization.py'),
              'ledger_sha256': LEDGER_SHA, 'same_cohort_result_sha256': digest(BASE/'materialization_r1.json'),
              'baseline': dict(baseline), 'aggregate': {k: dict(v) for k,v in totals.items()}, 'tiles': tiles,
              'directed_missing_intersection_example': directed,
              'limits': ['Greedy CPU trial planner is deliberately exhaustive; its cost is not a hardware implementation.',
                         'Event dominance omits planning/matching and forwarding, so it is not latency/energy dominance.',
                         'The arithmetic-only path can require more parent-port service; report separately.',
                         'Pairwise intersection mining is inherited from CSE/SumMerge and related graph work.',
                         'Original same-cohort peak live state is17. No mapped SRAM area or physical capacity assertion.'],
              'claim_boundary': {'rtl_speedup': False, 'ppa': False, 'new_AEE': False, 'frozen_fp32_equivalence': False, 'new_blind_test': False}}
    with out.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'baseline': dict(baseline), 'aggregate': report['aggregate']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
