#!/usr/bin/env python3
"""Fixed-forest parent retention DSE. CPU only; no changed subset algorithm.

Charge order generation AFTER parent selection (one visit plus ancestor-return
steps), then use the existing two-stage tile-overlap equation. A hot result
slot has a 1152-bit payload; its accesses are counted separately, never free
energy. Eviction spills a still-needed result to the existing 1RW backing.
No oracle next-use prediction and no parent re-selection.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import m2259_c1_forest_lifetime_probe as base


def threaded_order(masks, parent):
    """Prepending children while writing the forest needs no subtree sorting."""
    children = [[] for _ in masks]
    roots = []
    for row, mask in enumerate(masks):
        if not mask:
            continue
        p = int(parent[row])
        (roots if p < 0 else children[p]).append(row)
    order = []
    def visit(row):
        order.append(row)
        for child in reversed(children[row]):
            visit(child)
    for root in reversed(roots):
        visit(root)
    # An intentionally charged standalone traversal. One visit per active row,
    # one return per edge; no costless DFS or presumed matcher overlap.
    build_cycles = len(order) + sum(len(c) for c in children)
    return order, build_cycles


def serve_hot(masks, residual, parent, order, slots):
    requirements = [int(parent[r]) for r in order if parent[r] >= 0]
    consumers = [i for i, r in enumerate(order) if parent[r] >= 0]
    remaining = Counter(requirements)
    # Values are admission timestamps: FIFO replacement, not future-use oracle.
    hot = {}
    written = set()
    queue = []
    pending = None
    request = cursor = beat = 0
    count = Counter({k: 0 for k in ('cycles', 'issues', 'stalls', 'reads',
        'writes', 'forwards', 'deadline_holds', 'hot_reads', 'hot_writes',
        'hot_releases', 'spill_writes')})
    work = {r: max(int(residual[r]).bit_count(), int(parent[r] >= 0)) for r in order}
    while cursor < len(order):
        row = order[cursor]
        p = int(parent[row])
        ready = p < 0 or bool(queue and queue[0] == p)
        final = ready and beat + 1 == work[row]
        capacity = len(queue) + int(pending is not None) < 2
        exists = request < len(requirements)
        asked = requirements[request] if exists else -1
        consumer = consumers[request] if exists else -1

        # Release the current parent only at child completion, not speculative
        # prefetch. Queue entries already hold their own immutable snapshot.
        retiring = p if final and p >= 0 and remaining[p] == 1 else -1
        survivors = [r for r in hot if r != retiring]
        admit = final and remaining[row] > 0
        victim = min(survivors, key=hot.get) if admit and len(survivors) >= slots else -1
        spill = victim >= 0 and victim not in written
        hot_ready = exists and asked in hot
        hold = bool(spill and exists and capacity and asked in written
                    and not hot_ready and consumer == cursor + 1)
        issue = ready and not hold
        last = issue and beat + 1 == work[row]
        forward = bool(last and exists and capacity and asked == row)
        hot_read = bool(not forward and exists and capacity and hot_ready)
        write = bool(last and spill)
        read = bool(not write and not forward and not hot_read and exists
                    and capacity and asked in written)
        assert not (read and write)
        if last and p >= 0:
            assert queue and queue.pop(0) == p
        if pending is not None:
            queue.append(pending)
        if forward or hot_read:
            queue.append(asked)
            request += 1
        if read:
            request += 1
        pending = asked if read else None
        assert len(queue) + int(pending is not None) <= 2
        if last:
            if p >= 0:
                remaining[p] -= 1
                if remaining[p] == 0 and p in hot:
                    del hot[p]
                    count['hot_releases'] += 1
            if admit:
                if victim >= 0:
                    del hot[victim]
                    if write:
                        written.add(victim)
                hot[row] = count['cycles']
                count['hot_writes'] += 1
        count.update(cycles=1, issues=int(issue), stalls=int(not issue),
                     reads=int(read), writes=int(write), forwards=int(forward),
                     hot_reads=int(hot_read), deadline_holds=int(hold),
                     spill_writes=int(write), hot_spill_reads=int(write))
        if issue:
            if last:
                cursor += 1
                beat = 0
            else:
                beat += 1
        assert len(hot) <= slots
        assert count['cycles'] <= sum(work.values()) + 4 * len(order) + 16
    assert request == len(requirements) and not queue and pending is None
    assert not hot and not any(remaining.values())
    assert count['issues'] == sum(work.values())
    return dict(count)


def storage_replay(masks, residual, parent, order, slots):
    """Independent value-carrying reference checks spill/release, not only tags.

    Actual 96x12-bit product-row payload. Values are deterministic signed INT8
    weights and stay in the exact 12-bit range (16 * 127 <= 2047).
    """
    weights = (np.arange(16 * 96, dtype=np.int64).reshape(16, 96) * 29) % 255 - 127
    remaining = Counter(int(parent[r]) for r in order if parent[r] >= 0)
    hot = {}; backing = {}; birth = {}; spills = 0
    for tick, row in enumerate(order):
        p = int(parent[row])
        value = np.zeros(96, dtype=np.int64) if p < 0 else (hot[p] if p in hot else backing[p]).copy()
        for bit in range(16):
            if int(residual[row]) >> bit & 1:
                value += weights[bit]
        dense = sum((weights[b] for b in range(16) if int(masks[row]) >> b & 1), np.zeros(96, dtype=np.int64))
        assert np.array_equal(value, dense)
        assert np.all((-2048 <= value) & (value <= 2047))
        if p >= 0:
            remaining[p] -= 1
            if not remaining[p] and p in hot:
                del hot[p]; del birth[p]
        if remaining[row]:
            if len(hot) == slots:
                victim = min(birth, key=birth.get)
                if victim not in backing:
                    backing[victim] = hot[victim].copy(); spills += 1
                del hot[victim]; del birth[victim]
            hot[row] = value.copy(); birth[row] = tick
    assert not hot
    return spills


def pipeline(pre, work):
    return int(pre[0] + sum(max(a, b) + 2 for a, b in zip(work[:-1], pre[1:])) + work[-1] + 2)


def self_test():
    count = base.self_test()
    cases = [[0] * 64, [1, 3, 7, 15], [3] * 64,
             [1, 2, 3, 4, 5, 7, 15, 0], [1, 3, 5, 9, 17, 33, 65, 129]]
    rng = np.random.default_rng(2260)
    cases.extend(rng.integers(0, 65536, size=64, dtype=np.uint16) for _ in range(30))
    for raw in cases:
        masks = np.asarray(raw, dtype=np.uint16)
        residual, parent = base.old.M504.cleanroom_subset(masks)
        stable, _ = base.forest_orders(masks, parent)
        dfs, _ = threaded_order(masks, parent)
        for order in (stable, dfs):
            for slots in (1, 2, 4, 64):
                result = serve_hot(masks, residual, parent, order, slots)
                assert result['writes'] == storage_replay(masks, residual, parent, order, slots)
                if slots == 64:
                    assert result['writes'] == result['reads'] == 0
        count += 1
    return count


def cohort_tiles(stream, sample, operator, layout):
    if layout == 'fixed_k':
        for partition in (0, 216, 431):
            phase = (sample * 4 + operator) * 432 + partition
            stream.seek(phase * 3000 * 9)
            masks = np.array([int(v, 16) & 65535 for v in stream.read(3000 * 9).splitlines()], dtype=np.uint16)
            assert len(masks) == 3000
            yield dict(partition=partition), [masks[i:i+64] for i in range(0, 3000, 64)]
    else:
        # Original compute-ledger flatten order is chunk -> K. Keep EVERY K
        # partition within each selected spatial chunk, not artificial fixed-K
        # neighbor pairs. Drain/fill between the three separated spatial chunks.
        for chunk in (0, 23, 46):
            n = min(64, 3000 - chunk * 64)
            tiles = []
            for partition in range(432):
                phase = (sample * 4 + operator) * 432 + partition
                stream.seek((phase * 3000 + chunk * 64) * 9)
                masks = np.array([int(v, 16) & 65535 for v in stream.read(n * 9).splitlines()], dtype=np.uint16)
                assert len(masks) == n
                tiles.append(masks)
            yield dict(chunk=chunk), tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=10)
    ap.add_argument('--layout', choices=('fixed_k', 'native_k'), default='native_k')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    tested = self_test()
    totals = {}; phases = []; peaks = Counter(); tiles = 0
    with base.LEDGER.open('rb') as stream:
        for sample in range(args.samples):
            for operator in range(4):
                for identity, tile_masks in cohort_tiles(stream, sample, operator, args.layout):
                    pp = {}; ww = {}
                    for masks in tile_masks:
                        residual, parent = base.old.M504.cleanroom_subset(masks)
                        stable, _ = base.forest_orders(masks, parent)
                        dfs, build = threaded_order(masks, parent)
                        n = len(masks); bitcap = (n + 7) // 8
                        frontend = bitcap + sum(int(m).bit_count() > 1 for m in masks) + 17 * bitcap + 2
                        pre = max(160, frontend) if np.any(masks) else frontend
                        for name, order, slots in [('stable', stable, 0), ('threaded', dfs, 0)] + [
                                (f'{label}_hot{k}', order, k) for label, order in [('stable', stable), ('threaded', dfs)] for k in (1, 2, 4)]:
                            event = serve_hot(masks, residual, parent, order, slots) if slots else base.serve(masks, residual, parent, order)
                            if slots:
                                assert event['writes'] == storage_replay(masks, residual, parent, order, slots)
                            else:
                                base.numeric(masks, residual, parent, order)
                            live = base.live_metrics(order, parent)
                            peaks[name] = max(peaks[name], live['peak_live_parent_vectors'])
                            event['live_vector_row_steps'] = live['live_vector_row_steps']
                            totals.setdefault(name, Counter()).update(event)
                            pp.setdefault(name, []).append(pre + (build if name.startswith('threaded') else 0))
                            # Parallel 384-bit refcount initialization, one cycle
                            # for each of eight 96-lane output-bank passes. This
                            # is an explicit new control/port assumption to verify.
                            ww.setdefault(name, []).append(8 * (event['cycles'] + int(slots > 0 and bool(order))))
                        tiles += 1
                    phase_result = dict(sample=sample, operator=operator, **identity,
                        cycles={name: pipeline(pp[name], ww[name]) for name in pp})
                    for name, cycles in phase_result['cycles'].items():
                        totals[name]['phase_pipeline_cycles'] += cycles
                        totals[name]['charged_order_cycles'] += sum(pp[name]) - sum(pp['stable'])
                        totals[name]['isolated_tile_cycles'] += sum(pp[name]) + sum(ww[name]) + 2 * len(pp[name])
                    phases.append(phase_result)
            print(f'completed sample {sample}', flush=True)
    reference = totals['stable']; comparisons = {}
    for name, t in totals.items():
        comparisons[name] = dict(service_ratio=reference['cycles'] / t['cycles'],
            phase_pipeline_ratio=reference['phase_pipeline_cycles'] / t['phase_pipeline_cycles'],
            no_intertile_overlap_ratio=reference['isolated_tile_cycles'] / t['isolated_tile_cycles'],
            parent_sram_access_reduction=1 - (t['reads'] + t['writes']) / (reference['reads'] + reference['writes']),
            live_vector_row_steps_reduction=1 - t['live_vector_row_steps'] / reference['live_vector_row_steps'],
            max_dependency_live_vectors=peaks[name])
    result = dict(samples=args.samples, tiles=tiles, phases=phases, self_test_cases=tested,
        scalar_mismatches=0, totals=totals, comparisons=comparisons,
        layout=args.layout,
        cohort=('all 432 contiguous K tasks within each spatial chunk 0/23/46, four Conv operators, each sample'
                if args.layout == 'native_k' else 'spatial tiles within K 0/216/431; artificial adjacency, not native compute-ledger order'),
        scope='CPU service model, fixed exact forest; selected spatial/K population, not full layer, RTL cycles, mapped area, or energy',
        accounting=dict(hot_payload_bytes_per_slot=144, existing_scratch_retained=True,
            active_refcount_bits=384, initial_refcount_double_bank_bits=768,
            threading_first_child_next_sibling_double_bank_bits_with_valid=1792,
            generated_order_double_bank_bits_with_valid=896,
            metadata_subtotal_bits_excluding_tags_arbiters_and_wiring=3840,
            refcount_reload_cycles_per_nonempty_tile=8,
            output_row_id_preserved=True, timestep_neuron_order_changed=False),
        caveats=['Parent selection and residuals unchanged; no multi-parent or eviction-rescue sparsity claim',
            'Standalone DFS visit/return cycles charged before tile execution; overlap only across contiguous tiles by old pipeline equation',
            'New directory/metadata/slot control ports and timing not synthesized; no area or energy claim',
            'Hot hit and victim spill can read different slots while a result is written: explicitly 2R1W register paths; spill reads counted separately',
            'Architectural destination commit remains outside service model; phase ratios are not complete layer or network speedups',
            'Logical lifetime reduction alone does not reduce minimum compiled SRAM macro area'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('totals', 'phases')}, indent=2))


if __name__ == '__main__':
    main()
