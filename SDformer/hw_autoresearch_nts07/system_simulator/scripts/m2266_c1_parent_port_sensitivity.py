#!/usr/bin/env python3
"""Isolate parent SRAM ports on the existing exact C1 forest, CPU only.

All variants retain one-cycle reads, two response credits, dead-write policy,
arithmetic issues and stable order. Two-bank uses row-id parity, not an oracle.
True dual-port and a double-pumped SP could realize the same service schedule;
this script proves neither clock feasibility nor same physical area/energy.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import m2259_c1_forest_lifetime_probe as base
from m2260_c1_hot_parent_probe import cohort_tiles, pipeline


def serve(masks, residual, parent, order, organization, check_values=False, probe_window=0):
    needs = [int(parent[r]) for r in order if parent[r] >= 0]
    consumers = [i for i, r in enumerate(order) if parent[r] >= 0]
    used = set(needs)
    work = {r: max(int(residual[r]).bit_count(), int(parent[r] >= 0)) for r in order}
    queue = []; pending = None; written = set()
    request = cursor = beat = 0
    count = Counter({k: 0 for k in ('cycles', 'issues', 'stalls', 'reads', 'writes',
                                   'forwards', 'deadline_holds', 'concurrent_rw')})
    # Optional arithmetic sanity check of row reconstruction, not a response-
    # payload or RTL miter. A parent result is immutable within this task.
    # Nonperiodic sources avoid hiding a source-id +/-8 substitution.
    weights = None
    if check_values:
        weights = np.random.default_rng(2266).integers(-128,128,size=(16,96))
        weights[0,0] = -128; weights[-1,-1] = 127
        assert not np.array_equal(weights[:8],weights[8:])
    backing = {}; values = {}; acc = None
    while cursor < len(order):
        row = order[cursor]; p = int(parent[row])
        ready = p < 0 or bool(queue and queue[0] == p)
        final = ready and beat+1 == work[row]
        capacity = len(queue)+int(pending is not None) < 2
        exists = request < len(needs)
        asked = needs[request] if exists else -1
        consumer = consumers[request] if exists else -1
        parallel = organization == '1r1w' or (
            organization == 'two_1rw_parity' and asked >= 0 and (asked & 1) != (row & 1))
        hold = bool(final and row in used and exists and capacity and asked in written
                    and asked != row and consumer == cursor+1 and not parallel)
        issue = ready and not hold
        last = issue and beat+1 == work[row]
        forward = bool(last and exists and capacity and asked == row)
        write = bool(last and row in used)
        read = bool((not write or parallel) and not forward and exists and capacity and asked in written)
        if probe_window and not ready and beat == 0:
            # A counterfactual opportunity count, NOT an out-of-order replay:
            # only already queued parent payloads or root rows count as ready.
            # A parent merely in backing SRAM is not an immediate candidate.
            others = order[cursor+1:cursor+probe_window]
            root_ready = any(parent[r] < 0 for r in others)
            response_ready = any(parent[r] >= 0 and int(parent[r]) in queue for r in others)
            count.update(start_blocked_cycles=1,
                         start_blocked_with_root_in_window=int(root_ready),
                         start_blocked_with_queued_parent_in_window=int(response_ready),
                         start_blocked_with_any_ready_in_window=int(root_ready or response_ready))
        assert not (read and write and asked == row)
        if read and write:
            assert parallel
        if check_values and issue:
            if beat == 0:
                acc = np.zeros(96, dtype=np.int64) if p < 0 else values[p].copy()
            bits = [b for b in range(16) if int(residual[row]) >> b & 1]
            if beat < len(bits):
                acc += weights[bits[beat]]
            if last:
                expected = sum((weights[b] for b in range(16) if int(masks[row]) >> b & 1), np.zeros(96, dtype=np.int64))
                assert np.array_equal(acc, expected)
                assert np.all((-2048 <= acc) & (acc <= 2047))
                if write:
                    backing[row] = acc.copy()
                if forward:
                    values[row] = acc.copy()
        if last and p >= 0:
            assert queue and queue.pop(0) == p
        if pending is not None:
            queue.append(pending)
        if forward:
            queue.append(asked); request += 1
        if read:
            if check_values:
                values[asked] = backing[asked].copy()
            request += 1
        pending = asked if read else None
        if write:
            written.add(row)
        assert len(queue)+int(pending is not None) <= 2
        count.update(cycles=1, issues=int(issue), stalls=int(not issue), reads=int(read),
                     writes=int(write), forwards=int(forward), deadline_holds=int(hold),
                     concurrent_rw=int(read and write))
        if issue:
            if last:
                cursor += 1; beat = 0
            else:
                beat += 1
        assert count['cycles'] <= sum(work.values())+2*len(order)+8
    assert request == len(needs) and not queue and pending is None
    assert count['issues'] == sum(work.values())
    return dict(count)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', type=int, default=1)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--probe-ready-window', type=int, default=0)
    args = ap.parse_args()
    axes = ('1rw', 'two_1rw_parity', '1r1w')
    rng = np.random.default_rng(2266)
    cases = [[0]*64, [1,3,7,15], [3]*64, [1,2,3,4,5,7,15,0]]
    cases.extend(rng.integers(0,65536,size=64,dtype=np.uint16) for _ in range(32))
    for raw in cases:
        masks = np.asarray(raw,dtype=np.uint16)
        residual, parent = base.old.M504.cleanroom_subset(masks)
        order, _ = base.forest_orders(masks,parent)
        reference = base.serve(masks,residual,parent,order)
        for axis in axes:
            actual = serve(masks,residual,parent,order,axis,True)
            if axis == '1rw':
                assert all(actual[k] == reference[k] for k in actual if k in reference)
    totals = {axis: Counter() for axis in axes}; phases = []; tiles = 0
    with base.LEDGER.open('rb') as stream:
        for sample in range(args.samples):
            for operator in range(4):
                for identity, tile_masks in cohort_tiles(stream,sample,operator,'native_k'):
                    pre = []; works = {axis: [] for axis in axes}
                    for masks in tile_masks:
                        residual, parent = base.old.M504.cleanroom_subset(masks)
                        order, _ = base.forest_orders(masks,parent)
                        reference = base.serve(masks,residual,parent,order)
                        bitcap = (len(masks)+7)//8
                        frontend = bitcap+sum(int(m).bit_count()>1 for m in masks)+17*bitcap+2
                        pre.append(max(160,frontend) if np.any(masks) else frontend)
                        for axis in axes:
                            actual = serve(masks,residual,parent,order,axis,probe_window=args.probe_ready_window)
                            if axis == '1rw':
                                assert all(actual[k] == reference[k] for k in actual if k in reference)
                            totals[axis].update(actual)
                            works[axis].append(8*actual['cycles'])
                        tiles += 1
                    points = {axis: pipeline(pre,works[axis]) for axis in axes}
                    for axis in axes:
                        totals[axis]['phase_pipeline_cycles'] += points[axis]
                        totals[axis]['no_overlap_cycles'] += sum(pre)+sum(works[axis])+2*len(pre)
                    phases.append(dict(sample=sample,operator=operator,**identity,cycles=points))
            print(f'finished sample {sample}',flush=True)
    comparisons = {}
    for axis in axes:
        comparisons[axis] = {key+'_ratio': totals['1rw'][key]/totals[axis][key]
                             for key in ('cycles','phase_pipeline_cycles','no_overlap_cycles')}
    result = dict(scope='Selected ep34 C1 parent-service port sensitivity, not complete Conv/RTL/PPA',
        cohort='samples 0..N-1 x four Conv x spatial chunks 0/23/46 x all contiguous 432 K tasks',
        samples=args.samples,tiles=tiles,self_test_cases=len(cases),totals=totals,
        ready_window_probe=dict(entries_including_blocked_row=args.probe_ready_window,
            scope='Fixed-schedule opportunity only; no scheduling decision, new cycles, window metadata cost or RTL claim'),
        comparisons=comparisons,phases=phases,
        boundaries=['Same exact forest, stable order, two response credits, one-cycle parent-read latency',
            'Input/weight/prior-psum availability and output acceptance remain unstalled service assumptions',
            'Two-bank row-parity needs two independent full-width banks, not the existing nine width slices',
            '1r1w scheduling neither establishes a generated dual-port macro nor a double-clock SP timing result',
            'Physical macro depth/width rounding, muxes, clocks, area and energy are not priced',
            'No re-training, SNN timestep reorder or new product-sparsity definition'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(comparisons,indent=2))


if __name__ == '__main__':
    main()
