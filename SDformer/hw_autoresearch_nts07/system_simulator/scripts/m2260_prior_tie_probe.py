#!/usr/bin/env python3
"""CPU-only first-index versus Prosperity-paper largest-index tie probe.

Paper: https://arxiv.org/html/2503.03379v1#S3.SS4 (III-D pruning rules).
This ports that tie rule into the existing C1 model, NOT official hardware.
The local official Python artifact uses torch.argmax/torch.max first-index ties;
its CUDA scan also retains the first maximum. C1 M504 uses NumPy argmax.
Both variants retain C1's equal-mask p < row and popcount(row) < 2 guards.
Only this script and the requested small JSON result are created.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

import m2260_c1_hot_parent_probe as model


def largest_index_subset(masks):
    masks = np.asarray(masks, dtype=np.uint16)
    population = np.array([int(m).bit_count() for m in masks], dtype=np.int16)
    indices = np.arange(len(masks))
    residual = masks.copy()
    parent = np.full(len(masks), -1, dtype=np.int16)
    for row, current in enumerate(masks):
        if population[row] < 2:
            continue
        eligible = ((masks & current) == masks) & (population > 0)
        eligible &= ~((masks == current) & (indices >= row))
        candidates = indices[eligible]
        if candidates.size:
            best_popcount = int(population[candidates].max())
            chosen = int(candidates[population[candidates] == best_popcount][-1])
            parent[row] = chosen
            residual[row] = current ^ masks[chosen]
    return residual, parent


def check_forests(masks, first, latest):
    r_first, p_first = first
    r_latest, p_latest = latest
    assert np.array_equal(p_first >= 0, p_latest >= 0)
    assert [int(r).bit_count() for r in r_first] == [int(r).bit_count() for r in r_latest]
    assert np.array_equal((r_first == 0) & (p_first >= 0),
                          (r_latest == 0) & (p_latest >= 0))
    for residual, parent in (first, latest):
        for row, p in enumerate(parent):
            if p < 0:
                assert residual[row] == masks[row]
            else:
                assert int(masks[row]).bit_count() >= 2 and masks[p] != 0
                assert masks[p] & masks[row] == masks[p]
                assert masks[p] != masks[row] or p < row
                assert residual[row] == masks[row] ^ masks[p]


def numeric_replay(masks, residual, parent, order, slots, weights):
    """Independent 96-lane integer row/storage oracle; not a cycle miter."""
    bit_indices = np.arange(16)
    dense = ((masks.astype(np.int64)[:, None] >> bit_indices) & 1) @ weights
    adds = ((residual.astype(np.int64)[:, None] >> bit_indices) & 1) @ weights
    remaining = Counter(int(parent[row]) for row in order if parent[row] >= 0)
    hot = {}
    backing = {}
    birth = {}
    spills = 0
    for tick, row in enumerate(order):
        p = int(parent[row])
        value = adds[row].copy()
        if p >= 0:
            value += hot[p] if p in hot else backing[p]
        assert np.array_equal(value, dense[row])
        assert np.all((-2048 <= value) & (value <= 2047))
        if p >= 0:
            remaining[p] -= 1
            if remaining[p] == 0 and p in hot:
                del hot[p]
                del birth[p]
        if remaining[row]:
            if not slots:
                backing[row] = value.copy()
                spills += 1
            else:
                if len(hot) == slots:
                    victim = min(birth, key=birth.get)
                    assert victim not in backing
                    backing[victim] = hot[victim].copy()
                    spills += 1
                    del hot[victim]
                    del birth[victim]
                hot[row] = value.copy()
                birth[row] = tick
    assert not hot and not any(remaining.values())
    return spills, len(order) * 96


def self_test():
    cases = [([3, 3, 3], [-1, 0, 1]),
             ([1, 2, 3], [-1, -1, 1]),
             ([3, 1, 2], [2, -1, -1]),
             ([0, 1, 1, 3, 3], [-1, -1, -1, 2, 3]),
             ([0] * 64, [-1] * 64)]
    for raw, expected in cases:
        masks = np.asarray(raw, dtype=np.uint16)
        latest = largest_index_subset(masks)
        assert latest[1].tolist() == expected
        check_forests(masks, model.base.old.M504.cleanroom_subset(masks), latest)
    rng = np.random.default_rng(226061)
    for _ in range(24):
        masks = rng.integers(0, 256, size=64, dtype=np.uint16)
        forests = (model.base.old.M504.cleanroom_subset(masks), largest_index_subset(masks))
        check_forests(masks, *forests)
        weights = rng.integers(-128, 128, size=(16, 96), dtype=np.int64)
        for residual, parent in forests:
            stable, _ = model.base.forest_orders(masks, parent)
            threaded, _ = model.threaded_order(masks, parent)
            for order, slots in ((stable, 0), (stable, 2), (threaded, 2)):
                event = model.serve_hot(masks, residual, parent, order, slots) if slots else model.base.serve(masks, residual, parent, order)
                writes, _ = numeric_replay(masks, residual, parent, order, slots, weights)
                assert writes == event['writes']
    return len(cases) + 24


def compare(reference, candidate):
    return dict(service_ratio=reference['cycles'] / candidate['cycles'],
                phase_pipeline_ratio=reference['phase_pipeline_cycles'] / candidate['phase_pipeline_cycles'],
                no_intertile_overlap_ratio=reference['isolated_tile_cycles'] / candidate['isolated_tile_cycles'],
                parent_sram_access_reduction=1 - (candidate['reads'] + candidate['writes']) /
                    (reference['reads'] + reference['writes']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert 1 <= args.samples <= 10
    tested = self_test()
    totals = {}
    forest_totals = {label: Counter() for label in ('first', 'latest')}
    changed = Counter()
    phases = []
    tiles = scalar_comparisons = 0
    rng = np.random.default_rng(226062)
    with model.base.LEDGER.open('rb') as stream:
        for sample in range(args.samples):
            for operator in range(4):
                for identity, cohort in model.cohort_tiles(stream, sample, operator, 'native_k'):
                    pre_cycles = {}
                    work_cycles = {}
                    for masks in cohort:
                        first = model.base.old.M504.cleanroom_subset(masks)
                        latest = largest_index_subset(masks)
                        check_forests(masks, first, latest)
                        parent_changes = int(np.count_nonzero(first[1] != latest[1]))
                        changed.update(parent_rows=parent_changes,
                                       residual_pattern_rows=int(np.count_nonzero(first[0] != latest[0])),
                                       tiles=int(parent_changes > 0))
                        cap = (len(masks) + 7) // 8
                        frontend = 18 * cap + sum(int(m).bit_count() > 1 for m in masks) + 2
                        prep = max(160, frontend) if np.any(masks) else frontend
                        weights = rng.integers(-128, 128, size=(16, 96), dtype=np.int64)
                        weights[:, :3] = [-128, 127, 0]
                        for policy, (residual, parent) in (('first', first), ('latest', latest)):
                            stable, _ = model.base.forest_orders(masks, parent)
                            threaded, build = model.threaded_order(masks, parent)
                            stats = forest_totals[policy]
                            residual_issues = sum(int(r).bit_count() for r in residual)
                            synthetic_issues = int(np.count_nonzero((residual == 0) & (parent >= 0)))
                            stats.update(rows=len(masks), active_rows=len(stable),
                                         residual_issues=residual_issues,
                                         synthetic_parent_issues=synthetic_issues,
                                         total_issues=residual_issues + synthetic_issues,
                                         parent_edges=int(np.count_nonzero(parent >= 0)),
                                         unique_parent_rows=len(set(int(p) for p in parent if p >= 0)))
                            for label, order, slots in (('stable', stable, 0),
                                                        ('stable_hot2', stable, 2),
                                                        ('threaded_hot2', threaded, 2)):
                                name = policy + '_' + label
                                event = model.serve_hot(masks, residual, parent, order, slots) if slots else model.base.serve(masks, residual, parent, order)
                                writes, compared = numeric_replay(masks, residual, parent, order, slots, weights)
                                assert writes == event['writes']
                                assert event['issues'] == residual_issues + synthetic_issues
                                scalar_comparisons += compared
                                live = model.base.live_metrics(order, parent)
                                total = totals.setdefault(name, Counter())
                                total.update(event)
                                total['live_vector_row_steps'] += live['live_vector_row_steps']
                                total['max_dependency_live_vectors'] = max(total['max_dependency_live_vectors'], live['peak_live_parent_vectors'])
                                order_charge = build if label == 'threaded_hot2' else 0
                                reload_charge = 8 * int(slots > 0 and bool(order))
                                total['charged_order_cycles'] += order_charge
                                total['refcount_reload_cycles'] += reload_charge
                                pre_cycles.setdefault(name, []).append(prep + order_charge)
                                work_cycles.setdefault(name, []).append(8 * event['cycles'] + reload_charge)
                        tiles += 1
                    phase = dict(sample=sample, operator=operator, **identity, cycles={})
                    for name in pre_cycles:
                        cycles = model.pipeline(pre_cycles[name], work_cycles[name])
                        phase['cycles'][name] = cycles
                        totals[name]['phase_pipeline_cycles'] += cycles
                        totals[name]['isolated_tile_cycles'] += sum(pre_cycles[name]) + sum(work_cycles[name]) + 2 * len(cohort)
                    phases.append(phase)
                    print(f'completed sample={sample} operator={operator} chunk={identity["chunk"]}', flush=True)
    assert forest_totals['first']['total_issues'] == forest_totals['latest']['total_issues']
    reference = totals['first_stable']
    comparisons = {name: compare(reference, total) for name, total in totals.items()}
    matched_ties = {label: compare(totals['first_' + label], totals['latest_' + label])
                    for label in ('stable', 'stable_hot2', 'threaded_hot2')}
    within_latest = {label: compare(totals['latest_stable'], totals['latest_' + label])
                     for label in ('stable_hot2', 'threaded_hot2')}
    result = dict(samples=args.samples, sample_ids=list(range(args.samples)), tiles=tiles,
                  cohort='native chunk->K order; four Conv operators, chunks 0/23/46, all 432 K; drain/fill between chunks',
                  self_test_cases=tested, scalar_comparisons=scalar_comparisons, scalar_mismatches=0,
                  parent_and_residual_changes=dict(changed), forest_totals=forest_totals,
                  totals=totals, comparisons_vs_first_stable=comparisons,
                  latest_vs_first_matched_configuration=matched_ties,
                  within_latest_vs_latest_stable=within_latest, phases=phases,
                  provenance=dict(paper='https://arxiv.org/html/2503.03379v1#S3.SS4',
                                  paper_rule='maximum subset popcount, then largest original index',
                                  artifact_python='third_party/Prosperity/simulator/simulator.py:792; torch.argmax/torch.max first maximum',
                                  artifact_cuda='third_party/Prosperity/kernels/prosparsity_cuda.cu:94; ascending scan and strict >',
                                  c1_first='analyze_m504_h67_single_port_parent_scratch.py:73; numpy.argmax'),
                  accounting=dict(output_bank_passes=8, service_counters='per 96-lane bank, summed over tiles; pipeline work multiplies by eight',
                                  refcount_reload_cycles_per_nonempty_hot_tile=8,
                                  order_build='one visit per active node plus one return per forest edge',
                                  hot_storage='two 1152-bit slots, 2R1W register paths, backing SRAM retained'),
                  caveats=['CPU transplantation of paper tie rule into C1 service/cost model; not an official Prosperity hardware rerun',
                           'Tie policies have identical residual popcounts and synthetic-parent issue counts; changes are topology/lifetime/service only',
                           'Equal-mask prefixes still require p<row; rows with popcount<2 have no prefix in both variants',
                           'Largest-index tie comparator cost is assumed equal to first-index tie; not synthesized',
                           'Numeric checks independently replay integer row/cache values, not cycle payload timing; use the separate payload regression for that',
                           'No sink commit, full-layer/network speedup, energy, area, or novelty claim'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('totals', 'phases')}, indent=2))


if __name__ == '__main__':
    main()
