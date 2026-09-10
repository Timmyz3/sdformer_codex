"""Read existing whole/H8 CMVM DAGs; do not compile graphs or claim cycles/PPA."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def width(lo, hi):
    bits = 1
    while lo < -(1 << (bits-1)) or hi >= 1 << (bits-1):
        bits += 1
    return bits


def parents(node):
    return () if node['kind'] == 'input' else (node['lhs'], node['rhs'])


def order_for(graph, widths, mode):
    nodes, nin = graph['nodes'], graph['n_input']
    if mode == 'output_group_first':
        result, seen = [], set(range(nin))
        def visit(ident):
            if ident in seen:
                return
            for operand in parents(nodes[ident]):
                visit(operand)
            seen.add(ident)
            result.append(ident)
        for output in graph['outputs']:
            visit(output['node'])
        return result
    uses = Counter(i for n in nodes for i in parents(n))
    children = [set() for _ in nodes]
    pending = {}
    for node in nodes[nin:]:
        pending[node['id']] = len(set(i for i in parents(node) if i >= nin))
        for operand in set(parents(node)):
            children[operand].add(node['id'])
    ready = {i for i, count in pending.items() if count == 0}
    order = []
    while ready:
        def priority(ident):
            consumed = Counter(parents(nodes[ident]))
            freed = sum(widths[i] for i, count in consumed.items() if i >= nin and uses[i] == count)
            allocated = widths[ident] if uses[ident] else 0
            return (freed-allocated, bool(nodes[ident].get('output_consumers')), -ident)
        ident = min(ready) if mode == 'official_order' else max(ready, key=priority)
        ready.remove(ident)
        order.append(ident)
        for operand in parents(nodes[ident]):
            uses[operand] -= 1
        for child in children[ident]:
            pending[child] -= 1
            if pending[child] == 0:
                ready.add(child)
    assert len(order) == len(nodes)-nin
    return order


def account(graph, widths, group_masks, order):
    """One scalar graph, two bypass registers with lazy RF writeback.

    All 96 inputs already exist in their common cache. Outputs drain with
    backpressure stalling further compute; no unbounded output collection.
    These are traffic/lifetime counts for a legal order, not cycle timings.
    """
    nodes, nin = graph['nodes'], graph['n_input']
    uses = Counter(i for ident in order for i in parents(nodes[ident]))
    rf, cache, dirty = set(), [], set()
    live = set()
    result = dict(input_reads=0, temporary_reads=0, temporary_writes=0,
                  input_read_bits=0, temporary_read_bits=0, temporary_write_bits=0,
                  peak_RF_values=0, peak_RF_bits=0, peak_live_values=0,
                  peak_live_bits=0, peak_live_plus_result_bits=0,
                  cross_H8_peak_live_values=0, cross_H8_peak_live_bits=0)
    created, last_used = {}, {}
    def rf_peak():
        result['peak_RF_values'] = max(result['peak_RF_values'], len(rf))
        result['peak_RF_bits'] = max(result['peak_RF_bits'], sum(widths[i] for i in rf))
    def cache_add(ident):
        if ident in cache:
            cache.remove(ident)
        else:
            if len(cache) == 2:
                dead = [i for i in cache if i >= nin and uses[i] == 0]
                victim = dead[0] if dead else cache[0]
                cache.remove(victim)
                if victim in dirty:
                    dirty.remove(victim)
                    if uses[victim]:
                        rf.add(victim)
                        result['temporary_writes'] += 1
                        result['temporary_write_bits'] += widths[victim]
                        rf_peak()
        cache.append(ident)
    for step, ident in enumerate(order):
        operands = parents(nodes[ident])
        for operand in dict.fromkeys(operands):
            if operand not in cache:
                if operand < nin:
                    result['input_reads'] += 1
                    result['input_read_bits'] += widths[operand]
                else:
                    assert operand in rf
                    result['temporary_reads'] += 1
                    result['temporary_read_bits'] += widths[operand]
                cache_add(operand)
            else:
                cache.remove(operand)
                cache.append(operand)
            last_used[operand] = step
        # Result exists before its old operands are released. This includes
        # a result register, not a mandatory new SRAM word at every node.
        result['peak_live_plus_result_bits'] = max(result['peak_live_plus_result_bits'],
            sum(widths[i] for i in live)+widths[ident])
        for operand in operands:
            uses[operand] -= 1
            if uses[operand] == 0:
                live.discard(operand)
                rf.discard(operand)
                dirty.discard(operand)
        created[ident] = step
        if uses[ident]:
            live.add(ident)
            cache_add(ident)
            dirty.add(ident)
        result['peak_live_values'] = max(result['peak_live_values'], len(live))
        result['peak_live_bits'] = max(result['peak_live_bits'], sum(widths[i] for i in live))
        cross = [i for i in live if group_masks[i].bit_count() > 1]
        result['cross_H8_peak_live_values'] = max(result['cross_H8_peak_live_values'], len(cross))
        result['cross_H8_peak_live_bits'] = max(result['cross_H8_peak_live_bits'], sum(widths[i] for i in cross))
    assert not live and not rf
    result['RF_fixed_max_width_storage_upper_bits'] = result['peak_RF_values']*max(widths)
    result['two_bypass_register_bits_upper'] = 2*max(widths)
    result['one_result_register_bits_upper'] = max(widths)
    spans = [(last_used.get(i, created[i])-created[i], i) for i in created if group_masks[i].bit_count() > 1]
    result['largest_cross_H8_lifespans_in_issued_nodes'] = [
        dict(node=i, span=span, width=widths[i], output_groups=group_masks[i].bit_count())
        for span, i in sorted(spans, reverse=True)[:8]]
    return result


def audit(path, expected_matrix, xlo, xhi, global_offset=0):
    graph = json.loads(path.read_text())
    nodes, nin = graph['nodes'], graph['n_input']
    assert nin == 96
    widths, mismatches, interval_mismatches = [], [], []
    recurrence_mismatches, declared_interval_failures = [], []
    group_masks = [0]*len(nodes)
    for node in nodes:
        coeff = np.asarray(node['coeff'], np.int64)
        lo = int(np.maximum(coeff, 0).sum())*xlo+int(np.minimum(coeff, 0).sum())*xhi
        hi = int(np.maximum(coeff, 0).sum())*xhi+int(np.minimum(coeff, 0).sum())*xlo
        bits = width(lo, hi)
        widths.append(bits)
        if (lo, hi) != (node['static_min'], node['static_max']):
            interval_mismatches.append(dict(node=node['id'], independent=[lo, hi],
                exported=[node['static_min'], node['static_max']]))
        if bits != node['signed_bits']:
            mismatches.append(dict(node=node['id'], independent=bits, exported=node['signed_bits']))
        if node['kind'] != 'input':
            left = np.asarray(nodes[node['lhs']]['coeff'], np.int64) << node['lhs_shift']
            right = np.asarray(nodes[node['rhs']]['coeff'], np.int64) << node['rhs_shift']
            calculated = left-right if node['subtract'] else left+right
            if not np.array_equal(calculated, coeff):
                recurrence_mismatches.append(node['id'])
        stored_lo, stored_hi = (lo, hi) if node['DAIS_stored_sign'] == 1 else (-hi, -lo)
        declared_lo, declared_hi = node['DAIS_declared_integer_interval']
        if declared_lo > stored_lo or declared_hi < stored_hi:
            declared_interval_failures.append(node['id'])
    output_widths = []
    for index, output in enumerate(graph['outputs']):
        coeff = np.asarray(nodes[output['node']]['coeff'], np.int64)*int(output['sign'])
        shift = int(output['shift'])
        if shift >= 0:
            coeff = coeff << shift
        else:
            assert np.all(coeff % (1 << -shift) == 0)
            coeff = coeff // (1 << -shift)
        assert np.array_equal(coeff, expected_matrix[index])
        group_masks[output['node']] |= 1 << ((global_offset+index)//8)
        lo = int(np.maximum(coeff, 0).sum())*xlo+int(np.minimum(coeff, 0).sum())*xhi
        hi = int(np.maximum(coeff, 0).sum())*xhi+int(np.minimum(coeff, 0).sum())*xlo
        output_widths.append(width(lo, hi))
    for node in reversed(nodes[nin:]):
        for operand in parents(node):
            group_masks[operand] |= group_masks[node['id']]
    fanouts = Counter(i for node in nodes for i in parents(node))
    cross = [n['id'] for n in nodes[nin:] if group_masks[n['id']].bit_count() > 1]
    result = dict(file=str(path), nodes=len(nodes)-nin, inputs=nin, outputs=len(graph['outputs']),
        symbolic_output_matches_Wq=True, width_mismatches=mismatches,
        interval_mismatches=interval_mismatches,
        operation_recurrence_mismatches=recurrence_mismatches,
        DAIS_sign_corrected_interval_failures=declared_interval_failures,
        strict_input_bits=sorted(set(widths[:nin])), strict_output_bits=sorted(set(output_widths)),
        arithmetic_node_width_histogram=dict(sorted(Counter(widths[nin:]).items())),
        cross_H8_arithmetic_nodes=len(cross), cross_H8_node_bit_sum=sum(widths[i] for i in cross),
        cross_H8_width_histogram=dict(sorted(Counter(widths[i] for i in cross).items())),
        cross_H8_final_group_count_histogram=dict(sorted(Counter(group_masks[i].bit_count() for i in cross).items())),
        input_peak_fanout_edges=max(fanouts[i] for i in range(nin)),
        arithmetic_peak_fanout_edges=max(fanouts[i] for i in range(nin, len(nodes))),
        add_dependency_depth=max(n['depth'] for n in nodes),
        expanded_arithmetic_result_bit_sum=sum(widths[nin:]),
        expanded_bit_sum_scope='Sum of node result widths, not adder area, RF size, placed routing or pipeline registers.',
        orders={})
    for mode in ('official_order', 'pressure_first', 'output_group_first'):
        order = order_for(graph, widths, mode)
        result['orders'][mode] = account(graph, widths, group_masks, order)
    return result


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=here)
    args = parser.parse_args()
    p = np.load(args.directory.parent/'projection_control_parameters.npz')
    matrix = p['Wq'].astype(np.int64)
    xlo, xhi = [int(i) for i in p['input_integer_range']]
    whole = audit(args.directory/'whole_integer_dag.json', matrix, xlo, xhi)
    local = [audit(args.directory/f'h8_{i:02d}_integer_dag.json', matrix[i*8:(i+1)*8], xlo, xhi, i*8) for i in range(12)]
    comparison = {}
    for mode in whole['orders']:
        records = [g['orders'][mode] for g in local]
        comparison[mode] = dict(
            independent_H8_sum_input_reads=sum(r['input_reads'] for r in records),
            independent_H8_sum_temporary_reads=sum(r['temporary_reads'] for r in records),
            independent_H8_sum_temporary_writes=sum(r['temporary_writes'] for r in records),
            independent_H8_max_live_bits=max(r['peak_live_bits'] for r in records),
            independent_H8_max_RF_values=max(r['peak_RF_values'] for r in records),
            independent_H8_max_RF_bits=max(r['peak_RF_bits'] for r in records))
    result = dict(scope='One (p,t) CMVM instance, all96 input channels. Independent H8 graphs run serially with one common 96x12 input cache; scratch maximum, not sum. No pipeline timing, EDA, GPU, new graph compilation or PPA.',
        common_input_cache_bits=96*12,
        ordinary_H8_MAC_accumulator_bits=8*25,
        scheduler='Three fixed legal graph orders; two bypass registers, lazy RF writeback, immutable input cache. Final outputs stream through one result register and backpressure stalls compute. No optimality claim or whole96-output/Y storage requirement.',
        full=whole, independent_H8=local,
        sum_independent_H8_nodes=sum(g['nodes'] for g in local),
        independent_H8_folded_comparison=comparison)
    (args.directory/'audit_physical_gap.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
