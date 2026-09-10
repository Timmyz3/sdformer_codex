"""Mask-only, at-most-two-live-F source word opportunity. No cycle model."""
from pathlib import Path
import json
from collections import Counter
from functools import reduce
import numpy as np
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
P, C, BLOCKS, WORDS_PER_BLOCK = 4, 384, 24, 3


def bits(x):
    return x.bit_count()


def fixed_pairs(masks):
    graph = nx.Graph()
    graph.add_nodes_from(range(len(masks)))
    for a in range(len(masks)):
        for b in range(a + 1, len(masks)):
            graph.add_edge(a, b, weight=bits(masks[a] & masks[b]))
    pairs = sorted(tuple(sorted(p)) for p in
                   nx.max_weight_matching(graph, maxcardinality=True))
    used = {i for p in pairs for i in p}
    batches = pairs + [(i,) for i in range(len(masks)) if i not in used]
    reads = sum(bits(masks[p[0]] | (masks[p[1]] if len(p) == 2 else 0))
                for p in batches)
    return batches, reads


def interval(masks, order):
    position = {c: i for i, c in enumerate(order)}
    ranges = []
    occupancy = np.zeros(BLOCKS, dtype=np.int64)
    for m in masks:
        loc = [position[c] for c in range(BLOCKS) if m >> c & 1]
        lo, hi = min(loc), max(loc)
        ranges.append([lo, hi])
        occupancy[lo:hi + 1] += 1
    return dict(peak=int(occupancy.max()), ranges=ranges,
                span_sum=sum(hi - lo + 1 for lo, hi in ranges))


def greedy_column_order(masks):
    """Ordinary static frontier heuristic; reorder whole C16 addresses only."""
    remaining = masks.copy()
    started = [False] * len(masks)
    order = []
    for _ in range(BLOCKS):
        choices = []
        for c in set(range(BLOCKS)) - set(order):
            flag = 1 << c
            touched = [i for i, m in enumerate(remaining) if m & flag]
            before = sum(started[i] and remaining[i] != 0 for i in range(len(masks)))
            opens = sum(not started[i] for i in touched)
            closes = sum(remaining[i] == flag for i in touched)
            choices.append(((before + opens - closes, -closes, opens, c), c))
        _, c = min(choices)
        flag = 1 << c
        for i in range(len(masks)):
            if remaining[i] & flag:
                remaining[i] ^= flag
                started[i] = True
        order.append(c)
    return order


def rolling(masks, order, admission):
    """Repeated C sweeps; current block serves only the two admitted groups.

    A group receives every required block once, runs its full consumer and then
    releases S. Consumer latency is unpriced. A replacement cannot reuse the
    just-consumed C16 block for free: its next visit causes a new bank read.
    """
    pending = set(range(len(masks)))
    active = {}
    intervals = {}
    reads = 0
    dependencies = 0
    per_block = Counter()
    peak = 0
    scans = 0

    def admit():
        nonlocal peak
        while pending and len(active) < 2:
            if admission == 'native' or not active:
                i = min(pending)
            else:
                union = 0
                for m in active.values():
                    union |= m
                i = min(pending, key=lambda j: (-bits(masks[j] & union), j))
            pending.remove(i)
            active[i] = masks[i]
            intervals[i] = [reads, None]
            peak = max(peak, len(active))

    admit()
    while active:
        scans += 1
        for c in order:
            flag = 1 << c
            recipients = [i for i, m in active.items() if m & flag]
            if not recipients:
                continue
            reads += 1
            per_block[c] += 1
            dependencies += len(recipients)
            assert len(recipients) <= 2
            for i in recipients:
                active[i] ^= flag
            completed = [i for i, m in active.items() if m == 0]
            for i in completed:
                del active[i]
                intervals[i][1] = reads
            admit()
        assert scans <= len(masks) * 2
    assert dependencies == sum(map(bits, masks))
    assert all(end is not None for _, end in intervals.values())
    return dict(block_reads=reads, words=reads * WORDS_PER_BLOCK,
                scans=scans, live_peak=peak, dependencies=dependencies,
                per_block_reads=[per_block[c] for c in range(BLOCKS)],
                lifetime_read_steps=[intervals[i] for i in range(len(masks))])


def analyze(masks, cache_f):
    stripes = []
    for first in range(0, len(masks), cache_f):
        mm = masks[first:first + cache_f]
        serial = sum(map(bits, mm))
        pairs, pair_reads = fixed_pairs(mm)
        order = greedy_column_order(mm)
        normal = interval(mm, list(range(BLOCKS)))
        reordered = interval(mm, order)
        native = rolling(mm, list(range(BLOCKS)), 'native')
        ordinary = rolling(mm, order, 'overlap')
        disjoint = sum(mm[a] & mm[b] == 0 for a in range(len(mm))
                       for b in range(a + 1, len(mm)))
        per_c = [sum(m >> c & 1 for m in mm) for c in range(BLOCKS)]
        # Conditional lower bound for a visit that serves <= two current F.
        relaxed = sum((count + 1) // 2 for count in per_c)
        best = min(pair_reads, native['block_reads'], ordinary['block_reads'])
        stripes.append(dict(
            first_h=first * 8, groups=len(mm), dependency_occurrences=serial,
            serial_words=serial * 3, union_words=bits(reduce(int.__or__, mm, 0)) * 3,
            disjoint_group_pairs=disjoint,
            all_order_peak_equals_groups_if_pairwise_intersect=(disjoint == 0),
            native_interval=normal, ordinary_interval=reordered,
            ordinary_C16_order=order,
            fixed_best_pairing_words=pair_reads * 3,
            fixed_best_pairs=[[first + i for i in p] for p in pairs],
            rolling_native=native, rolling_ordinary=ordinary,
            best_of_tested_words=best * 3,
            relaxed_two_recipients_words=relaxed * 3,
            one_visit_per_block_pass_count_lower_bound=max(
                (serial + 2 * BLOCKS - 1) // (2 * BLOCKS),
                max((count + 1) // 2 for count in per_c)),
        ))
    summary = dict(
        F_cache=cache_f, stripes=len(stripes),
        serial_words_per_P4=sum(x['serial_words'] for x in stripes),
        fixed_best_pairing_words_per_P4=sum(x['fixed_best_pairing_words'] for x in stripes),
        rolling_native_words_per_P4=sum(x['rolling_native']['words'] for x in stripes),
        rolling_ordinary_words_per_P4=sum(x['rolling_ordinary']['words'] for x in stripes),
        best_tested_words_per_P4=sum(x['best_of_tested_words'] for x in stripes),
        relaxed_two_recipients_words_per_P4=sum(x['relaxed_two_recipients_words'] for x in stripes),
        native_interval_peaks=[x['native_interval']['peak'] for x in stripes],
        ordinary_interval_peaks=[x['ordinary_interval']['peak'] for x in stripes],
        disjoint_group_pairs=sum(x['disjoint_group_pairs'] for x in stripes),
        observed_live_peak=max(x['rolling_ordinary']['live_peak'] for x in stripes),
        source_DRAM_bytes_full_frame=1200 * 384 * 3 // 8 * len(stripes),
    )
    summary['extra_saved_over_best_fixed_pairs'] = (
        1 - summary['best_tested_words_per_P4'] / summary['fixed_best_pairing_words_per_P4'])
    for key in ('serial', 'fixed_best_pairing', 'rolling_native', 'rolling_ordinary', 'best_tested'):
        count = summary[key + '_words_per_P4']
        summary[key + '_words_full_frame'] = count * (1200 // P)
        summary[key + '_repeated_words_full_frame'] = (
            count - sum(x['union_words'] for x in stripes)) * (1200 // P)
    return dict(summary=summary, stripes=stripes)


def main():
    result = dict(
        kind='STATIC_WEIGHT_DEPENDENCY_AND_SOURCE_WORD_OPPORTUNITY_NOT_CYCLES',
        scope='s2b3 complete H/C mask, eight-tile union per k-ID, P4; no source-value zero skipping',
        source_word_bits=64, C16_source_bits=P * 16 * 3,
        source_words_per_C16=3, P4_groups_per_frame=300,
        source_bank_bytes=1024, resident_source_bytes=576,
        resources=dict(PEs=64, S_words_per_PE=56, S_bits=15,
                       class_R=7, time_R=6, class_two_live_S_words=56,
                       time_two_live_S_words=48,
                       both_max_live_at_P4=2, both_max_live_at_P8=1,
                       threshold_register='one T10 row per tile; consumer reload remains',
                       W_ports_per_tile=2, source_read_ports_per_ID=1),
        limitations=[
            'No W/S operation reduction credited; full T10 consumers remain; no FC2 or AEE claim',
            'Consumer service, output backpressure, context tags, ordering metadata and arbitration unpriced',
            'A read broadcasts to all eight tiles; next block waits for both admitted F groups',
            'Common source decode may fan to two F groups; extra destinations and W requests are not free',
            'Relaxed bound assumes <= two admitted groups per C16 visit; not a universal cycle lower bound',
            'C16 reordering changes bank addresses, not source storage layout; no free channel repacking',
            'Source DRAM is unchanged by two-live scheduling at fixed F_cache',
            'Current resident implementation has F_live=1; this is an unimplemented opportunity probe',
        ], modes={})
    specs = [('original', 'original_trained', 19),
             ('block16', 'broadcast8_C16_half_trained', 36),
             ('hidden', 'hidden_H_half_trained', 19)]
    for name, filename, actual_f in specs:
        path = ROOT / 'algorithm/pruning_probe' / (filename + '.npz')
        with np.load(path) as z:
            W = z['weight_int8'][z['hidden_keep']]
        dep = (W.reshape(-1, 8, C) != 0).any(axis=1)
        blockdep = dep.reshape(-1, BLOCKS, 16).any(2)
        # These three actual union masks retain complete C16, so three words is exact.
        assert np.array_equal(dep, np.repeat(blockdep, 16, axis=1))
        masks = [sum(int(v) << c for c, v in enumerate(row)) for row in blockdep]
        actual = analyze(masks, actual_f)
        current = json.loads((ROOT / 'psn' / f'pruning_{name}_resident10.json').read_text())
        reference = next(r for r in current['records'] if r['block'] == 3 and r['route'] == 'class')
        assert actual['summary']['serial_words_full_frame'] == reference['source_bank_word_reads']
        assert actual['summary']['source_DRAM_bytes_full_frame'] == reference['source_bytes']
        result['modes'][name] = dict(
            mask_file=str(path), H=len(W), H8_groups=len(masks),
            required_C16_per_group=sorted(set(map(bits, masks))),
            weight_nonzeros=int(np.count_nonzero(W)),
            full_T10_gate_outputs=1200 * len(W) * 10,
            current_W_peak_bytes=reference['weight_peak'],
            common_F_cache16=analyze(masks, 16), actual_F_cache=actual)
    # Keep one inspectable interval example and compact all complete-stripe counts.
    fields = ['first_h', 'groups', 'dependency_occurrences', 'serial_words',
              'fixed_best_pairing_words', 'rolling_native_words',
              'rolling_ordinary_words', 'best_of_tested_words',
              'relaxed_two_recipients_words', 'pass_count_lower_bound']
    for name, mode in result['modes'].items():
        for point in ('common_F_cache16', 'actual_F_cache'):
            item = mode[point]
            stripes = item['stripes']
            if name == 'block16' and point == 'actual_F_cache':
                x = stripes[0]
                item['first_stripe_interval_example'] = dict(
                    first_h=x['first_h'], C16_order=x['ordinary_C16_order'],
                    native_intervals=x['native_interval']['ranges'],
                    reordered_intervals=x['ordinary_interval']['ranges'],
                    fixed_best_pairs=x['fixed_best_pairs'],
                    rolling_reordered_intervals_in_read_steps=x['rolling_ordinary']['lifetime_read_steps'])
            item['stripe_fields'] = fields
            item['stripes'] = [[
                x['first_h'], x['groups'], x['dependency_occurrences'],
                x['serial_words'], x['fixed_best_pairing_words'],
                x['rolling_native']['words'], x['rolling_ordinary']['words'],
                x['best_of_tested_words'], x['relaxed_two_recipients_words'],
                x['one_visit_per_block_pass_count_lower_bound'],
            ] for x in stripes]
    out = ROOT / 'psn/pruning_two_live_probe.json'
    out.write_text(json.dumps(result, indent=2) + '\n')
    for name, mode in result['modes'].items():
        for point in ('common_F_cache16', 'actual_F_cache'):
            print(name, point, json.dumps(mode[point]['summary']))
    print(out)


if __name__ == '__main__':
    main()
