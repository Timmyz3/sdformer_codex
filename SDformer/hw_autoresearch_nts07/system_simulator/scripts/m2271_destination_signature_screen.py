#!/usr/bin/env python3
"""CPU arithmetic-graph screen of B4 common destination signatures.

Mailman/common-subexpression elimination is prior art. This measures an
opportunity, NOT cycles, traffic, energy, area, or a new sparsity definition.
Inputs are existing ep34 descriptors. Values used for equality are diagnostic
signed INT8, not an accuracy-bound network weight export.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import m2259_c1_forest_lifetime_probe as c1

HW = Path(__file__).resolve().parents[2]
POP = np.array([i.bit_count() for i in range(16)], dtype=np.int64)


def bits(words):
    return ((words[..., None] >> np.arange(16, dtype=np.uint16)) & 1).reshape(4, -1)


def analyze(x, verify=False):
    """Compare binary additions; first assignment is free on BOTH sides.

    A signature is the 4-bit set of outputs using one source weight. Build a
    shared partial only for signature fanout >=2 and >=2 member sources.
    Distinct signature partials have disjoint source sets; outputs may overlap.
    """
    signatures = (x.astype(np.int64) * np.array([1, 2, 4, 8])[:, None]).sum(0)
    hist = np.bincount(signatures, minlength=16)
    share = (hist >= 2) & (POP >= 2)
    reductions = int((hist[share]-1).sum())
    output_operands = x.sum(1, dtype=np.int64)
    plain = int(np.maximum(output_operands-1, 0).sum())
    for m in np.flatnonzero(share):
        output_operands -= ((m >> np.arange(4)) & 1) * (hist[m]-1)
    grouped = reductions + int(np.maximum(output_operands-1, 0).sum())
    assert grouped <= plain
    out = Counter(blocks=1, input_contributions=int(x.sum()),
                  distinct_live_sources=int((signatures != 0).sum()),
                  plain_binary_adds=plain, shared_binary_adds=grouped,
                  saved_binary_adds=plain-grouped,
                  shared_groups=int(share.sum()),
                  shareable_source_columns=int(hist[share].sum()),
                  multi_destination_source_columns=int(hist[POP >= 2].sum()))
    # K8's real parallel reduction is not a scalar serial chain. Report a
    # separate arithmetic-stage lower bound with free packing/ports; never use
    # it as measured candidate issue cycles.
    direct_words = x.reshape(4, -1, 8).sum(2)
    out['ordinary_nonempty_k8_bundles'] = int((direct_words > 0).sum())
    out['ideal_shared_stage_issues'] = int(
        ((hist[share]+7)//8).sum() + ((output_operands+7)//8).sum())
    # A stricter service-intent screen preserves the original eight-source
    # bank group. No free arbitrary gather of eight members of one signature.
    # Shared partials are accumulated as groups arrive. Nonshared sources can
    # still use the ordinary destination K8 path. At close, each shared vector
    # is added separately to each consumer (no free wide Acc24 compressor).
    class_updates = 0
    bypass_updates = 0
    for start in range(0, x.shape[1], 8):
        codes = signatures[start:start+8]
        class_updates += int(share[np.unique(codes)].sum())
        remaining = x[:,start:start+8] & (~share[codes])[None,:]
        bypass_updates += int(remaining.any(1).sum())
    scatters = int(POP[share].sum())
    out['bank_preserving_class_update_intents'] = class_updates
    out['bank_preserving_bypass_update_intents'] = bypass_updates
    out['shared_vector_scatter_intents'] = scatters
    out['bank_preserving_total_update_intents'] = class_updates+bypass_updates+scatters
    if verify:
        rng = np.random.default_rng(2271+x.shape[1])
        weights = rng.integers(-128, 128, (x.shape[1], 96), dtype=np.int64)
        reference = x.astype(np.int64) @ weights
        operands = [[] for _ in range(4)]
        counted = 0
        for m in range(1, 16):
            members = np.flatnonzero(signatures == m)
            if not len(members):
                continue
            if share[m]:
                partial = weights[members[0]].copy()
                for member in members[1:]:
                    partial += weights[member]
                    counted += 1
                for dest in range(4):
                    if m >> dest & 1:
                        operands[dest].append(partial)
            else:
                for dest in range(4):
                    if m >> dest & 1:
                        operands[dest].extend(weights[members])
        actual = np.zeros_like(reference)
        for dest in range(4):
            if operands[dest]:
                actual[dest] = operands[dest][0]
                for operand in operands[dest][1:]:
                    actual[dest] += operand
                    counted += 1
        assert np.array_equal(reference, actual)
        assert counted == grouped
        out['checked_int_outputs'] = actual.size
    return out


def finish(c):
    r = dict(c)
    r['binary_add_reduction_fraction'] = c['saved_binary_adds']/max(1,c['plain_binary_adds'])
    r['ordinary_over_shared_arithmetic_count'] = c['plain_binary_adds']/max(1,c['shared_binary_adds'])
    r['bank_preserving_over_ordinary_update_intents'] = c['bank_preserving_total_update_intents']/max(1,c['ordinary_nonempty_k8_bundles'])
    carry = c['cross_window_output_adds']
    r['plain_dot_binary_adds'] = c['plain_binary_adds']+carry
    r['shared_dot_binary_adds'] = c['shared_binary_adds']+carry
    r['dot_binary_add_reduction_fraction'] = c['saved_binary_adds']/max(1,r['plain_dot_binary_adds'])
    r['ordinary_over_shared_dot_arithmetic_count'] = r['plain_dot_binary_adds']/max(1,r['shared_dot_binary_adds'])
    if 'existing_64row_forest_binary_adds' in c:
        r['existing_forest_dot_binary_adds'] = c['existing_64row_forest_binary_adds']+carry
        r['shared_over_existing_forest_dot_arithmetic_count'] = r['shared_dot_binary_adds']/max(1,r['existing_forest_dot_binary_adds'])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    fixtures = HW / 'tb_m2018/fixtures'
    capture = HW / 'results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901'
    layer_by_id = {r['layer_id']: r for r in json.loads((capture/'layers.json').read_text())['layers']}
    totals = {}; by_sequence = {}; weighted = {}; workloads = 0
    for prefix, extent in (('m2051_ep34_tsbg_full40_s1920',48),
                           ('m2067_ep34_fc2_exact_continuation_s960',192)):
        meta = json.loads((fixtures/(prefix+'.json')).read_text())
        words = np.array([int(v,16)&65535 for v in (fixtures/(prefix+'.memh')).read_text().split()],dtype=np.uint16)
        words = words.reshape(len(meta['rows']),4,extent)
        for row in meta['rows']:
            target = row['target'] if extent == 48 else 'FC2'
            layer = layer_by_id[row['layer_id']]
            assert layer['target'] == target
            output_tiles = layer['weight_layout']['output_tile_count']
            assert output_tiles*96 == layer['output_channels']
            assert row['negative_codes'] == 0
            n = int(row['source_groups'])
            x = bits(words[row['slot'],:,:n])
            workloads += 1
            for window in (16,64,768):
                key = f"{target}_window{window}"
                count = Counter()
                live_windows = np.zeros(4, dtype=np.int64)
                for begin in range(0,x.shape[1],window):
                    tile = x[:,begin:begin+window]
                    # The real dimensions are all multiples of 16.
                    result = analyze(tile, verify=row['slot']<2)
                    count.update(result)
                    live_windows += tile.any(1)
                count['cross_window_output_adds'] = int(np.maximum(live_windows-1,0).sum())
                assert count['plain_binary_adds']+count['cross_window_output_adds'] == int(np.maximum(x.sum(1,dtype=np.int64)-1,0).sum())
                totals.setdefault(key,Counter()).update(count)
                by_sequence.setdefault(row['sequence']+'/'+key,Counter()).update(count)
                weighted.setdefault(key,Counter()).update({k:v*output_tiles for k,v in count.items() if k != 'checked_int_outputs'})
    # Native C1 spatial rows: same preset cohort as M2266, three samples,
    # four Conv layers, three spatial chunks, ALL 432 K partitions.
    conv = Counter()
    with c1.LEDGER.open('rb') as stream:
        for sample in range(3):
            for op in range(4):
                for chunk in (0,23,46):
                    n = min(64,3000-chunk*64)
                    active_partitions = np.zeros(n, dtype=np.int64)
                    for k in range(432):
                        phase = (sample*4+op)*432+k
                        stream.seek((phase*3000+chunk*64)*9)
                        masks = np.array([int(v,16)&65535 for v in stream.read(n*9).splitlines()],dtype=np.uint16)
                        active_partitions += masks != 0
                        residual, parent = c1.old.M504.cleanroom_subset(masks)
                        forest_adds = sum(int(residual[i]).bit_count() if parent[i]>=0 else max(int(residual[i]).bit_count()-1,0) for i in range(n))
                        count = Counter(existing_64row_forest_binary_adds=forest_adds,forest_tiles=1)
                        for begin in range(0,n,4):
                            group = np.pad(masks[begin:begin+4],(0,max(0,begin+4-n)))
                            count.update(analyze(bits(group[:,None]),verify=sample==op==chunk==k==0))
                        conv.update(count)
                    conv['cross_window_output_adds'] += int(np.maximum(active_partitions-1,0).sum())
    report = dict(
        scope='Arithmetic operation-count opportunity only; NOT workload speedup.',
        fc_workloads=workloads, fc= {k:finish(v) for k,v in totals.items()},
        fc_weighted_by_96lane_output_tiles={k:finish(v) for k,v in weighted.items()},
        fc_by_sequence={k:finish(v) for k,v in by_sequence.items()}, conv=finish(conv),
        bounds_and_costs=[
            'Compared to nonzero arithmetic with free first assignment, not a dense zero baseline.',
            'Local-window and full-dot arithmetic counts are separate; full-dot counts include cross-window/K-partition accumulation on both sides.',
            'FC is 40 samples x 24 layers x first/middle/last B4 quartets, not every token in the network.',
            'fc is descriptor-template weighted; fc_weighted_by_96lane_output_tiles additionally covers actual output-channel geometry, not population token-count extrapolation.',
            'TSBG already shares weight fetch; signature grouping claims NO additional weight-read reduction.',
            'ideal_shared_stage_issues assumes free packing and ports; not comparable as timed K8 execution.',
            'bank_preserving intents preserve eight-source groups and charge separate shared-vector scatter; still NOT cycles: classification, RAM read/write latency and contention, vector width, pipeline and flush costs are unpriced.',
            'A full B4 signature table has 15 vectors = 4320 B at 96 lanes x Acc24, vs 1152 B for four outputs; reuse/aliasing not assumed.',
            'Classifier, grouping, table ports, producer scheduling, scatter and overflow still require implementation.',
            'Integer reassociation is exact only with adequate intermediate range and matching overflow/rounding semantics.',
            'FC windows use causal arrived descriptors, not learned patterns selected on validation data.',
            'C1 comparison is against its 64-row parent forest, not merely four-row zero skip.',
            'Diagnostic weights prove the integer graph identity, not frozen-model quantization or AEE.'
        ])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('fc_by_sequence','bounds_and_costs')},indent=2))


if __name__=='__main__':
    main()
