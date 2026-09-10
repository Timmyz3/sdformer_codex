#!/usr/bin/env python3.12
"""Exact/greedy selective partial admission after M2271; intents, never cycles."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import time
import numpy as np

HW = Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07')
BASE = Path(__file__).resolve().parents[1]
POPS = np.array([i.bit_count() for i in range(16)], dtype=np.int64)
CAPS = (0, 1, 2, 4, 11)
FEES = (0, 1, 2, 4, 8, 16)
SUBSETS = {n: (np.arange(1 << n, dtype=np.int32), np.array([i.bit_count() for i in range(1 << n)], dtype=np.int16)) for n in range(12)}


def direct_cost(codes, selected):
    """Independent literal per-half reference; no subset transform."""
    class_updates = bypass = 0
    for half in codes.reshape(-1, 8):
        class_updates += len(set(int(c) for c in half if int(c) in selected))
        raw = 0
        for c in half:
            if int(c) not in selected:
                raw |= int(c)
        bypass += raw.bit_count()
    scatter = sum(c.bit_count() for c in selected)
    return class_updates + bypass + scatter


def prepare(codes):
    half = codes.reshape(-1, 8)
    present = np.any(half[:, :, None] == np.arange(16)[None, None, :], axis=1)
    hist = np.bincount(codes, minlength=16)
    ordinary = int(POPS[np.bitwise_or.reduce(half, axis=1)].sum())
    old = (hist >= 2) & (POPS >= 2)
    old_raw = np.where(old[half], 0, half).astype(np.uint8)
    old_cost = int(present[:, old].sum() + POPS[old].sum() + POPS[np.bitwise_or.reduce(old_raw, axis=1)].sum())
    occurrences = present.sum(axis=0)
    # Removing a class whose maximum possible bypass saving is <= its charge
    # cannot worsen the optimum; prefer empty/fewer states on all ties.
    eligible = np.flatnonzero(old & ((POPS - 1) * occurrences > POPS))
    n = len(eligible)
    masks, sizes = SUBSETS[n]
    charges = np.zeros(1 << n, dtype=np.int64)
    mapping = np.zeros(16, dtype=np.int32)
    for bit, code in enumerate(eligible):
        mapping[code] = 1 << bit
        k = 1 << bit
        charges[k:2*k] = charges[:k] + int(occurrences[code] + POPS[code])
    frequency = np.zeros(1 << n, dtype=np.int64)
    mapped = mapping[half]
    for d in range(4):
        live = ((half >> d) & 1).astype(bool)
        veto = np.any(live & (mapped == 0), axis=1)
        blockers = np.bitwise_or.reduce(np.where(live, mapped, 0), axis=1)
        valid = ~veto & (blockers != 0)
        frequency += np.bincount(blockers[valid], minlength=1 << n)
    saving = frequency.copy()
    for bit in range(n):
        v = saving.reshape(-1, 2, 1 << bit)
        v[:, 1, :] += v[:, 0, :]
    costs = ordinary + charges - saving
    assert costs[0] == ordinary and np.all(costs >= 0)
    return ordinary, old_cost, eligible, masks, sizes, costs


def select(costs, sizes, cap, fee=0):
    allowed = sizes <= cap
    objective = costs + fee * sizes
    # The first criterion is charged cost; fewer retained partials break ties.
    ranked = objective * 12 + sizes
    chosen = int(np.argmin(np.where(allowed, ranked, np.iinfo(np.int64).max)))
    return chosen, int(objective[chosen])


def greedy(costs, n, cap):
    mask = 0
    for _ in range(min(n, cap)):
        options = [mask | (1 << j) for j in range(n) if not mask >> j & 1]
        if not options:
            break
        best = min(options, key=lambda v: (costs[v], v))
        if costs[best] >= costs[mask]:
            break
        mask = best
    return mask


def check_arithmetic(codes, selected, rng):
    weights = rng.integers(-128, 128, (len(codes), 16), dtype=np.int64)
    if len(codes) >= 2:
        weights[0] = -128
        weights[1] = 127
    x = ((codes[None, :] >> np.arange(4)[:, None]) & 1).astype(np.int64)
    reference = x @ weights
    output = np.zeros((4, 16), dtype=np.int64)
    partials = {s: np.zeros(16, dtype=np.int64) for s in selected}
    # Stream through the source order. Natural source signs are all positive;
    # negative values here are weights and never encoded via a 9-bit bridge.
    for code, w in zip(codes, weights):
        code = int(code)
        if code in partials:
            partials[code] += w
        else:
            for dest in range(4):
                if code >> dest & 1:
                    output[dest] += w
    for code, p in partials.items():
        for dest in range(4):
            if code >> dest & 1:
                output[dest] += p
    assert np.array_equal(reference, output)
    assert np.max(np.abs(output)) < (1 << 23)
    return output.size


def mathematical_checks():
    rng = np.random.default_rng(20260906)
    cases = [np.zeros(64,dtype=np.uint8), np.full(64,15,dtype=np.uint8), np.tile(np.array([3,5,6,7,9,10,12,15],dtype=np.uint8),8)]
    # Independent review supplied a complementary-admission counterexample:
    # either class alone costs 17, both cost 14, ordinary and greedy cost 15.
    complementary = np.tile(np.array([3,5,0,0,0,0,0,0],dtype=np.uint8),5)
    o, _, e, _, sz, co = prepare(complementary)
    assert o == 15 and sorted(co.tolist()) == [14,15,17,17]
    assert co[greedy(co,len(e),2)] == 15 and select(co,sz,2)[1] == 14
    cases.append(complementary)
    cases += [rng.integers(0,16,8*(1+i%16),dtype=np.uint8) for i in range(240)]
    subset_checks = arithmetic = 0
    for codes in cases:
        ordinary, old, eligible, masks, sizes, costs = prepare(codes)
        all_old = set(int(c) for c in np.flatnonzero((np.bincount(codes,minlength=16)>=2)&(POPS>=2)))
        assert old == direct_cost(codes, all_old)
        for mask in masks:
            selected = {int(c) for j,c in enumerate(eligible) if int(mask) >> j & 1}
            assert costs[mask] == direct_cost(codes, selected)
            subset_checks += 1
        for cap in CAPS:
            chosen, cost = select(costs,sizes,cap)
            assert cost <= ordinary
            selected = {int(c) for j,c in enumerate(eligible) if chosen >> j & 1}
            arithmetic += check_arithmetic(codes,selected,rng)
    return {'synthetic_cases':len(cases), 'independent_literal_subset_checks':subset_checks, 'directed_integer_outputs':arithmetic, 'mismatches':0}


def add(totals, key, values, factor):
    totals[key].update({k:int(v)*factor for k,v in values.items()})


def summarize(c):
    r = dict(c)
    if c['ordinary']:
        r['charged_intents_over_ordinary'] = c['charged'] / c['ordinary']
        r['update_intent_reduction_fraction'] = (c['ordinary']-c['charged'])/c['ordinary']
    r['fixed_classifier_headroom_per_input_half'] = (c['ordinary']-c['charged'])/max(1,c['input_halves'])
    r['fixed_classifier_headroom_per_window'] = (c['ordinary']-c['charged'])/max(1,c['windows'])
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit('Refusing to overwrite existing result')
    start = time.monotonic()
    checks = mathematical_checks()
    print('Independent mathematical checks passed',flush=True)
    layers_path = HW/'results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901/layers.json'
    layers = {x['layer_id']:x for x in json.loads(layers_path.read_text())['layers']}
    old_path = HW/'results/m2271_destination_signature_screen_20260905/result.json'
    old_results = json.loads(old_path.read_text())
    weighted = defaultdict(Counter)
    unweighted = defaultdict(Counter)
    by_target_layer = defaultdict(Counter)
    old_check = defaultdict(Counter)
    hist_selected = defaultdict(Counter)
    records = []
    workload_count = natural_checked = 0
    rng = np.random.default_rng(2273)
    for prefix, extent in [('m2051_ep34_tsbg_full40_s1920',48),('m2067_ep34_fc2_exact_continuation_s960',192)]:
        meta_path = HW/'tb_m2018/fixtures'/f'{prefix}.json'
        mem_path = meta_path.with_suffix('.memh')
        meta = json.loads(meta_path.read_text())
        words = np.array([int(v,16)&65535 for v in mem_path.read_text().split()],dtype=np.uint16).reshape(len(meta['rows']),4,extent)
        records.append({'path':str(meta_path),'rows':len(meta['rows']),'fixture_path':str(mem_path),'fixture_sha256':hashlib.sha256(mem_path.read_bytes()).hexdigest()})
        for row in meta['rows']:
            target = row['target'] if extent==48 else 'FC2'
            layer = layers[row['layer_id']]
            assert layer['target']==target and row['negative_codes']==0
            factor = layer['weight_layout']['output_tile_count']
            assert factor*96 == layer['output_channels']
            w = words[row['slot'],:,:int(row['source_groups'])]
            x = ((w[...,None] >> np.arange(16,dtype=np.uint16))&1).reshape(4,-1)
            sig = (x*np.array([1,2,4,8],dtype=np.uint8)[:,None]).sum(0).astype(np.uint8)
            for window in (16,64,768):
                old_key = f'{target}_window{window}'
                for begin in range(0,len(sig),window):
                    codes = sig[begin:begin+window]
                    ordinary, old, eligible, masks, sizes, costs = prepare(codes)
                    add(old_check,old_key,{'ordinary_nonempty_k8_bundles':ordinary,'bank_preserving_total_update_intents':old,'blocks':1},factor)
                    for cap in CAPS:
                        for fee in FEES:
                            chosen, charged = select(costs,sizes,cap,fee)
                            key = f'{old_key}_cap{cap}_fee{fee}'
                            values = {'windows':1,'input_halves':len(codes)//8,'ordinary':ordinary,'charged':charged,'actual_update_intents':costs[chosen],'retained_classes':sizes[chosen], 'improved_windows':int(charged<ordinary)}
                            add(weighted,key,values,factor)
                            add(unweighted,key,values,1)
                            if fee==0:
                                hist_selected[key][int(sizes[chosen])] += factor
                                add(by_target_layer,f'layer{row["layer_id"]}_window{window}_cap{cap}',values,factor)
                                if row['slot']==0:
                                    selected = {int(c) for j,c in enumerate(eligible) if chosen >> j & 1}
                                    natural_checked += check_arithmetic(codes,selected,rng)
                        g = greedy(costs,len(eligible),cap)
                        add(weighted,f'{old_key}_greedy_cap{cap}',{'windows':1,'input_halves':len(codes)//8,'ordinary':ordinary,'charged':costs[g],'actual_update_intents':costs[g],'retained_classes':sizes[g], 'improved_windows':int(costs[g]<ordinary)},factor)
            workload_count += 1
            if workload_count%240==0:
                print(f'processed {workload_count}/2880 descriptors; elapsed {time.monotonic()-start:.1f}s',flush=True)
    assert workload_count==2880
    for key, values in old_check.items():
        expected = old_results['fc_weighted_by_96lane_output_tiles'][key]
        for field,value in values.items():
            assert value==expected[field],(key,field,value,expected[field])
    report = {
        'status':'CPU_SERVICE_INTENT_UPPER_BOUND_AND_GREEDY_SCREEN_NOT_CYCLES',
        'scope':{'workloads':workload_count,'windows':[16,64,768],'caps':list(CAPS),'per_class_fee_update_equivalents':list(FEES),'selection_cost_free_in_oracle':True,'baseline':'M2271 K8-preserving service intents; matched descriptor cohort'},
        'formula':'C(S)=ordinary + sum_s_in_S(half_occurrences_s+fanout_s) - number_of_half_destination_bundles_fully_removed_by_S',
        'weighting':'Weighted totals count the actual number of 96-output logical tiles, as M2271. Each such vector update requires six 16-output slice transfers on the existing bridge; new routing/partial storage is not implemented.',
        'partial_state_payload_bytes':{'physical_16_lane_slice_per_class':48,'logical_96_lane_tile_per_class':288,'existing_four_output_contexts_physical':192,'existing_four_output_contexts_logical':1152,'scope':'48 bytes is only one slice. Holding all six slices needs 288 bytes per class; a 48-byte design requires reordered slice lifetime, not a free reduction of state.'},
        'checks':{**checks,'natural_diagnostic_integer_outputs':natural_checked,'sealed_m2271_baseline_totals':'MATCH_ALL_6_TARGET_WINDOW_POINTS'},
        'fixtures':records,
        'weighted':{k:summarize(v) for k,v in sorted(weighted.items())},
        'unweighted':{k:summarize(v) for k,v in sorted(unweighted.items())},
        'by_layer':{k:summarize(v) for k,v in sorted(by_target_layer.items())},
        'selected_class_histogram_weighted':{k:dict(v) for k,v in sorted(hist_selected.items())},
        'limitations':[
            'This changes admission relative to M2271; it does not execute new RTL or measure latency/energy.',
            'The exact search is an upper bound with free selection. Greedy selection also has unpriced classification and scoring.',
            'Empty selection guarantees no regression only in counted update intents. Classifier/descriptor scan costs remain even when no class is retained.',
            'Per-class fees are illustrative update-equivalent sensitivity; they are not cycles and exclude fixed classification costs.',
            'The current endpoint has no class Acc24 sink or direct Acc24 scatter inlet. It cannot legally pass partials through a 9-bit effective-weight bridge.',
            'No extra weight-read reduction is claimed: group-major TSBG already shares fetches.',
            'The fields omit fetch/RMW latency, arbitration, selector critical path, phase switches, flush, reset/epoch and physical state placement.',
            'Both sides omit common G48 continuation-wrapper reset/cache/retained-output accumulation services. These update totals are not a complete FC execution schedule.',
            'The cohort contains first/middle/last B4 quartets, not all network tokens. Diagnostic signed weights do not prove frozen FP or quantization AEE.',
            'No C1 forest experiment is repeated or replaced; this screen is confined to C2 FC descriptors.'
        ],
        'PPA_ADMISSION':0,'RTL_SPEEDUP_ADMISSION':0,'elapsed_seconds':time.monotonic()-start
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    brief={k:v for k,v in report['weighted'].items() if ('window768_cap4_fee0' in k or 'window768_greedy_cap4' in k or 'window64_cap4_fee0' in k)}
    print(json.dumps({'checks':report['checks'],'selected_results':brief,'elapsed_seconds':report['elapsed_seconds']},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
