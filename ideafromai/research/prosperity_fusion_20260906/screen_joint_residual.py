"""Prosperity parent-choice/residual-sharing screen. This is not a cycle model."""
from collections import Counter
from pathlib import Path
import hashlib
import importlib.util
import itertools
import json
import sys
import numpy as np

sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
HW = Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07')
LEDGER = HW / 'results/m1590_ep34_c1_same_ledger_cycle_model_r1_20260901/ep34_c1_support16_rows.memh'
LEDGER_SHA = 'daa6265115df9c0bae5d96e5a133a4b5fbc9786de75598e53ab2e5812bfdb835'
RELATION = HW / 'system_simulator/scripts/analyze_m504_h67_single_port_parent_scratch.py'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def parent_options(masks, rule):
    options = []
    for i, mask in enumerate(masks):
        if mask.bit_count() < 2:
            options.append([-1])
            continue
        candidates = [j for j, p in enumerate(masks)
                      if p and p & mask == p and (p != mask or j < i)]
        if not candidates:
            options.append([-1])
            continue
        size = max(masks[j].bit_count() for j in candidates)
        maximum = [j for j in candidates if masks[j].bit_count() == size]
        options.append(sorted(maximum, reverse=rule == 'paper_max_index'))
    return options


def submasks(mask):
    value = mask
    while value:
        if value.bit_count() > 1:
            yield value
        value = (value - 1) & mask


def dictionary_candidates(masks, options, subset):
    # Identical information is supplied to fixed and joint parent policies.
    possible = Counter()
    for mask, choices in zip(masks, options):
        residuals = {mask ^ (masks[p] if p >= 0 else 0) for p in choices}
        values = set()
        for residual in residuals:
            values.update(submasks(residual) if subset else
                          ([residual] if residual.bit_count() > 1 else []))
        possible.update(values)
    return sorted(d for d, count in possible.items() if count > 1)


def decide(masks, options, dictionary, joint, subset):
    decisions = []
    for mask, all_choices in zip(masks, options):
        choices = all_choices if joint else all_choices[:1]
        best = None
        for p in choices:
            residual = mask ^ (masks[p] if p >= 0 else 0)
            raw_adds = max(0, residual.bit_count() + int(p >= 0) - 1)
            candidates = [(raw_adds, p, residual, 0)]
            for d in dictionary:
                if (d & residual == d) if subset else (d == residual):
                    left = residual ^ d
                    adds = max(0, left.bit_count() + 1 + int(p >= 0) - 1)
                    candidates.append((adds, p, residual, d))
            for candidate in candidates:
                # Equal arithmetic costs preserve the original parent rule.
                if best is None or candidate[0] < best[0]:
                    best = candidate
        decisions.append(best)
    return decisions


def cost(decisions):
    used = {d for _, _, _, d in decisions if d}
    return sum(a for a, _, _, _ in decisions) + sum(d.bit_count() - 1 for d in used)


def choose(masks, options, candidates, capacity, joint, subset):
    chosen = []
    decisions = decide(masks, options, chosen, joint, subset)
    evaluations = 1
    for _ in range(capacity):
        best_cost, best_decisions, best_dict = cost(decisions), decisions, chosen
        for d in candidates:
            if d in chosen:
                continue
            proposal = decide(masks, options, sorted(chosen + [d]), joint, subset)
            evaluations += 1
            value = cost(proposal)
            if value < best_cost:
                best_cost, best_decisions = value, proposal
                best_dict = sorted({v[3] for v in proposal if v[3]})
        if best_cost >= cost(decisions):
            break
        chosen, decisions = best_dict, best_decisions
    return decisions, chosen, evaluations


def numeric_and_account(masks, decisions):
    # Eight diagnostic lanes; source-dependent rational theta is retained exactly.
    # All numerators share 10^6. This is not a new ep34 integer deployment.
    phi = np.array([[(999883 + 17*c) * (((c+3)*(h+5)*13) % 256 - 128)
                     for h in range(8)] for c in range(16)], dtype=np.int64)
    def value(mask):
        indices = [c for c in range(16) if mask >> c & 1]
        return phi[indices].sum(0) if indices else np.zeros(8, dtype=np.int64)
    used = sorted({d for _, _, _, d in decisions if d})
    cached = {d: value(d) for d in used}
    output = {}
    order = sorted(range(len(masks)), key=lambda i: (masks[i].bit_count(), i))
    consumers = Counter(p for _, p, _, _ in decisions if p >= 0)
    live = set()
    high_water = 0
    for i in order:
        _, p, r, d = decisions[i]
        assert (masks[p] if p >= 0 else 0) & r == 0
        assert (masks[p] if p >= 0 else 0) | r == masks[i]
        result = value(r ^ d)
        if p >= 0:
            assert p in output and p in live
            result = result + output[p]
            consumers[p] -= 1
            if consumers[p] == 0:
                live.remove(p)
        if d:
            assert d & r == d
            result = result + cached[d]
        assert np.array_equal(result, value(masks[i]))
        output[i] = result
        if consumers[i] > 0:
            live.add(i)
        high_water = max(high_water, len(live))
    assert not live and len(output) == len(masks)
    return {
        'binary_vector_adds': cost(decisions),
        'weight_vector_reads': sum((r ^ d).bit_count() for _, _, r, d in decisions)
                               + sum(d.bit_count() for d in used),
        'parent_vector_reads_no_forwarding_credit': sum(p >= 0 for _, p, _, _ in decisions),
        'parent_vector_writes_live_only': len({p for _, p, _, _ in decisions if p >= 0}),
        'dictionary_vector_reads': sum(d != 0 for _, _, _, d in decisions),
        'dictionary_vector_writes': len(used),
        'max_live_parent_vectors': high_water,
        'dictionary_vectors_retained_for_tile': len(used),
        'all_original_destination_commits': len(masks),
        'nonzero_destination_commits': sum(m != 0 for m in masks),
        'diagnostic_rational_lane_values_verified': len(masks) * 8,
        'diagnostic_mismatches': 0,
    }


def run_tile(masks, rule, capacities):
    options = parent_options(masks, rule)
    raw = decide(masks, options, [], False, False)
    variants = {'plain': numeric_and_account(masks, raw)}
    search = {}
    for subset in [False, True]:
        label = 'subset' if subset else 'exact'
        candidates = dictionary_candidates(masks, options, subset)
        search[label + '_candidate_masks'] = len(candidates)
        for cap in capacities:
            for joint in [False, True]:
                mode = 'joint' if joint else 'fixed'
                key = f'{mode}_{label}_cap{cap}'
                dec, dictionary, evaluations = choose(masks, options, candidates, cap, joint, subset)
                variants[key] = numeric_and_account(masks, dec)
                variants[key]['chosen_dictionary'] = dictionary
                variants[key]['parent_changes'] = sum(a[1] != b[1] for a, b in zip(raw, dec))
                search[key + '_full_assignment_evaluations'] = evaluations
                if joint:
                    fixed = decide(masks, options, dictionary, False, subset)
                    paired = f'fixed_same_joint_dict_{label}_cap{cap}'
                    variants[paired] = numeric_and_account(masks, fixed)
                    variants[paired]['chosen_dictionary_available'] = dictionary
    return {
        'variants': variants, 'search': search,
        'rows_with_multiple_maximum_parent_choices': sum(len(o) > 1 for o in options),
        'maximum_parent_choices_for_one_row': max(map(len, options)),
        'plain_residual_popcount_histogram': dict(sorted(Counter(r.bit_count() for _, _, r, _ in raw).items())),
        'parent': [d[1] for d in raw], 'residual': [d[2] for d in raw]
    }


def toy():
    P = [(1 << (2*i)) | (1 << (2*i+1)) for i in range(4)]
    A = [(1 << (2*i)) | (1 << 8) for i in range(4)]
    shared = (1 << 8) | (1 << 9) | (1 << 10)
    Q = [p | shared for p in P]
    masks = P + A + Q
    result = run_tile(masks, 'paper_max_index', [1])
    v = result['variants']
    # Account the complete tile, including all real parent outputs, in the record.
    assert v['joint_exact_cap1']['binary_vector_adds'] < v['fixed_subset_cap1']['binary_vector_adds']
    return {'supports': masks, 'rule': 'paper_max_index', 'result': result}


def main():
    out = BASE / 'joint_residual_r1.json'
    assert not out.exists(), 'Do not silently overwrite a research result.'
    plan = json.loads((BASE / 'screen_plan.json').read_text())
    cfg = plan['c1']
    assert digest(LEDGER) == LEDGER_SHA
    spec = importlib.util.spec_from_file_location('sealed_m504_relation', RELATION)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    toy_result = toy()
    rows = []
    with LEDGER.open('rb') as f:
        for sample, operator, k, chunk in itertools.product(cfg['samples'], cfg['operators'],
                                                           cfg['k_partitions'], cfg['spatial_chunks']):
            phase = (sample * 4 + operator) * 432 + k
            n = min(cfg['row_tile'], 3000 - chunk * cfg['row_tile'])
            f.seek((phase * 3000 + chunk * cfg['row_tile']) * 9)
            lines = f.read(n * 9).splitlines()
            assert len(lines) == n and all(len(line) == 8 for line in lines)
            masks = [int(line, 16) for line in lines]
            assert all(0 <= mask < 1 << 16 for mask in masks)
            ref_r, ref_p = reference.cleanroom_subset(np.array(masks, dtype=np.uint16))
            by_rule = {}
            for rule in cfg['parent_rules']:
                result = run_tile(masks, rule, cfg['dictionary_capacities'])
                if rule == 'official_min_index':
                    assert np.array_equal(ref_r, result['residual'])
                    assert np.array_equal(ref_p, result['parent'])
                by_rule[rule] = result
            rows.append({'sample': sample, 'operator': operator, 'k_partition': k,
                         'spatial_chunk': chunk, 'rows': n, 'rules': by_rule})
            if len(rows) % 48 == 0:
                print('verified tiles', len(rows), flush=True)
    aggregate = {}
    for rule in cfg['parent_rules']:
        combined = {}
        for row in rows:
            for name, metrics in row['rules'][rule]['variants'].items():
                acc = combined.setdefault(name, Counter())
                for key, value in metrics.items():
                    if isinstance(value, int):
                        if key.startswith('max_'):
                            acc[key] = max(acc[key], value)
                        else:
                            acc[key] += value
        baseline = combined['plain']['binary_vector_adds']
        for metrics in combined.values():
            metrics['binary_adds_over_plain'] = metrics['binary_vector_adds'] / baseline
        aggregate[rule] = combined
        print(rule, json.dumps({name: data['binary_adds_over_plain'] for name, data in combined.items()}), flush=True)
    result = {
        'date': '2026-09-07', 'plan_sha256': digest(BASE / 'screen_plan.json'),
        'script_sha256': digest(Path(__file__)), 'ledger_sha256': LEDGER_SHA,
        'crosschecked_relation_source_sha256': digest(RELATION),
        'official_artifact_relation_crosscheck': 'Matched sealed NumPy cleanroom relation on every selected tile; full official Simulator.run_fc was not invoked.',
        'scope': '192 selected tiles. Offline greedy maximum-parent-tie/residual-cache opportunity, no latency claim.',
        'numeric_scope': 'Eight diagnostic lanes, integer numerators for theta_c=(999883+17*c)/10^6 and signed W; not checkpoint weights.',
        'limits': ['No complete Phi or Transitive Array cycle implementation in this screen.',
                   'No macro port scheduler; read/write and add counts must not be called cycles.',
                   'Greedy dictionary selection is not globally optimal. Fixed and joint policies receive identical candidates.',
                   'Candidate enumeration and all repeated assignment evaluations require a future realizable planner.',
                   'Ordinary forwarding, zero handling, allocation metadata and wide-result bit widths need identical mapped implementations.'],
        'toy': toy_result, 'aggregate': aggregate, 'tiles': rows,
        'claim_boundary': {'rtl_speedup': False, 'ppa': False, 'full_network': False,
                           'frozen_fp32_equivalence': False, 'new_AEE': False}
    }
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
