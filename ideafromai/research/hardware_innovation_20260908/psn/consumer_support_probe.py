"""Real source-support specialization of PSN constants; arithmetic opportunities only."""
from collections import Counter
import json
from pathlib import Path
import sys
import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
from gustavsnn_reference import identities, read_torch
from shared_adder_service_model import csd


def digits(a):
    return sum(len(csd(int(v))) for v in np.asarray(a).flat)


def key(row, kind):
    row = np.asarray(row, dtype=np.int64)
    if not np.any(row):
        return None
    if kind in ('signed', 'dyadic') and row[row != 0][0] < 0:
        row = -row
    if kind == 'dyadic':
        while np.all(row % 2 == 0):
            row = row//2
    return tuple(row.tolist())


def relation_groups(B, kind):
    groups = {}
    for t, row in enumerate(B):
        k = key(row, kind)
        if k is not None:
            groups.setdefault(k, []).append(t)
    return list(groups.values())


def cost(mask, B, D, A, plan):
    live = np.asarray([(mask >> r) & 1 for r in range(7)], dtype=bool)
    restricted = B[:, live]
    folded_D = {}
    time_live = 0
    for r, row in enumerate(D[:, live]):
        if not np.any(row):
            continue
        time_live += 1
        k = tuple(row.tolist())
        folded_D.setdefault(k, []).append(r)
    time_zero = A[:, np.any(D[:, live], axis=1)]
    folded_A = np.stack([A[:, cols].sum(axis=1) for cols in folded_D.values()], axis=1) if folded_D else np.zeros((10, 0), dtype=np.int64)
    groups = {kind: relation_groups(restricted, kind) for kind in ('equal', 'signed', 'dyadic')}
    # All candidate output relations are exact in independent class-S coordinates.
    # Different tau[t,h] are never combined: ten threshold decisions remain.
    class_shared = sum(min(digits(restricted[t]) for t in g) for g in groups['dyadic'])
    time_shared = sum(min(digits(folded_A[t]) for t in g) for g in groups['dyadic'])

    # Ordinary existing disjoint-subset CSE, restricted to live source classes.
    # Empty operands alias their other input; duplicate results alias old values.
    known = {1 << r for r in range(7) if live[r]}
    adds = 0
    for step in plan['addition_steps']:
        left = step['left_subset'] & mask
        right = step['right_subset'] & mask
        target = step['result_subset'] & mask
        assert target == (left | right) and not (left & right)
        if target and target not in known:
            assert left in known and right in known
            adds += 1
            known.add(target)
    targets = {int(v) & mask for v in plan['target_subsets']}
    assert all(v == 0 or v in known for v in targets)
    return {
        'class_live_S': int(live.sum()), 'time_live_S_before_equal_fold': time_live,
        'time_distinct_live_S': len(folded_D),
        'B_live_projection_rows': int(np.any(restricted, axis=1).sum()),
        'B_equal_projection_groups': len(groups['equal']),
        'B_signed_projection_groups': len(groups['signed']),
        'B_dyadic_projection_groups': len(groups['dyadic']),
        'class_zero_S_CSD_terms': digits(restricted),
        'time_zero_S_CSD_terms': digits(time_zero),
        'time_equal_input_fold_CSD_terms': digits(folded_A),
        'class_projection_relation_CSD_terms': class_shared,
        'time_input_and_projection_relation_CSD_terms': time_shared,
        'ordinary_restore_CSE_additions': adds,
        'ordinary_restore_then_A_arithmetic_terms': adds+digits(folded_A),
        'ordinary_restore_then_A_load_add_upper_terms': 2*adds+digits(folded_A),
        'retained_threshold_decisions': 10,
    }


def main():
    params = read_torch(ROOT/'algorithm/stage2_temporal_codes/integer_parameters.pt')
    books = np.load(ROOT/'algorithm/stage2_temporal_codes/codebooks.npz')
    basis = np.load(ROOT/'algorithm/stage2_temporal_codes/signed_basis.npz')
    consumers = json.loads((ROOT/'algorithm/stage2_class_shift/consumers.json').read_text())
    controls = json.loads((ROOT/'psn/class_reconstruct_control.json').read_text())['records']
    records = []
    totals = Counter()
    for b in range(6):
        W, routes, identity = identities(b, params, books, basis, consumers)
        H = len(W)
        B = routes['packed_class'][1]
        D, A = routes['packed_time']
        D = D[:, 1:]
        assert np.array_equal(A@D, B)
        control = next(x for x in controls if x['variant'] == 'bits3' and x['block'] == b)
        plan = control['plan']
        histogram, p4_histogram = Counter(), Counter()
        files = sorted((ROOT/'algorithm/direct_code_integer/deployment/capture10').glob(f'*_s2b{b}.npz'))
        assert len(files) == 10
        for filename in files:
            codes = np.load(filename)['codes']
            assert codes.shape == (1200, 384)
            masks = sum(((codes == r+1).any(axis=1).astype(np.int64) << r) for r in range(7))
            histogram.update(masks.tolist())
            p4_histogram.update(np.bitwise_or.reduce(masks.reshape(-1, 4), axis=1).tolist())
        masks_seen = set(histogram) | set(p4_histogram) | {127}
        costs = {m: cost(m, B, D, A, plan) for m in masks_seen}
        counts = Counter()
        for mask, n in histogram.items():
            counts.update({k: v*n for k, v in costs[mask].items()})
        p4_counts = Counter()
        for mask, n in p4_histogram.items():
            p4_counts.update({k: v*n*4 for k, v in costs[mask].items()})
        positions = sum(histogram.values())
        base = costs[127]
        counts['positions'] = positions
        counts['static_class_CSD_terms'] = base['class_zero_S_CSD_terms']*positions
        counts['static_time_CSD_terms'] = base['time_zero_S_CSD_terms']*positions
        counts['static_best_class_tail_terms'] = min(base['class_zero_S_CSD_terms'], control['restore_one_read_found'])*positions
        # Full-H application count is a count, not a PE cycle conversion.
        totals.update({k: v*H for k, v in counts.items()})
        records.append({'block': b, 'H': H, 'files': len(files), 'positions': positions,
            'static_cost': base, 'weighted_counts_per_output_h': dict(counts),
            'P4_common_union_mask_counts_per_output_h': dict(p4_counts),
            'mask_histogram': {str(k): v for k, v in sorted(histogram.items())},
            'observed_mask_costs': {str(k): costs[k] for k in sorted(histogram)}})
        print(b, 'terms/h/position', {k: round(counts[k]/positions, 4) for k in (
            'static_class_CSD_terms', 'class_zero_S_CSD_terms', 'class_projection_relation_CSD_terms',
            'static_time_CSD_terms', 'time_zero_S_CSD_terms', 'time_equal_input_fold_CSD_terms',
            'time_input_and_projection_relation_CSD_terms')}, flush=True)
    output = {'kind': 'SOURCE_SUPPORT_SYMBOLIC_CONSUMER_OPPORTUNITY', 'capture_files': 60,
        'student': 'six-source integer bits3; same power2 trained B and actual theta-g interface',
        'support': 'classes occurring over full C384 at each p; shared by eight output-row tiles at the same k-ID; excludes h-specific numeric cancellation',
        'identity': 'B[:,M] == A*D[:,M]; missing classes are exact zero S for every W; no model arithmetic change',
        'tau': 'ten distinct tau[t,h] decisions retained, including zero projections; equal/negative/dyadic projection shares never merge gate outputs',
        'ordinary_controls': 'both class/time get zero-S skipping, restricted D equality folding, existing disjoint-subset restoration CSE and output proportionality CSE',
        'representation': 'dyadic grouping permits common projection and separate rescaled integer thresholds; materialized shifts/branches and signal fanout are unpriced',
        'cost_boundary': 'counts of CSD coefficient terms, existing restoration additions/loads, and threshold decisions; not cycle, energy, or mapped area',
        'implementation_options_to_test': ['ordinary per-(t,r) class-presence jump with one common coefficient table; no 128 program copies',
            'a small fixed set of guarded projection-sharing tails must pay guard selection, opcode delivery, cache/U liveness and eight k-ID control paths'],
        'unpriced': ['support production while filling each source bank, at least P*7 support bits per k-ID',
            'multioutput-CSE selection/control and shared-projection register', 'actual instruction/metadata ports and backpressure',
            'FC1 execution is complete before this consumer-only opportunity; no claimed early state release or W request reduction',
            'ordinary general add/sub MCM can be stronger than the existing positive-subset CSE plan'],
        'aggregate_full_H_terms': dict(totals), 'records': records}
    target = ROOT/'psn/consumer_support_probe.json'
    target.write_text(json.dumps(output, indent=2)+'\n')
    print(target)


if __name__ == '__main__':
    main()
