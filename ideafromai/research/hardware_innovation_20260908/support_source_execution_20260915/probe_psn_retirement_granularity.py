#!/opt/anaconda3/bin/python3.12
"""New trace probe: exact gate-certificate retirement at actual H1/H12/H96 groups.

Declared question before running: scalar certificate depth may overstate useful
work cancellation when a physical bank/vector waits for its last consumer.
This is not BitL RTL, a new certificate formula, a code/time contraction, or AEE.
"""
from pathlib import Path
import csv
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'support_lut_execution_20260915'
WIDTHS = (1, 12, 96)  # Existing scalar, bank and vector obligations; no parameter search.


def histogram(x):
    v, n = np.unique(x, return_counts=True)
    return {str(int(a)): int(b) for a, b in zip(v, n)}


def execute(y, a, tau, width, reference):
    p, s, h = y.shape
    groups = h // width
    magnitude = np.abs(y).max(axis=1)
    raw_exp = np.zeros_like(magnitude)
    for bit in range(24):
        raw_exp = np.where(magnitude >= 1 << bit, bit+1, raw_exp)
    exp_group = raw_exp.reshape(p, groups, width).max(axis=-1)
    exponent = np.repeat(exp_group, width, axis=1)
    effective_exp = np.maximum(exponent, 1)
    sign = y < 0
    v = -np.einsum('ts,psh->pth', a, sign.astype(np.int64))
    locked = np.zeros_like(v, dtype=bool)
    decisions = np.zeros_like(v, dtype=bool)
    depth = np.zeros_like(v, dtype=np.int64)
    positive = np.maximum(a, 0).sum(axis=1)[None, :, None]
    negative = np.minimum(a, 0).sum(axis=1)[None, :, None]
    # Literal subset-tree lower count: first term assignment and wiring shifts free.
    header_adds = int(sign.sum()) * a.shape[0]
    full_plane_adds = cert_plane_adds = comparisons = 0
    for step in range(1, int(effective_exp.max())+1):
        active_h = step <= effective_exp
        m = np.maximum(effective_exp-step, 0)
        bits = (y >> m[:, None, :]) & 1
        dot = np.einsum('ts,psh->pth', a, bits)
        nv = 2*v + dot
        v = np.where(active_h[:, None, :], nv, v)
        suffix = (1 << m[:, None, :])-1
        lo = (v << m[:, None, :]) + negative*suffix
        hi = (v << m[:, None, :]) + positive*suffix
        undecided = (~locked) & active_h[:, None, :]
        # Inclusive >= threshold contract, real cases are positive and nonconstant.
        can_decide = undecided & ((lo >= tau[None]) | (hi < tau[None]))
        assert np.all((lo <= reference) | ~active_h[:, None, :])
        assert np.all((hi >= reference) | ~active_h[:, None, :])
        decisions[can_decide] = (lo >= tau[None])[can_decide]
        depth[can_decide] = step
        pop = bits.sum(axis=1)
        full_plane_adds += int((pop*active_h).sum())*a.shape[0]
        cert_plane_adds += int((undecided*pop[:, None, :]).sum())
        comparisons += 2*int(undecided.sum())
        locked |= can_decide
    assert locked.all() and np.array_equal(v, reference)
    assert np.array_equal(decisions, reference >= tau[None])
    # Shared group includes all T outputs and every h belonging to that group.
    group_depth = depth.reshape(p, a.shape[0], groups, width).max(axis=(1, 3))
    full_depth = np.maximum(exp_group, 1)
    assert np.all(group_depth <= full_depth)
    return {
        'H_group': width, 'physical_groups': p*groups,
        'full_data_planes': int(full_depth.sum()), 'cert_last_consumer_planes': int(group_depth.sum()),
        'full_header_plus_planes': int((full_depth+1).sum()),
        'cert_header_plus_planes': int((group_depth+1).sum()),
        'group_with_early_retirement': int((group_depth < full_depth).sum()),
        'group_last_consumer_depth_histogram': histogram(group_depth),
        'individual_gate_lock_depth_histogram': histogram(depth),
        'individual_gates_locked_before_last_plane': int((depth < effective_exp[:, None, :]).sum()),
        'gate_values_checked': int(v.size), 'mismatches': 0,
        'literal_subset_tree_scalar_adds_full': header_adds + full_plane_adds,
        'literal_subset_tree_scalar_adds_cert': header_adds + cert_plane_adds,
        'literal_subset_tree_ALU96_capacity_floor_full': math.ceil((header_adds+full_plane_adds)/96),
        'literal_subset_tree_ALU96_capacity_floor_cert': math.ceil((header_adds+cert_plane_adds)/96),
        'certificate_comparisons_excluding_header': comparisons,
        'bounds_and_comparisons_not_counted_as_free_hardware': True,
    }


def main():
    data = np.load(DATA / 'cases.npz', allow_pickle=False)
    indices = np.flatnonzero(data['is_real'])
    assert len(indices) == 32
    a = data['A'].astype(np.int64)
    assert a.shape == (10, 10) and np.count_nonzero(a) == 100
    rtl = {(x['case'], int(x['mode']), int(x['bp'])): x
           for x in csv.DictReader((DATA / 'rtl_cycles.csv').open())}
    cases = []
    aggregate = {w: {} for w in WIDTHS}
    psn_mac = psn_cycles = total_cycles = 0
    for index in indices:
        name = str(data['case_name'][index])
        source = data['S'][index].astype(np.int64)
        y = source @ data['W'][index].astype(np.int64)
        assert np.array_equal(y, data['Y'][index])
        u = np.einsum('ts,psh->pth', a, y)
        assert np.array_equal(u, data['U'][index])
        assert np.all(data['positive_gain'][index]) and not np.any(data['constant_channels'][index])
        assert np.array_equal(u >= data['tau'][index][None], data['gold'][index])
        rows = [execute(y, a, data['tau'][index].astype(np.int64), w, u) for w in WIDTHS]
        base = rtl[(name, 2, 0)]
        psn_mac += int(base['mac']); psn_cycles += int(base['psn_cycles']); total_cycles += int(base['cycles'])
        cases.append({'case': name, 'frame_file': str(data['frame_file'][index]),
            'tile_index': int(data['tile_index'][index]), 'hblock': int(data['hblock'][index]),
            'baseline_PSN_MAC96': int(base['mac']), 'baseline_PSN_cycles': int(base['psn_cycles']),
            'granularities': rows})
        for row in rows:
            out = aggregate[row['H_group']]
            for k, value in row.items():
                if isinstance(value, int) and not isinstance(value, bool) and k != 'H_group':
                    out[k] = out.get(k, 0) + value
                elif k.endswith('_histogram'):
                    out.setdefault(k, {})
                    for key, count in value.items():
                        out[k][key] = out[k].get(key, 0) + count
    assert psn_mac == 83400 and psn_cycles == 118344 and total_cycles == 184184
    for out in aggregate.values():
        out['plane_reduction_fraction'] = 1-out['cert_last_consumer_planes']/out['full_data_planes']
        out['header_inclusive_reduction_fraction'] = 1-out['cert_header_plus_planes']/out['full_header_plus_planes']
        out['literal_ALU_floor_vs_native_MAC_issues'] = out['literal_subset_tree_ALU96_capacity_floor_cert']/psn_mac
    result = {
        'status': 'PASS', 'scope': 'CPU exact certificate retirement/granularity and literal-tree operation counts, not simulated latency',
        'inputs': str(DATA/'cases.npz'), 'reference_RTL': str(DATA/'rtl_cycles.csv'),
        'cases': cases, 'aggregate_by_H_group': aggregate,
        'real_commands':32, 'independent_source_tiles':8, 'distinct_H96_functions_per_source':4,
        'baseline': {'PSN_MAC96_issues':psn_mac, 'PSN_cycles':psn_cycles, 'complete_cycles':total_cycles},
        'coverage': {'unique_U_values':32*32*10*96, 'gate_checks_across_three_granularities':32*32*10*96*3},
        'arithmetic': 'same signed prefix/residual interval as existing T5; exact A, tau and positive gain; no intermediate RNE',
        'holding': {'all_T10_for_H96_bytes_Y24':10*96*3, 'current_single_row_yhold_bytes':96*3,
            'additional_holding_if_added_as_registers_bytes':9*96*3, 'no_storage_or_port_allocation_performed':True},
        'limitations': [
            'Y is independently recomputed from actual S/W. This probe does not implement its production/last-writer protocol.',
            'Three H widths are the existing scalar, H12 bank, H96 vector obligations; no fitted threshold or width search.',
            'Plane sums across H groups are work indicators. Different widths require different concurrent resources; they are not directly comparable cycle counts.',
            'Literal subset-tree add counts grant arbitrary lane packing and ignore bounds/compare/control costs. They do not bound optimized constant-MVM, BitL, or subset-lookup algorithms.',
            'Actual support RTL has 96 multipliers and 96 shared 48-bit adders; the old T5 parallel add trees cannot be replicated 96 times without charging resources.',
            'No extra SRAM/table ports, inverse layout, lookup precomputation, exponent update during production, or output scheduling are implemented.',
            'Real source/consumer integer identity only; no new training, AEE, RTL or ASIC PPA.'
        ]}
    target = HERE/'probe_psn_retirement_granularity.json'
    target.write_text(json.dumps(result, ensure_ascii=False, separators=(',', ':'))+'\n')
    print(json.dumps({'status':'PASS','result':str(target),'aggregates':aggregate},ensure_ascii=False))


if __name__ == '__main__':
    main()
