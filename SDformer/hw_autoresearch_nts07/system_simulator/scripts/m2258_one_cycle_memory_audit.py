#!/usr/bin/env python3
"""Validate all nine one-cycle SRAM anchors, then replay fixed G48 population."""
from collections import Counter
import json
from pathlib import Path
from m2252_masked_c2_cycle_model import chunks, run_chunk

HW = Path(__file__).resolve().parents[2]


def main():
    anchors = {}
    for directory in ('m2249_bank_selective_mlj_ck8y', 'm2249_bank_selective_ct0ijuvj'):
        result = json.loads((HW / 'results' / directory / 'result.json').read_text())
        assert result['status'] == 'PASS'
        assert result['memory_latency_cycles'] == 1 and result['bank_ready_unstalled']
        anchors.update({(r['axis'], r['slot']): r for r in result['rows']})
    totals = {axis: Counter() for axis in ('ordinary', 'tsbg_demand', 'tsbg')}
    checks = count = 0; tails = Counter(); worst = None
    for prefix, row, words in chunks():
        points = {}
        for axis in totals:
            point = run_chunk(words, int(axis != 'ordinary'),
                prefetch_union=axis == 'tsbg', memory_latency=1, always_ready=True)
            points[axis] = point
            anchor = anchors.get((axis, row['slot'])) if prefix.startswith('m2051') else None
            if anchor:
                assert all(point[k] == anchor[k] for k in ('cycles', 'bank_reads'))
                checks += 1
            totals[axis].update(point)
        demand = points['tsbg_demand']; union = points['tsbg']
        assert demand['bank_reads'] == union['bank_reads']
        ratio = demand['cycles'] / union['cycles']
        tails['faster' if ratio > 1 else 'tie' if ratio == 1 else 'slower'] += 1
        if worst is None or ratio < worst['ratio']:
            worst = dict(fixture=prefix, slot=row['slot'], ratio=ratio,
                demand_cycles=demand['cycles'], union_cycles=union['cycles'])
        count += 1
    assert checks == 9 and count == 4320
    result = dict(chunks=count, vcs_cycle_and_read_matches=checks, totals=totals,
        union_vs_group_demand_cycle_ratio=totals['tsbg_demand']['cycles']/totals['tsbg']['cycles'],
        union_vs_ordinary_cycle_ratio=totals['ordinary']['cycles']/totals['tsbg']['cycles'],
        union_extra_bank_read_reduction=0,
        refill_transaction_reduction=1-totals['tsbg']['refill_beats']/totals['tsbg_demand']['refill_beats'],
        per_chunk_cycles=tails, worst=worst,
        scope='4320 independently reset G48 chunks; CPU model calibrated to nine directed VCS anchors',
        service='one-cycle bank response, bank ready always high; unchanged issue/commit backpressure',
        no_full_network_area_or_energy_claim=True)
    out = HW / 'results/m2258_one_cycle_memory_audit/result.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
