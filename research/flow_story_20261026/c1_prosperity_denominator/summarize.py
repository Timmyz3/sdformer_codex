#!/usr/bin/env python3
import csv, json, math
from pathlib import Path
OUT = Path.home() / 'work/sdformer_codex/research/flow_story_20261026/c1_prosperity_denominator'
HW = Path.home() / 'work/sdformer_codex/SDformer/hw_autoresearch_nts07'
RES = HW / 'results/m1590_ep34_c1_same_ledger_cycle_model_r1_20260901'
c1 = json.loads((RES / 'm1579_ep34_c1_same_ledger_cycle_model_result_r1.json').read_text())['aggregate_cycles']
C1 = c1['m505_dead_write_only_1rw_cycles']; ZERO = c1['m468_strong_zero_cycles']
CEIL = c1['m473_fused_concurrent_1r1w_ceiling_cycles']; BIT = c1['m473_same_coordinate_bit_cycles']
assert C1 == 382848700 and ZERO == 648741051
with (RES / 'sample_major_cycles.csv').open() as f:
    c1s = {int(r['sample']): int(r['m505_dead_write_only_1rw_cycles']) for r in csv.DictReader(f)}
runs = {}
for p in sorted((OUT / 'runs').glob('*.json')):
    r = json.loads(p.read_text()); runs.setdefault(r['config'], []).append(r)
summary = dict(c1_reference=dict(c1_finite_1rw=C1, strong_zero=ZERO, same_coord_bit=BIT, concurrent_ceiling=CEIL,
                                 source=str(RES / 'm1579_ep34_c1_same_ledger_cycle_model_result_r1.json')),
               configs={})
for cfg, rs in sorted(runs.items()):
    assert len(rs) == 40, (cfg, len(rs))
    tot = lambda k: sum(r[k] for r in rs)
    acc = rs[0]['accelerator']
    n = acc['adder_array_size']
    d = dict(accelerator=acc, layers=len(rs), total_cycles=tot('total_cycles'), compute_cycles=tot('compute_cycles'),
             preprocess_stall_cycles=tot('preprocess_stall_cycles'), mem_stall_cycles=tot('mem_stall_cycles'),
             num_ops=tot('num_ops'), im2col_nnz=tot('im2col_nnz'), dram_reads_bits=sum(r['reads'].get('dram', 0) for r in rs))
    d['ratio_prosperity_over_C1'] = d['total_cycles'] / C1
    d['ratio_prosperity_compute_only_over_C1'] = d['compute_cycles'] / C1
    d['ratio_prosperity_over_strong_zero'] = d['total_cycles'] / ZERO
    if n != 96:
        d['lane_normalized_to_96_total_cycles'] = d['total_cycles'] * n / 96
        d['lane_normalized_ratio_over_C1'] = d['total_cycles'] * n / 96 / C1
    per = {}
    for r in rs:
        per[r['sample']] = per.get(r['sample'], 0) + r['total_cycles']
    ratios = [per[s] / c1s[s] for s in sorted(per)]
    d['per_sample_ratio_over_C1'] = dict(min=min(ratios), max=max(ratios),
                                         geomean=math.exp(sum(map(math.log, ratios)) / len(ratios)))
    if 'global_counters' in rs[0]:
        tn = math.ceil(768 / n)
        g = {k: sum(r['global_counters'][k] for r in rs) // tn for k in
             ('ori_nnz', 'processed_nnz', 'ops_nnz', 'EM_rows', 'PM_rows', 'all_zero_rows', 'prefix_edges')}
        d['per_matrix_counters'] = g
    summary['configs'][cfg] = d
(OUT / 'SUMMARY.json').write_text(json.dumps(summary, indent=1))
for cfg, d in summary['configs'].items():
    print(f"{cfg:20s} total={d['total_cycles']:>13,d} compute={d['compute_cycles']:>13,d} memstall={d['mem_stall_cycles']:>11,d} "
          f"ppstall={d['preprocess_stall_cycles']:>11,d} P/C1={d['ratio_prosperity_over_C1']:.4f} "
          f"norm={d.get('lane_normalized_ratio_over_C1', float('nan')):.4f} persample={d['per_sample_ratio_over_C1']}")
    if 'per_matrix_counters' in d: print('   ', d['per_matrix_counters'])
