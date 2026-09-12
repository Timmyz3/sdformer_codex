"""Summarize completed fixed projection and source RTL without rerunning them."""
from pathlib import Path
import csv
import json

HERE=Path(__file__).resolve().parent
rows=[]
totals=dict(RTL_cases=0,gate_bits_checked=0,RF_vector_writebacks_checked=0)
for structure in ('dense','lifting40'):
    r=json.loads((HERE/structure/'results.json').read_text())
    q=r['quantization'];checks=r['cpu_checks']
    parent={(x['window'],x['mode']):x for x in r['parent_RTL_rows']}
    assert all(x['differences']==0 for x in r['rows'])
    for mode in ('ready','stress'):
        new=[x for x in r['rows'] if x['mode']==mode]
        old=[parent[x['window'],mode] for x in new]
        for a,b in zip(new,old):
            for key in ('tiles','SR64_reads','SW64_writes','gate_bits_checked'):
                assert a[key]==b[key],(structure,mode,key)
        tiles=sum(x['tiles'] for x in new)
        cycles=sum(x['cycles'] for x in new)
        parent_cycles=sum(x['cycles'] for x in old)
        row=dict(structure=structure,mode=mode,changed_coefficients=q['changed_coefficients'],
            total_coefficients=q['coefficients'],exponent=q['exponent'],
            parent_addsub=r['parent_accounting']['operation_slots']['addsub'],
            new_addsub=r['accounting']['operation_slots']['addsub'],
            parent_program_instructions=r['parent_accounting']['instructions'],
            new_program_instructions=r['accounting']['instructions'],
            parent_peak_RF_without_gate=r['parent_accounting']['peak_live_words'],
            new_peak_RF_without_gate=r['accounting']['peak_live_words'],
            gate_collector_RF_vectors=1,materialized_norm24=r['lifting_materialized_norm24'],
            tiles=tiles,parent_cycles=parent_cycles,new_cycles=cycles,
            parent_cycles_per_tile=parent_cycles/tiles,new_cycles_per_tile=cycles/tiles,
            source_cycle_reduction=1-cycles/parent_cycles,
            gate_bit_differences_vs_parent=sum(x['gate_bit_differences_vs_parent'] for x in checks),
            distinct_input_gate_bits=sum(x['gate_bits'] for x in checks),
            changed_gate_rate=sum(x['gate_bit_differences_vs_parent'] for x in checks)/sum(x['gate_bits'] for x in checks),
            SR64_reads=sum(x['SR64_reads'] for x in new),SW64_writes=sum(x['SW64_writes'] for x in new),
            RTL_differences=sum(x['differences'] for x in new),new_AEE10=None,new_AEE825=None)
        rows.append(row)
    totals['RTL_cases']+=len(r['rows'])
    for key in ('gate_bits_checked','RF_vector_writebacks_checked'):
        totals[key]+=sum(x[key] for x in r['rows'])
summary=dict(complete_CPU_RTL=True,complete_AEE=False,rows=rows,totals=totals,
    new_X=False,training=False,parameter_sweep=False,
    scope='New signed-power source functions; same source-only CSE/RTL comparison, no full-chain service or inherited AEE.',
    strong_ordinary_reference='Original matched contiguous34 source remains319-versus303 relevant context; no quality equivalence between structures.',
    handoff='dense/deployed_constants.npz and lifting40/deployed_constants.npz load directly into LiteralForward with the matching structure; per-arm cpu_gold.npz is T,C,H,W.')
(HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (HERE/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(summary,indent=2))
