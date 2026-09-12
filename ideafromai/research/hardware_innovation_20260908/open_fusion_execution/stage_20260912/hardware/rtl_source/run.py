"""Compile and replay actual ordinary/lifting programs on identical RTL."""
from pathlib import Path
import json
import subprocess
import sys

HERE=Path(__file__).resolve().parent
INPUT=HERE.parent/'source_rtl_inputs'
def execute(args):
    return subprocess.run(list(map(str,args)),cwd=HERE,check=True,capture_output=True,text=True)
execute([sys.executable,'prepare.py'])
v=execute(['verilator','-cc','--exe','-O3','-Wall','-Wno-fatal','--top-module',
           'temporal_source','temporal_source.sv','tb.cpp','-CFLAGS','-O2 -std=c++17'])
(HERE/'lint.log').write_text(v.stdout+v.stderr)
execute(['make','-C','obj_dir','-f','Vtemporal_source.mk','-j4'])
rows=[]
for case in json.loads((INPUT/'manifest.json').read_text())['cases']:
    axis,window=case['axis'],case['window']
    for mode in ('ready','stress','observed'):
        args=['obj_dir/Vtemporal_source',f'generated/{axis}.txt',
              INPUT/case['input_file'],INPUT/case['gate_gold_file'],mode]
        if mode=='observed':args+=[f'generated/{axis}_{window}_sink_trace.waits']
        row=json.loads(execute(args).stdout)
        row.update(axis=axis,window=window)
        rows.append(row)
        print(json.dumps(row),flush=True)
extra='generated/ordinary_interior_stress_sink_trace.waits'
if (HERE/extra).exists():
    case=next(c for c in json.loads((INPUT/'manifest.json').read_text())['cases'] if c['axis']=='ordinary' and c['window']=='interior')
    row=json.loads(execute(['obj_dir/Vtemporal_source','generated/ordinary.txt',
        INPUT/case['input_file'],INPUT/case['gate_gold_file'],'observed',extra]).stdout)
    row.update(axis='ordinary',window='interior',trace='integrated_stress_SW64_relative_waits')
    rows.append(row)
directed=[]
for mode in ('ready','stress'):
    directed.append(json.loads(execute(['obj_dir/Vtemporal_source','generated/directed.txt',
                                       'generated/directed_i24.bin','generated/directed_gate.bin',mode]).stdout))
result=dict(evidence='Verilator isolated source RTL; NOT full chain RTL, VCS/DC/PT/Formality speedup or PPA.',
    interface='1R64 request/response, 1W64 ready/valid; 96x8 signed48 RF, 512x128 programmable ROM; identical datapath for both programs.',
    charged='Instruction execution, 35 lifting intermediate RNE/sat operations, all physical source reads, both gate writes, RF hazards, supplied port backpressure.',
    excluded='External input DMA and consumer arithmetic are charged only by the separate integrated CPU model. One-time program configuration is excluded by BOTH models (fixed resident ROM). Shared SRAM macro implementation is not evaluated by either model.',
    observed_mode='Replay per-write observed wait durations only, not the original absolute integrated timeline or a concurrent consumer.',
    ready='No external stalls; deterministic identical environment for both programs.',
    stress='Read request blocked in cycle%32>=24; write blocked in cycle%32>=28; response latency cycles 1..3. Functional stress, not workload measurement.',
    rows=rows,directed_RNE_saturation_threshold=directed)
(HERE/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS',len(rows),'real cases;',sum(r['RF_vector_writebacks_checked']*8 for r in rows),'checked RF values')
