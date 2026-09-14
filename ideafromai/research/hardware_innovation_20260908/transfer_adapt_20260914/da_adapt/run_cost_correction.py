"""Incremental mode11 verification; run.py reproduces the full five-mode experiment."""
from pathlib import Path
import json,subprocess,concurrent.futures
H=Path(__file__).resolve().parent
with (H/'build_cost_correction.log').open('w') as log:
    for cmd in (['verilator','-Wall','--cc','--exe','--top-module','decomp_core','decomp_core.sv','tb.cpp','-CFLAGS','-O3'],['make','-C','obj_dir','-f','Vdecomp_core.mk','-j2']):
        subprocess.run(cmd,cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
cases=json.loads((H/'definition.json').read_text())['fixtures']
def run(task):
    case,stall,second=task
    name=f'{case}_m11_s{stall}'+(f'_to_{second}' if second else '')
    trace=H/'traces'/(name+'.hex')
    cmd=[str(H/'obj_dir/Vdecomp_core'),str(H/'fixtures'/case),'11',str(stall),str(trace)]
    if second:cmd.append(str(H/'fixtures'/second))
    r=subprocess.run(cmd,capture_output=True,text=True)
    if r.returncode:raise RuntimeError((task,r.returncode,r.stdout,r.stderr))
    a=[dict(json.loads(x),fixture=(second if i==1 and second else case),sequence=name,reconfigured=bool(second),trace=str(trace.relative_to(H))) for i,x in enumerate(r.stdout.splitlines())]
    trace_values=trace.read_text().split();assert len(a)==2 and len(trace_values)==7680
    for i,row in enumerate(a):assert trace_values[i*3840:(i+1)*3840]==(H/'fixtures'/row['fixture']/'gold.hex').read_text().split()
    print('PASS',name,[x['cycles'] for x in a],flush=True);return a
rows=[r for r in json.loads((H/'results.json').read_text()) if r['mode']!=11]
tasks=[(x['name'],s,None) for x in cases for s in [0,1]]
tasks += [(a,1,b) for a,b in [('small_dense_positive','q2_all_zero'),('q2_all_zero','small_dense_negative'),('small_dense_negative','padding_poison'),('padding_poison','rank2_small')]]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    for a in pool.map(run,tasks):rows.extend(a)
(H/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
print('COMPLETE',len(rows),'commands',sum(r['outputs'] for r in rows),'values')
