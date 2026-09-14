from pathlib import Path
import json,subprocess,concurrent.futures
H=Path(__file__).resolve().parent
with (H/'build.log').open('w') as log:
    for cmd in (['verilator','-Wall','--cc','--exe','--top-module','decomp_core','decomp_core.sv','tb.cpp','-CFLAGS','-O3'],['make','-C','obj_dir','-f','Vdecomp_core.mk','-j2']):
        subprocess.run(cmd,cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
(H/'traces').mkdir(exist_ok=True)
cases=json.loads((H/'definition.json').read_text())['fixtures']
def run(task):
    case,mode,stall,second=task
    name=f'{case}_m{mode}_s{stall}'+(f'_to_{second}' if second else '')
    trace=H/'traces'/(name+'.hex')
    cmd=[str(H/'obj_dir/Vdecomp_core'),str(H/'fixtures'/case),str(mode),str(stall),str(trace)]
    if second:cmd.append(str(H/'fixtures'/second))
    r=subprocess.run(cmd,capture_output=True,text=True)
    if r.returncode:raise RuntimeError((task,r.returncode,r.stdout,r.stderr))
    a=[dict(json.loads(x),fixture=(second if i==1 and second else case),sequence=name,reconfigured=bool(second),trace=str(trace.relative_to(H))) for i,x in enumerate(r.stdout.splitlines())]
    assert len(a)==2
    trace_values=trace.read_text().split();assert len(trace_values)==7680
    for i,row in enumerate(a):
        assert trace_values[i*3840:(i+1)*3840]==(H/'fixtures'/row['fixture']/'gold.hex').read_text().split()
    print('PASS',name,[v['cycles'] for v in a],flush=True);return a
rows=[]
tasks=[(x['name'],m,s,None) for x in cases for m in [14,15,13,12,11] for s in [0,1]]
# Warm commands with changed source, padding, Q1 and Q2 exercise LUT invalidation.
pairs=[('small_dense_positive','q2_all_zero'),('q2_all_zero','small_dense_negative'),('small_dense_negative','padding_poison'),('padding_poison','rank2_small')]
tasks += [(a,m,1,b) for a,b in pairs for m in [14,15,13,12,11]]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    for result in pool.map(run,tasks):rows.extend(result)
(H/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
print('COMPLETE',len(rows),'commands',sum(r['outputs'] for r in rows),'values',flush=True)
