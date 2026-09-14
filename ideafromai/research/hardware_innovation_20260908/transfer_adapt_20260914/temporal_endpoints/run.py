from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess

H=Path(__file__).resolve().parent
with (H/'build.log').open('w') as log:
    subprocess.run(['verilator','-Wall','--cc','--exe','--top-module','decomp_core','--Mdir','obj','decomp_core.sv','tb.cpp','-CFLAGS','-O2 -std=c++14'],cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run(['make','-C','obj','-f','Vdecomp_core.mk','-j2'],cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
cases=json.loads((H/'fixtures.json').read_text())
(H/'logs').mkdir(exist_ok=True)
def run(job):
    c,m,s,n=job
    result=subprocess.run([str(H/'obj/Vdecomp_core'),c['path'],str(m),str(s),str(n)],capture_output=True,text=True)
    (H/f"logs/{c['name']}_{m}_{n}_{s}.log").write_text(result.stdout+result.stderr)
    assert result.returncode==0,(job,result.returncode,result.stdout,result.stderr)
    return [dict(json.loads(l),fixture=c['name'],cross_mode=m!=n) for l in result.stdout.splitlines()]
jobs=[(c,m,s,m) for c in cases for m in [0,1,2,3,4,5] for s in [0,1]]
jobs += [(c,m,1,n) for c in cases if c['name'] in ['mixed_runs','late_endpoint_runs','negative4_falling','random_padding_poison','zero'] for m,n in [(0,2),(2,1),(1,0),(3,0),(1,3),(3,2)]]
jobs += [(c,m,1,n) for c in cases if c['name'] in ['inverse_time_tag','late_endpoint_runs','real_6'] for m,n in [(0,4),(4,0),(3,4),(4,3)]]
jobs += [(c,m,1,n) for c in cases if c['name'] in ['inverse_time_tag','late_endpoint_runs','real_6'] for m,n in [(0,5),(5,4),(4,5),(5,0)]]
with ThreadPoolExecutor(max_workers=3) as pool:
    rows=[r for part in pool.map(run,jobs) for r in part]
(H/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
summary=dict(passed=True,commands=len(rows),raw_values=sum(r['outputs'] for r in rows),real={})
for m in [0,1,2,3,4,5]:
    for s in [0,1]:
        rs=[r for r in rows if r['fixture'].startswith('real_') and r['mode']==m and r['stall']==s and r['command']==0 and not r['cross_mode']]
        summary['real'][f'm{m}_s{s}']={k:sum(r[k] for r in rs) for k in rs[0] if k not in ['fixture','mode','stall','command','state_cycles','cross_mode']}
(H/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
