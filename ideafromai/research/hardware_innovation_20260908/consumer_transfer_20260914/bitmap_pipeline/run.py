from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,subprocess

H=Path(__file__).resolve().parent;B=H.parents[1]
P=B/'transfer_adapt_20260914/pair_sparse'
with (H/'build.log').open('w') as log:
    for cmd in [
        ['verilator','-Wall','--cc','--exe','--top-module','decomp_core','--Mdir','obj','decomp_core.sv','tb.cpp','-CFLAGS','-O2 -std=c++14'],
        ['make','-C','obj','-f','Vdecomp_core.mk','-j2'],
    ]:subprocess.run(cmd,cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
cases=[]
for rec in json.loads((P/'fixtures.json').read_text()):
    p=P/'fixtures'/rec['name']
    if not p.exists():p=B/'fusion_review_followup_20260914/pair_dictionary/fixtures'/rec['name']
    cases.append(p)
def run(job):
    p,m,s=job
    r=subprocess.run([str(H/'obj/Vdecomp_core'),str(p),str(m),str(s)],capture_output=True,text=True)
    if r.returncode:raise RuntimeError((p.name,m,s,r.returncode,r.stdout[-1500:],r.stderr))
    a=[dict(json.loads(line),fixture=str(p)) for line in r.stdout.splitlines()]
    assert len(a)==2
    return a
with ThreadPoolExecutor(max_workers=3) as pool:
    rows=[r for a in pool.map(run,[(p,m,s) for p in cases for m in [14,15,11,12,13,10] for s in [0,1]]) for r in a]
(H/'results.json').write_text('[\n'+',\n'.join(json.dumps(r,separators=(',',':')) for r in rows)+'\n]\n')
summary={}
for m in [14,15,11,12,13,10]:
 for s in [0,1]:
    a=[r for r in rows if r['mode']==m and r['stall']==s and r['command']==0 and Path(r['fixture']).name.startswith('real_')]
    v={k:sum(r[k] for r in a) for k in a[0] if k not in ['mode','stall','command','fixture','state_cycles']}
    v['service_cycles']=v['cycles']+v['configuration_cycles']+len(a)
    summary[f'm{m}_s{s}']=v
(H/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(commands=len(rows),raw_values=sum(r['outputs'] for r in rows),real8={k:dict(core=v['cycles'],service=v['service_cycles'],plane=v['first_issues'],overlap=v['aux_events']) for k,v in summary.items()}),indent=2))
