import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE=Path(__file__).resolve().parent
subprocess.run(['verilator','-Wall','--cc','--exe','--top-module','integer_factor','integer_factor.sv','tb.cpp','-CFLAGS','-O3'],cwd=HERE,check=True)
with (HERE/'build.log').open('w') as log:
    subprocess.run(['make','-C','obj_dir','-f','Vinteger_factor.mk','-j4'],cwd=HERE,stdout=log,stderr=subprocess.STDOUT,check=True)
cases=json.loads((HERE/'definition.json').read_text())['fixtures']
jobs=[(c['name'],m,s) for c in cases for m in [6,7,8] for s in [0,1]]
def run(job):
    name,m,s=job
    done=subprocess.run([str(HERE/'obj_dir/Vinteger_factor'),str(HERE/'fixtures'/name),str(m),str(s)],capture_output=True,text=True)
    if done.returncode: raise RuntimeError((job,done.returncode,done.stdout,done.stderr))
    return [dict(json.loads(line),fixture=name) for line in done.stdout.splitlines()]
rows=[]
with ThreadPoolExecutor(max_workers=4) as pool:
    for result in pool.map(run,jobs):rows.extend(result)
(HERE/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
summary={'runs':len(rows),'outputs_compared':sum(r['outputs'] for r in rows),'real':{}}
for m in [6,7,8]:
    summary['real'][str(m)]={}
    for s in [0,1]:
        x=[r for r in rows if r['mode']==m and r['stall']==s and r['command']==0 and r['fixture'].startswith('real_')]
        fields=['cycles','source_words','weight_words','psum_reads','psum_writes','z_reads','z_writes','first_issues','mac_issues','second_weight_words','source_stalls','weight_stalls','output_stalls']
        summary['real'][str(m)][str(s)]={k:sum(r[k] for r in x) for k in fields}
        summary['real'][str(m)][str(s)]['fresh_source_origin_cycles']=sum(r['cycles'] for r in x)+1537*len(x)
(HERE/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
