from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,subprocess
H=Path(__file__).resolve().parent;S=H.parent/'pair_sparse'
with (H/'build.log').open('w') as log:
    for cmd in [['verilator','-Wall','--cc','--exe','--top-module','decomp_core','--Mdir','obj','decomp_core.sv','tb.cpp','-CFLAGS','-O3'],['make','-C','obj','-f','Vdecomp_core.mk','-j2']]:subprocess.run(cmd,cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
cases=json.loads((H/'fixtures.json').read_text())
def run(job):
    c,m,stall,reference=job
    binary=S/'obj/Vdecomp_core' if reference else H/'obj/Vdecomp_core'
    d=c['reference_dir'] if reference else str(H/'fixtures'/c['name'])
    p=subprocess.run([str(binary),d,str(m),str(stall)],text=True,capture_output=True)
    if p.returncode:raise RuntimeError((job,p.returncode,p.stdout,p.stderr))
    a=[dict(json.loads(line),fixture=c['name'],reference=reference) for line in p.stdout.splitlines()]
    print('PASS',c['name'],m,stall,reference,[r['cycles'] for r in a],flush=True);return a
jobs=[(c,m,stall,False) for c in cases for m in [14,20] for stall in [0,1]]
# Executable old19 and unchanged old metadata, not old19 on incompatible count8 classes.
jobs += [(c,19,stall,True) for c in cases if c['reference_dir'] for stall in [0,1]]
rows=[]
with ThreadPoolExecutor(max_workers=3) as pool:
    for a in pool.map(run,jobs):rows+=a
(H/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
summary={}
for m in [14,19,20]:
 for stall in [0,1]:
  a=[r for r in rows if r['fixture'].startswith('real_') and r['mode']==m and r['stall']==stall and r['command']==0]
  v={k:sum(r[k] for r in a) for k in a[0] if k not in ['fixture','mode','stall','command','state_cycles','reference']}
  v['start_beats']=len(a);v['service_cycles']=v['cycles']+v['configuration_cycles']+len(a)
  summary[f'm{m}_s{stall}']=v
(H/'SUMMARY.json').write_text(json.dumps(dict(commands=len(rows),outputs_compared=sum(r['outputs'] for r in rows),real_first_commands=summary),indent=2)+'\n')
print(json.dumps({k:{'core':v['cycles'],'service':v['service_cycles'],'updates':v['count_checks'],'retire':v['aux_events']} for k,v in summary.items()},indent=2))
