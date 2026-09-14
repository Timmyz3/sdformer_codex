from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,subprocess,argparse
H=Path(__file__).resolve().parent;O=H.parent;N=O.parent
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['small','short','held','disjoint'],default='small');args=p.parse_args()
stage=args.stage
if stage in ['held','disjoint']:assert json.loads((H/'checks_short.json').read_text())['passed']
obj='obj' if stage=='small' else 'obj_stream'
if stage in ['small','short']:
 with (H/f'build_{stage}.log').open('w') as log:
  for cmd in [['verilator','-Wall','--cc','--exe','--top-module','decomp_core','--Mdir',obj,'decomp_core.sv','tb.cpp' if stage=='small' else 'stream_tb.cpp','-CFLAGS','-O3'],['make','-C',obj,'-f','Vdecomp_core.mk','-j2']]:subprocess.run(cmd,cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
if stage=='small':
 fixtures=[O/'fixtures'/r['name'] for r in json.loads((O/'fixtures.json').read_text())]
 fixtures+=[H/'fixtures/t_unique']
 jobs=[(m,s,d) for d in fixtures for m in [14,20,21] for s in [0,1]]
else:
 manifest={'short':O/'small_manifest.txt','held':N/'pair_sparse/fixtures/stream/manifest.txt','disjoint':N/'audit/fixtures/disjoint_4000/manifest.txt'}[stage]
 paths=[Path(x) for x in manifest.read_text().splitlines()]
 jobs=[(m,s,O/'fixtures/real_0') for m in [14,20,21] for s in [0,1]]
def run(job):
 m,s,d=job;cmd=[str(H/obj/'Vdecomp_core'),str(d),str(m),str(s)]
 if stage!='small':cmd.append(str(manifest))
 proc=subprocess.run(cmd,text=True,capture_output=True)
 if proc.returncode:raise RuntimeError((job,proc.returncode,proc.stdout,proc.stderr))
 a=[dict(json.loads(line),fixture=str(d if stage=='small' else paths[i%len(paths)])) for i,line in enumerate(proc.stdout.splitlines())]
 assert len(a)==(2 if stage=='small' else 2*len(paths))
 print('PASS',stage,m,s,d.name,len(a),flush=True);return a
rows=[]
with ThreadPoolExecutor(max_workers=3) as pool:
 for a in pool.map(run,jobs):rows+=a
(H/f'results_{stage}.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(dict(commands=len(rows),raw_values=sum(r['outputs'] for r in rows))))
