from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json,subprocess
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['small','short','held','disjoint','sequences'],default='small');p.add_argument('--skip-build',action='store_true');a=p.parse_args()
if not a.skip_build:
 with (H/'build.log').open('w') as f:
  subprocess.run(['verilator','-Wall','--cc','--exe','--top-module','spatial_core','--Mdir','obj','spatial_core.sv','tb.cpp','-CFLAGS','-O3 -std=c++14'],cwd=H,stdout=f,stderr=subprocess.STDOUT,check=True)
  subprocess.run(['make','-C','obj','-f','Vspatial_core.mk','-j2'],cwd=H,stdout=f,stderr=subprocess.STDOUT,check=True)
def run(job):
 mode,stall=job
 r=subprocess.run([str(H/'obj/Vspatial_core'),str(H),str(H/f'{a.stage}.txt'),str(stall),str(mode)],capture_output=True,text=True)
 (H/f'raw_{a.stage}_m{mode}_s{stall}.jsonl').write_text(r.stdout);(H/f'raw_{a.stage}_m{mode}_s{stall}.err').write_text(r.stderr)
 if r.returncode:raise RuntimeError((job,r.returncode,r.stderr,r.stdout[-1500:]))
 rows=[json.loads(l) for l in r.stdout.splitlines()]
 return rows
rows=[]
with ThreadPoolExecutor(max_workers=2) as pool:
 for rr in pool.map(run,[(m,s) for m in [0,1] for s in [0,1]]):
  rows+=rr;print(json.dumps(dict(mode=rr[0]['mode'],stall=rr[0]['stall'],commands=len(rr),cycles=sum(r['cycles'] for r in rr))),flush=True)
summary=dict(passed=True,commands=len(rows),raw_values=sum(r['outputs'] for r in rows))
(H/f'SUMMARY_raw_{a.stage}.json').write_text(json.dumps(summary)+'\n');print(json.dumps(summary),flush=True)
