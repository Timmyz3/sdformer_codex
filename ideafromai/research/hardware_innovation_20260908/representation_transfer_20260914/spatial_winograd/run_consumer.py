from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json,subprocess
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['small','short','held','disjoint','sequences'],default='small');p.add_argument('--skip-build',action='store_true');a=p.parse_args()
if not a.skip_build:
 with (H/'build_consumer.log').open('w') as f:
  subprocess.run(['verilator','-Wall','--cc','--exe','--top-module','spatial_stream','--Mdir','obj_stream','spatial_stream.sv','spatial_core.sv','i24_consumer.sv','wide_phase_alu.sv','stream_tb.cpp','-CFLAGS','-O3 -std=c++14'],cwd=H,stdout=f,stderr=subprocess.STDOUT,check=True)
  subprocess.run(['make','-C','obj_stream','-f','Vspatial_stream.mk','-j2'],cwd=H,stdout=f,stderr=subprocess.STDOUT,check=True)
def run(job):
 mode,stall=job
 r=subprocess.run([str(H/'obj_stream/Vspatial_stream'),str(H),str(H/f'{a.stage}.txt'),str(stall),str(mode)],capture_output=True,text=True)
 (H/f'consumer_{a.stage}_m{mode}_s{stall}.jsonl').write_text(r.stdout);(H/f'consumer_{a.stage}_m{mode}_s{stall}.err').write_text(r.stderr)
 if r.returncode:raise RuntimeError((job,r.returncode,r.stderr,r.stdout[-1500:]))
 return [json.loads(l) for l in r.stdout.splitlines()]
rows=[]
with ThreadPoolExecutor(max_workers=2) as pool:
 for rr in pool.map(run,[(m,s) for m in [0,1] for s in [0,1]]):
  rows+=rr;print(json.dumps(dict(mode=rr[0]['mode'],stall=rr[0]['stall'],commands=len(rr),consumer_cycles=sum(r['c_cycles'] for r in rr))),flush=True)
summary=dict(passed=True,commands=len(rows),raw_J20_wide_I24_each=sum(r['raw_values'] for r in rows))
(H/f'SUMMARY_consumer_{a.stage}.json').write_text(json.dumps(summary)+'\n');print(json.dumps(summary),flush=True)
