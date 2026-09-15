from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json,subprocess

H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['small','held','disjoint','sequences','swap'],default='small');p.add_argument('--skip-build',action='store_true');a=p.parse_args()
if not a.skip_build:
 with (H/'build.log').open('w') as log:
  subprocess.run(['verilator','-Wall','--cc','--exe','--top-module','interleave_stream','--Mdir','obj','interleave_stream.sv','rr_context.sv','i24_consumer.sv','wide_phase_alu.sv','tb.cpp','-CFLAGS','-O3 -std=c++14'],cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run(['make','-C','obj','-f','Vinterleave_stream.mk','-j2'],cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
jobs=[(m,s,None) for m in [2,4,21,7] for s in [0,1]]
if a.stage=='swap':jobs += [(a,s,b) for a,b in [(2,7),(7,21),(21,4),(4,7)] for s in [0,1]]
def run(job):
 mode,stall,next_mode=job
 cmd=[str(H/'obj/Vinterleave_stream'),str(H/'parameters'),str(H/f'{a.stage}.txt'),str(mode),str(stall)]
 if next_mode is not None:cmd.append(str(next_mode))
 r=subprocess.run(cmd,capture_output=True,text=True)
 name=f'{a.stage}_m{mode}_s{stall}'+(f'_to{next_mode}' if next_mode is not None else '')
 (H/'logs').mkdir(exist_ok=True)
 (H/'logs'/f'{name}.jsonl').write_text(r.stdout);(H/'logs'/f'{name}.err').write_text(r.stderr)
 if r.returncode:raise RuntimeError((name,r.returncode,r.stderr,r.stdout[-1000:]))
 rows=[dict(json.loads(l),stage=a.stage,initial_mode=mode,next_mode=next_mode) for l in r.stdout.splitlines()]
 print(json.dumps(dict(job=name,cycles=[x['total_cycles'] for x in rows])),flush=True)
 return rows
rows=[]
with ThreadPoolExecutor(max_workers=2) as pool:
 for rr in pool.map(run,jobs):rows+=rr
(H/f'results_{a.stage}.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
(H/f'SUMMARY_{a.stage}.json').write_text(json.dumps(dict(passed=True,commands=len(rows),raw_J_wide_I24_each=sum(r['outputs'] for r in rows)),separators=(',',':'))+'\n')
