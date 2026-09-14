from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import subprocess

H = Path(__file__).resolve().parent
LOCKED = H.parent
p = argparse.ArgumentParser()
p.add_argument('--kind',choices=['raw','consumer'],required=True)
p.add_argument('--arm',choices=['moment','native_tap','both'],default='both')
p.add_argument('--stage',choices=['small','held','disjoint','sequences'],default='small')
p.add_argument('--skip-build',action='store_true')
a = p.parse_args()
top = 'spatial_core' if a.kind == 'raw' else 'spatial_stream'
obj = H / f'obj_{a.kind}'
if not a.skip_build:
    sources = ['spatial_core.sv','tb.cpp'] if a.kind == 'raw' else ['spatial_stream.sv','spatial_core.sv','i24_consumer.sv','wide_phase_alu.sv','stream_tb.cpp']
    with (H/f'build_{a.kind}.log').open('w') as log:
        subprocess.run(['verilator','-Wall','--cc','--exe','--top-module',top,'--Mdir',str(obj),
                        *[str(LOCKED/name) for name in sources],'-CFLAGS','-O3 -std=c++14'],
                       cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run(['make','-C',str(obj),'-f',f'V{top}.mk','-j2'],cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
arms = ['moment','native_tap'] if a.arm == 'both' else [a.arm]
def run(job):
    arm,mode,stall=job
    root=H/arm
    result=subprocess.run([str(obj/f'V{top}'),str(root),str(root/f'{a.stage}.txt'),str(stall),str(mode)],capture_output=True,text=True)
    prefix=root/f'{a.kind}_{a.stage}_m{mode}_s{stall}'
    prefix.with_suffix('.jsonl').write_text(result.stdout)
    prefix.with_suffix('.err').write_text(result.stderr)
    if result.returncode:
        raise RuntimeError((job,result.returncode,result.stderr,result.stdout[-1000:]))
    rows=[json.loads(line) for line in result.stdout.splitlines()]
    report=dict(arm=arm,kind=a.kind,stage=a.stage,mode=mode,stall=stall,commands=len(rows),
                cycles=sum(r['c_cycles' if a.kind=='consumer' else 'cycles'] for r in rows))
    print(json.dumps(report),flush=True)
    return report
with ThreadPoolExecutor(max_workers=2) as pool:
    reports=list(pool.map(run,[(arm,m,s) for arm in arms for m in [0,1] for s in [0,1]]))
(H/f'SUMMARY_{a.kind}_{a.stage}.json').write_text(json.dumps(dict(passed=True,runs=reports),separators=(',',':'))+'\n')
