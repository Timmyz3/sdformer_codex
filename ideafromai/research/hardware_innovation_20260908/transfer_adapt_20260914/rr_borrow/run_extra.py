"""Reproduce cross-mode/cross-row, same-module controls and the contended pair."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
import subprocess
import json

parser=argparse.ArgumentParser()
parser.add_argument('--adapt',action='store_true')
args=parser.parse_args()
ROOT=Path(__file__).resolve().parent
H=ROOT/'adapt_rr' if args.adapt else ROOT
B=ROOT.parents[1]
D=B/'r8_consumer_fusion_20260914/data'
P=B/'fusion_ten_trials_20260914/phase_borrow/fixtures/real_0'
base=[str(H/'obj_stream/Vinterleave_stream'),*[str(D/(n+'.npy')) for n in ['first_source_words','identity_fp32_full','raw_p_full','i24_new_full','identity_q20_full']],str(P)]

def run(job):
    first,count,mode,nxt,stall=job
    r=subprocess.run(base+[str(mode),str(first),str(count),str(stall),'2','10000000',str(nxt)],capture_output=True,text=True)
    (H/f'logs/extra_{first}_{mode}_{nxt}_{stall}.log').write_text(r.stdout+r.stderr)
    assert r.returncode==0,(job,r.returncode,r.stdout,r.stderr)
    return [json.loads(l) for l in r.stdout.splitlines()]

transitions=[(1,4),(4,3),(2,4),(4,2)] if args.adapt else [(1,3),(3,2),(2,3),(3,1)]
jobs=[('results_cross.json',[(f,3,m,n,1) for f in [159,19197] for m,n in transitions])]
if args.adapt:
    jobs.append(('results_controls.json',[(128,64,m,m,s) for m in [1,2,3] for s in [0,1]]))
for name, tasks in jobs:
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows=[r for part in pool.map(run,tasks) for r in part]
    (H/name).write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(file=name,passed=True,commands=len(rows))),flush=True)
if args.adapt:
    rows=[]
    for m in [1,2,3,4]:
        for s in [0,1]:
            r=subprocess.run([str(H/'audit_rtl/obj/Vinterleave_stream'),str(H/'pair_fixture'),str(m),str(s),'6460','2'],capture_output=True,text=True)
            (H/f'audit_rtl/pair_m{m}_s{s}.log').write_text(r.stdout+r.stderr)
            assert r.returncode==0,(m,s,r.returncode,r.stdout,r.stderr)
            rows.extend(dict(json.loads(l),fixture='pair_fixture',first_tile=6460,tiles=2) for l in r.stdout.splitlines() if l.startswith('{'))
    (H/'pair_results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(file='pair_results.json',passed=True,commands=len(rows))),flush=True)
