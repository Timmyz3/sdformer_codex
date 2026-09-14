"""No-reset source-set/context reuse, using original native source and full gold."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,subprocess
H=Path(__file__).resolve().parent;B=H.parents[1];D=B/'r8_consumer_fusion_20260914/data'
jobs=[]
for mode in [2,4,21,8,7]:
 for first,second in [(128,4000),(4000,128)]:
  for stall in [0,1]:jobs.append((mode,mode,first,second,stall))
for mode,nextmode in [(8,7),(7,8),(21,7),(7,21)]:
 for stall in [0,1]:jobs.append((mode,nextmode,128,4000,stall))
def run(job):
 mode,nextmode,first,second,stall=job
 name=f'swap_m{mode}_to{nextmode}_{first}_to{second}_s{stall}'
 cmd=[str(H/'obj_stream/Vinterleave_stream')]+[str(D/f) for f in ['first_source_words.npy','identity_fp32_full.npy','raw_p_full.npy','i24_new_full.npy','identity_q20_full.npy']]
 cmd += [str(H/'fixtures/real_0'),str(mode),str(first),'2',str(stall),'2','12000000',str(nextmode),str(second)]
 r=subprocess.run(cmd,capture_output=True,text=True)
 (H/'logs'/f'{name}.jsonl').write_text(r.stdout);(H/'logs'/f'{name}.err').write_text(r.stderr)
 if r.returncode:raise RuntimeError((name,r.returncode,r.stderr,r.stdout[-1200:]))
 rows=[]
 for l in r.stdout.splitlines():
  x=json.loads(l);f=second if x['command'] else first
  rows.append(dict(x,fixture=f'native_{f}',first_tile=f,tiles=2,cross_mode=mode!=nextmode,cross_source=True))
 return rows
rows=[]
with (H/'results_swap.jsonl').open('w') as f,ThreadPoolExecutor(max_workers=3) as pool:
 for i,rr in enumerate(pool.map(run,jobs)):
  rows+=rr
  for r in rr:f.write(json.dumps(r,separators=(',',':'))+'\n')
  f.flush()
  print(json.dumps(dict(job=i,of=len(jobs),mode=rr[0]['mode'],cycles=[r['total_cycles'] for r in rr])),flush=True)
summary=dict(passed=True,commands=len(rows),values_each_stage=sum(r['outputs'] for r in rows))
(H/'SUMMARY_swap.json').write_text(json.dumps(summary)+'\n');print(json.dumps(summary))
