from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json,subprocess
H=Path(__file__).resolve().parent;B=H.parents[1]
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['small','short','held','disjoint'],default='small');p.add_argument('--skip-build',action='store_true');a=p.parse_args()
stream=a.stage!='small';obj='obj_stream' if stream else 'obj';tb='stream_tb.cpp' if stream else 'tb.cpp'
if not a.skip_build:
 with (H/f'build_{a.stage}.log').open('w') as f:
  subprocess.run(['verilator','-Wall','--cc','--exe','--top-module','consumer_stream','--Mdir',obj,'consumer_stream.sv','decomp_core.sv','i24_consumer.sv',tb,'-CFLAGS','-O3 -std=c++14'],cwd=H,stdout=f,stderr=subprocess.STDOUT,check=True)
  subprocess.run(['make','-C',obj,'-f','Vconsumer_stream.mk','-j2'],cwd=H,stdout=f,stderr=subprocess.STDOUT,check=True)
cases=json.loads((H.parent/'count_rr/fixtures.json').read_text());modes=[14,15,13,10];jobs=[]
if not stream:
 for c in cases:
  for mode in modes:
   for stall in [0,1]:jobs.append((c,mode,stall,1,None))
 for name in ['real_6','count_inverse_time_unique','count_mixed_class_255_256']:
  c=next(c for c in cases if c['name']==name)
  for m,n in [(14,10),(10,14),(15,13),(13,15),(13,10),(10,13)]:
   for stall in [0,1]:jobs.append((c,m,stall,1,n))
else:
 ranges={'short':[(159,3),(19197,3)],'held':[(128,64)],'disjoint':[(4000,64)]}[a.stage]
 for first,count in ranges:
  for mode in modes:
   for stall in [0,1]:jobs.append((dict(name=f'native_{first}',tile_id=first,path=str(H.parent/'count_rr/fixtures/real_0')),mode,stall,count,None))
(H/'logs').mkdir(exist_ok=True)
def run(job):
 c,m,s,n,nxt=job;name=f"{a.stage}_{c['name']}_n{n}_m{m}_s{s}"+(f'_to{nxt}' if nxt is not None else '')
 cmd=[str(H/obj/'Vconsumer_stream')]
 if stream:
  D=B/'r8_consumer_fusion_20260914/data';cmd+=[str(D/f) for f in ['first_source_words.npy','identity_fp32_full.npy','raw_p_full.npy','i24_new_full.npy','identity_q20_full.npy']]
  cmd += [c['path'],str(m),str(c['tile_id']),str(n),str(s),'2','16000000']
 else:cmd += [c['path'],str(m),str(s),str(c['tile_id']),str(n)]+([] if nxt is None else [str(nxt)])
 r=subprocess.run(cmd,capture_output=True,text=True);(H/'logs'/f'{name}.jsonl').write_text(r.stdout);(H/'logs'/f'{name}.err').write_text(r.stderr)
 if r.returncode:raise RuntimeError((name,r.returncode,r.stderr,r.stdout[-1600:]))
 return [dict(json.loads(l),fixture=c['name'],first_tile=c['tile_id'],tiles=n,cross_mode=nxt is not None) for l in r.stdout.splitlines()]
rows=[]
with (H/f'results_{a.stage}.jsonl').open('w') as f,ThreadPoolExecutor(max_workers=3) as pool:
 for i,rr in enumerate(pool.map(run,jobs)):
  rows+=rr
  for r in rr:f.write(json.dumps(r,separators=(',',':'))+'\n')
  f.flush()
  if i%20==0 or stream:print(json.dumps({'job':i,'of':len(jobs),'mode':rr[0]['mode'],'first':rr[0]['first_tile'],'cycles':[r['total_cycles'] for r in rr]}),flush=True)
summary=dict(passed=True,commands=len(rows),values_each_stage=sum(r['outputs'] for r in rows))
if stream:summary['results']=[{k:r[k] for k in ['mode','stall','command','first_tile','tiles','total_cycles','core_cycles','consumer_cycles','consumer_join_wait_cycles','core_weight_stalls','core_output_stalls']} for r in rows]
(H/f'SUMMARY_{a.stage}.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
