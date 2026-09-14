from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json,subprocess
H=Path(__file__).resolve().parent;B=H.parents[1]
ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['small','short','held','disjoint'],default='small');ap.add_argument('--skip-build',action='store_true');a=ap.parse_args()
stream=a.stage!='small';obj='obj_stream' if stream else 'obj';tb='stream_tb.cpp' if stream else 'tb.cpp'
if not a.skip_build:
 with (H/f'build_{a.stage}.log').open('w') as f:
  subprocess.run(['verilator','-Wall','--cc','--exe','--top-module','interleave_stream','--Mdir',obj,'interleave_stream.sv','rr_context.sv','i24_consumer.sv','wide_phase_alu.sv',tb,'-CFLAGS','-O3 -std=c++14'],cwd=H,stdout=f,stderr=subprocess.STDOUT,check=True)
  subprocess.run(['make','-C',obj,'-f','Vinterleave_stream.mk','-j2'],cwd=H,stdout=f,stderr=subprocess.STDOUT,check=True)
cases=json.loads((H/'fixtures.json').read_text());modes=[2,4,21,8,7]
jobs=[]
if not stream:
 for c in cases:
  for mode in modes:
   for stall in [0,1]:jobs.append((c,mode,stall,1,None))
 for c in cases:
  if c['name'] in ['one','extreme_1','extreme_-1','count_mixed_class_255_256','count_q2_zero_overlay']:
   for mode in [2,4,21,8,7]:
    for stall in [0,1]:jobs.append((c,mode,stall,2,None))
 for name in ['real_6','count_inverse_time_unique','count_mixed_class_255_256']:
  c=next(c for c in cases if c['name']==name)
  for mode,next_mode in [(2,8),(8,2),(4,8),(8,4),(21,8),(8,21),(2,7),(7,2),(4,7),(7,4),(21,7),(7,21),(8,7),(7,8)]:
   for stall in [0,1]:jobs.append((c,mode,stall,1,next_mode))
else:
 ranges={'short':[(159,3),(19197,3)],'held':[(128,64)],'disjoint':[(4000,64)]}[a.stage]
 for first,count in ranges:
  for mode in modes:
   for stall in [0,1]:jobs.append((dict(name=f'native_{first}',tile_id=first,path=str(H/'fixtures/real_0')),mode,stall,count,None))
(H/'logs').mkdir(exist_ok=True)
def run(job):
 c,mode,stall,count,nxt=job;name=f"{a.stage}_{c['name']}_n{count}_m{mode}_s{stall}"+(f'_to{nxt}' if nxt is not None else '')
 cmd=[str(H/obj/'Vinterleave_stream')]
 if stream:
  D=B/'r8_consumer_fusion_20260914/data'
  cmd += [str(D/f) for f in ['first_source_words.npy','identity_fp32_full.npy','raw_p_full.npy','i24_new_full.npy','identity_q20_full.npy']]
  cmd += [c['path'],str(mode),str(c['tile_id']),str(count),str(stall),'2','12000000']
 else:cmd += [c['path'],str(mode),str(stall),str(c['tile_id']),str(count)]+([] if nxt is None else [str(nxt)])
 r=subprocess.run(cmd,capture_output=True,text=True)
 (H/'logs'/f'{name}.jsonl').write_text(r.stdout);(H/'logs'/f'{name}.err').write_text(r.stderr)
 if r.returncode:raise RuntimeError((name,r.returncode,r.stderr,r.stdout[-2000:]))
 return [dict(json.loads(line),fixture=c['name'],first_tile=c['tile_id'],tiles=count,cross_mode=nxt is not None) for line in r.stdout.splitlines()]
rows=[]
with (H/f'results_{a.stage}.jsonl').open('w') as out,ThreadPoolExecutor(max_workers=3) as pool:
 for i,result in enumerate(pool.map(run,jobs)):
  rows+=result
  for r in result:out.write(json.dumps(r,separators=(',',':'))+'\n')
  out.flush()
  if i%20==0 or stream:print(json.dumps({'job':i,'of':len(jobs),'fixture':result[0]['fixture'],'mode':result[0]['mode'],'cycles':[r['total_cycles'] for r in result]}),flush=True)
summary=dict(passed=True,commands=len(rows),values_each_stage=sum(r['outputs'] for r in rows),results=[{k:r[k] for k in ['fixture','tiles','mode','stall','command','cross_mode','total_cycles','static_words','window_cycles','core_aux_issues','core_metadata_reads','conflict_cycles','core_arbitration_stalls','core_weight_stalls','consumer_join_wait_cycles']} for r in rows])
(H/f'SUMMARY_{a.stage}.json').write_text(json.dumps(summary,separators=(',',':'))+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='results'}),flush=True)
