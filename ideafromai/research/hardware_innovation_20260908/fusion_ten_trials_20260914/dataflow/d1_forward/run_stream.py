import argparse,json,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
H=Path(__file__).resolve().parent;D=H.parents[2]/'r8_consumer_fusion_20260914/data'
p=argparse.ArgumentParser();p.add_argument('--full',action='store_true');args=p.parse_args()
for f in ['first_source_words.npy','identity_fp32_full.npy','raw_p_full.npy','i24_new_full.npy','identity_q20_full.npy']:
 if not (D/f).exists():raise FileNotFoundError(D/f)
subprocess.run(['verilator','-Wall','--cc','--exe','--top-module','consumer_stream','--Mdir','obj_stream_fp','consumer_stream.sv','i24_consumer.sv','forward_core.sv','stream_tb.cpp','-CFLAGS','-O3'],cwd=H,check=True)
with (H/'build_stream.log').open('w') as f:subprocess.run(['make','-C','obj_stream_fp','-f','Vconsumer_stream.mk','-j4'],cwd=H,stdout=f,stderr=subprocess.STDOUT,check=True)
first,count,repeats=(0,19200,1) if args.full else (128,64,2)
jobs=[(m,s) for m in [1,2] for s in ([0] if args.full else [0,1])]
name='full' if args.full else '64';(H/'logs').mkdir(exist_ok=True)
def run(job):
 m,s=job
 cmd=[str(H/'obj_stream_fp/Vconsumer_stream')]+[str(D/f) for f in ['first_source_words.npy','identity_fp32_full.npy','raw_p_full.npy','i24_new_full.npy','identity_q20_full.npy']]+[str(H/'fixtures/real_0'),str(m),str(first),str(count),str(s),str(repeats),'6000000000']
 t=time.monotonic()
 with (H/'logs'/f'{name}_m{m}_s{s}.err').open('w') as err:
  r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=err,text=True)
 if r.returncode:raise RuntimeError((job,r.returncode,r.stdout))
 (H/'logs'/f'{name}_m{m}_s{s}.jsonl').write_text(r.stdout)
 out=[dict(json.loads(l),identity_input="IEEE_binary32",job_wall_seconds=time.monotonic()-t) for l in r.stdout.splitlines()]
 print(json.dumps({'completed':job,'wall':time.monotonic()-t,'total_cycles':[x['total_cycles'] for x in out]}),flush=True)
 return out
start=time.monotonic();rows=[]
with ThreadPoolExecutor(max_workers=4) as p:
 for result in p.map(run,jobs):rows.extend(result)
(H/f'results_{name}.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps({'jobs':len(rows),'outputs':sum(r['outputs'] for r in rows),'raw_outputs':sum(r['raw_outputs'] for r in rows),'suite_wall_seconds':time.monotonic()-start}),flush=True)
