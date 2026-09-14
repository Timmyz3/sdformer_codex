from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json,subprocess,argparse
H=Path(__file__).resolve().parent;S=H.parent/'pair_sparse';B=H.parents[1]
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['small','64'],default='small');args=p.parse_args()
if args.stage=='64':
 small=json.loads((H/'stream_checks_small.json').read_text());assert small['passed']
 summary=json.loads((H/'SUMMARY.json').read_text())['real_first_commands'];assert summary['m20_s0']['cycles']<summary['m14_s0']['cycles']
else:
 with (H/'build_stream.log').open('w') as log:
  for cmd in [['verilator','-Wall','--cc','--exe','--top-module','decomp_core','--Mdir','obj_stream','decomp_core.sv','stream_tb.cpp','-CFLAGS','-O3'],['make','-C','obj_stream','-f','Vdecomp_core.mk','-j2']]:subprocess.run(cmd,cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
manifest=H/'small_manifest.txt' if args.stage=='small' else S/'fixtures/stream/manifest.txt'
paths=[Path(line) for line in manifest.read_text().splitlines()];nt=len(paths)
assert nt==(4 if args.stage=='small' else 64)
def run(job):
 m,s=job
 binary=S/'obj_stream/Vdecomp_core' if m==19 else H/'obj_stream/Vdecomp_core'
 params=B/'fusion_review_followup_20260914/pair_dictionary/fixtures/real_0' if m==19 else H/'fixtures/real_0'
 proc=subprocess.run([str(binary),str(params),str(m),str(s),str(manifest)],text=True,capture_output=True)
 if proc.returncode:raise RuntimeError((job,proc.returncode,proc.stdout,proc.stderr))
 a=[dict(json.loads(line),fixture=str(paths[i%nt]),reference=(m==19)) for i,line in enumerate(proc.stdout.splitlines())]
 assert len(a)==nt*2;print('PASS',args.stage,m,s,len(a),flush=True);return a
rows=[]
with ThreadPoolExecutor(max_workers=3) as pool:
 for a in pool.map(run,[(m,s) for m in [14,19,20] for s in [0,1]]):rows+=a
(H/f'results_stream_{args.stage}.json').write_text(json.dumps(rows,indent=2)+'\n')
summary={}
for m in [14,19,20]:
 for s in [0,1]:
  for repeat in [0,1]:
   a=[r for r in rows if r['mode']==m and r['stall']==s and r['command']//nt==repeat]
   v={k:sum(r[k] for r in a) for k in a[0] if k not in ['mode','stall','command','state_cycles','fixture','reference']}
   v['start_beats']=len(a);v['service_cycles']=v['cycles']+v['configuration_cycles']+len(a)
   summary[f'm{m}_s{s}_repeat{repeat}']=v
(H/f'SUMMARY_stream_{args.stage}.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:{'core':v['cycles'],'service':v['service_cycles'],'updates':v['count_checks']} for k,v in summary.items()},indent=2))
