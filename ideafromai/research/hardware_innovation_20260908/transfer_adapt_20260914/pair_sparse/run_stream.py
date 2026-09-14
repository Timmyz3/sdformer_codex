from pathlib import Path
import json,subprocess
from concurrent.futures import ThreadPoolExecutor
H=Path(__file__).resolve().parent; B=H.parents[1]
with (H/'build_stream.log').open('w') as log:
 subprocess.run(['verilator','-Wall','--cc','--exe','--top-module','decomp_core','--Mdir','obj_stream','decomp_core.sv','stream_tb.cpp','-CFLAGS','-O3'],cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run(['make','-C','obj_stream','-f','Vdecomp_core.mk','-j2'],cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
def run(job):
 m,s=job
 r=subprocess.run([str(H/'obj_stream/Vdecomp_core'),str(B/'fusion_review_followup_20260914/pair_dictionary/fixtures/real_0'),str(m),str(s),str(H/'fixtures/stream/manifest.txt')],capture_output=True,text=True,check=True)
 return [json.loads(l)for l in r.stdout.splitlines()]
rows=[]
with ThreadPoolExecutor(max_workers=3)as pool:
 for a in pool.map(run,[(m,s)for m in [14,15,18,19]for s in [0,1]]):rows+=a
(H/'results_stream.json').write_text(json.dumps(rows,indent=2)+'\n')
summary={}
for m in [14,15,18,19]:
 for s in [0,1]:
  for repeat in [0,1]:
   a=[r for r in rows if r['mode']==m and r['stall']==s and r['command']//64==repeat]
   v={k:sum(r[k]for r in a)for k in a[0]if k not in ['mode','stall','command','state_cycles']}
   v['start_beats']=len(a);v['service_cycles']=v['cycles']+v['configuration_cycles']+len(a)
   summary[f'm{m}_s{s}_repeat{repeat}']=v
(H/'SUMMARY_stream.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:{'core':v['cycles'],'service':v['service_cycles']}for k,v in summary.items()},indent=2))
