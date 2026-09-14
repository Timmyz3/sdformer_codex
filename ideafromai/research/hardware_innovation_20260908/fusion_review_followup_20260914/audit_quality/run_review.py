from pathlib import Path
import shutil,subprocess,json,concurrent.futures
B=Path(__file__).resolve().parents[2]
O=B/'fusion_ten_trials_20260914/algorithm_sparse';H=B/'fusion_review_followup_20260914/audit_quality'
for f in ['lossy_tile.sv','lossy_r8.sv','i24_consumer.sv','tb.cpp']:shutil.copy2(O/f,H/f)
with (H/'build.log').open('w') as log:
 for cmd in [['verilator','-Wall','--cc','--exe','--top-module','lossy_tile','lossy_tile.sv','lossy_r8.sv','i24_consumer.sv','tb.cpp','-CFLAGS','-O3'],['make','-C','obj_dir','-f','Vlossy_tile.mk','-j2']]:subprocess.run(cmd,cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
def one(t):
 name,m,s=t;p=subprocess.run([str(H/'obj_dir/Vlossy_tile'),str(O/'fixtures'/name),str(O/'constants'),str(m),str(s)],capture_output=True,text=True,check=True)
 return [dict(json.loads(l),fixture=name) for l in p.stdout.splitlines()]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=sum(pool.map(one,[(n,m,s) for n in ['real_6','zero','poison_corner'] for m in range(9) for s in [0,1]]),[])
old=json.loads((O/'results.json').read_text());exact=0
for r in rows:
 p=next(o for o in old if all(o[k]==r[k] for k in ['fixture','mode','stall','command']))
 for k,v in p.items():
  if isinstance(v,int):assert r[k]==v,(r['fixture'],r['mode'],k,v,r[k])
 exact+=1
(H/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
s=dict(commands=len(rows),raw_values=sum(r['outputs'] for r in rows),J_values=sum(r['outputs'] for r in rows),I24_values=sum(r['outputs'] for r in rows),all_integer_counters_match=exact)
(H/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print(s)
