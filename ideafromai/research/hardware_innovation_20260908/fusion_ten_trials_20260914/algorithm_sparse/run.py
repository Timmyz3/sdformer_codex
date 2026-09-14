from pathlib import Path
import subprocess,json,concurrent.futures
H=Path(__file__).resolve().parent
with (H/'build.log').open('w') as log:
 for cmd in [['verilator','-Wall','--cc','--exe','--top-module','lossy_tile','lossy_tile.sv','lossy_r8.sv','i24_consumer.sv','tb.cpp','-CFLAGS','-O3'],['make','-C','obj_dir','-f','Vlossy_tile.mk','-j2']]:subprocess.run(cmd,cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
fixtures=json.loads((H/'definition.json').read_text())['fixtures']
def run(task):
 name,mode,stall=task;p=subprocess.run([str(H/'obj_dir/Vlossy_tile'),str(H/'fixtures'/name),str(H/'constants'),str(mode),str(stall)],capture_output=True,text=True)
 if p.returncode:raise RuntimeError((task,p.returncode,p.stdout,p.stderr))
 rows=[dict(json.loads(line),fixture=name) for line in p.stdout.splitlines()];print('PASS',task,rows[0]['cycles'],flush=True);return rows
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 for rr in pool.map(run,[(x['name'],m,s) for x in fixtures for m in range(9) for s in [0,1]]):rows.extend(rr)
(H/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
fields=[k for k,v in rows[0].items() if isinstance(v,int) and k not in ['mode','stall','command']]
real={str(m):{str(s):{k:sum(r[k] for r in rows if r['fixture'].startswith('real') and r['mode']==m and r['stall']==s and r['command']==0) for k in fields} for s in [0,1]} for m in range(9)}
summary=dict(complete=True,runs=len(rows),raw_values=sum(r['outputs'] for r in rows),J_values=sum(r['outputs'] for r in rows),I24_values=sum(r['outputs'] for r in rows),real=real,scope='8 actual heldout tiles, completeK864C96N96T10 with actualFP32identity and I24 consumer; three lossyfunctions plus strongcontrols; no fullframe/PPA',modes={0:'exact',1:'groupdrop',2:'rankdrop',3:'K4prototype_residual',4:'zeroproto_rank',5:'temporalgroup_fullrefresh',6:'temporalrank_fullrefresh',7:'temporalrank_signed13delta',8:'temporalgroup_signed13delta'})
(H/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))
