from pathlib import Path
import subprocess,json,concurrent.futures
H=Path(__file__).resolve().parent
with (H/'build.log').open('w') as log:
 for cmd in [['verilator','-Wall','--cc','--exe','--top-module','consumer_stream','consumer_stream.sv','kron_core.sv','i24_consumer.sv','tb.cpp','-CFLAGS','-O3'],['make','-C','obj_dir','-f','Vconsumer_stream.mk','-j2']]:subprocess.run(cmd,cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
def run(t):
 f,m,s=t;p=subprocess.run([str(H/'obj_dir/Vconsumer_stream'),str(H/'fixtures'/f['name']),str(m),str(s),str(f['tile_id'])],capture_output=True,text=True)
 if p.returncode:raise RuntimeError((f,m,s,p.returncode,p.stdout,p.stderr))
 return [dict(json.loads(l),fixture=f['name']) for l in p.stdout.splitlines()]
fixtures=json.loads((H/'fixtures.json').read_text())
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=sum(pool.map(run,[(f,m,s) for f in fixtures for m in [0,1] for s in [0,1]]),[])
(H/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
fields=[k for k,v in rows[0].items() if isinstance(v,int) and k not in ['mode','stall','command']]
real={str(m):{str(s):{str(c):{k:sum(r[k] for r in rows if r['fixture'].startswith('real') and r['mode']==m and r['stall']==s and r['command']==c) for k in fields} for c in [0,1]} for s in [0,1]} for m in [0,1]}
summary=dict(complete=True,commands=len(rows),raw_values=sum(r['raw_outputs'] for r in rows),J_values=sum(r['J_outputs'] for r in rows),I24_values=sum(r['outputs'] for r in rows),real=real)
(H/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(dict(commands=len(rows),cold={m:real[m]['0']['0']['total_cycles'] for m in real},warm={m:real[m]['0']['1']['total_cycles'] for m in real}))
