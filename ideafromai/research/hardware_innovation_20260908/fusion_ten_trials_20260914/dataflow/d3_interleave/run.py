import json,subprocess,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
H=Path(__file__).resolve().parent
subprocess.run(['verilator','-Wall','--cc','--exe','--top-module','interleave_stream','--Mdir','obj_unit','interleave_stream.sv','thread_context.sv','i24_consumer.sv','tb.cpp','-CFLAGS','-O3'],cwd=H,check=True)
with (H/'build.log').open('w') as f:subprocess.run(['make','-C','obj_unit','-f','Vinterleave_stream.mk','-j4'],cwd=H,stdout=f,stderr=subprocess.STDOUT,check=True)
cases=json.loads((H/'fixtures.json').read_text());jobs=[(c,m,s) for c in cases for m in [0,1,2] for s in [0,1]]
def run(job):
 c,m,s=job;cmd=[str(H/'obj_unit/Vinterleave_stream'),str(H/'fixtures'/c['name']),str(m),str(s),str(c['tile_id'])]
 r=subprocess.run(cmd,capture_output=True,text=True)
 if r.returncode:raise RuntimeError((cmd,r.returncode,r.stdout,r.stderr))
 return [dict(json.loads(l),fixture=c['name'],identity_input='IEEE_binary32') for l in r.stdout.splitlines()]
t=time.monotonic();rows=[]
with ThreadPoolExecutor(max_workers=4) as p:
 for x in p.map(run,jobs):rows.extend(x)
(H/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps({'runs':len(rows),'outputs_each':sum(x['outputs'] for x in rows),'wall_seconds':time.monotonic()-t}))
