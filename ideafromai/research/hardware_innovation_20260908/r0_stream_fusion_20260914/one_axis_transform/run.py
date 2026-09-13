from pathlib import Path
import concurrent.futures,json,subprocess,time
H=Path(__file__).resolve().parent
with (H/'build.log').open('w') as log:
 for cmd in (['verilator','--cc','--exe','-Wall','-CFLAGS','-O3','--top-module','one_axis_tile','--Mdir',str(H/'obj_dir'),str(H/'one_axis_tile.sv'),str(H/'tb.cpp')],['make','-C',str(H/'obj_dir'),'-f','Vone_axis_tile.mk','-j2']):subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
fixtures=list(json.loads((H/'fixture_summary.json').read_text()))
def run(task):
 case,kind,stress=task;t=time.monotonic()
 p=subprocess.run([str(H/'obj_dir/Vone_axis_tile'),str(H/'fixtures'/case),kind,str(stress)],check=True,text=True,capture_output=True)
 r=json.loads(p.stdout);r['wall_seconds']=time.monotonic()-t;assert r['cycles']==sum(r['state_cycles']);print('PASS',case,kind,stress,r['cycles'],flush=True);return r
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(run,[(f,k,s) for f in fixtures for k in ('direct','one_axis') for s in (0,1)]))
(H/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
print('COMPLETE',len(rows),sum(r['values_checked'] for r in rows))
