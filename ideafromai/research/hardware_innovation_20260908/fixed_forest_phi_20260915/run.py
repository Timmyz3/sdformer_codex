from pathlib import Path
import sys,subprocess,json,csv
H=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(H/'prepare_runs.py')],check=True)
with (H/'run.log').open('w') as log:
 for cmd in [['verilator','--cc','--exe','--top-module','forest_eval','-Wno-fatal','-CFLAGS','-std=c++17 -O2','forest_eval.sv','tb.cpp'],
             ['make','-C','obj_dir','-f','Vforest_eval.mk','-j2'],
             [str(H/'obj_dir/Vforest_eval'),str(H/'runs.bin'),str(H/'results.tsv')]]:
  r=subprocess.run(cmd,cwd=H,stdout=log,stderr=subprocess.STDOUT);print(cmd[0],r.returncode,flush=True)
  if r.returncode:sys.exit(r.returncode)
rows=[{k:v if k=='case' else int(v) for k,v in r.items()} for r in csv.DictReader((H/'results.tsv').open(),delimiter='\t')]
result=[]
for kind in [0,1,2]:
 for held in [0,1]:
  for lazy in [0,1]:
   for bp in [0,1]:
    group=[r for r in rows if (r['kind'],r['held'],r['lazy'],r['bp'])==(kind,held,lazy,bp)]
    if group:
     result.append(dict(kind=kind,held=held,lazy=lazy,bp=bp,modes={str(m):{k:sum(r[k] for r in group if r['mode']==m) for k in ['cycles','configuration','mem_beats','center_beats','pwp_beats','query','alu','fields','build','parent_reads','joins','negative_fields','pwp_rows','mem_wait','out_wait']} for m in range(8)}))
(H/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print((H/'run.log').read_text().splitlines()[-1])
for r in result:
 if r['held']:print(r['kind'],r['lazy'],r['bp'],{m:v['cycles'] for m,v in r['modes'].items()})
