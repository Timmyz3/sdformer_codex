#!/opt/anaconda3/bin/python3.12
from pathlib import Path
import subprocess,json,csv,time
D=Path(__file__).resolve().parent
cases=['synthetic_zero','synthetic_dense','synthetic_sparse']+[f'real_{i:02d}' for i in range(8)]
kinds=['direct','wino','masked_wino','masked_expanded']
rows=[]
with (D/'rtl_results.jsonl').open('w') as out:
 for case in cases:
  for kind in kinds:
   for stress in (0,1):
    p=subprocess.run([str(D/'obj_dir/Vwinograd_tile'),str(D/'fixtures'/case),kind,str(stress)],cwd=D,text=True,capture_output=True)
    if p.returncode:raise RuntimeError(f'{case} {kind} {stress}: {p.stdout} {p.stderr}')
    row=json.loads(p.stdout);assert row['cycles']==sum(row['state_cycles'])
    rows.append(row);out.write(json.dumps(row)+'\n');out.flush()
    print(case,kind,stress,row['cycles'],flush=True)
(D/'rtl_results.json').write_text(json.dumps(rows,indent=2)+'\n')
fields=[x for x in rows[0] if x!='state_cycles']
with (D/'cycles.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:v for k,v in r.items() if k in fields} for r in rows)
