from pathlib import Path
import argparse,json,subprocess
H=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--streams',action='store_true');o=ap.parse_args()
if not o.streams:
 subprocess.run(['/opt/anaconda3/bin/python3.12','-B',str(H/'prepare.py')],check=True)
 with (H/'build.log').open('w') as out:
  subprocess.run(['verilator','--cc','--exe','--top-module','direct_core','-Wno-fatal','--Mdir',str(H/'obj'),str(H/'direct_core.sv'),str(H/'tb.cpp'),'-CFLAGS','-O2'],stdout=out,stderr=subprocess.STDOUT,check=True,cwd=H)
  subprocess.run(['make','-C',str(H/'obj'),'-f','Vdirect_core.mk','-j4'],stdout=out,stderr=subprocess.STDOUT,check=True,cwd=H)
sets=['held','disjoint'] if o.streams else ['small']
for name in sets:
 for stall in [0,1]:
  p=H/f'results_{name}_{stall}.jsonl'
  with p.open('w') as out:subprocess.run([str(H/'obj/Vdirect_core'),str(H),str(H/f'{name}.txt'),str(stall)],stdout=out,check=True,cwd=H)
  rows=[json.loads(v) for v in p.read_text().splitlines()]
  print(json.dumps(dict(set=name,stall=stall,commands=len(rows),outputs=sum(v['outputs'] for v in rows),core_cycles=sum(v['cycles'] for v in rows))),flush=True)
