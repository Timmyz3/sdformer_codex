from pathlib import Path
import argparse,json,subprocess
H=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--streams',action='store_true');o=ap.parse_args()
if not o.streams:
 subprocess.run(['/opt/anaconda3/bin/python3.12','-B',str(H/'prepare_os.py')],check=True)
 with (H/'build_os.log').open('w') as out:
  subprocess.run(['verilator','--cc','--exe','--top-module','os_core','-Wno-fatal','--Mdir',str(H/'obj_os'),str(H/'os_core.sv'),str(H/'tb.cpp'),'-CFLAGS','-O2 -DOS_ARM'],stdout=out,stderr=subprocess.STDOUT,check=True,cwd=H)
  subprocess.run(['make','-C',str(H/'obj_os'),'-f','Vos_core.mk','-j4'],stdout=out,stderr=subprocess.STDOUT,check=True,cwd=H)
for name in (['held','disjoint'] if o.streams else ['small']):
 for stall in [0,1]:
  p=H/f'os_results_{name}_{stall}.jsonl'
  with p.open('w') as out:subprocess.run([str(H/'obj_os/Vos_core'),str(H),str(H/f'{name}.txt'),str(stall)],stdout=out,check=True,cwd=H)
  rows=[json.loads(v) for v in p.read_text().splitlines()]
  print(json.dumps(dict(set=name,stall=stall,commands=len(rows),outputs=sum(v['outputs'] for v in rows),core_cycles=sum(v['cycles'] for v in rows))),flush=True)
