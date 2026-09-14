from pathlib import Path
import argparse,json,subprocess
H=Path(__file__).resolve().parent;A=H.parent/'spatial_r16_rtl'
ap=argparse.ArgumentParser();ap.add_argument('--function',choices=['q11','q13'],default='q11');ap.add_argument('--streams',action='store_true');ap.add_argument('--reuse-build',action='store_true');ap.add_argument('--sets',nargs='+');o=ap.parse_args()
D=H/'q11' if o.function=='q11' else H
if not o.streams:
 if o.function=='q11':subprocess.run(['/opt/anaconda3/bin/python3.12','-B',str(D/'prepare.py')],check=True)
 if not o.reuse_build:
  subprocess.run(['/opt/anaconda3/bin/python3.12','-B',str(H/'implement_consumer.py')],check=True)
  with (H/'build_consumer.log').open('w') as out:
   subprocess.run(['verilator','--cc','--exe','--top-module','os_stream','-Wno-fatal','--Mdir',str(H/'obj_stream'),str(H/'os_core.sv'),str(H/'os_stream.sv'),str(A/'i24_consumer.sv'),str(A/'wide_phase_alu.sv'),str(H/'stream_tb.cpp'),'-CFLAGS','-O2'],stdout=out,stderr=subprocess.STDOUT,check=True,cwd=H)
   subprocess.run(['make','-C',str(H/'obj_stream'),'-f','Vos_stream.mk','-j4'],stdout=out,stderr=subprocess.STDOUT,check=True,cwd=H)
for name in (o.sets or (['held','disjoint'] if o.streams else ['small'])):
 for stall in [0,1]:
  p=D/f'consumer_{name}_{stall}.jsonl'
  with p.open('w') as out:subprocess.run([str(H/'obj_stream/Vos_stream'),str(D),str(D/f'{name}.txt'),str(stall)],stdout=out,check=True,cwd=D)
  rows=[json.loads(v) for v in p.read_text().splitlines()]
  print(json.dumps(dict(function=o.function,set=name,stall=stall,commands=len(rows),i24_values=sum(v['i24_values'] for v in rows),cycles=sum(v['c_cycles'] for v in rows))),flush=True)
