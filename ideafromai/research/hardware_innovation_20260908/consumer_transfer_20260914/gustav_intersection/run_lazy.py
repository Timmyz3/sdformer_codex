from pathlib import Path
import subprocess,sys,csv,json
H=Path(__file__).resolve().parent;B=H.parents[1];OLD=B/'psn/rtl/gp_slice'
cmds=[[sys.executable,'-B','implement_lazy.py'],['verilator','--cc','--exe','--top-module','gp_slice','-Wno-fatal','--Mdir','obj_lazy','-CFLAGS','-std=c++17 -O2','gp_slice_lazy.sv',str(OLD/'gp_slice_pe.sv'),'lazy_scoreboard.cpp'],['make','-C','obj_lazy','-f','Vgp_slice.mk','-j2'],[str(H/'obj_lazy/Vgp_slice'),str(H/'cases.bin'),str(H/'lazy_results.tsv')]]
with (H/'run_lazy.log').open('w') as log:
 for cmd in cmds:
  proc=subprocess.run(cmd,cwd=H,text=True,stdout=log,stderr=subprocess.STDOUT)
  if proc.returncode:raise SystemExit(f'failed {cmd[0]}, see run_lazy.log')
  print('PASS',cmd[0],flush=True)
rows=[]
for row in csv.DictReader((H/'lazy_results.tsv').open(),delimiter='\t'):
 row={k:(v if k=='case' else int(v)) for k,v in row.items()};row['frontend']=row.pop('intersection');row['service_including_configuration']=row['last_gate']+row['configuration_beats']+1;rows.append(row)
(H/'lazy_results.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
print(json.dumps(dict(tasks=len(rows),gate_bits=len(rows)*320)))
