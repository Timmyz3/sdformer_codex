from pathlib import Path
import subprocess,sys,csv,json
H=Path(__file__).resolve().parent;B=H.parents[1];OLD=B/'psn/rtl/gp_slice'
cmds=[[sys.executable,'-B','implement.py'],[sys.executable,'-B','implement_tb.py'],[sys.executable,'-B','prepare.py'],['verilator','--cc','--exe','--top-module','gp_slice','-Wno-fatal','--Mdir','obj','-CFLAGS','-std=c++17 -O2','gp_slice.sv',str(OLD/'gp_slice_pe.sv'),'scoreboard.cpp'],['make','-C','obj','-f','Vgp_slice.mk','-j2'],[str(H/'obj/Vgp_slice'),str(H/'cases.bin'),str(H/'results.tsv')]]
with (H/'run.log').open('w') as log:
 for cmd in cmds:
  proc=subprocess.run(cmd,cwd=H,text=True,stdout=log,stderr=subprocess.STDOUT)
  if proc.returncode:raise SystemExit(f'failed {cmd[0]}, see run.log')
  print('PASS',cmd[0],flush=True)
rows=[]
for row in csv.DictReader((H/'results.tsv').open(),delimiter='\t'):
 row={k:(v if k=='case' else int(v)) for k,v in row.items()};row['frontend']=row.pop('intersection');row['service_including_configuration']=row['last_gate']+row['configuration_beats']+1;rows.append(row)
(H/'results.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
print(json.dumps(dict(tasks=len(rows),gate_bits=len(rows)*320)))
