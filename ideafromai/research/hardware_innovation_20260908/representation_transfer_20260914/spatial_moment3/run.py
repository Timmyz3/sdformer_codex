from pathlib import Path
import argparse,json,subprocess
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--consumer',action='store_true');p.add_argument('--stage',choices=['small','held','disjoint','sequences'],default='small');p.add_argument('--skip-build',action='store_true');p.add_argument('--input',type=Path);a=p.parse_args();D=a.input.resolve() if a.input else H
top='spatial_stream' if a.consumer else 'spatial_core';obj='obj_stream' if a.consumer else 'obj';prefix='consumer' if a.consumer else 'raw'
if not a.skip_build:
 sources=['spatial_core.sv','spatial_stream.sv','i24_consumer.sv','wide_phase_alu.sv','stream_tb.cpp'] if a.consumer else ['spatial_core.sv','tb.cpp']
 with (H/f'build_{prefix}.log').open('w') as out:
  subprocess.run(['verilator','-Wall','--cc','--exe','--top-module',top,'--Mdir',obj,*sources,'-CFLAGS','-O3 -std=c++14'],cwd=H,stdout=out,stderr=subprocess.STDOUT,check=True)
  subprocess.run(['make','-C',obj,'-f',f'V{top}.mk','-j2'],cwd=H,stdout=out,stderr=subprocess.STDOUT,check=True)
for stall in [0,1]:
 with (D/f'{prefix}_{a.stage}_s{stall}.jsonl').open('w') as out:
  subprocess.run([str(H/obj/f'V{top}'),str(D),str(D/f'{a.stage}.txt'),str(stall),'1'],stdout=out,check=True,cwd=H)
 rows=[json.loads(x) for x in (D/f'{prefix}_{a.stage}_s{stall}.jsonl').read_text().splitlines()]
 print(json.dumps(dict(arm='moment3',stage=a.stage,consumer=a.consumer,stall=stall,commands=len(rows),cycles=sum(r['c_cycles' if a.consumer else 'cycles'] for r in rows))),flush=True)
