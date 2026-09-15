from pathlib import Path
import subprocess,argparse,json
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--stage',default='small',choices=['small','held','disjoint','sequences','swap']);p.add_argument('--skip-build',action='store_true');a=p.parse_args()
if not a.skip_build:
 with (H/'build.log').open('w') as o:
  subprocess.run(['verilator','-Wall','--cc','--exe','--top-module','interleave_stream','--Mdir','obj','interleave_stream.sv','rr_context.sv','i24_consumer.sv','wide_phase_alu.sv','stream_tb.cpp','-CFLAGS','-O3 -std=c++14'],cwd=H,stdout=o,stderr=subprocess.STDOUT,check=True)
  subprocess.run(['make','-C','obj','-f','Vinterleave_stream.mk','-j2'],cwd=H,stdout=o,stderr=subprocess.STDOUT,check=True)
manifests=[H/f'{a.stage}.txt'] if a.stage!='swap' else [H/'held.txt',H/'disjoint.txt',H/'sequences.txt',H/'small.txt']
for borrow in [0,1]:
 for stall in [0,1]:
  path=H/f'results_{a.stage}_b{borrow}_s{stall}.jsonl'
  with path.open('w') as out:subprocess.run([str(H/'obj/Vinterleave_stream'),str(H),str(stall),str(borrow),*[str(p) for p in manifests]],cwd=H,stdout=out,check=True)
  rows=[json.loads(x) for x in path.read_text().splitlines()]
  print(json.dumps(dict(stage=a.stage,borrow=borrow,stall=stall,jobs=len(rows),services=[r['total_cycles']+1 for r in rows])),flush=True)
