from pathlib import Path
import argparse,subprocess,sys,concurrent.futures
H=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--stage',choices=['raw','stream','all'],default='all');p.add_argument('--prepare',action='store_true');a=p.parse_args()
def call(cmd,log=None):
 if log:
  with (H/log).open('w') as f:subprocess.run(cmd,cwd=H,stdout=f,stderr=subprocess.STDOUT,check=True)
 else:subprocess.run(cmd,cwd=H,check=True)
if a.prepare:call([sys.executable,'prepare.py']);call([sys.executable,'implement_consumer.py'])
for stage in (['raw','stream'] if a.stage=='all' else [a.stage]):
 stream=stage=='stream';top='spatial_stream' if stream else 'spatial_core';obj='obj_stream' if stream else 'obj'
 src=['spatial_core.sv']+(['i24_consumer.sv','wide_phase_alu.sv','spatial_stream.sv','stream_tb.cpp'] if stream else ['tb.cpp'])
 call(['verilator','-Wall','--cc','--exe','--top-module',top,'--Mdir',obj,*src,'-CFLAGS','-O2 -std=c++14'],f'rebuild_{stage}.log')
 call(['make','-C',obj,'-f',f'V{top}.mk','-j2'],f'remake_{stage}.log')
 def one(pair):
  name,stall=pair;prefix='stream' if stream else 'results'
  with (H/f'{prefix}_{name}_{stall}.jsonl').open('w') as out,(H/f'{prefix}_{name}_{stall}.err').open('w') as err:
   subprocess.run([str(H/obj/f'V{top}'),str(H),str(H/f'{name}.txt'),str(stall)],cwd=H,stdout=out,stderr=err,check=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(one,[(n,s) for n in ['small','held','disjoint'] for s in [0,1]]))
 call([sys.executable,f'verify_{stage}.py'])
if (H/'raw_verification.json').exists() and (H/'stream_verification.json').exists():call([sys.executable,'finalize.py'])
