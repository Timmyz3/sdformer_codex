#!/usr/bin/env python3
from pathlib import Path
import argparse,concurrent.futures,json,subprocess,time
H=Path(__file__).resolve().parent

def execute(arm,mode,first,count,stall,repeats):
 manifest=json.loads((H/'inputs.json').read_text())[arm]
 cmd=[str(H/'obj_dir/Vstream_wrapper'),manifest['source'],manifest['gold'],str(Path(manifest['weight']).parent),str(mode),str(first),str(count),str(stall),str(repeats),str(2000000*count+20000)]
 log=H/f'run_{arm}_m{mode}_first{first}_n{count}_s{stall}.log'
 start=time.monotonic()
 with log.open('w') as err:
  p=subprocess.run(cmd,check=True,text=True,stdout=subprocess.PIPE,stderr=err)
 rows=[]
 for line in p.stdout.splitlines():
  r=json.loads(line);r['arm']=arm;r['process_wall_seconds']=time.monotonic()-start;rows.append(r)
 assert len(rows)==repeats
 print('STREAM_PASS',arm,mode,count,stall,'seconds',round(time.monotonic()-start,3),flush=True)
 return rows

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--first',type=int,default=128);parser.add_argument('--count',type=int,default=64)
 parser.add_argument('--stalls',type=int,nargs='+',default=[0,1]);parser.add_argument('--repeats',type=int,default=2)
 parser.add_argument('--arms',nargs='+',default=['dense_q16','block_magnitude25','cin_fullcost25']);parser.add_argument('--modes',type=int,nargs='+',default=[5,6]);parser.add_argument('--workers',type=int,default=3);parser.add_argument('--output',default='results64.json');args=parser.parse_args()
 tasks=[(a,m,args.first,args.count,s,args.repeats) for a in args.arms for m in args.modes for s in args.stalls]
 rows=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
  for group in pool.map(lambda x:execute(*x),tasks):
   rows.extend(group);(H/args.output).write_text(json.dumps(rows,indent=2)+'\n')
 print(json.dumps(dict(runs=len(rows),checked_outputs=sum(r['checked_outputs'] for r in rows),max_seconds=max(r['process_wall_seconds'] for r in rows))))
