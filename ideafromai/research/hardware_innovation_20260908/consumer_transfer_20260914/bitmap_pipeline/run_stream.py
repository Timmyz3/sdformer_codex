from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,json,subprocess
H=Path(__file__).resolve().parent;B=H.parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['short','64','disjoint'],default='short');args=parser.parse_args()
params=B/'fusion_review_followup_20260914/pair_dictionary/fixtures/real_0'
if args.stage=='short':
    # Actual source switching makes every live metadata word cross lifetimes.
    paths=[params,params.parent/'zero',params.parent/'real_1',params.parent/'zero']
    (H/'fixtures').mkdir(exist_ok=True)
    manifest=H/'fixtures/short_manifest.txt';manifest.write_text('\n'.join(map(str,paths))+'\n')
    s=(H/'tb.cpp').read_text().replace('if(argc!=4)','if(argc!=5)')
    s=s.replace('for(unsigned i=0;i<src.size();i++){data[0]=src[i];cfg(0,i);}\n data[0]=(origin[0]&65535)|((origin[1]&65535)<<16);cfg(3,0);','')
    s=s.replace('d.cfg_valid=0;unsigned checked=0;', '''d.cfg_valid=0;unsigned checked=0;
 std::ifstream mf(argv[4]);std::vector<std::string> paths;std::string path;while(mf>>path)paths.push_back(path);
 if(paths.empty())return 22;''')
    s=s.replace('for(unsigned command=0;command<2;command++){', '''for(unsigned command=0;command<2*paths.size();command++){
  unsigned static_cycles=command==0?cfg_cycles:0;cfg_cycles=0;
  src=read(paths[command%paths.size()]+"/source.hex");gold=read(paths[command%paths.size()]+"/gold.hex");origin=read(paths[command%paths.size()]+"/origin.hex");
  for(unsigned i=0;i<src.size();i++){data[0]=src[i];cfg(0,i);}
  data[0]=(origin[0]&65535)|((origin[1]&65535)<<16);cfg(3,0);d.cfg_valid=0;
  unsigned this_config=static_cycles+cfg_cycles;''')
    s=s.replace('(command==0?cfg_cycles:0)','this_config').replace('checked==7680','checked==3840*2*paths.size()')
    (H/'stream_tb.cpp').write_text(s)
    with (H/'build_stream.log').open('w') as log:
        for cmd in [['verilator','-Wall','--cc','--exe','--top-module','decomp_core','--Mdir','obj_stream','decomp_core.sv','stream_tb.cpp','-CFLAGS','-O2 -std=c++14'],['make','-C','obj_stream','-f','Vdecomp_core.mk','-j2']]:
            subprocess.run(cmd,cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
else:
    manifest=B/'transfer_adapt_20260914'/('pair_sparse/fixtures/stream/manifest.txt' if args.stage=='64' else 'audit/fixtures/disjoint_4000/manifest.txt')
paths=manifest.read_text().split();nt=len(paths)
def run(job):
    mode,stall=job
    r=subprocess.run([str(H/'obj_stream/Vdecomp_core'),str(params),str(mode),str(stall),str(manifest)],capture_output=True,text=True)
    if r.returncode:raise RuntimeError((job,r.returncode,r.stdout[-1200:],r.stderr))
    rows=[dict(json.loads(l),fixture=paths[i%nt]) for i,l in enumerate(r.stdout.splitlines())]
    assert len(rows)==2*nt
    print('PASS',args.stage,mode,stall,len(rows),flush=True);return rows
with ThreadPoolExecutor(max_workers=3) as pool:
    rows=[r for a in pool.map(run,[(m,s) for m in [14,15,13,10] for s in [0,1]]) for r in a]
(H/f'results_{args.stage}.json').write_text('[\n'+',\n'.join(json.dumps(r,separators=(',',':')) for r in rows)+'\n]\n')
summary={}
for m in [14,15,13,10]:
 for s in [0,1]:
  for repeat in [0,1]:
    a=[r for r in rows if r['mode']==m and r['stall']==s and r['command']//nt==repeat]
    v={k:sum(r[k] for r in a) for k in a[0] if k not in ['mode','stall','command','fixture','state_cycles']}
    v['service_cycles']=v['cycles']+v['configuration_cycles']+nt
    summary[f'm{m}_s{s}_repeat{repeat}']=v
(H/f'SUMMARY_{args.stage}.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:{'core':v['cycles'],'service':v['service_cycles']} for k,v in summary.items()},indent=2))
