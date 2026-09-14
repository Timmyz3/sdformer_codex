"""Same-frame held128..191, existing native source/gold, real RTL inverse drain."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import subprocess
import argparse

H=Path(__file__).resolve().parent
B=H.parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--disjoint',action='store_true')
args=parser.parse_args()
assert json.loads((H/'verification.json').read_text())['gate_64']
s=(H/'tb.cpp').read_text().replace('if(argc!=4&&argc!=5)','if(argc!=5)')
s=s.replace('for(unsigned i=0;i<src.size();i++){data[0]=src[i];cfg(0,i);}\n data[0]=(origin[0]&65535)|((origin[1]&65535)<<16);cfg(3,0);','')
s=s.replace('d.cfg_valid=0;unsigned checked=0;bool permutation_loaded=false;', '''d.cfg_valid=0;unsigned checked=0;bool permutation_loaded=false;
 std::ifstream mf(argv[4]);std::vector<std::string> tiles;std::string td;while(mf>>td)tiles.push_back(td);
 if(tiles.empty())return 22;''')
s=s.replace('for(unsigned command=0;command<2;command++){', '''for(unsigned command=0;command<2*tiles.size();command++){
  unsigned static_cycles=command==0?cfg_cycles:0;cfg_cycles=0;
  src=read(tiles[command%tiles.size()]+"/source.hex");origin=read(tiles[command%tiles.size()]+"/origin.hex");gold=read(tiles[command%tiles.size()]+"/gold.hex");
  for(unsigned i=0;i<src.size();i++){data[0]=src[i];cfg(0,i);}
  data[0]=(origin[0]&65535)|((origin[1]&65535)<<16);cfg(3,0);d.cfg_valid=0;''')
s=s.replace('unsigned runmode=command&&argc==5?std::stoul(argv[4]):mode;', 'unsigned runmode=mode;')
s=s.replace('unsigned command_cfg=command==0?cfg_cycles:0,', 'unsigned command_cfg=static_cycles+cfg_cycles,')
s=s.replace('checked==7680','checked==3840*2*tiles.size()')
(H/'stream_tb.cpp').write_text(s)
with (H/'build_stream.log').open('w') as log:
    subprocess.run(['verilator','-Wall','--cc','--exe','--top-module','decomp_core','--Mdir','obj_stream','decomp_core.sv','stream_tb.cpp','-CFLAGS','-O2 -std=c++14'],cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run(['make','-C','obj_stream','-f','Vdecomp_core.mk','-j2'],cwd=H,stdout=log,stderr=subprocess.STDOUT,check=True)
manifest=B/('transfer_adapt_20260914/audit/fixtures/disjoint_4000/manifest.txt' if args.disjoint else 'transfer_adapt_20260914/pair_sparse/fixtures/stream/manifest.txt')
tile_paths=manifest.read_text().split()
first=4000 if args.disjoint else 128
assert [int(Path(p).name) for p in tile_paths]==list(range(first,first+64))
modes=[0,4,5] if args.disjoint else [0,3,4,5]
tag='disjoint' if args.disjoint else 'stream'
def run(job):
    mode,stall=job
    r=subprocess.run([str(H/'obj_stream/Vdecomp_core'),str(B/'fusion_review_followup_20260914/pair_dictionary/fixtures/real_0'),str(mode),str(stall),str(manifest)],capture_output=True,text=True)
    (H/f'logs/{tag}_m{mode}_s{stall}.log').write_text(r.stdout+r.stderr)
    assert r.returncode==0,(job,r.returncode,r.stdout[-2000:],r.stderr)
    return [dict(json.loads(l),fixture=f'native_{first+i%64}',tile_id=first+i%64) for i,l in enumerate(r.stdout.splitlines())]
with ThreadPoolExecutor(max_workers=3) as pool:
    rows=[r for group in pool.map(run,[(m,s) for m in modes for s in [0,1]]) for r in group]
(H/f'results_{tag}.json').write_text(json.dumps(rows,indent=2)+'\n')
summary={}
for mode in modes:
    for stall in [0,1]:
        for repeat in [0,1]:
            rs=[r for r in rows if r['mode']==mode and r['stall']==stall and r['command']//64==repeat]
            v={k:sum(r[k] for r in rs) for k in rs[0] if k not in ['mode','stall','command','state_cycles','fixture','tile_id']}
            v['start_beats']=len(rs);v['service_cycles']=v['cycles']+v['configuration_cycles']+len(rs)
            summary[f'm{mode}_s{stall}_repeat{repeat}']=v
(H/f'SUMMARY_{tag}.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:{'core':v['cycles'],'service':v['service_cycles'],'updates':v['first_issues']} for k,v in summary.items()}),flush=True)
