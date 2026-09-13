"""One fixed real P2 tile, three pairings, three exact RTL execution modes."""
from pathlib import Path
import json
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
EXPORT=BASE/'open_fusion_execution/stage_20260912/algorithm/hardware_exports/ordinary'

def write_hex(path,values):
    path.write_text(''.join(f'{int(v)&0xffffffff:08x}\n' for v in values))

def pairs_greedy(bits,objective):
    remaining=set(range(len(bits))); result=[]
    while remaining:
        a=min(remaining,key=lambda k:(-bits[k].bit_count(),k));remaining.remove(a)
        if objective=='conflict':
            key=lambda b:((bits[a]&bits[b]).bit_count(),(bits[a]|bits[b]).bit_count(),b)
        else:
            key=lambda b:((bits[a]|bits[b]).bit_count(),-(bits[a]&bits[b]).bit_count(),b)
        b=min(remaining,key=key);remaining.remove(b);result.append((a,b))
    return result

def main():
    cap=np.load(EXPORT/'000_zurich_city_09_a_0001.npz')
    coeff=np.load(EXPORT/'deployed_constants.npz')
    g=cap['interior_sn2_gate'].astype(np.int64)
    w=coeff['U_conv2_theta_q16'].astype(np.int64).reshape(16,864)
    def patch(y,x): return g[:,:,y-1:y+2,x-1:x+2].reshape(10,864).T
    src=np.concatenate([patch(1,1),patch(1,3)],axis=1)
    calibration=np.concatenate([patch(y,x) for y in (1,3,5,7) for x in (1,3,5,7)
        if (y,x) not in ((1,1),(1,3))],axis=1)
    bits=[sum(int(v)<<j for j,v in enumerate(row)) for row in calibration]
    maps={'adjacent':list(zip(range(0,864,2),range(1,864,2))),
          'coactivity':pairs_greedy(bits,'conflict'),'union':pairs_greedy(bits,'union')}
    datasets={'real':src,'all_one_control':np.ones_like(src),
              'all_zero_control':np.zeros_like(src)}
    metadata={'capture':str(EXPORT/'000_zurich_city_09_a_0001.npz'),
      'constants':str(EXPORT/'deployed_constants.npz'),'K':864,'N':16,'T':10,'P':2,
      'local_centers':[[1,1],[1,3]],'source_layout':'C,dy,dx; P-major,T-minor mask',
      'calibration':'other 14 interior 3x3 centers of this same frame; spatial holdout only',
      'spikes':int(src.sum()),'union_sources':int(np.any(src,axis=1).sum()),
      'pair_stats':{}}
    runs=[]
    for data_name,s in datasets.items():
      for name,pairs in maps.items():
        if data_name!='real' and name!='adjacent': continue
        out=HERE/'generated'/data_name/name;out.mkdir(parents=True,exist_ok=True)
        masks=[sum(int(v)<<j for j,v in enumerate(row)) for row in s]
        write_hex(out/'source.hex',masks)
        write_hex(out/'mapping.hex',[a|(b<<10) for a,b in pairs])
        write_hex(out/'weights.hex',[(int(w[o,a])&65535)|((int(w[o,b])&65535)<<16)
             for o in range(16) for a,b in pairs])
        write_hex(out/'gold.hex',(w@s).reshape(-1))
        if data_name=='real':
            metadata['pair_stats'][name]={
              'active_pairs':sum(bool(masks[a]|masks[b]) for a,b in pairs),
              'both_sources_live':sum(bool(masks[a]) and bool(masks[b]) for a,b in pairs),
              'collision_pairs':sum(bool(masks[a]&masks[b]) for a,b in pairs),
              'coincident_bits':sum((masks[a]&masks[b]).bit_count() for a,b in pairs)}
        for mode in (0,1,2):
            for stall in (0,1):
                runs.append((out,data_name,name,mode,stall))
    build=HERE/'obj_dir'; build.mkdir(exist_ok=True)
    with (HERE/'build.log').open('w') as log:
        subprocess.run(['verilator','--cc','--exe','-Wno-fatal','--top-module','pair_execution',
          '--Mdir',str(build),str(HERE/'pair_execution.sv'),str(HERE/'tb.cpp')],check=True,stdout=log,stderr=subprocess.STDOUT)
        subprocess.run(['make','-C',str(build),'-f','Vpair_execution.mk','-j2'],check=True,stdout=log,stderr=subprocess.STDOUT)
    results=[]
    for out,dataset,pairing,mode,stall in runs:
        completed=subprocess.run([str(build/'Vpair_execution'),str(out),str(mode),str(stall)],check=True,text=True,capture_output=True)
        r=json.loads(completed.stdout);r.update(dataset=dataset,pairing=pairing);results.append(r)
    (HERE/'results.json').write_text(json.dumps({'metadata':metadata,'rtl':results},indent=2)+'\n')
    print(json.dumps({'metadata':metadata,'rtl':[r for r in results if r['dataset']=='real']},indent=2))

if __name__=='__main__':main()
