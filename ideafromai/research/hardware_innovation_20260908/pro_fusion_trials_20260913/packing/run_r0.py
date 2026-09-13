"""Extend the same exact integer leaf to costly r0 conv2, all K and N.

32 held-out spatial positions form 16 P2 tiles; no full-image performance claim.
The new Q16 export is a diagnostic function, not a validated deployment model.
"""
from pathlib import Path
import json
import subprocess
import numpy as np
from run import HERE,BASE,write_hex,pairs_greedy

def main():
    file=BASE/'open_fusion_execution/major_operator_fusions_20260913/root_owned/sttmultires_unet_encoders_swin3d_patch_embed_residual_encoding_resblocks_0_conv2_0.npz'
    d=np.load(file)
    g=d['input'].astype(np.int64)
    w=np.rint(d['weight'].astype(np.float64).reshape(96,864)*65536).astype(np.int64)
    cal=g[:,:32,:].reshape(320,864).T
    bits=[sum(int(v)<<j for j,v in enumerate(row)) for row in cal]
    maps={'adjacent':list(zip(range(0,864,2),range(1,864,2))),
          'coactivity':pairs_greedy(bits,'conflict'),'union':pairs_greedy(bits,'union')}
    build=HERE/'obj_dir/r0';build.mkdir(parents=True,exist_ok=True)
    with (HERE/'r0_build.log').open('w') as log:
        subprocess.run(['verilator','--cc','--exe','-Wno-fatal','--top-module','pair_execution','-GO=96',
          '--Mdir',str(build),str(HERE/'pair_execution.sv'),str(HERE/'tb.cpp')],check=True,stdout=log,stderr=subprocess.STDOUT)
        subprocess.run(['make','-C',str(build),'-f','Vpair_execution.mk','-j2'],check=True,stdout=log,stderr=subprocess.STDOUT)
    results=[];stats=[];error=0.;signal=0.
    for position in range(32,64,2):
        src=g[:,position:position+2,:].transpose(1,0,2).reshape(20,864).T
        exact=w@src
        float_y=d['weight'].astype(np.float64).reshape(96,864)@src
        error+=float(np.sum((exact/65536-float_y)**2));signal+=float(np.sum(float_y**2))
        masks=[sum(int(v)<<j for j,v in enumerate(row)) for row in src]
        for name,pairs in maps.items():
            out=HERE/'generated/r0'/f'positions_{position}_{position+1}'/name
            out.mkdir(parents=True,exist_ok=True)
            write_hex(out/'source.hex',masks)
            write_hex(out/'mapping.hex',[a|(b<<10) for a,b in pairs])
            write_hex(out/'weights.hex',[(int(w[o,a])&65535)|((int(w[o,b])&65535)<<16)
                   for o in range(96) for a,b in pairs])
            write_hex(out/'gold.hex',exact.reshape(-1))
            stats.append(dict(tile=position//2-16,pairing=name,spikes=int(src.sum()),
                active_pairs=sum(bool(masks[a]|masks[b]) for a,b in pairs),
                both_sources_live=sum(bool(masks[a]) and bool(masks[b]) for a,b in pairs),
                collision_pairs=sum(bool(masks[a]&masks[b]) for a,b in pairs)))
            for mode in (0,1,2):
                for stall in (0,1):
                    done=subprocess.run([str(build/'Vpair_execution'),str(out),str(mode),str(stall)],check=True,text=True,capture_output=True)
                    r=json.loads(done.stdout);r.update(tile=position//2-16,pairing=name);results.append(r)
    aggregate=[]
    for name in maps:
      for mode in (0,1,2):
        for stall in (0,1):
          selected=[r for r in results if r['pairing']==name and r['mode']==mode and r['stall']==stall]
          row=dict(pairing=name,mode=mode,stall=stall,tiles=len(selected))
          for field in ('cycles','weight_words','issue','replay','outputs','input_stall','output_stall','request_stall'):
              row[field]=sum(r[field] for r in selected)
          aggregate.append(row)
    report={'capture':str(file),'scope':'r0 conv2 fixed 16 P2 tiles, complete K864/N96/T10 per tile; pre-gathered source words, not full-image/native-im2col',
      'calibration_positions':d['positions'][:32].tolist(),'evaluation_positions':d['positions'][32:].tolist(),
      'coefficient_contract':'round_even(original FP32 weight*2^16) -> signed16; raw sum signed32; diagnostic export, no inherited AEE',
      'coefficient_min':int(w.min()),'coefficient_max':int(w.max()),
      'local_quant_output_nrmse':float(np.sqrt(error/signal)),
      'aggregate':aggregate,'pair_stats':stats,'rtl':results}
    (HERE/'r0_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('scope','local_quant_output_nrmse','aggregate')},indent=2))

if __name__=='__main__':main()
