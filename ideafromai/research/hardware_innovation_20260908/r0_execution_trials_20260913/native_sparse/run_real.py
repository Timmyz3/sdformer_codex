#!/usr/bin/env python3
"""Run fixed native captures and supplied model masks; never invokes a model."""
import argparse
import json
from pathlib import Path
import numpy as np
from run import HERE,fixture,build,execute
from analyze import analyze

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--mask',action='append',default=[],help='arm_name:npz_path:key (one fixed arm, no parameter search)')
    parser.add_argument('--no-build',action='store_true')
    args=parser.parse_args()
    manifest_path=args.capture.parent/'capture_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    assert manifest.get('complete'), 'Only a completed accepted capture can enter the real result table'
    capture=np.load(args.capture,allow_pickle=False)
    sources=capture['source_bits'];w=capture['weight_q16'];origins=capture['input_origin_yx']
    assert sources.ndim==5 and sources.shape[1:]==(10,96,4,4)
    assert w.shape==(96,96,3,3)
    reconstructed=np.rint(capture['weight_fp32'].astype(np.float64)*float(capture['theta'])*65536)
    assert np.array_equal(w,reconstructed)
    assert np.array_equal(capture['source_fp32'],sources.astype(np.float32)*float(capture['theta']))
    arms={'dense':np.ones((12,24),dtype=bool)};mask_sources={}
    for descriptor in args.mask:
        name,path,key=descriptor.split(':',2)
        mask=np.load(path,allow_pickle=False)[key]
        assert mask.shape==(12,24) and np.all((mask==0)|(mask==1))
        assert name not in arms
        arms[name]=mask.astype(bool);mask_sources[name]={'path':path,'key':key,'live_blocks':int(mask.sum())}
    names=[]
    for arm,mask in arms.items():
        for tile,source in enumerate(sources):
            name=f'real_{arm}_tile{tile}'
            path=fixture(name,source,w,mask,{'capture':str(args.capture.resolve()),'arm':arm,'tile':tile,
                'full_k':864,'snapshot':manifest.get('source_snapshot'),'mask_source':mask_sources.get(arm)},origin=origins[tile])
            # Dense integer gold is independently regenerated from native
            # source and is also checked against the capture's original gold.
            if arm=='dense':
                bits=np.array([int(s,16) for s in (path/'gold.hex').read_text().split()],dtype=np.uint32).view(np.int32)
                wanted=capture['golden_accum'][tile].reshape(10,12,8,4).transpose(1,3,0,2).reshape(-1)
                assert np.array_equal(bits,wanted)
            names.append(name)
    if not args.no_build:build()
    execute(names,'real_results.json')
    analyze('real_results.json')
    rows=json.loads((HERE/'real_results.json').read_text())
    summary={}
    for arm in arms:
        summary[arm]={}
        for mode in range(3):
            selected=[r for r in rows if r['fixture'].startswith(f'real_{arm}_tile') and r['mode']==mode and r['stall']==0 and r['command']==0]
            assert len(selected)==len(sources)
            keys=['cycles','source_words','weight_words','psum_reads','psum_writes','sum_issues','update_issues','control_cycles']
            totals={key:sum(r[key] for r in selected) for key in keys}
            totals['fresh_source_origin_load_cycles']=1537*len(sources)
            totals['cycles_with_fresh_source_origin']=totals['cycles']+1537*len(sources)
            totals['cycles_first_batch_with_weight_mask_load']=totals['cycles_with_fresh_source_origin']+10656
            summary[arm][str(mode)]=totals
    output={'capture':str(args.capture.resolve()),'manifest':str(manifest_path.resolve()),'tile_count':len(sources),
            'mask_sources':mask_sources,'runs':len(rows),'checked_outputs':sum(r['outputs'] for r in rows),
            'unique_outputs_per_arm':len(sources)*3840,'summary_no_stalls_command0':summary,
            'precision':'signed32 raw dot; Wq=RNE(theta*Wfp32*65536); no original norm/bias/residual included',
            'repeat_command':'functional verification only; per-fresh-tile source/origin loading explicitly added'}
    (HERE/'real_summary.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(output,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
