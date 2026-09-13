"""Fixed three72-block controls, trained only by one preselected train capture."""
import json
import numpy as np
from model_access import HERE,BASE
PREVIOUS=BASE/'r0_execution_trials_20260913/data_and_quality'

def costs(source,valid):
    F=source.shape[0];words=np.sum(source.astype(np.int64)*(1<<np.arange(10))[None,:,None,None,None],axis=1)
    rows=[]
    for cg in range(24):
        emitted=0;weight=0;sums=0;updates=0;consumer_control=0;active=0
        for f in range(F):
            for pair in range(2):
                for y in range(4):
                    for x in range(4):
                        if not valid[f,y,x]:continue
                        a,b=[int(words[f,4*cg+pair*2+i,y,x]) for i in (0,1)]
                        union=a|b
                        if not union:continue
                        P=sum(0<=y-p//2<3 and 0<=x-p%2<3 for p in range(4))
                        M=union.bit_count();active+=1
                        emitted+=P*(3+int(bool(a))+int(bool(b))+int(bool(a&b))+3*M)
                        weight+=12*P*(int(bool(a))+int(bool(b)))
                        sums+=12*P*int(bool(a&b));updates+=12*P*M
                        consumer_control+=12*P*(3+M)
        sr=4*int(valid.sum());ctrl=158*F-sr+consumer_control
        complete=158*F+12*emitted
        assert complete==sr+weight+sums+2*updates+ctrl
        rows.append(dict(Cin4=cg,execution_cost_per_live_O8=emitted,source_words_removed=sr,
            W_words_removed=weight,sum_issues_removed=sums,psum_updates_removed=updates,
            control_cycles_removed=ctrl,total_core_cycles_saved=complete,active_pair_source=active))
    return rows

def golden(source,w):
    out=np.empty((len(source),10,96,2,2),np.int64)
    for y in range(2):
        for x in range(2):out[:,:,:,y,x]=np.einsum('ftcij,ncij->ftn',source[:,:,:,y:y+3,x:x+3].astype(np.int64),w,optimize=True)
    return out

def main():
    cal=np.load(HERE/'calibration_grid_t10.npz');ev=np.load(PREVIOUS/'r0_contiguous_t10.npz')
    w=cal['weight_q16'].astype(np.int64);s=cal['source_bits'];assert np.array_equal(w,ev['weight_q16'])
    contributions=np.empty((24,len(s),10,96,2,2),np.int64)
    for cg in range(24):
        for y in range(2):
            for x in range(2):
                contributions[cg,:,:,:,y,x]=np.einsum('ftcij,ncij->ftn',s[:,:,cg*4:cg*4+4,y:y+3,x:x+3].astype(np.int64),w[:,cg*4:cg*4+4],optimize=True)
    assert np.array_equal(contributions.sum(axis=0),cal['golden_accum'])
    rows=costs(s,cal['source_valid_yx']);cycle=np.array([r['total_core_cycles_saved'] for r in rows])
    energy=np.square(contributions.astype(np.float64)).sum(axis=(1,2,3,4,5))
    magnitude=np.square(w.astype(np.float64)).reshape(12,8,24,4,3,3).sum(axis=(1,3,4,5))
    masks={}
    block=np.ones((12,24),np.bool_);block.ravel()[np.argsort(magnitude.ravel(),kind='stable')[:72]]=False
    masks['block_magnitude25']=block
    cinmag=np.ones((12,24),np.bool_);cinmag[:,np.argsort(magnitude.sum(axis=0),kind='stable')[:6]]=False
    masks['cin_magnitude25']=cinmag
    cincost=np.ones((12,24),np.bool_);cincost[:,np.argsort(energy/cycle,kind='stable')[:6]]=False
    masks['cin_fullcost25']=cincost
    masks={'dense_q16':np.ones((12,24),np.bool_),**masks}
    metadata={}
    for name,live in masks.items():
        qw=w*live.repeat(8,axis=0).repeat(4,axis=1)[:,:,None,None]
        out=golden(ev['source_bits'],qw);calout=golden(s,qw)
        np.savez_compressed(HERE/(name+'.npz'),source_bits=ev['source_bits'],source_valid_yx=ev['source_valid_yx'],
            input_origin_yx=ev['input_origin_yx'],output_origin_yx=ev['output_origin_yx'],
            frame=ev['frame'],weight_q16=qw.astype(np.int16),weight_fp32=ev['weight_fp32'],theta=ev['theta'],
            weight_exponent=ev['weight_exponent'],live=live,golden_accum=out,
            calibration_frame=cal['frame'],calibration_source_bits=s,calibration_golden_accum=calout)
        metadata[name]=dict(file=name+'.npz',dropped_blocks=int((~live).sum()),retired_Cin4=np.flatnonzero(~live.any(axis=0)).tolist(),
            coefficient_nonzero=int(np.count_nonzero(qw)),calibration_output_SSE=float(np.square((calout-cal['golden_accum']).astype(np.float64)).sum()),
            evaluation_tile_output_SSE=float(np.square((out-ev['golden_accum']).astype(np.float64)).sum()))
    np.savez_compressed(HERE/'calibration_scores.npz',Cin4_contribution=contributions,output_SSE=energy,
        complete_cycle_saving=cycle,score=energy/cycle,block_magnitude=magnitude,**{n+'_live':m for n,m in masks.items()})
    manifest=dict(complete=True,calibration_frame=str(cal['frame']),evaluation_frame=str(ev['frame']),
        calibration_evaluation_frame_overlap=False,calibration_evaluation_sequence_overlap=False,
        calibration_source_ones=int(s.sum()),calibration_zero_contribution_Cin4=np.flatnonzero(energy==0).tolist(),
        selection_tie_policy='stable original flattened Ogroup,Cgroup order, or original Cgroup order; no resampling despite sparse calibration',
        fixed_deleted_blocks=72,selected_Cin4_count=6,rate_search=False,training=False,
        cost_basis='fixed prior mode3 source-major live-consumer enumeration, no stalls, full core work; newC4reuse hardware not used to refit masks',
        cost_formula='for F calibrationtiles: full-Cin4 removal saves158F+12*execution_cost;execution_cost=sum_inbounds_pairXY P*(3Iunion+Ia+Ib+Iintersection+3popcountunion)',
        loss='fullO96/T10/P4 exact integer contribution SSE; cost arm is independent per-Cin4 ranking, not greedy recomputation',
        cost_rows=rows,arms=metadata)
    (HERE/'controls_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='cost_rows'},indent=2))
if __name__=='__main__':main()
