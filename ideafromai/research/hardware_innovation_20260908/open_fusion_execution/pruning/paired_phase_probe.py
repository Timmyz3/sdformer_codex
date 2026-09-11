"""One bounded alternative mask interface: equal masks inside horizontal P2.

Fixed two deleted H8 groups, no pruning-ratio sweep. The global control searches
all 66 group pairs. Row-phase search uses one fixed coordinate pass, 66 choices
per row parity; it can differ vertically while retaining horizontal broadcast.
Actual full-K integer consumer distortion is the objective. No new AEE/train.
"""
from pathlib import Path
from itertools import combinations
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'review'))
from phase_channel_probe import FULL,features,downstream,metrics,mask_weights,check_baseline


def main():
    cal=[(3+14*y,3+19*x) for y in range(8) for x in range(8)]
    val=[(10+14*y,12+19*x) for y in range(8) for x in range(8)]
    pairs=list(combinations(range(12),2))
    result=dict(scope=__doc__,calibration_points=cal,validation_points=val,axes={})
    masks={}
    for axis in ['ordinary','lifting_raw']:
        path=FULL/'capture_full_producers'/axis
        with np.load(path/'000_zurich_city_09_a_0001.npz') as z:
            data={k:z[k] for k in ['full_sn2_words','full_I24','full_updated_I24','full_continuous_q24','full_proj_words']}
        with np.load(path/'parameters.npz') as z:q={k:z[k] for k in z.files}
        x,raw=features(data,cal);u=x@q['U_conv2_theta_q16'].T;ref=downstream(u,raw,q)
        check_baseline(data,cal,ref)
        delta=np.empty((4,12)+u.shape,np.float64)
        for ph in range(4):
            offsets=[ky*3+kx for ky in range(3) for kx in range(3) if 2*((ky+1)%2)+(kx+1)%2==ph]
            for group in range(12):
                cols=[c*9+off for c in range(group*8,(group+1)*8) for off in offsets]
                delta[ph,group]=x[:,:,cols]@q['U_conv2_theta_q16'][:,cols].T
        def search(phases,already):
            costs=[]
            for a,b in pairs:
                du=sum(delta[ph,a]+delta[ph,b] for ph in phases)
                costs.append(metrics(downstream(u-already-du,raw,q),ref)['score'])
            index=int(np.argmin(costs));a,b=pairs[index]
            return pairs[index],sum(delta[ph,a]+delta[ph,b] for ph in phases),costs
        gp,_,gc=search(range(4),np.zeros_like(u))
        global_drop=np.zeros((4,96),bool)
        for g in gp:global_drop[:,g*8:(g+1)*8]=True
        paired=np.zeros((4,96),bool);already=np.zeros_like(u);steps=[]
        for phases in [(0,1),(2,3)]:
            chosen,du,costs=search(phases,already);already+=du
            for ph in phases:
                for g in chosen:paired[ph,g*8:(g+1)*8]=True
            steps.append(dict(phases=phases,chosen=chosen,candidate_costs=costs))
        vx,vr=features(data,val);vref=downstream(vx@q['U_conv2_theta_q16'].T,vr,q)
        check_baseline(data,val,vref)
        record=dict(global_candidates=66,paired_candidates=132,global_chosen=gp,paired_search=steps,variants={})
        masks[axis]={}
        for name,drop in [('global_joint_pair',global_drop),('row_phase_joint_pair',paired)]:
            w=mask_weights(q,drop)
            rec=dict(calibration=metrics(downstream(x@w.T,raw,q),ref),
                validation=metrics(downstream(vx@w.T,vr,q),vref),
                horizontal_P2_mask_equal=bool(np.array_equal(drop[0],drop[1]) and np.array_equal(drop[2],drop[3])))
            record['variants'][name]=rec
            masks[axis][name]=dict(mask_uint8=drop.astype(int).tolist(),drop_channels=[np.flatnonzero(d).tolist() for d in drop])
        result['axes'][axis]=record
        print(axis,{k:v['validation'] for k,v in record['variants'].items()},flush=True)
    (HERE/'paired_phase_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    (HERE/'paired_phase_masks.json').write_text(json.dumps(masks,indent=2)+'\n')


if __name__=='__main__':main()
