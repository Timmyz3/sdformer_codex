"""One-frame, untrained phase-channel mask probe on actual r1 consumers.

Calibrate 4 x 12 removals on 64 predeclared anchor locations, then measure a
separate 64-anchor grid from the same frame. This is neither AEE nor timing.
All complete K864 terms, integer RNE/sat boundaries, residual, projection gate,
and continuous U32/V96 are retained. FP64 only holds exact integer arithmetic;
all legal matrix accumulations here remain below 2**47 (<2**53).
"""
from pathlib import Path
import json
import numpy as np
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
FULL=BASE/'algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain'


def rne(x,shift):
    assert np.max(np.abs(x))<2**47
    return np.clip(np.rint(x/(2.**shift)),-2**23,2**23-1)


def features(data,points):
    padded=np.pad(data['full_sn2_words'],((0,0),(1,1),(1,1)))
    words=np.stack([padded[:,2*y:2*y+3,2*x:2*x+3].reshape(864) for y,x in points])
    gate=np.stack([(words>>t)&1 for t in range(10)],axis=1).astype(np.float64)
    raw=np.stack([data['full_I24'][:,:,2*y,2*x] for y,x in points]).astype(np.float64)
    return gate,raw


def downstream(uacc,raw,q):
    lat=rne(uacc,int(q['U_conv2_theta_exponent'])-14)
    branch=rne(lat @ q['F_q16'].T,int(q['F_exponent']))
    updated=rne(raw+branch+q['BN2_constant_q24'],0)
    perm=q['consumer_permutation'].astype(int)
    threshold=q['consumer_threshold'][None,:,None]
    direction=q['consumer_direction'][None,:,None]
    constant=q['consumer_constant'][None,:,None]
    proj=np.where(constant>=0,constant.astype(bool),np.where(direction>0,updated[:,perm]>=threshold,updated[:,perm]<=threshold))
    up=rne(updated @ q['U_ped_q16'].T,int(q['U_ped_exponent']))
    ped=rne(up @ q['V_ped_q16'].T,int(q['V_ped_exponent']))
    ped=rne(ped+q['PED_bias_q24'],0)
    return updated,proj,ped


def metrics(value,ref):
    updated,g,ped=value;u0,g0,p0=ref
    relative_ped_mse=float(np.sum((ped-p0)**2)/max(float(np.sum(p0**2)),1.))
    gate_flip_fraction=float(np.mean(g!=g0))
    return dict(gate_flips=int(np.count_nonzero(g!=g0)),gate_count=int(g.size),gate_flip_fraction=gate_flip_fraction,PED_relative_squared_error=relative_ped_mse,PED_q24_RMSE=float(np.sqrt(np.mean((ped-p0)**2))),updated_q24_RMSE=float(np.sqrt(np.mean((updated-u0)**2))),score=relative_ped_mse+gate_flip_fraction)


def mask_weights(q,drop):
    # Output anchors are even/even. Kernel (ky,kx) reads source phase
    # ((ky-1)%2,(kx-1)%2), equivalently ((ky+1)%2,(kx+1)%2).
    w=q['U_conv2_theta_q16'].reshape(16,96,3,3).copy()
    for py in range(2):
        for px in range(2):
            channels=np.flatnonzero(drop[py*2+px])
            for ky in range(3):
                for kx in range(3):
                    if ((ky+1)%2,(kx+1)%2)==(py,px):w[:,channels,ky,kx]=0
    return w.reshape(16,864)


def check_baseline(data,points,value):
    u,g,p=value
    eu=np.stack([data['full_updated_I24'][:,:,2*y,2*x] for y,x in points])
    ep=np.stack([data['full_continuous_q24'][:,:,y,x] for y,x in points])
    ew=np.stack([data['full_proj_words'][:,2*y,2*x] for y,x in points])
    eg=np.stack([(ew>>t)&1 for t in range(10)],axis=1)
    result=dict(updated_diff=int(np.count_nonzero(u!=eu)),gate_diff=int(np.count_nonzero(g!=eg)),PED_diff=int(np.count_nonzero(p!=ep)))
    assert not any(result.values()),result
    return result


def main():
    calibration=[(3+14*y,3+19*x) for y in range(8) for x in range(8)]
    validation=[(10+14*y,12+19*x) for y in range(8) for x in range(8)]
    assert not(set(calibration)&set(validation))
    result=dict(scope=__doc__,calibration_points=calibration,validation_points=validation,score='PED relative squared error + projection-gate flip fraction; fixed unit weights, no tuning. Individual damage selection; combined mask tested after selection.',removed_channels_per_phase=12,axes={},timing=False,AEE=False,training=False)
    masks={}
    for axis in ['ordinary','lifting_raw']:
        path=FULL/'capture_full_producers'/axis
        with np.load(path/'000_zurich_city_09_a_0001.npz') as z:data={k:z[k] for k in ['full_sn2_words','full_I24','full_updated_I24','full_continuous_q24','full_proj_words']}
        with np.load(path/'parameters.npz') as z:q={k:z[k] for k in z.files}
        x,raw=features(data,calibration);xu=x @ q['U_conv2_theta_q16'].T
        ref=downstream(xu,raw,q);check=check_baseline(data,calibration,ref)
        damages=np.empty((4,96));detail=[]
        for ph in range(4):
            kyx=[(ky,kx) for ky in range(3) for kx in range(3) if ((ky+1)%2)*2+(kx+1)%2==ph]
            for c in range(96):
                cols=[c*9+ky*3+kx for ky,kx in kyx]
                du=x[:,:,cols] @ q['U_conv2_theta_q16'][:,cols].T
                m=metrics(downstream(xu-du,raw,q),ref);damages[ph,c]=m['score']
                detail.append(dict(phase=ph,channel=c,**m))
        drop=np.zeros((4,96),bool)
        for ph in range(4):drop[ph,np.argsort(damages[ph],kind='stable')[:12]]=True
        # Global channel control drops 12 channels across every spatial phase,
        # same total 48/384 producer phase-channel classes; select on its own
        # exact joint downstream errors, not simply sum phase scores.
        global_damage=[]
        for c in range(96):
            cols=list(range(c*9,c*9+9));du=x[:,:,cols] @ q['U_conv2_theta_q16'][:,cols].T
            global_damage.append(metrics(downstream(xu-du,raw,q),ref)['score'])
        global_drop=np.zeros((4,96),bool);global_drop[:,np.argsort(global_damage,kind='stable')[:12]]=True
        # Magnitude control has the identical legal phase mask family/emitter.
        mag_drop=np.zeros((4,96),bool)
        w=q['U_conv2_theta_q16'].reshape(16,96,3,3).astype(np.float64)
        for ph in range(4):
            score=np.zeros(96)
            for ky in range(3):
                for kx in range(3):
                    if ((ky+1)%2)*2+(kx+1)%2==ph:score+=np.sum(w[:,:,ky,kx]**2,axis=0)
            mag_drop[ph,np.argsort(score,kind='stable')[:12]]=True
        vx,vr=features(data,validation);vref=downstream(vx @ q['U_conv2_theta_q16'].T,vr,q)
        vcheck=check_baseline(data,validation,vref)
        rec=dict(baseline_calibration_check=check,baseline_validation_check=vcheck,controls={})
        for name,mask in [('phase_joint',drop),('global_channel12',global_drop),('phase_magnitude',mag_drop)]:
            mw=mask_weights(q,mask)
            cal=metrics(downstream(x @ mw.T,raw,q),ref)
            val=metrics(downstream(vx @ mw.T,vr,q),vref)
            rec['controls'][name]=dict(calibration=cal,validation=val,weight_nnz_before=int(np.count_nonzero(q['U_conv2_theta_q16'])),weight_nnz_after=int(np.count_nonzero(mw)),deleted_phase_channel_classes=int(mask.sum()),physical_issue_savings_unknown=True)
            masks.setdefault(axis,{})[name]=dict(drop_channels=[np.flatnonzero(row).tolist() for row in mask],mask_uint8=mask.astype(int).tolist())
        rec['individual_damage']=detail;result['axes'][axis]=rec
        print(axis,{k:v['validation'] for k,v in rec['controls'].items()},flush=True)
    (HERE/'phase_channel_masks.json').write_text(json.dumps(masks,indent=2)+'\n')
    (HERE/'phase_channel_probe.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
