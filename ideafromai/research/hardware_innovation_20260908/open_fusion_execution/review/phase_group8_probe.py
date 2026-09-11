"""Physical-H8 variant: each phase removes 2 of 12 consecutive H8 groups.

One fixed follow-up to scattered C12 masks, not a parameter sweep. Reuses the
same 64 calibration / 64 separate-grid anchors, exact downstream arithmetic,
score, and controls. No arbitrary channel repacking is assumed.
"""
import json
import numpy as np
from phase_channel_probe import HERE,FULL,features,downstream,metrics,mask_weights,check_baseline


def expand(groups):
    return np.repeat(groups,8,axis=1)


def main():
    calibration=[(3+14*y,3+19*x) for y in range(8) for x in range(8)]
    validation=[(10+14*y,12+19*x) for y in range(8) for x in range(8)]
    result=dict(scope=__doc__,calibration_points=calibration,validation_points=validation,score='PED relative squared error + projection-gate flip fraction; fixed unit weights, no tuning. Individual H8 damage then combined mask measured.',removed_groups_per_phase=2,group_width=8,axes={},timing=False,AEE=False,training=False)
    masks={}
    for axis in ['ordinary','lifting_raw']:
        path=FULL/'capture_full_producers'/axis
        with np.load(path/'000_zurich_city_09_a_0001.npz') as z:data={k:z[k] for k in ['full_sn2_words','full_I24','full_updated_I24','full_continuous_q24','full_proj_words']}
        with np.load(path/'parameters.npz') as z:q={k:z[k] for k in z.files}
        x,raw=features(data,calibration);xu=x @ q['U_conv2_theta_q16'].T
        ref=downstream(xu,raw,q);check=check_baseline(data,calibration,ref)
        damages=np.empty((4,12));detail=[]
        for ph in range(4):
            kyx=[(ky,kx) for ky in range(3) for kx in range(3) if ((ky+1)%2)*2+(kx+1)%2==ph]
            for group in range(12):
                cols=[c*9+ky*3+kx for c in range(group*8,group*8+8) for ky,kx in kyx]
                du=x[:,:,cols] @ q['U_conv2_theta_q16'][:,cols].T
                m=metrics(downstream(xu-du,raw,q),ref);damages[ph,group]=m['score']
                detail.append(dict(phase=ph,group8=group,**m))
        selected=np.zeros((4,12),bool)
        for ph in range(4):selected[ph,np.argsort(damages[ph],kind='stable')[:2]]=True
        global_damage=[]
        for group in range(12):
            cols=list(range(group*8*9,(group*8+8)*9));du=x[:,:,cols] @ q['U_conv2_theta_q16'][:,cols].T
            global_damage.append(metrics(downstream(xu-du,raw,q),ref)['score'])
        global_selected=np.zeros((4,12),bool);global_selected[:,np.argsort(global_damage,kind='stable')[:2]]=True
        mag=np.zeros((4,12),bool);w=q['U_conv2_theta_q16'].reshape(16,96,3,3).astype(np.float64)
        for ph in range(4):
            score=np.zeros(96)
            for ky in range(3):
                for kx in range(3):
                    if ((ky+1)%2)*2+(kx+1)%2==ph:score+=np.sum(w[:,:,ky,kx]**2,axis=0)
            score=score.reshape(12,8).sum(1);mag[ph,np.argsort(score,kind='stable')[:2]]=True
        vx,vr=features(data,validation);vref=downstream(vx @ q['U_conv2_theta_q16'].T,vr,q)
        vcheck=check_baseline(data,validation,vref)
        rec=dict(baseline_calibration_check=check,baseline_validation_check=vcheck,controls={},individual_damage=detail)
        for name,group_mask in [('phase_joint',selected),('global_group2',global_selected),('phase_magnitude',mag)]:
            mask=expand(group_mask);mw=mask_weights(q,mask)
            cal=metrics(downstream(x @ mw.T,raw,q),ref);val=metrics(downstream(vx @ mw.T,vr,q),vref)
            rec['controls'][name]=dict(calibration=cal,validation=val,weight_nnz_before=int(np.count_nonzero(q['U_conv2_theta_q16'])),weight_nnz_after=int(np.count_nonzero(mw)),deleted_phase_group_classes=int(group_mask.sum()),deleted_phase_channel_classes=int(mask.sum()),physical_issue_savings_unknown=True)
            masks.setdefault(axis,{})[name]=dict(drop_groups=[np.flatnonzero(row).tolist() for row in group_mask],drop_channels=[np.flatnonzero(row).tolist() for row in mask],mask_uint8=mask.astype(int).tolist(),group_mask_uint8=group_mask.astype(int).tolist())
        result['axes'][axis]=rec
        print(axis,{k:v['validation'] for k,v in rec['controls'].items()},flush=True)
    (HERE/'phase_group8_masks.json').write_text(json.dumps(masks,indent=2)+'\n')
    (HERE/'phase_group8_probe.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
