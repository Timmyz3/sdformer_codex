"""One pre-V conservative certificate at the true gate-only sn2 boundary.

S1 opportunity measurement, not service timing. Grouped interval propagation
and suffix norm costs are deliberately NOT called free hardware speedups.
"""
import json
import numpy as np
from run_chain import HERE,read_npz,tf32_round
from machine import vector_fma


def probe(data,p,label):
    z=tf32_round(data[label+'_preview_Z_shared'])
    h,w=z.shape[2:];max_z=np.max(np.abs(z.astype(np.float64)),axis=1)
    weights=tf32_round(p['preview_v'][:32])
    weight_l1=np.sum(np.abs(weights.astype(np.float64)),axis=0)
    # FMA summation error, including an absolute underflow allowance. The
    # dyadic coefficient sum and captured TF32 max are exactly representable
    # at these measured exponent ranges; each following FP64 step rounds up.
    gamma=np.nextafter(32/(2**24-32),np.inf)
    radius=np.nextafter(max_z[:,None,:,:]*weight_l1[None,:,None,None],np.inf)
    radius=np.nextafter(radius+np.nextafter(gamma*radius,np.inf)+32*2**-149,np.inf)
    # Actual zero dot is an ordinary exact-zero baseline, with no round error.
    radius=np.where(max_z[:,None,:,:]==0,0,radius)
    low=np.nextafter(np.float32(-radius),np.float32(-np.inf))
    high=np.nextafter(np.float32(radius),np.float32(np.inf))
    low=np.where(radius==0,np.float32(0),low);high=np.where(radius==0,np.float32(0),high)
    inv=np.float32(1)/np.sqrt(np.float32(p['bn1_var'])+np.float32(p['bn1_eps']))
    scale=np.float32(p['bn1_gamma']*inv)
    bias=np.float32(p['bn1_beta']-np.float32(p['bn1_mean']*scale))
    aa=np.float32(low*scale[None,:,None,None]);bb=np.float32(high*scale[None,:,None,None])
    low=np.float32(np.minimum(aa,bb)+bias[None,:,None,None])
    high=np.float32(np.maximum(aa,bb)+bias[None,:,None,None])
    lower=np.zeros_like(low);upper=np.zeros_like(high)
    for t in range(10):
        for s in range(10):
            a=p['preview_A'][t,s]
            if a==0:continue
            lower[t]=vector_fma((low[s] if a>0 else high[s]).reshape(-1),a,lower[t].reshape(-1)).reshape(96,h,w)
            upper[t]=vector_fma((high[s] if a>0 else low[s]).reshape(-1),a,upper[t].reshape(-1)).reshape(96,h,w)
    b=p['preview_b'][:,None,None,None];theta=np.float32(p['preview_theta_output'])
    lower=np.float32(np.float32(lower+b)-theta);upper=np.float32(np.float32(upper+b)-theta)
    certified=(lower>=0)|(upper<0);predicted=lower>=0;gold=data[label+'_sn2_gate']
    errors=int(np.count_nonzero(certified&(predicted!=gold)));assert errors==0
    empty_position=np.all(max_z==0,axis=0)
    # Real P2 grouping, including one P1 at the odd interior edge.
    groups=accepted=baseline=additional=active_v_slots=saved_v_slots=0
    detail=[]
    for y in range(h):
        for x in range(0,w,2):
            count=min(2,w-x);ordinary_constant=bool(empty_position[y,x:x+count].all())
            active=int(np.count_nonzero(max_z[:,y,x:x+count]))
            selected=[]
            for h0 in range(0,96,8):
                accept=bool(certified[:,h0:h0+8,y,x:x+count].all())
                groups+=1;accepted+=accept;baseline+=ordinary_constant
                additional+=accept and not ordinary_constant
                active_v_slots+=32*active
                if accept:saved_v_slots+=32*active
                if accept:selected.append(h0)
            if selected:detail.append(dict(y=y,x=x,positions=count,accepted_H8=selected,ordinary_allzero=ordinary_constant))
    return dict(gate_bits=int(gold.size),certified_gate_bits=int(certified.sum()),certificate_errors=errors,
        P2_H8_groups=groups,certified_groups=accepted,ordinary_constant_groups=baseline,additional_groups=additional,
        active_V_vector_FMAs=active_v_slots,removable_V_vector_FMAs_before_certificate_cost=saved_v_slots,
        opportunity_only=True,net_service_reduction=None,detail=detail,
        decision='Stop this pre-V L_inf envelope; retain other ranges/interfaces.' if additional==0 else 'Need exact certificate cost and strongest same-group interval baseline before promotion.')


def main():
    result=dict(scope=__doc__,axes={},new_training=False,new_quantization=False,RTL=False)
    for axis in ('ordinary','lifting_raw'):
        path=HERE.parent/'capture'/axis;data=read_npz(path/'000_zurich_city_09_a_0001.npz');p=read_npz(path/'live_parameters.npz')
        result['axes'][axis]={}
        for label in ('corner','interior'):
            r=probe(data,p,label);result['axes'][axis][label]=r
            print(axis,label,{k:v for k,v in r.items() if k!='detail'},flush=True)
    (HERE/'latent_gate_probe.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
