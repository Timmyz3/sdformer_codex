"""One original real P2 tile checks the reusable U/V MAC interface."""
import json
import numpy as np
from probe import FULL,HERE,ProbeMachine,resident_mac,pack24,read24,rne_sat

p=FULL/'capture_full_producers/ordinary'
with np.load(p/'parameters.npz') as z:
    q={k:z[k] for k in z.files}
with np.load(p/'000_zurich_city_09_a_0001.npz') as z:
    x=np.stack([z['full_updated_I24'][:,:,0,v] for v in (0,2)]).astype(np.int64)
    captured=np.stack([z['full_continuous_q24'][:,:,0,v] for v in (0,1)]).astype(np.int64)
u=q['U_ped_q16'].astype(np.int64);v=q['V_ped_q16'].astype(np.int64)
bias=q['PED_bias_q24'].astype(np.int64)
expected_u=rne_sat(x@u.T,int(q['U_ped_exponent']))
expected_v=np.clip(rne_sat(expected_u@v.T,int(q['V_ped_exponent']))+bias,-(1<<23),(1<<23)-1)
assert np.array_equal(expected_v,captured)
rows=[]
for stress in (False,True):
    m=ProbeMachine(stress)
    blob=u.T.astype('<i2').tobytes()+v.T.astype('<i2').tobytes()+bias.astype('<i4').tobytes()
    base={'U_ped':0,'V_ped':u.size*2,'PED_bias':(u.size+v.size)*2}
    m.dma_input(pack24(x),0);m.dma_input(blob,0,True)
    resident_mac(m,base,'U_ped',96,32,0,8192,2,int(q['U_ped_exponent']))
    au=read24(m,8192,(2,10,32))
    resident_mac(m,base,'V_ped',32,96,8192,16384,2,int(q['V_ped_exponent']),bias='PED_bias')
    av=read24(m,16384,(2,10,96))
    assert np.array_equal(au,expected_u)
    assert np.array_equal(av,captured)
    rows.append(dict(stress=stress,U_values=au.size,PED_values=av.size,U_differences=0,PED_differences=0,
                     scope='Callable smoke check on one original real P2; not a full-chain performance table.'))
(HERE/'callable_check.json').write_text(json.dumps(rows,indent=2)+'\n')
print('CALLABLE_U_V_REAL_P2_PASS',flush=True)
