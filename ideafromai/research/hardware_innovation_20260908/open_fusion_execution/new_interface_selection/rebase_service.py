"""Static calibrated R24 vs original R32: finite original PED U/V machine.

No dynamic oracle, gate/BN/projection producer or system claim. Both arms
use the already-preserved stronger resident-MAC implementation unmodified.
"""
import importlib.util
import json
import numpy as np
from rebase_probe import FULL,HERE,execute

spec=importlib.util.spec_from_file_location('ped_callable',HERE.parent/'ped_bitplanes/probe.py')
ped=importlib.util.module_from_spec(spec)
spec.loader.exec_module(ped)


def main():
    rows=[]
    for axis in ('ordinary','lifting_raw'):
        with np.load(FULL/'capture'/axis/'parameters.npz') as z:q={k:z[k] for k in z.files}
        with np.load(HERE/(axis+'_rebase_parameters.npz')) as z:nq={k:z[k] for k in z.files}
        with np.load(FULL/'capture_full_producers'/axis/'000_zurich_city_09_a_0001.npz') as z:
            full=z['full_updated_I24']
        for label,(y,x0) in [('corner',(0,0)),('interior',(120,160))]:
            x=np.stack([full[:,:,y,x0],full[:,:,y,x0+2]]).astype(np.int64)
            for stress in (False,True):
                for name,rank,u,v in [('original32',32,q['U_ped_q16'],q['V_ped_q16']),
                        ('activation_whitened24',24,nq['activation_whitened_U'][:24],nq['activation_whitened_V'][:,:24])]:
                    m=ped.ProbeMachine(stress)
                    bias=q['PED_bias_q24']
                    packed=u.T.astype('<i2').tobytes()+v.T.astype('<i2').tobytes()+bias.astype('<i4').tobytes()
                    base={'U_ped':0,'V_ped':u.size*2,'PED_bias':(u.size+v.size)*2}
                    m.dma_input(ped.pack24(x),0)
                    m.dma_input(packed,0,True)
                    cold=m.time
                    ped.resident_mac(m,base,'U_ped',96,rank,0,8192,2,16)
                    ped.resident_mac(m,base,'V_ped',rank,96,8192,16384,2,15,bias='PED_bias')
                    actual=ped.read24(m,16384,(2,10,96))
                    ref,_=execute(u,v,x.transpose(2,0,1).reshape(96,-1),bias)
                    ref=ref.reshape(96,2,10).transpose(1,2,0)
                    assert np.array_equal(actual,ref),(axis,label,stress,name)
                    m.phase='common_output_egress'
                    for off in range(0,2*10*96*3,32):
                        for j in (0,8,16,24):m.read_word(16384+off+j)
                        for _ in range(5):m.advance(tag='DMA_output_slots')
                    row=dict(axis=axis,window=label,stress=stress,arm=name,rank=rank,
                        service_slots=m.time,cold_fill_slots=cold,body_and_output_slots=m.time-cold,
                        phases=dict(m.stages),counts=dict(m.count),input_bytes=x.size*3,
                        coefficient_bytes=len(packed),output_bytes=2*10*96*3,
                        latent_materialized_bytes=2*10*rank*3,
                        state_region_end_bytes=16384+2*10*96*3,RF_capacity_vectors=96,
                        used_accumulator_vectors_max=80,
                        U_accumulator_vectors=20*(rank//8),V_accumulator_vectors=80,
                        source_cache_vectors=4,checked_values=actual.size,value_differences=0,
                        timeline=m.timeline)
                    rows.append(row)
                    print(axis,label,stress,name,m.time,'PASS',flush=True)
    pairs=[]
    for i in range(0,len(rows),2):
        a,b=rows[i:i+2]
        pairs.append(dict(axis=a['axis'],window=a['window'],stress=a['stress'],
            base_slots=a['service_slots'],candidate_slots=b['service_slots'],
            service_reduction=1-b['service_slots']/a['service_slots'],
            body_and_output_reduction=1-b['body_and_output_slots']/a['body_and_output_slots']))
    result=dict(scope=__doc__,evidence='CPU payload-executing finite issue/port/response model; NOT VCS/RTL/PPA or complete-network performance.',
        resource=dict(lanes=8,accumulator_bits=48,integer_latency=2,shared_issue_per_slot=1,
            SR_bytes_per_slot=8,SW_bytes_per_slot=8,coefficient_read_bytes_per_slot=32,
            SRAM_bytes=131072,coefficient_SRAM_bytes=131072,RF_vectors=96,
            original_resident_MAC='../ped_bitplanes/probe.py:resident_mac'),
        same_R24_other_basis_contract='Weight-SVD and original-order static R24 have the same loop/control and physical resource privileges; not separate titles.',
        rows=rows,pairs=pairs)
    (HERE/'rebase_service.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(pairs,indent=2))


if __name__=='__main__':main()
