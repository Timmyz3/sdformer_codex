"""Close the selected ordinary fixed-R24 service control after real AEE.

Same four existing P2/pressure points; exact same callable resident MAC and
port machine as rebase_service.py. The prior ordinary R32 trace is the control.
"""
from pathlib import Path
import sys,json
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from rebase_service import ped,FULL

with np.load(HERE.parent/'ordinary_rebase_parameters.npz') as z:
    u=z['original_ordered_U'][:24];v=z['original_ordered_V'][:,:24];bias=z['PED_bias_q24']
with np.load(HERE/'cpu_fixture.npz') as z:fixture={k:z[k] for k in z.files}
reference=json.loads((HERE.parent/'rebase_service.json').read_text())
rows=[]
for window in ('corner','interior'):
    x=fixture['ordinary_'+window+'_x'][:,:,0,:2].transpose(2,0,1).astype(np.int64)
    expected=fixture['ordinary_'+window+'_original_ordered24'][:,:,0,:2].transpose(2,0,1)
    for stress in (False,True):
        m=ped.ProbeMachine(stress)
        packed=u.T.astype('<i2').tobytes()+v.T.astype('<i2').tobytes()+bias.astype('<i4').tobytes()
        base={'U_ped':0,'V_ped':u.size*2,'PED_bias':(u.size+v.size)*2}
        m.dma_input(ped.pack24(x),0);m.dma_input(packed,0,True)
        cold=m.time
        ped.resident_mac(m,base,'U_ped',96,24,0,8192,2,16)
        ped.resident_mac(m,base,'V_ped',24,96,8192,16384,2,15,bias='PED_bias')
        out=ped.read24(m,16384,(2,10,96))
        assert np.array_equal(out,expected)
        m.phase='common_output_egress'
        for off in range(0,5760,32):
            for j in (0,8,16,24):m.read_word(16384+off+j)
            for _ in range(5):m.advance(tag='DMA_output_slots')
        ref=next(r for r in reference['rows'] if r['axis']=='ordinary' and r['window']==window and r['stress']==stress and r['arm']=='original32')
        rows.append(dict(window=window,stress=stress,arm='original_ordered24',values=out.size,differences=0,
            service_slots=m.time,cold_fill_slots=cold,baseline_service_slots=ref['service_slots'],
            service_reduction=1-m.time/ref['service_slots'],phases=dict(m.stages),counts=dict(m.count),
            timeline=m.timeline))
        print(window,stress,m.time,'PASS',flush=True)
(HERE/'service_original24.json').write_text(json.dumps(dict(scope=__doc__,evidence='CPU finite payload-executing issue/port model only; not VCS/PPA/full network.',rows=rows),indent=2)+'\n')
