"""Before GPU, compare the actual helper methods against saved NumPy q24."""
from pathlib import Path
import sys,json
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CHAIN=ROOT/'algorithm/patch_probe/residual_consumer_probe/projection_chain'
FULL=CHAIN/'fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain'
sys.path.insert(0,str(CHAIN))
sys.path.insert(0,str(HERE.parent))
from fixed_temporal_coordinates import FixedTemporalForward
from rebase_probe import execute
from ped_rebase_adapter import select,MODES,check_fixture

fixture={};rows=[]
for axis in ('ordinary','lifting_raw'):
    with np.load(FULL/'capture'/axis/'parameters.npz') as z:original={k:z[k] for k in z.files}
    with np.load(HERE.parent/(axis+'_rebase_parameters.npz')) as z:q={k:z[k] for k in z.files}
    with np.load(FULL/'capture_full_producers'/axis/'000_zurich_city_09_a_0001.npz') as z:full=z['full_updated_I24']
    for window,(y,x) in [('corner',(0,0)),('interior',(120,160))]:
        inp=full[:,:,y:y+8:2,x:x+8:2].copy()
        key=axis+'_'+window
        fixture[key+'_x']=inp
        flat=inp.transpose(1,0,2,3).reshape(96,-1)
        for mode in MODES:
            u,v=select(q,mode,original)
            output,_=execute(u,v,flat,original['PED_bias_q24'])
            fixture[key+'_'+mode]=output.reshape(96,10,4,4).transpose(1,0,2,3).astype(np.int32)
    helper=object.__new__(FixedTemporalForward)
    helper.device=torch.device('cpu');helper.matrices={};helper.matrix_metadata={};helper.constants={}
    helper.projection_bias=torch.as_tensor(original['PED_bias_q24'],dtype=torch.float64)
    rows.extend(check_fixture(helper,q,original,fixture,axis))
np.savez_compressed(HERE/'cpu_fixture.npz',**fixture)
(HERE/'cpu_check.json').write_text(json.dumps(dict(scope=__doc__,rows=rows,all_PASS=True),indent=2)+'\n')
print('EXACT_CPU_WRAPPER_PASS',sum(r['values'] for r in rows),flush=True)
