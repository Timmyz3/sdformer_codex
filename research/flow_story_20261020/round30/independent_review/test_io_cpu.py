import copy
from pathlib import Path
import unittest
import numpy as np
import reduce_joint_raw_v1 as r

def fixture():
    ids=[str(i) for i in range(64)]
    f=dict(auditor={'sha256':'auditor'},auditor_freeze={'sha256':'freeze'})
    a=dict(schema=r.AUDIT_SCHEMA,status='PASS',quality_analysis_allowed=True,
        engineering_smoke=False,auditor_sha256='auditor',auditor_freeze_sha256='freeze',
        graph_id=r.GRAPH,source_result_sha256='result',source_mask_contract_sha256='mask',
        counts=dict(optimizer_updates=512,quality_predictions=256,formal_model_forwards=1289),
        student_mode_counts=dict(Q4=256,Q8=128,M=128),
        predictions=[dict(state=label,target_id=i) for _,label in r.STAGES.values() for i in ids])
    return ids,f,a

class Test(unittest.TestCase):
    def test_actual_admission_complete_and_rejections(self):
        ids,f,a=fixture()
        self.assertEqual(len(r.validate_audit(a,f,ids,'result','mask')),256)
        mutations=[
            lambda x:x.update(engineering_smoke=True),
            lambda x:x.update(quality_analysis_allowed=False),
            lambda x:x.update(status='PARTIAL'),
            lambda x:x['counts'].update(optimizer_updates=511),
            lambda x:x['counts'].update(quality_predictions=255),
            lambda x:x['counts'].update(formal_model_forwards=1288),
            lambda x:x.update(source_mask_contract_sha256='other'),
            lambda x:x.update(graph_id='old_graph'),
            lambda x:x.update(auditor_sha256='other'),
            lambda x:x.update(auditor_freeze_sha256='other'),
            lambda x:x['predictions'].__setitem__(0,copy.deepcopy(x['predictions'][1])),
            lambda x:x['predictions'][0].update(state='recovered_Q4'),
            lambda x:x['student_mode_counts'].update(Q4=255),
        ]
        for corrupt in mutations:
            b=copy.deepcopy(a);corrupt(b)
            with self.assertRaises(ValueError):r.validate_audit(b,f,ids,'result','mask')
    def test_new_state_mapping_and_relative_paths(self):
        self.assertEqual({s for s,_ in r.STAGES.values()},set(r.core.NEW_STATES))
        self.assertEqual({s for _,s in r.STAGES.values()},
            {'initial_M','recovered_joint_Q4','recovered_joint_Q8','recovered_joint_M'})
        self.assertEqual(r.safe(Path('/tmp/actual'),'predictions/a.npz'),Path('/tmp/actual/predictions/a.npz'))
        for bad in ('../old.npz','/tmp/outside.npz'):
            with self.assertRaises(ValueError):r.safe(Path('/tmp/actual'),bad)
    def test_actual_raw_metric_signed_direction_and_empty(self):
        gt=np.zeros((2,3,4),dtype=np.float32);pred=gt.copy();pred[0,:,:]=3;pred[1,:,:]=4
        valid=np.ones((3,4),dtype=bool)
        masks={k:valid.copy() for k in r.core.old.REGIONS};masks['B'][:]=False
        measured=r.raw.measure(pred,gt,masks)
        self.assertEqual(measured['all_valid'],dict(valid_pixels=12,error_sum=60.,AEE=5.))
        self.assertEqual(measured['B'],dict(valid_pixels=0,error_sum=0.,AEE=None))
        self.assertEqual(r.core.old.ratio(5.,4.),.25)

if __name__=='__main__':unittest.main()
