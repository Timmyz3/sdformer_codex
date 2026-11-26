import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import run_native_scores as run


def fixture(core):
    records=[]
    for i,tid in enumerate(core.IDS):
        for arm in ('native','FF','CC')+tuple('fine_%d_%d'%p for p in core.TILES):
            records.append(dict(target_id=tid,arm=arm,execution='native' if arm=='native' else 'compact',quality_endpoint=True,endpoint_arm=arm))
            if i==0 and arm=='CC':
                records.extend(dict(target_id=tid,arm='CC_repeat%d'%j,execution='compact',quality_endpoint=False,endpoint_arm='CC') for j in (1,2))
    for i,r in enumerate(records):
        r.update(forward_id=i+1,prediction_file_sha256='a'*64,tensor_records={'u':{'sha256':r['target_id']}})
    result=dict(schema='flow_r19_compact_graph_v1',status='COMPACT_GRAPH122_COMPLETE',graph_id=run.GRAPH,
         binding=dict(graph_id=run.GRAPH,source_freeze_sha256='freeze',h8_sha256=run.H8_SHA,BN_mode='NATIVE_DENSE_CURRENT'),
         full_forward_count=122,full_forward_attempts=122,initial_allcurrent_state_sha256=run.STATE_SHA,
         final_state_sha256=run.STATE_SHA,all_states_unchanged=True,all_hooks_removed=True,original_modules_restored=True,
         forward_records=records)
    audit=dict(schema='flow_r19_compact_graph_independent_execution_audit_v1',status='PASS',quality_analysis_allowed=True,
          graph_id=run.GRAPH,source_result_sha256='result',source_freeze_sha256='freeze',
          counts=dict(full_forwards=122,native=8,compact=114,expanded=0,quality_endpoints=120,determinism_replays=2,
            FC1_rows=67021520,FC2_rows=67021520,BN_calls=9516,archives=122,head_payloads=488,capture_payloads=98,
            optimizer_steps=0,calibration_forwards=0),
          numerical_checks=[dict(pass_=True,kind=k) for k in ['allfine_vs_native']*8+['CC_exact_replay']*2],
          predictions=[{k:r[k] for k in ('forward_id','target_id','arm','execution','quality_endpoint','endpoint_arm')}|
                       {'file_sha256':r['prediction_file_sha256']} for r in records])
    for check in audit['numerical_checks']:check['pass']=check.pop('pass_')
    return result,audit


class Checks(unittest.TestCase):
    def test_exact_frozen_math_and_synthetic_policy(self):
        core=run.math_module();self.assertEqual(run.sha(run.MATH),run.MATH_SHA)
        u=np.zeros((10,1,30,40,1),np.float32);s=np.zeros_like(u)
        result=core.score_frame(u,s)
        self.assertEqual(len(result['policies']),20)
        for name in core.SCORES:self.assertEqual(result['policies'][name]['selected_tile'],[0,0])

    def test_own_complete_graph_admission_and_rejections(self):
        core=run.math_module();r,a=fixture(core)
        self.assertEqual([x['target_id'] for x in run.validate_admission(r,a,'result','freeze',core)],list(core.IDS))
        for kind in ('r18','graph','prefix','count','numeric','rawsha','u','dup','repeat_quality','state','freezes','audit_status'):
            rr,aa=copy.deepcopy(r),copy.deepcopy(a)
            if kind=='r18':rr['schema']='flow_r18_g0_cached_ordinary_v1'
            elif kind=='graph':aa['graph_id']='r17_dense'
            elif kind=='prefix':aa['quality_analysis_allowed']=False
            elif kind=='count':aa['counts']['full_forwards']=120
            elif kind=='numeric':aa['numerical_checks'][0]['pass']=False
            elif kind=='rawsha':aa['predictions'][0]['file_sha256']='bad'
            elif kind=='u':rr['forward_records'][2]['tensor_records']['u']['sha256']='bad'
            elif kind=='dup':rr['forward_records'][1]['arm']='native'
            elif kind=='repeat_quality':rr['forward_records'][3]['quality_endpoint']=True
            elif kind=='state':rr['final_state_sha256']='bad'
            elif kind=='freezes':aa['source_freeze_sha256']='bad'
            else:aa['status']='PASS_PREFIX_EVIDENCE_ONLY'
            with self.assertRaises(ValueError,msg=kind):run.validate_admission(rr,aa,'result','freeze',core)

    def test_two_payload_lazy_loading_and_binding(self):
        core=run.math_module()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'native.npz';u=np.zeros((10,1,30,40,384),np.float32)
            forbidden=np.array([{'do_not_load':True}],dtype=object)
            np.savez_compressed(path,u=u,sn1=u,**{k:forbidden for k in ('head0','head1','head2','head3','branch','sn2')})
            record=dict(prediction_file=path.name,prediction_file_sha256=run.sha(path),tensor_records={
                k:dict(shape=list(u.shape),dtype='torch.float32',sha256=core.payload_sha(u)) for k in ('u','sn1')})
            a,b=run.load_native(path,record,core);self.assertEqual(a.shape,u.shape);self.assertEqual(b.dtype,np.float32)
            bad=copy.deepcopy(record);bad['tensor_records']['sn1']['dtype']='torch.float64'
            with self.assertRaises(ValueError):run.load_native(path,bad,core)
            bad=copy.deepcopy(record);bad['prediction_file_sha256']='bad'
            with self.assertRaises(ValueError):run.load_native(path,bad,core)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args();assert not args.output.exists()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    out=dict(schema='flow_r19_native_score_cpu_v1',status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,
             failures=len(result.failures),errors=len(result.errors),source_sha256={n:run.sha(run.HERE/n) for n in run.OWN},
             math_sha256=run.MATH_SHA,actual_R19_arrays_loaded=False,GT_or_quality_loaded=False,GPU_calls=0,model_forwards=0)
    args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
    raise SystemExit(0 if result.wasSuccessful() else 1)
