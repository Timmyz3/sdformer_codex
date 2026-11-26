"""Synthetic admission failures before any prediction/GT numerical access."""
import copy
import unittest
from pathlib import Path
from unittest.mock import patch
import reduce_raw_v1 as r

class Tests(unittest.TestCase):
    def setUp(self):
        self.members=r.read(r.HERE/'membership_v1.json')
        self.regions=r.read(r.REGIONS_PATH)
        self.freeze={'auditor':{'sha256':'auditor'},'audit_freeze':{'sha256':'freeze'}}
        self.inputs={'schema':'flow_r27_raw_quality_inputs_v1','execution_audit':{'path':'fake_audit','sha256':'actual'},
                     'runs':{k:{'path':k,'sha256':k} for k in r.ARMS},'source_input_root':'unopened_source'}
        self.audit={'schema':r.AUDIT_SCHEMA,'status':'PASS','quality_analysis_allowed':True,
                    'auditor_sha256':'auditor','auditor_freeze_sha256':'freeze','pre_result_freeze_sha256':r.ANALYSIS_SHA,
                    'counts':{'optimizer_updates':1536,'quality_predictions':384,'quality_archives':384},
                    'source_results':{k:k for k in r.ARMS},
                    'predictions':[{'state':s,'target_id':m['target_id']} for s in r.core.STATES for m in self.members['dev_members']]}
    def reject_without_arrays(self, audit=None, inputs=None):
        def fake_read(p):
            if Path(p)==r.HERE/'membership_v1.json':return self.members
            if Path(p)==r.REGIONS_PATH:return self.regions
            if Path(p)==Path('fake_audit'):return audit or self.audit
            raise AssertionError('Unexpected downstream input read '+str(p))
        def fake_sha(p):return r.MEMBERS_SHA if Path(p).name=='membership_v1.json' else r.REGIONS_SHA
        with patch.object(r,'read',fake_read),patch.object(r,'sha',fake_sha),\
             patch.object(r,'checked_descriptor',lambda d:Path(d['path'])),\
             patch.object(r.np,'load',side_effect=AssertionError('Numerical access before admission')):
            with self.assertRaises(ValueError):r.bind_artifacts(inputs or self.inputs,self.freeze)
    def test_incomplete_or_unbound_actual_rejected(self):
        for key,value in (('status','RUNNING'),('quality_analysis_allowed',False),('auditor_sha256','other'),('auditor_freeze_sha256','other'),('pre_result_freeze_sha256','other')):
            a=copy.deepcopy(self.audit);a[key]=value;self.reject_without_arrays(a)
        for key,value in (('optimizer_updates',1535),('quality_predictions',383),('quality_archives',385)):
            a=copy.deepcopy(self.audit);a['counts'][key]=value;self.reject_without_arrays(a)
    def test_all_384_unique_state_members_required(self):
        a=copy.deepcopy(self.audit);a['predictions'][-1]=copy.deepcopy(a['predictions'][0]);self.reject_without_arrays(a)
        a=copy.deepcopy(self.audit);a['predictions'][0]['target_id']=self.members['sealed_ids'][0];self.reject_without_arrays(a)
        a=copy.deepcopy(self.audit);a['predictions'][0]['state']='best_Q6';self.reject_without_arrays(a)
        i=copy.deepcopy(self.inputs);del i['runs']['native'];self.reject_without_arrays(inputs=i)
    def test_source_paths_cannot_leave_snapshot(self):
        for text in ('../sealed.npy','/absolute/gt.npy','x/../../event.npy'):
            with self.assertRaises(ValueError):r.safe_child('/allowed',text)
        self.assertEqual(r.safe_child('/allowed','saved_flow_data/train/gt.npy'),Path('/allowed/saved_flow_data/train/gt.npy'))

if __name__=='__main__':unittest.main()
