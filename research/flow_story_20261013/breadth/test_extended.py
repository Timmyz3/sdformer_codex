import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys
sys.dont_write_bytecode=True
import ast
import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
import evaluate_breadth_extended as x
from test_evaluate_breadth import Checks, ART, HERE


class ExtendedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Checks.setUpClass();q=Checks
        cls.config=q().validate()
        cls.context=(q.native,q.trainer,q.metrics,q.rows,q.hashes,cls.config,q.snapshot_run,q.side,None,{'source_sha256':q.sources})
        cls.args=SimpleNamespace(bn_schedule=ART/'bn_calibration_schedule.json',canonical=ART/'common_targets.json',
            roles=ART/'train_auxiliary_roles.csv',checkpoint_sha256=q.side['checkpoint_sha256'],geometry='full9',
            prior_root=HERE.parents[1]/'flow_story_20261012',bn_payload=None,svs_payload=None,
            svs_collector_sha256=x.SVS_COLLECTOR_SHA,static_extension_sha256=x.STATIC_EXTENSION_SHA)

    def fixture(self):
        model=torch.nn.Sequential(torch.nn.Linear(3,3),torch.nn.BatchNorm1d(3))
        ids=x.read_pin(self.args.bn_schedule,x.SCHEDULE_SHA)['stages']['initial']['target_ids']
        records={r['target_id']:r for r in Checks.records}
        before=x.non_bn_digest(model,{'1'})
        artifact={'schema':'flow_breadth_bn_calibration_v1','status':'COMPLETE','method':'cma_native_batch_stats_reset',
            'checkpoint_sha256':self.args.checkpoint_sha256,'data_contract_sha256':x.digest(self.config['data_contract']),
            'source_sha256':Checks.sources,'geometry':'full9','fit_target_ids':ids,'fit_target_ids_sha256':x.digest(ids),
            'fit_role':'bn_calibration','schedule_sha256':x.SCHEDULE_SHA,'fit_stage':'initial','completed_ids':ids,
            'fit_uses_final_valid':False,'optimizer_steps':0,'script_sha256':x.BN_COLLECTOR_SHA,'evaluator_sha256':x.BASE_SHA,
            'GPU_executed':True,'non_bn_state_unchanged':True,'source_non_bn_state_sha256':before,'after_non_bn_state_sha256':before,
            'input_evidence':[{'target_id':t,'relative':Checks.native.target_files(records[t])['input'],
                'sha256':Checks.hashes[Checks.native.target_files(records[t])['input']]['sha256']} for t in ids],
            'buffers':{'1':{'running_mean':[1.,2.,3.],'running_var':[0.,.5,2.],'num_batches_tracked':176}}}
        return model,artifact

    def test_bound_BN_accepts_fixed176_not_evaluation32(self):
        model,artifact=self.fixture();x.validate_bn_artifact(artifact,self.args,self.context)
        self.assertEqual(len(artifact['fit_target_ids']),176)
        self.assertFalse(set(artifact['fit_target_ids'])&set(Checks.sample['target_ids']))

    def test_wrong_BN_bindings_and_nonfinite_rejected(self):
        _,artifact=self.fixture()
        for key,value in [('status','RUNNING'),('geometry','crop_center9'),('checkpoint_sha256','f'*64),
            ('schedule_sha256','e'*64),('non_bn_state_unchanged',False),('script_sha256','0'*64),('completed_ids',artifact['completed_ids'][:-1])]:
            bad=copy.deepcopy(artifact);bad[key]=value
            with self.assertRaises(ValueError):x.validate_bn_artifact(bad,self.args,self.context)
        bad=copy.deepcopy(artifact);bad['input_evidence'][0]['sha256']='d'*64
        with self.assertRaisesRegex(ValueError,'native inventory'):x.validate_bn_artifact(bad,self.args,self.context)
        bad=copy.deepcopy(artifact);bad['buffers']['1']['running_var'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'variance'):x.validate_bn_artifact(bad,self.args,self.context)

    def test_only_BN_buffers_change_and_zero_variance_legal(self):
        model,artifact=self.fixture();before={k:v.clone() for k,v in model.state_dict().items()}
        report=x.apply_bn_buffers(model,artifact,torch)
        self.assertTrue(report['non_bn_state_unchanged']);self.assertEqual(model[1].num_batches_tracked.item(),176)
        for key in before:
            if key not in ('1.running_mean','1.running_var','1.num_batches_tracked'):self.assertTrue(torch.equal(before[key],model.state_dict()[key]))
        self.assertEqual(model[1].running_var[0].item(),0.)

    def test_model_name_shape_count_and_foreign_weights_rejected(self):
        for mutation in ('name','shape','negative','count','weights'):
            model,artifact=self.fixture();before=model[1].running_mean.clone()
            if mutation=='name':artifact['buffers']['bad']=artifact['buffers'].pop('1')
            if mutation=='shape':artifact['buffers']['1']['running_mean']=[0,1]
            if mutation=='negative':artifact['buffers']['1']['running_var'][0]=-1
            if mutation=='count':artifact['buffers']['1']['num_batches_tracked']=175
            if mutation=='weights':
                with torch.no_grad():model[0].weight.add_(1)
            with self.assertRaises(ValueError):x.apply_bn_buffers(model,artifact,torch)
            self.assertTrue(torch.equal(before,model[1].running_mean))

    def test_shared_inference_body_only_construction_changed(self):
        old=inspect.getsource(x.base.run_inference);new=inspect.getsource(x.run_inference)
        def remove_construction(text):
            start=text.index('            model.load_state_dict(state,strict=True)')
            stop=text.index('            model.to(device)',start)
            return text[:start]+text[stop:]
        self.assertEqual(remove_construction(old),remove_construction(new))
        self.assertEqual(x.sha(x.base.__file__),x.BASE_SHA)

    def test_SVS_dispatch_binds_actual_BN_and_source(self):
        model,_=self.fixture()
        with torch.no_grad():model[1].num_batches_tracked.fill_(3)
        args=copy.copy(self.args);ids=x.read_pin(HERE/'svs_gate_fit2.json',x.SVS_FIT_SHA)['target_ids']
        args.svs_payload={'fit_ids':ids,'evidence':{'evaluator_sha256':x.BASE_SHA,'static_methods_sha256':x.STATIC_SHA,
            'collector_sha256':x.SVS_COLLECTOR_SHA,'CPU_only_SVD':True,'cuda_initialized':False,'sample_count':2,
            'source_site':'sn2 binary output before drop2/fc2','score_input':'lossless uint8 counts; float64/10; no spatial approximation'}}
        captured={}
        class FakeMethods:
            def configure_prior_root(self,path):pass
            def build_static_candidate(self,model,name,paths,**kwargs):captured.update(kwargs);return model,{}
        with patch.object(x,'import_file',return_value=FakeMethods()):
            x.build_static_extended(model,{'static_method':'svs','bn':'D','keep_ratio':.5},args,self.context,torch)
        expected=captured['svs_expected_provenance']
        self.assertEqual(expected['checkpoint_sha256'],args.checkpoint_sha256)
        self.assertEqual(expected['source_sha256'],x.digest(Checks.sources))
        self.assertEqual(expected['fit_ids_sha256'],x.digest(ids))
        args.svs_payload['evidence']['collector_sha256']='0'*64
        with patch.object(x,'import_file',return_value=FakeMethods()),self.assertRaisesRegex(ValueError,'SVS source'):
            x.build_static_extended(model,{'static_method':'svs','bn':'D'},args,self.context,torch)


if __name__=='__main__':
    torch.set_num_threads(1)
    r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExtendedTests))
    (HERE/'extended_cpu_checks.json').write_text(json.dumps({'passed':r.wasSuccessful(),'tests':r.testsRun,
        'errors':len(r.errors),'failures':len(r.failures),'GPU_initialized':torch.cuda.is_initialized(),
        'extended_entry_sha256':x.sha(x.__file__),'frozen_base_sha256':x.sha(x.base.__file__),
        'scope':'Real protocol metadata plus synthetic BN fixtures and dispatch probes; no calibration/inference result'},indent=2)+'\n')
    raise SystemExit(0 if r.wasSuccessful() else 1)
