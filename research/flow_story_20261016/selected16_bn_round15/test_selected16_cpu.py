#!/usr/bin/env python3
"""Selected16 state/real tiny native hooks/source regressions; no GPU/quality."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import ast
import builtins
import copy
import dis
import hashlib
import json
from pathlib import Path
import sys
import types
import unittest
sys.dont_write_bytecode=True
import numpy as np
import torch
import run_selected16 as r
core=r.core
OLD,K,H0,RUNTIME=r.runtime();R13=RUNTIME[0]
require=core.require
require(r.sha(K.HERE/'test_deployment_cpu.py')=='22d5cc794d7abd9fbbc9b899b24b23dc3fe845b76bdb423669f679c34687b5cc','Pinned existing tiny model fixture')
sys.path.insert(0,str(K.HERE));FIX=r.load(K.HERE/'test_deployment_cpu.py','_r15_existing_tiny_model_fixture')
MEASUREMENTS=[]


def own_toy():
    model,targets,shapes,x,functional=FIX.toy()
    all_bn,private=FIX.FIX.scope.bn_inventory(model,torch);values={}
    for name in private:
        bn=all_bn[name]
        values[name+'.running_mean']=np.linspace(-.1,.1,bn.num_features,dtype=np.float32)
        values[name+'.running_var']=np.linspace(.7,1.3,bn.num_features,dtype=np.float32)
        values[name+'.num_batches_tracked']=np.array(176,dtype=np.int64)
    return model,targets,shapes,x,functional,values


def current_fixture():
    members={'check_members':[{'target_id':'id'+str(i),'sequence':'seq'+str(i%2)} for i in range(8)]}
    common={k:k+'_sha' for k in ('membership_sha256','runtime_manifest_sha256','region_sha256')}
    frames=[dict(row,input_evidence={'events':'e','gt':'g','mask':'m'},prediction_file=row['target_id']+'.npz',prediction_file_sha256='f'*64,
                 arms={'native':{'prediction_sha256':hashlib.sha256(row['target_id'].encode()).hexdigest()}}) for row in members['check_members']]
    state={'checkpoint_sha256':core.CHECKPOINT_SHA,'source_state_sha256':core.SOURCE_SHA,'initial_allcurrent_state_sha256':core.NON_BN_SHA,
           'final_allcurrent_state_sha256':core.NON_BN_SHA,'BN_scope':{'scope':'ALL78_CURRENT'},'evaluations':{'endpoint':{'frames':frames}}}
    value={'schema':'flow_bn_conversion_round14_v1','status':'ROUND14_BN_CONVERSION_COMPLETE','binding':common,
           'runtime':{'AMP':False,'TF32':False},'checkpoints':{core.STEP:state}}
    audit={'schema':'flow_bn_conversion_round14_independent_execution_audit_v1','status':'PASS','source_result_sha256':core.CURRENT_RESULT_SHA,
           'full_forwards':16,'quality_prediction_count':16,'quality_archive_count':16,'optimizer_updates':0,'calibration_forwards':0,'dual_forwards':0,
           'source_states':{core.STEP:{'checkpoint_sha256':core.CHECKPOINT_SHA,'source_state_sha256':core.SOURCE_SHA,'allcurrent_state_sha256':core.NON_BN_SHA}},
           'quality_predictions':[{'checkpoint_step':271482,'target_id':f['target_id'],'layout':'native','array_sha256':f['arms']['native']['prediction_sha256']} for f in frames]}
    for key in ('all_states_unchanged','all_BN_buffers_removed','all_hooks_removed','original_modules_restored'):value[key]=audit[key]=True
    return members,common,value,audit


class Tests(unittest.TestCase):
    def test_fixed_actual_membership_eight_and_real_ten_source_keys(self):
        path=r.HERE.parent/'coverage_recovery_round13/data_protocol/manifest_v1.json';m=r.read(path)
        canonical=FIX.FIX.ROOT/'research/flow_story_20261011/preflight/data_protocol/artifacts/common_targets.json'
        roles=FIX.FIX.ROOT/'research/flow_story_20261011/preflight/data_protocol/artifacts/train_auxiliary_roles.csv'
        self.assertEqual(len(m['source_sha256']),10);OLD.validate_manifest_sources(m,canonical,roles)
        plan=core.plan(m);self.assertEqual(len(plan),8);self.assertEqual({p['checkpoint_step'] for p in plan},{271482})
        self.assertEqual([p['target_id'] for p in plan],[x['target_id'] for x in m['check_members']]);self.assertTrue(all(p['layout']=='native' for p in plan))
        per_native=sum(d*10*h*w for d,h,w in ((2,120,160),(2,60,80),(6,30,40),(2,15,20)))
        self.assertEqual(8*per_native,4464000)
        bad=copy.deepcopy(m);bad['check_members'][1]=bad['check_members'][0]
        with self.assertRaises(ValueError):core.plan(bad)
        bad=copy.deepcopy(m);bad['source_sha256']['roles']='bad'
        with self.assertRaises(ValueError):OLD.validate_manifest_sources(bad,canonical,roles)
        MEASUREMENTS.append({'real_manifest_sha256':r.sha(path),'source_key_count':10,'full_forwards':8,'each_FC_rows':4464000})

    def test_true_CPU_reconstruction_and_own_receipt_identity(self):
        expected=r.read(r.HERE/'expected_state.json');state=expected['selected_state']
        self.assertEqual(expected['source_sha256'],{'source_state_cpu.py':r.sha(r.HERE/'source_state_cpu.py'),'selected_core.py':r.sha(r.HERE/'selected_core.py')})
        self.assertEqual((state['selected16_state_sha256'],state['selected16_tensor_count'],state['selected16_elements']),(core.SELECTED_STATE_SHA,525,54944214))
        self.assertEqual(len(state['retained_buffer_payloads']),48);self.assertEqual(state['retained_buffer_elements'],24976)
        self.assertEqual(state['frozen_BN_names'],sorted(core.mlp_bn_names((0,2))));self.assertEqual(state['private_current_BN_names'],sorted(core.mlp_bn_names((1,3))))
        mp=r.HERE.parent/'coverage_recovery_round13/data_protocol/manifest_v1.json';m=r.read(mp)
        receipt=r.read(K.HERE/'run_v1/bn_receipt.json')
        common={'membership_sha256':r.sha(mp),'runtime_manifest_sha256':'c0538f18c92a526d196138c4bb8a92d9dbef9335e9830683d91b109b36eb8d5e'}
        self.assertEqual(r.sha(K.HERE/'run_v1/bn_receipt.json'),core.OWN_RECEIPT_SHA)
        core.validate_own_receipt(receipt,m,common)
        for key,value in (('checkpoint_step',135930),('buffers_sha256','bad'),('GT_used',True),('calibration_forwards',175),('non_bn_state_after_sha256','bad')):
            bad=copy.deepcopy(receipt);bad[key]=value
            with self.assertRaises(ValueError):core.validate_own_receipt(bad,m,common)
        MEASUREMENTS.append({'actual_source_CPU_record_sha256':r.sha(r.HERE/'expected_state.json'),'same_independent525_state_sha256':core.SELECTED_STATE_SHA,'own_receipt_rejections':5,'no_quality_read':True})

    def test_installer_atomic_rejections_validate_all72(self):
        model,targets,shapes,x,functional,values=own_toy();before=R13.state_digest(model)
        state,chosen,record=core.construct_state(model.state_dict(),values,H0.state_dict_digest,torch,np,strict=False)
        badkey=next(n for n in values if '.layers.1.' in n and n.endswith('running_var'))
        for change in ('negative_unused_var','missing_unused','wrong_dtype','wrong_counter','bad_expected'):
            bad={n:v.copy() for n,v in values.items()};expected=record['selected16_state_sha256']
            if change=='negative_unused_var':bad[badkey][0]=-1
            elif change=='missing_unused':bad.pop(badkey)
            elif change=='wrong_dtype':bad[badkey]=bad[badkey].astype(np.float64)
            elif change=='wrong_counter':bad[next(n for n in bad if n.endswith('num_batches_tracked'))]=np.array(175,dtype=np.int64)
            else:expected='bad'
            with self.assertRaises(ValueError):core.install(model,bad,FIX.FIX.scope,FIX.FIX.base,H0.state_dict_digest,torch,np,expected,strict=False)
            self.assertEqual(R13.state_digest(model),before)
        MEASUREMENTS.append({'full72_validated_before_subset':True,'atomic_install_rejections':5})

    def test_real_native_H0_with_selected16_all78_observers_and_closure(self):
        historical_forward=H0.forward;historical_require=H0.forward.__globals__['require']
        model,targets,shapes,x,functional,values=own_toy()
        state,chosen,expected=core.construct_state(model.state_dict(),values,H0.state_dict_digest,torch,np,strict=False)
        bn,chosen,record=core.install(model,values,FIX.FIX.scope,FIX.FIX.base,H0.state_dict_digest,torch,np,expected['selected16_state_sha256'],strict=False)
        self.assertEqual((len(bn['frozen_BN_names']),len(bn['current_stat_BN_names'])),(16,62))
        actual=model.state_dict();self.assertEqual(sum(n.endswith(tuple('.'+s for s in core.BUFFERS)) for n in actual),48)
        for name,value in chosen.items():np.testing.assert_array_equal(actual[name].numpy(),value)
        for name,module in targets.items():
            if H0.selected_layer(name):FIX.FIX.MERGE.validate_native(module)
            else:
                self.assertIsNone(module.bn1.norm_layer.running_mean)
                with self.assertRaises(ValueError):FIX.FIX.MERGE.validate_native(module)
        before=R13.state_digest(model);signature=FIX.FIX.base.bn_signature(model,torch)
        with torch.inference_mode():
            functional.reset_net(model);direct=[v.detach().cpu().clone() for v in model(x)['flow']]
            first,observed=core.native_with_bn_records(lambda forward:forward(RUNTIME,model,functional,targets,shapes,x,'native',FIX.FIX.MERGE,torch),model,torch,H0)
        self.assertEqual(len(observed),78);self.assertTrue(all(x['calls']==1 and len(x['input_shapes'])==1 for x in observed.values()))
        self.assertEqual(sum(x['current_stat'] for x in observed.values()),62)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(direct,first[0])))
        self.assertEqual((len(first[0]),len(first[1]['branch_records']),first[1]['layers']),(4,8,{}))
        K.core.check_closure(model,targets,before,R13.state_digest,signature,FIX.FIX.base.bn_signature,torch)
        def broken(forward):raise RuntimeError('intentional observer exception')
        with self.assertRaises(RuntimeError):core.native_with_bn_records(broken,model,torch,H0)
        self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()))
        def twice(forward):
            with torch.inference_mode():model(x);return model(x)
        with self.assertRaises(ValueError):core.native_with_bn_records(twice,model,torch,H0)
        extra=model.external[0].register_forward_hook(lambda m,a,o:None)
        try:
            with torch.inference_mode(),self.assertRaisesRegex(ValueError,'Only known78'):
                core.native_with_bn_records(lambda forward:forward(RUNTIME,model,functional,targets,shapes,x,'native',FIX.FIX.MERGE,torch),model,torch,H0)
        finally:extra.remove()
        K.core.check_closure(model,targets,before,R13.state_digest,signature,FIX.FIX.base.bn_signature,torch)
        self.assertIs(H0.forward,historical_forward);self.assertIs(H0.forward.__globals__['require'],historical_require)
        MEASUREMENTS.append({'tiny_actual_native_forwards':5,'actual_H0_native_heads_equal_direct':4,'actual_H0_branches':8,'all78_BN_called_once':True,'current_stat_count':62,
                             'selected8_guard_valid_unselected4_current':True,'duplicate_call_rejected':True,'exception_hooks_closed':True,'BN_and_fullstate_unchanged':True,
                             'unknown_hook_rejected':True,'historical_forward_and_require_identities_unchanged':True})

    def test_current_reference_contract_bad_scope_and_pairing(self):
        m,common,value,audit=current_fixture();desc=core.current_reference(value,audit,m,common)
        self.assertEqual(len(desc['cached_native_predictions']),8)
        for change in ('result','audit','sha','count','closure','source','state','precision','scope','order','payload','data'):
            v=copy.deepcopy(value);a=copy.deepcopy(audit)
            if change=='result':v['status']='INCOMPLETE'
            elif change=='audit':a['status']='FAIL'
            elif change=='sha':a['source_result_sha256']='bad'
            elif change=='count':a['full_forwards']=8
            elif change=='closure':v['all_states_unchanged']=False
            elif change=='source':v['checkpoints'][core.STEP]['checkpoint_sha256']='bad'
            elif change=='state':v['checkpoints'][core.STEP]['final_allcurrent_state_sha256']='bad'
            elif change=='precision':v['runtime']['AMP']=True
            elif change=='scope':v['checkpoints'][core.STEP]['BN_scope']['scope']='private24'
            elif change=='order':a['quality_predictions'].reverse()
            elif change=='payload':a['quality_predictions'][0]['array_sha256']='bad'
            else:v['binding']['region_sha256']='bad'
            with self.assertRaises(ValueError):core.current_reference(v,a,m,common)
        MEASUREMENTS.append({'same_W_current_reference_rejections':12,'native_only_exact8_pairing':True})

    def test_source_dependency_globals_and_no_training_calibration_entry(self):
        for module in (r,core):
            for name,fn in vars(module).items():
                if isinstance(fn,types.FunctionType) and fn.__module__==module.__name__:
                    needed=set()
                    def scan(code):
                        needed.update(i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL')
                        for c in code.co_consts:
                            if isinstance(c,types.CodeType):scan(c)
                    scan(fn.__code__);self.assertFalse(needed-set(fn.__globals__)-set(vars(builtins)),name)
        source=(r.HERE/'run_selected16.py').read_text();tree=ast.parse(source)
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr in ('backward','AdamW','SGD','step','reset_bn_cma','finish_calibration','apply_fresh_bn') for n in ast.walk(tree)))
        for forbidden in ('old.prepare(', 'old.execute(', 'k.prepare(', 'k.execute('):self.assertNotIn(forbidden,source)
        self.assertFalse(torch.cuda.is_initialized())
        MEASUREMENTS.append({'all_source_global_names_bound':True,'no_training_or_calibration_or_old_prepare':True,'GPU_initialized':False})


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261016)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    receipt={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'GPU_initialized':torch.cuda.is_initialized(),'torch':str(torch.__version__),
             'source_sha256':{str(r.HERE/n):r.sha(r.HERE/n) for n in r.OWN},'measurements':MEASUREMENTS,
             'errors':[str(x) for x in result.errors],'failures':[str(x) for x in result.failures],
             'scope':'Real source-reconstruction receipt/metadata plus tiny native arithmetic and synthetic rejections; no new actual quality, SSH or GPU.'}
    r.write(r.HERE/'cpu_results.json',receipt);raise SystemExit(0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1)
