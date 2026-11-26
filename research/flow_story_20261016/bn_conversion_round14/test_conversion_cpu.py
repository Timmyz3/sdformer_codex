#!/usr/bin/env python3
"""New two-source contract/current-stat switch tests; no actual quality/GPU."""
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
import tempfile
import types
import unittest
sys.dont_write_bytecode=True
import numpy as np
import torch
import run_conversion as r
core=r.core
K,H0,RUNTIME=r.runtime();R13=RUNTIME[0]
sys.path.insert(0,str(r.KDIR));FIX=r.load(r.KDIR/'test_deployment_cpu.py','_r14_existing_tiny_fixture')
MEASUREMENTS=[]


def reference_fixture(step):
    m={'check_members':[{'target_id':'id'+str(i),'sequence':'s'+str(i%2)} for i in range(8)]}
    common={k:k+'sha' for k in ('membership_sha256','runtime_manifest_sha256','region_sha256')}
    b=dict(common,checkpoint_sha256=core.CHECKPOINT_SHA[step],BN_buffers_sha256='b'*64,condition='diversity',arm='native',mode='recover')
    frames=[dict(row,input_evidence={'events':'e','gt':'g','mask':'m'},prediction_file=row['target_id']+'.npz',prediction_file_sha256='f'*64,
                 arms={'native':{'prediction_sha256':hashlib.sha256(row['target_id'].encode()).hexdigest()}}) for row in m['check_members']]
    scope={'BN_count':78,'MLP_BN_count':24,'frozen_BN_names':['private'+str(i) for i in range(24)],
           'current_stat_BN_names':['external'+str(i) for i in range(54)],'non_bn_state_sha256':core.CURRENT_STATE_SHA[step]}
    value={'binding':b,'BN_scope':scope,'runtime':{'AMP':False,'TF32':False},
           **{k:True for k in ('BN_buffers_unchanged','original_modules_restored','all_hooks_removed')}}
    if step=='135930':
        scope['scope']='MLP_R13_COMMON_CMA176'
        value.update(schema='flow_round13_coverage_recovery_v1',status='ROUND13_COVERAGE_RECOVERY_COMPLETE',initial_model_state_sha256='i'*64,
                     evaluations={'initial':{'frames':frames}})
        entries=[{'condition':c,'mode':m,'arm':a,'result_sha256':'r'*64,'initial_model_state_sha256':'i'*64,
                  'predictions':[{'target_id':f['target_id'],'layout':'native','label':'initial','array_sha256':f['arms']['native']['prediction_sha256']} for f in frames]}
                 for c in ('reference','diversity') for m in ('smoke','recover') for a in ('native','FF','CC','fine_s2b2')]
        audit={'schema':'flow_round13_independent_execution_audit_v1','status':'PASS','scope':{'conditions':['reference','diversity'],'modes':['smoke','recover'],'runs':16},
               'results':entries,'common_initial_model_sha256':'i'*64,'new_BN_actual_sha256':'b'*64}
    else:
        scope['scope']='MLP_271482_OWN_CMA176'
        value.update(schema='flow_later_checkpoint_deployment_v1',status='LATER_CHECKPOINT_193_FORWARD_COMPLETE',non_bn_state_unchanged=True,
                     deployment_initial_state_sha256='i'*64,deployment_final_state_sha256='i'*64,own_BN_buffers={'sha256':'b'*64},evaluations={'endpoint':{'frames':frames}})
        audit={'schema':'flow_later_checkpoint_independent_execution_audit_v1','status':'PASS','source_result_sha256':'r'*64,'full_forwards':193,'optimizer_updates':0,
               'calibration_forwards':176,'source_non_BN_state_sha256':core.CURRENT_STATE_SHA[step],'source_state_sha256':core.SOURCE_STATE_SHA[step],
               'deployment_state_sha256':'i'*64,'BN_buffers_sha256':'b'*64,
               'quality_predictions':[{'target_id':f['target_id'],'layout':'native','array_sha256':f['arms']['native']['prediction_sha256']} for f in frames]}
    return m,common,value,audit


class Tests(unittest.TestCase):
    def test_fixed16_and_two_original_checkpoint_inventory_contract(self):
        m=reference_fixture('135930')[0];plan=core.plan(m)
        self.assertEqual(len(plan),16);self.assertEqual([p['checkpoint_step'] for p in plan],[135930]*8+[271482]*8)
        self.assertEqual([p['target_id'] for p in plan[:8]],[p['target_id'] for p in plan[8:]])
        self.assertEqual(16*558000,8928000)
        bad=copy.deepcopy(m);bad['check_members'][0]=bad['check_members'][1]
        with self.assertRaises(ValueError):core.plan(bad)
        # Inventory-only synthetic state uses meta tensors: no real source PT
        # bytes or quality are touched. Actual full-state SHA is separately
        # independently measured from the two source checkpoints.
        state={n+'.'+f:torch.empty((),device='meta') for n in ['bn'+str(i) for i in range(78)] for f in core.BUFFERS}
        state.update({'p'+str(i):torch.empty((1 if i else 54919238-476,),device='meta') for i in range(477)})
        digest=lambda s:'full' if len(s)==711 else 'current'
        got=core.strip_bn_state(state,digest,'full','current');self.assertEqual(got['allcurrent_tensor_count'],477)
        with self.assertRaises(ValueError):core.strip_bn_state(state,digest,'wrong','current')
        state.pop('bn0.running_mean')
        with self.assertRaises(ValueError):core.strip_bn_state(state,digest)
        MEASUREMENTS.append({'full_forwards':16,'each_FC_rows':8928000,'zero_calibration_optimizer_dual':True,'synthetic_inventory_not_real_PT':True})

    def test_same_W_cache_provenance_and_rejection_paths(self):
        for step in core.CHECKPOINTS:
            m,common,value,audit=reference_fixture(step)
            descriptor=core.reference_contract(step,value,'r'*64,audit,m,common)
            self.assertEqual(descriptor['cache_phase'],'initial' if step=='135930' else 'endpoint')
            for case in ('checkpoint','data','precision','BNscope','closure','audit','source','payload','order'):
                v=copy.deepcopy(value);a=copy.deepcopy(audit)
                if case=='checkpoint':v['binding']['checkpoint_sha256']='bad'
                elif case=='data':v['binding']['membership_sha256']='bad'
                elif case=='precision':v['runtime']['AMP']=True
                elif case=='BNscope':v['BN_scope']['frozen_BN_names'][0]=v['BN_scope']['frozen_BN_names'][1]
                elif case=='closure':v['all_hooks_removed']=False
                elif case=='audit':a['status']='FAIL'
                elif case=='source':
                    if step=='135930':v['BN_scope']['non_bn_state_sha256']='wrong'
                    else:a['source_non_BN_state_sha256']='wrong'
                else:
                    phase='initial' if step=='135930' else 'endpoint';frames=v['evaluations'][phase]['frames']
                    if case=='payload':frames[0]['arms']['native']['prediction_sha256']='bad'
                    else:frames.reverse()
                with self.assertRaises(ValueError):core.reference_contract(step,v,'r'*64,a,m,common)
            with self.assertRaises(ValueError):core.reference_contract(step,value,'wrong-result-SHA',audit,m,common)
        MEASUREMENTS.append({'two_cache_sources_metadata_only':True,'rejection_cases':20,'135930_uses_initial_not_recovered_endpoint':True})

    def test_real78_switch_preserves_affine_native_allcurrent_and_closure(self):
        model,targets,shapes,x,functional=FIX.toy()
        old={k:v.detach().clone() for k,v in model.state_dict().items()}
        expected={k:v for k,v in old.items() if not k.endswith(tuple('.'+s for s in core.BUFFERS))}
        before=H0.state_dict_digest(expected)
        scope=core.switch_allcurrent(model,FIX.FIX.base,FIX.FIX.scope,R13.state_digest,torch,before,expected_count=len(expected))
        self.assertEqual(scope['BN_count'],78);self.assertTrue(all(not v.track_running_stats for v in model.external))
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,old[k]))
        signature=FIX.FIX.base.bn_signature(model,torch)
        captured={};handles=[]
        try:
            for name,module in targets.items():
                if not H0.selected_layer(name):continue
                def capture(layer,args,out,n=name):captured[n]=(args[0].detach().clone(),out.detach().clone())
                handles.append(module.register_forward_hook(capture))
            functional.reset_net(model)
            with torch.inference_mode():direct=[v.detach().clone() for v in model(x)['flow']]
        finally:
            for handle in handles:handle.remove()
        with torch.inference_mode():
            first=H0.forward(RUNTIME,model,functional,targets,shapes,x,'native',FIX.FIX.MERGE,torch)
            second=H0.forward(RUNTIME,model,functional,targets,shapes,x,'native',FIX.FIX.MERGE,torch)
        H0.check_pair(first,second,torch)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(direct,first[0])))
        self.assertEqual(set(captured),set(first[2]))
        for name,(input_value,branch) in captured.items():
            self.assertTrue(torch.equal(branch,first[2][name]))
            self.assertEqual(FIX.FIX.MERGE.tensor_sha(input_value),first[3][name])
        self.assertEqual(first[1]['layers'],{});self.assertEqual(len(first[0]),4)
        K.core.check_closure(model,targets,before,R13.state_digest,signature,FIX.FIX.base.bn_signature,torch)
        self.assertTrue(all(v['running_mean'] is None and v['running_var'] is None and v['num_batches_tracked'] is None for v in signature.values()))
        MEASUREMENTS.append({'actual_tiny_native_all78current_calls':3,'direct_native_all4heads_8input_8branch_equals_H0':True,'native_helper_has_no_frozen_private_BN_requirement':True,
                             'affine_and_nonBN_exact':True,'all78_no_buffers_no_updates':True,'original_hooks_and_modules_closed':True})

    def test_cache_NPZ_actual_payload_and_nested_source_dependencies(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'native.npz';arr=np.ones((2,480,640),dtype=np.float32);np.savez_compressed(p,native=arr)
            record={'prediction_file':p.name,'prediction_file_sha256':r.sha(p),'prediction_sha256':hashlib.sha256(arr.tobytes()).hexdigest()}
            desc={'cached_native_predictions':[record]};r.verify_cache_arrays(root,desc,np)
            for key,value in [('prediction_file','../bad.npz'),('prediction_file_sha256','bad'),('prediction_sha256','bad')]:
                bad=copy.deepcopy(desc);bad['cached_native_predictions'][0][key]=value
                with self.assertRaises(ValueError):r.verify_cache_arrays(root,bad,np)
        for module in (r,core):
            for name,fn in vars(module).items():
                if isinstance(fn,types.FunctionType) and fn.__module__==module.__name__:
                    needed=set()
                    def scan(code):
                        needed.update(i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL')
                        for v in code.co_consts:
                            if isinstance(v,types.CodeType):scan(v)
                    scan(fn.__code__);self.assertFalse(needed-set(fn.__globals__)-set(vars(builtins)),name)
        source=(r.HERE/'run_conversion.py').read_text();tree=ast.parse(source)
        self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr in ('backward','AdamW','SGD','step','reset_bn_cma','finish_calibration') for n in ast.walk(tree)))
        self.assertNotIn('k.prepare(',source);self.assertNotIn('k.execute(',source);self.assertNotIn('apply_fresh_bn(',source)
        self.assertFalse(torch.cuda.is_initialized())
        MEASUREMENTS.append({'cache_payload_rejections':3,'nested_globals_bound':True,'no_training_calibration_or_old_prepare_calls':True})


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261016)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    receipt={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'GPU_initialized':torch.cuda.is_initialized(),'torch':str(torch.__version__),
             'source_sha256':{str(r.HERE/n):r.sha(r.HERE/n) for n in r.OWN},'measurements':MEASUREMENTS,
             'errors':[str(x) for x in result.errors],'failures':[str(x) for x in result.failures],
             'scope':'Synthetic cache metadata/NPZ and existing small native MLP arithmetic; no real new quality, full source PT, SSH or GPU.'}
    r.write(r.HERE/'cpu_results.json',receipt);raise SystemExit(0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1)
