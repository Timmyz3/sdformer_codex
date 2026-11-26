#!/usr/bin/env python3
"""New capture/CMA/closure checks on existing small native MLPs; no real quality."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import ast
import builtins
import copy
import dis
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
import numpy as np
import torch
import run_deployment as r
core=r.core
H0,RUNTIME=r.load_runtime()
R13=RUNTIME[0]
R13DIR=r.H0DIR.parents[1]/'implementation'
sys.path.insert(0,str(R13DIR))
FIX=r.load(R13DIR/'test_coverage_cpu.py','_later_existing_cpu_fixture')
CMA=r.load(FIX.ROOT/'research/flow_story_20261013/breadth/collect_bn_cma.py','_later_cpu_cma')
MEASUREMENTS=[]


def toy():
    model,targets,shapes,x,functional,_=FIX.toy()
    native_forward=model.forward
    def four_heads(value):
        output=native_forward(value)
        # Exercise every one of the 54 external BN instances once per model
        # call. The original fixture already uses all 24 native private BNs.
        flat=value.reshape(-1,4)
        for layer in model.external:layer(flat)
        a,b=output['flow']
        return {'flow':[a,b,a*.5,b*.5]}
    model.forward=four_heads
    model.requires_grad_(False)
    return model,targets,shapes,x,functional


def calibrated():
    model,targets,shapes,x,functional=toy()
    bns,private=FIX.scope.bn_inventory(model,torch)
    CMA.reset_bn_cma(model,torch);before=CMA.non_bn_digest(model,bns)
    for step in range(176):
        ex=core.native_calibration_forward(model,functional,targets,shapes,x,torch)
        if set(int(v.num_batches_tracked) for v in bns.values())!={step+1}:raise AssertionError('Actual 78 counter progression')
    buffers=core.finish_calibration(model,bns,private,CMA,before,torch)
    return model,targets,shapes,x,functional,bns,private,buffers,ex,before


class Tests(unittest.TestCase):
    def test_fixed_real_member_order_and_independent_geometry(self):
        members,rows,roles=FIX.manifest_inputs();R13.validate_members(members,rows,roles)
        plan=core.plan(members)
        self.assertEqual(len(plan),193)
        self.assertEqual([v['target_id'] for v in plan[:176]],[v['target_id'] for v in members['bn_members']])
        self.assertEqual([v['forward_id'] for v in plan],list(range(1,194)))
        self.assertEqual(sum(v['mode']=='dual_reference' for v in plan),1)
        self.assertEqual((plan[177]['layout'],plan[178]['mode']),( 'FF','dual_reference'))
        # Independent production geometry, not the implementation row helper.
        per_stage=[(2,120,160),(2,60,80),(6,30,40),(2,15,20)]
        native=sum(d*10*h*w for d,h,w in per_stage)
        coarse_extra=sum(d*10*sum(((min(9,h-y)+2)//3)*((min(9,w-x)+2)//3)
                            for y in range(0,h,9) for x in range(0,w,9))
                         for d,h,w in (per_stage[0],per_stage[2]))
        self.assertEqual((native,coarse_extra),(558000,51600))
        self.assertEqual(192*native+native+coarse_extra,107745600)
        for key in ('bn_members','check_members'):
            bad=copy.deepcopy(members);bad[key].pop()
            with self.assertRaises(ValueError):core.plan(bad)
        MEASUREMENTS.append({'full_forwards':193,'calibration':176,'evaluation':16,'dual':1,'each_FC_total_rows':107745600,'optimizer_steps':0})

    def test_payload_source_contract_and_rejections(self):
        run={'contract':{'encoding':'native'},'provenance':{'source':'pinned'},'effective_config':{'T':10}}
        payload=dict(run,schema_version='flow_a_native_encoding_v2_checkpoint_v1',contract_sha256=r.CONTRACT_SHA,
                     training={'completed_optimizer_steps':271482},stop_reason='wall_budget',model_state_dict={'a':torch.ones(2),'b':torch.zeros(3)})
        side={'contract_sha256':r.CONTRACT_SHA,'completed_optimizer_steps':271482}
        audit=r.check_payload(payload,run,side,torch,2,5);self.assertEqual(audit['model_elements'],5)
        for case in ('schema','step','provenance','contract','config','finite','inventory','stop','nontensor'):
            bad=copy.deepcopy(payload)
            if case=='schema':bad['schema_version']='old'
            elif case=='step':bad['training']['completed_optimizer_steps']=135930
            elif case=='provenance':bad['provenance']['source']='other'
            elif case=='contract':bad['contract_sha256']='other'
            elif case=='config':bad['effective_config']['T']=1
            elif case=='finite':bad['model_state_dict']['a'][0]=float('nan')
            elif case=='inventory':bad['model_state_dict'].pop('a')
            elif case=='stop':bad['stop_reason']='unknown'
            else:bad['model_state_dict']['a']='not tensor'
            with self.assertRaises(ValueError):r.check_payload(bad,run,side,torch,2,5)
        MEASUREMENTS.append({'actual_checkpoint_not_loaded_in_synthetic_unit_test':True,'contract_rejections':9})

    def test_real_176_CMA_calls_78_counters_72_own_buffers_and_54_current(self):
        model,targets,shapes,x,functional,bns,private,buffers,ex,before=calibrated()
        self.assertEqual(len(buffers),72)
        self.assertEqual(sum(k.endswith(('running_mean','running_var','num_batches_tracked')) for k in model.state_dict()),72)
        self.assertEqual(CMA.non_bn_digest(model,bns),before)
        self.assertTrue(all(not v.training for v in model.modules()))
        self.assertTrue(all(v.running_mean is None and not v.track_running_stats for n,v in bns.items() if n not in private))
        self.assertTrue(all(int(bns[n].num_batches_tracked)==176 for n in private))
        for key,value in buffers.items():np.testing.assert_array_equal(model.state_dict()[key].numpy(),value)
        self.assertEqual(ex['summed_fc1_rows'],sum(np.prod(v[:-1]) for v in shapes.values()))
        self.assertEqual(len(ex['all12_observed_fc_rows']),12)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'own.npz';np.savez_compressed(path,**buffers)
            with np.load(path,allow_pickle=False) as saved:
                self.assertEqual(set(saved.files),set(buffers))
                for key in saved.files:np.testing.assert_array_equal(saved[key],buffers[key])
        MEASUREMENTS.append({'actual_tiny_native_model_calls':176,'all78_counter_progression':True,'own_buffers':72,'current_stat_external':54,
                             'nonBN_unchanged':True,'no_optimizer':True})

    def test_real_pinned_H0_capture_actual_dual_raw_payloads_and_closure(self):
        model,targets,shapes,x,functional,bns,private,buffers,ex,before=calibrated()
        state=R13.state_digest(model);bn=FIX.base.bn_signature(model,torch)
        source_probe_globals=dict(RUNTIME[2].forward_probe.__globals__)
        with torch.inference_mode():
            plain=H0.forward(RUNTIME,model,functional,targets,shapes,x,'FF',FIX.MERGE,torch)
            actual,inputs=core.forward_with_raw_inputs(H0,RUNTIME,model,functional,targets,shapes,x,FIX.MERGE,torch,'actual',lambda:None)
            dual,dinputs=core.forward_with_raw_inputs(H0,RUNTIME,model,functional,targets,shapes,x,FIX.MERGE,torch,'dual_reference',lambda:None)
        self.assertEqual(set(inputs),set(actual[2]));self.assertEqual(len(inputs),8)
        H0.check_pair(plain,actual,torch);H0.check_pair(actual,dual,torch)
        self.assertEqual(source_probe_globals,RUNTIME[2].forward_probe.__globals__)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            ar=core.save_raw_pair_side(root/'actual.npz',actual,inputs,np,r.sha)
            dr=core.save_raw_pair_side(root/'dual.npz',dual,dinputs,np,r.sha)
            self.assertEqual(ar['arrays'],dr['arrays']);self.assertEqual(len(ar['arrays']),20)
            self.assertEqual(ar['branch_names_in_index_order'],sorted(inputs))
            with np.load(root/'actual.npz',allow_pickle=False) as saved:
                self.assertEqual(set(saved.files),set(ar['arrays']))
                for key in saved.files:self.assertEqual(__import__('hashlib').sha256(saved[key].tobytes()).hexdigest(),ar['arrays'][key]['payload_sha256'])
            wrong=dict(inputs);name=next(iter(wrong));wrong[name]=wrong[name]+1
            with self.assertRaisesRegex(ValueError,'binding'):core.save_raw_pair_side(root/'bad.npz',actual,wrong,np,r.sha)
            truncated=(actual[0][:-1],*actual[1:])
            with self.assertRaisesRegex(ValueError,'Four flow'):core.save_raw_pair_side(root/'bad.npz',truncated,inputs,np,r.sha)
        core.check_closure(model,targets,state,R13.state_digest,bn,FIX.base.bn_signature,torch)
        MEASUREMENTS.append({'real_pinned_H0_clone_test':True,'tiny_shape':list(next(iter(shapes.values()))),'four_heads':4,
                             'eight_whole_inputs_and_branches':True,'actual_dual_exact':True,'raw_array_count_per_side':20,'historical_globals_unchanged':True})

    def test_failures_remove_hooks_and_restore_original_wrappers(self):
        model,targets,shapes,x,functional=toy();all_bn,private=FIX.scope.bn_inventory(model,torch)
        CMA.reset_bn_cma(model,torch);before=CMA.non_bn_digest(model,all_bn)
        with self.assertRaisesRegex(ValueError,'Incomplete'):core.finish_calibration(model,all_bn,private,CMA,before,torch)
        layer=next(iter(targets.values()))
        with patch.object(layer.fc1,'forward',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'):core.native_calibration_forward(model,functional,targets,shapes,x,torch)
        self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()))
        # Prepare legal eval BN without invoking any external frozen prepare.
        for m in all_bn.values():m.num_batches_tracked.fill_(176)
        core.finish_calibration(model,all_bn,private,CMA,before,torch)
        with patch.object(layer.fc1,'forward',side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError,'injected'):
                with torch.inference_mode():core.forward_with_raw_inputs(H0,RUNTIME,model,functional,targets,shapes,x,FIX.MERGE,torch,'actual',lambda:None)
        self.assertTrue(all(model.get_submodule(n) is m for n,m in targets.items()))
        self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()))
        MEASUREMENTS.append({'calibration_and_raw_capture_exception_hooks_clean':True,'incomplete_CMA_rejected':True,'fake176_used_only_for_exception_fixture':True})

    def test_globals_no_training_source_and_preflight_filesystem(self):
        for module in (r,core):
            for name,value in vars(module).items():
                if isinstance(value,types.FunctionType) and value.__module__==module.__name__:
                    needed=set()
                    def scan(code):
                        needed.update(i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL')
                        for c in code.co_consts:
                            if isinstance(c,types.CodeType):scan(c)
                    scan(value.__code__)
                    self.assertFalse(needed-set(value.__globals__)-set(vars(builtins)),name)
        for name in ('run_deployment.py','deployment_core.py'):
            source=(r.HERE/name).read_text();tree=ast.parse(source)
            self.assertFalse(any(isinstance(n,ast.Attribute) and n.attr in ('backward','AdamW','SGD','step') for n in ast.walk(tree)))
            self.assertNotIn('apply_fresh_bn(',source);self.assertNotIn('h0.prepare(',source)
        self.assertIn('core.native_calibration_forward(', (r.HERE/'run_deployment.py').read_text())
        with tempfile.TemporaryDirectory() as d:
            future=Path(d)/'run_v1';path=Path(d)/'preflight'/'receipt.json'
            r.write(path,{'status':'PREFLIGHT_ONLY'});self.assertTrue(path.is_file());self.assertFalse(future.exists())
            self.assertFalse(path.resolve().is_relative_to(future.resolve()))
        self.assertFalse(torch.cuda.is_initialized())
        MEASUREMENTS.append({'nested_globals_bound':True,'training_calls_absent':True,'preflight_parent_creation':True,'GPU_initialized':False})


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261016)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    receipt={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'GPU_initialized':torch.cuda.is_initialized(),
             'torch':str(torch.__version__),'source_sha256':{str(r.HERE/n):r.sha(r.HERE/n) for n in r.OWN},
             'measurements':MEASUREMENTS,'errors':[str(x) for x in result.errors],'failures':[str(x) for x in result.failures],
             'scope':'CPU synthetic metadata and actual small native MLP math; no real checkpoint/evaluation quality/data arrays, SSH or GPU.'}
    r.write(r.HERE/'cpu_results.json',receipt)
    raise SystemExit(0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1)
