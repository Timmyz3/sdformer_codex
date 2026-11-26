#!/usr/bin/env python3
"""Synthetic metadata and native small-MLP checks; no real quality/data/GPU."""
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
from unittest.mock import patch
sys.dont_write_bytecode=True
import numpy as np
import torch
import run_native_lr as r
sys.path.insert(0,str(r.R13DIR))
R13TEST=r.load_file(r.R13DIR/'test_coverage_cpu.py','_native_lr_existing_fixture')
MEASUREMENTS=[]


def fixture():
    manifest,_,_=R13TEST.manifest_inputs()
    common={k:'a'*64 for k in ('checkpoint_sha256','wrapper_sha256','runtime_manifest_sha256','BN_buffers_sha256','BN_receipt_sha256',
                             'loss_sha256','region_sha256')}
    common.update(membership_sha256=r.MEMBER_SHA,allowed_parameter_count=r.R13.PARAMETERS,membership_audit={'synthetic':True})
    binding=dict(common,runner_sha256=r.sha(r.HERE/'run_native_lr.py'),execute_sha256=r.sha(r.HERE/'execute_recovery.py'),
                 runner_checks_sha256='b'*64,source_pins_sha256=r.sha(r.HERE/'inherited_pins.json'),recipe=r.RECIPE,
                 condition='diversity',arm='native',mode='recover')
    old=dict(common,runner_sha256=r.R13_SHA,execute_sha256=r.R13_EXEC_SHA,recipe=r.R13.RECIPE,condition='diversity',arm='native',mode='recover',
             fit_ids=[v['target_id'] for v in manifest['train_order_diversity']],check_ids=[v['target_id'] for v in manifest['check_members']])
    frames=[{'target_id':v['target_id'],'sequence':v['sequence'],'arms':{'native':{'prediction_sha256':hashlib.sha256(v['target_id'].encode()).hexdigest()}}}
            for v in manifest['check_members']]
    baseline={'schema':r.R13.SCHEMA,'status':'ROUND13_COVERAGE_RECOVERY_COMPLETE','completed_optimizer_steps':200,'training_performed':True,
              'binding':old,'initial_model_state_sha256':'c'*64,'evaluations':{'initial':{'frames':frames}},
              **{k:True for k in ('frozen_state_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed')}}
    return manifest,binding,baseline


def smoke_fixture(binding,baseline_sha):
    own=dict(binding,mode='smoke',baseline_comparison={'result_sha256':baseline_sha,'initial_state_sha256':'c'*64})
    return {'schema':r.SCHEMA,'status':'NATIVE_LR_SMOKE_COMPLETE','completed_optimizer_steps':5,'training_performed':True,'full_forward_count':9,
            'binding':own,'initial_model_state_sha256':'c'*64,'initial_numerical_pair':{'all_flow_outputs_exact':True},
            'final_numerical_pair':{'all_flow_outputs_exact':True},'gradient_coverage':{'complete':True},
            'actual_optimizer_hyperparameters':{'lr':r.RECIPE['lr'],'weight_decay':r.RECIPE['weight_decay'],'betas':list(r.RECIPE['betas']),'eps':r.RECIPE['eps']},
            **{k:True for k in ('frozen_state_unchanged','BN_buffers_unchanged','original_modules_restored','all_hooks_removed',
                              'smoke_weights_discarded','paired_initial_state_exact')}}


class Tests(unittest.TestCase):
    def test_fixed_single_axis_budget_and_execute_source(self):
        self.assertEqual({k for k in r.RECIPE if r.RECIPE[k]!=r.R13.RECIPE[k]},{'lr'})
        self.assertEqual(r.RECIPE['lr'],r.R13.RECIPE['lr']/10)
        self.assertEqual(sum(r.budget('smoke').values()),9);self.assertEqual(sum(r.budget('recover').values()),228)
        self.assertEqual(558000*(9+228),132246000)
        source=(r.HERE/'execute_recovery.py').read_text();ast.parse(source)
        self.assertIn("evaluate('endpoint',('native','FF'))",source)
        self.assertIn('apply_fresh_bn(',source);self.assertNotIn('scope.apply_scope',source)
        self.assertNotIn('optimizer.load_state_dict',source);self.assertNotIn('torch.load',source)
        self.assertIn('flow_native_lr_stability_recovered_state_v1',source)
        self.assertNotIn('flow_round13_recovered_shared_native_state_v1',source)
        MEASUREMENTS.append({'only_recipe_change':'lr','smoke_forward_budget':9,'formal_forward_budget':228,'combined':237,
                             'per_FC_rows_total':132246000,'new_BN_calibration_forwards':0,'no_resumed_state_or_optimizer':True})

    def test_baseline_exact_source_identity_and_initial_prediction_contract(self):
        manifest,binding,baseline=fixture();pair=r.baseline_contract(baseline,binding,manifest,'d'*64)
        self.assertEqual(len(pair['initial_native_prediction_sha256']),8)
        self.assertTrue(pair['R13_check8_now_development']);self.assertFalse(pair['recovered_baseline_weights_used_as_initialization'])
        for name in ('condition','recipe','BN','order','frames','digest','closure','step'):
            bad=copy.deepcopy(baseline)
            if name=='condition':bad['binding']['condition']='reference'
            elif name=='recipe':bad['binding']['recipe']['lr']=r.RECIPE['lr']
            elif name=='BN':bad['binding']['BN_buffers_sha256']='wrong'
            elif name=='order':bad['binding']['fit_ids'].reverse()
            elif name=='frames':bad['evaluations']['initial']['frames'].pop()
            elif name=='digest':bad['evaluations']['initial']['frames'][0]['arms']['native']['prediction_sha256']='invalid'
            elif name=='closure':bad['all_hooks_removed']=False
            else:bad['completed_optimizer_steps']=5
            with self.assertRaises(ValueError):r.baseline_contract(bad,binding,manifest,'d'*64)
        MEASUREMENTS.append({'baseline_contract_rejection_cases':8,'initial_prediction_payloads_paired':8,'synthetic_only':True})

    def test_single_new_smoke_not_old_four_arm_gate(self):
        _,binding,_=fixture();value=smoke_fixture(binding,'d'*64)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'smoke.json';p.write_text(json.dumps(value));r.validate_smoke([p],binding,'d'*64)
            for paths in ([],[p,p]):
                with self.assertRaises(ValueError):r.validate_smoke(paths,binding,'d'*64)
            for name in ('status','lr','baseline','count','state','discard'):
                bad=copy.deepcopy(value)
                if name=='status':bad['status']='ROUND13_SMOKE_COMPLETE'
                elif name=='lr':bad['actual_optimizer_hyperparameters']['lr']=r.R13.RECIPE['lr']
                elif name=='baseline':bad['binding']['baseline_comparison']['result_sha256']='wrong'
                elif name=='count':bad['full_forward_count']=11
                elif name=='state':bad['initial_model_state_sha256']='a'*64
                else:bad['smoke_weights_discarded']=False
                p.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):r.validate_smoke([p],binding,'d'*64)
        MEASUREMENTS.append({'single_smoke_rejection_cases':8,'old_R13_smoke_cannot_authorize_new_recipe':True})

    def test_private_prepare_own_sources_and_exact_24_endpoint_count(self):
        manifest,binding,baseline=fixture()
        # A metadata-only stand-in checks the private namespace passed to the
        # frozen prepare code. Full real data preflight remains root's job.
        def metadata_prepare(a,collector,probe,parent):
            binding=dict(parent)
            binding.update(runner_sha256=sha(__file__),execute_sha256=sha(HERE/'execute_recovery.py'),recipe=RECIPE,
                           source_pins_sha256=sha(HERE/'inherited_pins.json'),expected_evaluation_forwards=40,
                           own_files=list(OWN),mode=a.mode)
            if a.mode=='recover':binding['smoke_receipts_sha256']=validate_smokes(a.smoke_receipt,binding)
            rows=collector['train_members_diversity'];checks=[] if a.mode=='smoke' else collector['check_members']
            inherited=(None,None,None,None,None,None,None,binding,None)
            return inherited,rows,checks,binding,collector
        # Explicit globals mirror names used by actual frozen prepare.
        namespace=dict(metadata_prepare.__globals__);namespace.update(sha=r.sha,HERE=r.R13DIR,RECIPE=r.R13.RECIPE,OWN=r.R13.OWN,
                    validate_smokes=r.R13.validate_smokes,__file__=str(r.R13DIR/'run_coverage_recovery.py'))
        fake=types.FunctionType(metadata_prepare.__code__,namespace)
        with tempfile.TemporaryDirectory() as d:
            baseline_path=Path(d)/'baseline.json';baseline_path.write_text(json.dumps(baseline));baseline_sha=r.sha(baseline_path)
            smoke=smoke_fixture(binding,baseline_sha);p=Path(d)/'smoke.json';p.write_text(json.dumps(smoke))
            for mode in ('smoke','recover'):
                a=types.SimpleNamespace(condition='diversity',arm='native',membership_sha256=r.MEMBER_SHA,baseline_result=baseline_path,
                                        baseline_result_sha256=baseline_sha,mode=mode,smoke_receipt=[] if mode=='smoke' else [p])
                with patch.object(r.R13,'prepare',fake):result=r.prepare(a,manifest,None,binding)
                got=result[3]
                self.assertEqual(got['runner_sha256'],r.sha(r.HERE/'run_native_lr.py'));self.assertEqual(got['execute_sha256'],r.sha(r.HERE/'execute_recovery.py'))
                self.assertEqual(got['recipe'],r.RECIPE);self.assertEqual(got['reused_prepare_source_sha256'],r.R13_SHA)
                self.assertEqual(got['expected_evaluation_forwards'],0 if mode=='smoke' else 24)
                self.assertIs(result[0][7],got);self.assertTrue(got['fresh_W0_and_optimizer'])
        MEASUREMENTS.append({'private_prepare_globals_bind_new_entry_execute_recipe_sources':True,'mock_metadata_only_no_full_preflight':True})

    def test_fresh_initial_native_lower_lr_five_updates_and_all_layout_math(self):
        def fresh():
            torch.manual_seed(r.RECIPE['seed'])
            model,targets,shapes,x,functional,values=R13TEST.toy()
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'bn.npz';receipt=Path(d)/'receipt.json';receipt.write_text('{}');np.savez(p,**values)
                r.apply_fresh_bn(model,p,receipt,{},'',R13TEST.scope,R13TEST.base,torch,validate_receipt=False)
            selected=r.select_parameters(model,targets,expected=3104)
            return model,targets,shapes,x,functional,selected
        model,targets,shapes,x,functional,selected=fresh();initial=r.state_digest(model)
        frozen=r.state_digest(model,selected);bn=R13TEST.base.bn_signature(model,torch);before={n:p.detach().clone() for n,p in selected.items()}
        optimizer=torch.optim.AdamW(list(selected.values()),lr=r.RECIPE['lr'],weight_decay=r.RECIPE['weight_decay'],betas=tuple(r.RECIPE['betas']),eps=r.RECIPE['eps'])
        steps=[]
        for phase in ('initial','endpoint'):
            with torch.no_grad():
                for arm in ('native','FF'):
                    output,ex,branch=r.forward_train(model,functional,targets,shapes,x,arm,capture=True)
                    ref=r.reference_forward(R13TEST.R9,R13TEST.R8,R13TEST.PARENT,model,functional,targets,shapes,x,arm,R13TEST.MERGE,torch)
                    r.compare_all(output['flow'],branch,ex['capture_input_sha256'],ref,torch)
            if phase=='endpoint':break
            for index in range(5):
                optimizer.zero_grad(set_to_none=True);output,_,_=r.forward_train(model,functional,targets,shapes,x+index*.03,'native')
                sum((v-.37).square().mean() for v in output['flow']).backward();steps.append({'gradient_audit':r.gradient_audit(model,selected,torch)})
                torch.nn.utils.clip_grad_norm_(list(selected.values()),100.,error_if_nonfinite=True);optimizer.step()
                self.assertEqual(r.state_digest(model,selected),frozen);self.assertEqual(R13TEST.base.bn_signature(model,torch),bn)
            optimizer.zero_grad(set_to_none=True)
        self.assertTrue(r.gradient_coverage(steps,[n for n in targets if r.tile.is_selected_layer(n)],True)['complete'])
        drift=r.parameter_drift(selected,before,torch);self.assertTrue(all(v['relative_L2']>0 for v in drift['categories'].values()))
        fresh_model,*_=fresh();self.assertEqual(r.state_digest(fresh_model),initial);self.assertNotEqual(r.state_digest(model),initial)
        fresh_optimizer=torch.optim.AdamW([p for p in fresh_model.parameters() if p.requires_grad],lr=r.RECIPE['lr'])
        self.assertEqual(len(fresh_optimizer.state),0)
        self.assertTrue(all(model.get_submodule(n) is m for n,m in targets.items()));self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        MEASUREMENTS.append({'native_small_MLP_steps':5,'same_new_BN_native_FF_initial_and_final_old_math_exact':True,
                             'fresh_W0_restart_not_smoke_state':True,'fresh_optimizer_empty':True,'frozen_complement_BN_hooks_closed':True,
                             'scope':'real native MLP mathematics, synthetic two-head loss; not full GPU loop or real four-head data'})

    def test_parameter_drift_independent_scalar_values(self):
        names=('x.mlp.fc1.weight','x.mlp.sn1.weight','x.mlp.bn1.weight')
        initial={n:torch.tensor([3.,4.]) for n in names};selected={n:torch.nn.Parameter(torch.tensor([6.,8.])) for n in names}
        result=r.parameter_drift(selected,initial,torch)
        for value in result['categories'].values():
            self.assertEqual(value['initial_squared_L2'],25.);self.assertEqual(value['delta_squared_L2'],25.)
            self.assertEqual(value['relative_L2'],1.);self.assertEqual(value['max_abs_delta'],4.)

    def test_nested_global_dependencies_and_paired_forward_assertions(self):
        fn=r.execute_function();functions=[fn]+[v for v in vars(r).values() if isinstance(v,types.FunctionType) and v.__module__==r.__name__]
        for function in functions:
            dependencies=set()
            def scan(code):
                dependencies.update(i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL')
                for value in code.co_consts:
                    if isinstance(value,types.CodeType):scan(value)
            scan(function.__code__)
            self.assertFalse(dependencies-set(function.__globals__)-set(vars(builtins)),function.__name__)
        source=(r.HERE/'execute_recovery.py').read_text()
        self.assertIn("initial==binding['baseline_comparison']['initial_state_sha256']",source)
        self.assertIn("binding['baseline_comparison']['initial_native_prediction_sha256'][row['target_id']]",source)
        self.assertFalse(torch.cuda.is_initialized())
        MEASUREMENTS.append({'all_entry_and_execute_nested_globals_bound':True,'paired_initial_state_and_per_frame_SHA_assertions_present':True})


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(r.RECIPE['seed'])
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    receipt={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'GPU_initialized':torch.cuda.is_initialized(),
             'torch':str(torch.__version__),'source_sha256':{str(r.HERE/n):r.sha(r.HERE/n) for n in r.OWN},'measurements':MEASUREMENTS,
             'errors':[str(v) for v in result.errors],'failures':[str(v) for v in result.failures],
             'scope':'CPU synthetic prior/source contracts and real small-native-MLP gradient/math, no real event/GT/quality, full model, SSH or GPU; root must run actual preflight/smoke.'}
    (r.HERE/'runner_cpu_results.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1)
