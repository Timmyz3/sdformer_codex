#!/usr/bin/env python3
"""Native small-MLP autograd, isolation and loss tests; never loads a full model."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
import torch
import run_recovery as runner
from source_contract import HERE,ROOT,sha,source_path,verify_all
TILE=runner.tile
C,PROBE,PARENT,R9,R8=runner.load_runtime()
FP=source_path('fixture');sys.path.insert(0,str(FP.parent))
spec=importlib.util.spec_from_file_location('_r12_small_fixture',FP)
FIX=importlib.util.module_from_spec(spec);spec.loader.exec_module(FIX)
MERGE=PARENT.load_frozen_merge()
MEASUREMENTS=[]


def build():
    torch.manual_seed(20261016)
    model,targets,shapes,x,functional=FIX.build()
    selected=runner.select_parameters(model,targets,expected=3104)
    return model,targets,shapes,x,functional,selected


class Tests(unittest.TestCase):
    def test_four_arms_five_updates_old_actual_dual_and_frozen_complement(self):
        for arm in runner.ARMS:
            model,targets,shapes,x,functional,selected=build()
            frozen=runner.state_digest(model,selected)
            before={n:p.detach().clone() for n,p in selected.items()}
            optimizer=torch.optim.AdamW(list(selected.values()),lr=runner.RECIPE['lr'],weight_decay=.01)
            records=[];forwards=0
            for endpoint in ('initial','final'):
                with torch.no_grad():
                    out,ex,branch=runner.forward_train(model,functional,targets,shapes,x,arm,capture=True);forwards+=1
                    flows=out['flow']
                    for mode in ('actual',) if arm=='native' else ('actual','dual_reference'):
                        ref=runner.reference_forward(R9,R8,PARENT,model,functional,targets,shapes,x,arm,MERGE,torch,mode=mode)
                        runner.compare_all(flows,branch,ex['capture_input_sha256'],ref,torch);forwards+=1
                if endpoint=='final':break
                for step in range(5):
                    optimizer.zero_grad(set_to_none=True)
                    output,service,_=runner.forward_train(model,functional,targets,shapes,x+step*.03,arm);forwards+=1
                    # Synthetic two-head task, independent of any optical-flow data.
                    loss=sum((o-.37).square().mean() for o in output['flow'])
                    self.assertTrue(loss.requires_grad)
                    loss.backward();audit=runner.gradient_audit(model,selected,torch)
                    torch.nn.utils.clip_grad_norm_(list(selected.values()),100.,error_if_nonfinite=True);optimizer.step()
                    records.append({'gradient_audit':audit})
                    self.assertEqual(runner.state_digest(model,selected),frozen)
                    self.assertTrue(all(model.get_submodule(n) is m for n,m in targets.items()))
                    self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
                optimizer.zero_grad(set_to_none=True)
            coverage=runner.gradient_coverage(records,[n for n in targets if TILE.is_selected_layer(n)],True)
            self.assertTrue(coverage['complete'])
            self.assertTrue(all(any(not torch.equal(before[n],p) for n,p in selected.items() if n.startswith(layer+'.'))
                                for layer in targets if TILE.is_selected_layer(layer)))
            self.assertEqual(forwards,9 if arm=='native' else 11)
            MEASUREMENTS.append({'arm':arm,'toy_steps':5,'old_actual_exact':True,
                'old_dual_exact':None if arm=='native' else True,'native_has_no_tiled_dual':arm=='native',
                'two_synthetic_outputs_and_all8_branches_exact':True,'frozen_complement_unchanged':True,
                'full_toy_forwards':forwards,'gradient_coverage':coverage,'toy_fc_rows':service['summed_fc1_rows']})

    def test_independent_cell_backward_input_and_every_native_parameter(self):
        # Independent per-cell call and CopySlices assembly, preserving actual T10.
        original=FIX.fixture.native(dtype=torch.float64).requires_grad_(True)
        u=torch.randn(10,1,11,14,4,dtype=torch.float64,requires_grad=True)
        independent=copy.deepcopy(original);v=u.detach().clone().requires_grad_(True)
        actual,_=TILE.tile_core(original,u,False)
        expected=torch.empty_like(v)
        for sy in range(0,11,9):
            for sx in range(0,14,9):
                for y in range(sy,min(sy+9,11),3):
                    for x in range(sx,min(sx+9,14),3):
                        ey,ex=min(y+3,sy+9,11),min(x+3,sx+9,14)
                        value=v[:,:,y:ey,x:ex].sum((2,3),keepdim=True)/((ey-y)*(ex-x))
                        expected[:,:,y:ey,x:ex]=independent(value.contiguous())
        torch.testing.assert_close(actual,expected,atol=1e-11,rtol=1e-11)
        weight=torch.randn_like(actual)
        gradients=torch.autograd.grad((actual*weight).sum(),[u]+list(original.parameters()))
        references=torch.autograd.grad((expected*weight).sum(),[v]+list(independent.parameters()))
        for a,b in zip(gradients,references):torch.testing.assert_close(a,b,atol=1e-10,rtol=1e-10)
        self.assertTrue(bool(gradients[0].abs().sum()>0))
        MEASUREMENTS.append({'independent_per_cell_input_and_parameter_backward':True,'dtype':'float64',
                             'temporal_steps':10,'shape':[10,1,11,14,4],'atol':1e-10,'rtol':1e-10})

    def test_frozen_middle_preserves_upstream_gradient_and_detach_is_rejected(self):
        model,targets,shapes,x,functional,selected=build()
        out,_,_=runner.forward_train(model,functional,targets,shapes,x,'CC')
        sum(y.square().mean() for y in out['flow']).backward()
        audit=runner.gradient_audit(model,selected,torch)
        self.assertTrue(all(v>0 for n,v in audit['per_layer_squared_L2'].items() if '.layers.0.' in n))
        model.zero_grad(set_to_none=True)
        middle=model.sttmultires_unet.encoders.swin3d.layers[1]
        real=middle.forward
        with patch.object(middle,'forward',side_effect=lambda value:real(value).detach()):
            out,_,_=runner.forward_train(model,functional,targets,shapes,x,'CC')
            sum(y.square().mean() for y in out['flow']).backward()
            with self.assertRaisesRegex(ValueError,'Missing/nonfinite'):runner.gradient_audit(model,selected,torch)

    def test_zero_gradient_allowed_per_step_but_smoke_coverage_required(self):
        model,targets,shapes,x,functional,selected=build()
        for p in selected.values():p.grad=torch.zeros_like(p)
        audit=runner.gradient_audit(model,selected,torch)
        names=[n for n in targets if TILE.is_selected_layer(n)]
        self.assertFalse(runner.gradient_coverage([{'gradient_audit':audit}],names)['complete'])
        with self.assertRaisesRegex(ValueError,'coverage incomplete'):
            runner.gradient_coverage([{'gradient_audit':audit}]*5,names,True)
        selected[next(iter(selected))].grad.flatten()[0]=float('nan')
        with self.assertRaisesRegex(ValueError,'nonfinite'):runner.gradient_audit(model,selected,torch)

    def test_exception_restores_modules_hooks_and_gradient_flags(self):
        model,targets,shapes,x,functional,selected=build();before=runner.state_digest(model)
        failing=targets[list(targets)[4]].fc2
        with patch.object(failing,'forward',side_effect=RuntimeError('injected producer')):
            with self.assertRaisesRegex(RuntimeError,'injected producer'):
                runner.forward_train(model,functional,targets,shapes,x,'fine_s2b2',capture=True)
        self.assertEqual(runner.state_digest(model),before)
        self.assertTrue(all(model.get_submodule(n) is m for n,m in targets.items()))
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        with self.assertRaisesRegex(RuntimeError,'freeze failure'):
            with runner.frozen_parameters(model):raise RuntimeError('freeze failure')
        self.assertEqual({n for n,p in model.named_parameters() if p.requires_grad},set(selected))
        with patch.object(list(targets.values())[1].fc1,'register_forward_pre_hook',side_effect=RuntimeError('registration')):
            with self.assertRaisesRegex(RuntimeError,'registration'):
                runner.forward_train(model,functional,targets,shapes,x,'CC')
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))

    def test_actual_deployed_loss_all_four_heads_all_valid_including_large_flow(self):
        path=ROOT/'SDformer/third_party/SDformerFlow/loss/flow_supervised.py'
        self.assertEqual(sha(path),runner.LOSS_SHA)
        spec=importlib.util.spec_from_file_location('_r12_cpu_actual_loss',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        loss=module.flow_loss_supervised({'metrics':{'flow_scaling':1},'loss':{'lambda_mod':1,'lambda_ang':1}},torch.device('cpu'))
        gt=torch.tensor([[[[500.,3.],[2.,4.]],[[0.,4.],[0.,3.]]]])
        mask=torch.tensor([[[[1.,1.],[0.,1.]]]])
        preds=[torch.full_like(gt,float(i),requires_grad=True) for i in range(4)]
        actual=loss(preds,gt,mask,gamma=None)
        independent=sum(((p-gt).square().sum(1)+1e-8).sqrt().mul(mask[:,0]).sum()/(mask.sum()+1e-9) for p in preds)/4
        self.assertTrue(torch.equal(actual,independent));actual.backward()
        self.assertTrue(all(bool(p.grad[:,:,0,0].abs().sum()>0) for p in preds))
        self.assertTrue(all(bool((p.grad[:,:,1,0]==0).all()) for p in preds))
        MEASUREMENTS.append({'deployed_loss_sha256':sha(path),'all_four_head_gradients':True,
            'valid_500px_GT_retained':True,'invalid_GT_zero_gradient':True,'no_event_input_or_filter':True})

    def test_production_rows_membership_source_and_safe_import(self):
        model,_,_,_,_,_=build();_,production=PARENT.target_layers(model)
        self.assertEqual({a:runner.expected_rows(production,a) for a in runner.ARMS},runner.ROWS)
        self.assertEqual({runner.expected_rows(production,a,True) for a in runner.ARMS if a!='native'},{609600})
        self.assertEqual(sha(HERE/'check_membership_v1.json'),runner.MEMBER_SHA)
        self.assertEqual(sha(HERE/'selection_manifest_v1.json'),runner.SELECTION_SHA)
        member=json.loads((HERE/'check_membership_v1.json').read_text())
        self.assertEqual(len(member['groups']['fit']),20);self.assertEqual(len(member['groups']['check']),8)
        self.assertEqual(len(verify_all()),14)
        self.assertEqual(sum((9 if a=='native' else 11) for a in runner.ARMS),42)
        self.assertEqual(sum((244 if a=='native' else 246) for a in runner.ARMS),982)
        code="import os,sys;os.environ['CUDA_VISIBLE_DEVICES']='marker';sys.path.insert(0,"+repr(str(HERE))+");import run_recovery as r;r.load_runtime();import torch;assert os.environ['CUDA_VISIBLE_DEVICES']=='marker';assert not torch.cuda.is_initialized()"
        result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_stale_receipts_and_smoke_incomplete_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cpu.json'
            sources=[HERE/n for n in runner.OWN]
            value={'passed':True,'GPU_initialized':False,'source_sha256':{p.name:sha(p) for p in sources}}
            path.write_text(json.dumps(value));PARENT.checked_cpu_receipt(path,sources)
            value['source_sha256']['run_recovery.py']='0'*64;path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'current source'):PARENT.checked_cpu_receipt(path,sources)
            paths=[];binding={'runner_sha256':sha(runner.__file__),'recipe':runner.RECIPE,'checkpoint_sha256':C.CHECKPOINT_SHA,
                'wrapper_sha256':sha(TILE.__file__),'runner_checks_sha256':'synthetic_cpu_receipt',
                'source_pins_sha256':sha(HERE/'source_pins.json'),'loss_sha256':runner.LOSS_SHA,
                'membership_sha256':runner.MEMBER_SHA,'selection_sha256':runner.SELECTION_SHA,'allowed_parameter_count':runner.PARAMETERS}
            for arm in runner.ARMS:
                p=Path(directory)/(arm+'.json');paths.append(p)
                p.write_text(json.dumps({'schema':runner.SCHEMA,'arm':arm,'status':'ROUND12_SMOKE_COMPLETE',
                    'completed_optimizer_steps':5,'training_performed':True,'binding':{**binding,'arm':arm,'mode':'smoke'},
                    'frozen_state_unchanged':True,'BN_buffers_unchanged':True,'original_modules_restored':True,'all_hooks_removed':True,
                    'smoke_weights_discarded':True,'gradient_coverage':{'complete':True},'initial_numerical_pair':{'all_flow_outputs_exact':True},
                    'final_numerical_pair':{'all_flow_outputs_exact':True}}))
            self.assertEqual(len(runner.validate_smokes(paths,binding)),4)
            wrong={**binding,'wrapper_sha256':'0'*64}
            with self.assertRaisesRegex(ValueError,'source/recipe'):runner.validate_smokes(paths,wrong)
            changed=json.loads(paths[0].read_text());changed['completed_optimizer_steps']=4;paths[0].write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'Incomplete smoke'):runner.validate_smokes(paths,binding)


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261016)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    sources=[HERE/n for n in runner.OWN]
    receipt={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'GPU_initialized':torch.cuda.is_initialized(),
        'torch':str(torch.__version__),'full_model_run':False,'source_sha256':{str(p):sha(p) for p in sources},
        'inherited_sources':verify_all(),'measurements':MEASUREMENTS,'errors':[str(x) for x in result.errors],
        'failures':[str(x) for x in result.failures],
        'scope':'Twelve C4 synthetic native MLPs, independent cell backward, same production trainable and frozen reference APIs; no full model, data, checkpoint, SSH, GPU or quality result'}
    (HERE/'runner_cpu_results.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1)
