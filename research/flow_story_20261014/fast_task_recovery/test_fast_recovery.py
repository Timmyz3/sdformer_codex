"""Small CPU boundary/gradient tests; not native GPU recovery evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import tempfile
import os
import subprocess
import unittest

import torch
from torch import nn
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parent/'fast_warmstart'))
import run_fast_recovery as runner
import trainable_fast_tile as tile
ADAPTER=HERE.parents[1]/'flow_story_20261012/transfer_experiments/dynamicswin'
FastBranch=runner.load_fast_class(ADAPTER)


class SyntheticTeacher(nn.Module):
    def __init__(self):
        super().__init__();self.fc1=nn.Linear(8,16);self.fc2=nn.Linear(16,8)
    def forward(self,x):
        return self.fc2(torch.tanh(self.fc1(x+.2*x.mean(0,keepdim=True))))


def fixture():
    torch.manual_seed(17)
    native=SyntheticTeacher().eval().requires_grad_(False)
    fast=FastBranch(8).eval()
    u=torch.randn(10,1,10,13,8)
    keep=torch.tensor([[True,False],[False,True]])
    return native,fast,u,keep


def bank_fixture():
    fitpath=HERE.parents[1]/'flow_story_20261013/recovery/manifests/svs_fit20.json'
    fit=json.loads(fitpath.read_text())
    rows=[{'target_id':t,'sequence':t.rsplit('_',1)[0],'protocol_split':'train'} for t in fit['target_ids']]
    dev=[{'target_id':f'{seq}_{i:04d}','sequence':seq,'protocol_split':'train'}
         for i in range(4) for seq in ('zurich_city_01_a','zurich_city_05_a')]
    roles={r['target_id']:'gate_fit' for r in rows};roles.update({r['target_id']:'gate_development' for r in dev})
    return fit,rows,dev,roles


class RecoveryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_actual_dual_equal_full_time_tail_and_MSE_target(self):
        native,fast,u,keep=fixture()
        with torch.no_grad():
            actual,missing,ar=tile.tile_core(native,fast,u,keep,'actual')
            dual,mse,dr=tile.tile_core(native,fast,u,keep,'dual')
        torch.testing.assert_close(actual,dual,atol=0,rtol=0)
        self.assertIsNone(missing);self.assertEqual(ar['slow_points'],85);self.assertEqual(ar['fast_points'],45)
        self.assertEqual(ar['observed'],{'slow_fc1_rows':850,'slow_fc2_rows':850,'fast_fc_rows':450})
        self.assertEqual(dr['observed'],{'slow_fc1_rows':1300,'slow_fc2_rows':1300,'fast_fc_rows':1300})
        self.assertEqual(dr['MSE_elements'],10*45*8)
        mask=(~keep).repeat_interleave(9,0).repeat_interleave(9,1)[:10,:13]
        reference=(fast(u)-native(u)).square()[:,:,mask,:].mean()
        torch.testing.assert_close(mse,reference,rtol=1e-6,atol=1e-7)
        self.assertFalse(dr['external_residual_added_here'])

    def test_R0_learning_keeps_teacher_input_and_frozen_suffix_unchanged(self):
        native,fast,u,keep=fixture();u_before=u.clone();teacher_before=runner.source_digest(native)
        opt=torch.optim.AdamW(fast.parameters(),lr=.003,weight_decay=.0001)
        values=[]
        for _ in range(15):
            opt.zero_grad(set_to_none=True)
            out,mse,rec=tile.tile_core(native,fast,u,keep,'dual')
            mse.backward();self.assertTrue(all(p.grad is None for p in native.parameters()))
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in fast.parameters()))
            values.append(float(mse.detach()));opt.step()
        self.assertLess(values[-1],values[0]);self.assertEqual(teacher_before,runner.source_digest(native))
        torch.testing.assert_close(u,u_before,atol=0,rtol=0)

    def test_R1_gradient_crosses_frozen_suffix_and_eval_BN(self):
        native,fast,u,keep=fixture()
        # External current-stat BN remains differentiable in eval although its parameters are frozen.
        suffix=nn.Sequential(nn.Linear(8,5),nn.BatchNorm1d(5,track_running_stats=False),nn.Linear(5,2)).eval().requires_grad_(False)
        state=runner.source_digest(suffix)
        out,_,_=tile.tile_core(native,fast,u,keep,'dual')
        prediction=suffix((u+out).reshape(-1,8));target=torch.ones_like(prediction)*.25
        loss=(prediction-target).square().mean();loss.backward()
        self.assertTrue(loss.requires_grad)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in fast.parameters()),0)
        self.assertTrue(all(p.grad is None for p in native.parameters()))
        self.assertTrue(all(p.grad is None for p in suffix.parameters()))
        self.assertEqual(state,runner.source_digest(suffix))

    def test_allkeep_allfast_null_MSE_and_invalid_input_guards(self):
        native,fast,u,keep=fixture()
        out,missing,rec=tile.tile_core(native,fast,u,torch.ones_like(keep),'dual')
        self.assertIsNone(missing);self.assertEqual(rec['fast_points'],0)
        out,mse,rec=tile.tile_core(native,fast,u,torch.zeros_like(keep),'dual')
        self.assertEqual(rec['MSE_elements'],u.numel());self.assertIsNotNone(mse)
        for bad in (u[:9],u.double(),u.clone().requires_grad_(True),u*float('nan')):
            with self.assertRaises(ValueError):tile.tile_core(native,fast,bad,keep,'dual')
        native.fc1.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError,'Frozen'):tile.tile_core(native,fast,u,keep,'dual')

    def test_fit_dev_membership_and_fixed_order_guards(self):
        fit,rows,dev,roles=bank_fixture()
        self.assertEqual(runner.membership(fit,rows+dev,roles,dev),rows)
        badroles=dict(roles);badroles[rows[0]['target_id']]='gate_development'
        with self.assertRaisesRegex(ValueError,'outside'):runner.membership(fit,rows+dev,badroles,dev)
        bad=copy.deepcopy(fit);bad['target_ids'].reverse()
        with self.assertRaisesRegex(ValueError,'ordered'):runner.membership(bad,rows+dev,roles,dev)
        baddev=copy.deepcopy(dev);baddev[0]=copy.deepcopy(rows[0])
        with self.assertRaises(ValueError):runner.membership(fit,rows+dev,roles,baddev)

    def test_mask_bank_rejects_edited_source_role_and_budget(self):
        fit,rows,dev,roles=bank_fixture()
        prior=json.loads((HERE.parent/'fast_warmstart/runtime_v1/results.json').read_text())
        selector=prior['frames'][0]['arms']['independent_trained']['runtime_selector']
        keep=prior['frames'][0]['arms']['independent_trained']['execution'][0]['keep']
        binding={'frozen_parent':'synthetic','recipe':runner.RECIPE}
        bank={'schema':'flow_fast_recovery_mask_bank_v1','status':'MASK_BANK_COMPLETE','binding':binding,
              'fit_ids':[r['target_id'] for r in rows],'dev_ids':[r['target_id'] for r in dev],'frames':[]}
        for i,row in enumerate(rows+dev):
            bank['frames'].append({'target_id':row['target_id'],'sequence':row['sequence'],
                'role':'fit' if i<20 else 'dev','keep':copy.deepcopy(keep),'selector':copy.deepcopy(selector),
                'u_sha256':selector['input_sha256'],'slow_points':900,'target_cost':900})
        self.assertEqual(len(runner.validate_bank(bank,binding,rows,dev)),28)
        # Synthetic non900 but internally consistent target demonstrates cap-vs-exact distinction.
        flexible=copy.deepcopy(bank);f=flexible['frames'][0]
        y,x=next((y,x) for y in range(4) for x in range(5) if f['keep'][y][x])
        f['keep'][y][x]=False
        actual=900-min(9,30-y*9)*min(9,40-x*9)
        f['slow_points']=f['target_cost']=actual
        f['selector']['selection']['slow_mask']=copy.deepcopy(f['keep'])
        f['selector']['selection']['slow_service_cost']=actual
        f['selector']['common_budget']['target_cost']=actual
        self.assertEqual(runner.validate_bank(flexible,binding,rows,dev)[f['target_id']]['slow_points'],actual)
        for key in ('slow_points','target_cost'):
            bad=copy.deepcopy(flexible);bad['frames'][0][key]+=1
            with self.subTest(mismatch=key),self.assertRaises(ValueError):runner.validate_bank(bad,binding,rows,dev)
        bad=copy.deepcopy(flexible);bad['frames'][0]['selector']['common_budget']['target_cost']+=1
        with self.assertRaises(ValueError):runner.validate_bank(bad,binding,rows,dev)
        for kind in ('source','role','budget','input'):
            bad=copy.deepcopy(bank)
            if kind=='source':bad['binding']['frozen_parent']='changed'
            if kind=='role':bad['frames'][20]['role']='fit'
            if kind=='budget':bad['frames'][0]['keep'][0][0]^=True
            if kind=='input':bad['frames'][0]['u_sha256']='0'*64
            with self.subTest(kind=kind),self.assertRaises(ValueError):runner.validate_bank(bad,binding,rows,dev)
        self.assertFalse(torch.cuda.is_initialized())

    def test_full_run_requires_matching_arm_smoke_not_other_weights(self):
        binding={'recipe':runner.RECIPE,'source':'fixed'}
        smoke={'schema':'flow_fast_task_recovery_results_v1','status':'SMOKE_COMPLETE','arm':'R0',
            'completed_optimizer_steps':1,'binding':binding,'mask_bank_sha256':'a'*64,
            'original_model_unchanged':True,'BN_buffers_unchanged':True,
            'initial_numerical_pair':{'all_flow_outputs_exact':True},'final_numerical_pair':{'all_flow_outputs_exact':True}}
        runner.validate_smoke(smoke,'R0',binding,'a'*64)
        with self.assertRaises(ValueError):runner.validate_smoke(smoke,'R1',binding,'a'*64)
        with self.assertRaises(ValueError):runner.validate_smoke(smoke,'R0',binding,'b'*64)
        bad=copy.deepcopy(smoke);bad['BN_buffers_unchanged']=False
        with self.assertRaises(ValueError):runner.validate_smoke(bad,'R0',binding,'a'*64)

    def test_real_inherited_parser_accepts_new_output_preserves_old_contract(self):
        breadth=HERE.parents[1]/'flow_story_20261013/breadth'
        sys.path.insert(0,str(breadth))
        import evaluate_breadth_extended as extended
        parent=json.loads((HERE.parent/'fast_warmstart/runtime_contract_v1.json').read_text())
        args=list(parent['base_args'])
        # A real existing artifact reproduces the parser's original rejection condition.
        args[args.index('--output')+1]=str(HERE.parent/'fast_warmstart/runtime_v1/results.json')
        original=list(args)
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'new'/'results.json'
            parsed=runner.parse_recovery_args(args,out,extended)
            self.assertEqual(parsed.output,out.resolve())
            self.assertEqual(args,original)
            self.assertFalse(out.exists())
            self.assertEqual(parsed.geometry,'full9')
        with self.assertRaises(ValueError):runner.parse_recovery_args(args+['--output','extra'],Path('/tmp/new'),extended)

    def test_cold_import_of_runtime_topology_preserves_CUDA_environment(self):
        # No CUDA API is invoked: fresh interpreter catches module import side effects.
        code=("import os,sys; sys.path.insert(0,"+repr(str(HERE))+"); "
              "import run_fast_recovery as r; before=os.environ['CUDA_VISIBLE_DEVICES']; "
              "c=r.load_fast_class("+repr(str(ADAPTER))+"); "
              "assert os.environ['CUDA_VISIBLE_DEVICES']==before=='0'; "
              "assert 'fit_fast_cache' not in sys.modules; "
              "import torch; assert not torch.cuda.is_initialized(); "
              "assert c(8).fc1[1].out_features==8")
        env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']='0'
        child=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=30)
        self.assertEqual(child.returncode,0,child.stderr)


if __name__=='__main__':
    started=time.monotonic();outcome=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryChecks))
    report={'schema':'flow_fast_task_recovery_CPUchecks_v1','status':'PASS' if outcome.wasSuccessful() else 'FAIL',
        'tests_run':outcome.testsRun,'failures':len(outcome.failures),'errors':len(outcome.errors),'wall_seconds':time.monotonic()-started,
        'source_sha256':{p.name:runner.sha(p) for p in (Path(runner.__file__),Path(tile.__file__),Path(__file__))},
        'scope':'Synthetic gradient/teacher/BN/tail and metadata guards; no native full model, no GPU, no scientific recovery result'}
    (HERE/'cpu_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
