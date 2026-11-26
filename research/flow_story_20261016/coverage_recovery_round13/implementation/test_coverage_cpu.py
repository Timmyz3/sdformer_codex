#!/usr/bin/env python3
"""R13 identity, BN and differential runtime tests without full model or data arrays."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import ast
import builtins
import dis
import types
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.dont_write_bytecode=True
import numpy as np
import torch
import run_coverage_recovery as r
HERE=r.HERE
ROOT=HERE.parents[3]
C,PROBE,PARENT,R9,R8=r.load_runtime()
FIXPATH=r.OLD.source_path('fixture');sys.path.insert(0,str(FIXPATH.parent))
FIX=r.load_file(FIXPATH,'_r13_cpu_fixture')
sys.path.insert(0,str(ROOT/'research/flow_story_20261013/breadth'))
import run_bn_scope as scope
import evaluate_breadth as base
MERGE=PARENT.load_frozen_merge()
MEASUREMENTS=[]


def manifest_inputs():
    manifest=json.loads((HERE.parent/'data_protocol/manifest_v1.json').read_text())
    canonical=json.loads((ROOT/'research/flow_story_20261011/preflight/data_protocol/artifacts/common_targets.json').read_text())
    # Canonical format is validated independently by the existing metric reader.
    metrics=r.load_file(ROOT/'research/flow_story_20261012/evaluation/flow_eval_contract.py','_r13_cpu_metrics')
    rows=metrics.canonical_rows(ROOT/'research/flow_story_20261011/preflight/data_protocol/artifacts/common_targets.json')
    roles=metrics.auxiliary_roles(ROOT/'research/flow_story_20261011/preflight/data_protocol/artifacts/train_auxiliary_roles.csv',rows)
    return manifest,rows,roles


def toy():
    model,targets,shapes,x,functional=FIX.build()
    model.external=torch.nn.ModuleList([torch.nn.BatchNorm1d(4) for _ in range(54)])
    model.eval()
    all_bn,mlp=scope.bn_inventory(model,torch)
    values={}
    for n in mlp:
        m=all_bn[n]
        values[n+'.running_mean']=np.linspace(-.1,.1,m.num_features,dtype=np.float32)
        values[n+'.running_var']=np.linspace(.7,1.3,m.num_features,dtype=np.float32)
        values[n+'.num_batches_tracked']=np.array(176,dtype=np.int64)
    return model,targets,shapes,x,functional,values


class Tests(unittest.TestCase):
    def test_exact_member_roles_nested_orders_and_failure_cases(self):
        m,rows,roles=manifest_inputs();audit=r.validate_members(m,rows,roles)
        self.assertTrue(audit['per_step_sequence_matched'])
        for mutation in ('overlap','order','counts','role','canonical'):
            bad=copy.deepcopy(m);badroles=dict(roles)
            if mutation=='overlap':bad['split_sequences']['check'][0]=bad['split_sequences']['train'][0]
            elif mutation=='order':bad['train_order_diversity'][0],bad['train_order_diversity'][1]=bad['train_order_diversity'][1],bad['train_order_diversity'][0]
            elif mutation=='counts':bad['train_order_reference'][0]['target_id']=bad['train_order_reference'][1]['target_id']
            elif mutation=='role':badroles[bad['bn_members'][0]['target_id']]='gate_fit'
            else:bad['check_members'][0]['v2_target_filename']='altered.npy'
            with self.assertRaises(ValueError):r.validate_members(bad,rows,badroles)
        MEASUREMENTS.append(audit)

    def test_fresh_bn_only_24_buffers_external54_and_atomic_reject(self):
        model,targets,shapes,x,functional,values=toy()
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'bn.npz';receipt=Path(directory)/'receipt.json';receipt.write_text('{}')
            np.savez(p,**values)
            before=r.state_digest(model)
            bad=dict(values);key=next(k for k in bad if k.endswith('running_var'));bad[key]=bad[key].copy();bad[key][0]=-1
            bp=Path(directory)/'bad.npz';np.savez(bp,**bad)
            with self.assertRaisesRegex(ValueError,'variance'):r.apply_fresh_bn(model,bp,receipt,{},'',scope,base,torch,validate_receipt=False)
            self.assertEqual(r.state_digest(model),before)
            audit=r.apply_fresh_bn(model,p,receipt,{},'',scope,base,torch,validate_receipt=False)
            self.assertEqual((audit['BN_count'],audit['MLP_BN_count']),(78,24))
            self.assertFalse(audit['old_CMA_loaded'])
            self.assertTrue(all(m.running_mean is None and not m.track_running_stats for m in model.external))
            self.assertTrue(all(not m.training for m in model.modules()))
            self.assertEqual(sum(k.endswith(('running_mean','running_var','num_batches_tracked')) for k in model.state_dict()),72)
            for key,value in values.items():np.testing.assert_array_equal(model.state_dict()[key].numpy(),value)
            MEASUREMENTS.append({'fresh_buffers':72,'private_BN':24,'external_current_BN':54,'negative_variance_atomic_reject':True})

    def test_same_fresh_bn_all_arms_actual_dual_exact_and_training_gradient(self):
        for arm in r.ARMS:
            model,targets,shapes,x,functional,values=toy()
            with tempfile.TemporaryDirectory() as directory:
                p=Path(directory)/'bn.npz';receipt=Path(directory)/'receipt.json';receipt.write_text('{}');np.savez(p,**values)
                r.apply_fresh_bn(model,p,receipt,{},'',scope,base,torch,validate_receipt=False)
            selected=r.select_parameters(model,targets,expected=3104)
            frozen=r.state_digest(model,selected);bn_before=base.bn_signature(model,torch)
            with torch.no_grad():
                out,ex,branch=r.forward_train(model,functional,targets,shapes,x,arm,capture=True)
                for mode in ('actual',) if arm=='native' else ('actual','dual_reference'):
                    ref=r.reference_forward(R9,R8,PARENT,model,functional,targets,shapes,x,arm,MERGE,torch,mode=mode)
                    r.compare_all(out['flow'],branch,ex['capture_input_sha256'],ref,torch)
            optimizer=torch.optim.AdamW(list(selected.values()),lr=r.RECIPE['lr'],weight_decay=r.RECIPE['weight_decay'])
            out,_,_=r.forward_train(model,functional,targets,shapes,x,arm)
            sum((v-.37).square().mean() for v in out['flow']).backward()
            grad=r.gradient_audit(model,selected,torch)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            self.assertEqual(r.state_digest(model,selected),frozen)
            self.assertEqual(base.bn_signature(model,torch),bn_before)
            self.assertTrue(all(model.get_submodule(n) is m for n,m in targets.items()))
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
            MEASUREMENTS.append({'arm':arm,'newBN_actual_dual_exact':True,'gradients_finite':grad['all_allowed_gradients_present_finite'],
                                 'frozen_complement_and_BN_unchanged':True})

    def test_bn_receipt_rejects_legacy_and_wrong_member_or_payload(self):
        m,_,_=manifest_inputs()
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'buffers';p.write_bytes(b'not_arrays_receipt_only_test')
            v={'schema':'flow_round13_common_bn_v1','status':'COMMON_BN_COMPLETE','checkpoint_sha256':r.CHECKPOINT_SHA,
               'membership_sha256':'member','bn_target_ids':[x['target_id'] for x in m['bn_members']],
               'buffers_sha256':r.sha(p),'calibration_forwards':176,'calibration_bn_count':78,'retained_mlp_bn_count':24,
               'BN_scope':r.BN_SCOPE,'calibration_definition':'all78_reset_CMA_train_nonBN_eval','old_CMA_loaded':False,'GT_used':False}
            r.validate_bn_receipt(v,p,m,'member')
            for key,value in (('old_CMA_loaded',True),('GT_used',True),('membership_sha256','wrong'),('buffers_sha256','wrong'),('calibration_forwards',175)):
                bad=copy.deepcopy(v);bad[key]=value
                with self.assertRaises(ValueError):r.validate_bn_receipt(bad,p,m,'member')

    def test_execute_function_global_dependencies_are_bound(self):
        module=r.load_file(HERE/'execute_recovery.py','_r13_cpu_execute_dependency_check')
        namespace=dict(module.execute.__globals__)
        names=('require','RECIPE','BN_SCOPE','ARMS','LOSS_SHA','tile','apply_fresh_bn','state_digest','select_parameters',
               'forward_train','reference_forward','compare_all','gradient_audit','gradient_coverage','sha')
        namespace.update({k:getattr(r,k) for k in names})
        dependencies=set()
        def scan(code):
            dependencies.update(i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL')
            for value in code.co_consts:
                if isinstance(value,types.CodeType):scan(value)
        scan(module.execute.__code__)
        self.assertFalse(dependencies-set(namespace)-set(vars(builtins)))
        self.assertIn('json',dependencies)
        self.assertEqual(namespace['json'].loads(namespace['json'].dumps({'step':1})),{'step':1})
        MEASUREMENTS.append({'execute_all_nested_global_dependencies_bound':True,'dependencies':sorted(dependencies),
                             'step_logging_json_roundtrip':True,'scope':'bytecode dependency and logging primitive; no GPU/full-loop execution'})

    def test_forward_budget_and_execute_source_contract(self):
        m,_,_=manifest_inputs();n=len(m['check_members']);self.assertEqual(n,8)
        formal=sum(200+(4 if arm=='native' else 6)+5*n for arm in r.ARMS)
        smoke=sum(5+(4 if arm=='native' else 6) for arm in r.ARMS)
        self.assertEqual((formal,smoke),(982,42));self.assertEqual(2*(formal+smoke)+176,2224)
        text=(HERE/'execute_recovery.py').read_text();ast.parse(text)
        self.assertNotIn('scope.apply_scope',text);self.assertNotIn("'MLP_CMA176'",text)
        self.assertIn('row=train_rows[index]',text);self.assertIn('apply_fresh_bn(',text)
        self.assertIn("evaluate('endpoint',ARMS)",text)
        self.assertIn("expected_evaluation_forwards':0 if a.mode=='smoke' else 5*len(checks)",(HERE/'run_coverage_recovery.py').read_text())
        self.assertFalse(torch.cuda.is_initialized())
        MEASUREMENTS.append({'per_condition_smoke_forwards':smoke,'per_condition_formal_forwards':formal,
                             'two_conditions_plus_shared_calibration_full_forwards':2224,'not_independent_sample_count':True})


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(20261016)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
    receipt={'passed':result.wasSuccessful(),'tests_run':result.testsRun,'GPU_initialized':torch.cuda.is_initialized(),
             'torch':str(torch.__version__),'source_sha256':{str(HERE/n):r.sha(HERE/n) for n in r.OWN},
             'inherited_pins_sha256':r.sha(HERE/'inherited_pins.json'),'measurements':MEASUREMENTS,
             'errors':[str(x) for x in result.errors],'failures':[str(x) for x in result.failures],
             'scope':'CPU toy native MLPs, identity metadata and new BN source validation; no real event/GT arrays, SSH, GPU, full model or quality read.'}
    (HERE/'runner_cpu_results.json').write_text(json.dumps(receipt,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() and not torch.cuda.is_initialized() else 1)
