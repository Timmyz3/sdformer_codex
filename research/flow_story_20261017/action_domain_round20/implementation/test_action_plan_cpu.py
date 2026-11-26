"""R20 bounded-plan and real original-MLP integration checks; CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import types
import unittest
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
SOURCE=ROOT/'SDformer/third_party/SDformerFlow'
PRIMITIVE=HERE.parents[1]/'current_bn_round16/implementation'
sys.path[:0]=[str(ROOT/'research/flow_story_20261011/preflight/cpu_numerics/deps'),str(SOURCE),str(PRIMITIVE)]
import torch
from spikingjelly.activation_based import functional
from models.STSwinNet_SNN.Spiking_swin_transformer3D import MS_Spiking_Mlp
import native_dense_bn_v2 as primitive
import run_action_domain as p

MEASUREMENTS=[]


def micro_model():
    torch.manual_seed(17)
    mlp=MS_Spiking_Mlp(4,hidden_features=16,norm_layer='BN',drop=0.,num_steps=10,
         spike_norm='BN',neuron_type='psn',v_th=.1,v_reset=None,
         surrogate_fun='surrogate.ATan()',tau=2.,detach_reset=True)
    functional.set_step_mode(mlp,'m')
    for wrapper in (mlp.bn1,mlp.bn2):
        bn=wrapper.norm_layer
        bn.track_running_stats=False;bn.running_mean=None;bn.running_var=None;bn.num_batches_tracked=None
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();node=self
            names=p.TARGET.split('.')
            for name in names[:-1]:
                child=torch.nn.Module();node.add_module(name,child);node=child
            node.add_module(names[-1],mlp)
            self.external=torch.nn.ModuleList([torch.nn.BatchNorm1d(1,track_running_stats=False) for _ in range(76)])
        def forward(self,u):
            for bn in self.external:bn(torch.tensor([[0.],[1.]],dtype=u.dtype))
            branch=self.get_submodule(p.TARGET)(u)
            # Four synthetic heads only exercise capture/compare plumbing.
            return {'flow':[branch.mean(-1) for _ in range(4)]}
    model=Model().eval().requires_grad_(False)
    return model,mlp,torch.randn(10,1,30,40,4)


class Checks(unittest.TestCase):
    def test_plan_and_independent_geometry(self):
        rows=[dict(target_id=str(i)) for i in range(8)]
        items=p.plan(rows,'action44')
        self.assertEqual(len(items),44)
        self.assertEqual([v['arm'] for v in items[:9]],['native','FF','CC','CC_repeat1','CC_repeat2','F12','T8','T8_repeat1','T8_repeat2'])
        self.assertEqual(sum(v['quality_endpoint'] for v in items),40)
        self.assertEqual(sum(v['execution']=='native' for v in items),8)
        self.assertEqual(sum(v['execution']=='compact' for v in items),36)
        self.assertEqual(sum(p.expected_rows(v) for v in items),24343920)
        self.assertEqual(44*78,3432)
        self.assertEqual(sum((4 if v['execution']=='native' else 6) if p.save_captures(v,'0') else 0 for v in items),116)
        self.assertEqual(44*4+116,292)
        self.assertEqual([v['forward_id'] for v in items if v['arm']=='CC'][:1],[3])
        self.assertEqual([v['forward_id'] for v in items if v['arm']=='T8'][:1],[7])
        for tid in map(str,range(8)):
            self.assertEqual([x['endpoint_arm'] for x in items if x['target_id']==tid and x['quality_endpoint']],list(p.ARMS))
        for v in items:
            if not v['quality_endpoint']:
                self.assertEqual(v['target_id'],'0')
                self.assertEqual(v['arm'].split('_')[0],v['endpoint_arm'])
        # Independent true-tail area and ceil cover; never call original pack here.
        for arm in p.ARMS[1:]:
            mask=p.mask_for(arm,torch);reps=0;covered=0
            for y in range(0,30,9):
                for x in range(0,40,9):
                    hh=min(9,30-y);ww=min(9,40-x)
                    stride=1 if bool(mask[y//9,x//9]) else 3
                    reps+=((hh+stride-1)//stride)*((ww+stride-1)//stride);covered+=hh*ww
            self.assertEqual(covered,1200)
            item=dict(arm=arm,execution='compact')
            self.assertEqual(546000+10*reps,p.expected_rows(item))
            self.assertEqual(reps,dict(FF=1200,CC=140,F12=1004,T8=336)[arm])
        self.assertTrue(torch.equal(p.mask_for('F12',torch),~p.mask_for('T8',torch)))
        self.assertEqual(int(p.mask_for('F12',torch).sum()),12)
        self.assertEqual(int(p.mask_for('T8',torch).sum()),8)
        for name in ('CC_repeat1','CC_repeat2','T8_repeat1','T8_repeat2'):
            self.assertTrue(torch.equal(p.mask_for(name,torch),p.mask_for(name.split('_')[0],torch)))
        for bad in ('fine_0_0','F12_repeat1','unknown','native'):
            with self.assertRaises(ValueError):p.mask_for(bad,torch)
        with self.assertRaises(ValueError):p.plan(rows,'compact122')
        with self.assertRaises(ValueError):p.plan(rows[:-1],'action44')
        self.assertEqual(p.GRAPH_ID,'r20_W271482_all78current_s2b2_R16V2_action_domain')

    def test_original_primitive_pins_and_no_expanded_call(self):
        self.assertEqual(p.sha(PRIMITIVE/'native_dense_bn_v2.py'),p.PRIMITIVE_SHA)
        self.assertEqual(p.sha(PRIMITIVE/'compact_current_mlp.py'),p.SHARED_MATH_SHA)
        parsed=ast.parse((HERE/'run_action_domain.py').read_text())
        attrs=[n.func.attr for n in ast.walk(parsed) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertIn('compact_mlp',attrs);self.assertNotIn('expanded_mlp',attrs)
        actual=p.load_frozen_primitive(PRIMITIVE/'native_dense_bn_v2.py')
        shared=sys.modules['compact_current_mlp']
        self.assertIs(actual.pack,shared.pack)
        fake=types.ModuleType('compact_current_mlp');fake.__file__='/wrong/import/compact_current_mlp.py'
        for name in ('validate','pack','expand','expanded_mlp','require'):setattr(fake,name,getattr(shared,name))
        sys.modules['compact_current_mlp']=fake
        try:
            with self.assertRaisesRegex(ValueError,'Actually imported'):
                p.load_frozen_primitive(PRIMITIVE/'native_dense_bn_v2.py')
        finally:sys.modules['compact_current_mlp']=shared

    def test_original_numeric_gate_and_bitwise_replay(self):
        capture={n:torch.zeros(2) for n in ('u','branch','sn1','sn2')}
        a=([torch.zeros(2) for _ in range(4)],capture,{})
        b=([torch.zeros(2) for _ in range(4)],{k:v.clone() for k,v in capture.items()},{})
        self.assertTrue(p.numerical_pair(a,b,torch)['pass'])
        b[1]['sn1'][0]=1;self.assertFalse(p.numerical_pair(a,b,torch)['pass']);b[1]['sn1'][0]=0
        b[0][3][0]=1e-3;self.assertFalse(p.numerical_pair(a,b,torch)['pass']);b[0][3][0]=0
        for obj in (a,b):
            obj[1]['sn1_packed']=torch.zeros(2);obj[1]['sn2_packed']=torch.zeros(2)
        self.assertTrue(p.deterministic_pair(a,b,torch)['pass'])
        b[1]['sn1_packed'][0]=-0.
        self.assertFalse(p.deterministic_pair(a,b,torch)['pass'])
        b[1]['sn1_packed'][0]=0.
        b[0][0][0]=torch.nextafter(torch.tensor(0.),torch.tensor(1.))
        self.assertFalse(p.deterministic_pair(a,b,torch)['pass'])
        with self.assertRaises(ValueError):p.compare(torch.tensor([float('nan')]),torch.zeros(1),torch)

    def test_real_original_mlp_wrapper_capture_and_replay(self):
        model,mlp,u=micro_model();targets={p.TARGET:mlp};shapes={p.TARGET:tuple(u.shape)}
        before={k:p.tensor_sha(v) for k,v in model.state_dict().items()}
        outputs={};references={};checks=[]
        for i,arm in enumerate(('native','FF','CC','CC_repeat1','CC_repeat2','F12','T8','T8_repeat1','T8_repeat2')):
            item=dict(forward_id=i+1,target_id='synthetic',arm=arm,
                      execution='native' if arm=='native' else 'compact',
                      quality_endpoint='_repeat' not in arm,endpoint_arm=arm.split('_')[0] if '_repeat' in arm else arm)
            out=p.forward(model,functional,targets,shapes,u,item,primitive,'NATIVE_DENSE_CURRENT',torch)
            outputs[arm]=out
            check=p.numerical_check(item,out,outputs.get('native'),references,'synthetic',torch)
            if check:checks.append(dict(check,forward_id=i+1))
            self.assertEqual(sum(out[2]['native_BN_calls'].values()),78)
            self.assertEqual(out[2]['all12_observed_fc_rows'][p.TARGET]['fc1_rows'],p.expected_rows(item)-546000)
            self.assertEqual(out[2]['tensor_records']['u']['sha256'],p.tensor_sha(u))
            self.assertIs(model.get_submodule(p.TARGET),mlp)
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
            if arm!='native':
                audit=out[2]['target_execution']
                self.assertFalse(audit['native_MLP_shortcut']);self.assertEqual(sum(audit['areas']),1200)
                for sn in ('sn1','sn2'):
                    self.assertTrue(torch.equal(out[1][sn],primitive.expand(out[1][sn+'_packed'],audit['cells'],30,40)))
        self.assertTrue(p.numerical_pair(outputs['FF'],outputs['native'],torch)['pass'])
        for arm in ('CC','T8'):
            for suffix in ('_repeat1','_repeat2'):
                self.assertTrue(p.deterministic_pair(outputs[arm+suffix],outputs[arm],torch)['pass'])
        with self.assertRaisesRegex(ValueError,'Replay shape/dtype'):
            p.deterministic_pair(outputs['T8_repeat1'],outputs['CC'],torch)
        self.assertEqual([c['reference_forward_id'] for c in checks if c['kind']=='CC_exact_replay'],[3,3])
        self.assertEqual([c['reference_forward_id'] for c in checks if c['kind']=='T8_exact_replay'],[7,7])
        self.assertEqual(len(checks),5)
        self.assertTrue(all(c['pass'] for c in checks))
        bad=dict(outputs['T8_repeat1'][2],target_id='wrong-frame')
        with self.assertRaisesRegex(ValueError,'Actual first matching'):
            p.numerical_check(bad,outputs['T8_repeat1'],outputs['native'],references,'synthetic',torch)
        with self.assertRaisesRegex(ValueError,'arm/frame'):
            p.numerical_check(outputs['T8_repeat1'][2],outputs['T8_repeat1'],outputs['native'],{'T8':outputs['CC']},'synthetic',torch)
        self.assertEqual(before,{k:p.tensor_sha(v) for k,v in model.state_dict().items()})
        MEASUREMENTS.append(dict(real_original_MLP_PSN=True,synthetic_78BN_fixture=True,
                                 local_micro_forwards=9,full_model_forwards=0,
                                 actual_compact_rows=[outputs[n][2]['all12_observed_fc_rows'][p.TARGET]['fc1_rows'] for n in outputs]))

    def test_exception_restores_target_and_hooks(self):
        model,mlp,u=micro_model()
        def fail(*args):raise RuntimeError('deliberate primitive failure')
        fake=types.SimpleNamespace(compact_mlp=fail)
        item=dict(arm='CC',execution='compact')
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            p.forward(model,functional,{p.TARGET:mlp},{p.TARGET:tuple(u.shape)},u,item,fake,'NATIVE_DENSE_CURRENT',torch)
        self.assertIs(model.get_submodule(p.TARGET),mlp)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);a=parser.parse_args()
    assert not a.output.exists()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    out=dict(schema='flow_r20_orchestration_cpu_v1',status='PASS' if result.wasSuccessful() else 'FAIL',
             graph_id=p.GRAPH_ID,source_sha256={n:p.sha(HERE/n) for n in ('run_action_domain.py','test_action_plan_cpu.py','PROTOCOL_V1.md')},
             primitive_sha256=p.sha(PRIMITIVE/'native_dense_bn_v2.py'),shared_math_sha256=p.sha(PRIMITIVE/'compact_current_mlp.py'),
             tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),measurements=MEASUREMENTS,
             planned_forwards=44,quality_endpoints=40,expected_FC_rows_each=24343920,expected_BN_calls=3432,
             full_model_forwards=0,GPU_used=False,GT_read=False,quality_read=False,
             wall_seconds=time.monotonic()-start)
    assert not torch.cuda.is_initialized()
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
    raise SystemExit(0 if result.wasSuccessful() else 1)
