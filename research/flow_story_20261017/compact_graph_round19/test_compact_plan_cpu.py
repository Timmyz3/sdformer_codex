"""R19 bounded-plan and real original-MLP integration checks; CPU only."""
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
ROOT=HERE.parents[2]
SOURCE=ROOT/'SDformer/third_party/SDformerFlow'
PRIMITIVE=HERE.parent/'current_bn_round16/implementation'
sys.path[:0]=[str(ROOT/'research/flow_story_20261011/preflight/cpu_numerics/deps'),str(SOURCE),str(PRIMITIVE)]
import torch
from spikingjelly.activation_based import functional
from models.STSwinNet_SNN.Spiking_swin_transformer3D import MS_Spiking_Mlp
import native_dense_bn_v2 as primitive
import run_compact_graph as p

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
        items=p.plan(rows,'compact122')
        self.assertEqual(len(items),122)
        self.assertEqual([v['arm'] for v in items[:6]],['native','FF','CC','CC_repeat1','CC_repeat2','fine_0_0'])
        self.assertEqual(sum(v['quality_endpoint'] for v in items),120)
        self.assertEqual(sum(v['execution']=='native' for v in items),8)
        self.assertEqual(sum(v['execution']=='compact' for v in items),114)
        self.assertEqual(sum(p.expected_rows(v) for v in items),67021520)
        self.assertEqual(122*78,9516)
        # Independent ceil cover for true full/tail tiles, without calling pack.
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
            if arm.startswith('fine_'):
                self.assertEqual(int(mask.sum()),1);self.assertFalse(mask[3].any());self.assertFalse(mask[:,4].any())
        self.assertEqual(p.GRAPH_ID,'r19_W271482_all78current_s2b2_R16V2_compact')

    def test_original_primitive_pins_and_no_expanded_call(self):
        self.assertEqual(p.sha(PRIMITIVE/'native_dense_bn_v2.py'),p.PRIMITIVE_SHA)
        self.assertEqual(p.sha(PRIMITIVE/'compact_current_mlp.py'),p.SHARED_MATH_SHA)
        parsed=ast.parse((HERE/'run_compact_graph.py').read_text())
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
        outputs={}
        for i,arm in enumerate(('native','FF','CC','CC_repeat1','fine_2_3')):
            item=dict(forward_id=i+1,target_id='synthetic',arm=arm,
                      execution='native' if arm=='native' else 'compact',
                      quality_endpoint=arm!='CC_repeat1',endpoint_arm='CC' if arm=='CC_repeat1' else arm)
            out=p.forward(model,functional,targets,shapes,u,item,primitive,'NATIVE_DENSE_CURRENT',torch)
            outputs[arm]=out
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
        self.assertTrue(p.deterministic_pair(outputs['CC_repeat1'],outputs['CC'],torch)['pass'])
        self.assertEqual(before,{k:p.tensor_sha(v) for k,v in model.state_dict().items()})
        MEASUREMENTS.append(dict(real_original_MLP_PSN=True,synthetic_78BN_fixture=True,
                                 local_micro_forwards=5,full_model_forwards=0,
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
    out=dict(schema='flow_r19_orchestration_cpu_v1',status='PASS' if result.wasSuccessful() else 'FAIL',
             graph_id=p.GRAPH_ID,source_sha256={n:p.sha(HERE/n) for n in ('run_compact_graph.py','test_compact_plan_cpu.py','PROTOCOL_V1.md')},
             primitive_sha256=p.sha(PRIMITIVE/'native_dense_bn_v2.py'),shared_math_sha256=p.sha(PRIMITIVE/'compact_current_mlp.py'),
             tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),measurements=MEASUREMENTS,
             planned_forwards=122,quality_endpoints=120,expected_FC_rows_each=67021520,expected_BN_calls=9516,
             full_model_forwards=0,GPU_used=False,GT_read=False,quality_read=False,
             wall_seconds=time.monotonic()-start)
    assert not torch.cuda.is_initialized()
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
    raise SystemExit(0 if result.wasSuccessful() else 1)
