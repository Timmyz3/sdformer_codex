"""R24 bounded-plan and real original-MLP integration checks; CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import argparse
import tempfile
from unittest import mock
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
import run_two_slot_domain as p
primitive = p.load_frozen_primitive(HERE/'two_slot_mlp.py', PRIMITIVE/'native_dense_bn_v2.py')

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
    def test_source_preflight_does_not_stat_the_dataset(self):
        calls=[]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';source.mkdir();sources={}
            for i in range(63):
                path=source/('s%d.py'%i);path.write_text('source fixture\n');sources[path.name]=p.sha(path)
            cp=root/'checkpoint.pt';cp.write_bytes(b'synthetic source-contract checkpoint fixture')
            p.write(cp.with_suffix('.json'),{'checkpoint_sha256':p.sha(cp),'checkpoint_file':cp.name})
            p.write(root/'run_contract.json',{'contract':'fixture'})
            def bundle(*args,**kwargs):
                self.assertIs(kwargs['check_current_files'],False)
                calls.append('accepted_metadata_only');return [],{}, {},{'current_file_stat_checked':False}
            def checked_path(parent,relative):
                self.assertEqual(Path(parent),source,'Dataset path access prohibited in source preflight')
                calls.append('source_only');return source/relative
            native=types.SimpleNamespace(validate_native_bundle=bundle,checked_path=checked_path,
                                         pinned_json=lambda *args:sources)
            trainer=types.SimpleNamespace(CORE_SHA='core',load_core=lambda:types.SimpleNamespace(SOURCE_MANIFEST_SHA='source'))
            def contract(run,side,binding,sources_in,trainer_in,train,hashes):
                self.assertEqual(binding,{'entrypoint_sha256':'entry','adapter_sha256':'adapter',
                    'training_core_sha256':'core','checkpoint_schema':'schema','checkpoint_creator':'creator'})
                self.assertEqual(sources_in,sources);calls.append('original_contract');return {'validated':True}
            base=types.SimpleNamespace(SCHEMA='schema',CREATOR='creator',validate_contract=contract,
                support_modules=lambda prior:(native,trainer,object(),{'baseline/train_native_baseline.py':'entry','baseline/native_data_acceptance.py':'adapter'}))
            b=types.SimpleNamespace(geometry='full9',prior_root=root,data_root=root/'never-touch-dataset',canonical=root/'metadata',
                acceptance=root/'metadata',acceptance_sha256='acceptance',archive_audit=root/'metadata',
                source_manifest=root/'metadata',source_root=source,checkpoint=cp,checkpoint_sha256=p.sha(cp))
            with mock.patch('numpy.load',side_effect=AssertionError('No data loads')):
                context=p.source_preflight(b,base)
            self.assertEqual(context[5],{'validated':True});self.assertEqual(context[9]['sealed_confirmation_array_paths_visited'],0)
            self.assertEqual(calls.count('source_only'),63);self.assertIn('original_contract',calls)


    def test_plan_and_independent_geometry(self):
        rows=[dict(target_id=str(i)) for i in range(8)]
        items=p.plan(rows,'domain122')
        self.assertEqual(len(items),122)
        self.assertEqual([v['arm'] for v in items[:6]],['native','FF','T2','T2_repeat1','T2_repeat2','fine_0_0'])
        self.assertEqual(sum(v['quality_endpoint'] for v in items),120)
        self.assertEqual(sum(v['execution']=='native' for v in items),8)
        self.assertEqual(sum(v['execution']=='compact' for v in items),114)
        self.assertEqual(sum(p.expected_rows(v) for v in items),67335120)
        self.assertEqual(122*78,9516)
        # Independent ceil cover for true full/tail tiles, without calling pack.
        for arm in p.ARMS[1:]:
            mask=p.mask_for(arm,torch);reps=0;covered=0
            for y in range(0,30,9):
                for x in range(0,40,9):
                    hh=min(9,30-y);ww=min(9,40-x)
                    reps += hh*ww if bool(mask[y//9,x//9]) else 18
                    covered += hh*ww
            self.assertEqual(covered,1200)
            item=dict(arm=arm,execution='compact')
            self.assertEqual(546000+10*reps,p.expected_rows(item))
            if arm.startswith('fine_'):
                self.assertEqual(int(mask.sum()),9);self.assertTrue(mask[3].all());self.assertTrue(mask[:,4].all());self.assertEqual(int(mask[:3,:4].sum()),1)
        self.assertEqual(p.GRAPH_ID,'r24_W271482_all78current_s2b2_fixedH1_two_slot_T2_k1_H8')

    def test_original_math_pins_private_pack_only_and_no_expanded_call(self):
        self.assertEqual(p.sha(HERE/'two_slot_mlp.py'),p.PRIMITIVE_SHA)
        self.assertEqual(p.sha(PRIMITIVE/'native_dense_bn_v2.py'),p.BASE_PRIMITIVE_SHA)
        self.assertEqual(p.sha(PRIMITIVE/'compact_current_mlp.py'),p.SHARED_MATH_SHA)
        parsed=ast.parse((HERE/'run_two_slot_domain.py').read_text())
        attrs=[n.func.attr for n in ast.walk(parsed) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertIn('compact_mlp',attrs);self.assertNotIn('expanded_mlp',attrs)
        actual=p.load_frozen_primitive(HERE/'two_slot_mlp.py',PRIMITIVE/'native_dense_bn_v2.py')
        shared=sys.modules['compact_current_mlp']
        self.assertIs(actual._base_module.pack,shared.pack)
        self.assertIs(actual.expand,shared.expand)
        self.assertIs(actual._native_compact.__code__,actual._base_module.compact_mlp.__code__)
        original=actual._base_module.compact_mlp.__globals__
        private=actual._native_compact.__globals__
        self.assertEqual([key for key in private if private[key] is not original[key]],['pack'])
        self.assertIs(private['pack'],actual.pack_two_slot)
        self.assertIs(original['pack'],shared.pack)
        fake=types.ModuleType('compact_current_mlp');fake.__file__='/wrong/import/compact_current_mlp.py'
        for name in ('validate','pack','expand','expanded_mlp','require'):setattr(fake,name,getattr(shared,name))
        sys.modules['compact_current_mlp']=fake
        try:
            with self.assertRaisesRegex(ValueError,'Actually imported'):
                p.load_frozen_primitive(HERE/'two_slot_mlp.py',PRIMITIVE/'native_dense_bn_v2.py')
        finally:sys.modules['compact_current_mlp']=shared

    def test_real_pack_h1_support_values_all_T_C_and_tail_guard(self):
        u=torch.arange(10*2*30*40*4,dtype=torch.float64).reshape(10,2,30,40,4)
        for arm,reps in (('FF',1200),('T2',444),('fine_1_2',507)):
            packed,areas,cells=primitive.pack_two_slot(u,p.mask_for(arm,torch))
            self.assertEqual(tuple(packed.shape),(10,2,reps,1,4))
            cover=torch.zeros(30,40,dtype=torch.int64)
            for i,(y0,y1,x0,x1,area) in enumerate(cells):
                cover[y0:y1,x0:x1]+=1
                self.assertEqual(area,(y1-y0)*(x1-x0))
                if area>1:
                    self.assertLess(y1,28);self.assertLessEqual(x1,36)
                    self.assertEqual(x1-x0,3)
                    self.assertIn((y1-y0,area),((1,3),(2,6)))
                    self.assertEqual(y0%3,0 if area==3 else 1)
                expected=u[:,:,y0:y1,x0:x1,:].mean((2,3))
                self.assertTrue(torch.equal(packed[:,:,i,0,:],expected))
            self.assertTrue(torch.equal(cover,torch.ones_like(cover)))
            self.assertEqual(int(areas.sum()),1200)
            dense=primitive.expand(packed,cells,30,40)
            self.assertTrue(torch.equal(dense[:,:,27:,:,:],u[:,:,27:,:,:]))
            self.assertTrue(torch.equal(dense[:,:,:,36:,:],u[:,:,:,36:,:]))
            weighted=(packed*areas.view(1,1,-1,1,1)).sum((2,3))
            self.assertTrue(torch.equal(weighted,u.sum((2,3))))
        bad=p.mask_for('T2',torch);bad[3,0]=False
        with self.assertRaisesRegex(ValueError,'tail'):
            primitive.pack_two_slot(u,bad)

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
        for i,arm in enumerate(('native','FF','T2','T2_repeat1','fine_2_3')):
            item=dict(forward_id=i+1,target_id='synthetic',arm=arm,
                      execution='native' if arm=='native' else 'compact',
                      quality_endpoint=arm!='T2_repeat1',endpoint_arm='T2' if arm=='T2_repeat1' else arm)
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
        self.assertTrue(torch.equal(outputs['FF'][1]['branch'],outputs['native'][1]['branch']))
        for name in ('sn1','sn2'):
            self.assertTrue(torch.equal(outputs['FF'][1][name],outputs['native'][1][name]))
        for arm in ('FF','T2','T2_repeat1','fine_2_3'):
            rec=outputs[arm][2]
            for bn in ('bn1','bn2'):
                shape=rec['native_BN_inputs'][p.TARGET+'.'+bn+'.norm_layer'][0]['shape']
                self.assertEqual(shape,[10,1,16 if bn=='bn1' else 4,30,40])
            self.assertEqual(rec['target_execution']['full_domain_BN_elements'],10*1200*(16+4))
            self.assertEqual(rec['target_execution']['private_globals_overrides'],['pack'])
            self.assertFalse(rec['target_execution']['original_globals_mutated'])
        self.assertTrue(p.deterministic_pair(outputs['T2_repeat1'],outputs['T2'],torch)['pass'])
        self.assertEqual(before,{k:p.tensor_sha(v) for k,v in model.state_dict().items()})
        MEASUREMENTS.append(dict(real_original_MLP_PSN=True,synthetic_78BN_fixture=True,
                                 local_micro_forwards=5,full_model_forwards=0,
                                 actual_compact_rows=[outputs[n][2]['all12_observed_fc_rows'][p.TARGET]['fc1_rows'] for n in outputs]))

    def test_exception_restores_target_and_hooks(self):
        model,mlp,u=micro_model()
        def fail(*args):raise RuntimeError('deliberate primitive failure')
        fake=types.SimpleNamespace(compact_mlp=fail)
        item=dict(arm='T2',execution='compact')
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            p.forward(model,functional,{p.TARGET:mlp},{p.TARGET:tuple(u.shape)},u,item,fake,'NATIVE_DENSE_CURRENT',torch)
        self.assertIs(model.get_submodule(p.TARGET),mlp)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);a=parser.parse_args()
    assert not a.output.exists()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    out=dict(schema='flow_r24_two_slot_orchestration_cpu_v1',status='PASS' if result.wasSuccessful() else 'FAIL',
             graph_id=p.GRAPH_ID,source_sha256={n:p.sha(HERE/n) for n in ('run_two_slot_domain.py','two_slot_mlp.py','test_two_slot_cpu.py','PROTOCOL_V1.md')},
             primitive_sha256=p.sha(HERE/'two_slot_mlp.py'),base_primitive_sha256=p.sha(PRIMITIVE/'native_dense_bn_v2.py'),shared_math_sha256=p.sha(PRIMITIVE/'compact_current_mlp.py'),
             tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),measurements=MEASUREMENTS,
             planned_forwards=122,quality_endpoints=120,expected_FC_rows_each=67335120,expected_BN_calls=9516,
             full_model_forwards=0,GPU_used=False,GT_read=False,quality_read=False,
             wall_seconds=time.monotonic()-start)
    assert not torch.cuda.is_initialized()
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
    raise SystemExit(0 if result.wasSuccessful() else 1)
