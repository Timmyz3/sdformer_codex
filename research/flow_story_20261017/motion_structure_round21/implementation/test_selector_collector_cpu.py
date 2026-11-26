"""R21 bounded-plan and real original-MLP integration checks; CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import argparse
import copy
import csv
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
import native_dense_bn_v2 as primitive
import run_selector_collector as p

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
    def test_actual_role_metadata_and_no_confirmation_file_access(self):
        rolepath=HERE.parents[1]/'action_domain_round20/independent_review/SELECTOR_ROLES_V1.json'
        self.assertEqual(p.sha(rolepath),p.ROLE_SHA)
        selector=p.read(rolepath)
        canonical_path=Path(selector['source_sha256']['canonical']['path'])
        deployment_path=Path(selector['source_sha256']['deployment']['path'])
        self.assertEqual(p.sha(canonical_path),selector['source_sha256']['canonical']['sha256'])
        self.assertEqual(p.sha(deployment_path),selector['source_sha256']['deployment']['sha256'])
        canonical=p.read(canonical_path)['records']
        with deployment_path.open(newline='') as f:
            hashes={x['relative_path']:{'sha256':x['sha256'],'bytes':int(x['bytes'])} for x in csv.DictReader(f)}
        native=p.load(ROOT/'research/flow_story_20261012/baseline/native_data_acceptance.py','_r21_test_native_metadata')
        selected,roles=p.selector_members(selector,canonical,hashes,native)
        self.assertEqual(len(selected),24)
        self.assertEqual([roles[r['target_id']] for r in selected],['fit']*16+['development']*8)
        sealed={r['target_id'] for r in selector['groups']['sealed_confirmation']}
        self.assertFalse(sealed & set(roles))
        bad=copy.deepcopy(selector);bad['groups']['fit'][0]=bad['groups']['sealed_confirmation'][0]
        with self.assertRaisesRegex(ValueError,'disjoint'):p.selector_members(bad,canonical,hashes,native)
        bad=copy.deepcopy(selector);bad['records'][0]['accepted_array_metadata'][0]['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'acceptance'):p.selector_members(bad,canonical,hashes,native)
        visited=[]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);synthetic={}
            for row in selected:
                for kind,relative in native.target_files(row).items():
                    path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
                    path.write_bytes(('synthetic fixture '+row['target_id']+' '+kind).encode())
                    synthetic[relative]={'bytes':path.stat().st_size,'sha256':p.sha(path)}
            allowed=set(synthetic)
            def guarded_path(root,relative):
                self.assertIn(relative,allowed,'A sealed/unselected array path must never be visited')
                visited.append(relative);return native.checked_path(root,relative)
            proxy=types.SimpleNamespace(target_files=native.target_files,checked_path=guarded_path)
            with mock.patch('numpy.load',side_effect=AssertionError('No array loads in metadata/SHA preflight')):
                evidence=p.verify_selected_files(selected,proxy,root,synthetic,roles)
            self.assertEqual(len(evidence),24);self.assertEqual(len(visited),72);self.assertEqual(set(visited),allowed)
            self.assertTrue(all(set(r['input_evidence'])=={'events','gt','mask'} for r in evidence))
            first=next(iter(allowed));(root/first).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'size/SHA'):p.verify_selected_files(selected,proxy,root,synthetic,roles)
        MEASUREMENTS.append({'actual_role_manifest_sha256':p.ROLE_SHA,'selected_identity_count':24,
                             'real_selected_data_arrays_read':0,'synthetic_allowed_file_hashes':72,'sealed_array_path_visits':0})

    def test_single_background_contract(self):
        for bg in ('CC','T8'):
            inp={'background':bg};freeze={'background':bg};admission={'permitted_backgrounds':[bg]}
            p.background_contract(bg,inp,freeze,admission)
            with self.assertRaises(ValueError):p.background_contract(bg,inp,{'background':'CC' if bg=='T8' else 'T8'},admission)
            with self.assertRaises(ValueError):p.background_contract(bg,inp,freeze,{'permitted_backgrounds':['CC','T8']})

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
        rows=[dict(target_id=str(i)) for i in range(24)]
        roles={str(i):'fit' if i<16 else 'development' for i in range(24)}
        for background,total in (('CC',198874960),('T8',199490400)):
            items=p.plan(rows,'selector362',background,roles)
            self.assertEqual(len(items),362)
            self.assertEqual([v['arm'] for v in items[:6]],['native','FF',background,background+'_repeat1',background+'_repeat2','fine_0_0'])
            self.assertEqual(sum(v['quality_endpoint'] for v in items),360)
            self.assertEqual(sum(v['execution']=='native' for v in items),24)
            self.assertEqual(sum(v['execution']=='compact' for v in items),338)
            self.assertEqual(sum(p.expected_rows(v) for v in items),total)
            self.assertEqual(362*78,28236)
            payloads=[p.payload_names(v,'0') for v in items]
            self.assertEqual(sum(sum(k.startswith('head') for k in x) for x in payloads),515)
            self.assertEqual(sum(sum(not k.startswith('head') for k in x) for x in payloads),258)
            self.assertEqual(sum(map(len,payloads)),773)
            for tid in map(str,range(24)):
                self.assertEqual([x['endpoint_arm'] for x in items if x['target_id']==tid and x['quality_endpoint']],list(p.arms(background)))
            for arm in p.arms(background)[1:]:
                mask=p.mask_for(arm,torch,background);reps=0;covered=0
                for y in range(0,30,9):
                    for x in range(0,40,9):
                        hh=min(9,30-y);ww=min(9,40-x)
                        step=1 if bool(mask[y//9,x//9]) else 3
                        reps+=((hh+step-1)//step)*((ww+step-1)//step);covered+=hh*ww
                self.assertEqual(covered,1200)
                self.assertEqual(546000+10*reps,p.expected_rows(dict(arm=arm,execution='compact',background=background)))
                if arm.startswith('fine_'):
                    self.assertEqual(int(mask[:3,:4].sum()),1)
                    self.assertEqual(int(mask.sum()),1 if background=='CC' else 9)
            for bad in ('F12','fine_3_0','fine_0_4','native', 'T8' if background=='CC' else 'CC'):
                with self.assertRaises(ValueError):p.mask_for(bad,torch,background)
            self.assertIn(background,p.graph_id(background))
        with self.assertRaises(ValueError):p.plan(rows,'selector362','T8',{**roles,'0':'sealed_confirmation'})
        with self.assertRaises(ValueError):p.plan(rows[:-1],'selector362','T8',roles)

    def test_original_primitive_pins_and_no_expanded_call(self):
        self.assertEqual(p.sha(PRIMITIVE/'native_dense_bn_v2.py'),p.PRIMITIVE_SHA)
        self.assertEqual(p.sha(PRIMITIVE/'compact_current_mlp.py'),p.SHARED_MATH_SHA)
        parsed=ast.parse((HERE/'run_selector_collector.py').read_text())
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
        for background in ('CC','T8'):
            model,mlp,u=micro_model();targets={p.TARGET:mlp};shapes={p.TARGET:tuple(u.shape)}
            before={k:p.tensor_sha(v) for k,v in model.state_dict().items()}
            outputs={};references={};checks=[]
            for i,arm in enumerate(('native','FF',background,background+'_repeat1',background+'_repeat2','fine_2_3')):
                item=dict(forward_id=i+1,target_id='synthetic',arm=arm,background=background,selector_role='fit',
                          execution='native' if arm=='native' else 'compact',quality_endpoint='_repeat' not in arm,
                          endpoint_arm=background if '_repeat' in arm else arm)
                out=p.forward(model,functional,targets,shapes,u,item,primitive,'NATIVE_DENSE_CURRENT',torch)
                outputs[arm]=out
                check=p.numerical_check(item,out,outputs.get('native'),references,'synthetic',torch)
                if check:checks.append(check)
                self.assertEqual(sum(out[2]['native_BN_calls'].values()),78)
                self.assertEqual(out[2]['all12_observed_fc_rows'][p.TARGET]['fc1_rows'],p.expected_rows(item)-546000)
                self.assertEqual(out[2]['tensor_records']['u']['sha256'],p.tensor_sha(u))
                self.assertTrue(torch.equal(out[1]['u'],u))
                self.assertTrue(torch.equal(out[1]['u'].double().mean(0),u.double().mean(0)))
                self.assertIs(model.get_submodule(p.TARGET),mlp)
                self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
                self.assertTrue(all(h['dtype']=='torch.float32' for h in out[2]['head_records']))
                if arm!='native':
                    audit=out[2]['target_execution']
                    self.assertFalse(audit['native_MLP_shortcut']);self.assertEqual(sum(audit['areas']),1200)
                    for sn in ('sn1','sn2'):
                        self.assertTrue(torch.equal(out[1][sn],primitive.expand(out[1][sn+'_packed'],audit['cells'],30,40)))
            self.assertEqual(len(checks),3);self.assertTrue(all(c['pass'] for c in checks))
            self.assertEqual([c['reference_forward_id'] for c in checks if c['kind'].endswith('exact_replay')],[3,3])
            with self.assertRaisesRegex(ValueError,'Actual first matching'):
                p.numerical_check(dict(outputs[background+'_repeat1'][2],target_id='sealed'),outputs[background+'_repeat1'],outputs['native'],references,'synthetic',torch)
            with self.assertRaisesRegex(ValueError,'arm/frame'):
                p.numerical_check(outputs[background+'_repeat1'][2],outputs[background+'_repeat1'],outputs['native'],{background:outputs['FF']},'synthetic',torch)
            self.assertEqual(before,{k:p.tensor_sha(v) for k,v in model.state_dict().items()})
            MEASUREMENTS.append(dict(background=background,real_original_MLP_PSN=True,synthetic_78BN_fixture=True,
                                     local_micro_forwards=6,full_model_forwards=0,
                                     actual_compact_rows=[outputs[n][2]['all12_observed_fc_rows'][p.TARGET]['fc1_rows'] for n in outputs]))

    def test_exception_restores_target_and_hooks(self):
        model,mlp,u=micro_model()
        def fail(*args):raise RuntimeError('deliberate primitive failure')
        fake=types.SimpleNamespace(compact_mlp=fail)
        item=dict(arm='CC',execution='compact',background='CC')
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            p.forward(model,functional,{p.TARGET:mlp},{p.TARGET:tuple(u.shape)},u,item,fake,'NATIVE_DENSE_CURRENT',torch)
        self.assertIs(model.get_submodule(p.TARGET),mlp)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);a=parser.parse_args()
    assert not a.output.exists()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    out=dict(schema='flow_r21_orchestration_cpu_v1',status='PASS' if result.wasSuccessful() else 'FAIL',
             graph_ids=[p.graph_id(x) for x in ('CC','T8')],source_sha256={n:p.sha(HERE/n) for n in ('run_selector_collector.py','test_selector_collector_cpu.py','PROTOCOL_V1.md')},
             primitive_sha256=p.sha(PRIMITIVE/'native_dense_bn_v2.py'),shared_math_sha256=p.sha(PRIMITIVE/'compact_current_mlp.py'),
             tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),measurements=MEASUREMENTS,
             planned_forwards=362,quality_endpoints=360,expected_FC_rows_each=p.TOTAL_ROWS,expected_BN_calls=28236,
             full_model_forwards=0,GPU_used=False,GT_read=False,quality_read=False,
             wall_seconds=time.monotonic()-start)
    assert not torch.cuda.is_initialized()
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
    raise SystemExit(0 if result.wasSuccessful() else 1)
