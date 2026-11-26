"""R22 bounded-plan and real original-MLP integration checks; CPU only."""
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
HELPER=HERE.parents[1]/'tail_fine_feasibility_round21/implementation/run_tail_fine_domain.py'
sys.path[:0]=[str(ROOT/'research/flow_story_20261011/preflight/cpu_numerics/deps'),str(SOURCE)]
import torch
from spikingjelly.activation_based import functional
from models.STSwinNet_SNN.Spiking_swin_transformer3D import MS_Spiking_Mlp
import run_representation as p

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

    def test_native26_plan_and_work(self):
        rows=[dict(target_id=str(i)) for i in range(24)]
        roles={str(i):'fit' if i<16 else 'development' for i in range(24)}
        items=p.plan(rows,roles)
        self.assertEqual(len(items),26)
        self.assertEqual([x['arm'] for x in items[:4]],['native','native_repeat1','native_repeat2','native'])
        self.assertEqual(sum(x['quality_endpoint'] for x in items),24)
        self.assertTrue(all(x['execution']==x['endpoint_arm']=='native' for x in items))
        self.assertEqual([x['target_id'] for x in items if x['quality_endpoint']],list(map(str,range(24))))
        self.assertEqual(26*558000,14508000);self.assertEqual(26*78,2028);self.assertEqual(26*8,208)
        with self.assertRaises(ValueError):p.plan(rows,{**roles,'0':'sealed_confirmation'})
        with self.assertRaises(ValueError):p.plan(rows[:-1]+[rows[0]],roles)

    def test_frozen_helper_and_native_only_entry(self):
        self.assertEqual(p.sha(HELPER),p.NATIVE_HELPER_SHA)
        module=p.load(HELPER,'_r22_test_frozen_native_helper')
        self.assertTrue(callable(module.forward));self.assertTrue(callable(module.source_preflight))
        parsed=ast.parse((HERE/'run_representation.py').read_text())
        calls=[n.func.attr for n in ast.walk(parsed) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('compact_mlp',calls);self.assertNotIn('expanded_mlp',calls)
        self.assertNotIn('load_frozen_primitive',calls)
        self.assertEqual(calls.count('source_preflight'),1)
        self.assertEqual(calls.count('forward'),1)

    def test_bytewise_replay_including_signed_zero(self):
        caps={k:torch.zeros(2) for k in ('u','branch','sn1','sn2')}
        a=([torch.zeros(2) for _ in range(4)],caps,{})
        b=([x.clone() for x in a[0]],{k:v.clone() for k,v in caps.items()},{})
        self.assertTrue(p.replay(a,b,torch)['pass'])
        b[1]['u'][0]=-0.;self.assertFalse(p.replay(a,b,torch)['pass']);b[1]['u'][0]=0.
        b[0][3][0]=torch.nextafter(torch.tensor(0.),torch.tensor(1.))
        self.assertFalse(p.replay(a,b,torch)['pass'])
        del b[1]['sn2']
        with self.assertRaises(ValueError):p.replay(a,b,torch)

    def test_actual_native_helper_capture_without_primitive(self):
        helper=p.load(HELPER,'_r22_test_actual_native_forward')
        model,mlp,u=micro_model();targets={p.TARGET:mlp};shapes={p.TARGET:tuple(u.shape)}
        before={k:helper.tensor_sha(v) for k,v in model.state_dict().items()};outs=[]
        for i,arm in enumerate(('native','native_repeat1','native_repeat2')):
            item=dict(forward_id=i+1,target_id='synthetic',selector_role='fit',arm=arm,
                      execution='native',quality_endpoint=i==0,endpoint_arm='native')
            # The actual frozen helper must execute its native branch with no primitive object.
            out=helper.forward(model,functional,targets,shapes,u,item,None,'NATIVE_DENSE_CURRENT',torch)
            outs.append(out)
            self.assertEqual(out[2]['target_execution'],{})
            self.assertEqual(sum(out[2]['native_BN_calls'].values()),78)
            self.assertEqual(out[2]['all12_observed_fc_rows'][p.TARGET],{'fc1_rows':12000,'fc2_rows':12000})
            self.assertEqual(set(out[1]),{'u','branch','sn1','sn2'})
            self.assertTrue(torch.equal(out[1]['u'],u))
            self.assertTrue(torch.equal(out[1]['u'].double().mean(0),u.double().mean(0)))
            self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))
        self.assertTrue(p.replay(outs[1],outs[0],torch)['pass']);self.assertTrue(p.replay(outs[2],outs[0],torch)['pass'])
        self.assertEqual(before,{k:helper.tensor_sha(v) for k,v in model.state_dict().items()})
        MEASUREMENTS.append(dict(real_original_MLP_PSN=True,synthetic_78BN_fixture=True,local_native_micro_forwards=3,
                                 full_model_forwards=0,primitive_calls=0,target_actual_FC_rows=[12000]*3,
                                 full_u_and_CPU_time_mean_equal=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);a=parser.parse_args();assert not a.output.exists()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    out=dict(schema='flow_r22_representation_cpu_v1',status='PASS' if result.wasSuccessful() else 'FAIL',
             graph_id=p.GRAPH_ID,source_sha256={n:p.sha(HERE/n) for n in ('run_representation.py','test_representation_cpu.py','PROTOCOL_V1.md')},
             native_helper_sha256=p.sha(HELPER),roles_sha256=p.ROLE_SHA,tests_run=result.testsRun,
             failures=len(result.failures),errors=len(result.errors),measurements=MEASUREMENTS,
             planned_forwards=26,quality_endpoints=24,expected_FC_rows_each=14508000,expected_BN_calls=2028,
             full_model_forwards=0,GPU_used=False,GT_read=False,quality_read=False,wall_seconds=time.monotonic()-start)
    assert not torch.cuda.is_initialized()
    a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
    raise SystemExit(0 if result.wasSuccessful() else 1)

