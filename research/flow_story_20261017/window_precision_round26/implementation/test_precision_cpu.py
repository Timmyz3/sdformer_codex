"""New integer graph checks, real original MSMLP with synthetic 78BN wrapper; no GPU/GT."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
import argparse,ast,hashlib,json,sys,time,types,unittest
from pathlib import Path
from unittest import mock
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'research/flow_story_20261011/preflight/cpu_numerics/deps'),str(ROOT/'SDformer/third_party/SDformerFlow')]
import torch
from spikingjelly.activation_based import functional
from models.STSwinNet_SNN.Spiking_swin_transformer3D import MS_Spiking_Mlp
import bitplane_linear as core
import run_precision as p
MEASUREMENTS=[]


def micro_model():
    torch.manual_seed(17)
    mlp=MS_Spiking_Mlp(8,hidden_features=16,norm_layer='BN',drop=0.,num_steps=10,
        spike_norm='BN',neuron_type='psn',v_th=.1,v_reset=None,surrogate_fun='surrogate.ATan()',tau=2.,detach_reset=True)
    functional.set_step_mode(mlp,'m')
    for wrapper in (mlp.bn1,mlp.bn2):
        bn=wrapper.norm_layer;bn.track_running_stats=False
        bn.running_mean=bn.running_var=bn.num_batches_tracked=None
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();node=self;names=p.TARGET.split('.')
            for name in names[:-1]:child=torch.nn.Module();node.add_module(name,child);node=child
            node.add_module(names[-1],mlp)
            self.external=torch.nn.ModuleList([torch.nn.BatchNorm1d(1,track_running_stats=False) for _ in range(76)])
        def forward(self,u):
            for bn in self.external:bn(torch.tensor([[0.],[1.]],dtype=u.dtype))
            branch=self.get_submodule(p.TARGET)(u)
            return {'flow':[branch.mean(-1) for _ in range(4)]}
    return Model().eval().requires_grad_(False),mlp,torch.randn(10,1,30,40,8)


class Checks(unittest.TestCase):
    def test_01_plan_raw_budget_and_work(self):
        rows=[{'target_id':str(i)} for i in range(8)];plan=p.plan(rows,'precision42')
        self.assertEqual(len(plan),42);self.assertEqual(sum(x['quality_endpoint'] for x in plan),40)
        self.assertEqual([x['arm'] for x in plan[:7]],['native','Q8ref','Q8serial','Q6','Q6_repeat1','Q6_repeat2','Q4'])
        payload=work=integercalls=0
        for item in plan:
            arm=item['endpoint_arm']
            payload+=4+(4 if arm=='native' else 6) if p.keep_captures(item,'0') else 4
            if arm in ('Q8serial','Q6','Q4'):
                drop={'Q8serial':0,'Q6':2,'Q4':4}[arm]
                work+=(8-drop)*12000+drop*2280;integercalls+=8
            elif arm=='Q8ref':integercalls+=1
        self.assertEqual((payload,work,integercalls),(314,1990560,216))
        counts=p.expected_counts();self.assertEqual(counts['expected_native_BN_calls'],42*78)
        self.assertEqual(counts['expected_original_FC_rows_each'],42*546000+8*12000)
        self.assertEqual(counts['expected_full_grid_logical_FC_rows_each'],42*558000)
        with self.assertRaises(ValueError):p.plan(rows,'domain122')

    def test_02_no_compact_or_GT_forward_and_source_pins(self):
        self.assertEqual(p.sha(HERE/'bitplane_linear.py'),p.PRIMITIVE_SHA)
        helper=HERE.parents[1]/'two_slot_action_round24/implementation/run_two_slot_domain.py'
        self.assertEqual(p.sha(helper),p.DATA_HELPER_SHA)
        tree=ast.parse((HERE/'run_precision.py').read_text())
        attrs=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
        self.assertNotIn('compact_mlp',attrs);self.assertNotIn('expanded_mlp',attrs);self.assertNotIn('prepare',attrs)
        self.assertIn('source_preflight',attrs)
        loads=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load' and isinstance(n.func.value,ast.Name) and n.func.value.id=='np']
        self.assertEqual(len(loads),1);self.assertEqual(ast.unparse(loads[0].args[0]),'path')

    def test_03_true_integer_truncation_full_tail_and_scale(self):
        torch.manual_seed(101);linear=torch.nn.Linear(8,16,bias=True).eval()
        with torch.no_grad():
            linear.weight[0].zero_();linear.weight[1]=torch.tensor([-127.,127.,-126.,126.,-1.,1.,0.,3.])
        spec=core.place_spec(core.quantize_source(linear),'cpu')
        u=torch.randint(0,2,(10,1,30,40,8)).float()
        tailmask=torch.zeros((30,40),dtype=torch.bool);tailmask[27:]=True;tailmask[:,36:]=True
        self.assertEqual(int(tailmask.sum()),228)
        flat=u.reshape(-1,8).to(torch.int64);q=spec['q8'].to(torch.int64)
        outputs={}
        for mode,drop in [('Q8ref',0),('Q8serial',0),('Q6',2),('Q4',4)]:
            module=core.QuantLinear(linear,spec,mode);out=module(u);outputs[mode]=out
            low=(q>>drop)<<drop
            expected=(flat@low.T).reshape(10,1,30,40,16)
            dense=(flat@q.T).reshape_as(expected)
            expected[:,:,tailmask,:]=dense[:,:,tailmask,:]
            self.assertTrue(torch.equal(module.last_integer,expected.to(torch.int32)))
            self.assertTrue(torch.equal(out,expected.float()*spec['scale']+linear.bias))
            self.assertEqual(module.last_audit['original_linear_forward_calls'],0)
            self.assertEqual(module.last_audit['bias_applications_per_output_element'],1)
        self.assertTrue(torch.equal(outputs['Q8ref'].view(torch.uint8),outputs['Q8serial'].view(torch.uint8)))
        self.assertEqual(int(((q[1,0]>>2)<<2)),-128)
        self.assertEqual(float(spec['scale'][0]),0.)
        self.assertTrue(torch.equal(outputs['Q6'][:,:,tailmask,:],outputs['Q8serial'][:,:,tailmask,:]))
        MEASUREMENTS.append({'synthetic_affine_modes':4,'full_grid_rows':12000,'tail_rows':2280,'independent_int64_reference':True})

    def test_04_invalid_spikes_and_exact_bytes(self):
        layer=torch.nn.Linear(8,16,bias=False);spec=core.place_spec(core.quantize_source(layer),'cpu')
        for value in (.5,float('nan'),float('inf')):
            u=torch.zeros(10,1,30,40,8);u[0,0,0,0,0]=value
            with self.assertRaises(ValueError):core.QuantLinear(layer,spec,'Q6')(u)
        ac={k:torch.zeros(2,dtype=torch.int32 if 'integer' in k else torch.float32) for k in ('u','branch','sn1','sn2','fc1_integer','fc2_integer')}
        a=([torch.zeros(2) for _ in range(4)],ac,{})
        b=([x.clone() for x in a[0]],{k:x.clone() for k,x in ac.items()},{})
        self.assertTrue(p.exact_pair(a,b,torch)['pass']);b[1]['branch'][0]=-0.
        self.assertFalse(p.exact_pair(a,b,torch)['pass']);b[1]['branch'][0]=0.;b[1]['fc1_integer'][0]=1
        self.assertFalse(p.exact_pair(a,b,torch)['pass'])

    def test_05_real_native_MLP_all78BN_q8_exact_and_closure(self):
        model,mlp,u=micro_model();targets={p.TARGET:mlp};shapes={p.TARGET:tuple(u.shape)}
        original_fc=(mlp.fc1,mlp.fc2);before={k:p.tensor_sha(x) for k,x in model.state_dict().items()}
        specs={fc:core.place_spec(core.quantize_source(getattr(mlp,fc)),'cpu') for fc in ('fc1','fc2')}
        with torch.inference_mode():native_direct=model(u)
        outputs={}
        for i,arm in enumerate(('native','Q8ref','Q8serial','Q6','Q6_repeat1','Q4')):
            mode='Q6' if arm.startswith('Q6_repeat') else arm
            item=dict(forward_id=i+1,target_id='synthetic',arm=arm,endpoint_arm=mode,execution='native' if mode=='native' else 'integer_quantized',quality_endpoint=arm!='Q6_repeat1')
            out=p.forward(model,functional,targets,shapes,u,item,core,specs,torch);outputs[arm]=out
            record=out[2];self.assertEqual(sum(record['native_BN_calls'].values()),78)
            self.assertEqual(record['all12_original_fc_observations'][p.TARGET]['fc1_calls'],int(mode=='native'))
            self.assertEqual(record['all12_full_grid_logical_fc_rows'][p.TARGET]['fc1_rows'],12000)
            for bn,c in (('bn1',16),('bn2',8)):
                self.assertEqual(record['native_BN_inputs'][p.TARGET+'.'+bn+'.norm_layer'][0]['shape'],[10,1,c,30,40])
            if mode!='native':self.assertEqual(record['target_active_quantized_rows'],{'fc1':12000,'fc2':12000})
            self.assertIs(mlp.fc1,original_fc[0]);self.assertIs(mlp.fc2,original_fc[1])
            self.assertEqual(before,{k:p.tensor_sha(x) for k,x in model.state_dict().items()})
        for actual,ref in zip(outputs['native'][0],native_direct['flow']):self.assertTrue(torch.equal(actual,ref))
        self.assertTrue(p.exact_pair(outputs['Q8serial'],outputs['Q8ref'],torch)['pass'])
        self.assertTrue(p.exact_pair(outputs['Q6_repeat1'],outputs['Q6'],torch)['pass'])
        MEASUREMENTS.append({'real_original_MLP_PSN':True,'synthetic_78BN_fixture':True,'local_micro_forwards':7,
                             'full_model_forwards':0,'all_native_BN_calls_per_micro':78,'Q8_all10_arrays_exact':True,'Q6_all10_arrays_exact':True})

    def test_06_exception_restores_original_FC_and_hooks(self):
        model,mlp,u=micro_model();original=(mlp.fc1,mlp.fc2)
        specs={fc:core.place_spec(core.quantize_source(getattr(mlp,fc)),'cpu') for fc in ('fc1','fc2')}
        item={'endpoint_arm':'Q6','arm':'Q6'}
        with mock.patch.object(core,'integer_mm',side_effect=RuntimeError('deliberate integer backend failure')):
            with self.assertRaisesRegex(RuntimeError,'deliberate'):
                p.forward(model,functional,{p.TARGET:mlp},{p.TARGET:tuple(u.shape)},u,item,core,specs,torch)
        self.assertIs(mlp.fc1,original[0]);self.assertIs(mlp.fc2,original[1])
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args();assert not args.output.exists()
    torch.set_num_threads(2);torch.set_num_interop_threads(1);start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    receipt=dict(schema='flow_r26_precision_cpu_v1',status='PASS' if result.wasSuccessful() else 'FAIL',
      graph_id=p.GRAPH_ID,source_sha256={n:p.sha(HERE/n) for n in ('bitplane_linear.py','run_precision.py','test_precision_cpu.py','PROTOCOL_V1.md')},
      tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),measurements=MEASUREMENTS,
      expected_counts=p.expected_counts(),GPU_used=False,full_model_forwards=0,GT_read=False,quality_read=False,
      torch=str(torch.__version__),wall_seconds=time.monotonic()-start)
    assert not torch.cuda.is_initialized();args.output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
    raise SystemExit(0 if result.wasSuccessful() else 1)
