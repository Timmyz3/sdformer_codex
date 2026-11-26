#!/usr/bin/env python3
"""Small independent resource/geometry/source checks; no full model or GPU."""
import dataclasses, importlib.util, json, os, sys, time, unittest
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import unified_service as s
DETAILS=[]

class Tests(unittest.TestCase):
 def test_constructed_pipeline_and_capacity_rejection(self):
  m=dataclasses.replace(s.Machine(),dma_startup=0)
  jobs=[s.make_job(100,64,64,0,m,load_bytes=64,store_bytes=64,input_slot_bytes=64,output_slot_bytes=64) for _ in range(3)]
  r=s.pipeline(jobs,m,trace=True)
  self.assertEqual(r['ticks'],302);self.assertEqual(r['PE_arithmetic_ticks'],300);self.assertEqual(r['DMA_bytes'],384)
  comps=[v for v in r['trace'] if v['kind']=='compute'];self.assertEqual([(x['start'],x['end']) for x in comps],[(1,101),(101,201),(201,301)])
  bad=dict(jobs[0],input_slot_bytes=m.io_slot_bytes+1)
  with self.assertRaises(ValueError):s.pipeline([bad],m)
  with self.assertRaises(ValueError):s.pipeline(jobs,dataclasses.replace(m,weight_slots=1))
  bad=dict(jobs[0],ticks=0)
  with self.assertRaises(ValueError):s.pipeline([bad],m)
  DETAILS.append(dict(test='actual_abstract_calendar',expected_ticks=302,actual=r))

 def test_weight_pingpong_and_stats_barrier(self):
  m=dataclasses.replace(s.Machine(),dma_startup=0)
  jobs=[]
  for group in range(3):
   for j in range(2):jobs.append(s.make_job(100,64,64,0,m,load_bytes=64,store_bytes=64,input_slot_bytes=64,output_slot_bytes=64,weight_group=group,weight_load_bytes=64 if j==0 else 0,weight_slot_bytes=64))
  r=s.pipeline(jobs,m,trace=True)
  comps={x['job']:x for x in r['trace'] if x['kind']=='compute'}
  loads={x['job']:x for x in r['trace'] if x['kind']=='load'}
  self.assertGreaterEqual(loads[4]['start'],comps[1]['end'])
  self.assertLess(loads[2]['start'],comps[1]['end']) # legal prefetch into other paid slot
  bn=s.bn_service(dict(elements=256,channels=16),m,trace=True)
  self.assertEqual(bn['global_barriers'],2);self.assertEqual([p['name'] for p in bn['phases']],['variance','normalization'])
  self.assertTrue(bn['Z_retained']);self.assertEqual(bn['DMA_bytes'],1152) # output +16gamma/16beta
  with self.assertRaises(ValueError):s.bn_service(dict(elements=100000,channels=100000),m)

 def test_inventory_and_independent_dot_ceil_scope(self):
  inv=s.inventory();self.assertEqual(len(inv['affines']),83);self.assertEqual(len(inv['BN']),78)
  self.assertEqual(sum(x['elements'] for x in inv['BN']),1414344960)
  self.assertEqual(len(inv['PSN']),105);self.assertEqual(sum(not x['diagnostic_only'] for x in inv['PSN']),93)
  totals={scope:dict(parallel=0,serial8=0,serial6=0) for scope in ('MLP12','patch4')}
  for op in inv['affines']:
   if op['scope'] not in totals:continue
   d=op['row_domains'][0];r,k,n=d['rows'],d['K'],op['Cout'];full=d['complete_rows'];tail=r-full
   t=totals[op['scope']]
   # No production helper or model schedule; fixed integer loops/ceil counts.
   t['parallel']+=r*((k+15)//16)*((n+15)//16)
   t['serial8']+=r*((k+127)//128)*((n+15)//16)*8
   t['serial6']+=(full*6+tail*8)*((k+127)//128)*((n+15)//16)
  self.assertEqual(totals['MLP12'],dict(parallel=663552000,serial8=700416000,serial6=562100400))
  self.assertEqual(totals['patch4'],dict(parallel=995328000,serial8=1032192000,serial6=784526400))
  DETAILS.append(dict(test='independent_root_integer_ceil_crosscheck',totals=totals))

 def test_all_source_PSN_weight_contracts(self):
  import torch
  p=s.REPO/'research/flow_story_20261016/baseline_checkpoint_verification/source_snapshot/checkpoint_step_000271482.pt'
  self.assertEqual(s.sha(p),'1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977')
  state=torch.load(p,map_location='cpu',weights_only=True)['model_state_dict'];inv=s.inventory()
  expected={x['name']+'.spiking_neuron.weight':x for x in inv['PSN']}
  actual={k:v for k,v in state.items() if k.endswith('.spiking_neuron.weight')}
  self.assertEqual(set(actual),set(expected))
  for n,v in actual.items():self.assertEqual(list(v.shape),[expected[n]['T']]*2)
  DETAILS.append(dict(test='true_checkpoint_PSN_shapes',count=len(actual),checkpoint_sha256=s.sha(p)))

 def test_real_native_four_residual_binary_operands(self):
  import torch
  sys.path[:0]=[str(s.REPO/'research/flow_story_20261011/preflight/cpu_numerics/deps'),str(s.REPO/'SDformer/third_party/SDformerFlow')]
  from spikingjelly.activation_based import functional
  from models.STSwinNet_SNN.Spiking_modules import MS_ResBlock
  torch.manual_seed(27019)
  kw=dict(num_steps=10,spike_norm='BN',neuron_type='psn',v_th=.1,v_reset=None,surrogate_fun='surrogate.ATan()',tau=2.,detach_reset=True)
  blocks=torch.nn.Sequential(MS_ResBlock(8,8,**kw),MS_ResBlock(8,8,**kw)).eval().requires_grad_(False)
  functional.set_step_mode(blocks,'m');calls=[];handles=[]
  for mod in blocks.modules():
   if isinstance(mod,torch.nn.modules.batchnorm._BatchNorm):mod.track_running_stats=False;mod.running_mean=mod.running_var=mod.num_batches_tracked=None
  for i,b in enumerate(blocks):
   for j in (1,2):
    def capture(module,inputs,name=f'res{i}.conv{j}'):
     x=inputs[0];calls.append(dict(name=name,shape=list(x.shape),binary=bool(((x==0)|(x==1)).all()),values=sorted(torch.unique(x).tolist())))
    handles.append(getattr(b,'conv'+str(j))[0].register_forward_pre_hook(capture))
  try:
   with torch.no_grad():y=blocks(torch.randn(10,1,8,11,14))
   self.assertTrue(torch.isfinite(y).all());self.assertEqual(len(calls),4)
   self.assertTrue(all(x['binary'] and x['shape']==[10,1,8,11,14] for x in calls))
  finally:
   for h in handles:h.remove()
  DETAILS.append(dict(test='native_residual_binary_domain',actual_block_forwards=2,full_model_forwards=0,calls=calls))

 def test_parallel_is_stronger_than_serial_for_real_shape(self):
  m=s.Machine();self.assertLess(s.dot_ticks(192000,96,384,'int8_parallel',8,m),s.dot_ticks(192000,96,384,'bitserial',8,m))
  self.assertLess(s.dot_ticks(12000,384,1536,'bitserial',6,m),s.dot_ticks(12000,384,1536,'int8_parallel',8,m))

 def test_scale_parameters_resident_and_smallN_packing(self):
  # A real stage3 FC1 weight stripe previously fit weights alone but not its
  # stored scales. The finite-SRAM assertion must include both.
  m=s.Machine();op=next(v for v in s.inventory()['affines'] if v['name'].endswith('layers.3.swin_blocks.0.mlp.fc1'))
  r=s.affine_service(op,m,'bitserial','Q6')
  self.assertEqual(r['rows'],3000);self.assertEqual(r['logical_row_planes'],3000*8-1620*2)
  mp=dataclasses.replace(m,small_output_row_packing=True)
  self.assertEqual(s.dot_ticks(128,2,2,'native',8,mp),20)
  self.assertEqual(s.dot_ticks(128,2,2,'native',8,m),132)
  # Same port limits still apply to the faster row-packed arithmetic.
  p=s.psn_service(dict(T=2,features=128,elements=256),mp)
  self.assertGreaterEqual(p['PE_reserved_ticks'],s.ceil(256*4,m.input_port_bytes_tick))

if __name__=='__main__':
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1)
 out=Path(sys.argv[1]);assert not out.exists();start=time.monotonic()
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
 data=dict(schema='flow_r27_abstract_machine_cpu_checks_v1',status='PASS' if result.wasSuccessful() else 'FAIL',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),seconds=time.monotonic()-start,source_sha256={str(HERE/n):s.sha(HERE/n) for n in ('unified_service.py','test_unified_service.py')},GPU_used=False,GT_values_read=False,full_model_forwards=0,details=DETAILS)
 out.write_text(json.dumps(data,indent=2)+'\n');sys.exit(0 if result.wasSuccessful() else 1)
