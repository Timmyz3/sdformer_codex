import copy,unittest,math,types,importlib.util
import numpy as np
import audit_precision as a
M=[dict(target_id='synthetic_'+str(i)) for i in range(8)]
Q={fc:{'q8_sha256':'a'*64,'scale_sha256':'b'*64} for fc in a.SHAPES}
def fc_record(fc,mode):
 cin,cout=a.SHAPES[fc];drop={'Q8ref':0,'Q8serial':0,'Q6':2,'Q4':4}[mode];ref=mode=='Q8ref';calls=[]
 for i in range(1 if ref else 8):
  tail=not ref and i<drop;rows=2280 if tail else 12000
  r={'kind':'integer_dense_reference' if ref else 'binary_weight_plane','bit':None if ref else i,'coefficient':None if ref else -128 if i==7 else 1<<i,'rows':rows,'Cin':cin,'Cout':cout,'input_dtype':'torch.int8','weight_dtype':'torch.int8','output_dtype':'torch.int32','weight_shape':[cin,cout],'explicit_padding_rows':0,'explicit_padding_channels':0,'input_spike_ones':rows}
  if not ref:r['row_scope']='tail_only_Q8' if tail else 'all_spatial_rows'
  calls.append(r)
 return {'mode':mode,'actual_input_shape':[10,1,30,40,cin],'input_rows':12000,'full_tile_rows':9720,'tail_rows':2280,'original_linear_forward_calls':0,'int_mm_call_count':len(calls),'int_mm_calls':calls,'row_plane_work':0 if ref else 96000-drop*9720,'integer_dense_rows':12000 if ref else 0,'binary_input_verified':True,'full_T_B_C_spatial_positions_retained':True,'scale_applications_per_output_element':1,'bias_applications_per_output_element':0,**Q[fc],'integer_output_shape':[10,1,30,40,cout],'integer_output_sha256':'c'*64,'scaled_output_sha256':'d'*64,'software_plane_storage_dtype':'torch.int8','user_padding':False,'internal_kernel_padding':'not_observed_not_a_cycle_claim','float_matmul_fallback':False}
def records():
 _,bn=a.S.A15.names_and_shapes();out=[]
 for row in a.plan(M):
  r=dict(row);native=r['endpoint_arm']=='native';original={};logical={}
  for k,shape in a.S.G.GEOMETRY.items():
   n=math.prod(shape[:-1]);o=0 if k==a.S.TARGET and not native else n
   original[k]={'fc1_rows':o,'fc2_rows':o,'fc1_calls':int(o>0),'fc2_calls':int(o>0)};logical[k]={'fc1_rows':n,'fc2_rows':n}
  r.update(all12_original_fc_observations=original,all12_full_grid_logical_fc_rows=logical,target_active_quantized_rows={fc:0 if native else 12000 for fc in a.SHAPES},
   target_execution={'mode':r['endpoint_arm'],'full_spatial_grid_retained':True,'old_compact_called':False,'BN_mode':'ALL78_NATIVE_CURRENT','FC':{} if native else {fc:fc_record(fc,r['endpoint_arm']) for fc in a.SHAPES}},
   summed_original_fc1_rows=sum(v['fc1_rows'] for v in original.values()),summed_original_fc2_rows=sum(v['fc2_rows'] for v in original.values()),summed_full_grid_logical_fc1_rows=558000,summed_full_grid_logical_fc2_rows=558000,
   native_BN_calls={k:1 for k in bn},native_BN_inputs={k:[{'shape':s,'dtype':'torch.float32','stride':[1]*len(s)}] for k,s in bn.items()},state_and_BN_unchanged=True,original_modules_restored_and_hooks_removed=True)
  out.append(r)
 return out,bn
class Checks(unittest.TestCase):
 def test_plan_and_actual_vs_geometric_counters(self):
  rows,bn=records();t=a.check_records(rows,M,bn,Q)
  self.assertEqual(t['full_grid_FC1_rows'],23436000);self.assertEqual(t['original_FC1_rows'],23028000);self.assertEqual(t['target_row_plane_FC1_work'],1990560);self.assertEqual(t['model_int_mm_calls'],432)
  self.assertEqual(t['BN_calls'],3276);self.assertEqual([r['arm'] for r in rows[:7]],['native','Q8ref','Q8serial','Q6','Q6_repeat1','Q6_repeat2','Q4'])
  self.assertEqual(a.check_records(rows[:6],M,bn,Q)['full_forwards'],6)
 def test_contract_damage_rejections(self):
  rows,bn=records();mutations=[lambda r:r[3]['target_execution']['FC']['fc1']['int_mm_calls'][0].__setitem__('rows',12000),lambda r:r[3]['target_execution']['FC']['fc1']['int_mm_calls'][7].__setitem__('coefficient',128),lambda r:r[3]['target_execution']['FC']['fc1'].__setitem__('int_mm_call_count',6),lambda r:r[3]['target_execution']['FC']['fc1'].__setitem__('float_matmul_fallback',True),lambda r:r[3]['target_execution']['FC']['fc1'].__setitem__('bias_applications_per_output_element',1),lambda r:r[3]['target_execution']['FC']['fc1'].__setitem__('scale_applications_per_output_element',8),lambda r:r[3]['all12_original_fc_observations'][a.S.TARGET].__setitem__('fc1_calls',1),lambda r:r[3]['target_active_quantized_rows'].__setitem__('fc1',2280),lambda r:r[3]['native_BN_calls'].__setitem__(next(iter(bn)),0),lambda r:r[3].__setitem__('state_and_BN_unchanged',False),lambda r:r[4].__setitem__('quality_endpoint',True),lambda r:r[5].__setitem__('arm','Q6_repeat1')]
  for change in mutations:
   rr=copy.deepcopy(rows);change(rr)
   with self.assertRaises(ValueError):a.check_records(rr,M,bn,Q)
 def test_exact_mixed_dtype_bytes(self):
  original={k:np.zeros(3,dtype=np.int32 if k.startswith('fc') else np.float32) for k in ('head0','head1','head2','head3','u','branch','sn1','sn2','fc1_integer','fc2_integer')}
  self.assertTrue(a.exact_pair(original,copy.deepcopy(original))['pass'])
  for k in original:
   bad=copy.deepcopy(original);bad[k][0]=1 if k.startswith('fc') else -0.0
   self.assertFalse(a.exact_pair(bad,original)['pass'])
 def test_true_tail_allT_plane_rows(self):
  self.assertEqual(int(a.TAIL.sum()),2280)
  for p,expected in ((8,96000),(6,76560),(4,57120)):
   counts=[sum(1 for t in range(10) for y in range(30) for x in range(40) if k>=8-p or y>=27 or x>=36) for k in range(8)]
   self.assertEqual(sum(counts),expected);self.assertEqual(counts[7],12000)
 def test_sampled_dot_binds_sign_tail_and_any_channel(self):
  arrays={};weights={};r={'endpoint_arm':'Q6','target_execution':{'FC':{}}}
  for fc,(cin,cout) in a.SHAPES.items():
   sn=np.zeros((10,1,30,40,cin),dtype=np.float32);sn[...,0]=1
   q=np.zeros((cout,cin),dtype=np.int8);q[:,0]=np.where(np.arange(cout)%2==0,-127,127)
   expected=np.where(a.TAIL[:,None],q[:,0][None,:].astype(np.int32),np.floor_divide(q[:,0].astype(np.int32),4)[None,:]*4)
   arrays['sn1' if fc=='fc1' else 'sn2']=sn;arrays[fc+'_integer']=expected.reshape(10,1,30,40,cout);weights[fc]={'q8':q};r['target_execution']['FC'][fc]=fc_record(fc,'Q6')
  check=a.sampled_integer(arrays,r,weights);self.assertEqual(sum(c['samples'] for c in check),700)
  arrays['fc1_integer'][9,0,29,39,-1]+=1
  with self.assertRaises(ValueError):a.sampled_integer(arrays,r,weights)
 def test_real_source_weights_numpy_quantization(self):
  import torch
  torch.set_num_threads(2);self.assertFalse(torch.cuda.is_initialized());cp=a.OLD/'baseline_checkpoint_verification/source_snapshot/checkpoint_step_000271482.pt';self.assertEqual(a.sha(cp),a.S.CP_SHA)
  state=torch.load(cp,map_location='cpu',weights_only=True)['model_state_dict']
  path=a.RUN/'implementation/bitplane_linear.py';self.assertEqual(a.sha(path),a.CORE_SHA);s=importlib.util.spec_from_file_location('_r26_independent_source_quant_only',path);k=importlib.util.module_from_spec(s);s.loader.exec_module(k)
  for fc,(cin,cout) in a.SHAPES.items():
   w=state[a.S.TARGET+'.'+fc+'.weight'];npw=w.numpy();scale=np.max(np.abs(npw),1)/np.float32(127);den=np.where(scale==0,np.float32(1),scale);q=np.clip(np.rint(npw/den[:,None]),-127,127).astype(np.int8)
   actual=k.quantize_source(types.SimpleNamespace(weight=w,bias=None));self.assertEqual(q.tobytes(),actual['q8'].numpy().tobytes());self.assertEqual(scale.tobytes(),actual['scale'].numpy().tobytes())
