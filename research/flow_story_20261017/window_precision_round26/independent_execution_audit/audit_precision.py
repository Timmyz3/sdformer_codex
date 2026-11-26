#!/usr/bin/env python3
"""Independent R26 original-state, integer-graph/count/raw audit. No GT/forward."""
import argparse,hashlib,importlib.util,json,math
from collections import Counter
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;RUN=HERE.parent;DATE=RUN.parent;REPO=DATE.parents[1];OLD=REPO/'research/flow_story_20261016'
PRIOR=DATE/'current_bn_round16/pilot/independent_execution_audit/audit_pilot_v2.py'
PRIOR_SHA='1e660183e7f1c518524dcde2eb798385b2ad86be7d9ceb9eed443810580ff7f3'
ROOT_FREEZE='98c4b750bead5a12f6d89f9d84f8780f370bc9b84bb22d5f96cd4a594e172bc0';INPUTS_SHA='cc7ab752c7cd1e39ce8b2be023b7701e561ec5e127b600691e22baae109b3f4a';RUNNER_SHA='66ce0f8f2385ab49ed114adde799e2a11c1d88934727baad62cf20495da3379d'
CORE_SHA='759eaa6917ad6f02b25793f11c26c961bf375eaf62249c6299c7b014dd78b916'
HELPER_SHA='4677fe4c1fb7f296ffea62887381ada7bcab46b8e641e14cbd8f31de86abde66'
GRAPH='r26_W271482_all78current_s2b2_fullspatial_weightplanes_H8'
ARMS=('native','Q8ref','Q8serial','Q6','Q4');SHAPES={'fc1':(384,1536),'fc2':(1536,384)}
SAMPLE_XY=((0,0),(0,35),(0,36),(26,35),(27,0),(29,39),(17,17))
def require(v,m):
 if not v:raise ValueError(m)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def digest(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
require(sha(PRIOR)==PRIOR_SHA,'Frozen independent source-state helper')
spec=importlib.util.spec_from_file_location('_r26_independent_source_helper',PRIOR);S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
TAIL=np.tile(np.array([y>=27 or x>=36 for y in range(30) for x in range(40)]),10)
def plan(members):
 out=[]
 for i,r in enumerate(members):
  for arm in ARMS:
   out.append(dict(target_id=r['target_id'],arm=arm,endpoint_arm=arm,execution='native' if arm=='native' else 'integer_quantized',quality_endpoint=True))
   if i==0 and arm=='Q6':out.extend(dict(target_id=r['target_id'],arm='Q6_repeat%d'%j,endpoint_arm='Q6',execution='integer_quantized',quality_endpoint=False) for j in (1,2))
 require(len(members)==8 and len(out)==42,'Fixed42 plan')
 return [dict(r,forward_id=i+1) for i,r in enumerate(out)]
def source_pins():
 require(sha(RUN/'source_freeze_v1.json')==ROOT_FREEZE and sha(RUN/'inputs_v1.json')==INPUTS_SHA,'Root source/input bytes')
 require(sha(RUN/'implementation/run_precision.py')==RUNNER_SHA and sha(RUN/'implementation/bitplane_linear.py')==CORE_SHA,'Actual runner/core bytes')
 out={}
 for r in read(RUN/'source_freeze_v1.json')['files']:
  remote=r['path']
  if '/research_20261017/' in remote:p=DATE/remote.split('/research_20261017/',1)[1]
  elif '/research_20261016/' in remote:p=OLD/remote.split('/research_20261016/',1)[1]
  else:raise ValueError('Unrecognized source relocation')
  require(sha(p)==r['sha256'],'Actual frozen source '+str(p));out[str(p)]=sha(p)
 out.update(S.A14.source_pins())
 for p in (Path(__file__),PRIOR,HERE/'SOURCE_REVIEW_V1.md',HERE/'INTEGER_SAMPLING_PRE_RESULT_V1.md',RUN/'source_freeze_v1.json',RUN/'inputs_v1.json'):out[str(p)]=sha(p)
 return out

def quantization(result,path):
 import torch
 cp=OLD/'baseline_checkpoint_verification/source_snapshot/checkpoint_step_000271482.pt'
 require(sha(cp)==S.CP_SHA,'Actual source weight checkpoint')
 state=torch.load(cp,map_location='cpu',weights_only=True)['model_state_dict']
 rp=path.parent/result['quantization_receipt']['path'];ap=path.parent/result['quantization_archive']['path']
 require(rp.name=='quantization.json' and ap.name=='quantization.npz','Fixed quantization artifacts')
 require(sha(rp)==result['quantization_receipt']['sha256'] and sha(ap)==result['quantization_archive']['sha256'],'Actual quantization bytes')
 r=read(rp);require(r['schema']=='flow_r26_weight_quantization_v1' and r['checkpoint_sha256']==S.CP_SHA and r['original_allcurrent_state_sha256']==S.CURRENT,'Own original-state quantization')
 require(r['arrays']=='quantization.npz' and r['arrays_sha256']==sha(ap) and set(r['FC'])==set(SHAPES),'Quantization binding')
 expected={};summary={}
 with np.load(ap,allow_pickle=False) as z:
  require(set(z.files)==set(r['arrays_records'])=={fc+'__'+k for fc in SHAPES for k in ('q8','scale')},'Four exact q/scale arrays')
  for fc,(cin,cout) in SHAPES.items():
   name=S.TARGET+'.'+fc;require(name+'.bias' not in state,'Production source FC has no bias')
   w=state[name+'.weight'].numpy();require(w.dtype==np.float32 and w.shape==(cout,cin) and np.isfinite(w).all(),'Original FP32 weight geometry')
   scale=np.max(np.abs(w),axis=1).astype(np.float32)/np.float32(127)
   denominator=np.where(scale==0,np.float32(1),scale)
   q=np.clip(np.rint(w/denominator[:,None]),-127,127).astype(np.int8)
   expected[fc]={'q8':q,'scale':scale};m=r['FC'][fc]
   want={'source_weight_sha256':digest(w),'source_bias_sha256':None,'q8_sha256':digest(q),'scale_sha256':digest(scale),'zero_output_channels':int((scale==0).sum()),'source_quantization_device':'cpu','rounding':'torch.round_nearest_ties_to_even','quantized_range':[-127,127],'scale_rule':'output_channel_max_abs_div_127; zero_channel_scale0_q0','low_precision_rule':'signed_arithmetic_right_shift_then_left_shift_no_reclamp_no_correction'}
   require(m==want,'Independent original-weight q/scale reconstruction '+fc)
   for key,v in expected[fc].items():
    name=fc+'__'+key;actual=z[name]
    require(actual.shape==v.shape and actual.dtype==v.dtype and actual.tobytes()==v.tobytes(),'Actual integer/FP32 quantization bytes '+name)
    require(r['arrays_records'][name]=={'shape':list(v.shape),'dtype':str(v.dtype),'sha256':digest(v)},'Quantization payload descriptor')
   summary[fc]=want
 return expected,{'receipt_sha256':sha(rp),'archive_sha256':sha(ap),'FC':summary,'independently_reconstructed_from_original_PT':True,'payload_count':4}

def check_FC(ex,mode,fc,quant):
 cin,cout=SHAPES[fc];drop={'Q8ref':0,'Q8serial':0,'Q6':2,'Q4':4}[mode]
 want={'mode':mode,'actual_input_shape':[10,1,30,40,cin],'input_rows':12000,'full_tile_rows':9720,'tail_rows':2280,'original_linear_forward_calls':0,
 'int_mm_call_count':1 if mode=='Q8ref' else 8,'row_plane_work':0 if mode=='Q8ref' else 96000-drop*9720,'integer_dense_rows':12000 if mode=='Q8ref' else 0,
 'binary_input_verified':True,'full_T_B_C_spatial_positions_retained':True,'scale_applications_per_output_element':1,'bias_applications_per_output_element':0,
 'q8_sha256':quant[fc]['q8_sha256'],'scale_sha256':quant[fc]['scale_sha256'],'integer_output_shape':[10,1,30,40,cout],
 'software_plane_storage_dtype':'torch.int8','user_padding':False,'internal_kernel_padding':'not_observed_not_a_cycle_claim','float_matmul_fallback':False}
 for k,v in want.items():require(ex[k]==v,'Actual FC integer contract '+fc+'/'+k)
 require(S.G.hexdigest(ex['integer_output_sha256']) and S.G.hexdigest(ex['scaled_output_sha256']),'Full actual integer/scaled output hashes')
 require(len(ex['int_mm_calls'])==want['int_mm_call_count'],'Every actual integer call recorded')
 rows=0
 for i,r in enumerate(ex['int_mm_calls']):
  ref=mode=='Q8ref';tail=not ref and i<drop;n=2280 if tail else 12000
  w={'kind':'integer_dense_reference' if ref else 'binary_weight_plane','bit':None if ref else i,'coefficient':None if ref else -128 if i==7 else 1<<i,
    'rows':n,'Cin':cin,'Cout':cout,'input_dtype':'torch.int8','weight_dtype':'torch.int8','output_dtype':'torch.int32','weight_shape':[cin,cout],'explicit_padding_rows':0,'explicit_padding_channels':0}
  if not ref:w['row_scope']='tail_only_Q8' if tail else 'all_spatial_rows'
  for k,v in w.items():require(r[k]==v,'Actual retained plane identity/row/C/sign '+fc+'/'+k)
  require(type(r['input_spike_ones']) is int and 0<=r['input_spike_ones']<=n*cin,'Actual binary operand population bound');rows+=n
 return rows

def check_records(records,members,bn_shapes,quant):
 require(0<len(records)<=42,'Nonempty bounded prefix');tot=Counter()
 for r,w in zip(records,plan(members)):
  require(all(r[k]==v for k,v in w.items()),'Fixed ordered42 record');mode=r['endpoint_arm'];native=mode=='native'
  require(set(r['all12_original_fc_observations'])==set(r['all12_full_grid_logical_fc_rows'])==set(S.G.GEOMETRY),'All12 FC observer population')
  geometric=original=0
  for name,shape in S.G.GEOMETRY.items():
   n=math.prod(shape[:-1]);o=0 if name==S.TARGET and not native else n
   require(r['all12_original_fc_observations'][name]=={'fc1_rows':o,'fc2_rows':o,'fc1_calls':int(o>0),'fc2_calls':int(o>0)},'True original FC hook count '+name)
   require(r['all12_full_grid_logical_fc_rows'][name]=={'fc1_rows':n,'fc2_rows':n},'Fullgrid geometry is separate from original FC calls')
   geometric+=n;original+=o
  require(geometric==558000 and original==(558000 if native else 546000),'Independent native/replacement row total')
  require(r['summed_original_fc1_rows']==r['summed_original_fc2_rows']==original and r['summed_full_grid_logical_fc1_rows']==r['summed_full_grid_logical_fc2_rows']==geometric,'Per-forward independent row sums')
  require(r['target_active_quantized_rows']=={'fc1':0 if native else 12000,'fc2':0 if native else 12000},'Actual full-grid wrapper input rows')
  ex=r['target_execution'];require(ex['mode']==mode and ex['full_spatial_grid_retained'] is True and ex['old_compact_called'] is False and ex['BN_mode']=='ALL78_NATIVE_CURRENT','Only target FC callables change')
  require(set(ex['FC'])==(set() if native else set(SHAPES)),'Actual integer FC scope')
  if native:tot.update(native=1,target_native_FC1_rows=12000,target_native_FC2_rows=12000)
  else:
   tot['integer_reference' if mode=='Q8ref' else 'bitserial']+=1
   for j,fc in enumerate(SHAPES,1):
    n=check_FC(ex['FC'][fc],mode,fc,quant);tot['model_int_mm_calls']+=len(ex['FC'][fc]['int_mm_calls'])
    tot['target_integer_dense_FC%d_rows'%j if mode=='Q8ref' else 'target_row_plane_FC%d_work'%j]+=n
  require(set(r['native_BN_calls'])==set(r['native_BN_inputs'])==set(bn_shapes),'All78 real BN observers')
  for name,shape in bn_shapes.items():
   entries=r['native_BN_inputs'][name];require(r['native_BN_calls'][name]==1 and len(entries)==1 and entries[0]['shape']==shape and entries[0]['dtype']=='torch.float32','Full native BN domain '+name)
   require(len(entries[0]['stride'])==len(shape) and all(type(v) is int and v>=0 for v in entries[0]['stride']),'Actual native BN strides');tot['BN_input_elements']+=math.prod(shape)
  require(r['state_and_BN_unchanged'] is True and r['original_modules_restored_and_hooks_removed'] is True,'Per-forward full-state/BN/module/hook closure')
  tot.update(full_forwards=1,full_grid_FC1_rows=geometric,full_grid_FC2_rows=geometric,original_FC1_rows=original,original_FC2_rows=original,BN_calls=78)
  tot['quality_endpoints' if r['quality_endpoint'] else 'determinism_replays']+=1
 if len(records)==42:
  for k,v in {'native':8,'integer_reference':8,'bitserial':26,'full_grid_FC1_rows':23436000,'full_grid_FC2_rows':23436000,'original_FC1_rows':23028000,'original_FC2_rows':23028000,'target_native_FC1_rows':96000,'target_native_FC2_rows':96000,'target_integer_dense_FC1_rows':96000,'target_integer_dense_FC2_rows':96000,'target_row_plane_FC1_work':1990560,'target_row_plane_FC2_work':1990560,'model_int_mm_calls':432,'BN_calls':3276,'BN_input_elements':59402488320,'quality_endpoints':40,'determinism_replays':2}.items():require(tot[k]==v,'Complete independent work sum '+k)
 tot.update(optimizer_steps=0,calibration_forwards=0,compact=0,expanded=0)
 return tot

def exact_pair(actual,reference):
 keys={'head0','head1','head2','head3','u','branch','sn1','sn2','fc1_integer','fc2_integer'};require(set(actual)==set(reference)==keys,'All ten actual integer-graph arrays')
 out={}
 for name,a in actual.items():
  b=reference[name];require(a.shape==b.shape and a.dtype==b.dtype,'Exact pair type/shape')
  dif=int(np.count_nonzero(np.ascontiguousarray(a).view(np.uint8)!=np.ascontiguousarray(b).view(np.uint8)))
  out[name]={'bitwise_equal':dif==0,'different_bytes':dif,'shape':list(a.shape),'dtype':'torch.int32' if a.dtype==np.int32 else 'torch.float32','actual_sha256':digest(a),'reference_sha256':digest(b)}
 return {'pass':all(v['bitwise_equal'] for v in out.values()),'array_checks':out}

def sampled_integer(arrays,record,weights):
 mode=record['endpoint_arm'];out=[]
 for fc,(cin,cout) in SHAPES.items():
  sn=arrays['sn1' if fc=='fc1' else 'sn2'];ints=arrays[fc+'_integer'];q=weights[fc]['q8'].astype(np.int64)
  channels=(0,cout//3,cout//2,2*cout//3,cout-1);checked=0
  for t in range(10):
   for y,x in SAMPLE_XY:
    vector=sn[t,0,y,x].astype(np.int64)
    require(np.isin(vector,(0,1)).all(),'Actual saved binary vector')
    w=q[np.asarray(channels)]
    if mode in ('Q6','Q4') and y<27 and x<36:
     divisor=4 if mode=='Q6' else 16;w=np.floor_divide(w,divisor)*divisor
    expected=w@vector;got=ints[t,0,y,x,np.asarray(channels)]
    require(np.array_equal(got,expected),'Fixed source-weight int64 dot versus actual int32 output');checked+=len(channels)
  flat=sn.reshape(12000,cin)
  for call in record['target_execution']['FC'][fc]['int_mm_calls']:
   data=flat[TAIL] if call['rows']==2280 else flat
   require(call['input_spike_ones']==int(data.sum(dtype=np.int64)),'Original-array actual call binary population')
  out.append({'FC':fc,'samples':checked,'coordinates':[list(v) for v in SAMPLE_XY],'time_steps':list(range(10)),'output_channels':list(channels),'independent_int64_dot_exact':True})
 return out

def audit(path):
 sources=source_pins();state=S.source_state();_,members,evidence=S.h8_inputs();_,bn_shapes=S.A15.names_and_shapes()
 d=read(path);b=d['binding'];inputs=read(RUN/'inputs_v1.json');records=d.get('forward_records',[]);n=len(records);complete=d['status']=='WINDOW_PRECISION42_COMPLETE'
 require(d['schema']=='flow_r26_window_precision_v1' and (complete or d['status']=='INCOMPLETE_OR_NUMERICAL_FAILURE'),'Own precision graph result')
 require(d['graph_id']==b['graph_id']==GRAPH and b['plan']=='precision42' and b['forward_plan']==plan(members),'Exact fixed graph and plan')
 require(d['training_performed'] is False and d['optimizer_steps']==d['calibration_forwards']==0 and d['hardware_benefit_claim'] is False and d['flow_specific_increment_claim'] is False,'Zero training/calibration/no unsupported benefit')
 require(d['inference_executed'] is True and d['full_forward_count']==n and n<=d['full_forward_attempts']<=min(n+1,42),'Recorded nonempty model prefix')
 want={'inputs_sha256':INPUTS_SHA,'source_freeze_sha256':ROOT_FREEZE,'primitive_sha256':CORE_SHA,'data_only_helper_sha256':HELPER_SHA,'old_spatial_compact_called':False,'source_quantization_device':'cpu','BN':'all78_current','BN_mode':'ALL78_NATIVE_CURRENT','target':S.TARGET,'other_11_MLPs':'native','quality_endpoints':40,'determinism_replays':2,'optimizer_steps':0,'calibration_forwards':0,'h8_sha256':inputs['h8']['sha256'],'membership_sha256':inputs['membership']['sha256'],'region_sha256':inputs['regions']['sha256'],'independent_admission_sha256':inputs['independent_admission']['sha256'],'model_forwards':42,'native_forwards':8,'integer_reference_forwards':8,'bitplane_forwards':26,'expected_native_BN_calls':3276,'expected_full_grid_logical_FC_rows_each':23436000,'expected_original_FC_rows_each':23028000,'expected_target_native_FC_rows_each':96000,'expected_target_integer_dense_rows_each':96000,'expected_target_row_plane_work_each':1990560,'expected_model_int_mm_calls_each_targetFC':216,'backend_operator_smoke_calls':4,'backend_smoke_model_forwards':0,'expected_saved_arrays':314}
 for k,v in want.items():require(b[k]==v,'Exact graph binding '+k)
 for k,v in {'full_accepted_metadata_validated':True,'all_dataset_current_file_stat_checked':False,'allowed8_current_files_checked_separately':True,'allowed8_current_files_SHA_checked':24,'sealed_confirmation_array_paths_visited':0,'GT_array_loads':0}.items():require(b['data_preflight'][k]==v,'Source-only accepted data contract '+k)
 require(b['precision']=='FP32 activations/currentBN, exact int8 x int8 -> int32 FC; no AMP/TF32; T10/B1/full9' and 0<b['wall_minutes']<=20,'Fixed numeric/budget contract')
 for k,v in {'checkpoint_sha256':S.CP_SHA,'checkpoint_step':271482,'source_state_sha256':S.SOURCE,'allcurrent_state_sha256':S.CURRENT,'source_tensor_count':711,'removed_BN_buffers':234,'allcurrent_tensor_count':477,'allcurrent_elements':54919238,'optimizer_restored':False,'external_BN_artifact_loaded':False,'source_BN_buffers_removed_before_inference':True}.items():require(b['checkpoint_cpu_audit'][k]==v,'Actual source state '+k)
 S.A14.verify_scope(d['BN_scope'],set(bn_shapes));require(d['initial_allcurrent_state_sha256']==S.CURRENT,'Independent original477 state')
 require(d['runtime']['AMP'] is False and d['runtime']['TF32'] is False,'Actual disabled AMP/TF32')
 require(d['data_evidence']==[dict(target_id=r['target_id'],input_evidence=evidence[r['target_id']]) for r in members],'Actual H8 input bytes')
 expected_smoke=[{'rows':rows,'Cin':cin,'Cout':cout,'input_dtype':'torch.int8','output_dtype':'torch.int32','explicit_padding_rows':0,'passed_exact_column_sum':True,'model_forward_count':0} for cin,cout in SHAPES.values() for rows in (12000,2280)]
 require(d['backend_operator_smoke']==expected_smoke,'Four real-shape integer operator checks separate from model count')
 weights,quant=quantization(d,path);tot=check_records(records,members,bn_shapes,quant['FC']);tot.update(backend_operator_calls=4,quantization_archives=1,quantization_payloads=4)
 if complete:require(n==d['full_forward_attempts']==42 and d['final_state_sha256']==S.CURRENT and all(d[k] is True for k in ('all_states_unchanged','all_hooks_removed','original_modules_restored')),'Complete source/state closure')
 native=reference=q6=None;checks=[];predictions=[];samples=[];seen=set()
 for rec in records:
  tid,arm,mode=rec['target_id'],rec['arm'],rec['endpoint_arm'];require(rec['input_evidence']==evidence[tid],'Actual record input source')
  name=rec['prediction_file'];require(Path(name).name==name and name not in seen,'Unique original archive');seen.add(name)
  archive=path.parent/name;require(sha(archive)==rec['prediction_file_sha256'],'Actual compressed prediction bytes')
  capture=arm in ('native','Q8ref','Q8serial') or (tid==members[0]['target_id'] and mode=='Q6')
  floatkeys={'u','branch','sn1','sn2'};intkeys=set() if mode=='native' else {'fc1_integer','fc2_integer'};capkeys=floatkeys|intkeys
  keys={'head%d'%i for i in range(4)}|(capkeys if capture else set())
  require(set(rec['saved_array_names'])==keys and len(rec['saved_array_names'])==len(keys),'Actual saved-array inventory')
  with np.load(archive,allow_pickle=False) as z:require(set(z.files)==keys,'Exact actual archive payloads');arrays={k:z[k] for k in z.files}
  require(len(rec['head_records'])==4 and set(rec['tensor_records'])==capkeys,'All original tensor descriptors')
  for i,h in enumerate(rec['head_records']):
   a=arrays['head%d'%i];require(a.shape==(1,2,480,640) and a.dtype==np.float32 and np.isfinite(a).all() and h=={'shape':list(a.shape),'dtype':'torch.float32','sha256':digest(a)},'Four full FP32 head arrays')
  for key,meta in rec['tensor_records'].items():
   integer=key in intkeys;width=1536 if key in ('sn2','fc1_integer') else 384;shape=[10,1,30,40,width];dtype=np.int32 if integer else np.float32
   require(meta['shape']==shape and meta['dtype']==('torch.int32' if integer else 'torch.float32') and S.G.hexdigest(meta['sha256']),'Actual full-grid tensor metadata')
   if capture:
    a=arrays[key];require(list(a.shape)==shape and a.dtype==dtype and np.isfinite(a).all() and digest(a)==meta['sha256'],'Original actual float/int payload')
    if key in ('sn1','sn2'):require(np.isin(a,(0,1)).all(),'Actual binary spike payload')
   if integer:require(meta['sha256']==rec['target_execution']['FC'][key[:3]]['integer_output_sha256'],'Actual saved integer linked to primitive output')
  if mode=='native':native=(tid,arrays)
  else:require(native is not None and native[0]==tid and rec['tensor_records']['u']['sha256']==digest(native[1]['u']),'Exact common predecision u')
  if capture and mode!='native':samples.append({'forward_id':rec['forward_id'],'arm':arm,'FC_checks':sampled_integer(arrays,rec,weights)})
  if arm=='Q8ref':reference=(rec['forward_id'],tid,arrays)
  elif arm=='Q8serial':
   require(reference is not None and reference[1]==tid,'Own same-frame integer reference')
   checks.append(dict(target_id=tid,forward_id=rec['forward_id'],arm=arm,kind='Q8serial_vs_Q8ref_exact',reference_forward_id=reference[0],**exact_pair(arrays,reference[2])))
  if arm=='Q6' and tid==members[0]['target_id']:q6=(rec['forward_id'],arrays)
  elif arm.startswith('Q6_repeat'):
   require(q6 is not None and q6[0]==4,'Fixed firstQ6 reference4')
   checks.append(dict(target_id=tid,forward_id=rec['forward_id'],arm=arm,kind='Q6_exact_replay',reference_forward_id=q6[0],**exact_pair(arrays,q6[1])))
  tot.update(archives=1,head_payloads=4,float_capture_payloads=4 if capture else 0,integer_capture_payloads=len(intkeys) if capture else 0,saved_payloads=len(keys))
  predictions.append({k:rec[k] for k in ('forward_id','target_id','arm','execution','quality_endpoint','endpoint_arm')}|{'file':str(archive),'file_sha256':sha(archive),'head3_sha256':digest(arrays['head3'])})
 require(checks==d['numerical_checks'],'Independently repeated complete bytewise numerical reports')
 failures=[c for c in checks if not c['pass']]
 if complete:
  require(not failures and len(checks)==10 and len(samples)==19 and sum(c['samples'] for r in samples for c in r['FC_checks'])==13300,'Full exact checks and predeclared integer dot samples')
  for k,v in {'archives':42,'head_payloads':168,'float_capture_payloads':108,'integer_capture_payloads':38,'saved_payloads':314}.items():require(tot[k]==v,'Complete payload count '+k)
  require(0<d['wall_seconds']<=1260 and 0<d['max_allocated_bytes']<=d['max_reserved_bytes'],'Actual engineering resources')
 elif failures:require(len(failures)==1 and failures[0]['forward_id']==n==d['full_forward_attempts'],'Immediate exact gate stop')
 return {'schema':'flow_r26_window_precision_independent_execution_audit_v1','graph_id':GRAPH,'status':'PASS' if complete else 'PASS_PREFIX_EVIDENCE_ONLY','quality_analysis_allowed':complete,'source_result_sha256':sha(path),'source_freeze_sha256':ROOT_FREEZE,'auditor_sha256':sha(__file__),'pre_result_freeze_sha256':sha(HERE/'PRE_RESULT_FREEZE_V1.json'),'source_pins':sources,'source_state':state,'quantization':quant,'counts':dict(tot),'numerical_checks':checks,'sampled_integer_checks':samples,'predictions':predictions,'input_evidence':d['data_evidence'],'runtime':d['runtime'],'resources':{k:d.get(k) for k in ('wall_seconds','max_allocated_bytes','max_reserved_bytes')},'limitations':['Only CPU source/raw audit; no new model forward, GPU or GT/AEE read. H8 is reused development.','All source q/scale and raw integer outputs are bound.13300 original integer values independently recomputed; remaining matrices use full pair/replay byte equality, not complete independent matmul replay.','Q4 and later Q6 retain source/call counters and tensor hashes, not full prefix/integer raw captures.','Per-forward state closure is from frozen runtime assertions; independent full state reconstruction covers source and reported initial/final hashes.','All FC spatial rows and native currentBN remain. Integer plane terms cannot be added to11 FP32 MLP cycles; no RTL/DC/packing or net-chip speed proof.']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'Exclusive audit result')
 for f,h in read(HERE/'PRE_RESULT_FREEZE_V1.json')['source_sha256'].items():require(sha(f)==h,'Frozen actual auditor bytes')
 out=audit(a.result)
 with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({k:out[k] for k in ('status','quality_analysis_allowed','source_result_sha256','counts')}))
if __name__=='__main__':main()
