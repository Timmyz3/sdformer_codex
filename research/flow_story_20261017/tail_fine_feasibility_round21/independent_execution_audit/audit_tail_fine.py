#!/usr/bin/env python3
"""R21 independent122 execution/raw numerical audit. Never reads GT/AEE."""
import argparse,hashlib,importlib.util,json,math
from collections import Counter
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;RUN=HERE.parent;DATE=RUN.parent;REPO=DATE.parents[1]
OLD=REPO/'research/flow_story_20261016'
PRIOR=DATE/'current_bn_round16/pilot/independent_execution_audit/audit_pilot_v2.py'
PRIOR_SHA='1e660183e7f1c518524dcde2eb798385b2ad86be7d9ceb9eed443810580ff7f3'
ROOT_FREEZE='6d5238440c0a3e9a4fa05a061cd28d9c7a73caa591033efd97e0ea327c2888f4'
INPUTS_SHA='ecbc786927b3a79b505560e38411c4b47349ffdcca725d413ed083bcef66359b'
RUNNER_SHA='7e650140ebd50a1af50aed0f46189a3c218683ad8c1e814e7c7a131326532411'
GRAPH_ID='r21_W271482_all78current_s2b2_R16V2_T8_k1_H8'
PRIMITIVE_SHA='e874e976ce4833a0638092ca78612e8224139eb1cc3b49d9ce0098488a3bb731'
MATH_SHA='ad85fab7c0de64bad5b1fca15baca1af3347e32e226fb17ec18487485cc837ec'
def require(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
require(sha(PRIOR)==PRIOR_SHA,'Frozen independent source/numeric helpers')
spec=importlib.util.spec_from_file_location('_r21_independent_helpers',PRIOR);S=importlib.util.module_from_spec(spec);spec.loader.exec_module(S)
ARMS=('native','FF','T8')+tuple('fine_%d_%d'%(y,x) for y in range(3) for x in range(4))
def plan(members):
 out=[]
 for i,r in enumerate(members):
  for arm in ARMS:
   out.append(dict(target_id=r['target_id'],arm=arm,execution='native' if arm=='native' else 'compact',quality_endpoint=True,endpoint_arm=arm))
   if i==0 and arm=='T8':
    out.extend(dict(target_id=r['target_id'],arm='T8_repeat%d'%j,execution='compact',quality_endpoint=False,endpoint_arm='T8') for j in (1,2))
 require(len(members)==8 and len(out)==122,'Fixed8x15 plus exactly2 replay budget')
 return [dict(x,forward_id=i+1) for i,x in enumerate(out)]
def cells(arm):
 require(arm in ARMS[1:] or arm in ('T8_repeat1','T8_repeat2'),'Known compact action')
 target=tuple(map(int,arm.split('_')[1:])) if arm.startswith('fine_') else None
 out=[]
 for sy in range(0,30,9):
  for sx in range(0,40,9):
   ey,ex=min(sy+9,30),min(sx+9,40);step=1 if arm=='FF' or sy//9==3 or sx//9==4 or target==(sy//9,sx//9) else 3
   for y in range(sy,ey,step):
    for x in range(sx,ex,step):
     yy,xx=min(y+step,ey),min(x+step,ex);out.append([y,yy,x,xx,(yy-y)*(xx-x)])
 require(sum(v[4] for v in out)==1200,'Real-tail complete cover')
 return out
def source_pins():
 require(sha(RUN/'source_freeze_v1.json')==ROOT_FREEZE and sha(RUN/'inputs_v1.json')==INPUTS_SHA and sha(RUN/'implementation/run_tail_fine_domain.py')==RUNNER_SHA,'Exact R21 source/input')
 pins={}
 for item in read(RUN/'source_freeze_v1.json')['files']:
  remote=item['path']
  if '/research_20261017/' in remote:p=DATE/remote.split('/research_20261017/',1)[1]
  elif '/research_20261016/' in remote:p=OLD/remote.split('/research_20261016/',1)[1]
  else:raise ValueError('Unrecognized remote source mapping')
  require(sha(p)==item['sha256'],'Actual frozen source '+str(p));pins[str(p)]=sha(p)
 pins.update(S.A14.source_pins())
 for p in (RUN/'source_freeze_v1.json',RUN/'inputs_v1.json',PRIOR,Path(__file__),HERE/'SOURCE_REVIEW_V1.md'):pins[str(p)]=sha(p)
 return pins
def check_execution(ex,arm):
 if arm=='native':require(ex=={},'Native no wrapper');return 12000
 c=cells(arm);r=len(c);n=10*r
 require(ex['cells']==c and ex['areas']==[v[4] for v in c],'Actual complete T/true-tail compact cell descriptors')
 want={'representatives':r,'statistics_barriers':2,'FC1_retained_elements':n*1536,'FC2_retained_elements':n*384,
  'full_T_and_all_channels':True,'native_MLP_shortcut':False,'BN_mode':'NATIVE_DENSE_CURRENT','native_BN_calls':2,
  'full_domain_BN_elements':23040000,'expand_written_elements':23040000,'BN_output_dense_elements':23040000,
  'production_BN1_gather_read_elements':n*1536,'production_BN1_gather_written_elements':n*1536,
  'diagnostic_BN2_gather_read_elements':n*384,'cell_exact_audit_read_elements':46080000,
  'each_cell_BN1_and_BN2_bitwise_equal':True,'packed_input_shape':[10,1,r,1,384],
  'diagnostic_moments_full_domain_elements_per_scan':23040000,'diagnostic_moments_scans':2}
 for k,v in want.items():require(ex[k]==v,'Actual compact full-domain service '+k)
 require(S.G.hexdigest(ex['packed_input_sha256']),'Packed producer actual hash')
 require(ex['schedule']=='Compact PSN/FC; expand each FC output over original HW; native dense BN; sample all T/B/C at each cell top-left for PSN2. No producer recomputation.','Frozen primitive schedule')
 return n
def check_count_records(records,members,bn_shapes):
 require(0<len(records)<=122,'Recorded nonempty bounded prefix')
 totals=Counter()
 for r,want in zip(records,plan(members)):
  require(all(r[k]==v for k,v in want.items()),'Exact fixed new122 order')
  nr=check_execution(r['target_execution'],r['arm']);rows=0
  require(set(r['all12_observed_fc_rows'])==set(S.G.GEOMETRY),'All12 actual FC observers')
  for name,shape in S.G.GEOMETRY.items():
   count=nr if name==S.TARGET else math.prod(shape[:-1]);require(r['all12_observed_fc_rows'][name]=={'fc1_rows':count,'fc2_rows':count},'Actual service rows '+name);rows+=count
  require(r['summed_fc1_rows']==r['summed_fc2_rows']==rows,'Per-forward FC row sum')
  require(set(r['native_BN_calls'])==set(r['native_BN_inputs'])==set(bn_shapes),'All78 actual native BN observers')
  for name,shape in bn_shapes.items():
   entries=r['native_BN_inputs'][name]
   require(r['native_BN_calls'][name]==1 and len(entries)==1 and entries[0]['shape']==shape and entries[0]['dtype']=='torch.float32','Unreduced full-domain current BN '+name)
   require(len(entries[0]['stride'])==len(shape) and all(type(x) is int and x>=0 for x in entries[0]['stride']),'Actual BN stride inventory')
   totals['BN_input_elements']+=math.prod(shape)
  require(r['state_and_BN_unchanged'] is True and r['original_modules_restored_and_hooks_removed'] is True,'Every completed forward closure')
  totals.update(full_forwards=1,FC1_rows=rows,FC2_rows=rows,BN_calls=78,observed_FC_modules=24)
  totals[r['execution']]+=1;totals['quality_endpoints' if r['quality_endpoint'] else 'determinism_replays']+=1
  if r['execution']=='compact':
   totals.update(compact_target_calls=1,target_full_domain_BN_elements=23040000,target_expand_written_elements=23040000,production_BN1_gather_elements=nr*1536)
 require(totals['native']<=8 and totals['compact']<=114,'Actual execution population')
 if len(records)==122:require(totals['FC1_rows']==totals['FC2_rows']==67229280 and totals['BN_calls']==9516 and totals['native']==8 and totals['compact']==114 and totals['quality_endpoints']==120 and totals['determinism_replays']==2,'Independent final service/endpoint population')
 totals.update(expanded=0,optimizer_steps=0,calibration_forwards=0)
 return totals
def replay(actual,reference):
 require(set(actual)==set(reference)=={'head0','head1','head2','head3','u','branch','sn1','sn2','sn1_packed','sn2_packed'},'Ten complete replay arrays')
 checks={}
 for k,a in actual.items():
  b=reference[k];require(a.shape==b.shape and a.dtype==b.dtype,'Replay raw shape/dtype')
  aa=np.ascontiguousarray(a).view(np.uint8);bb=np.ascontiguousarray(b).view(np.uint8);different=int(np.count_nonzero(aa!=bb))
  checks[k]={'bitwise_equal':different==0,'different_bytes':different,'shape':list(a.shape),'dtype':'torch.float32','actual_sha256':S.digest(a),'reference_sha256':S.digest(b)}
 return {'pass':all(v['bitwise_equal'] for v in checks.values()),'array_checks':checks}
def expand_spikes(packed,c):
 out=np.empty((10,1,30,40,packed.shape[-1]),dtype=np.float32)
 for i,(y,yy,x,xx,_) in enumerate(c):out[:,:,y:yy,x:xx,:]=packed[:,:,i:i+1,:,:]
 return out
def audit(path):
 sources=source_pins();state=S.source_state();_,members,evidence=S.h8_inputs();_,bn_shapes=S.A15.names_and_shapes()
 d=read(path);b=d['binding'];inputs=read(RUN/'inputs_v1.json');n=len(d['forward_records']);complete=d['status']=='TAIL_FINE_DOMAIN122_COMPLETE'
 require(d['graph_id']==b['graph_id']==GRAPH_ID,'Distinct frozen approximate algorithm graph')
 require(d['schema']=='flow_r21_tail_fine_domain_v1' and (complete or d['status']=='INCOMPLETE_OR_NUMERICAL_FAILURE'),'Actual R21 complete or stopped-prefix schema')
 require(d['full_forward_count']==n and n<=d['full_forward_attempts']<=min(n+1,122) and d['inference_executed'] is True and d['training_performed'] is False,'Recorded forward/attempt prefix')
 require(d['optimizer_steps']==d['calibration_forwards']==0 and d['hardware_benefit_claim'] is False and d['flow_specific_increment_claim'] is False,'No training or unsupported contribution')
 require(b['plan']=='domain122' and b['forward_plan']==plan(members) and b['target']==S.TARGET and b['other_11_MLPs']=='native' and b['BN']=='all78_current' and b['BN_mode']=='NATIVE_DENSE_CURRENT','Own compact graph fixed scope')
 for k,v in {'inputs_sha256':INPUTS_SHA,'source_freeze_sha256':ROOT_FREEZE,'primitive_sha256':PRIMITIVE_SHA,'shared_math_sha256':MATH_SHA,'independent_admission_sha256':inputs['independent_admission']['sha256'],'membership_sha256':inputs['membership']['sha256'],'h8_sha256':inputs['h8']['sha256'],'region_sha256':inputs['regions']['sha256'],'compact_forwards':114,'native_forwards':8,'quality_endpoints':120,'determinism_replays':2,'expected_FC_rows_each':67229280,'expected_BN_calls':9516}.items():require(b[k]==v,'Actual graph binding '+k)
 for k,v in {'full_accepted_metadata_validated':True,'all_dataset_current_file_stat_checked':False,'allowed8_current_files_checked_separately':True,'allowed8_current_files_SHA_checked':24,'sealed_confirmation_array_paths_visited':0,'GT_array_loads':0}.items():require(b['data_preflight'][k]==v,'H8-only metadata/source contract '+k)
 require(b['precision']=='FP32; no AMP/TF32; T10/B1/full9' and b['optimizer_steps']==b['calibration_forwards']==0 and 0<b['wall_minutes']<=20,'Fixed precision/budget')
 cpu=b['checkpoint_cpu_audit']
 for k,v in {'checkpoint_sha256':S.CP_SHA,'checkpoint_step':271482,'source_state_sha256':S.SOURCE,'allcurrent_state_sha256':S.CURRENT,'source_tensor_count':711,'removed_BN_buffers':234,'allcurrent_tensor_count':477,'allcurrent_elements':54919238,'optimizer_restored':False,'external_BN_artifact_loaded':False,'source_BN_buffers_removed_before_inference':True}.items():require(cpu[k]==v,'Actual original state inventory '+k)
 S.A14.verify_scope(d['BN_scope'],set(bn_shapes));require(d['initial_allcurrent_state_sha256']==S.CURRENT,'Independent original current477 state')
 if complete:require(n==d['full_forward_attempts']==122 and d['final_state_sha256']==S.CURRENT and all(d[k] is True for k in ('all_states_unchanged','all_hooks_removed','original_modules_restored')),'Complete actual closure')
 require(d['runtime']['AMP'] is False and d['runtime']['TF32'] is False,'Actual precision')
 require(d['data_evidence']==[dict(target_id=r['target_id'],input_evidence=evidence[r['target_id']]) for r in members],'Fixed H8 source bytes')
 totals=check_count_records(d['forward_records'],members,bn_shapes);native=None;t8=None;predictions=[];checks=[];seen=set()
 for rec in d['forward_records']:
  tid,arm,mode=rec['target_id'],rec['arm'],rec['execution'];require(rec['input_evidence']==evidence[tid],'Actual per-forward source identity')
  name=rec['prediction_file'];require(name==Path(name).name and name not in seen,'New unique archive');seen.add(name)
  archive=Path(path).parent/name;require(sha(archive)==rec['prediction_file_sha256'],'Actual archive bytes')
  captures=arm in ('native','FF') or (tid==members[0]['target_id'] and arm.startswith('T8'))
  c=[] if mode=='native' else cells(arm);capkeys={'u','branch','sn1','sn2'}|({'sn1_packed','sn2_packed'} if mode=='compact' else set())
  keys={'head%d'%i for i in range(4)}|(capkeys if captures else set())
  with np.load(archive,allow_pickle=False) as z:
   require(set(z.files)==keys,'Exact new saved payload set');arrays={k:z[k] for k in z.files}
  require(len(rec['head_records'])==4 and set(rec['tensor_records'])==capkeys,'Complete actual tensor metadata')
  for i,h in enumerate(rec['head_records']):
   arr=arrays['head%d'%i];require(arr.shape==(1,2,480,640) and arr.dtype==np.float32 and np.isfinite(arr).all() and h=={'shape':list(arr.shape),'sha256':S.digest(arr)},'Actual full head/bytes')
  for key,meta in rec['tensor_records'].items():
   width=1536 if key.startswith('sn2') else 384;shape=[10,1,len(c),1,width] if key.endswith('_packed') else [10,1,30,40,width]
   require(meta['shape']==shape and meta['dtype']=='torch.float32' and S.G.hexdigest(meta['sha256']),'FullT actual capture metadata')
   if captures:
    arr=arrays[key];require(list(arr.shape)==shape and arr.dtype==np.float32 and np.isfinite(arr).all() and S.digest(arr)==meta['sha256'],'Actual raw target bytes')
    if key.startswith('sn'):require(np.isin(arr,(0,1)).all(),'Binary complete PSN')
  if captures and mode=='compact':
   for sn in ('sn1','sn2'):require(S.digest(expand_spikes(arrays[sn+'_packed'],c))==S.digest(arrays[sn]),'Actual packed spikes expand with real cells')
  if mode=='native':native=(tid,arrays)
  else:
   require(native is not None and native[0]==tid and rec['tensor_records']['u']['sha256']==S.digest(native[1]['u']),'All actions same actual predecision u')
   if arm=='FF':
    checks.append(dict(forward_id=rec['forward_id'],target_id=tid,kind='allfine_vs_native',arm='FF',**S.numerical(arrays,native[1])))
   if tid==members[0]['target_id'] and arm=='T8':t8=(rec['forward_id'],arrays)
   if arm.startswith('T8_repeat'):
    require(t8 is not None,'Actual own-graph T8 reference exists')
    checks.append(dict(forward_id=rec['forward_id'],target_id=tid,kind='T8_exact_replay',arm=arm,reference_forward_id=t8[0],**replay(arrays,t8[1])))
  totals.update(archives=1,head_payloads=4,capture_payloads=len(capkeys) if captures else 0)
  predictions.append({k:rec[k] for k in ('forward_id','target_id','arm','execution','quality_endpoint','endpoint_arm')}|{'file':str(archive),'file_sha256':sha(archive),'head3_sha256':S.digest(arrays['head3'])})
 require(len(checks)==len(d['numerical_checks']),'Every actual numerical check independently repeated')
 for got,want in zip(d['numerical_checks'],checks):
  require(all(got[k]==want[k] for k in ('forward_id','target_id','kind','arm','pass')),'Exact numerical pair identity/status')
  if want['kind']=='allfine_vs_native':S.check_reported(got,want)
  else:require(got==want,'Actual bytewise T8 replay report')
 failures=[x for x in checks if not x['pass']]
 if complete:
  require(not failures and len(checks)==10,'All8 FF+2 replay gates')
  require(totals['archives']==122 and totals['head_payloads']==488 and totals['capture_payloads']==98,'Complete586 actual payload inventory')
  require(0<d['wall_seconds']<=1260 and 0<d['max_allocated_bytes']<=d['max_reserved_bytes'],'Finite actual engineering resources')
 elif failures:require(len(failures)==1 and failures[0]['forward_id']==n==d['full_forward_attempts'],'Stop immediately at original failed numerical gate')
 return {'schema':'flow_r21_tail_fine_independent_execution_audit_v1','status':'PASS' if complete else 'PASS_PREFIX_EVIDENCE_ONLY','quality_analysis_allowed':complete,
  'admission':'ADMITTED_T8_K1_DOMAIN120_ENDPOINTS_ONLY' if complete else 'NOT_ADMITTED_INCOMPLETE_OR_NUMERICAL_FAILURE',
  'source_result_sha256':sha(path),'source_freeze_sha256':ROOT_FREEZE,'auditor_sha256':sha(__file__),'pre_result_freeze_sha256':sha(HERE/'PRE_RESULT_FREEZE_V1.json'),
  'source_pins':sources,'source_state':state,'counts':dict(totals),'numerical_checks':checks,'predictions':predictions,
  'input_evidence':d['data_evidence'],'runtime':d['runtime'],'resources':{k:d.get(k) for k in ('wall_seconds','max_allocated_bytes','max_reserved_bytes')},
  'limitations':['Independent compact approximate graph. R16 expanded equivalence remains failed; no R17/R18 quality or hardware saving transferred.',
   'Original PT independently reconstructed; per-forward full-state closure comes from frozen runtime assertions, not saved full states or independent model replays.',
   'Saved heads and native/FF/firstT8 raw tensors verified; other compact actions retain target hashes, not full target raw payloads. Internal BN hidden kernel moments not replayed.',
   'Every compact call keeps full-domain BN and expansion/gather; counters are not accelerator cycles or a complete-network speedup.',
   'No GT/AEE read here; no new neural forward/GPU. H8 is exposed development, not independent confirmation.']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'Exclusive new audit output')
 for f,digest in read(HERE/'PRE_RESULT_FREEZE_V1.json')['source_sha256'].items():require(sha(f)==digest,'Frozen auditor source')
 out=audit(a.result)
 with a.output.open('x') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({k:out[k] for k in ('status','quality_analysis_allowed','source_result_sha256','counts')}))
if __name__=='__main__':main()
