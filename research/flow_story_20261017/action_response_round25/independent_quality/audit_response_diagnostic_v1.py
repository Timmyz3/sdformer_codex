#!/usr/bin/env python3
"""Independent post-hoc R24 response audit. No model, training or new policy.

Only the immutable original-array loader and region-mask definition are reused.
Response reductions, errors, rankings and summaries below do not import the
producer diagnostic. NumPy FP32 response/pooling is its explicitly fixed graph.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
DATE=HERE.parents[1]
REPO=HERE.parents[3]
R24=DATE/'two_slot_action_round24'
R13=REPO/'research/flow_story_20261016/coverage_recovery_round13'
PINS={
 'diagnostic_source':(HERE.parent/'response_representation_diagnostic.py','9c670f3d4c26bb9965f454f1048fa3cc4916817f57f7cac3e976ed3cdd578951'),
 'result':(R24/'actual_v1/run_v1/results.json','e4e45b5a0608aa7c388ef196e3a84cb1b12e2894d0eada806fd9b56860692ef9'),
 'quality':(R24/'independent_review/ACTUAL_RAW_QUALITY_V1.json','43bdf87024a7c4aeb659cceb3cae285bff5f97dc58e65afd1143f5c4a1f136a4'),
 'execution_audit':(R24/'independent_execution_audit/ACTUAL_EXECUTION_AUDIT_V1.json','1f575892c126fbef50939e4a395602d1af1b755728b0e5ad6393ad6c320d992f'),
 'metric':(R13/'independent_quality_review/reduce_raw_quality.py','26a2e5c97b988f10685e1c440a11506354d9591cb9b36a567292abb6d0348088'),
 'h8':(R13/'next_action_headroom/implementation_h0/h8_manifest.json','1083d2a3f9b940aee0ebf271cb30b062eeed7bc4bcbf51429d573172573569f5'),
 'snapshot':(R13/'next_action_headroom/h8_input_snapshot_v1/receipt.json','7f0cc5d74ce171aaa82cc602ec877e8fd58456e7481a9a158bc397cb8efa7b5d'),
 'regions':(R13/'preparation/regions_v1.json','2e4934f6d82663f16ca1ae6c3aea238a31e699d4f49916052f6ea6b5e9c1d41b'),
}
ARMS=tuple(f'fine_{r}_{c}' for r in range(3) for c in range(4))
METHODS=('pooled_oracle_gain','full_oracle_gain','GT_direction_taylor_gain','response_magnitude','all_pixel_response_magnitude')
COUNT=0
MAX_DIFF=0.0

def require(ok,msg):
 if not ok: raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def check(a,b,path):
 global COUNT,MAX_DIFF
 if isinstance(a,dict):
  require(set(a)==set(b),path+' keys')
  for k in a:check(a[k],b[k],path+'/'+k)
 elif isinstance(a,(list,tuple)):
  require(len(a)==len(b),path+' length')
  for i,(x,y) in enumerate(zip(a,b)):check(x,y,path+'/'+str(i))
 elif a is None or isinstance(a,(str,bool,int)):
  require(a==b,path+' exact');COUNT+=1
 else:
  require(np.isfinite(a) and np.isfinite(b),path+' finite')
  d=abs(float(a)-float(b));MAX_DIFF=max(MAX_DIFF,d)
  require(d<=1e-10+1e-9*abs(float(b)),path+' numerical mismatch '+str((a,b)));COUNT+=1

def mean_error(pred,gt,mask):
 n=int(np.count_nonzero(mask))
 if not n:return None
 e=np.asarray(pred,dtype=np.float64)-np.asarray(gt,dtype=np.float64)
 err=np.sqrt(e[0]*e[0]+e[1]*e[1])
 return float(np.sum(err[mask],dtype=np.float64)/n)
def ranks(v):
 order=np.argsort(np.asarray(v),kind='stable');r=np.empty(len(v),np.float64);i=0
 while i<len(v):
  j=i+1
  while j<len(v) and v[order[j]]==v[order[i]]:j+=1
  r[order[i:j]]=(i+j-1)/2;i=j
 return r
def statistics(y,s):
 y=np.asarray(y,np.float64);s=np.asarray(s,np.float64)
 a=ranks(y);b=ranks(s);a-=a.mean();b-=b.mean()
 den=np.sqrt(np.dot(a,a)*np.dot(b,b))
 rho=float(np.dot(a,b)/den) if den>0 else None
 selected=int(np.flatnonzero(s==np.max(s))[0]);best=int(np.flatnonzero(y==np.max(y))[0])
 return dict(spearman=rho,selected_index=selected,oracle_best_index=best,top1_match=selected==best,
  true_gain_at_selected=float(y[selected]),regret_AEE=float(y[best]-y[selected]),gain_mae=float(np.sum(np.abs(y-s))/len(y)))

def run(diagnostic):
 for name,(p,h) in PINS.items():require(sha(p)==h,'Frozen source '+name)
 d=read(diagnostic);result=read(PINS['result'][0]);q=read(PINS['quality'][0]);audit=read(PINS['execution_audit'][0])
 require(d['source_sha256']==PINS['diagnostic_source'][1] and d['result_sha256']==PINS['result'][1] and d['quality_sha256']==PINS['quality'][1],'Producer source bindings')
 require(audit['status']=='PASS' and audit['quality_analysis_allowed'] and audit['source_result_sha256']==PINS['result'][1],'Actual audited complete R24')
 require(q['resource_decision']=='STOP_THIS_T2_K1_SELECTOR_FAMILY' and d['R24_STOP_remains'] is True,'No reopened family')
 require(d['actual_new_model_forwards']==d['optimizer_steps']==0 and d['runtime_policy'] is False,'Post-hoc only')
 spec=importlib.util.spec_from_file_location('immutable_original_masks',PINS['metric'][0]);metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
 members=read(PINS['h8'][0])['members'];ids=[v['target_id'] for v in members]
 snap=PINS['snapshot'][0].parent;inputs,evidence=metric.load_snapshot(snap,PINS['snapshot'][0],ids)
 cfg=read(PINS['regions'][0]);truth={f['target_id']:f for f in q['frames']}
 require([f['target_id'] for f in d['frames']]==ids,'All8 in fixed order')
 records={(v['target_id'],v['arm']):v for v in result['forward_records'] if v['quality_endpoint'] and v['arm']!='FF'}
 require(len(records)==112,'Native/T2/all12 for all8')
 listed={v['file']:v['sha256'] for v in d['archives']};require(len(listed)==112,'Complete producer file ledger')
 predictions={}
 for key,r in records.items():
  p=PINS['result'][0].parent/r['prediction_file'];require(sha(p)==r['prediction_file_sha256']==listed[p.name],'Real archive SHA')
  require(r['input_evidence']==evidence[key[0]],'Original input binding')
  with np.load(p,allow_pickle=False) as z:a=z['head3']
  require(a.shape==(1,2,480,640) and a.dtype==np.float32 and np.isfinite(a).all(),'Head graph')
  require(hashlib.sha256(a.tobytes()).hexdigest()==r['head_records'][3]['sha256'],'Actual head payload')
  predictions[key]=a[0]
 frames=[];original_metric_checks=0;zero_residual_valid=0
 for df in d['frames']:
  tid=df['target_id'];src=inputs[tid];gt=src['gt'];gt64=gt.astype(np.float64)
  masks=metric.masks_from_original(gt,src['mask'],src['events'],cfg['boundary_threshold_pixels'],cfg['large_displacement_threshold_pixels'])
  bg=predictions[tid,'T2'];base={k:mean_error(bg,gt,v) for k,v in masks.items()}
  e=bg.astype(np.float64)-gt64;en=np.sqrt(e[0]*e[0]+e[1]*e[1]);zero_residual_valid+=int(np.count_nonzero((en==0)&masks['all_valid']))
  for arm in ('native','T2')+ARMS:
   for name,mask in masks.items():
    actual={'AEE':mean_error(predictions[tid,arm],gt,mask),'valid_pixels':int(mask.sum())}
    old=truth[tid]['arms'][arm][name]
    check(actual,{'AEE':old['AEE'],'valid_pixels':old['valid_pixels']},'old quality/'+tid+'/'+arm+'/'+name);original_metric_checks+=1
  actions=[]
  for arm in ARMS:
   f=predictions[tid,arm];delta=np.subtract(f,bg,dtype=np.float32)
   coarse=delta.reshape(2,30,16,40,16).sum(axis=(2,4),dtype=np.float32)/np.float32(256)
   restored=np.add(bg,np.repeat(np.repeat(coarse,16,axis=1),16,axis=2),dtype=np.float32)
   full=np.add(bg,delta,dtype=np.float32)
   dv=delta.astype(np.float64);norm=np.sqrt(dv[0]*dv[0]+dv[1]*dv[1]);taylor=-(e[0]*dv[0]+e[1]*dv[1])/np.maximum(en,1e-12)
   regions={}
   for name,mask in masks.items():
    n=int(mask.sum())
    if n:
     actual=mean_error(f,gt,mask)
     regions[name]=dict(valid_pixels=n,true_gain=base[name]-actual,
      pooled_oracle_gain=base[name]-mean_error(restored,gt,mask),full_oracle_gain=base[name]-mean_error(full,gt,mask),
      GT_direction_taylor_gain=float(np.sum(taylor[mask],dtype=np.float64)/n),response_magnitude=float(np.sum(norm[mask],dtype=np.float64)/n))
     old=truth[tid]['arms']['T2'][name]['AEE']-truth[tid]['arms'][arm][name]['AEE']
     check(regions[name]['true_gain'],old,'old signedgain/'+tid+'/'+arm+'/'+name)
    else:regions[name]=dict(valid_pixels=0,true_gain=None,pooled_oracle_gain=None,full_oracle_gain=None,GT_direction_taylor_gain=None,response_magnitude=None)
   actions.append(dict(arm=arm,regions=regions,all_pixel_response_magnitude=float(np.sum(norm,dtype=np.float64)/norm.size)))
  diags={}
  for name,mask in masks.items():
   if not np.any(mask):diags[name]=None;continue
   y=[a['regions'][name]['true_gain'] for a in actions]
   diags[name]={method:statistics(y,[a[method] if method=='all_pixel_response_magnitude' else a['regions'][name][method] for a in actions]) for method in METHODS}
  reconstructed=dict(target_id=tid,actions=actions,diagnostics=diags);check(reconstructed,df,'producer frame/'+tid);frames.append(reconstructed)
 summary={}
 for name in metric.REGIONS:
  good=[f['diagnostics'][name] for f in frames if f['diagnostics'][name] is not None];summary[name]={}
  for method in METHODS if good else ():
   z=[x[method] for x in good];rho=[x['spearman'] for x in z if x['spearman'] is not None]
   summary[name][method]=dict(frames=len(z),top1_matches=sum(x['top1_match'] for x in z),mean_spearman=float(sum(rho)/len(rho)) if rho else None,
     mean_regret_AEE=float(sum(x['regret_AEE'] for x in z)/len(z)),mean_gain_mae=float(sum(x['gain_mae'] for x in z)/len(z)))
 check(summary,d['summaries'],'all16 summaries')
 return dict(schema='flow_r25_independent_posthoc_response_audit_v1',status='PASS',auditor_sha256=sha(__file__),diagnostic_sha256=sha(diagnostic),
  pins={k:dict(path=str(p),sha256=h) for k,(p,h) in PINS.items()},counts=dict(frames=8,actions=96,region_action_cells=1536,raw_archives=112,original_metric_cells=original_metric_checks,checked_scalar_fields=COUNT),
  max_scalar_absolute_difference=MAX_DIFF,comparison_tolerance=dict(atol=1e-10,rtol=1e-9),exact_zero_baseline_residual_valid_pixels=zero_residual_valid,
  summaries=summary,L_support=[int(truth[tid]['arms']['T2']['L']['valid_pixels']) for tid in ids],
  scope='Independent raw response/error/rank recomputation; original immutable loader and region masks reused. No producer diagnostic functions imported.',
  R24_family_remains_stopped=True,deployable_policy_demonstrated=False,new_model_forwards=0,optimizer_steps=0)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--diagnostic',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'New audit output only')
 out=run(a.diagnostic);a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps({k:out[k] for k in ('status','counts','max_scalar_absolute_difference','L_support')}))
