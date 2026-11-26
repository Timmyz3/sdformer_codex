"""R27 preregistered64-frame ordinary recovery quality; pure scalar math only."""
import math
STATES=('original_native','initial_Q8','initial_Q6','recovered_native','recovered_Q8','recovered_Q6')
REGIONS=('all_valid','B','L','B_and_L','B_only','L_only_B_known','neither_B_known','B_unknown','voxel_nonzero','voxel_zero','B_voxel_nonzero','B_voxel_zero','L_voxel_nonzero','L_voxel_zero','B_unknown_voxel_nonzero','B_unknown_voxel_zero')
BANDS=('empty','1_to_31','32_to_255','256_plus')
def require(ok,msg):
 if not ok:raise ValueError(msg)
def validate(v):
 n,e,a=v['valid_pixels'],v['error_sum'],v['AEE'];require(type(n)is int and n>=0 and math.isfinite(e) and e>=0,'Finite support/error')
 require((a is None)==(n==0),'Empty is null')
 require((n==0 and e==0) or (n>0 and math.isfinite(a) and a>=0 and math.isclose(a,e/n,rel_tol=1e-12,abs_tol=1e-10)),'AEE=sum/count')
def aggregate(rows):
 for r in rows:validate(r)
 n=sum(r['valid_pixels'] for r in rows);e=math.fsum(r['error_sum'] for r in rows);good=[r for r in rows if r['valid_pixels']]
 return dict(valid_pixels=n,error_sum=e,frame_count=len(rows),positive_frames=len(good),empty_frames=len(rows)-len(good),pixel_weighted_AEE=e/n if n else None,frame_mean_AEE=math.fsum(r['AEE'] for r in good)/len(good) if good else None)
def band(n):return 'empty' if n==0 else '1_to_31' if n<32 else '32_to_255' if n<256 else '256_plus'
def difference(c,r):return None if c is None or r is None else c-r
def ratio(c,r):return None if c is None or r is None or r==0 else c/r-1
def quantiles(vals):
 x=sorted(v for v in vals if v is not None)
 if not x:return dict(count=0,p50=None,p90=None,max=None)
 def q(frac):
  i=(len(x)-1)*frac;lo=int(i);hi=min(lo+1,len(x)-1);return x[lo]+(x[hi]-x[lo])*(i-lo)
 return dict(count=len(x),p50=q(.5),p90=q(.9),max=x[-1])
def one_gate(c,r,limit,support):
 positive=support>0 and c is not None and r is not None
 return dict(candidate_AEE=c,reference_AEE=r,relative_change=ratio(c,r),signed_delta_AEE=difference(c,r),valid_pixels=support,limit=limit,positive_support=positive,**{'pass':bool(positive and c<=r*(1+limit))})
def comparison(frames,summaries,candidate,reference,seqs):
 c=summaries[candidate]['overall'];r=summaries[reference]['overall'];g={}
 for key,region,measure,limit in (('allGT_frame_mean','all_valid','frame_mean_AEE',.03),('allGT_pixel_weighted','all_valid','pixel_weighted_AEE',.03),('B_pixel_weighted','B','pixel_weighted_AEE',.05),('L_pixel_weighted','L','pixel_weighted_AEE',.05)):
  require(c[region]['valid_pixels']==r[region]['valid_pixels'],'Same aggregate support');g[key]=one_gate(c[region][measure],r[region][measure],limit,r[region]['valid_pixels'])
 seqflags={}
 for region in ('B','L'):
  pairs=[(summaries[candidate][s][region]['pixel_weighted_AEE'],summaries[reference][s][region]['pixel_weighted_AEE']) for s in seqs]
  good=[(a,b) for a,b in pairs if a is not None and b is not None];ca=math.fsum(a for a,_ in good)/len(good) if good else None;ra=math.fsum(b for _,b in good)/len(good) if good else None
  seqflags[region]=dict(candidate_sequence_macro_AEE=ca,reference_sequence_macro_AEE=ra,positive_sequences=len(good),empty_sequences=len(seqs)-len(good),relative_change=ratio(ca,ra),signed_delta_AEE=difference(ca,ra),review_flag=bool(good and ca>ra*1.05),not_a_hard_gate=True)
 rows=[];high=[]
 for f in frames:
  regs={}
  for region in REGIONS:
   cc,rr=f['states'][candidate][region],f['states'][reference][region];n=rr['valid_pixels'];delta=difference(cc['AEE'],rr['AEE']);rel=ratio(cc['AEE'],rr['AEE']);limit=.03 if region=='all_valid' else .05
   flag=region in ('all_valid','B','L') and n>=256 and (cc['AEE']>rr['AEE']*1.2 or delta>1.)
   regs[region]=dict(valid_pixels=n,support_band=band(n),candidate_AEE=cc['AEE'],reference_AEE=rr['AEE'],signed_delta_AEE=delta,relative_change=rel,legacy_threshold_exceeded=bool(n and cc['AEE']>rr['AEE']*(1+limit)),legacy_limit=limit,high_support_tail_review_flag=flag)
   if flag:high.append(dict(target_id=f['target_id'],sequence=f['sequence'],region=region,**regs[region]))
  rows.append(dict(target_id=f['target_id'],sequence=f['sequence'],regions=regs))
 strata={}
 for region in REGIONS:
  strata[region]={}
  for b in BANDS:
   selected=[v['regions'][region] for v in rows if v['regions'][region]['support_band']==b]
   cmetrics=[f['states'][candidate][region] for f in frames if band(f['states'][reference][region]['valid_pixels'])==b];rmetrics=[f['states'][reference][region] for f in frames if band(f['states'][reference][region]['valid_pixels'])==b]
   strata[region][b]=dict(frame_count=len(selected),candidate=aggregate(cmetrics),reference=aggregate(rmetrics),signed_delta_AEE=quantiles([v['signed_delta_AEE'] for v in selected]),relative_change=quantiles([v['relative_change'] for v in selected]),legacy_threshold_exceedance_count=sum(v['legacy_threshold_exceeded'] for v in selected))
 return dict(candidate=candidate,reference=reference,aggregate_gates=g,aggregate_pass=all(v['pass'] for v in g.values()),sequence_balanced_review=seqflags,high_support_tail_review=high,frames=rows,support_strata=strata)

def analyze(frames):
 require(len(frames)==len({f['target_id'] for f in frames})==64,'Exactly64 unique newdev')
 seqs=list(dict.fromkeys(f['sequence'] for f in frames));require(len(seqs)==8 and all(sum(f['sequence']==s for f in frames)==8 for s in seqs),'8sequences x8 frames')
 for f in frames:
  require(set(f['states'])==set(STATES),'All6 initial/final states')
  for state in STATES:
   require(set(f['states'][state])==set(REGIONS),'Complete16 regions')
   for region in REGIONS:
    validate(f['states'][state][region]);require(f['states'][state][region]['valid_pixels']==f['states']['original_native'][region]['valid_pixels'],'Allstate common GT support')
 scopes=['overall']+seqs
 summaries={state:{s:{region:aggregate([f['states'][state][region] for f in frames if s=='overall' or f['sequence']==s]) for region in REGIONS} for s in scopes} for state in STATES}
 pairs=[('recovered_native','original_native'),('recovered_Q8','original_native'),('recovered_Q8','recovered_native'),('recovered_Q6','original_native'),('recovered_Q6','recovered_native'),('initial_Q8','original_native'),('initial_Q6','original_native'),('recovered_Q8','initial_Q8'),('recovered_Q6','initial_Q6'),('recovered_Q6','recovered_Q8')]
 comparisons={c+'__vs__'+r:comparison(frames,summaries,c,r,seqs) for c,r in pairs}
 passed=lambda c,r:comparisons[c+'__vs__'+r]['aggregate_pass']
 stable=passed('recovered_native','original_native');q8=passed('recovered_Q8','original_native') and passed('recovered_Q8','recovered_native');q6=passed('recovered_Q6','original_native') and passed('recovered_Q6','recovered_native')
 if not stable:decision='MATCHED_NATIVE_RECIPE_UNSTABLE'
 elif not q8:decision='Q8_RECOVERY_ANCHOR_NOT_QUALIFIED'
 elif not q6:decision='Q6_RECOVERY_NOT_QUALIFIED'
 else:decision='AGGREGATE_ORDINARY_RECOVERY_PASS_REVIEW_FLAGS_SEPARATE'
 return dict(schema='flow_r27_ordinary_recovery_scalar_quality_v1',frames=frames,summaries=summaries,comparisons=comparisons,native_recovery_stable=stable,Q8_dual_reference_pass=q8,Q6_dual_reference_pass=q6,resource_decision=decision,quality_prediction_count=384,frame_region_cells=6144,summary_cells=864,selector_learning_demonstrated=False,old_R26_failure_unchanged=True,scope='Prospective new64 development aggregate protocol only. High-support/sequence flags separate and fully disclosed; no single small-support frame veto. Same backbone-training scenes; not official test or unseen-scene confirmation.')
