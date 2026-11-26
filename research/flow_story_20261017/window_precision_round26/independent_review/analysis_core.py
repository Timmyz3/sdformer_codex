"""Independent fixed Q6-primary/Q4-diagnostic precision quality; no fitting."""
import math
ARMS=('native','Q8ref','Q8serial','Q6','Q4')
REGIONS=('all_valid','B','L','B_and_L','B_only','L_only_B_known','neither_B_known','B_unknown',
 'voxel_nonzero','voxel_zero','B_voxel_nonzero','B_voxel_zero','L_voxel_nonzero','L_voxel_zero',
 'B_unknown_voxel_nonzero','B_unknown_voxel_zero')
def require(ok,message):
    if not ok:raise ValueError(message)
def validate(m):
    n,e,a=m['valid_pixels'],m['error_sum'],m['AEE']
    require(type(n) is int and n>=0 and math.isfinite(e) and e>=0,'Finite nonnegative support/error')
    require((a is None)==(n==0),'Empty support remains null')
    require((n==0 and e==0) or (n>0 and math.isfinite(a) and a>=0 and math.isclose(a,e/n,abs_tol=1e-10,rel_tol=1e-12)),'AEE sum/count')
def aggregate(rows):
    for r in rows:validate(r)
    good=[r for r in rows if r['valid_pixels']>0];n=sum(r['valid_pixels'] for r in rows);e=math.fsum(r['error_sum'] for r in rows)
    return dict(valid_pixels=n,error_sum=e,frames_with_region=len(good),empty_region_frames=len(rows)-len(good),
      frame_mean_AEE=math.fsum(r['AEE'] for r in good)/len(good) if good else None,pixel_weighted_AEE=e/n if n else None)
def gate(candidate,native):
    out={}
    for r,limit in (('all_valid',.03),('B',.05),('L',.05)):
        c,n=candidate[r],native[r];validate(c);validate(n);require(c['valid_pixels']==n['valid_pixels'],'Identical GT support')
        positive=n['valid_pixels']>0;cv,nv=c['AEE'],n['AEE']
        out[r]=dict(valid_pixels=n['valid_pixels'],candidate_AEE=cv,native_AEE=nv,relative_change=cv/nv-1 if positive and nv!=0 else None,
          limit=limit,positive_support=positive,**{'pass':positive and cv<=nv*(1+limit)})
    return dict(regions=out,all_three_pass=all(x['pass'] for x in out.values()))
def macro_as_metric(summary):
    return {r:dict(AEE=v['frame_mean_AEE'],valid_pixels=v['valid_pixels'],error_sum=v['frame_mean_AEE']*v['valid_pixels'] if v['valid_pixels'] else 0.) for r,v in summary.items()}
def contrast(candidate,reference):
    out={}
    for r in REGIONS:
        c,n=candidate[r],reference[r];require(c['valid_pixels']==n['valid_pixels'],'Same contrast support')
        size=n['valid_pixels'];e=c['error_sum']-n['error_sum']
        out[r]=dict(valid_pixels=size,signed_error_sum=e,signed_AEE=e/size if size else None)
    return out

def compare_scope(frames,summaries,candidate,reference):
 overall=gate(macro_as_metric(summaries[candidate]['overall']),macro_as_metric(summaries[reference]['overall']))
 rows=[dict(target_id=f['target_id'],sequence=f['sequence'],**gate(f['arms'][candidate],f['arms'][reference])) for f in frames]
 return dict(reference=reference,candidate=candidate,overall_frame_mean=overall,frames=rows,
   all_frames_and_overall_pass=overall['all_three_pass'] and all(v['all_three_pass'] for v in rows))
def analyze(frames,*,serial8_exact):
 require(serial8_exact is True,'All8 independent raw bitserial8/integer8 exact numerical checks required')
 require(len(frames)==len({f['target_id'] for f in frames})==len({f['sequence'] for f in frames})==8,'Fixed8 H8 development frames/sequences')
 for f in frames:
  require(set(f['arms'])==set(ARMS),'All5 actual precision endpoints')
  for arm in ARMS:
   require(set(f['arms'][arm])==set(REGIONS),'Full16 regions')
   for r in REGIONS:
    validate(f['arms'][arm][r]);require(f['arms'][arm][r]['valid_pixels']==f['arms']['native'][r]['valid_pixels'],'Same original validGT support')
  require(f['arms']['Q8serial']==f['arms']['Q8ref'],'Exact original integer8/serial8 arrays give identical region metrics')
 scopes=['overall']+[f['sequence'] for f in frames]
 summaries={a:{s:{r:aggregate([f['arms'][a][r] for f in frames if s=='overall' or f['sequence']==s]) for r in REGIONS} for s in scopes} for a in ARMS}
 primary={a:compare_scope(frames,summaries,a,'native') for a in ARMS[1:]}
 diagnostic={a:compare_scope(frames,summaries,a,'Q8ref') for a in ('Q6','Q4')}
 feasibility=[]
 for f in frames:
  mode_gates={a:gate(f['arms'][a],f['arms']['native']) for a in ('Q6','Q4')}
  modes=[a for a in ('Q6','Q4') if mode_gates[a]['all_three_pass']]
  feasibility.append(dict(target_id=f['target_id'],sequence=f['sequence'],mode_gates=mode_gates,feasible_mode_count=len(modes),feasible_modes=modes))
 anchor=primary['Q8ref']['all_frames_and_overall_pass'];q6=primary['Q6']['all_frames_and_overall_pass'];q4=primary['Q4']['all_frames_and_overall_pass']
 zero=[f['target_id'] for f in feasibility if f['feasible_mode_count']==0]
 if not anchor:decision='STOP_Q8_DEPLOYMENT_ANCHOR_FAILED'
 elif q6:decision='PROCEED_FIXED_Q6_PRIMARY_ACTION_DEVELOPMENT_ONLY'
 elif q4:decision='STOP_Q6_PRIMARY_Q4_DIAGNOSTIC_ONLY_NO_AUTOMATIC_LEARNING'
 else:decision='STOP_Q6_PRIMARY_NO_ADMITTED_FIXED_MODE'
 return dict(frames=frames,summaries=summaries,primary_vs_FP32_native=primary,diagnostic_vs_Q8_reference=diagnostic,
  per_frame_two_mode_feasibility=feasibility,zero_feasible_mode_frames=zero,Q8_conversion_anchor_pass=anchor,
  serial8_exact=True,Q6_primary_pass=q6,Q4_diagnostic_pass=q4,resource_decision=decision,
  Q6_learning_action_candidate=anchor and q6,Q4_promoted_to_primary=False,
  signed_changes_vs_native=[dict(target_id=f['target_id'],**{a:contrast(f['arms'][a],f['arms']['native']) for a in ARMS[1:]}) for f in frames],
  signed_changes_vs_Q8=[dict(target_id=f['target_id'],**{a:contrast(f['arms'][a],f['arms']['Q8ref']) for a in ('Q6','Q4')}) for f in frames],
  frame_region_cells=640,summary_cells=720,quality_prediction_count=40,numerical_replays_excluded=2,
  scope='Fixed precision action feasibility only. H8 is reused development; neither selector advantage nor flow-specific gain nor net cycles/RTL claim. Q6 primary/Q4 diagnostic fixed before quality.')
