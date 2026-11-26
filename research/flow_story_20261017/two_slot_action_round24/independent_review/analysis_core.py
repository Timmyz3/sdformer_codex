"""Independent fixed fixed-H1-two-slot+k1 feasibility; no selection or fitting."""
import math
SINGLETONS=tuple(f'fine_{r}_{c}' for r in range(3) for c in range(4))
ARMS=('native','FF','T2')+SINGLETONS
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
def analyze(frames):
    require(len(frames)==len({f['target_id'] for f in frames})==len({f['sequence'] for f in frames})==8,'Fixed eight distinct frames/sequences')
    for f in frames:
        require(set(f['arms'])==set(ARMS),'All15 actual quality endpoints')
        for a in ARMS:
            require(set(f['arms'][a])==set(REGIONS),'All16 regions')
            for r in REGIONS:
                validate(f['arms'][a][r]);require(f['arms'][a][r]['valid_pixels']==f['arms']['native'][r]['valid_pixels'],'Paired full support')
    scopes=['overall']+[f['sequence'] for f in frames]
    summaries={a:{s:{r:aggregate([f['arms'][a][r] for f in frames if s=='overall' or f['sequence']==s]) for r in REGIONS} for s in scopes} for a in ARMS}
    gates={a:dict(overall_frame_mean=gate(macro_as_metric(summaries[a]['overall']),macro_as_metric(summaries['native']['overall'])),
      frames=[dict(target_id=f['target_id'],sequence=f['sequence'],**gate(f['arms'][a],f['arms']['native'])) for f in frames]) for a in ARMS[1:]}
    feasibility=[];diagnostics=[]
    for f in frames:
        actions={a:gate(f['arms'][a],f['arms']['native']) for a in SINGLETONS};good=[a for a in SINGLETONS if actions[a]['all_three_pass']]
        feasibility.append(dict(target_id=f['target_id'],sequence=f['sequence'],T2=gate(f['arms']['T2'],f['arms']['native']),FF=gate(f['arms']['FF'],f['arms']['native']),singleton_actions=actions,feasible_singleton_count=len(good),all_feasible_singleton_ids=good))
        diagnostics.append(dict(target_id=f['target_id'],sequence=f['sequence'],T2_added_damage=contrast(f['arms']['T2'],f['arms']['native']),
          singleton_signed_gain={a:contrast(f['arms']['T2'],f['arms'][a]) for a in SINGLETONS}))
    zero=[f['target_id'] for f in feasibility if f['feasible_singleton_count']==0]
    return dict(frames=frames,summaries=summaries,protection_vs_native=gates,feasibility=feasibility,signed_action_diagnostics=diagnostics,
      zero_feasible_singleton_frames=zero,all_eight_have_feasible_singleton=not zero,
      resource_decision='PROCEED_FIXED_FIT16_DEV8_DEVELOPMENT_SCREEN' if not zero else 'STOP_THIS_T2_K1_SELECTOR_FAMILY',
      frame_region_cells=1920,summary_cells=2160,quality_prediction_count=120,singleton_actions_checked=96,T2_actions_checked=8,
      scope='GT-dependent fixed-domain existence only, not deployable selection, same-budget algorithm victory, unseen confirmation or paper success or independent confirmation.')
