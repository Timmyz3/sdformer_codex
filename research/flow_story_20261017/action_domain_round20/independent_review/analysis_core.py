"""R20 fixed five-layout metric algebra; no learned strategy or budget winner."""
import math

ARMS=('native','FF','CC','F12','T8')
REGIONS=('all_valid','B','L','B_and_L','B_only','L_only_B_known','neither_B_known','B_unknown',
 'voxel_nonzero','voxel_zero','B_voxel_nonzero','B_voxel_zero','L_voxel_nonzero','L_voxel_zero',
 'B_unknown_voxel_nonzero','B_unknown_voxel_zero')

def require(ok,message):
    if not ok:raise ValueError(message)

def validate(m):
    n,e,a=m['valid_pixels'],m['error_sum'],m['AEE']
    require(type(n) is int and n>=0 and math.isfinite(e) and e>=0,'Finite nonnegative metric/support')
    require((a is None)==(n==0),'Empty is null')
    require((n==0 and e==0) or (n>0 and math.isfinite(a) and a>=0 and math.isclose(a,e/n,abs_tol=1e-10,rel_tol=1e-12)),'AEE sum/count')

def aggregate(rows):
    for r in rows:validate(r)
    good=[r for r in rows if r['valid_pixels']>0];n=sum(r['valid_pixels'] for r in rows);e=math.fsum(r['error_sum'] for r in rows)
    return {'valid_pixels':n,'error_sum':e,'frames_with_region':len(good),'empty_region_frames':len(rows)-len(good),
      'frame_mean_AEE':math.fsum(r['AEE'] for r in good)/len(good) if good else None,'pixel_weighted_AEE':e/n if n else None}

def protection(candidate,native):
    out={}
    for region,limit in (('all_valid',.03),('B',.05),('L',.05)):
        c,n=candidate[region],native[region];validate(c);validate(n)
        require(c['valid_pixels']==n['valid_pixels'],'Same candidate/reference support')
        good=n['valid_pixels']>0;cv,nv=c['AEE'],n['AEE']
        out[region]={'candidate_AEE':cv,'native_AEE':nv,'support':n['valid_pixels'],'positive_support':good,'limit':limit,
          'relative_change':cv/nv-1 if good and nv!=0 else None,'pass':good and cv<=nv*(1+limit)}
    return {'regions':out,'all_three_pass':all(r['pass'] for r in out.values())}

def macro_as_metric(s):
    # Synthetic error_sum preserves validation while the actual aggregate sum
    # stays separately reported in summaries. Protection uses macro AEE only.
    return {r:{'AEE':x['frame_mean_AEE'],'valid_pixels':x['valid_pixels'],
      'error_sum':x['frame_mean_AEE']*x['valid_pixels'] if x['valid_pixels'] else 0.} for r,x in s.items()}

def analyze(frames):
    require(len(frames)==len({f['target_id'] for f in frames})==len({f['sequence'] for f in frames})==8,'Eight fixed frame/sequences')
    for f in frames:
        require(set(f['arms'])==set(ARMS),'All five actual layouts')
        for arm in ARMS:
            require(set(f['arms'][arm])==set(REGIONS),'All sixteen regions')
            for reg in REGIONS:
                validate(f['arms'][arm][reg]);require(f['arms'][arm][reg]['valid_pixels']==f['arms']['native'][reg]['valid_pixels'],'Paired actual support')
    scopes=['overall']+[f['sequence'] for f in frames]
    summaries={a:{scope:{r:aggregate([f['arms'][a][r] for f in frames if scope=='overall' or f['sequence']==scope]) for r in REGIONS} for scope in scopes} for a in ARMS}
    guards={}
    for arm in ARMS[1:]:
        rows=[{'target_id':f['target_id'],'sequence':f['sequence'],**protection(f['arms'][arm],f['arms']['native'])} for f in frames]
        overall=protection(macro_as_metric(summaries[arm]['overall']),macro_as_metric(summaries['native']['overall']))
        guards[arm]={'overall_frame_mean':overall,'frames':rows,'overall_and_all_frames':overall['all_three_pass'] and all(x['all_three_pass'] for x in rows)}
    contrasts={
      'CC_added_damage':{'CC':1,'native':-1},'F12_gain_vs_CC':{'CC':1,'F12':-1},
      'T8_gain_vs_CC':{'CC':1,'T8':-1},'FF_minus_native':{'FF':1,'native':-1},
      'layout_nonadditivity':{'FF':1,'F12':-1,'T8':-1,'CC':1}}
    diagnostics={}
    for name,coefs in contrasts.items():
        values=[]
        for f in frames:
            regs={}
            for r in REGIONS:
                n=f['arms']['native'][r]['valid_pixels'];e=math.fsum(k*f['arms'][a][r]['error_sum'] for a,k in coefs.items())
                regs[r]={'valid_pixels':n,'signed_error_sum_contrast':e,'signed_AEE_contrast':e/n if n else None}
            values.append({'target_id':f['target_id'],'sequence':f['sequence'],'regions':regs})
        summary={}
        for r in REGIONS:
            good=[v['regions'][r] for v in values if v['regions'][r]['valid_pixels']]
            n=sum(x['valid_pixels'] for x in good);e=math.fsum(x['signed_error_sum_contrast'] for x in good)
            summary[r]={'valid_pixels':n,'frames_with_region':len(good),'empty_region_frames':8-len(good),
              'frame_mean_signed_contrast':math.fsum(x['signed_AEE_contrast'] for x in good)/len(good) if good else None,
              'pixel_weighted_signed_contrast':e/n if n else None}
        diagnostics[name]={'coefficients':coefs,'frames':values,'overall':summary}
    return {'frames':frames,'summaries':summaries,'protection_vs_native':guards,'signed_layout_diagnostics':diagnostics,
      'frame_region_cells':640,'summary_cells':720,'quality_prediction_count':40,
      'scope':'Fixed unequal-cost layout diagnostics. No same-budget winner, causal BN/motion attribution, learned flow increment or unseen-scene confirmation.'}
