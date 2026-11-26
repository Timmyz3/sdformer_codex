"""Fixed pilot123 descriptive mathematics, independent of experiment runner.

Inputs are 120 independently measured region dictionaries, not predictions
from another analysis. No policy fitting, tile selection or hypothesis test.
"""
import math
import statistics

ARMS=('native','FF','CC')+tuple(f'fine_{r}_{c}' for r in range(3) for c in range(4))
REGIONS=('all_valid','B','L','B_and_L','B_only','L_only_B_known','neither_B_known',
         'B_unknown','voxel_nonzero','voxel_zero','B_voxel_nonzero','B_voxel_zero',
         'L_voxel_nonzero','L_voxel_zero','B_unknown_voxel_nonzero','B_unknown_voxel_zero')

def require(ok,message):
    if not ok: raise ValueError(message)

def aggregate(values):
    present=[v for v in values if v['valid_pixels']>0]
    pixels=sum(v['valid_pixels'] for v in values)
    error=sum(v['error_sum'] for v in values)
    return {'frames_with_region':len(present),'empty_region_frames':len(values)-len(present),
            'valid_pixels':pixels,'error_sum':error,
            'frame_mean_AEE':sum(v['AEE'] for v in present)/len(present) if present else None,
            'pixel_weighted_AEE':error/pixels if pixels else None}

def delta(candidate,reference):
    require((candidate is None)==(reference is None),'Paired valid-GT support')
    return None if candidate is None else candidate-reference

def protection(candidate,reference):
    out={}
    for name,limit in (('all_valid',.03),('B',.05),('L',.05)):
        a,b=candidate[name]['AEE'],reference[name]['AEE']
        require(candidate[name]['valid_pixels']==reference[name]['valid_pixels'],'Fixed output support')
        require((a is None)==(b is None),'Matched empty region')
        out[name]={'candidate':a,'reference_native':b,'support':candidate[name]['valid_pixels'],
                   'limit':limit,'relative_change':a/b-1 if b not in (None,0) else None,
                   'positive_support':a is not None,'pass':None if a is None else a<=b*(1+limit)}
    return {'regions':out,'all_three_positive_support':all(v['positive_support'] for v in out.values()),
            'diagnostic_all_three_pass':all(v['positive_support'] and v['pass'] for v in out.values()),
            'scope':'Diagnostic protection only, not full-network compression/deployment admission.'}

def analyze(frames):
    require(len(frames)==8 and len({f['target_id'] for f in frames})==8,'Complete H8')
    sequences=[f['sequence'] for f in frames]
    require(len(set(sequences))==8,'One fixed frame per each of eight sequences')
    for f in frames:
        require(tuple(f['arms'])==ARMS,'All fixed actions in order')
        for arm in ARMS:
            require(set(f['arms'][arm])==set(REGIONS),'Complete sixteen regions')
            for name in REGIONS:
                m=f['arms'][arm][name];n=m['valid_pixels'];e=m['error_sum'];v=m['AEE']
                require(type(n) is int and n>=0 and math.isfinite(e) and e>=0,'Finite valid region')
                require((v is None)==(n==0),'Empty regions stay null')
                if n:require(math.isfinite(v) and v>=0 and math.isclose(v,e/n,rel_tol=1e-12,abs_tol=1e-10),'AEE definition')
                require(n==f['arms']['native'][name]['valid_pixels'],'Same all-output support for every action')
        require(f['arms']['native']['all_valid']['valid_pixels']>0,'Positive all-GT frame')
    scopes={'overall':frames,**{s:[f for f in frames if f['sequence']==s] for s in sequences}}
    summaries={arm:{scope:{region:aggregate([f['arms'][arm][region] for f in group])
               for region in REGIONS} for scope,group in scopes.items()} for arm in ARMS}
    action_values=[]
    for f in frames:
        values={}
        for region in REGIONS:
            e=lambda arm:f['arms'][arm][region]['AEE']
            gains=[delta(e('CC'),e(arm)) for arm in ARMS[3:]]
            present=[g for g in gains if g is not None]
            values[region]={'native_AEE':e('native'),'CC_added_damage':delta(e('CC'),e('native')),
                'FF_minus_native':delta(e('FF'),e('native')),'CC_minus_FF':delta(e('CC'),e('FF')),
                'singleton_gains':dict(zip(ARMS[3:],gains)),
                'positive':sum(g>0 for g in present),'zero':sum(g==0 for g in present),
                'negative':sum(g<0 for g in present),'minimum':min(present) if present else None,
                'maximum':max(present) if present else None,'median':statistics.median(present) if present else None,
                'range':max(present)-min(present) if present else None}
        action_values.append({'target_id':f['target_id'],'sequence':f['sequence'],'regions':values,
            'protection_vs_native':{arm:protection(f['arms'][arm],f['arms']['native']) for arm in ARMS[1:]}})
    overall_values={region:{'CC_added_damage':delta(summaries['CC']['overall'][region]['frame_mean_AEE'],summaries['native']['overall'][region]['frame_mean_AEE']),
       'singleton_gains':{arm:delta(summaries['CC']['overall'][region]['frame_mean_AEE'],summaries[arm]['overall'][region]['frame_mean_AEE']) for arm in ARMS[3:]}}
       for region in REGIONS}
    return {'frames':frames,'summaries':summaries,'action_values':action_values,'overall_action_values':overall_values,
            'frame_region_cells':8*15*16,'summary_cells':15*9*16,'actual_prediction_count':120,
            'scope':'Conditional signed singleton values over whole valid output support. Ninety-six labels share eight frames; no pair/oracle/selector/independent-generalization claim.'}
