"""Independent fixed G0 score mathematics and actual-endpoint expectations."""
import math
import numpy as np

SCORES=('amplitude_abs_mean','feature_rms','pooling_residual_mse',
 'spatial_neighbor_abs_difference','temporal_neighbor_abs_difference',
 'psn1_activity','psn1_spatial_neighbor_hamming')
TILES=tuple((r,c) for r in range(3) for c in range(4))
REGIONS=('all_valid','B','L','B_and_L','B_only','L_only_B_known','neither_B_known',
 'B_unknown','voxel_nonzero','voxel_zero','B_voxel_nonzero','B_voxel_zero',
 'L_voxel_nonzero','L_voxel_zero','B_unknown_voxel_nonzero','B_unknown_voxel_zero')

def require(ok,message):
    if not ok: raise ValueError(message)

def independently_score(u,sn1):
    require(u.shape==sn1.shape==(10,1,30,40,384),'Full native tensor geometry')
    require(np.isfinite(u).all() and np.isfinite(sn1).all(),'Finite captures')
    require(np.all((sn1==0)|(sn1==1)),'Binary native PSN1')
    out=[]
    for r,c in TILES:
        x=np.asarray(u[:,:,r*9:(r+1)*9,c*9:(c+1)*9,:],dtype=np.float64)
        s=sn1[:,:,r*9:(r+1)*9,c*9:(c+1)*9,:]
        restored=np.empty_like(x)
        # Explicit cells provide an independent implementation of pool/restore.
        for y in (0,3,6):
            for z in (0,3,6):
                cell=x[:,:,y:y+3,z:z+3,:]
                mean=cell.sum(axis=2,keepdims=True).sum(axis=3,keepdims=True)/9.0
                restored[:,:,y:y+3,z:z+3,:]=mean
        vals=(float(np.abs(x).mean()),float(np.sqrt((x*x).mean())),
          float(((x-restored)**2).mean()),
          float((np.abs(x[:,:,1:,:,:]-x[:,:,:-1,:,:]).mean()+np.abs(x[:,:,:,1:,:]-x[:,:,:,:-1,:]).mean())/2),
          float(np.abs(x[1:]-x[:-1]).mean()),float(s.astype(np.float64).mean()),
          float(((s[:,:,1:,:,:]!=s[:,:,:-1,:,:]).mean()+(s[:,:,:,1:,:]!=s[:,:,:,:-1,:]).mean())/2))
        require(all(math.isfinite(v) and v>=0 for v in vals),'Nonnegative finite scores')
        out.append({'tile':[r,c],**dict(zip(SCORES,vals))})
    return out

def select(scores,name):
    require([tuple(v['tile']) for v in scores]==list(TILES),'Complete row-col score order')
    require(name in SCORES,'Fixed score name')
    return max(range(12),key=lambda i:(scores[i][name],-i))

def policy_weights(scores):
    out={}
    for name in SCORES:
        chosen=select(scores,name);out[name]=[float(i==chosen) for i in range(12)]
    for i,(r,c) in enumerate(TILES):out[f'fixed_{r}_{c}']=[float(i==j) for j in range(12)]
    out['uniform_expectation']=[1/12]*12
    return out

def validate_metric(m):
    n,e,a=m['valid_pixels'],m['error_sum'],m['AEE']
    require(type(n) is int and n>=0 and math.isfinite(e) and e>=0,'Region support/error')
    require((a is None)==(n==0),'Null iff empty')
    require((n==0 and e==0) or (n>0 and math.isfinite(a) and a>=0 and math.isclose(a,e/n,abs_tol=1e-10,rel_tol=1e-12)),'AEE=sum/count')

def expectation(metrics,weights):
    require(len(metrics)==len(weights)==12,'Twelve actual singleton endpoints')
    require(all(math.isfinite(w) and w>=0 for w in weights) and math.isclose(sum(weights),1,abs_tol=1e-14),'Probability distribution')
    for m in metrics:validate_metric(m)
    n=metrics[0]['valid_pixels'];require(all(m['valid_pixels']==n for m in metrics),'Same output support')
    e=math.fsum(w*m['error_sum'] for w,m in zip(weights,metrics))
    return {'valid_pixels':n,'error_sum':e,'AEE':e/n if n else None}

def aggregate(metrics):
    for m in metrics:validate_metric(m)
    populated=[m for m in metrics if m['valid_pixels']>0]
    n=sum(m['valid_pixels'] for m in metrics);e=math.fsum(m['error_sum'] for m in metrics)
    return {'valid_pixels':n,'error_sum':e,'frames_with_region':len(populated),
       'empty_region_frames':len(metrics)-len(populated),
       'frame_mean_AEE':math.fsum(m['AEE'] for m in populated)/len(populated) if populated else None,
       'pixel_weighted_AEE':e/n if n else None}

def protection(candidate,native):
    rows={}
    for region,limit in (('all_valid',.03),('B',.05),('L',.05)):
        c,n=candidate[region],native[region]
        require(c['valid_pixels']==n['valid_pixels'],'Fixed protection support')
        cv,nv=c['AEE'],n['AEE'];require((cv is None)==(nv is None),'Paired null')
        rows[region]={'candidate':cv,'reference_native':nv,'support':n['valid_pixels'],
          'limit':limit,'relative_change':cv/nv-1 if nv not in (None,0) else None,
          'positive_support':nv is not None,'pass':None if nv is None else cv<=nv*(1+limit)}
    return {'regions':rows,'diagnostic_all_three_pass':all(v['positive_support'] and v['pass'] for v in rows.values())}

def analyze(frames,weights_by_id):
    require(len(frames)==8 and len({f['target_id'] for f in frames})==8 and len({f['sequence'] for f in frames})==8,'Fixed eight frames/sequences')
    ids=[f['target_id'] for f in frames];require(set(weights_by_id)==set(ids),'All policy frames')
    expected=list(SCORES)+[f'fixed_{r}_{c}' for r,c in TILES]+['uniform_expectation']
    policies={name:[] for name in expected}
    for f in frames:
        tid=f['target_id'];require(set(weights_by_id[tid])==set(expected),'Twenty fixed policies')
        for name in expected:
            w=weights_by_id[tid][name]
            metrics={reg:expectation([f['arms'][f'fine_{r}_{c}'][reg] for r,c in TILES],w) for reg in REGIONS}
            native=f['arms']['native'];cc=f['arms']['CC']
            d=lambda a,b:None if a is None else a-b
            policies[name].append({'target_id':tid,'sequence':f['sequence'],'weights':w,'metrics':metrics,
                'gain_vs_CC':{reg:d(cc[reg]['AEE'],metrics[reg]['AEE']) for reg in REGIONS},
                'delta_vs_native':{reg:d(metrics[reg]['AEE'],native[reg]['AEE']) for reg in REGIONS},
                'protection_vs_native':protection(metrics,native)})
    sequences=[f['sequence'] for f in frames];scopes=['overall']+sequences
    summaries={name:{scope:{reg:aggregate([f['metrics'][reg] for f in pf if scope=='overall' or f['sequence']==scope]) for reg in REGIONS} for scope in scopes} for name,pf in policies.items()}
    native_summary={reg:aggregate([f['arms']['native'][reg] for f in frames]) for reg in REGIONS}
    wrap=lambda m:{reg:{'AEE':m[reg]['frame_mean_AEE'],'valid_pixels':m[reg]['valid_pixels']} for reg in REGIONS}
    guards={name:{'overall':protection(wrap(summaries[name]['overall']),wrap(native_summary)),
       'all_eight_frames':all(f['protection_vs_native']['diagnostic_all_three_pass'] for f in policies[name])} for name in expected}
    for row in guards.values():row['overall_and_all_frames']=row['overall']['diagnostic_all_three_pass'] and row['all_eight_frames']
    return {'policies':policies,'summaries':summaries,'protection_vs_native':guards,
       'frame_region_cells':2560,'summary_cells':2880,'policy_count':20,
       'scope':'Exposed H8 development analysis; uniform is expected actual endpoint error, not an averaged prediction; no learned generic or hardware benefit claim.'}
