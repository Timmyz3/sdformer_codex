"""R22 independent proxy geometry/metric math; no fitter imports."""
import math
import numpy as np
METHODS=('global_vector','position_prior','q','q_pool_expand3')
REGIONS=('all_valid','B','L','B_and_L','B_only','L_only_B_known','neither_B_known','B_unknown',
 'voxel_nonzero','voxel_zero','B_voxel_nonzero','B_voxel_zero','L_voxel_nonzero','L_voxel_zero',
 'B_unknown_voxel_nonzero','B_unknown_voxel_zero')
def require(ok,message):
    if not ok:raise ValueError(message)
def lift(q):
    require(q.shape==(30,40,2) and np.isfinite(q).all(),'Finite original-grid proxy')
    return q.repeat(16,axis=0).repeat(16,axis=1).transpose(2,0,1)
def pool3(q):
    require(q.shape==(30,40,2),'Full original grid')
    out=np.empty_like(q)
    for y in range(0,30,3):
        for x in range(0,40,3):
            out[y:y+3,x:x+3]=q[y:y+3,x:x+3].mean((0,1),dtype=np.float32)
    return out
def measure(q,gt,masks):
    require(set(masks)==set(REGIONS) and gt.shape==(2,480,640),'Complete original GT/regions')
    full=lift(q).astype(np.float64);vectors=gt.astype(np.float64);e=np.sqrt(np.square(full-vectors).sum(0))
    grid=q.astype(np.float64);hwc=vectors.transpose(1,2,0);out={}
    for r in REGIONS:
        mask=masks[r];require(mask.shape==(480,640) and mask.dtype==np.bool_,'Original region mask')
        n=int(mask.sum());err=float(e[mask].sum(dtype=np.float64))
        counts=mask.reshape(30,16,40,16).sum((1,3),dtype=np.int64)
        sums=np.where(mask[...,None],hwc,0.).reshape(30,16,40,16,2).sum((1,3),dtype=np.float64)
        good=counts>0;cells=int(good.sum());mean=sums/np.maximum(counts,1)[...,None]
        ce=np.sqrt(np.square(grid-mean).sum(-1));weighted=float((ce[good]*counts[good]).sum(dtype=np.float64));unweighted=float(ce[good].sum(dtype=np.float64))
        require(n==int(counts.sum()) and ((cells==0)==(n==0)),'Cell/pixel support agrees')
        require(weighted<=err+1e-9+1e-12*abs(err),'Jensen/index support consistency')
        out[r]=dict(pixel=dict(valid_pixels=n,error_sum=err,AEE=err/n if n else None),
          grid=dict(valid_pixels=n,nonempty_cells=cells,weighted_error_sum=weighted,cell_error_sum=unweighted,
            pixel_weighted_cell_AEE=weighted/n if n else None,unweighted_cell_AEE=unweighted/cells if cells else None))
    return out
def aggregate_pixel(rows):
    good=[x for x in rows if x['valid_pixels']];n=sum(x['valid_pixels'] for x in rows);e=math.fsum(x['error_sum'] for x in rows)
    return dict(valid_pixels=n,error_sum=e,frames_with_region=len(good),empty_region_frames=len(rows)-len(good),
      frame_mean_AEE=math.fsum(x['AEE'] for x in good)/len(good) if good else None,pixel_weighted_AEE=e/n if n else None)
def aggregate_grid(rows):
    good=[x for x in rows if x['valid_pixels']];n=sum(x['valid_pixels'] for x in rows);c=sum(x['nonempty_cells'] for x in rows)
    w=math.fsum(x['weighted_error_sum'] for x in rows);u=math.fsum(x['cell_error_sum'] for x in rows)
    return dict(valid_pixels=n,nonempty_cells=c,weighted_error_sum=w,cell_error_sum=u,frames_with_region=len(good),empty_region_frames=len(rows)-len(good),
      frame_mean_pixel_weighted_cell_AEE=math.fsum(x['pixel_weighted_cell_AEE'] for x in good)/len(good) if good else None,
      frame_mean_unweighted_cell_AEE=math.fsum(x['unweighted_cell_AEE'] for x in good)/len(good) if good else None,
      pixel_weighted_cell_AEE=w/n if n else None,unweighted_cell_AEE=u/c if c else None)
def difference(candidate,reference):
    require((candidate is None)==(reference is None),'Same comparison support')
    return dict(candidate=candidate,reference=reference,signed_difference=candidate-reference if candidate is not None else None,
      relative_change=candidate/reference-1 if reference not in (None,0.) else None)
def analyze(frames):
    require(len(frames)==len({f['target_id'] for f in frames})==24,'Exactly24 distinct identities')
    require(set(f['role'] for f in frames)=={'fit','development'},'No sealed data')
    summaries={};comparisons={};native={}
    for role,count in (('fit',16),('development',8)):
        selected=[f for f in frames if f['role']==role];seqs=sorted({f['sequence'] for f in selected})
        require(len(selected)==count and len(seqs)==8 and all(sum(f['sequence']==s for f in selected)==count//8 for s in seqs),'Fixed8x2/8x1 roles')
        label='fit_sequence_OOF' if role=='fit' else 'development_final_fit16'
        summaries[label]={};native[label]={};comparisons[label]={}
        for scope in ['overall']+seqs:
            group=[f for f in selected if scope=='overall' or f['sequence']==scope]
            summaries[label][scope]={};native[label][scope]={}
            for method in METHODS:
                summaries[label][scope][method]={r:dict(pixel=aggregate_pixel([f['representations'][method][r]['pixel'] for f in group]),grid=aggregate_grid([f['representations'][method][r]['grid'] for f in group])) for r in REGIONS}
            native[label][scope]={r:aggregate_pixel([f['native_head3'][r] for f in group]) for r in REGIONS}
            comparisons[label][scope]={}
            for name,ca,re in (('q_vs_global','q','global_vector'),('q_vs_position','q','position_prior'),('pool_vs_q','q_pool_expand3','q')):
                comparisons[label][scope][name]={}
                for r in REGIONS:
                    a,b=summaries[label][scope][ca][r],summaries[label][scope][re][r]
                    comparisons[label][scope][name][r]={'valid_pixels':a['pixel']['valid_pixels'],
                      'pixel_frame_mean':difference(a['pixel']['frame_mean_AEE'],b['pixel']['frame_mean_AEE']),
                      'pixel_weighted':difference(a['pixel']['pixel_weighted_AEE'],b['pixel']['pixel_weighted_AEE']),
                      'grid_pixel_weighted':difference(a['grid']['pixel_weighted_cell_AEE'],b['grid']['pixel_weighted_cell_AEE']),
                      'grid_unweighted':difference(a['grid']['unweighted_cell_AEE'],b['grid']['unweighted_cell_AEE'])}
    return dict(frames=frames,summaries=summaries,native_head3_context=native,fixed_representation_comparisons=comparisons,
      representation_frame_region_cells=1536,native_frame_region_cells=384,representation_scope_method_region_cells=1152,
      scope='Representation error only; no allocation/selector/compression benefit, no T8+k1 admission, no formal or scene-independent validation.')
