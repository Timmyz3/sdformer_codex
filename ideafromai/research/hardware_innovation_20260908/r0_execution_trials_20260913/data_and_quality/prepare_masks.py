"""One fixed 25% pruning point. Calibration is the same eight RTL tiles."""
import json
import numpy as np
from model_access import HERE
BT=np.array([[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]],dtype=np.int64)
AT=np.array([[1,1,1,0],[0,1,-1,-1]],dtype=np.int64)
G2=np.array([[2,0,0],[1,1,1],[1,-1,1],[0,0,2]],dtype=np.int64)

def expand_u4(u4):
    # n,a,b,c,p,q; channel order required by pixel_shuffle is n,(a*2+b).
    h=np.einsum('ai,bj,ncij,ip,jq->nabcpq',AT,AT,u4,BT,BT,optimize=True)
    return h.reshape(384,96,4,4)

def rne2(x):
    lo=x//4; rem=x%4
    return lo+((rem>2)|((rem==2)&((lo&1)!=0)))

def main():
    cap=np.load(HERE/'r0_contiguous_t10.npz')
    s=cap['source_bits'].astype(np.int64);w=cap['weight_q16'].astype(np.int64)
    assert s.shape==(8,10,96,4,4) and float(cap['theta'])==1
    cost=np.zeros(24,np.int64); native=np.zeros(24,np.int64)
    energy=np.zeros((12,24),np.float64)
    for cg in range(24):
        cs=s[:,:,cg*4:(cg+1)*4]
        for ky in range(3):
            for kx in range(3):
                patch=cs[:,:,:,ky:ky+2,kx:kx+2]
                cost[cg]+=np.any(patch,axis=(1,3,4)).sum()
                native[cg]+=np.any(patch,axis=1).sum()
        # Exact integer contribution before float squaring. Complete9tap/P4/T10.
        for og in range(12):
            wg=w[og*8:(og+1)*8,cg*4:(cg+1)*4]
            for py in range(2):
                for px in range(2):
                    y=np.einsum('ftcij,ncij->ftn',cs[:,:,:,py:py+3,px:px+3],wg,optimize=True)
                    energy[og,cg]+=np.square(y.astype(np.float64)).sum()
    scores=np.full((12,24),np.inf)
    np.divide(energy,cost[None],out=scores,where=cost[None]>0)
    cost_live=np.ones((12,24),np.bool_);cost_live.ravel()[np.argsort(scores.ravel(),kind='stable')[:72]]=False
    mag=np.square(w.astype(np.float64)).reshape(12,8,24,4,3,3).sum(axis=(1,3,4,5))
    mag_live=np.ones((12,24),np.bool_);mag_live.ravel()[np.argsort(mag.ravel(),kind='stable')[:72]]=False
    dense_live=np.ones((12,24),np.bool_)
    u4=np.einsum('ia,ncab,jb->ncij',G2,w,G2,optimize=True)
    v=np.einsum('ip,ftcpq,jq->ftcij',BT,s,BT,optimize=True)
    coord_energy=np.zeros((12,16),np.float64)
    gain=np.outer(np.square(AT).sum(axis=0),np.square(AT).sum(axis=0))
    for og in range(12):
        z=np.einsum('ncij,ftcij->ftnij',u4[og*8:(og+1)*8],v,optimize=True)
        coord_energy[og]=(np.square(z.astype(np.float64)).sum(axis=(0,1,2))*gain/16).ravel()
    coord_live=np.ones((12,16),np.bool_)
    for og in range(12):coord_live[og,np.argsort(coord_energy[og],kind='stable')[:4]]=False
    masked_u4=u4*coord_live.repeat(8,axis=0)[:,None,:].reshape(96,1,4,4)
    z4=np.einsum('ai,ftnij,bj->ftnab',AT,np.einsum('ncij,ftcij->ftnij',masked_u4,v,optimize=True),AT,optimize=True)
    expanded=expand_u4(masked_u4)
    # Exact integer equality of transform and phase-aware expanded definition.
    expanded_z4=np.einsum('ftcpq,ncpq->ftn',s,expanded,optimize=True).reshape(8,10,96,2,2)
    assert np.array_equal(expanded_z4,z4)
    exact4=expand_u4(u4)
    exact_out=np.einsum('ftcpq,ncpq->ftn',s,exact4,optimize=True).reshape(8,10,96,2,2)
    assert np.array_equal(exact_out,cap['golden_accum']*4)
    shared=dict(source_bits=cap['source_bits'],source_valid_yx=cap['source_valid_yx'],
        weight_fp32=cap['weight_fp32'],theta=cap['theta'],weight_exponent=cap['weight_exponent'],
        output_origin_yx=cap['output_origin_yx'],input_origin_yx=cap['input_origin_yx'],frame=cap['frame'])
    arms={}
    for name,live in [('dense_q16',dense_live),('physical25_q16',cost_live),('magnitude25_q16',mag_live)]:
        qw=w*live.repeat(8,axis=0).repeat(4,axis=1)[:,:,None,None]
        golden=np.empty((8,10,96,2,2),np.int64)
        for py in range(2):
            for px in range(2):golden[:,:,:,py,px]=np.einsum('ftcij,ncij->ftn',s[:,:,:,py:py+3,px:px+3],qw,optimize=True)
        np.savez_compressed(HERE/(name+'.npz'),**shared,weight_q16=qw.astype(np.int16),live=live,golden_accum=golden)
        arms[name]=dict(file=name+'.npz',dropped_groups=int((~live).sum()),coefficient_nonzero=int(np.count_nonzero(qw)),
            removed_weight_major_words=int((~live*cost[None]).sum()),
            remaining_weight_major_words=int((live*cost[None]).sum()),
            remaining_native_source_words=int((live*native[None]).sum()),
            cal_output_squared_error=float(np.square((golden-cap['golden_accum']).astype(np.float64)).sum()))
    np.savez_compressed(HERE/'coordinate25_q16.npz',**shared,weight_q16=w.astype(np.int16),
        live=coord_live,coordinate_live=coord_live,U4_q16=u4,masked_U4_q16=masked_u4,
        phase_kernel4_q16=expanded,golden_accum4=z4,golden_rne_q16=rne2(z4))
    arms['coordinate25_q16']=dict(file='coordinate25_q16.npz',dropped_coordinates_per_O8=(~coord_live).sum(axis=1).tolist(),
        cal_output_squared_error_before_rne=float(np.square((z4/4-cap['golden_accum'])).sum()),
        cal_fraction_requiring_RNE=float(np.mean(z4%4!=0)),U4_q16_range=[int(u4.min()),int(u4.max())],
        phase_kernel4_q16_range=[int(expanded.min()),int(expanded.max())])
    np.savez_compressed(HERE/'selection_scores.npz',physical_cost_per_C4=cost,native_cost_per_C4=native,
        output_contribution_energy=energy,physical_score=scores,magnitude_score=mag,
        coordinate_contribution_energy=coord_energy,physical_live=cost_live,magnitude_live=mag_live,coordinate_live=coord_live)
    report=dict(complete=True,calibration_frame=str(cap['frame']),calibration_tiles=cap['output_origin_yx'].tolist(),
        calibration_overlaps_diverse10=True,rtl_measurement_uses_same_calibration_tiles=True,
        heldout_spatial_tiles=False,fixed_pruning_fraction=0.25,training=False,search_or_sweep=False,
        group_layout='O8 x C4 x full3x3;72of288 removed once',
        score='sum_T,tile,P4,O8(group linear contribution^2) / actual weight-major128bit W-word demand',
        cost='sum_tile,c in C4,tap any_{T10,P4}(S); no x8 factor; one128bit word has8 output banks',
        zero_cost_policy='infinite score for zero-cost group; tie break stable flatOgroup,Cgroup index',
        magnitude_control='same72 physical groups by Q16 coefficient squared norm; not unstructured matching',
        coordinate_layout='O8 x xi16 across allC96;4/16 removed perO8 by inverse-transform output contribution energy',
        coordinate_convention='U4=G2 Wq G2^T; output4=A^T(sum_C U4 * B^T S B) A; xi=i*4+j;phase=a*2+b',
        phase_kernel='E4[n,a,b,c,p,q]=sum_ij AT[a,i]AT[b,j]U4[n,c,i,j]BT[i,p]BT[j,q]; Fconv2d E4/(4*65536),pad1,stride2 thenpixel_shuffle2',
        exact_integer_phase_check=True,arms=arms)
    (HERE/'mask_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
