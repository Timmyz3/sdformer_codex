"""V2 shared representative producers, original dense current-BN kernels.

V1 is frozen and remains a negative result. This changes the execution
mechanism, not its matrix/tolerances. Device-neutral; no data/training logic.
"""
import torch
from compact_current_mlp import validate,pack,expand,expanded_mlp,require


def gather_exact(dense,cells):
    """Keep every T/B/C, sampling only spatially identical covered cells."""
    points=[];checked=0;all_equal=True
    for y,y1,x,x1,area in cells:
        block=dense[:,:,y:y1,x:x1,:]
        representative=dense[:,:,y:y+1,x:x+1,:]
        integer_dtype=torch.int32 if dense.dtype==torch.float32 else torch.int64
        all_equal=bool(torch.equal(block.contiguous().view(integer_dtype),
                                  representative.expand_as(block).contiguous().view(integer_dtype))) and all_equal
        points.append(representative[:,:,0,0,:]);checked+=block.numel()
    packed=torch.stack(points,dim=2).unsqueeze(3).contiguous()
    require(packed.shape[0:2]==dense.shape[0:2] and packed.shape[-1]==dense.shape[-1],
            'Gather preserves complete T/B/channel axes')
    return packed,checked,all_equal


def native_current_bn(z,cells,height,width,wrapper):
    dense_z=expand(z,cells,height,width)
    # Exactly the native MLP's channel-first view and the actual BN wrapper.
    dense_output=wrapper(dense_z.permute(0,1,4,2,3)).permute(0,1,3,4,2)
    packed,checked,all_equal=gather_exact(dense_output,cells)
    moments={'mean':dense_z.mean((0,1,2,3)),
        'population_variance':dense_z.var((0,1,2,3),unbiased=False),
        'denominator':torch.tensor(z.shape[0]*z.shape[1]*height*width,dtype=z.dtype,device=z.device),
        'epsilon':wrapper.norm_layer.eps,
        'measurement':'Reconstructed from actual native BN input; not the hidden kernel statistics.'}
    return packed,dense_output,dense_z,moments,checked,all_equal


def compact_mlp(native,u,fine_mask):
    validate(native,u,fine_mask);packed,areas,cells=pack(u,fine_mask)
    h,w=u.shape[2:4]
    spikes1=native.sn1(packed);z1=native.fc1(native.drop1(spikes1))
    norm1,dense_norm1,dense_z1,moments1,checked1,equal1=native_current_bn(z1,cells,h,w,native.bn1)
    spikes2=native.sn2(norm1);z2=native.fc2(native.drop2(spikes2))
    norm2,branch,dense_z2,moments2,checked2,equal2=native_current_bn(z2,cells,h,w,native.bn2)
    return branch,{'packed_input':packed,'areas':areas,'cells':cells,'BN1':moments1,'BN2':moments2,
        'BN1_output':norm1,'BN2_output':norm2,'BN1_dense_output':dense_norm1,'BN2_dense_output':branch,
        'BN1_dense_input':dense_z1,'BN2_dense_input':dense_z2,'representatives':len(cells),
        'statistics_barriers':2,'BN_mode':'NATIVE_DENSE_CURRENT','native_BN_calls':2,
        'schedule':'Compact PSN/FC; expand each FC output over original HW; native dense BN; sample all T/B/C at each cell top-left for PSN2. No producer recomputation.',
        'FC1_retained_elements':z1.numel(),'FC2_retained_elements':z2.numel(),
        'full_domain_BN_elements':dense_z1.numel()+dense_z2.numel(),
        'expand_written_elements':dense_z1.numel()+dense_z2.numel(),
        'BN_output_dense_elements':dense_norm1.numel()+branch.numel(),
        'production_BN1_gather_read_elements':norm1.numel(),
        'production_BN1_gather_written_elements':norm1.numel(),
        'diagnostic_BN2_gather_read_elements':norm2.numel(),
        'cell_exact_audit_read_elements':2*(checked1+checked2),
        'BN1_cell_bitwise_equal':equal1,'BN2_cell_bitwise_equal':equal2,
        'each_cell_BN1_and_BN2_bitwise_equal':equal1 and equal2,'full_T_and_all_channels':True,
        'native_MLP_shortcut':False}
