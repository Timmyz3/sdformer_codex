"""Device-neutral current-BN representative MLP; inference arithmetic only.

This is a shared ordinary executor, not a flow-specific score or task module.
No checkpoint, GT, scheduler, native-fine shortcut, or training is included.
"""
import torch

TILE=9
STRIDE=3


def require(ok,message):
    if not ok:raise ValueError(message)


def validate(native,u,mask):
    require(type(native).__name__=='MS_Spiking_Mlp' and native.norm_layer=='BN','Actual native MS_Spiking_Mlp/BN required')
    require(u.ndim==5 and u.shape[0]==10 and u.dtype in (torch.float32,torch.float64) and bool(torch.isfinite(u).all()),'Finite complete T10/B/H/W/C')
    require(mask.dtype==torch.bool and tuple(mask.shape)==((u.shape[2]+8)//9,(u.shape[3]+8)//9),'Whole spatial9tile Boolean mask')
    require(all(not m.training for m in native.modules()) and not any(p.requires_grad for p in native.parameters()),'Frozen eval/dropout-off inference only')
    require(native.fc1.in_features==u.shape[-1] and native.fc1.out_features==native.fc2.in_features
            and native.fc2.out_features==u.shape[-1],'Native full-channel widths')
    require(all(p.dtype==u.dtype and p.device==u.device for p in native.parameters()),'Same true dtype/device')
    for sn in (native.sn1.spiking_neuron,native.sn2.spiking_neuron):
        require(type(sn).__name__=='PSN' and sn.T==10 and tuple(sn.weight.shape)==(10,10),'Native complete temporal PSN')
    for wrapper in (native.bn1,native.bn2):
        bn=wrapper.norm_layer
        require(isinstance(bn,torch.nn.modules.batchnorm._BatchNorm) and not bn.track_running_stats
                and all(getattr(bn,k) is None for k in ('running_mean','running_var','num_batches_tracked')),'Current-stat BN only')
        require(bn.step_mode=='m','Native BN reduces full T/B/spatial domain')
        require(bn.affine and bn.eps>0,'Native eps and affine required')


def pack(u,mask):
    """All T/B/C share one deterministic cover; no padding or partial channel."""
    _,_,height,width,_=u.shape;cells=[];points=[]
    for sy in range(0,height,9):
        for sx in range(0,width,9):
            ey,ex=min(sy+9,height),min(sx+9,width)
            fine=bool(mask[sy//9,sx//9]);stride=1 if fine else 3
            for y in range(sy,ey,stride):
                for x in range(sx,ex,stride):
                    y1,x1=min(y+stride,ey),min(x+stride,ex);area=(y1-y)*(x1-x)
                    value=u[:,:,y:y1,x:x1,:].sum((2,3))/area
                    points.append(value);cells.append((y,y1,x,x1,area))
    packed=torch.stack(points,dim=2).unsqueeze(3).contiguous()
    areas=torch.tensor([c[4] for c in cells],dtype=u.dtype,device=u.device)
    require(sum(c[4] for c in cells)==height*width,'Exact real-cell coverage')
    return packed,areas,cells


def expand(packed,cells,height,width):
    require(packed.ndim==5 and packed.shape[2]==len(cells) and packed.shape[3]==1,'One representative per real cell')
    result=torch.empty((packed.shape[0],packed.shape[1],height,width,packed.shape[-1]),dtype=packed.dtype,device=packed.device)
    for i,(y,y1,x,x1,area) in enumerate(cells):result[:,:,y:y1,x:x1,:]=packed[:,:,i:i+1,:,:]
    return result


def weighted_current_bn(z,areas,bn):
    """Two-pass population variance over full T/B/original-area, actual dtype."""
    require(z.shape[2]==areas.numel() and z.shape[3]==1,'Packed spatial layout')
    weight=areas.view(1,1,-1,1,1)
    denominator=z.shape[0]*z.shape[1]*areas.sum()
    mean=(z*weight).sum((0,1,2,3))/denominator
    variance=((z-mean).square()*weight).sum((0,1,2,3))/denominator
    output=(z-mean)*torch.rsqrt(variance+bn.eps)*bn.weight+bn.bias
    return output,{'mean':mean,'population_variance':variance,'denominator':denominator,'epsilon':bn.eps}


def compact_mlp(native,u,fine_mask):
    validate(native,u,fine_mask)
    packed,areas,cells=pack(u,fine_mask)
    # Even all-fine uses these compact producers and weighted reductions.
    spikes1=native.sn1(packed);z1=native.fc1(native.drop1(spikes1))
    norm1,moments1=weighted_current_bn(z1,areas,native.bn1.norm_layer)
    spikes2=native.sn2(norm1);z2=native.fc2(native.drop2(spikes2))
    norm2,moments2=weighted_current_bn(z2,areas,native.bn2.norm_layer)
    branch=expand(norm2,cells,u.shape[2],u.shape[3])
    return branch,{'packed_input':packed,'areas':areas,'cells':cells,'BN1':moments1,'BN2':moments2,
                   'BN1_output':norm1,'BN2_output':norm2,'representatives':len(cells),'statistics_barriers':2,
                   'schedule':'Retain FC1 and FC2 representative tensors across their separate full-domain BN barriers; no producer recomputation.',
                   'FC1_retained_elements':z1.numel(),'FC2_retained_elements':z2.numel(),
                   'full_T_and_all_channels':True,'native_MLP_shortcut':False}


def expanded_input_reference(u,mask):
    """Independent dense assembly: never calls compact pack/cells/expand."""
    out=u.clone();height,width=u.shape[2:4]
    for ty in range(mask.shape[0]):
        for tx in range(mask.shape[1]):
            if bool(mask[ty,tx]):continue
            ystop,xstop=min((ty+1)*9,height),min((tx+1)*9,width)
            for y in range(ty*9,ystop,3):
                for x in range(tx*9,xstop,3):
                    yy,xx=min(y+3,ystop),min(x+3,xstop)
                    # Native reference dense grid, every real covered pixel.
                    out[:,:,y:yy,x:xx,:]=u[:,:,y:yy,x:xx,:].mean((2,3),keepdim=True)
    return out.contiguous()


def expanded_mlp(native,u,fine_mask):
    validate(native,u,fine_mask)
    dense=expanded_input_reference(u,fine_mask)
    return native(dense),dense
