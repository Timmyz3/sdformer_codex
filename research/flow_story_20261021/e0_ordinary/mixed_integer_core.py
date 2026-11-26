"""E0 two-domain integer affine and identity STE; default ordinary RNE4.
Derived from frozen R30 mixed_integer_core.py
ceec6924fd07a33fa7b806dce7ee4699a4a438a08d6a342b24137d216e7a13d5.
All spatial/T/channel positions, original current-BN and PSN are preserved.
Low variants: rne4 independent maxabs/7 scale; midpoint drop4+8;
drop4 exists only for CPU/source-equivalence checks. No GPU launch here.
"""
import contextlib
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F

PREFIX='sttmultires_unet.encoders.swin3d.'
LINEARS=tuple(PREFIX+f'layers.{s}.swin_blocks.{b}.mlp.{fc}'
              for s,depth in enumerate((2,2,6,2)) for b in range(depth) for fc in ('fc1','fc2'))
CONVS=tuple(PREFIX+f'patch_embed.residual_encoding.resblocks.{b}.conv{j}.0' for b in (0,1) for j in (1,2))
TARGETS=LINEARS+CONVS
MODES=('Q8','Q4','M')
VARIANTS=('rne4','midpoint','drop4')
GROUPS=tuple(n.rsplit('.',1)[0] for n in LINEARS[::2])+CONVS
MAX_ROWS_PER_KERNEL=65536
MAX_OUTPUT_ELEMENTS=8*1024*1024


def require(ok,why):
    if not ok:raise ValueError(why)


def digest(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def quantize(weight):
    """CPU FP32 once per actual callable invocation, no learned calibration."""
    w=weight.detach().reshape(weight.shape[0],-1).cpu()
    require(w.dtype==torch.float32 and bool(torch.isfinite(w).all()),'Finite FP32 weights')
    scale=w.abs().amax(1)/127.
    divisor=torch.where(scale==0,torch.ones_like(scale),scale)
    q=torch.round(w/divisor[:,None]).clamp(-127,127).to(torch.int8)
    return q.to(weight.device),scale.to(weight.device)


def tail_rows(t,b,h,w,device):
    require(t==10 and b>=1 and h>0 and w>0,'Native fullT10/B/H/W')
    spatial=torch.ones(h,w,dtype=torch.bool,device=device)
    spatial[:(h//9)*9,:(w//9)*9]=False
    return spatial[None,None].expand(t,b,h,w).reshape(-1)


def group_name(name):return name.rsplit('.',1)[0] if name in LINEARS else name


def row_masks(t,b,h,w,device,mode,selected=None):
    tail=tail_rows(t,b,h,w,device)
    require(mode in MODES,'Declared precision mode')
    require((selected is not None)==(mode=='M'),'Only mixed mode receives explicit mask')
    if mode=='Q8':high=torch.ones_like(tail);coords=[]
    elif mode=='Q4':high=tail;coords=[]
    else:
        require(selected.dtype==torch.bool and selected.device.type=='cpu' and tuple(selected.shape)==(h,w),'Explicit frozen CPU spatial bool mask')
        chosen=selected.detach();hh,ww=h//9*9,w//9*9
        require(not bool(chosen[hh:,:].any()) and not bool(chosen[:,ww:].any()),'Only complete tiles selected; true tails implicit Q8')
        counts=chosen[:hh,:ww].reshape(h//9,9,w//9,9).permute(0,2,1,3).sum((2,3))
        require(bool(((counts==0)|(counts==81)).all()),'Whole9x9 selection, no fragmented rows')
        coords=torch.nonzero(counts==81).tolist();chosen=chosen.to(device)
        high=tail|chosen[None,None].expand(t,b,h,w).reshape(-1)
    return tail,high,dict(selected_complete_tiles=coords,selected_spatial_mask_sha256=digest(selected) if selected is not None else None,
                          geometric_shape=[t,b,h,w],row_order='T,B,output_y,output_x')


def kernel_mm(a,b):
    require(a.ndim==b.ndim==2 and a.dtype==b.dtype==torch.int8 and a.shape[1]==b.shape[0],'Integer operand contract')
    n=len(a);padding=0
    if a.device.type=='cuda':
        require(a.shape[1]%8==0 and b.shape[1]%8==0,'CUDA native K/N multiple8; no channel padding')
        # Explicit zero rows only for a last chunk too small for the CUDA kernel.
        issued=max(24,((n+7)//8)*8);padding=issued-n
        if padding:a=F.pad(a,(0,0,0,padding))
    output=torch._int_mm(a.contiguous(),b.contiguous())
    require(output.dtype==torch.int32 and output.shape==(n+padding,b.shape[1]),'True int32 result')
    return output[:n],padding


def representations(weight,mode,variant):
    require(mode in MODES and variant in VARIANTS,'Declared mode and quantizer')
    q8,s8=quantize(weight)
    meta=dict(variant=variant,q8_sha256=digest(q8),s8_sha256=digest(s8),
              q8_shape=list(q8.shape),quantization_device='cpu',rounding='ties_to_even',
              high_scale='per_output_maxabs/127',low_representation_generated=mode!='Q8')
    if mode=='Q8':return q8,s8,q8,s8,0,meta
    if variant=='rne4':
        w=weight.detach().reshape(weight.shape[0],-1).cpu()
        s4=w.abs().amax(1)/7.;divisor=torch.where(s4==0,torch.ones_like(s4),s4)
        q4=torch.round(w/divisor[:,None]).clamp(-8,7).to(torch.int8)
        low=(q4.to(torch.int16)<<4).to(torch.int8).to(weight.device)
        scale=(s4/16.).to(weight.device);offset=0
        meta.update(q4_sha256=digest(q4),s4_sha256=digest(s4),
                    low_scale='per_output_maxabs/7/16',low_encoding='signed_q4_left_shift4',
                    low_rule='independent_rne4_maxabs7_clamp_minus8_plus7')
    else:
        low=((q8.to(torch.int16)>>4)<<4).to(torch.int8);scale=s8
        offset=8 if variant=='midpoint' else 0
        meta.update(low_scale='same_q8_scale',low_encoding='signed_q8_drop4',
                    low_rule='drop4_plus_8_popcount' if offset else 'drop4')
    meta.update(low_q_sha256=digest(low),low_scale_sha256=digest(scale),low_offset_per_active_input=offset)
    return q8,s8,low,scale,offset,meta


def integer_affine(x,q8,s8,tail,mode,*,high8=None,low_q=None,low_scale=None,
                   low_offset=0,representation=None,reference=False):
    """Two exclusive domains; actual low4 work never evaluates full Q8.

    rne4 low accumulator is dot(x, q4<<4), scaled once by s4/16.
    midpoint is dot(x, drop4(q8))+8*popcount(x); reference dots
    drop4(q8)+8 directly. Returned integer captures the corrected accumulator.
    """
    require(mode in MODES and x.ndim==2 and x.shape[1]==q8.shape[1] and tail.shape==(len(x),),'Affine shapes')
    require(x.dtype==torch.float32 and bool(torch.isfinite(x).all()) and bool(((x==0)|(x==1)).all()),'Actual binary operands')
    require(q8.dtype==torch.int8 and s8.dtype==torch.float32,'High integer encoding')
    if high8 is None:
        require(mode!='M','Mixed needs explicit mask');high8=torch.ones_like(tail) if mode=='Q8' else tail
    require(high8.dtype==torch.bool and high8.shape==tail.shape and bool((high8|~tail).all()),'True tails retain Q8')
    require(mode!='Q8' or bool(high8.all()),'Q8 all high')
    require(mode!='Q4' or torch.equal(high8,tail),'Q4 complete low / tail high')
    if low_q is None:low_q=((q8.to(torch.int16)>>4)<<4).to(torch.int8)
    if low_scale is None:low_scale=s8
    require(low_q.shape==q8.shape and low_q.dtype==torch.int8 and low_scale.shape==s8.shape and low_scale.dtype==torch.float32,'Independent low representation')
    require(low_offset in (0,8) and bool(((low_q.to(torch.int16)&15)==0).all()) if mode!='Q8' else low_offset==0,'Low four zero encoding bits')
    binary=x.to(torch.int8);limit=max(24,min(MAX_ROWS_PER_KERNEL,MAX_OUTPUT_ELEMENTS//q8.shape[0]));limit=(limit//8)*8
    integer=torch.zeros(len(x),q8.shape[0],device=x.device,dtype=torch.int32)
    output=torch.empty(len(x),q8.shape[0],device=x.device,dtype=torch.float32)
    calls=[];correction_rows=0
    for high in (True,False):
        indices=torch.nonzero(high8 if high else ~high8).flatten()
        if not len(indices):continue
        q,scale=(q8,s8) if high else (low_q,low_scale)
        offset=0 if high else low_offset;bits=range(8) if high else range(4,8)
        for start in range(0,len(indices),limit):
            ids=indices[start:start+limit];operand=binary.index_select(0,ids)
            accum=torch.zeros(len(ids),q8.shape[0],device=x.device,dtype=torch.int32)
            if reference:
                effective=(q.to(torch.int16)+offset).to(torch.int8)
                accum,pad=kernel_mm(operand,effective.T)
                calls.append(dict(kind='dense_integer_reference',bit=None,rows=len(ids),padding_rows=pad,scope='high8' if high else 'low4',coefficient=None))
            else:
                for bit in bits:
                    plane=((q.to(torch.int16)>>bit)&1).to(torch.int8).T.contiguous()
                    coeff=-128 if bit==7 else 1<<bit
                    part,pad=kernel_mm(operand,plane);accum.add_(part*coeff)
                    calls.append(dict(kind='binary_weight_plane',bit=bit,rows=len(ids),padding_rows=pad,scope='high8' if high else 'low4',coefficient=coeff))
                if offset:
                    counts=operand.to(torch.int32).sum(1,dtype=torch.int32)
                    accum.add_(counts[:,None]*offset);correction_rows+=len(ids)
            integer.index_copy_(0,ids,accum)
            output.index_copy_(0,ids,accum.float()*scale)
    audit=dict(mode=mode,reference=reference,rows=len(x),Cin=q8.shape[1],Cout=q8.shape[0],tail_rows=int(tail.sum()),
        complete_rows=int((~tail).sum()),high8_rows=int(high8.sum()),low4_rows=int((~high8).sum()),
        restored_complete_rows=int((high8&~tail).sum()),high8_rows_sha256=digest(high8),binary_verified=True,int32_accumulation=True,
        scale_applications=1,scale_applications_per_output=1,
        scale_variants=2 if mode!='Q8' and (representation or {}).get('variant')=='rne4' else 1,
        logical_row_plane_work=sum(c['rows'] for c in calls) if not reference else None,
        actual_integer_kernel_calls=len(calls),explicit_padded_rows=sum(c['padding_rows'] for c in calls),kernel_calls=calls,
        spatial_rows_retained=True,full_T_and_channels=True,quantization_device='cpu',software_plane_dtype='int8',cycle_claim=False,
        low_bit_rule=(representation or {}).get('low_rule','drop4'),representation=representation,
        midpoint_correction=dict(rows=correction_rows,inputs_per_row=x.shape[1],broadcast_outputs=q8.shape[0],offset=low_offset,
            reference_fused_into_effective_weight=reference and low_offset!=0),
        canonical_scatter_before_BN=True,independent_domain_execution=True)
    return output,binary,integer,audit


def effective_weights(q8,s8,low_q,low_scale,low_offset):
    return q8.float()*s8[:,None],(low_q.float()+low_offset)*low_scale[:,None]


class _LinearSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias,mode,holder):
        t,b,h,w,c=x.shape;tail,high,mask_audit=row_masks(t,b,h,w,x.device,mode,holder.selected);q,scale,lq,ls,off,rep=representations(weight,mode,holder.variant)
        y,binary,integer,audit=integer_affine(x.reshape(-1,c),q,scale,tail,mode,high8=high,low_q=lq,low_scale=ls,low_offset=off,representation=rep);audit.update(mask_audit)
        if bias is not None:y=y+bias
        ctx.save_for_backward(binary,q,scale,lq,ls,high);ctx.low_offset=off;ctx.mode=mode;ctx.shape=tuple(x.shape);ctx.has_bias=bias is not None
        holder.last_audit=audit;holder.last_integer=integer.detach() if holder.capture_integer else None
        return y.reshape(t,b,h,w,weight.shape[0])
    @staticmethod
    def backward(ctx,grad):
        x,q,scale,lq,ls,high=ctx.saved_tensors;g=grad.reshape(-1,q.shape[0]);full,low=effective_weights(q,scale,lq,ls,ctx.low_offset)
        dx=torch.empty(len(x),q.shape[1],device=g.device,dtype=g.dtype)
        dx[high]=g[high]@full;dx[~high]=g[~high]@low
        # Identity STE for q(W)*scale: d effective W / d master W := 1.
        # Scale/rounding/clipping are not differentiated; this is declared surrogate math.
        dw=g.T@x.float();db=g.sum(0) if ctx.has_bias else None
        return dx.reshape(ctx.shape),dw,db,None,None


class _ConvSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias,mode,holder):
        require(x.ndim==5,'Actual original multi-step Conv2d input')
        t,b,c,h,w=x.shape;require(tuple(weight.shape[-2:])==(3,3),'Fixed ordinary3x3 conv')
        tail,high,mask_audit=row_masks(t,b,h,w,x.device,mode,holder.selected);q,scale,lq,ls,off,rep=representations(weight,mode,holder.variant)
        flat=x.reshape(t*b,c,h,w)
        require(x.dtype==torch.float32 and bool(torch.isfinite(x).all()) and bool(((x==0)|(x==1)).all()),'Actual conv operand binary')
        # Native stride1,pad1,groups1,dilation1; zero padding is exact binary0.
        rows=F.unfold(flat,kernel_size=3,padding=1).transpose(1,2).reshape(-1,c*9)
        y,_,integer,audit=integer_affine(rows,q,scale,tail,mode,high8=high,low_q=lq,low_scale=ls,low_offset=off,representation=rep);audit.update(mask_audit)
        if bias is not None:y=y+bias
        ctx.save_for_backward(x.to(torch.int8),q,scale,lq,ls,high);ctx.low_offset=off;ctx.mode=mode;ctx.weight_shape=tuple(weight.shape);ctx.has_bias=bias is not None
        audit.update(operation='Conv2d',kernel=[3,3],stride=[1,1],padding=[1,1],groups=1,dilation=[1,1],
                     unfold_elements=rows.numel(),output_tile_origin='original_output_yx',padded_zero_input_taps_included=True)
        holder.last_audit=audit;holder.last_integer=integer.detach() if holder.capture_integer else None
        return y.reshape(t,b,h,w,weight.shape[0]).permute(0,1,4,2,3).contiguous()
    @staticmethod
    def backward(ctx,grad):
        x,q,scale,lq,ls,high=ctx.saved_tensors;t,b,c,h,w=x.shape;co=q.shape[0]
        g=grad.reshape(t*b,co,h,w);full,low=effective_weights(q,scale,lq,ls,ctx.low_offset)
        full,low=full.reshape(ctx.weight_shape),low.reshape(ctx.weight_shape)
        mask=high.reshape(t*b,1,h,w)
        if ctx.mode=='Q8':
            dx=torch.nn.grad.conv2d_input((t*b,c,h,w),full,g,padding=1)
        else:
            dx=torch.nn.grad.conv2d_input((t*b,c,h,w),full,g*mask,padding=1)+torch.nn.grad.conv2d_input((t*b,c,h,w),low,g*(~mask),padding=1)
        dw=torch.nn.grad.conv2d_weight(x.reshape(t*b,c,h,w).float(),ctx.weight_shape,g,padding=1)
        db=g.sum((0,2,3)) if ctx.has_bias else None
        return dx.reshape(t,b,c,h,w),dw,db,None,None


def deploy_forward(module,x,*,reference=False):
    """Autograd-free production entry, same integer math, independent call path."""
    q,scale,lq,ls,off,rep=representations(module.weight,module.mode,module.variant)
    if module.kind=='linear':
        t,b,h,w,c=x.shape;rows=x.reshape(-1,c)
    else:
        t,b,c,h,w=x.shape
        rows=F.unfold(x.reshape(t*b,c,h,w),3,padding=1).transpose(1,2).reshape(-1,c*9)
    tail,high,mask_audit=row_masks(t,b,h,w,x.device,module.mode,module.selected)
    y,_,integer,audit=integer_affine(rows,q,scale,tail,module.mode,high8=high,low_q=lq,low_scale=ls,low_offset=off,representation=rep,reference=reference);audit.update(mask_audit)
    if module.kind=='conv':audit.update(operation='Conv2d',kernel=[3,3],stride=[1,1],padding=[1,1],groups=1,dilation=[1,1],
       unfold_elements=rows.numel(),output_tile_origin='original_output_yx',padded_zero_input_taps_included=True)
    if module.bias is not None:y=y+module.bias
    module.last_audit=audit;module.last_integer=integer.detach() if module.capture_integer else None
    y=y.reshape(t,b,h,w,module.weight.shape[0])
    return y if module.kind=='linear' else y.permute(0,1,4,2,3).contiguous()


class STEAffine(nn.Module):
    def __init__(self,source,kind,mode,*,selected=None,ste=True,reference=False,variant="rne4"):
        super().__init__();require(mode in MODES,'Only predeclared Q8/Q4/M')
        self.weight=source.weight;self.bias=source.bias
        self.kind,self.mode=kind,mode;self.ste,self.reference=ste,reference
        require(variant in VARIANTS,"Known ordinary quantizer");self.variant=variant
        self.selected=selected
        self.capture_integer=False;self.last_integer=None;self.last_audit=None
        if kind=='linear':
            self.in_features,self.out_features=source.in_features,source.out_features
        else:
            require(tuple(source.kernel_size)==(3,3) and tuple(source.stride)==(1,1) and tuple(source.padding)==(1,1)
                    and tuple(source.dilation)==(1,1) and source.groups==1,'Only fixed patch residual conv geometry')
            self.in_channels,self.out_channels=source.in_channels,source.out_channels
    def forward(self,x):
        if not self.ste:
            require(not torch.is_grad_enabled(),'Deployment path is inference only')
            return deploy_forward(self,x,reference=self.reference)
        return (_LinearSTE if self.kind=='linear' else _ConvSTE).apply(x,self.weight,self.bias,self.mode,self)


@contextlib.contextmanager
def installed(model,mode,*,masks=None,ste=True,reference=False,variant="rne4"):
    """No BN or PSN changes; same master Parameter objects and state names."""
    require(mode in ('native',)+MODES,'One ordinary arm')
    require((masks is not None)==(mode=='M'),'Explicit masks only for M')
    if mode=='M':require(set(masks)==set(GROUPS),'All16 mask groups, no implicit quota')
    originals={name:model.get_submodule(name) for name in TARGETS}
    require(len(originals)==28 and all(m.bias is None for m in originals.values()),'Exactly28 no-bias original target affines')
    wrappers={}
    try:
        if mode!='native':
            for name,source in originals.items():
                wrapped=STEAffine(source,'linear' if name in LINEARS else 'conv',mode,
                                  selected=masks[group_name(name)] if mode=='M' else None,ste=ste,reference=reference,variant=variant)
                parent,attr=name.rsplit('.',1);setattr(model.get_submodule(parent),attr,wrapped);wrappers[name]=wrapped
        yield wrappers
    finally:
        for name,source in originals.items():
            parent,attr=name.rsplit('.',1);setattr(model.get_submodule(parent),attr,source)
        require(all(model.get_submodule(n) is m for n,m in originals.items()),'Original affine objects restored')


def trainable_parameters(model):
    selected=dict(model.named_parameters())
    for parameter in selected.values():parameter.requires_grad_(True)
    require(len(selected)==477 and len({id(p) for p in selected.values()})==477,'All477 original parameters; quantization scope remains28')
    return selected
