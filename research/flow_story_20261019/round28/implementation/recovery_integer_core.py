"""R28 candidate: exact integer forward, explicit identity-STE weight backward.

No GT, source checkpoint loading, optimizer, model launch, or BN changes here.
All original spatial/T/channel locations survive. Q4 skips low planes only on
complete output9x9 tiles; true tails use Q8. CPU quantization matches R26.
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
MODES=('Q8','Q4')
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


def integer_affine(x,q,scale,tail,mode,*,reference=False):
    """x is full row-major binary activation. Returned affine has no bias."""
    require(mode in MODES and x.ndim==2 and x.shape[1]==q.shape[1] and tail.shape==(len(x),),'Fixed affine inputs')
    require(x.dtype==torch.float32 and bool(torch.isfinite(x).all()) and bool(((x==0)|(x==1)).all()),'Actual inputs must be binary spikes')
    require(q.dtype==torch.int8 and scale.dtype==torch.float32,'Fixed integer encoding')
    binary=x.to(torch.int8)
    limit=max(24,min(MAX_ROWS_PER_KERNEL,MAX_OUTPUT_ELEMENTS//q.shape[0]))
    limit=(limit//8)*8
    drop=0 if mode=='Q8' else 4
    integer=torch.zeros(len(x),q.shape[0],device=x.device,dtype=torch.int32)
    calls=[]
    if reference:
        groups=[(False,torch.nonzero(~tail).flatten()),(True,torch.nonzero(tail).flatten())]
        for is_tail,indices in groups:
            weight=q if is_tail or not drop else ((q.to(torch.int16)>>drop)<<drop).to(torch.int8)
            for start in range(0,len(indices),limit):
                ids=indices[start:start+limit];part,pad=kernel_mm(binary.index_select(0,ids),weight.T)
                integer.index_copy_(0,ids,part)
                calls.append(dict(kind='dense_integer_reference',bit=None,rows=len(ids),padding_rows=pad,scope='tail' if is_tail else 'complete',coefficient=None))
    else:
        indices_tail=torch.nonzero(tail).flatten()
        for bit in range(8):
            indices=indices_tail if bit<drop else None
            n=len(indices_tail) if indices is not None else len(x)
            plane=((q.to(torch.int16)>>bit)&1).to(torch.int8).T.contiguous()
            coefficient=-128 if bit==7 else 1<<bit
            for start in range(0,n,limit):
                ids=indices[start:start+limit] if indices is not None else None
                operand=binary.index_select(0,ids) if ids is not None else binary[start:start+limit]
                part,pad=kernel_mm(operand,plane);part=part*coefficient
                if ids is None:integer[start:start+len(part)].add_(part)
                else:integer.index_add_(0,ids,part)
                calls.append(dict(kind='binary_weight_plane',bit=bit,rows=len(part),padding_rows=pad,
                                  scope='tail' if indices is not None else 'all',coefficient=coefficient))
    output=integer.float()*scale
    audit=dict(mode=mode,reference=reference,rows=len(x),Cin=q.shape[1],Cout=q.shape[0],tail_rows=int(tail.sum()),
               complete_rows=int((~tail).sum()),binary_verified=True,int32_accumulation=True,scale_applications=1,
               logical_row_plane_work=sum(c['rows'] for c in calls) if not reference else None,
               actual_integer_kernel_calls=len(calls),explicit_padded_rows=sum(c['padding_rows'] for c in calls),
               kernel_calls=calls,spatial_rows_retained=True,full_T_and_channels=True,
               low_bit_rule='arithmetic_shift_drop4_no_reclamp_no_offset',quantization_device='cpu',
               software_plane_dtype='int8',cycle_claim=False)
    return output,binary,integer,audit


def effective_weights(q,scale,mode):
    full=q.float()*scale[:,None]
    low=full if mode=='Q8' else ((q.to(torch.int16)>>4)<<4).float()*scale[:,None]
    return full,low


class _LinearSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias,mode,holder):
        t,b,h,w,c=x.shape;tail=tail_rows(t,b,h,w,x.device);q,scale=quantize(weight)
        y,binary,integer,audit=integer_affine(x.reshape(-1,c),q,scale,tail,mode)
        if bias is not None:y=y+bias
        ctx.save_for_backward(binary,q,scale,tail);ctx.mode=mode;ctx.shape=tuple(x.shape);ctx.has_bias=bias is not None
        holder.last_audit=audit;holder.last_integer=integer.detach() if holder.capture_integer else None
        return y.reshape(t,b,h,w,weight.shape[0])
    @staticmethod
    def backward(ctx,grad):
        x,q,scale,tail=ctx.saved_tensors;g=grad.reshape(-1,q.shape[0]);full,low=effective_weights(q,scale,ctx.mode)
        dx=torch.empty(len(x),q.shape[1],device=g.device,dtype=g.dtype)
        dx[tail]=g[tail]@full;dx[~tail]=g[~tail]@low
        # Identity STE for q(W)*scale: d effective W / d master W := 1.
        # Scale/rounding/clipping are not differentiated; this is declared surrogate math.
        dw=g.T@x.float();db=g.sum(0) if ctx.has_bias else None
        return dx.reshape(ctx.shape),dw,db,None,None


class _ConvSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias,mode,holder):
        require(x.ndim==5,'Actual original multi-step Conv2d input')
        t,b,c,h,w=x.shape;require(tuple(weight.shape[-2:])==(3,3),'Fixed ordinary3x3 conv')
        tail=tail_rows(t,b,h,w,x.device);q,scale=quantize(weight)
        flat=x.reshape(t*b,c,h,w)
        require(x.dtype==torch.float32 and bool(torch.isfinite(x).all()) and bool(((x==0)|(x==1)).all()),'Actual conv operand binary')
        # Native stride1,pad1,groups1,dilation1; zero padding is exact binary0.
        rows=F.unfold(flat,kernel_size=3,padding=1).transpose(1,2).reshape(-1,c*9)
        y,_,integer,audit=integer_affine(rows,q,scale,tail,mode)
        if bias is not None:y=y+bias
        ctx.save_for_backward(x.to(torch.int8),q,scale,tail);ctx.mode=mode;ctx.weight_shape=tuple(weight.shape);ctx.has_bias=bias is not None
        audit.update(operation='Conv2d',kernel=[3,3],stride=[1,1],padding=[1,1],groups=1,dilation=[1,1],
                     unfold_elements=rows.numel(),output_tile_origin='original_output_yx',padded_zero_input_taps_included=True)
        holder.last_audit=audit;holder.last_integer=integer.detach() if holder.capture_integer else None
        return y.reshape(t,b,h,w,weight.shape[0]).permute(0,1,4,2,3).contiguous()
    @staticmethod
    def backward(ctx,grad):
        x,q,scale,tail=ctx.saved_tensors;t,b,c,h,w=x.shape;co=q.shape[0]
        g=grad.reshape(t*b,co,h,w);full,low=effective_weights(q,scale,ctx.mode)
        full,low=full.reshape(ctx.weight_shape),low.reshape(ctx.weight_shape)
        mask=tail.reshape(t*b,1,h,w)
        if ctx.mode=='Q8':
            dx=torch.nn.grad.conv2d_input((t*b,c,h,w),full,g,padding=1)
        else:
            dx=torch.nn.grad.conv2d_input((t*b,c,h,w),full,g*mask,padding=1)+torch.nn.grad.conv2d_input((t*b,c,h,w),low,g*(~mask),padding=1)
        dw=torch.nn.grad.conv2d_weight(x.reshape(t*b,c,h,w).float(),ctx.weight_shape,g,padding=1)
        db=g.sum((0,2,3)) if ctx.has_bias else None
        return dx.reshape(t,b,c,h,w),dw,db,None,None


def deploy_forward(module,x,*,reference=False):
    """Autograd-free production entry, same integer math, independent call path."""
    q,scale=quantize(module.weight)
    if module.kind=='linear':
        t,b,h,w,c=x.shape;rows=x.reshape(-1,c)
    else:
        t,b,c,h,w=x.shape
        rows=F.unfold(x.reshape(t*b,c,h,w),3,padding=1).transpose(1,2).reshape(-1,c*9)
    y,_,integer,audit=integer_affine(rows,q,scale,tail_rows(t,b,h,w,x.device),module.mode,reference=reference)
    if module.bias is not None:y=y+module.bias
    module.last_audit=audit;module.last_integer=integer.detach() if module.capture_integer else None
    y=y.reshape(t,b,h,w,module.weight.shape[0])
    return y if module.kind=='linear' else y.permute(0,1,4,2,3).contiguous()


class STEAffine(nn.Module):
    def __init__(self,source,kind,mode,*,ste=True,reference=False):
        super().__init__();require(mode in MODES,'Only predeclared Q8/Q4')
        self.weight=source.weight;self.bias=source.bias
        self.kind,self.mode=kind,mode;self.ste,self.reference=ste,reference
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
def installed(model,mode,*,ste=True,reference=False):
    """No BN or PSN changes; same master Parameter objects and state names."""
    require(mode in ('native',)+MODES,'One ordinary arm')
    originals={name:model.get_submodule(name) for name in TARGETS}
    require(len(originals)==28 and all(m.bias is None for m in originals.values()),'Exactly28 no-bias original target affines')
    wrappers={}
    try:
        if mode!='native':
            for name,source in originals.items():
                wrapped=STEAffine(source,'linear' if name in LINEARS else 'conv',mode,ste=ste,reference=reference)
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
