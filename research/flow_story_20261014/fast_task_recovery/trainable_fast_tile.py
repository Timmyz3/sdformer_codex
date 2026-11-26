"""New differentiable, fixed-mask adapter; original model/adapters remain frozen.

Training uses the full per-tile native teacher and fast paths. The native
branch is detached; only the original LN+Linear fast parameters receive grads.
Actual inference computes only the selected branch with identical tile shapes.
No residual is added here; the untouched native block adds it exactly once.
"""
import contextlib
import hashlib
import math

import torch

LAYER = 'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.0.mlp'
SHAPE = (10, 1, 30, 40, 384)


def require(ok, message):
    if not ok: raise ValueError(message)


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def tile_core(native, fast, u, keep, mode):
    """Numerical kernel, tested with small synthetic C/H/W; production wrapper below fixes native shape."""
    require(mode in ('dual', 'actual'), 'Unknown branch execution')
    require(u.ndim == 5 and tuple(u.shape[:2]) == (10, 1) and not u.requires_grad,
            'Frozen upstream complete T10/B1 required')
    require(u.dtype == torch.float32 and bool(torch.isfinite(u).all()), 'Finite FP32 input required')
    t,b,h,w,c = u.shape
    require(isinstance(keep,torch.Tensor) and keep.dtype == torch.bool and
            tuple(keep.shape) == (math.ceil(h/9),math.ceil(w/9)), 'Whole tile mask required')
    decisions=keep.detach().cpu().tolist()
    require(not any(p.requires_grad for p in native.parameters()) and
            all(not m.training for m in native.modules()) and all(not m.training for m in fast.modules()),
            'Frozen native and dropout-off fast required')
    rows=[];loss_sum=None;loss_elements=0;slow_points=0;fast_points=0
    observed={'slow_fc1_rows':0,'slow_fc2_rows':0,'fast_fc_rows':0};handles=[]
    for module,key in ((native.fc1,'slow_fc1_rows'),(native.fc2,'slow_fc2_rows'),(fast.fc1[1],'fast_fc_rows')):
        def count(m,inputs,k=key):observed[k]+=inputs[0].numel()//inputs[0].shape[-1]
        handles.append(module.register_forward_pre_hook(count))
    try:
        for gy,row in enumerate(decisions):
            pieces=[]
            for gx,selected in enumerate(row):
                local=u[:,:,gy*9:min((gy+1)*9,h),gx*9:min((gx+1)*9,w),:].contiguous()
                points=local.shape[2]*local.shape[3]
                slow_points+=points*selected;fast_points+=points*(not selected)
                slow=quick=None
                if mode=='dual' or selected:
                    with torch.no_grad():slow=native(local)
                    require(slow.shape==local.shape and bool(torch.isfinite(slow).all()),'Teacher shape/finite failure')
                if mode=='dual' or not selected:
                    quick=fast(local)
                    require(quick.shape==local.shape and bool(torch.isfinite(quick).all()),'Fast shape/finite failure')
                if mode=='dual' and not selected:
                    error=(quick-slow.detach()).square().sum()
                    loss_sum=error if loss_sum is None else loss_sum+error
                    loss_elements+=local.numel()
                pieces.append(slow if selected else quick)
            rows.append(torch.cat(pieces,dim=3))
        output=torch.cat(rows,dim=2).contiguous()
    finally:
        for handle in handles:handle.remove()
    require(output.shape==u.shape,'Output shape differs')
    require(observed['slow_fc1_rows']==observed['slow_fc2_rows']==t*(h*w if mode=='dual' else slow_points),
            'Observed slow FC service differs')
    require(observed['fast_fc_rows']==t*(h*w if mode=='dual' else fast_points),'Observed fast service differs')
    mse=loss_sum/loss_elements if loss_elements else None
    return output,mse,{'mode':mode,'shape_TBHWC':list(u.shape),'keep':decisions,
        'slow_points':slow_points,'fast_points':fast_points,'MSE_elements':loss_elements,
        'observed':observed,'teacher_detached':True,'external_residual_added_here':False,
        'input_sha256':tensor_sha(u),'output_sha256':tensor_sha(output),
        'cost_scope':'dual training computes both candidates; actual only selected; counts are not PE cycles'}


class RecoveryTileMLP(torch.nn.Module):
    def __init__(self,native,fast,keep,mode,expected_u_sha=None):
        super().__init__()
        require(type(native).__name__=='MS_Spiking_Mlp' and native.norm_layer=='BN','Native BN MLP required')
        require((native.fc1.in_features,native.fc1.out_features,native.fc2.in_features,native.fc2.out_features)==
                (384,1536,1536,384),'Native stage2 widths differ')
        for sn in (native.sn1.spiking_neuron,native.sn2.spiking_neuron):
            require(type(sn).__name__=='PSN' and sn.T==10,'Native complete T10 PSN required')
        for m in native.modules():
            if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):
                require(not m.training and m.track_running_stats and m.running_mean is not None and m.running_var is not None,
                        'Only private frozen BN is legal')
        require(type(fast).__name__=='FastBranch' and isinstance(fast.fc1[0],torch.nn.LayerNorm) and
                isinstance(fast.fc1[1],torch.nn.Linear) and fast.fc1[1].in_features==fast.fc1[1].out_features==384,
                'Original C384 LNLinear fast topology required')
        self.native,self.fast=native,fast
        self.keep,self.mode,self.expected_u_sha=keep,mode,expected_u_sha
        self.records=[];self.mse=None
        self.eval()

    def forward(self,u):
        require(tuple(u.shape)==SHAPE,'Declared stage2 shape required')
        require(not self.records,'One root forward per wrapper required')
        if self.expected_u_sha is not None:
            require(tensor_sha(u)==self.expected_u_sha,'Frozen upstream input changed')
        out,self.mse,rec=tile_core(self.native,self.fast,u,self.keep,self.mode)
        self.records.append(rec)
        return out


@contextlib.contextmanager
def replace(model,fast,keep,mode,expected_u_sha=None):
    require(all(not m.training for m in model.modules()),'Backbone eval required')
    require(not any(p.requires_grad for p in model.parameters()),'Entire original backbone must be frozen')
    parent_name,child=LAYER.rsplit('.',1);parent=model.get_submodule(parent_name)
    require(getattr(parent,'cnf',None)=='ADD' and not getattr(parent,'use_checkpoint',False),'Native ADD/no-checkpoint required')
    original=getattr(parent,child)
    wrapper=RecoveryTileMLP(original,fast,keep,mode,expected_u_sha)
    try:
        setattr(parent,child,wrapper);yield wrapper
    finally:setattr(parent,child,original)
