"""GPU forward bridge. Dense torch kernels verify function, not sparse speed."""
from pathlib import Path
import numpy as np

TARGET='sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.0.conv2.0'

def install_conv(module,filename,trainable=False):
    """Return (factor nn.Module, restore callable). Accept 4D or T,B,C,H,W.

    Load e.g. results/spatial_nm_r8_w32.npz or flat_nm_r8_w32.npz.
    FP32 input required. The outer real BN, residual, neuron and theta stay live.
    Saved W8 factors are dequantized floats; this is not an INT8 kernel.
    """
    import torch
    from torch import nn
    import torch.nn.functional as F
    if (tuple(module.kernel_size),tuple(module.stride),tuple(module.padding),tuple(module.dilation),module.groups)!=((3,3),(1,1),(1,1),(1,1),1):
        raise ValueError('Bridge admits stride1 3x3 pad1 ungrouped Conv only')
    with np.load(filename) as z:arrays={k:z[k].copy() for k in z.files}
    old=module.forward
    class FactorConv(nn.Module):
        def __init__(self):
            super().__init__()
            for k in ['first','core','second','residual','weight','bias']:
                if k in arrays:
                    self.register_parameter(k,nn.Parameter(torch.as_tensor(arrays[k],device=module.weight.device,dtype=module.weight.dtype),requires_grad=trainable))
            self.coefficient_bits=int(arrays.get('coefficient_bits',32))
        def forward(self,x):
            shape=x.shape
            if x.ndim==5:z=x.flatten(0,1)
            elif x.ndim==4:z=x
            else:raise ValueError(str(tuple(shape)))
            if hasattr(self,'weight'):
                y=F.conv2d(z,self.weight,self.bias,padding=1)
            else:
                y=F.conv2d(z,self.first,padding=tuple(k//2 for k in self.first.shape[-2:]))
                if hasattr(self,'core'):y=F.conv2d(y,self.core,padding=1)
                y=F.conv2d(y,self.second,padding=tuple(k//2 for k in self.second.shape[-2:]))
                if hasattr(self,'residual'):y=y+F.conv2d(z,self.residual,padding=1)
                y=y+self.bias[None,:,None,None]
            return y.reshape(shape[0],shape[1],*y.shape[1:]) if x.ndim==5 else y
    factor=FactorConv();module.forward=factor.forward
    def restore():module.forward=old
    return factor,restore
