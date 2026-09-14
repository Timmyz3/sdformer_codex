"""Three surviving Winograd products, storing doubled outputs explicitly.

This is a two-phase spatial operator, not a shared ordinary 1x3 convolution.
Its independent coefficients and consumer scale define a separate lossy model.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch_integer import SpatialR16Integer


class Phase3Integer(SpatialR16Integer):
    def __init__(self, filename, device='cpu'):
        torch.nn.Module.__init__(self)
        with np.load(filename) as file:
            arrays = {key: file[key].copy() for key in file.files}
        assert 'q2' not in arrays, 'Phase3 coefficients must not masquerade as ordinary g'
        self.theta = float(arrays['theta'])
        self.register_buffer('q1_fp64', torch.as_tensor(arrays['q1'][:, :, :, None], dtype=torch.float64, device=device))
        self.register_buffer('physical_coeff3', torch.as_tensor(arrays['physical_coeff3'], dtype=torch.float64, device=device))
        for name in ['a_q40', 'b_q20', 'z_lower', 'z_upper', 'p_lower', 'p_upper']:
            self.register_buffer(name, torch.as_tensor(arrays[name].astype(np.int64), device=device)[None, :, None, None])
        for name in ['output_scale', 'bias']:
            self.register_buffer(name, torch.as_tensor(arrays[name], dtype=torch.float64, device=device)[None, :, None, None])

    @torch.no_grad()
    def raw_p(self, source, *, tile=False, return_z=False, check=True):
        x, leading = self._flat(source)
        if check:
            assert bool(((x == 0) | (x == self.theta)).all())
        if tile:
            assert tuple(x.shape[-2:]) == (4, 4)
        else:
            assert x.shape[-1] % 2 == 0
        with torch.backends.cudnn.flags(enabled=False):
            z = F.conv2d(x.ne(0).double(), self.q1_fp64, padding=0 if tile else (1, 0))
            padded = z if tile else F.pad(z, (1, 1, 0, 0))
            width = padded.shape[-1] - 2
            d0, d1, d2, d3 = [padded[:, :, :, offset:offset+width:2] for offset in range(4)]
            transformed = [d0-d2, d1+d2, d1-d3]
            products = [F.conv2d(value, self.physical_coeff3[:, :, m, None, None])
                        for m, value in enumerate(transformed)]
            even, odd = products[0]+products[1], products[1]-products[2]
            p = torch.stack([even, odd], dim=-1).flatten(-2)
        if check:
            assert bool((z == z.round()).all()) and bool(((z >= self.z_lower) & (z <= self.z_upper)).all())
            assert bool((p == p.round()).all()) and bool(((p >= self.p_lower) & (p <= self.p_upper)).all())
        p = self._restore(p.to(torch.int64), leading)
        return (p, self._restore(z.to(torch.int64), leading)) if return_z else p
