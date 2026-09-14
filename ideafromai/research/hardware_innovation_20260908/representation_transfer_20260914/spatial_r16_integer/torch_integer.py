"""Exact integer spatial-R16 forward using FP64 conv as an execution oracle.

This module does not install hooks or run a network. The caller owns the live
source/identity capture and reinjects I24 / 16384 at the residual-block output.
No floating BatchNorm/add result may replace consume()'s integer result.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


class SpatialR16Integer(torch.nn.Module):
    def __init__(self, filename=None, device='cpu'):
        super().__init__()
        filename = filename or Path(__file__).with_name('factors.npz')
        with np.load(filename) as f:
            arrays = {k: f[k].copy() for k in f.files}
        self.theta = float(arrays['theta'])
        self.register_buffer('q1_fp64', torch.as_tensor(arrays['q1'][:, :, :, None], dtype=torch.float64, device=device))
        self.register_buffer('q2_fp64', torch.as_tensor(arrays['q2'][:, :, None, :], dtype=torch.float64, device=device))
        for name in ['a_q40', 'b_q20', 'z_lower', 'z_upper', 'p_lower', 'p_upper']:
            self.register_buffer(name, torch.as_tensor(arrays[name].astype(np.int64), device=device)[None, :, None, None])
        self.register_buffer('output_scale', torch.as_tensor(arrays['output_scale'], dtype=torch.float64, device=device)[None, :, None, None])
        self.register_buffer('bias', torch.as_tensor(arrays['bias'], dtype=torch.float64, device=device)[None, :, None, None])

    @staticmethod
    def _flat(x):
        if x.ndim == 4:
            return x, None
        if x.ndim == 5:
            return x.flatten(0, 1), tuple(x.shape[:2])
        raise ValueError('Expected [N,C,H,W] or [T,B,C,H,W]')

    @staticmethod
    def _restore(x, leading):
        return x if leading is None else x.reshape(*leading, *x.shape[1:])

    @torch.no_grad()
    def raw_p(self, source, *, tile=False, return_z=False, check=True):
        """Live source values must be {0, theta}; theta is already in q2.

        Full frame: stride1, vertical pad(1,0), horizontal pad(0,1).
        tile=True: native 4x4 source halo -> Z 2x4 -> p 2x2, no padding;
        the caller supplies zero source words beyond the physical image edge.
        """
        x, leading = self._flat(source)
        if x.shape[1] != 96 or (tile and tuple(x.shape[-2:]) != (4, 4)):
            raise ValueError('Expected C96 and a 4x4 native halo in tile mode')
        if check and not bool(((x == 0) | (x == self.theta)).all()):
            raise ValueError('Source is not the fixed AT-LIF {0, theta} domain')
        g = x.ne(0).to(torch.float64)
        # Full-frame cuDNN may select a transformed convolution algorithm.
        # Parent A800 audit observed 2.728e-12 roundoff despite FP64 operands;
        # disable only this oracle's cuDNN path instead of rounding it away.
        with torch.backends.cudnn.flags(enabled=False):
            z = F.conv2d(g, self.q1_fp64, padding=0 if tile else (1, 0))
            p = F.conv2d(z, self.q2_fp64, padding=0 if tile else (0, 1))
        if check:
            assert bool((z == z.round()).all()) and bool(((z >= self.z_lower) & (z <= self.z_upper)).all())
            assert bool((p == p.round()).all()) and bool(((p >= self.p_lower) & (p <= self.p_upper)).all())
        # All products and arbitrary partial sums are exact integers < 2**53.
        # Casting is storage conversion, not a latent quantizer.
        p = self._restore(p.to(torch.int64), leading)
        return (p, self._restore(z.to(torch.int64), leading)) if return_z else p

    @torch.no_grad()
    def consume(self, p, identity, *, return_intermediates=False, check=True):
        """Return signed I24 integer codes; all tie handling is RNE.

        identity must be the original IEEE binary32 residual input. NaN/Inf
        are outside this contract; finite overflow saturates at J32 first.
        """
        pf, leading = self._flat(p)
        ident, identity_leading = self._flat(identity)
        if pf.dtype != torch.int64 or ident.dtype != torch.float32 or pf.shape != ident.shape or leading != identity_leading:
            raise ValueError('Expected matching int64 p and original float32 identity')
        if check:
            if not bool(torch.isfinite(ident).all()):
                raise ValueError('NaN/Inf identity is outside the finite deployment contract')
            assert bool(((pf >= self.p_lower) & (pf <= self.p_upper)).all())
        j = (ident.double() * (1 << 20)).round().clamp(-(1 << 31), (1 << 31) - 1).to(torch.int64)
        wide = pf * self.a_q40 + ((j + self.b_q20) << 20)
        quotient = torch.div(wide, 1 << 26, rounding_mode='floor')
        remainder = wide - quotient * (1 << 26)
        rounded = quotient + ((remainder > (1 << 25)) | ((remainder == (1 << 25)) & ((quotient & 1) != 0))).to(torch.int64)
        i24 = rounded.clamp(-(1 << 23), (1 << 23) - 1)
        if return_intermediates:
            return {k: self._restore(v, leading) for k, v in [('p', pf), ('J', j), ('wide', wide), ('i24', i24)]}
        return self._restore(i24, leading)

    @torch.no_grad()
    def forward(self, source, identity, *, tile=False, return_intermediates=False, check=True):
        return self.consume(self.raw_p(source, tile=tile, check=check), identity, return_intermediates=return_intermediates, check=check)

    @staticmethod
    def reader_value(i24):
        """Exactly representable binary32 value for the existing I24 reader."""
        return i24.to(torch.float32) / 16384


def validate_cpu():
    """Replay all existing true-input gold tiles on CPU; no GPU/network work."""
    import json
    torch.set_num_threads(1)
    model = SpatialR16Integer()
    with np.load(Path(__file__).with_name('gold_tiles.npz')) as gold:
        count = 0
        for index in range(len(gold['tile_ids'])):
            words = gold['source_words'][index]
            source = ((words[None] >> np.arange(10)[:, None, None, None]) & 1).astype(np.float32) * model.theta
            identity = gold['identity_fp32_bits'][index].copy().view(np.float32)
            p, z = model.raw_p(torch.from_numpy(source), tile=True, return_z=True)
            actual = model.consume(p, torch.from_numpy(identity), return_intermediates=True)
            for key, field in [('p', 'p_int'), ('J', 'J_q20'), ('wide', 'wide_int64'), ('i24', 'i24')]:
                assert np.array_equal(actual[key].numpy(), gold[field][index]), (int(gold['tile_ids'][index]), key)
            assert np.array_equal(z.numpy(), gold['z_halo_int'][index])
            # The real reader's RNE cannot change an already exact I24 value.
            assert torch.equal((model.reader_value(actual['i24']).double() * 16384).round().to(torch.int64), actual['i24'])
            count += p.numel()
        # Exercise T,B layout without changing the source or arithmetic.
        actual5 = model(torch.from_numpy(source)[:, None], torch.from_numpy(identity)[:, None], tile=True)
        assert torch.equal(actual5[:, 0], actual['i24'])
    result = dict(passed=True, device='cpu', torch=torch.__version__, tiles=index + 1, raw_values=count,
                  checked=['Z', 'p', 'J', 'wide', 'I24', 'reader_reinjection', 'T_B_layout'], differences=0,
                  network_AEE='not run', training=False, format_search=False)
    Path(__file__).with_name('torch_cpu_stats.json').write_text(json.dumps(result, separators=(',', ':')) + '\n')
    print(json.dumps(result, separators=(',', ':')))


if __name__ == '__main__':
    validate_cpu()
