"""One q11 choice permits exact F(2,3) inside existing 19x13 multipliers."""
from pathlib import Path
import json
import sys
import numpy as np

H = Path(__file__).resolve().parent
S = H.parent / 'spatial_r16_integer'
sys.path.insert(0, str(S))
from export import factor_chain, rne_shift


def main():
    with np.load(S / 'factors.npz') as ar:
        f = {k: ar[k].copy() for k in ar.files}
    base = H.parents[1]
    with np.load(base / 'open_fusion_execution/major_operator_fusions_20260913/decomposition_owned/results/spatial_r16_w8.npz') as ar:
        effective = ar['second'].astype(np.float64)[:, :, 0, :] * f['first_scale'][None, :, None] * f['theta']
    scale = np.max(abs(effective), axis=(1, 2)) / 1023
    q = np.rint(effective / scale[:, None, None]).astype(np.int64)
    q1 = f['q1'].astype(np.int64)
    expanded = np.einsum('orx,rcy->ocyx', q, q1)
    a = np.rint(scale * f['BN_gain'] * (1 << 40)).astype(np.int64)
    zabs = np.maximum(-f['z_lower'], f['z_upper'])
    lo = np.minimum(expanded, 0).sum((1, 2, 3))
    hi = np.maximum(expanded, 0).sum((1, 2, 3))
    prefix = (abs(q) * zabs[None, :, None]).sum((1, 2))
    wide = np.maximum(-lo, hi) * abs(a) + ((1 << 31) + abs(f['b_q20'].astype(np.int64))) * (1 << 20)
    transformed = np.stack([2*q[:, :, 0], q.sum(2), q[:, :, 0]-q[:, :, 1]+q[:, :, 2], 2*q[:, :, 2]], axis=2)
    # D=[d0-d2,d1+d2,d2-d1,d1-d3]; output equations divide by 2.
    assert abs(q).max() <= 1023 and abs(transformed).max() <= 3069
    transformed_prefix = (abs(transformed) * (2*zabs)[None, :, None]).sum(1)
    reconstruct_bound = np.maximum(transformed_prefix[:, 0]+transformed_prefix[:, 1]+transformed_prefix[:, 2],
                                   transformed_prefix[:, 1]+transformed_prefix[:, 2]+transformed_prefix[:, 3])
    assert (2*zabs).max() < (1 << 15) and reconstruct_bound.max() < (1 << 31)
    assert wide.max() < (1 << 63) and abs(a).max() < (1 << 31)
    f.update(q2=q.astype(np.int16), q2_bits=np.array(11), output_scale=scale, a_q40=a.astype(np.int32),
             expanded_int32=expanded.astype(np.int32), p_lower=lo, p_upper=hi,
             p_any_prefix_abs=prefix, wide_abs_bound=wide, winograd_q2=transformed.astype(np.int16),
             winograd_M_prefix_abs=transformed_prefix, winograd_reconstruction_abs=reconstruct_bound)
    np.savez_compressed(H / 'factors.npz', **f)
    with np.load(S / 'gold_tiles.npz') as ar:
        gold = {k: ar[k].copy() for k in ar.files if k not in ('qdq64_i24', 'integer_ideal_i24')}
    ps, wides, i24s, m_values = [], [], [], []
    for index, words in enumerate(gold['source_words']):
        g = ((words[None] >> np.arange(10)[:, None, None, None]) & 1).astype(np.int64)
        z, p = factor_chain(g, q1, q)
        assert np.array_equal(z, gold['z_halo_int'][index])
        d = np.stack([z[:, :, :, 0]-z[:, :, :, 2], z[:, :, :, 1]+z[:, :, :, 2],
                      z[:, :, :, 2]-z[:, :, :, 1], z[:, :, :, 1]-z[:, :, :, 3]], axis=3)
        M = np.einsum('trym,orm->toym', d, transformed)
        twice = np.stack([M[:, :, :, 0]+M[:, :, :, 1]+M[:, :, :, 2],
                           M[:, :, :, 1]-M[:, :, :, 2]-M[:, :, :, 3]], axis=3)
        assert not np.any(twice & 1) and np.array_equal(twice // 2, p)
        wide_i = p * a[None, :, None, None] + (gold['J_q20'][index].astype(np.int64) + f['b_q20'][None, :, None, None]) * (1 << 20)
        ps.append(p.astype(np.int32)); wides.append(wide_i); i24s.append(rne_shift(wide_i, 26, 24).astype(np.int32)); m_values.append(M)
    gold.update(p_int=np.stack(ps), wide_int64=np.stack(wides), i24=np.stack(i24s))
    np.savez_compressed(H / 'gold_tiles.npz', **gold)
    report = dict(passed=True, q2_bits=11, transformed_q2_range=[int(transformed.min()), int(transformed.max())],
                  transformed_Z_abs=int((2*zabs).max()), M_prefix_abs=int(transformed_prefix.max()),
                  output_reconstruction_abs=int(reconstruct_bound.max()),
                  raw_values=int(gold['p_int'].size), tiles=len(ps), winograd_differences=0, odd_reconstructions=0,
                  network_AEE='not run', RTL='not run', q13_quality_not_inherited=True, training=False,
                  bit_choice='one mathematically bounded q11 choice; not a sweep',
                  real_M_range=[int(np.min(m_values)), int(np.max(m_values))])
    (H / 'stats.json').write_text(json.dumps(report, indent=2) + '\n')
    arrays = {k: dict(shape=list(f[k].shape), dtype=str(f[k].dtype), values=f[k].tolist())
              for k in ['q1', 'q2', 'winograd_q2', 'a_q40', 'b_q20', 'output_scale']}
    (H / 'frozen_parameters.json').write_text(json.dumps(dict(arrays=arrays), separators=(',', ':')) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
