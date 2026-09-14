"""Fixed U2-zero phase3 model, CPU integer gold, no activity-based selection."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import importlib.util
import json
import numpy as np

H = Path(__file__).resolve().parent
P = H.parent
N = P.parent
S = N / 'spatial_winograd_inputs'
spec = importlib.util.spec_from_file_location('integer_gold_export', N / 'spatial_r16_integer/export.py')
integer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(integer)
factor_chain, rne_shift, identity_to_j = integer.factor_chain, integer.rne_shift, integer.identity_to_j


def read_npz(path):
    with np.load(path) as f:
        return {k: f[k].copy() for k in f.files}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def error(a, b, relative=True):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    d = a - b
    r = dict(RMSE=float(np.sqrt(np.mean(d*d))), MAE=float(np.mean(abs(d))),
             max_abs=float(np.max(abs(d))), different=int(np.count_nonzero(d)))
    if relative:
        den = float(np.linalg.norm(b.ravel()))
        r['relative_L2'] = float(np.linalg.norm(d.ravel()) / den) if den else None
    return r


def transform(q):
    return np.stack([2*q[:, :, 0], q.sum(2), q[:, :, 0]-q[:, :, 1]+q[:, :, 2],
                     2*q[:, :, 2]], axis=2)


def gold_from_p(g, p, f):
    j = identity_to_j(g['identity_fp32_bits'].view(np.float32))
    assert np.array_equal(j, g['J_q20'])
    a = f['a_q40'].astype(np.int64)[None, None, :, None, None]
    b = f['b_q20'].astype(np.int64)[None, None, :, None, None]
    wide = p*a + (j+b)*(1 << 20)
    i24 = rne_shift(wide, 26, 24).astype(np.int32)
    out = {k: g[k].copy() for k in ['source_words', 'output_origin_yx',
                                    'z_halo_int', 'identity_fp32_bits', 'J_q20']}
    for k in ['tile_ids', 'frame_indices', 'record_indices']:
        if k in g:
            out[k] = g[k].copy()
    out.update(p_int=p.astype(np.int32), wide_int64=wide, i24=i24)
    return out


def local_errors(out, g, f, parent, multiplier):
    p = out['p_int'].astype(np.int64)
    pp = g['p_int'].astype(np.int64)
    s = (f['output_scale']*f['BN_gain'])[None, None, :, None, None]
    sp = (parent['output_scale']*parent['BN_gain'])[None, None, :, None, None]
    physical = p*s-pp*sp
    r = dict(raw_error_vs_parent_at_same_integer_scale=error(p, pp*multiplier),
             raw_parent_multiplier=multiplier,
             bn_scaled_output_error=error(physical, np.zeros_like(physical), False),
             wide_error_vs_q11_parent=error(out['wide_int64'], g['wide_int64']),
             i24_error_vs_q11_parent=error(out['i24'], g['i24']),
             i24_saturations=int(np.count_nonzero((out['i24'] == -(1 << 23)) |
                                                (out['i24'] == (1 << 23)-1))),
             parent_i24_saturations=int(np.count_nonzero((g['i24'] == -(1 << 23)) |
                                                       (g['i24'] == (1 << 23)-1))))
    r['phase_errors'] = {str(x): dict(
        bn_scaled_output_error=error(physical[..., x], np.zeros_like(physical[..., x]), False),
        i24_error_vs_q11_parent=error(out['i24'][..., x], g['i24'][..., x])) for x in range(2)}
    return r


def emit_phase(g, f, parent):
    q1 = f['q1'].astype(np.int64)
    q = parent['q2'].astype(np.int64)
    coeff = f['physical_coeff3'].astype(np.int64)
    W = f['expanded_phase_int32'].astype(np.int64)
    mp_bound = f['M_any_prefix_abs'].astype(np.int64)
    rec_bound = f['p_any_prefix_abs'].astype(np.int64)
    ps, Ms, per = [], [], []
    max_d = max_m = max_rec = 0
    for idx, words in enumerate(g['source_words']):
        gates = ((words[None] >> np.arange(10)[:, None, None, None]) & 1).astype(np.int64)
        z, mother_p = factor_chain(gates, q1, q)
        assert np.array_equal(z, g['z_halo_int'][idx])
        assert np.array_equal(mother_p, g['p_int'][idx])
        D = np.stack([z[:, :, :, 0]-z[:, :, :, 2], z[:, :, :, 1]+z[:, :, :, 2],
                      z[:, :, :, 1]-z[:, :, :, 3]], axis=3)
        max_d = max(max_d, int(abs(D).max()))
        M = np.zeros((10, 96, 2, 3), np.int64)
        p = np.zeros((10, 96, 2, 2), np.int64)
        for stripe in range(2):
            ms = np.zeros_like(M)
            for r in range(8*stripe, 8*stripe+8):
                product = D[:, None, r, :, :] * coeff[None, :, r, None, :]
                assert abs(product).max() < 2**31
                ms += product
                assert np.all(abs(ms) <= mp_bound[None, :, None, :])
                max_m = max(max_m, int(abs(ms).max()))
            M += ms
            assert np.all(abs(M) <= mp_bound[None, :, None, :])
            stripe_p = np.stack([ms[:, :, :, 0]+ms[:, :, :, 1],
                                 ms[:, :, :, 1]-ms[:, :, :, 2]], axis=3)
            assert np.all(abs(stripe_p) <= rec_bound[None, :, None, None])
            p += stripe_p
            assert np.all(abs(p) <= rec_bound[None, :, None, None])
            max_rec = max(max_rec, int(abs(stripe_p).max()), int(abs(p).max()))
        assert np.array_equal(M, np.einsum('tryk,ork->toyk', D, coeff))
        direct = np.stack([np.stack([
            gates[:, :, y:y+3, :].reshape(10, 1152) @ W[x].reshape(96, 1152).T
            for x in range(2)], axis=2) for y in range(2)], axis=2)
        assert np.array_equal(p, direct), idx
        omitted_M2 = np.einsum('try,or->toy', z[:, :, :, 2]-z[:, :, :, 1],
                               q[:, :, 0]-q[:, :, 1]+q[:, :, 2])
        assert np.array_equal(p[..., 0]-2*mother_p[..., 0], -omitted_M2)
        assert np.array_equal(p[..., 1]-2*mother_p[..., 1], omitted_M2)
        assert np.array_equal(p.sum(3), 2*mother_p.sum(3))
        for x in range(2):
            assert np.all(p[..., x] >= f['p_phase_lower'][x][None, :, None])
            assert np.all(p[..., x] <= f['p_phase_upper'][x][None, :, None])
        ps.append(p.astype(np.int32))
        Ms.append(M.astype(np.int32))
        per.append(dict(record_index=idx, tile_id=int(g['tile_ids'][idx]),
                        raw_p2_odd_values=int(np.count_nonzero(p & 1))))
    out = gold_from_p(g, np.stack(ps).astype(np.int64), f)
    out['physical_M_int'] = np.stack(Ms)
    stats = dict(passed=True, records=len(ps), raw_values=int(out['p_int'].size),
                 phase_expanded_differences=0, full_vs_stripe_differences=0,
                 pair_sum_preservation_differences=0, omitted_M2_error_identity_differences=0,
                 source_Z_parent_differences=0, identity_J_differences=0,
                 raw_p2_odd_values=int(np.count_nonzero(out['p_int'] & 1)),
                 observed=dict(D_abs=max_d, M_stripe_prefix_abs=max_m,
                               stripe_and_accumulated_p2_abs=max_rec))
    stats.update(local_errors(out, g, f, parent, 2))
    for idx, row in enumerate(per):
        row['i24_error_vs_q11_parent'] = error(out['i24'][idx], g['i24'][idx])
    return out, stats, per


def emit_native_sequence(g, f, parent):
    q1, q = f['q1'].astype(np.int64), f['q2'].astype(np.int64)
    U = transform(q)
    W = np.einsum('orx,rcy->ocyx', q, q1)
    ps, Ms = [], []
    for idx, words in enumerate(g['source_words']):
        gates = ((words[None] >> np.arange(10)[:, None, None, None]) & 1).astype(np.int64)
        z, p = factor_chain(gates, q1, q)
        assert np.array_equal(z, g['z_halo_int'][idx])
        direct = np.stack([np.stack([
            gates[:, :, y:y+3, x:x+3].reshape(10, 864) @ W.reshape(96, 864).T
            for x in range(2)], axis=2) for y in range(2)], axis=2)
        assert np.array_equal(p, direct)
        D = np.stack([z[:, :, :, 0]-z[:, :, :, 2], z[:, :, :, 1]+z[:, :, :, 2],
                      z[:, :, :, 2]-z[:, :, :, 1], z[:, :, :, 1]-z[:, :, :, 3]], axis=3)
        M = np.einsum('trym,orm->toym', D, U)
        twice = np.stack([M[:, :, :, 0]+M[:, :, :, 1]+M[:, :, :, 2],
                          M[:, :, :, 1]-M[:, :, :, 2]-M[:, :, :, 3]], axis=3)
        assert not np.any(twice & 1) and np.array_equal(twice//2, p)
        ps.append(p.astype(np.int32))
        Ms.append(M.astype(np.int32))
    out = gold_from_p(g, np.stack(ps).astype(np.int64), f)
    out['winograd_M_int'] = np.stack(Ms)
    stats = dict(passed=True, records=len(ps), raw_values=int(out['p_int'].size),
                 factor_vs_expanded_differences=0, factor_vs_winograd_differences=0)
    stats.update(local_errors(out, g, f, parent, 1))
    return out, stats


def main():
    parent = read_npz(S / 'factors.npz')
    q = parent['q2'].astype(np.int64)
    q1 = parent['q1'].astype(np.int64)
    U = transform(q)
    assert np.array_equal(U, parent['winograd_q2'])
    coeff = U[:, :, [0, 1, 3]]
    zero = np.zeros_like(coeff[:, :, 0])
    horizontal = np.stack([
        np.stack([coeff[:, :, 0], coeff[:, :, 1], coeff[:, :, 1]-coeff[:, :, 0], zero], axis=2),
        np.stack([zero, coeff[:, :, 1]-coeff[:, :, 2], coeff[:, :, 1], coeff[:, :, 2]], axis=2)], axis=0)
    W = np.einsum('forx,rcy->focyx', horizontal, q1)
    za = np.maximum(-parent['z_lower'], parent['z_upper']).astype(np.int64)
    mp = (abs(coeff)*(2*za)[None, :, None]).sum(1)
    rec = np.maximum(mp[:, 0]+mp[:, 1], mp[:, 1]+mp[:, 2])
    lo, hi = np.minimum(W, 0).sum((2, 3, 4)), np.maximum(W, 0).sum((2, 3, 4))
    scale = parent['output_scale']/2
    a_float = np.rint(scale*parent['BN_gain']*(1 << 40))
    assert np.all(a_float >= -(1 << 31)) and np.all(a_float < (1 << 31))
    a = a_float.astype(np.int32)
    wide = np.maximum(-lo, hi).max(0)*abs(a.astype(np.int64)) + \
        ((1 << 31)+abs(parent['b_q20'].astype(np.int64)))*(1 << 20)
    assert np.max(abs(coeff)) < 2**12
    assert int((2*za).max()) < 2**18
    assert max(int(mp.max()), int(rec.max()), int(abs(W).max()),
               int(abs(lo).max()), int(abs(hi).max())) < 2**31
    assert int(wide.max()) < 2**63
    keep = ['q1', 'b_q20', 'theta', 'bias', 'BN_gain', 'BN_offset', 'first_scale',
            'q1_bits', 'z_bits', 'p_bits', 'wide_bits', 'z_lower', 'z_upper']
    f = {k: parent[k].copy() for k in keep}
    f.update(function_type=np.array('phase3'), physical_coeff3=coeff.astype(np.int16),
             physical_component_indices=np.array([0, 1, 3], np.int8),
             physical_coeff_bits=np.array(13), output_scale=scale, a_q40=a,
             phase_horizontal_coeff_int32=horizontal.astype(np.int32),
             expanded_phase_int32=W.astype(np.int32), p_phase_lower=lo, p_phase_upper=hi,
             p_lower=lo.min(0), p_upper=hi.max(0),
             p_any_prefix_abs=rec, M_any_prefix_abs=mp, wide_abs_bound=wide,
             raw_p_is_unhalved=np.array(True), fixed_zero_component=np.array(2))
    assert 'q2' not in f and 'expanded_int32' not in f
    np.savez_compressed(H / 'factors.npz', **f)
    fields = ['q1', 'physical_coeff3', 'physical_component_indices', 'a_q40', 'b_q20',
              'output_scale', 'theta', 'bias', 'BN_gain', 'BN_offset', 'first_scale']
    write_json(H / 'frozen_parameters.json', dict(schema='spatial_r16_phase3_u2zero_i24_v1',
        function_type='phase3', q2_present=False, raw_p_contract='unhalved p2',
        arrays={k: dict(shape=list(f[k].shape), dtype=str(f[k].dtype), values=f[k].tolist()) for k in fields}))
    bounds = dict(physical_coeff3=[int(coeff.min()), int(coeff.max())],
                  phase_horizontal=[int(horizontal.min()), int(horizontal.max())],
                  Z=[int(parent['z_lower'].min()), int(parent['z_upper'].max())],
                  D_abs=int((2*za).max()), M_any_prefix_abs=int(mp.max()),
                  p2_reconstruction_and_rank_prefix_abs=int(rec.max()),
                  expanded_phase=[int(W.min()), int(W.max())],
                  p2_final=[int(lo.min()), int(hi.max())], wide_abs=int(wide.max()))
    write_json(H / 'static_bounds.json', dict(passed=True, bounds=bounds,
        proof='|D[r,k]|<=2*max(-Zlo[r],Zhi[r]); any rank prefix |M[o,k]|<=sum_r |U[o,r,k]|*Dabs[r]. '
              'Every stripe reconstruction and cross-stripe p2 prefix is bounded by max(M0abs+M1abs,M1abs+M3abs). '
              'Independent phase-expanded binary coefficients give final min/max; J spans signed32 and b is fixed.',
        arithmetic={'source': 'binary', 'Z': 'signed15 admitted fixed q1', 'D_operand': 'signed19',
                    'coeff': 'signed13', 'M_and_p2': 'signed32', 'wide': 'signed64'},
        a_recomputed_from_scale=True, parent_a_odd=int(np.count_nonzero(parent['a_q40'] & 1)),
        a_differences_from_integer_floor_half=int(np.count_nonzero(a != parent['a_q40']//2))))
    print('PARAMETERS_READY unconstrained/factors.npz function_type=phase3 no q2', flush=True)

    data = {'tiles135': read_npz(S / 'gold_tiles.npz'),
            'sequences36': read_npz(N / 'quality/q11/sequence_tiles.npz')}
    meta = json.loads((N / 'quality/q11/sequence_tiles.json').read_text())
    seq = data['sequences36']
    seq.update(tile_ids=np.array([x['tile_id'] for x in meta], np.int32),
               frame_indices=np.array([x['frame_index'] for x in meta], np.int32),
               record_indices=np.arange(len(meta), dtype=np.int32))
    write_json(H / 'sequence_tiles.json', meta)
    summary = dict(passed=True, function_type='phase3', q2_present=False, groups=192,
                   removed_component=2, removed_N8_vectors=192,
                   remaining_N8_vectors=int(np.count_nonzero(np.any(coeff.reshape(12, 8, 16, 3), axis=1))),
                   raw_contract='p_int is unhalved p2; physical M order [0,1,3]',
                   bounds=bounds, datasets={}, comparison={}, GPU='not run', RTL='not run',
                   training=False, activity_used_for_selection=False, quality_AEE='root evaluates separately')
    manifest = dict(function_type='phase3', schema='spatial_r16_phase3_u2zero_i24_v1', q2_present=False,
        physical_coeff3='[96,16,3] signed13; coefficient index order original U0,U1,U3, not native three taps',
        raw_p_contract='gold p_int stores unhalved raw p2, even=M0+M1, odd=M1-M3; no intermediate RNE or /2',
        scale_contract='output_scale=parent/2; a_q40=RNE(output_scale*BN_gain*2^40), b and J unchanged; one final RNE26/I24',
        expanded_phase_int32='[2,96,96,3,4], axes horizontal_phase,output,input,ky,kx; same 4-column source halo for both phases',
        phase_origin='Even-aligned output tiles; phase0/1 means x=0/1 in the 2x2 tile. Not a shared translation-invariant 1x3 convolution.',
        datasets={})
    for name, g in data.items():
        assert not np.any(g['output_origin_yx'] & 1), 'phase3 tile anchors must be even'
        original = gold_from_p(g, g['p_int'].astype(np.int64), parent)
        assert np.array_equal(original['wide_int64'], g['wide_int64'])
        assert np.array_equal(original['i24'], g['i24'])
        out, stats, per = emit_phase(g, f, parent)
        filename = 'gold_tiles.npz' if name == 'tiles135' else 'gold_sequences.npz'
        np.savez_compressed(H / filename, **out)
        write_json(H / (name + '_stats.json'), stats)
        (H / (name + '_stats.jsonl')).write_text(''.join(json.dumps(x) + '\n' for x in per))
        manifest['datasets'][name] = dict(file=filename, shapes={k: list(v.shape) for k, v in out.items()},
            dtypes={k: str(v.dtype) for k, v in out.items()},
            input_fields=['source_words', 'output_origin_yx', 'identity_fp32_bits'],
            oracle_only_fields=['z_halo_int', 'p_int', 'physical_M_int', 'J_q20', 'wide_int64', 'i24'])
        summary['datasets'][name] = stats
        summary['comparison'][name] = {'unconstrained': stats}
        print(json.dumps(dict(dataset=name, phase3_pass=True, i24=stats['i24_error_vs_q11_parent'])), flush=True)
    # Add 36-record controls; frozen models and their 135-record arrays remain untouched.
    for name in ['moment', 'native_tap']:
        ff = read_npz(P / name / 'factors.npz')
        out, stats = emit_native_sequence(seq, ff, parent)
        np.savez_compressed(P / name / 'gold_sequences.npz', **out)
        write_json(P / name / 'sequence_stats.json', stats)
        write_json(P / name / 'sequence_tiles.json', meta)
        summary['comparison']['sequences36'][name] = stats
        old = read_npz(P / name / 'gold_tiles.npz')
        oldstats = json.loads((P / name / 'stats.json').read_text())
        oldstats.update(local_errors(old, data['tiles135'], ff, parent, 1))
        summary['comparison']['tiles135'][name] = oldstats
        print(json.dumps(dict(dataset='sequences36', arm=name, pass_gold=True,
                              i24=stats['i24_error_vs_q11_parent'])), flush=True)
    write_json(H / 'manifest.json', manifest)
    write_json(H / 'SUMMARY.json', summary)


if __name__ == '__main__':
    main()
