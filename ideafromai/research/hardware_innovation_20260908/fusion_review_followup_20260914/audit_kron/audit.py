from pathlib import Path
import json
import shutil
import subprocess
import numpy as np

H = Path(__file__).resolve().parent
SRC = H.parent / 'kron_escape'
BASE = H.parents[1]

def raw_z(words, q1, origin):
    z = np.zeros((40, 8), np.int64)
    for pos in range(4):
        for y in range(3):
            for x in range(3):
                iy, ix = pos//2+y, pos%2+x
                if 0 <= origin[0]+iy < 240 and 0 <= origin[1]+ix < 320:
                    bits = ((words[:, iy, ix, None] >> np.arange(10)) & 1)
                    z[pos*10:pos*10+10] += bits.T @ q1[:, :, y, x].T
    return z

def inspect_fixture(p, origin):
    words = np.fromfile(p/'source.bin', '<u2').astype(np.int64).reshape(96, 4, 4)
    q1 = np.fromfile(p/'param4.bin', '<i4').astype(np.int64).reshape(864, 8).T.reshape(8, 96, 3, 3)
    w = np.fromfile(p/'param5.bin', '<i4').astype(np.int64).reshape(12, 8, 8).transpose(0, 2, 1).reshape(96, 8)
    aa = np.fromfile(p/'param8.bin', '<i4').astype(np.int64).reshape(12, 8)[:, :4]
    bb = np.fromfile(p/'param9.bin', '<i4').astype(np.int64).reshape(2, 8).T
    mask = np.fromfile(p/'param10.bin', '<i4')[0]
    escape = ((int(mask) >> np.arange(12)) & 1).astype(bool)
    wfull = np.kron(aa, bb)
    assert np.array_equal(w.reshape(12, 8, 8)[~escape], wfull.reshape(12, 8, 8)[~escape])
    z = raw_z(words, q1, origin)
    s = z.reshape(40, 4, 2) @ bb.T
    raw = np.concatenate([z @ w[8*o:8*o+8].T for o in range(12)])
    factor = np.concatenate([sum(aa[o, k] * s[:, k] for k in range(4)) for o in range(12)])
    for o in np.where(escape)[0]:
        factor[40*o:40*o+40] = z @ w[8*o:8*o+8].T
    assert np.array_equal(raw, factor)
    assert np.array_equal(raw.ravel(), np.fromfile(p/'raw.bin', '<i4'))
    assert np.all((s >= -(1<<18)) & (s < (1<<18)))
    fp = np.fromfile(p/'identity_fp32.bin', '<f4').astype(float).reshape(480, 8)
    j = np.clip(np.rint(fp*(1<<20)), -(1<<31), (1<<31)-1).astype(np.int64)
    assert np.array_equal(j.ravel(), np.fromfile(p/'identity.bin', '<i4'))
    ab = np.fromfile(p/'param7.bin', '<i4').astype(np.int64).reshape(12, 2, 8)
    al = np.repeat(ab[:, 0], 40, axis=0)
    bias = np.repeat(ab[:, 1], 40, axis=0)
    final = []
    for pv, av, bv, jv in zip(raw.flat, al.flat, bias.flat, j.flat):
        q, r = divmod(int(pv)*int(av)+(int(bv)+int(jv))*(1<<20), 1<<26)
        q += r > (1<<25) or (r == (1<<25) and q % 2 != 0)
        final.append(max(-(1<<23), min((1<<23)-1, q)))
    assert np.array_equal(final, np.fromfile(p/'gold.bin', '<i4'))
    # Independent native event obligations, without reading the old schedule checker.
    masks = []
    for k in range(864):
        channel, tap = divmod(k, 9)
        values = []
        for pos in range(4):
            y, x = pos//2+tap//3, pos%2+tap%3
            values.append(int(words[channel, y, x]) if 0 <= origin[0]+y < 240 and 0 <= origin[1]+x < 320 else 0)
        masks.append(values)
    livek = np.any(q1.reshape(8, 864) != 0, axis=0)
    assert np.array_equal(livek, np.fromfile(p/'param6.bin', '<i4').reshape(864, 8)[:, 0])
    K = int(livek.sum()); active = U = dual = 0
    for k, values in enumerate(masks):
        if livek[k]:
            active += bool(any(values))
            for a, b in [(values[0], values[1]), (values[2], values[3])]:
                U += (a | b).bit_count(); dual += (a & b).bit_count()
    vlive = np.any(w.reshape(12, 8, 8) != 0, axis=1)
    live_z = z != 0
    live_s = np.any(s != 0, axis=2)
    N = int((~escape).sum())
    direct_mac = int((live_z * vlive.sum(0)).sum())
    escape_mac = int((live_z * vlive[escape].sum(0)).sum())
    smac = int(live_z.sum())
    kmac = sum(int((live_s & (row != 0)).sum()) for row in aa[~escape])
    coeff_words = int((vlive & np.any(live_z, axis=0)).sum())
    escape_words = int((vlive[escape] & np.any(live_z, axis=0)).sum())
    base = 5437 + K + 2*active + 3*U
    common = dict(core_local_source_reads=K, core_first_issues=U, core_dual_updates=dual,
                  core_z_writes=20+U, core_z_vector_reads=40+U, core_psum_reads=480, core_psum_writes=480)
    pred = {
        0: dict(common, core_mac_issues=direct_mac, core_z_scalar_reads=direct_mac,
                core_second_weight_words=coeff_words, core_weight_words=active+coeff_words,
                core_s_build_cycles=0, core_s_mac_issues=0, core_s_writes=0, core_s_reads=0,
                core_a_words=0, core_factor_mac_issues=0, core_escape_mac_issues=0,
                base_cycles=base+direct_mac),
        1: dict(common, core_mac_issues=smac+kmac+escape_mac, core_z_scalar_reads=smac+escape_mac,
                core_second_weight_words=escape_words+N, core_weight_words=active+escape_words+N,
                core_s_build_cycles=320+smac, core_s_mac_issues=smac, core_s_writes=160,
                core_s_reads=kmac, core_a_words=N, core_factor_mac_issues=kmac, core_escape_mac_issues=escape_mac,
                base_cycles=base+320+smac+kmac+escape_mac-7*N)}
    return pred, dict(fixture=p.name, S_abs_max=int(np.abs(s).max()), W_abs_max=int(np.abs(w).max()), raw_abs_max=int(np.abs(raw).max()), escape_groups=np.where(escape)[0].tolist())

def math_audit():
    fit = np.load(SRC/'fitted.npz')
    assert np.array_equal(np.kron(fit['A'].astype(np.int64), fit['B'].astype(np.int64)), fit['W_kron'])
    assert np.array_equal(fit['W_hybrid'].reshape(12, 8, 8)[fit['escape']], fit['q2'].reshape(12, 8, 8)[fit['escape']])
    cal = BASE/'fusion_ten_trials_20260914/algorithm_sparse'
    captured = np.load(cal/'calibration.npz')['source_bits'].astype(np.int64)
    q1 = fit['q1'].astype(np.int64).reshape(8, 96, 3, 3)
    z = np.zeros((336, 4, 10, 8), np.int64)
    for pos in range(4):
        for y in range(3):
            for x in range(3):
                z[:, pos] += captured[:, :, :, pos//2+y, pos%2+x] @ q1[:, :, y, x].T
    assert np.array_equal(z, np.load(cal/'calibration_latent.npz')['z'])
    values = z.reshape(-1, 8)
    co = np.load(cal/'consumer_coefficients.npz')
    error = (values @ (fit['W_kron'].astype(np.int64)-fit['q2'].astype(np.int64)).T).astype(float)
    weights = (co['a_q40'].astype(float) / np.mean(co['a_q40']))**2
    group_error = np.mean(error**2, axis=0).reshape(12, 8) * weights.reshape(12, 8)
    ranked = np.argsort(-group_error.sum(1), kind='stable')
    assert sorted(ranked[:2]) == list(np.where(fit['escape'])[0]) == [6, 7]
    fit_report = json.loads((SRC/'fit.json').read_text())
    assert np.allclose(group_error.sum(1), fit_report['weighted_group_errors'])
    lo = np.minimum(fit['q1'].astype(np.int64), 0).sum(1)
    hi = np.maximum(fit['q1'].astype(np.int64), 0).sum(1)
    z_abs_bound = np.maximum(-lo, hi)
    s_bound = int((z_abs_bound.reshape(4, 2) @ np.abs(fit['B'].astype(np.int64)).T).max())
    p_bound = int((np.abs(fit['W_hybrid'].astype(np.int64)) @ z_abs_bound).max())
    assert s_bound == fit_report['S_abs_bound'] == 16451
    assert p_bound == fit_report['p_abs_bound'] == 67955134
    original = np.load(BASE/'r8_consumer_fusion_20260914/data/consumer_first8.npz')
    assert np.array_equal(fit['q1'], original['q1'])
    assert np.array_equal(fit['q2'], original['q2'])
    predictions = {}; bounds = []
    for c in json.loads((SRC/'fixtures.json').read_text()):
        if c['name'].startswith('real_'):
            index = int(c['name'].split('_')[-1])
            p = SRC/'fixtures'/c['name']
            srcbits = original['source_bits'][index].astype(np.int64)
            packed = np.sum(srcbits * (1 << np.arange(10))[:, None, None, None], axis=0)
            assert np.array_equal(packed.ravel(), np.fromfile(p/'source.bin', '<u2'))
            assert np.array_equal(c['input_origin'], original['input_origin_yx'][index])
            assert np.array_equal(np.fromfile(p/'param8.bin', '<i4').reshape(12, 8)[:, :4], fit['A'])
            assert np.array_equal(np.fromfile(p/'param9.bin', '<i4').reshape(2, 8).T, fit['B'])
            assert np.array_equal(np.fromfile(p/'param5.bin', '<i4').reshape(12, 8, 8).transpose(0, 2, 1).reshape(96, 8), fit['W_hybrid'])
            identity = original['identity_fp32'][index].reshape(10, 12, 8, 2, 2).transpose(1, 3, 4, 0, 2)
            assert np.array_equal(identity.ravel(), np.fromfile(p/'identity_fp32.bin', '<f4'))
        pred, bound = inspect_fixture(SRC/'fixtures'/c['name'], c['input_origin'])
        predictions[c['name']] = pred; bounds.append(bound)
    rows = json.loads((SRC/'results.json').read_text())
    for r in rows:
        pred = predictions[r['fixture']][r['mode']]
        for key, val in pred.items():
            if key != 'base_cycles':
                assert r[key] == val, (r['fixture'], r['mode'], key, r[key], val)
        assert r['core_cycles'] == pred['base_cycles'] + r['core_source_stalls'] + r['core_weight_stalls'] + r['core_output_stalls']
        assert r['static_words'] == (0 if r['command'] else 1848+15*r['mode'])
        assert r['consumer_cycles'] == 3385+r['consumer_join_wait_cycles']+r['consumer_output_stalls']
        assert r['total_cycles'] == r['consumer_cycles']+r['static_words']+r['parameter_stalls']+r['source_load_words']+r['origin_words']+r['source_load_stalls']+3
    result = dict(original_fixtures=len(bounds), native_raw_J_I24_each=len(bounds)*3840,
                  verified_rtl_command_obligations=len(rows), training_windows_reconstructed=336,
                  real_fixture_origin_source_identity_and_parameters_match=True,
                  fixed_all_input_bounds=dict(z_lower=lo.tolist(), z_upper=hi.tolist(), S_abs=s_bound, p_abs=p_bound),
                  weighted_fit_escape_groups=[6, 7], ranks=dict(A=int(np.linalg.matrix_rank(fit['A'])), B=int(np.linalg.matrix_rank(fit['B'])), W_kron=int(np.linalg.matrix_rank(fit['W_kron']))), bounds=bounds)
    (H/'math_results.json').write_text(json.dumps(result, indent=2)+'\n')
    return result

def build(folder):
    with (folder/'build.log').open('w') as log:
        subprocess.run(['verilator', '-Wall', '--cc', '--exe', '--top-module', 'consumer_stream', '--Mdir', 'obj', 'consumer_stream.sv', 'i24_consumer.sv', 'kron_core.sv', 'tb.cpp', '-CFLAGS', '-O3'], cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(['make', '-C', 'obj', '-f', 'Vconsumer_stream.mk', '-j2'], cwd=folder, stdout=log, stderr=subprocess.STDOUT, check=True)

if __name__ == '__main__':
    print(json.dumps(math_audit(), indent=2))
