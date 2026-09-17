#!/usr/bin/env python3
"""T19：全部 12 个 fc1 门路层的证书拍比广度（capture 40 序列）。

T17 只覆盖 stage0/3 的 block0（layer 8/28）；capture 实有 12 个 fc1 层
（stage0–3 全部 swin block）。本脚本用自有解析器读 fc_frames.bin 全部 fc1 源帧，
参数来自 sd5ai 抽取的 checkpoint npz（t19_gate_params.npz，theta_src 如实相乘
——layer30 为 0.99999875，其余 =1.0），跑与 t5/t17 同口径的证书递推+解析界。

每 (layer, sid) 一条 trace，G=20000 组；产出 results/t19_result.json。
capture 只读。自有代码。
"""
import json
import struct
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CAP = Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07'
           '/results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture'
           '_s40_r1_20260901')
PARAMS = ROOT / 't19_gate_params.npz'
JOUT = ROOT / 'results' / 't19_result.json'
T, G = 10, 20000
HEADER = struct.Struct('<8sHH11I')
LIDS = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]


def parse_sources(sids):
    """一遍扫描 fc_frames.bin → {(sid,lid): (payload bytes list, C)}（自有解析）。
    存解压后的 packed 行（n×br），按需 unpack 控内存。"""
    want = {(sid, lid) for sid in sids for lid in LIDS}
    chunks = {k: [] for k in want}
    meta = {}
    with (CAP / 'fc_frames.bin').open('rb') as f:
        while raw := f.read(HEADER.size):
            magic, ver, hs, lid, sid, fi, start, n, C, br, nnz, rb, cb, _ = \
                HEADER.unpack(raw)
            if (sid, lid) not in chunks:
                f.seek(cb, 1)
                continue
            payload = zlib.decompress(f.read(cb))
            if (sid, lid) in meta and meta[sid, lid][0] != C:
                raise ValueError('C changed')
            meta[sid, lid] = (C, br)
            chunks[sid, lid].append(payload[:n * br])
    out = {}
    for k, lst in chunks.items():
        if lst:
            C, br = meta[k]
            out[k] = (np.concatenate([np.frombuffer(p, np.uint8) for p in lst]), C, br)
    return out


def j_first_exact(Yq, thr_g, A_q, P_t, N_t):
    Gn = Yq.shape[0]
    j_first = np.full((Gn, T), -1, np.int8)
    frozen = np.zeros((Gn, T), bool)
    for j in range(23, -1, -1):
        Vtop = (Yq >> j) @ A_q.T
        Vmin = (Vtop << j) + N_t * ((1 << j) - 1)
        Vmax = (Vtop << j) + P_t * ((1 << j) - 1)
        lock = (Vmin >= thr_g) | (Vmax < thr_g)
        newly = lock & ~frozen
        j_first[newly] = j
        frozen |= lock
    assert frozen.all()
    return j_first


def j_first_analytical(delta, L1):
    js = np.arange(24)
    bound = L1[None, :, None] * ((np.int64(1) << js) - 1)
    d = delta[:, :, None]
    mask = ((d > 0) & (d >= bound)) | ((d < 0) & (-d > bound)) \
        | ((d == 0) & (js == 0))
    return (mask.sum(2) - 1).astype(np.int8)


def one_trace(packed, C, br, prm, sid):
    S = np.unpackbits(packed.reshape(-1, br), axis=1, bitorder='little')[:, :C]
    N = S.shape[0]
    P = N // T
    W = prm['W'].astype(np.float64)
    A, gamma, beta = prm['A'], prm['gamma'], prm['beta']
    bias, center = prm['bias'], prm['center']
    theta_src = float(np.asarray(prm['theta_src']).ravel()[0])
    H = W.shape[0]
    R = A.sum(1).reshape(T, 1)
    direction = np.sign(gamma)
    Sf = S.astype(np.float64)

    tau = np.zeros((T, H))
    for lo in range(0, H, 32):
        hi = min(H, lo + 32)
        Y = (Sf @ W[lo:hi].T).reshape(T, P, hi - lo) * theta_src
        mu = Y.mean((0, 1))
        var = ((Y - mu) ** 2).mean((0, 1))
        tau[:, lo:hi] = (mu * R + np.sqrt(var + 1e-5) / gamma[lo:hi]
                         * (1.0 + center - bias - beta[lo:hi] * R)) * direction[None, lo:hi]

    A_q = np.rint(A * 4096).astype(np.int64)
    A_q = np.where(A_q >= (1 << 15), A_q - (1 << 16), A_q)
    P_t = A_q.clip(min=0).sum(1)
    N_t = A_q.clip(max=0).sum(1)
    L1 = np.abs(A_q).sum(1)
    tau_q = np.rint(tau * (1 << 14)).astype(np.int64)
    thr = np.where(direction[None, :] < 0, -(tau_q << 12) + 1, tau_q << 12)
    thr = np.where(thr >= (1 << 47), thr - (1 << 48), thr)
    thr = np.where(thr < -(1 << 47), thr + (1 << 48), thr)
    Dflag = (direction > 0).astype(np.int64)

    rng = np.random.default_rng(20260914 + sid)
    ps = rng.integers(0, P, G)
    hs = rng.integers(0, H, G)
    Ysub = np.einsum('tgc,gc->tg',
                     Sf.reshape(T, P, C)[:, ps, :], W[hs]) * theta_src
    Yq = np.clip(np.rint(Ysub * (1 << 14)), -(1 << 23), (1 << 23) - 1).astype(np.int64)
    Yq = Yq.T
    Vfull = Yq @ A_q.T
    thr_g = thr[:, hs].T
    delta = Vfull - thr_g
    msb_g = np.array([int(np.abs(Yq[g]).max()).bit_length() for g in range(G)])

    jf = j_first_exact(Yq, thr_g, A_q, P_t, N_t)
    j_star = jf.min(1)
    dec_raw = np.zeros((G, T), bool)
    for j in range(24):
        sel = jf == j
        if not sel.any():
            continue
        Vtop = (Yq >> j) @ A_q.T
        Vmin = (Vtop << j) + N_t * ((1 << j) - 1)
        dec_raw[sel] = Vmin[sel] >= thr_g[sel]
    dec = np.where(Dflag[hs][:, None] > 0, dec_raw, ~dec_raw)
    dec_full = np.where(Dflag[hs][:, None] > 0, Vfull >= thr_g, ~(Vfull >= thr_g))
    mism = int((dec != dec_full).sum())

    jf_pred = j_first_analytical(delta, L1)
    assert (jf_pred <= jf).all()

    cyc = lambda js_: 1 + np.maximum(np.maximum(msb_g - js_, 0), 1)
    cyc_meas = cyc(j_star)
    cyc_bound = cyc(jf_pred.min(1))

    return {
        'lid': prm['lid'], 'stage': prm['stage'], 'block': prm['block'], 'sid': sid,
        'N': N, 'P': P, 'H': H, 'C': C,
        'cert_vs_full_mismatches': mism,
        'ratio_measured': float(cyc_meas.mean() / 24),
        'ratio_analytical_bound': float(cyc_bound.mean() / 24),
        'j_star_mean': float(j_star.mean()),
        'msb_mean': float(msb_g.mean()),
        'rho_median': float(np.median(np.abs(delta) / L1[None, :])),
    }


def main():
    z = np.load(PARAMS)
    prms = {}
    for lid in LIDS:
        mod = str(z['L%d_module' % lid])
        stage = int(mod.split('layers.')[1].split('.')[0])
        block = int(mod.split('swin_blocks.')[1].split('.')[0])
        prms[lid] = {
            'lid': lid, 'stage': stage, 'block': block,
            'W': z['L%d_W' % lid], 'A': z['L%d_A' % lid],
            'gamma': z['L%d_gamma' % lid], 'beta': z['L%d_beta' % lid],
            'bias': z['L%d_bias' % lid], 'center': z['L%d_center' % lid],
            'theta_src': z['L%d_theta_src' % lid],
        }
    sids = list(range(40))
    src = parse_sources(sids)
    print('parsed', len(src), '(sid,lid) pairs', flush=True)
    res = []
    for lid in LIDS:
        for sid in sids:
            if (sid, lid) not in src:
                continue
            packed, C, br = src[sid, lid]
            r = one_trace(packed, C, br, prms[lid], sid)
            res.append(r)
            del src[sid, lid]
            print('L%02d s%02d st%d blk%d P=%-6d H=%-5d meas=%.1f%% bound=%.1f%% mism=%d'
                  % (lid, sid, r['stage'], r['block'], r['P'], r['H'],
                     r['ratio_measured'] * 100, r['ratio_analytical_bound'] * 100,
                     r['cert_vs_full_mismatches']), flush=True)
    JOUT.write_text(json.dumps(res, indent=1))
    meas = [r['ratio_measured'] for r in res]
    print('\n%d traces: %.1f–%.1f%% (mean %.1f%%)'
          % (len(res), min(meas) * 100, max(meas) * 100, 100 * np.mean(meas)))
    print('saved', JOUT)


if __name__ == '__main__':
    main()
