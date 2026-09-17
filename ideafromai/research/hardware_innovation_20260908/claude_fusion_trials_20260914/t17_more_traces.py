#!/usr/bin/env python3
"""T17：跨序列 trace 扩充验证——拍比 17.x% 与解析模型在 capture 全部 40 样本上的稳定性。

背景：bn_state 只用了 capture 的 s0/s10 两条序列；m1707 capture（fc_frames.bin）
实际含 40 个样本。本脚本用自有解析代码直接读 capture 的 fc1 源帧（layer 8 =
stage0 block0 fc1、layer 28 = stage3 block0 fc1），按 t5_c1_cert_core_model 的
同一口径（终态 tau、A_q f12、thr 48bit 合同、组级 BFP 证书递推）重算每条序列：
- 实测拍比（精确递推）；
- T16 解析保守界拍比（j_first ≥ floor(log2(ρ+1))，ρ=|δ|/L1）；
- sid=0/10 与 T16 已有数字交叉校验（验证本解析器）。

capture 路径只读；不触碰生产树任何文件。自有代码。
"""
import json
import struct
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BN = ROOT.parents[0] / 'bn_state'
CAP = Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07'
           '/results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture'
           '_s40_r1_20260901')
JOUT = ROOT / 'results' / 't17_result.json'
T, G = 10, 20000
HEADER = struct.Struct('<8sHH11I')
LID = {0: 8, 3: 28}          # stage → fc1(swin_blocks.0) layer_id


def parse_sources(sids):
    """capture fc_frames.bin → {sid: {stage: S (N,C) uint8 bits}}（自有解析）。"""
    want = {(sid, lid) for sid in sids for lid in LID.values()}
    chunks = {k: [] for k in want}
    with (CAP / 'fc_frames.bin').open('rb') as f:
        while raw := f.read(HEADER.size):
            magic, ver, hs, lid, sid, fi, start, n, C, br, nnz, rb, cb, _ = \
                HEADER.unpack(raw)
            if (sid, lid) not in chunks:
                f.seek(cb, 1)
                continue
            payload = zlib.decompress(f.read(cb))
            bits = np.unpackbits(
                np.frombuffer(payload[:n * br], np.uint8).reshape(n, br),
                axis=1, bitorder='little')[:, :C]
            if start != sum(len(x) for x in chunks[sid, lid]):
                raise ValueError('noncontiguous frame')
            chunks[sid, lid].append(bits)
    out = {}
    for sid in sids:
        for st, lid in LID.items():
            if chunks[sid, lid]:
                out[sid, st] = np.concatenate(chunks[sid, lid])
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


def one_trace(S, st, sid):
    z = np.load(BN / f'trace_s0_stage{st}.npz')       # 阶段参数（sid 无关）
    W = z['W'].astype(np.float64)
    A, gamma, beta = z['A'], z['gamma'], z['beta']
    bias, center = z['bias'], z['center']
    N, C = S.shape
    P = N // T
    H = W.shape[0]
    theta = 1.0
    R = A.sum(1).reshape(T, 1)
    direction = np.sign(gamma)
    Sf = S.astype(np.float64)

    tau = np.zeros((T, H))
    for lo in range(0, H, 32):
        hi = min(H, lo + 32)
        Y = (Sf @ W[lo:hi].T).reshape(T, P, hi - lo)
        mu = Y.mean((0, 1))
        var = ((Y - mu) ** 2).mean((0, 1))
        tau[:, lo:hi] = (mu * R + np.sqrt(var + 1e-5) / gamma[lo:hi]
                         * (theta + center - bias - beta[lo:hi] * R)) * direction[None, lo:hi]

    A_q = np.rint(A * 4096).astype(np.int64)
    A_q = np.where(A_q >= (1 << 15), A_q - (1 << 16), A_q)
    P_t = A_q.clip(min=0).sum(1)
    N_t = A_q.clip(max=0).sum(1)
    L1 = np.abs(A_q).sum(1)
    tau_q = np.rint(tau * (1 << 14)).astype(np.int64)
    thr = np.where(direction[None, :] < 0,
                   -(tau_q << 12) + 1, tau_q << 12)
    thr = np.where(thr >= (1 << 47), thr - (1 << 48), thr)
    thr = np.where(thr < -(1 << 47), thr + (1 << 48), thr)
    Dflag = (direction > 0).astype(np.int64)

    rng = np.random.default_rng(20260914 + sid)
    ps = rng.integers(0, P, G)
    hs = rng.integers(0, H, G)
    Ysub = np.einsum('tgc,gc->tg',
                     Sf.reshape(T, P, C)[:, ps, :], W[hs])
    Yq = np.clip(np.rint(Ysub * (1 << 14)), -(1 << 23), (1 << 23) - 1).astype(np.int64)
    Yq = Yq.T                                          # (G,T)
    Vfull = Yq @ A_q.T
    thr_g = thr[:, hs].T
    delta = Vfull - thr_g
    msb_g = np.array([int(np.abs(Yq[g]).max()).bit_length() for g in range(G)])

    jf = j_first_exact(Yq, thr_g, A_q, P_t, N_t)
    j_star = jf.min(1)
    dec_raw = np.zeros((G, T), bool)
    # 精确判决（与递推一致）：锁定时 Vmin ≥ thr
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
    slack = jf.astype(np.int64) - jf_pred

    cyc = lambda js_: 1 + np.maximum(np.maximum(msb_g - js_, 0), 1)
    cyc_meas = cyc(j_star)
    planes = np.maximum(msb_g - j_star, 0)
    cyc_bound = cyc(jf_pred.min(1))
    pool = jf_pred.ravel()
    cyc_iid = cyc(rng.choice(pool, size=(G, T)).min(1))

    return {
        'sid': sid, 'stage': st, 'N': N, 'P': P, 'H': H, 'C': C,
        'cert_vs_full_mismatches': mism,
        'ratio_measured': float(cyc_meas.mean() / 24),
        'ratio_analytical_bound': float(cyc_bound.mean() / 24),
        'ratio_iid_marginal': float(cyc_iid.mean() / 24),
        'j_star_mean': float(j_star.mean()),
        'msb_mean': float(msb_g.mean()),
        'slack_mean': float(slack.mean()),
        'rho_median': float(np.median(np.abs(delta) / L1[None, :])),
        'planes_hist': {str(int(k)): int((planes == k).sum()) for k in np.unique(planes)},
    }


def main():
    sids = list(range(40))
    src = parse_sources(sids)
    print('parsed', sorted({k[0] for k in src}), 'sids')
    res = []
    for (sid, st), S in sorted(src.items()):
        r = one_trace(S, st, sid)
        res.append(r)
        print('sid=%2d stage%d P=%d H=%d  meas=%.1f%%  bound=%.1f%%  iid=%.1f%%  mism=%d'
              % (sid, st, r['P'], r['H'], r['ratio_measured'] * 100,
                 r['ratio_analytical_bound'] * 100, r['ratio_iid_marginal'] * 100,
                 r['cert_vs_full_mismatches']), flush=True)
    JOUT.write_text(json.dumps(res, indent=1))
    meas = [r['ratio_measured'] for r in res]
    bnd = [r['ratio_analytical_bound'] for r in res]
    print('\n%d traces: measured %.1f–%.1f%% (mean %.1f%%), bound %.1f–%.1f%%'
          % (len(res), min(meas) * 100, max(meas) * 100, 100 * np.mean(meas),
             min(bnd) * 100, max(bnd) * 100))
    print('saved', JOUT)


if __name__ == '__main__':
    main()
