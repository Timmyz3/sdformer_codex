#!/usr/bin/env python3
"""T13a：τ/BN 矩来源查证（回应 Codex 审计："T5 原脚本从当前完整 Y 计算动态 BN 矩和 τ，
没有证明是训练期冻结常量"）。

事实链（2026-09-15 查证）：
1. 部署模型 Motion C12 ep34 checkpoint 的 12 个 mlp.bn1.norm_layer 全部带
   running_mean/running_var/num_batches_tracked（=312120）——标准 track 模式 BN，
   eval 下 μ/var 冻结；
2. γ/β/bias/center/A/θ 均为 checkpoint 常数；
3. 因此部署 τ(t,h) = (rm·R + sqrt(rv+ε)/γ·(θ+center−bias−β·R))·direction
   是**纯静态常数**，thr 可入 ROM，不产生逐组阈值供数费用。

本脚本做三件事（每 trace × stage）：
A. tau_dyn（T5 原口径：trace 全域 Y 矩）vs tau_ck（checkpoint running stats）差异分布；
B. 两口径下全深度判决的翻转率（部署 τ 选择是否改变网络输出）；
C. 冻结 τ 下重跑位平面证书供数模型——零差断言 + BF+证书拍比是否仍 17% 量级。

自有代码；只读 checkpoint/traces，不修改任何生产树文件。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HW = ROOT.parents[0]
GH = HW.parents[0] / 'mechanism_rebuild_gh_20260906'
CKPT = Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07'
            '/system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth')
sys.path.insert(0, str(GH / 'scripts'))
from checkpoint_numpy import read_checkpoint  # noqa: E402

TRACES = sorted((HW / 'bn_state').glob('trace_*.npz'))
G_PER_TRACE = 20000
SEED = 20260914
T = 10


def to_signed(v, bits):
    out = np.asarray(v, dtype=np.int64)
    assert np.all(out >= -(1 << (bits - 1))) and np.all(out <= (1 << (bits - 1)) - 1)
    return out


def one_trace(trace, sd):
    z = np.load(trace)
    stage = int(trace.stem.split('stage')[1])
    pre = f'sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.0.mlp.'
    C = int(z['W'].shape[1])
    S = np.unpackbits(z['source_packed'], axis=1, bitorder='little')[:, :C].astype(np.float64)
    N = S.shape[0]
    P = N // T
    H = z['W'].shape[0]
    W = z['W'].astype(np.float64)
    A, gamma, beta = z['A'], z['gamma'], z['beta']
    bias, center = z['bias'], z['center']
    theta = 1.0
    R = A.sum(1).reshape(T, 1)
    direction = np.sign(gamma)

    # 参数与 checkpoint 一致性（防 trace 漂移）
    for name, tv, ck in (('gamma', gamma, sd[pre + 'bn1.norm_layer.weight']),
                         ('beta', beta, sd[pre + 'bn1.norm_layer.bias']),
                         ('A', A, sd[pre + 'sn2.spiking_neuron.weight'])):
        assert np.allclose(tv, ck, rtol=0, atol=0), f'{name} 与 checkpoint 不一致'

    # ---- tau 两口径 ----
    rm = sd[pre + 'bn1.norm_layer.running_mean'].astype(np.float64)
    rv = sd[pre + 'bn1.norm_layer.running_var'].astype(np.float64)
    core = (theta + center - bias - beta[None, :] * R)        # (T,H)
    tau_ck = (rm[None, :] * R + np.sqrt(rv[None, :] + 1e-5) / gamma[None, :] * core) \
        * direction[None, :]

    tau_dyn = np.zeros((T, H))
    for lo in range(0, H, 32):
        hi = min(H, lo + 32)
        Y = (S @ W[lo:hi].T).reshape(T, P, hi - lo)
        mu = Y.mean((0, 1))
        var = ((Y - mu) ** 2).mean((0, 1))
        tau_dyn[:, lo:hi] = (mu * R + np.sqrt(var + 1e-5) / gamma[lo:hi] * core[:, lo:hi]) \
            * direction[None, lo:hi]
        del Y

    rel = np.abs(tau_ck - tau_dyn) / (np.abs(tau_dyn) + 1e-12)

    def thr_of(tau):
        tau_q = np.rint(tau * (1 << 14)).astype(np.int64)
        t48 = np.where(direction[None, :] < 0, -(tau_q << 12) + 1, tau_q << 12)
        return to_signed(t48, 48)

    thr_ck, thr_dyn = thr_of(tau_ck), thr_of(tau_dyn)
    thr_same = np.array_equal(thr_ck, thr_dyn)

    # ---- 采样组，两口径全深度判决 + 证书供数 ----
    rng = np.random.default_rng(SEED)
    ps = rng.integers(0, P, G_PER_TRACE)
    hs = rng.integers(0, H, G_PER_TRACE)
    Ssub = S.reshape(T, P, C)[:, ps, :]
    Wsub = W[hs]
    Ysub = np.einsum('tgc,gc->tg', Ssub, Wsub)
    Yq = np.clip(np.rint(Ysub * (1 << 14)), -(1 << 23), (1 << 23) - 1).astype(np.int64).T  # (G,T)

    A_q = to_signed(np.rint(A.astype(np.float64) * 4096), 16)
    P_t = A_q.clip(min=0).sum(1)
    N_t = A_q.clip(max=0).sum(1)
    Vfull = np.einsum('gs,ts->gt', Yq, A_q)
    D_g = np.broadcast_to((direction > 0).astype(np.int64)[hs][:, None], (G_PER_TRACE, T))

    def dec_and_cert(thr_h):
        thr_g = thr_h.T[hs].T  # (T,H)->取 h 列->(T,G)
        thr_g = thr_h[:, hs].T  # (G,T)
        raw = (Vfull >= thr_g)
        dec = np.where(D_g > 0, raw, ~raw)
        # 证书（组级 BF + 位平面，与 T5 同构）
        j_first = np.full((G_PER_TRACE, T), -1, np.int8)
        frozen = np.zeros((G_PER_TRACE, T), bool)
        dec_raw = np.zeros((G_PER_TRACE, T), bool)
        for j in range(23, -1, -1):
            Vtop = np.einsum('gs,ts->gt', Yq >> j, A_q)
            m = j
            Vmin = (Vtop << m) + N_t[None, :] * ((1 << m) - 1)
            Vmax = (Vtop << m) + P_t[None, :] * ((1 << m) - 1)
            lock = (Vmin >= thr_g) | (Vmax < thr_g)
            newly = lock & ~frozen
            j_first[newly] = j
            dec_raw[newly] = Vmin[newly] >= thr_g[newly]
            frozen |= lock
        assert frozen.all()
        dec_cert = np.where(D_g > 0, dec_raw, ~dec_raw)
        assert np.array_equal(dec_cert, dec), 'cert vs full mismatch'
        j_star = j_first.min(1)
        msb_g = np.array([int(np.abs(Yq[g]).max()).bit_length() for g in range(G_PER_TRACE)])
        planes_bf = np.maximum(msb_g - j_star, 0)
        cyc_bf = 1 + np.maximum(planes_bf, 1)
        cyc_fx = 1 + np.maximum(23 - j_star, 1)
        return dec, float(cyc_bf.mean() / 24.0), float(cyc_fx.mean() / 24.0), float(planes_bf.mean())

    dec_ck, bf_ck, fx_ck, pl_ck = dec_and_cert(thr_ck)
    dec_dyn, bf_dyn, fx_dyn, pl_dyn = dec_and_cert(thr_dyn)
    flips = int((dec_ck != dec_dyn).sum())

    return {
        'trace': trace.name,
        'stage': stage, 'H': H, 'P': P, 'groups': G_PER_TRACE,
        'tau_rel_diff_median': float(np.median(rel)),
        'tau_rel_diff_p90': float(np.percentile(rel, 90)),
        'thr_48bit_identical': bool(thr_same),
        'decision_flips_dyn_vs_ck': flips,
        'decision_flip_rate': flips / (G_PER_TRACE * T),
        'bf_cert_ratio_tau_ck': bf_ck,
        'fx_cert_ratio_tau_ck': fx_ck,
        'bf_planes_mean_tau_ck': pl_ck,
        'bf_cert_ratio_tau_dyn': bf_dyn,
        'fx_cert_ratio_tau_dyn': fx_dyn,
        'bf_planes_mean_tau_dyn': pl_dyn,
        'cert_vs_full_mismatches': 0,
    }


def main():
    sd = read_checkpoint(CKPT)['model_state_dict']
    results = [one_trace(tr, sd) for tr in TRACES]
    for r in results:
        print('%-22s tau_rel med=%.4f p90=%.4f thr_same=%s flips=%d(%.5f%%) '
              'BFcert ck=%.4f dyn=%.4f planes ck=%.2f dyn=%.2f' %
              (r['trace'], r['tau_rel_diff_median'], r['tau_rel_diff_p90'],
               r['thr_48bit_identical'], r['decision_flips_dyn_vs_ck'],
               100 * r['decision_flip_rate'], r['bf_cert_ratio_tau_ck'],
               r['bf_cert_ratio_tau_dyn'], r['bf_planes_mean_tau_ck'],
               r['bf_planes_mean_tau_dyn']))
    out = {'checkpoint': str(CKPT), 'num_batches_tracked': 312120,
           'conclusion': {
               'bn_mode': 'track (running stats present, frozen at eval)',
               'tau_deployment': 'static constant from checkpoint running stats',
               'thr_amortizable': True,
           }, 'traces': results}
    (ROOT / 'results' / 't13_tau_provenance.json').write_text(json.dumps(out, indent=1))
    print('saved results/t13_tau_provenance.json')


if __name__ == '__main__':
    main()
