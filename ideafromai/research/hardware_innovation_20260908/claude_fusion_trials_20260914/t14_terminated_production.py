#!/usr/bin/env python3
"""T14：terminated production 静态预实验（通道维证书终止的融合生产者）。

机制（由 T13 结构结论解锁）：
- T13 证明 fc1→bn1→sn2 严格串行、门核是 Y 唯一消费者 → Y 无需物化；
- 融合形态：V[g,t] = Σ_c u[g,t,c]·W[h,c]，其中 u[g,t,c] = Σ_s A[t,s]·S[g,s,c]
  （10 位 S 列 → 2^10×10 f12 LUT 一次查出，A 静态）；
- 通道维证书：按静态序（|W| 降序 / 自然序）逐通道累加，
  未处理通道贡献界 = B_t · Σ_{未处理}|W[h,c]|（B_t=Σ_s|A[t,s]|，静态），
  锁定 = (P−R ≥ thr) 或 (P+R < thr)，全 10 单元锁定即停算；
- 省的是生产 MAC：1 − k*/C（每通道 10 MAC，基线同）；传输全免（无 Y 无平面）。

对照 C1（位平面供数证书）：C1 省传输不省生产；本机制省生产+传输。
两机制在证书框架上同构（区间终止），不确定维不同（低位 bit vs 未算通道）。

口径：float64 精确算术（预实验）；thr 用 T13a 冻结 τ（checkpoint running stats）。
三种 (顺序, 界) 组合：
  oracle_order+oracle_bound：按逐组 |贡献| 降序 + 精确剩余界（收益上界）；
  wdesc+realistic_bound：静态 |W| 降序 + B_t·suffix|W| 界（可实现主口径）；
  natural+realistic_bound：自然 c 序 + 同界（免排序的保守口径）。
杀门：wdesc+realistic 的 MAC 节省 < 15% 则不立项在线算术/RTL。
自有代码；只读 traces 与 checkpoint。
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
    A, gamma = z['A'], z['gamma']
    R = A.sum(1).reshape(T, 1)
    direction = np.sign(gamma)

    rm = sd[pre + 'bn1.norm_layer.running_mean'].astype(np.float64)
    rv = sd[pre + 'bn1.norm_layer.running_var'].astype(np.float64)
    beta = sd[pre + 'bn1.norm_layer.bias'].astype(np.float64)
    core = (1.0 + z['center'] - z['bias'] - beta[None, :] * R)
    tau = (rm[None, :] * R + np.sqrt(rv[None, :] + 1e-5) / gamma[None, :] * core) \
        * direction[None, :]
    # U 侧自然尺度比较阈值（T5 合同的去量化版）：
    # D=+1: thr=tau_folded, dec=(U≥thr)；D=−1: thr=−tau_folded, dec=(U<thr)
    thr_nat = np.where(direction[None, :] < 0, -tau, tau)             # (T,H)
    # 部署量化判决（T5 口径）用于翻转对照
    A_q = np.rint(A.astype(np.float64) * 4096).astype(np.int64)
    tau_q = np.rint(tau * (1 << 14)).astype(np.int64)
    thr_q = np.where(direction[None, :] < 0, -(tau_q << 12) + 1, tau_q << 12)  # (T,H) f26

    rng = np.random.default_rng(SEED)
    ps = rng.integers(0, P, G_PER_TRACE)
    hs = rng.integers(0, H, G_PER_TRACE)
    B_t = np.abs(A).sum(1)                                    # (T,) 界乘子

    CH = 500 if C > 128 else 2000
    kstar = {k: [] for k in ('oracle', 'wdesc', 'natural')}
    dec_flips = 0
    dec_total = 0
    for lo in range(0, G_PER_TRACE, CH):
        hi = min(G_PER_TRACE, lo + CH)
        gc = hi - lo
        Ssub = S.reshape(T, P, C)[:, ps[lo:hi], :]            # (T,gc,C)
        Wg = W[hs[lo:hi]]                                     # (gc,C)
        thr_g = thr_nat[:, hs[lo:hi]].T                       # (gc,T) 自然尺度
        D_g = np.broadcast_to((direction > 0)[hs[lo:hi]][:, None], (gc, T))
        u = np.einsum('ts,sgc->tgc', A, Ssub)                 # (T,gc,C)
        X = u * Wg[None]                                      # (T,gc,C) 逐通道贡献
        V_exact = X.sum(-1)                                   # (T,gc)

        # 部署量化判决（T5 口径）与融合自然判决的翻转对照
        Yw = np.einsum('sgc,gc->sg', Ssub, Wg)                # (T,gc) 每词 fc1 输出
        Yq = np.clip(np.rint(Yw * (1 << 14)), -(1 << 23), (1 << 23) - 1).astype(np.int64)
        V_q = np.einsum('gs,ts->gt', Yq.T, A_q)               # (gc,T) f26
        thrq_g = thr_q[:, hs[lo:hi]].T                        # (gc,T)
        dec_q = np.where(D_g, V_q >= thrq_g, ~(V_q >= thrq_g))
        dec_nat = np.where(D_g, V_exact.T >= thr_g, ~(V_exact.T >= thr_g))
        dec_flips += int((dec_q != dec_nat).sum())
        dec_total += dec_q.size

        # oracle 序：逐组按 Σ_t|X| 降序
        order_or = np.argsort(-np.abs(X).sum(0), axis=1)      # (gc,C)
        Xo = np.take_along_axis(X, order_or[None], axis=2)    # (T,gc,C)
        # wdesc 序：静态 |W| 降序（逐组）
        order_w = np.argsort(-np.abs(Wg), axis=1)             # (gc,C)
        Ws = np.take_along_axis(Wg, order_w, axis=1)          # (gc,C)
        Xw = np.take_along_axis(X, order_w[None], axis=2)
        # natural 序
        Xn = X

        for key, Xarr, Warr in (('oracle', Xo, None), ('wdesc', Xw, Ws),
                                ('natural', Xn, Wg)):
            P_acc = np.zeros((gc, T))
            if key == 'oracle':
                R_rem = np.abs(Xarr).sum(-1).T                 # (gc,T) 精确剩余界，逐通道递减
            else:
                suf = np.cumsum(np.abs(Warr)[:, ::-1], axis=1)[:, ::-1]  # (gc,C)
                suf = np.concatenate([suf[:, 1:], np.zeros((gc, 1))], axis=1)  # 处理后剩余
            ks = np.full(gc, C, np.int32)
            for c in range(C):
                P_acc += Xarr[:, :, c].T                       # (gc,T)
                if key == 'oracle':
                    R_rem -= np.abs(Xarr[:, :, c]).T
                else:
                    R_rem = B_t[None, :] * suf[:, c][:, None]
                lock = (P_acc - R_rem >= thr_g) | (P_acc + R_rem < thr_g)
                done = lock.all(1)
                ks[done & (ks == C)] = c + 1
            kstar[key].append(ks)

    out = {'trace': trace.name, 'C': C, 'groups': G_PER_TRACE,
           'dec_flips_fused_vs_deployed': dec_flips,
           'dec_flip_rate': dec_flips / dec_total}
    for key in ('oracle', 'wdesc', 'natural'):
        ks = np.concatenate(kstar[key])
        out[f'kstar_mean_{key}'] = float(ks.mean())
        out[f'mac_saving_{key}'] = float(1 - ks.mean() / C)
        out[f'kstar_p90_{key}'] = float(np.percentile(ks, 90))
    return out


def main():
    sd = read_checkpoint(CKPT)['model_state_dict']
    results = [one_trace(tr, sd) for tr in TRACES]
    for r in results:
        print('%-22s C=%4d  oracle: k*=%.1f 省=%.1f%% | wdesc: k*=%.1f 省=%.1f%% '
              '| natural: k*=%.1f 省=%.1f%% | 翻转=%.4f%%' %
              (r['trace'], r['C'], r['kstar_mean_oracle'], 100 * r['mac_saving_oracle'],
               r['kstar_mean_wdesc'], 100 * r['mac_saving_wdesc'],
               r['kstar_mean_natural'], 100 * r['mac_saving_natural'],
               100 * r['dec_flip_rate']))
    avg = {k: float(np.mean([r[k] for r in results]))
           for k in results[0] if k.startswith(('kstar', 'mac'))}
    print('\n4-trace 均值: oracle 省=%.1f%% | wdesc(realistic) 省=%.1f%% | natural 省=%.1f%%'
          % (100 * avg['mac_saving_oracle'], 100 * avg['mac_saving_wdesc'],
             100 * avg['mac_saving_natural']))
    verdict = 'PASS(≥15%)' if avg['mac_saving_wdesc'] >= 0.15 else 'FAIL(<15%)'
    print('wdesc+realistic 杀门: %s' % verdict)
    summary = {'traces': results, 'average': avg, 'gate': 'wdesc realistic >=15%',
               'verdict': verdict,
               'mechanism': 'fused terminated production: V=Σ_c u[t,c]·W[c], '
                            'channel-dim certificate, Y never materialized'}
    (ROOT / 'results' / 't14_terminated_production.json').write_text(json.dumps(summary, indent=1))
    print('saved results/t14_terminated_production.json')


if __name__ == '__main__':
    main()
