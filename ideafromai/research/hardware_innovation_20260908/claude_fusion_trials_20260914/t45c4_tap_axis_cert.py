#!/usr/bin/env python3
"""T45c-4：通道/抽头轴上的精确证书能不能早锁？（真实权重 + 真实输入脉冲）

## 为什么问这个

T45b 把靶点分布修正成三足鼎立后，一个自然的想法是"把 C1 的证书搬到全网"。
C1 的证书作用在**被逐位送达的多比特操作数**上（fc1→sn2 门路的 Y）。
但全网 83% 的算子（frontend / decoder / encoder 的 attention+downsample / bottleneck）
**输入是 0/1 脉冲**（本步实测 `input_first_binary01_ratio == 1.0`）——没有位平面可裁。

那么"退一步，在**抽头/通道轴**上做同样的精确证书"行不行？即：
把 in_ch×k×k 个抽头按 |w| 降序喂进累加器，用**剩余抽头的可达区间**做证书，
一旦区间与阈值单侧分离就锁定、跳过剩余抽头。

这就是 CGNet 通道前缀 / ICCD'24 "Early"（权重重要度早退）的**精确版**。
本脚本用**真实权重**（ep34 checkpoint）+ **真实输入脉冲张量**（m1458 捕获里保留的
payload，逐位打包）在瓶颈 resblock 的 768ch 3×3 卷积上实测它的锁定点。

## 判据（两条，都不需要 BN 建模）

1. **相对窗口 < 1**（阈值无关的**必要条件**）：第 m 个抽头后区间宽度
   `hi−lo = Σ_{j>m}|w_j|`，而量的尺度是 `|acc| = |Σ w s|`。
   只要 `Σ_{j>m}|w_j| ≥ |acc|`，**任何阈值规则都无法锁定**（区间比被比较的量还宽）。
   记最小的 m 为 m*/M。
2. **带阈值的锁定点**：阈值取"能复现该神经元实测发放率的经验分位数"，
   然后逐 (输出, 通道) 精确递推 `lo = part + Σ_{j>m} min(0,w_j)`、
   `hi = part + Σ_{j>m} max(0,w_j)`，锁定条件 `lo ≥ thr` 或 `hi < thr`。

用法：/opt/anaconda3/envs/pytorch310/bin/python t45c4_tap_axis_cert.py
"""
from __future__ import annotations

import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CODE = Path('/home/zhumd/work/sdformer_codex/SDformer')
CAP = (CODE / 'hw_autoresearch_nts07/results'
       / 'm1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831')
CKPT = (CODE / 'hw_autoresearch_nts07/system_handoff/incoming'
        / 'm2041_ep34_quant_binding_inputs/checkpoint_epoch34.pth')

# (conv 名, 对应 ATLIF 名) —— payload 只对 c1_conv3x3 / decoder_convtranspose 保留
CASES = [
    ('sttmultires_unet.resblocks.0.conv1.0',
     'sttmultires_unet.resblocks.0.sn2'),
    ('sttmultires_unet.resblocks.1.conv1.0',
     'sttmultires_unet.resblocks.1.sn2'),
]
NCH = 48           # 采样多少个输出通道（共 768）


def im2col3(x, T, H, W, k=3, pad=1):
    """x: (T*H*W, Cin)；返回 (T*H*W, Cin*k*k)，通道主序 i*9+kr*3+kc（不跨 t 混合）。"""
    P, C = x.shape
    assert P == T * H * W
    im = np.pad(x.reshape(T, H, W, C), ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    cols = np.empty((T, H, W, C, k, k), dtype=x.dtype)
    for kr in range(k):
        for kc in range(k):
            cols[:, :, :, :, kr, kc] = im[:, kr:kr + H, kc:kc + W, :]
    return cols.reshape(T * H * W, C * k * k)


def main():
    import torch

    st = torch.load(CKPT, map_location='cpu', weights_only=False)['model_state_dict']
    recs = [json.loads(l) for l in (CAP / 'unified_ordered_records.jsonl').read_text().splitlines()]
    by_name = {}
    for r in recs:
        if (r.get('payload') or {}).get('retained'):
            by_name.setdefault(r['name'], r)
    atlif = {e['name'].replace('.spiking_neuron', ''): e
             for e in json.loads((CAP / 'atlif_activity.json').read_text())}

    out = {'cases': []}
    for cname, neuron in CASES:
        rec = by_name[cname]
        assert rec['name'] == cname, (rec['name'], cname)
        p = rec['payload']
        z = (CAP / p['compressed_fp32']).read_bytes()
        raw = zlib.decompress(z)
        x = np.frombuffer(raw, '<f4')
        shape = rec['input']['shape']
        assert x.size == int(np.prod(shape)), (x.size, shape)
        x = x.reshape(shape)                                   # [T, 1, Cin, H, W]
        T, _, Cin, H, W = x.shape
        X = x.transpose(0, 3, 4, 2, 1).reshape(T * H * W, Cin)  # (P, Cin)

        Wt = st[cname + '.weight'].numpy().astype(np.float64)   # (Cout, Cin, 3, 3)
        Bt = (st[cname + '.bias'].numpy().astype(np.float64)
              if cname + '.bias' in st else np.zeros(Wt.shape[0]))
        Cout = Wt.shape[0]
        assert Wt.shape[1] == Cin and Wt.shape[2:] == (3, 3)

        P = im2col3(X.astype(np.float64), T, H, W)              # (P, Cin*9)
        acc = (P @ Wt.reshape(Cout, -1).T + Bt).T               # (Cout, P)

        rho = float((X != 0).mean())
        act = float(atlif[neuron]['activity']) if neuron in atlif else None
        thr = float(np.quantile(acc, 1.0 - act)) if act else 0.0

        rng = np.random.default_rng(0)
        chs = rng.choice(Cout, size=min(NCH, Cout), replace=False)
        M = Cin * 9
        m_rel = np.full((len(chs), acc.shape[1]), M, dtype=np.int32)   # 相对窗口<1 的首个 m
        m_thr = np.full((len(chs), acc.shape[1]), M, dtype=np.int32)   # 阈值锁定点
        m_sgn = np.full((len(chs), acc.shape[1]), M, dtype=np.int32)   # 零阈值（定符号）
        tail_frac = []                                                 # 尾 L1 占比曲线

        for ci, o in enumerate(chs):
            w = Wt[o].reshape(-1)
            perm = np.argsort(-np.abs(w))
            wp = w[perm]
            contrib = P[:, perm] * wp                                   # (P, M)
            part = np.cumsum(contrib, axis=1)
            pos = np.where(wp > 0, wp, 0.0)
            neg = np.where(wp < 0, wp, 0.0)
            tail_p = np.cumsum(pos[::-1])[::-1] - pos                   # Σ_{j>m} max(0,w)
            tail_n = np.cumsum(neg[::-1])[::-1] - neg                   # Σ_{j>m} min(0,w)
            absrev = np.cumsum(np.abs(wp)[::-1])[::-1]
            tail_abs = absrev - np.abs(wp)                              # Σ_{j>m} |w_j|
            acco = acc[o]
            rel = tail_abs[None, :] / np.maximum(np.abs(acco)[:, None], 1e-12)
            ok = rel < 1.0
            m_rel[ci] = np.where(ok.any(axis=1), ok.argmax(axis=1), M)
            hi = part + tail_p
            lo = part + tail_n
            lock = (lo >= thr) | (hi < thr)
            m_thr[ci] = np.where(lock.any(axis=1), lock.argmax(axis=1), M)
            l0 = (lo >= 0) | (hi < 0)
            m_sgn[ci] = np.where(l0.any(axis=1), l0.argmax(axis=1), M)
            if ci == 0:
                tail_frac = (tail_abs / np.abs(w).sum()).tolist()
                sig_rel = float(np.std(acco) / np.abs(w).sum())

        tot = m_thr.size
        out['cases'].append({
            'conv': cname, 'neuron': neuron,
            'input_shape': shape, 'Cin': Cin, 'Cout': Cout, 'T': T,
            'P_positions': int(acc.shape[1]), 'taps_per_out': M,
            'input_density_measured': rho, 'neuron_activity': act, 'thr': thr,
            'n_out_sampled': int(tot),
            'rel_window_lock_frac': float((m_rel < M).mean()),
            'rel_window_lock_mean_m_over_M': float(m_rel.mean() / M),
            'thr_lock_frac': float((m_thr < M).mean()),
            'thr_lock_mean_m_over_M': float(m_thr.mean() / M),
            'thr_lock_median_m_over_M': float(np.median(m_thr) / M),
            # 锁步阵列的真实代价：一个通道的所有位置同时推进 ⇒ 看逐通道 max
            'thr_lock_lockstep_m_over_M': float(np.median(m_thr.max(axis=1)) / M),
            'sgn_lock_mean_m_over_M': float(m_sgn.mean() / M),
            'sgn_lock_lockstep_m_over_M': float(np.median(m_sgn.max(axis=1)) / M),
            'sigma_over_L1': sig_rel,
            'predicted_1_minus_rho': 1.0 - rho,
        })
        # 相对窗口在小 m 处的值（尾 L1 尚未塌到 |acc| 以下）
        c = out['cases'][-1]
        c['tail_L1_frac_at_m_over_M'] = {
            f'{q:.2f}': float(tail_frac[int(q * (M - 1))]) for q in (0.25, 0.5, 0.75, 0.9, 0.95)
        }
        print('=== %s（density %.4f, 抽头 M=%d, 采样 %d 个输出通道）==='
              % (cname, rho, M, len(chs)))
        print('  相对窗口<1 的锁定率 %.3f，平均 m*/M = %.4f（预测下界 1-ρ = %.4f）'
              % (c['rel_window_lock_frac'], c['rel_window_lock_mean_m_over_M'],
                 c['predicted_1_minus_rho']))
        print('  带阈值锁定率 %.3f，m*/M 均值 %.4f 中位 %.4f'
              % (c['thr_lock_frac'], c['thr_lock_mean_m_over_M'],
                 c['thr_lock_median_m_over_M']))
        print('  锁步阵列（逐通道取 max）m*/M = %.4f ⇒ 可省抽头 %.1f%%（oracle 逐位置 %.1f%%）'
              % (c['thr_lock_lockstep_m_over_M'],
                 100 * (1 - c['thr_lock_lockstep_m_over_M']),
                 100 * (1 - c['thr_lock_mean_m_over_M'])))
        print('  零阈值(定符号) 均值 m*/M %.4f，锁步 %.4f'
              % (c['sgn_lock_mean_m_over_M'], c['sgn_lock_lockstep_m_over_M']))
        print('  σ(acc)/Σ|w| = %.3e  ⇒ 尾 L1 要降到该尺度需 m/M ≥ %.4f'
              % (c['sigma_over_L1'], 1 - c['sigma_over_L1']))
        print('  尾 L1 占比 @ m/M: ' + '  '.join(
            '%s→%.3f' % (k, v) for k, v in c['tail_L1_frac_at_m_over_M'].items()))
        print()

    (ROOT / 'results' / 't45c4_tap_axis_cert.json').write_text(
        json.dumps(out, indent=1) + '\n')
    print('wrote results/t45c4_tap_axis_cert.json')


if __name__ == '__main__':
    main()
