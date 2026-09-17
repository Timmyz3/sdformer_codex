"""T8b（L2 补充）：逐词 MSB 的跨序列稳定性（静态表可行性）。

T8 主试验结论：优先级序按词打包真实可达成 11.2–12.4%（元数据免费口径），
但逐词 MSB 传输（边际熵 2.7–3.2 bit/词 + 符号 + 指数 ≈ 45bit ≈ +5 拍）同端口
计费后 32%，杀。本补充测唯一替代路径：离线标定静态 MSB 表（部署态常数）。

- 安全性：assumed ≥ actual 时区间仍 sound（更宽界、只是延迟锁定）；
  assumed < actual = 溢出（区间过窄、判决可能错），必须头拍 flag + 回退全深。
- 测量：固定 (p,h) 采样（与 T7 同 seed），s0 标定 / s10 测试，
  assumed = max(MSB_s0, 1) + pad；报溢出率（回退）与 MSB 跨序列差分布。
- 存储可行性：逐 (p,h) 表 = P×H×10 词×~4bit（stage0 ≈ 295Mb）——先看数值再议。
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HW = ROOT.parents[0]
BN = HW / 'bn_state'
STAGES = {'stage0': ('trace_s0_stage0.npz', 'trace_s10_stage0.npz'),
          'stage3': ('trace_s0_stage3.npz', 'trace_s10_stage3.npz')}
G = 20000
SEED = 20260915


def load_msb(trace, ps, hs):
    z = np.load(trace)
    C = int(z['W'].shape[1])
    S = np.unpackbits(z['source_packed'], axis=1, bitorder='little')[:, :C].astype(np.float64)
    N = S.shape[0]
    T, P = 10, N // 10
    W = z['W'].astype(np.float64)
    Ssub = S.reshape(T, P, C)[:, ps, :]
    Ysub = np.einsum('tgc,gc->tg', Ssub, W[hs])
    Yq = np.clip(np.rint(Ysub * (1 << 14)), -(1 << 23), (1 << 23) - 1).astype(np.int64)
    Yq = Yq.T                                                     # (G,T)
    msb = np.zeros_like(Yq, dtype=np.int16)
    for b in range(23, -1, -1):
        msb = np.where((msb == 0) & ((np.abs(Yq) >> b) > 0), b + 1, msb)
    return msb, Yq


def main():
    rng = np.random.default_rng(SEED)
    results = []
    for stage, (f0, f1) in STAGES.items():
        z0 = np.load(BN / f0)
        P0 = z0['source_packed'].shape[0] // 10
        H0 = z0['W'].shape[0]
        ps = rng.integers(0, P0, G)
        hs = rng.integers(0, H0, G)
        msb_a, _ = load_msb(BN / f0, ps, hs)                      # s0 标定
        msb_b, _ = load_msb(BN / f1, ps, hs)                      # s10 测试

        gap = msb_a.astype(int) - msb_b.astype(int)               # >0: 测试更小（安全方向）
        overflow0 = float((msb_b > np.maximum(msb_a, 1)).any(1).mean())
        table = {}
        for pad in (0, 1, 2, 3):
            assumed = np.maximum(msb_a, 1) + pad
            ov = float((msb_b > assumed).any(1).mean())
            waste = float((assumed - msb_b).mean())               # 平均过估（冗余符号位）
            table['pad%d' % pad] = {'overflow': ov, 'overestimate_bits': waste}
        results.append({
            'stage': stage,
            'per_word_gap_mean': float(gap.mean()),
            'per_word_gap_std': float(gap.std()),
            'per_word_gap_hist': {str(int(k)): int(v) for k, v in
                                  zip(*np.unique(gap, return_counts=True)) if abs(k) <= 5},
            'any_word_overflow_pad0': overflow0,
            'pads': table,
            'storage_bits_per_group_raw': 40,
        })
        print(stage, 'gap mean=%.2f std=%.2f  overflow(pad0)=%.4f  P(|gap|<=1)=%.3f' %
              (gap.mean(), gap.std(), overflow0,
               float((np.abs(gap) <= 1).mean())))
        for k, v in table.items():
            print('  %s overflow=%.4f overest=%.2fb/词' % (k, v['overflow'], v['overestimate_bits']))

    out = {'trials': results, 'G': G, 'note': 'T8b 逐词 MSB 跨序列稳定性（静态表可行性）。'}
    (ROOT / 'results' / 't8b_static_msb.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
