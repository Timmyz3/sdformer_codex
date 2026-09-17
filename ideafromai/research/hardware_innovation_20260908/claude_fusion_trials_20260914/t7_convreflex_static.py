"""T7（L3，照抄对象 ConvReflex, SenSys26）：静态锁深调度 vs 运行时证书。

ConvReflex 的机制是编译期捷径（离线分析出必被 clamp 的卷积、运行时直接跳过）。
套用到 C1 门路：离线标定每 (p,h) 组的静态供数深度 d，运行时按表供数、免区间
证书逻辑。可行性取决于 j*（组首锁平面）的跨序列稳定性。

设计：
- 每层（stage0/stage3）固定同一组 (p,h) 采样（G=20000，seed 20260915），在
  s0 与 s10 两条 trace 上各算一次 j* 与组 MSB e（数据依赖）；
- 口径与 T5 一致（U 侧 thr、BF 形态、FX 基线 24 拍/组）；
- 对比三种形态的拍比与误 fire 率：
  (a) 运行时证书（T5 基准）：cycles = 1 + max(e−j*, 1)；
  (b) 全局静态深度（MINT 式逐层常数）：d = j*_train 的分位数；
  (c) 逐组静态表（ConvReflex 式）：d[(p,h)] = j*_train，跨 trace 测试；
      误 fire = j*_test < d（供数不足、判决未锁）；带回退计费
      （误 fire 组继续供到平面 0：cycles = 1 + e）。
- 另报 j*_s0 − j*_s10 的逐组差分布（稳定性证据）。

边界：数值试验（非 RTL）；tau 用各 trace 终态矩（与 T5 同口径）；仅门路。
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
FX_BASE = 24


def to_signed(v, bits):
    out = np.asarray(v, dtype=np.int64)
    assert np.all(out >= -(1 << (bits - 1))) and np.all(out <= (1 << (bits - 1)) - 1)
    return out


def load_groups(trace, ps, hs):
    """与 T5 同口径：终态 tau、U 侧 thr、位平面 j*、组 MSB e。"""
    z = np.load(trace)
    C = int(z['W'].shape[1])
    S = np.unpackbits(z['source_packed'], axis=1, bitorder='little')[:, :C].astype(np.float64)
    N = S.shape[0]
    T, P = 10, N // 10
    W = z['W'].astype(np.float64)
    A, gamma, beta = z['A'], z['gamma'], z['beta']
    bias, center = z['bias'], z['center']
    theta = 1.0
    R = A.sum(1).reshape(T, 1)
    direction = np.sign(gamma)

    tau = np.zeros((T, len(hs)))
    for lo in range(0, len(hs), 32):
        hi = min(len(hs), lo + 32)
        hh = hs[lo:hi]
        Y = (S @ W[hh].T).reshape(T, P, hi - lo)
        mu = Y.mean((0, 1))
        var = ((Y - mu) ** 2).mean((0, 1))
        tau[:, lo:hi] = (mu * R + np.sqrt(var + 1e-5) / gamma[hh]
                         * (theta + center - bias - beta[hh] * R)) * direction[None, hh]
        del Y
    A_q = to_signed(np.rint(A.astype(np.float64) * 4096), 16)
    P_t = A_q.clip(min=0).sum(1)
    N_t = A_q.clip(max=0).sum(1)
    tau_q = np.rint(tau * (1 << 14)).astype(np.int64)               # (T,Gh)
    thr = np.where(direction[hs][None, :] < 0,
                   -(tau_q << 12) + 1, tau_q << 12)                  # (T,Gh) U 侧
    thr = to_signed(thr, 48)

    Ssub = S.reshape(T, P, C)[:, ps, :]                              # (T,G,C)
    Wsub = W[hs]                                                     # (G,C)
    Ysub = np.einsum('tgc,gc->tg', Ssub, Wsub)                       # (T,G)
    Yq = np.clip(np.rint(Ysub * (1 << 14)), -(1 << 23), (1 << 23) - 1).astype(np.int64)
    Yq = Yq.T                                                        # (G,T) -> Y[s]

    j_first = np.full((G, T), -1, np.int8)
    frozen = np.zeros((G, T), bool)
    for j in range(23, -1, -1):
        Ytop = Yq >> j
        Vtop = np.einsum('gs,ts->gt', Ytop, A_q)
        m = j
        Vmin = (Vtop << m) + N_t[None, :] * ((1 << m) - 1)
        Vmax = (Vtop << m) + P_t[None, :] * ((1 << m) - 1)
        lock = (Vmin >= thr.T) | (Vmax < thr.T)
        newly = lock & ~frozen
        j_first[newly] = j
        frozen |= lock
    assert frozen.all()
    j_star = j_first.min(1)
    e_g = np.array([int(np.abs(Yq[g]).max()).bit_length() for g in range(G)])
    return j_star, e_g


def cycles_bf(e, d, misfire):
    """BF 形态：头拍 + 数据平面 e−1..d；误 fire 回退继续到平面 0。"""
    n = np.maximum(e - d, 1)
    n = np.where(misfire, e, n)
    return 1 + n


def main():
    rng = np.random.default_rng(SEED)
    results = []
    for stage, (f0, f1) in STAGES.items():
        z0 = np.load(BN / f0)
        P0 = z0['source_packed'].shape[0] // 10
        H0 = z0['W'].shape[0]
        ps = rng.integers(0, P0, G)
        hs = rng.integers(0, H0, G)
        j_a, e_a = load_groups(BN / f0, ps, hs)          # 标定 trace (s0)
        j_b, e_b = load_groups(BN / f1, ps, hs)          # 测试 trace (s10)

        # (a) 运行时证书基准（各自 trace）
        rt_a = float(cycles_bf(e_a, j_a, np.zeros(G, bool)).mean() / FX_BASE)
        rt_b = float(cycles_bf(e_b, j_b, np.zeros(G, bool)).mean() / FX_BASE)

        # (b) 全局静态深度（MINT 式）：标定 trace j* 分位数
        glob = {}
        for q in (0.5, 0.75, 0.9, 1.0):
            d = int(np.quantile(j_a, q))
            mis = j_b < d
            glob['q%d' % (q * 100)] = {
                'd': d,
                'ratio': float(cycles_bf(e_b, d, mis).mean() / FX_BASE),
                'misfire': float(mis.mean()),
            }

        # (c) 逐组静态表（ConvReflex 式）：d[(p,h)] = j*_train（含安全加深 pad）
        table = {}
        for pad in (0, 1, 2, 3, 5):
            d = j_a - pad
            mis = j_b < d
            table['pad%d' % pad] = {
                'ratio': float(cycles_bf(e_b, d, mis).mean() / FX_BASE),
                'misfire': float(mis.mean()),
                'no_fallback_ratio': float((1 + np.maximum(e_b - d, 1)).mean() / FX_BASE),
            }

        dj = j_a.astype(int) - j_b.astype(int)
        results.append({
            'stage': stage,
            'runtime_cert_ratio': {'s0': rt_a, 's10': rt_b},
            'global_static': glob,
            'per_group_static': table,
            'j_diff_mean': float(dj.mean()),
            'j_diff_std': float(dj.std()),
            'j_diff_abs_hist': {str(int(k)): int(v) for k, v in
                                zip(*np.unique(np.abs(dj), return_counts=True))},
            'j_star_mean': {'s0': float(j_a.mean()), 's10': float(j_b.mean())},
            'e_mean': {'s0': float(e_a.mean()), 's10': float(e_b.mean())},
        })
        print('%s: runtime(s10)=%.4f  per-group static pad0: ratio=%.4f misfire=%.4f' %
              (stage, rt_b, table['pad0']['ratio'], table['pad0']['misfire']))

    out = {'trials': results, 'G': G, 'seed': SEED, 'fx_base': FX_BASE,
           'note': 'T7/L3 ConvReflex 式静态锁深 vs 运行时证书；BF 形态；误 fire 带回退计费。'}
    (ROOT / 'results' / 't7_static_schedule.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out['trials'], indent=1))


if __name__ == '__main__':
    main()
