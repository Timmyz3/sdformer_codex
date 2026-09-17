"""T8（L2，照抄对象 AO-BFP DATE26 / DMP-BFP ICCD25）：离群感知组供数。

AO-BFP/DMP-BFP 的机制是运行时按指数/阈值调整 BFP 精度（离群值单独处理）。
套用到 C1：组级 BFP 的共享指数 e 由组内最大 |Y| 决定，小幅度词在高位平面
全是 0，占用 10-lane 供数口的空闲 lane。量化离群结构与"按词打包"空间：

测量（四 trace，与 T5 同口径采样 20000 组/trace）：
1. 组内幅度结构：m1..m10（|Y_s| 的 MSB 降序）的 m1−m2、m1−m5 分布；
2. 供数 lane 利用率：平面 p ∈ [j*, e) 上活跃词数（MSB_s > p）的均值占比；
3. **按词打包可达成上界**：同 j* 锁深下每词实际需要 bit 数 = max(MSB_s−j*, 0)
   （低于 j* 的位被区间界吸收，符号在头拍），打包进 10-lane 拍：
   cycles = 1 + ceil(Σ_s max(MSB_s−j*,0) / 10)。判决仍精确（终态区间与
   统一 j* 供数等价），中间早锁忽略（保守）。
4. 对照：统一平面供数 cycles = 1 + (e−j*)。

判读：若打包上界 ≈ 统一供数（无 headroom）→ 离群结构不可利用，杀；
若显著更低 → 设计真实打包机制（lane 重映射 + 逐词深度跟踪的异构区间）。

边界：数值试验；j* 取统一供数下的锁深（保守）；仅门路。
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HW = ROOT.parents[0]
TRACES = sorted((HW / 'bn_state').glob('trace_*.npz'))
G = 20000
SEED = 20260914
FX_BASE = 24


def to_signed(v, bits):
    out = np.asarray(v, dtype=np.int64)
    assert np.all(out >= -(1 << (bits - 1))) and np.all(out <= (1 << (bits - 1)) - 1)
    return out


def one_trace(trace):
    z = np.load(trace)
    C = int(z['W'].shape[1])
    S = np.unpackbits(z['source_packed'], axis=1, bitorder='little')[:, :C].astype(np.float64)
    N = S.shape[0]
    T, P = 10, N // 10
    H = z['W'].shape[0]
    W = z['W'].astype(np.float64)
    A, gamma, beta = z['A'], z['gamma'], z['beta']
    bias, center = z['bias'], z['center']
    theta = 1.0
    R = A.sum(1).reshape(T, 1)
    direction = np.sign(gamma)

    tau = np.zeros((T, H))
    for lo in range(0, H, 32):
        hi = min(H, lo + 32)
        Y = (S @ W[lo:hi].T).reshape(T, P, hi - lo)
        mu = Y.mean((0, 1))
        var = ((Y - mu) ** 2).mean((0, 1))
        tau[:, lo:hi] = (mu * R + np.sqrt(var + 1e-5) / gamma[lo:hi]
                         * (theta + center - bias - beta[lo:hi] * R)) * direction[None, lo:hi]
        del Y
    A_q = to_signed(np.rint(A.astype(np.float64) * 4096), 16)
    P_t = A_q.clip(min=0).sum(1)
    N_t = A_q.clip(max=0).sum(1)
    tau_q = np.rint(tau * (1 << 14)).astype(np.int64)
    thr = np.where(direction[None, :] < 0, -(tau_q << 12) + 1, tau_q << 12)
    thr = to_signed(thr, 48)

    rng = np.random.default_rng(SEED)
    ps = rng.integers(0, P, G)
    hs = rng.integers(0, H, G)
    Ssub = S.reshape(T, P, C)[:, ps, :]
    Wsub = W[hs]
    Ysub = np.einsum('tgc,gc->tg', Ssub, Wsub)
    Yq = np.clip(np.rint(Ysub * (1 << 14)), -(1 << 23), (1 << 23) - 1).astype(np.int64)
    Yq = Yq.T                                                        # (G,T)

    # j*（统一供数锁深，与 T5 同）
    thr_g = thr[:, hs].T
    j_first = np.full((G, T), -1, np.int8)
    frozen = np.zeros((G, T), bool)
    for j in range(23, -1, -1):
        Ytop = Yq >> j
        Vtop = np.einsum('gs,ts->gt', Ytop, A_q)
        Vmin = (Vtop << j) + N_t[None, :] * ((1 << j) - 1)
        Vmax = (Vtop << j) + P_t[None, :] * ((1 << j) - 1)
        lock = (Vmin >= thr_g) | (Vmax < thr_g)
        newly = lock & ~frozen
        j_first[newly] = j
        frozen |= lock
    assert frozen.all()
    j_star = j_first.min(1)

    # msb[s] = |Y_s| 的 bit_length（0 -> 0）
    msb = np.zeros_like(Yq, dtype=np.int8)
    for b in range(23, -1, -1):
        msb = np.where((msb == 0) & ((np.abs(Yq) >> b) > 0), b + 1, msb)

    e_g = msb.max(1)
    # 幅度结构
    msb_sorted = -np.sort(-msb, axis=1)                               # 降序
    d12 = msb_sorted[:, 0] - msb_sorted[:, 1]
    d15 = msb_sorted[:, 0] - msb_sorted[:, 4]
    # lane 利用率：平面 p 活跃词 = MSB_s > p（该平面此词有非零可能位）
    # 供数平面范围 [j*, e)；对每组取平均活跃占比
    act_frac = []
    bits_packed = np.zeros(G, np.int64)
    for g in range(G):
        js, eg = int(j_star[g]), int(e_g[g])
        if eg > js:
            planes = np.arange(js, eg)                                # 供数平面
            act = (msb[g][None, :] > planes[:, None]).sum(1)          # 每平面活跃词数
            act_frac.append(act.mean() / 10.0)
        bits_packed[g] = np.maximum(msb[g] - js, 0).sum()

    cyc_unified = 1 + np.maximum(e_g - j_star, 1)
    cyc_packed = 1 + np.maximum(np.ceil(bits_packed / 10.0).astype(np.int64), 1)
    # 每词需要的位深直方（打包机制的设计输入）
    need = np.maximum(msb - j_star[:, None], 0)

    # ---- 优先级序运行时仿真（真实可达成，非 oracle）----
    # bit (s,ℓ) 优先级 = ℓ + w_s（w_s = max_t|A_q[t,s]| 的 bit_length，静态），
    # 每 10 bit 一拍，拍尾做异构深度区间锁定检查：
    #   Vmin_t = Σ_s A·(Ytop_s<<j_s) + Σ_{A<0} A·(2^{j_s}−1)（Vmax 对称）
    #   Ytop_s init = −sign_s（符号扩展），j_s init = MSB_s（元数据给出）
    w_s = np.array([int(np.abs(A_q[:, s]).max()).bit_length() for s in range(T)])
    Abig = A_q.astype(np.int64)
    Apos = Abig.clip(min=0)
    Aneg = Abig.clip(max=0)
    thr_g = thr[:, hs].T
    cyc_prio = np.zeros(G, np.int64)
    bits_prio = np.zeros(G, np.int64)
    for g in range(G):
        Yg, mg = Yq[g], msb[g]
        sign = (Yg < 0).astype(np.int64)
        bits = []
        for s in range(T):
            for l in range(int(mg[s]) - 1, -1, -1):
                bits.append((l + int(w_s[s]), s, int((Yg[s] >> l) & 1)))
        bits.sort(key=lambda x: -x[0])
        Ytop = -sign.copy()
        js = mg.astype(np.int64).copy()
        dec_bits = 0
        ok = False
        for bi, (pr, s, b) in enumerate(bits):
            Ytop[s] = (Ytop[s] << 1) + b
            js[s] = pr - int(w_s[s])                       # = 该 bit 的层号 ℓ
            dec_bits += 1
            if dec_bits % 10 == 0 or bi == len(bits) - 1:
                Vt = Abig @ (Ytop << js)
                rem = (np.int64(1) << js) - 1
                Vmin = Vt + Aneg @ rem
                Vmax = Vt + Apos @ rem
                thr_t = thr_g[g]
                if int(((Vmin >= thr_t) | (Vmax < thr_t)).sum()) == T:
                    ok = True
                    break
        assert ok                                            # 全供数必锁
        cyc_prio[g] = 1 + max(-(-dec_bits // 10), 1)
        bits_prio[g] = dec_bits

    # ---- MSB 元数据成本（同端口计费）----
    # 条件于 e 的逐词边际熵之和（独立编码上界；联合熵只会更低）
    dm = (e_g[:, None] - msb)                              # (G,T) e−MSB_s ≥ 0
    dm_valid = dm[dm < 24]
    vals, cnts = np.unique(dm_valid, return_counts=True)
    p = cnts / cnts.sum()
    H_marg = float(-(p * np.log2(p)).sum())                # 每词平均 bit（边际）
    meta_bits = 10 + 5 + H_marg * 10                       # 符号+组指数+10 词 MSB
    return {
        'trace': str(trace.relative_to(HW)),
        'unified_ratio': float(cyc_unified.mean() / FX_BASE),
        'packed_oracle_ratio': float(cyc_packed.mean() / FX_BASE),
        'packed_prio_ratio': float(cyc_prio.mean() / FX_BASE),
        'bits_prio_mean': float(bits_prio.mean()),
        'msb_marginal_entropy_bits': H_marg,
        'meta_bits_mean': float(meta_bits),
        'packed_prio_ratio_with_meta': float(
            (1 + np.ceil((bits_prio + meta_bits) / 10.0)).mean() / FX_BASE),
        'lane_utilization_mean': float(np.mean(act_frac)),
        'bits_per_group_packed_mean': float(bits_packed.mean()),
        'bits_per_group_unified_mean': float((np.maximum(e_g - j_star, 0) * 10).mean()),
        'd12_mean': float(d12.mean()), 'd12_p90': float(np.quantile(d12, 0.9)),
        'd15_mean': float(d15.mean()),
        'e_mean': float(e_g.mean()), 'j_star_mean': float(j_star.mean()),
    }


def main():
    results = [one_trace(t) for t in TRACES]
    for r in results:
        print('%-28s unified=%.4f oracle=%.4f prio=%.4f prio+meta=%.4f '
              '(H_msb=%.2fb/词, lane_util=%.3f)' %
              (r['trace'], r['unified_ratio'], r['packed_oracle_ratio'],
               r['packed_prio_ratio'], r['packed_prio_ratio_with_meta'],
               r['msb_marginal_entropy_bits'], r['lane_utilization_mean']))
    out = {'traces': results, 'G': G,
           'note': 'T8/L2 离群感知按词打包（AO-BFP 式）：oracle=同j*位计数；'
                   'prio=优先级序运行时仿真；+meta=同端口计入 MSB 元数据'
                   '（10符号+5指数+边际熵×10词）。'}
    (ROOT / 'results' / 't8_outlier_packed.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
