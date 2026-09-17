"""T9（L5+L4，照抄对象 EITCE26 / ECHO Electronics24）：

L5 = EITCE 式通道序部分和 bound 终止 vs 我们的位平面序。
EITCE（FPGA CNN）：bit-parallel 通道累加，部分和 vs 剩余正贡献上界，可证
pre-activation ≤ 0 则无损跳过剩余 MAC。套到门路判决（V_t = Σ_s A[t,s]·Y_s
≥ thr_t，"通道"= 10 个贡献词）：
- 词序供数（整词），供应后该词不确定性归零；未供词按格式定界：
  (a) FX-24：Y_s ∈ [−2^23, 2^23)；
  (b) BFP-e（组指数 e 头拍传输，与 T5 同）：Y_s ∈ [−2^e, 2^e)；
- 供数顺序两版：静态（w_s = max_t|A_q[t,s]| 降序，部署态免元数据）与
  oracle（max_t|A·Y| 降序，上界）；
- 锁定 = 全 10 单元区间判定；cycles = 1 + ceil(bits/10)（10-lane 同端口），
  bits = k·24 或 k·(e+1)；
- 对照：位平面序（T5 bf_cert，1 + (e−j*)）。

L4 = ECHO 式仅符号终止：MSDF 在线算术的"输出符号确定即终止"等价于
零数据平面锁定（仅符号字+幅度界）。测 P(锁 | 0 数据平面)。

预期判读：位平面序对供数受限的门判决结构性占优（所有词同步粗分辨率精化，
界每平面减半；整词供数在尾部词供完前界不闭合）。
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
    Ysub = np.einsum('tgc,gc->tg', Ssub, W[hs])
    Yq = np.clip(np.rint(Ysub * (1 << 14)), -(1 << 23), (1 << 23) - 1).astype(np.int64)
    Yq = Yq.T
    thr_g = thr[:, hs].T                                            # (G,T)

    # 位平面序基准（T5 bf_cert）+ 零平面锁定率（L4/ECHO）
    j_first = np.full((G, T), -1, np.int8)
    frozen = np.zeros((G, T), bool)
    for j in range(23, -1, -1):
        Ytop = Yq >> j
        Vtop = np.einsum('gs,ts->gt', Ytop, A_q)
        Vmin = (Vtop << j) + N_t[None, :] * ((1 << j) - 1)
        Vmax = (Vtop << j) + P_t[None, :] * ((1 << j) - 1)
        lock = (Vmin >= thr_g) | (Vmax < thr_g)
        j_first[lock & ~frozen] = j
        frozen |= lock
    assert frozen.all()
    j_star = j_first.min(1)
    e_g = np.array([int(np.abs(Yq[g]).max()).bit_length() for g in range(G)])
    cyc_bitplane = 1 + np.maximum(e_g - j_star, 1)
    echo_zero_plane = float((j_star >= e_g).mean())                 # L4：仅符号锁定率

    # 通道序（L5/EITCE）
    Abig = A_q.astype(np.int64)
    Apos = Abig.clip(min=0)
    Aneg = Abig.clip(max=0)
    w_s = np.array([int(np.abs(A_q[:, s]).max()) for s in range(T)])
    static_order = np.argsort(-w_s)                                  # 部署态（静态）

    def channel_cycles(order_kind):
        """通道序：整词供数，未供词按格式定界（fx24: ±2^23；bfp-e: ±2^e）。
        逐单元剩余界：Vmin_t = Vt − remP_t·B + remN_t·(B−1)，
                      Vmax_t = Vt + remP_t·(B−1) − remN_t·B，
        remP/remN = 未供词 A 的正/负部列和。"""
        cyc = {'fx24': np.zeros(G, np.int64), 'bfpe': np.zeros(G, np.int64)}
        kwords = {'fx24': np.zeros(G, np.int64), 'bfpe': np.zeros(G, np.int64)}
        for g in range(G):
            Yg = Yq[g]
            if order_kind == 'static':
                order = static_order
            else:                                                    # oracle：按实际影响降序
                order = np.argsort(-np.abs(Abig * Yg[None, :]).max(0))
            thr_t = thr_g[g]
            Bmap = {'fx24': 1 << 23, 'bfpe': 1 << int(e_g[g])}
            bitsw = {'fx24': 24, 'bfpe': int(e_g[g]) + 1}
            Vt = np.zeros(T, np.int64)
            remP = Apos.sum(1).astype(np.int64)                       # (T,) 未供正部和
            remN = Aneg.sum(1).astype(np.int64)                       # (T,) 未供负部和
            got = {'fx24': False, 'bfpe': False}
            for idx, s in enumerate(order):
                s = int(s)
                Vt = Vt + Abig[:, s] * int(Yg[s])
                remP = remP - Apos[:, s]
                remN = remN - Aneg[:, s]
                k = idx + 1
                for fmt in ('fx24', 'bfpe'):
                    if got[fmt]:
                        continue
                    B = Bmap[fmt]
                    Vmin = Vt - remP * B + remN * (B - 1)
                    Vmax = Vt + remP * (B - 1) - remN * B
                    if int(((Vmin >= thr_t) | (Vmax < thr_t)).sum()) == T:
                        got[fmt] = True
                        kwords[fmt][g] = k
                if got['fx24'] and got['bfpe']:
                    break
            for fmt in ('fx24', 'bfpe'):                              # 全供必锁（k=10）
                if not got[fmt]:
                    kwords[fmt][g] = T
                cyc[fmt][g] = 1 + max(-(-int(kwords[fmt][g]) * bitsw[fmt] // 10), 1)
        cyc['kwords'] = kwords['bfpe']
        return cyc

    cs = channel_cycles('static')
    co = channel_cycles('oracle')
    return {
        'trace': str(trace.relative_to(HW)),
        'bitplane_ratio': float(cyc_bitplane.mean() / FX_BASE),
        'echo_zero_plane_lock_rate': echo_zero_plane,
        'channel_static_fx24_ratio': float(cs['fx24'].mean() / FX_BASE),
        'channel_static_bfpe_ratio': float(cs['bfpe'].mean() / FX_BASE),
        'channel_oracle_fx24_ratio': float(co['fx24'].mean() / FX_BASE),
        'channel_oracle_bfpe_ratio': float(co['bfpe'].mean() / FX_BASE),
        'channel_oracle_kwords_mean': float(co['kwords'].mean()),
    }


def main():
    results = [one_trace(t) for t in TRACES]
    for r in results:
        print('%-28s bitplane=%.4f echo0=%.4f | ch_static fx24=%.4f bfp-e=%.4f | '
              'ch_oracle fx24=%.4f bfp-e=%.4f k=%.2f' %
              (r['trace'], r['bitplane_ratio'], r['echo_zero_plane_lock_rate'],
               r['channel_static_fx24_ratio'], r['channel_static_bfpe_ratio'],
               r['channel_oracle_fx24_ratio'], r['channel_oracle_bfpe_ratio'],
               r['channel_oracle_kwords_mean']))
    out = {'traces': results, 'G': G,
           'note': 'T9 L5 EITCE 通道序 vs 位平面序 + L4 ECHO 仅符号锁定率。'}
    (ROOT / 'results' / 't9_channel_order.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
