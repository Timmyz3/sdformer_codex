"""T5（C1 卡 A' RTL 前置）：证书终止门核的整数模型 + Verilator 激励生成。

模型（与 RTL 位级一致）：
- 每组 (p,h)：10 个判决单元 t=0..9 共享 10 个贡献字 Y_q[s,p,h]（signed24/f14）；
- 供数 = 24 个位平面 MSB-first（平面 j=23..0，每平面 10 bit，bit s = Y[s] 的第 j 位）；
- 消费平面 j 后剩余 m=j 位：Vtop_t = Σ_s A_q[t,s]·(Y_q>>j)，
  V ∈ [Vtop·2^j + N_t·(2^j−1), Vtop·2^j + P_t·(2^j−1)]，
  P_t=Σ_{A>0}A_q, N_t=Σ_{A<0}A_q（f12）；
- 锁定 = (Vmin ≥ thr) 或 (Vmax < thr)；判决 = (Vmin ≥ thr)；
  thr（U 侧，48bit 合同）：D=+1 时 tau_q·2^12；D=−1 时 −(tau_q·2^12)+1，
  输出按 Dflag 取反（tau 已乘 direction 而 Vtop 未乘，方向只折一次）；
- 组终止 = 全部 10 单元锁定（锁定后区间嵌套单调，判决冻结）；
- 基线 = 符号头拍 + 23 数据平面（bit23=符号字不重发）= 24 拍/组，
  FX 证书 = 1 + max(23−j*, 1) 拍/组。

验证：证书判决 == 全深度整数判决（逐组逐 t 断言）；
tau 用终态 BN 矩（部署态静态常数，非 oracle 限制）。
生成 4 trace 激励到 results/t5_rtl/<trace>/。自有代码。
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HW = ROOT.parents[0]
TRACES = sorted((HW / 'bn_state').glob('trace_*.npz'))
G_PER_TRACE = 20000
SEED = 20260914


def to_signed(v, bits):
    out = np.asarray(v, dtype=np.int64)
    assert np.all(out >= -(1 << (bits - 1))) and np.all(out <= (1 << (bits - 1)) - 1), \
        'overflow %d bits' % bits
    return out


def one_trace(trace, outdir):
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

    # 终态 tau（部署态静态常数），按 h 分块控内存
    tau = np.zeros((T, H))
    for lo in range(0, H, 32):
        hi = min(H, lo + 32)
        Y = (S @ W[lo:hi].T).reshape(T, P, hi - lo)
        mu = Y.mean((0, 1))
        var = ((Y - mu) ** 2).mean((0, 1))
        tau[:, lo:hi] = (mu * R + np.sqrt(var + 1e-5) / gamma[lo:hi]
                         * (theta + center - bias - beta[lo:hi] * R)) * direction[None, lo:hi]
        del Y

    A_q = to_signed(np.rint(A.astype(np.float64) * 4096), 16)      # f12
    P_t = A_q.clip(min=0).sum(1)
    N_t = A_q.clip(max=0).sum(1)
    tau_q = np.rint(tau * (1 << 14)).astype(np.int64)              # f14
    # 方向折叠（审计修正 2026-09-15）：tau 已乘 direction（Vdir=U·D 侧），
    # 而 RTL 的 Vtop 是 U 侧（未乘 direction）。D=−1 时正确判决 = (U·D ≥ tau)
    # = (U ≤ tau_base)，tau_base = −tau_q，整数域取 thr_U = −(tau_q<<12)+1，
    # 判决 = ~(U ≥ thr_U)。D=+1 时 thr_U = tau_q<<12，判决 = (U ≥ thr_U)。
    thr = np.where(direction[None, :] < 0,
                   -(tau_q << 12) + 1, tau_q << 12)
    thr = to_signed(thr, 48)                                        # DUT 48bit 合同
    Dflag = (direction > 0).astype(np.int64)                       # (H,) 1: 同向

    # 采样组 (p,h)
    rng = np.random.default_rng(SEED)
    ps = rng.integers(0, P, G_PER_TRACE)
    hs = rng.integers(0, H, G_PER_TRACE)
    rows = (ps[None, :] + (np.arange(T)[:, None] * P)).ravel()     # T-major 展平
    Ssub = S.reshape(T, P, C)[:, ps, :]                            # (T,G,C)
    Wsub = W[hs]                                                   # (G,C)
    Ysub = np.einsum('tgc,gc->tg', Ssub, Wsub)                     # (T,G)
    sat = int((np.abs(Ysub) >= 512.0).sum())
    Yq = np.clip(np.rint(Ysub * (1 << 14)), -(1 << 23), (1 << 23) - 1).astype(np.int64)  # (T,G)
    Yq = Yq.T                                                       # (G,T) -> Y[s]

    # 全深度整数判决：V[g,t] = Σ_s A_q[t,s]·Yq[g,s]
    Vfull = np.einsum('gs,ts->gt', Yq, A_q)
    thr_g = thr[:, hs].T                                            # (G,T)
    D_g = np.broadcast_to(Dflag[hs][:, None], (G_PER_TRACE, T))     # (G,T)
    raw = (Vfull >= thr_g)
    dec_full = np.where(D_g > 0, raw, ~raw)

    # 位平面证书模型
    j_first = np.full((G_PER_TRACE, T), -1, np.int8)                # 每单元首锁平面
    dec_raw = np.zeros((G_PER_TRACE, T), bool)                      # D 折入 thr 的原始判决
    frozen = np.zeros((G_PER_TRACE, T), bool)
    for j in range(23, -1, -1):
        Ytop = Yq >> j                                              # 算术右移
        Vtop_new = np.einsum('gs,ts->gt', Ytop, A_q)
        m = j
        Vmin = (Vtop_new << m) + N_t[None, :] * ((1 << m) - 1)
        Vmax = (Vtop_new << m) + P_t[None, :] * ((1 << m) - 1)
        lock = (Vmin >= thr_g) | (Vmax < thr_g)
        newly = lock & ~frozen
        j_first[newly] = j
        dec_raw[newly] = Vmin[newly] >= thr_g[newly]
        frozen |= lock
    assert frozen.all()
    dec_cert_final = np.where(D_g > 0, dec_raw, ~dec_raw)
    assert np.array_equal(dec_cert_final, dec_full), 'cert vs full mismatch'

    j_star = j_first.min(1)                                         # 组终止=最慢单元首锁平面
    per_dec_supply = (24 - j_first).mean() / 24.0                   # 逐判决（各自终止）
    # 组级块浮点：10 词共享指数（组内最大 |Y| 的 MSB），供数从组 MSB 起算
    msb_g = np.array([int(np.abs(Yq[g]).max()).bit_length() for g in range(G_PER_TRACE)])
    planes_bf = np.maximum(msb_g - j_star, 0)                       # 数据平面供数数
    # 拍数口径（2026-09-15 修正，采纳 Codex cert_transport 的 bit23 冗余发现）：
    # signed24 的 bit23 即符号位，FX 首数据平面与 sop 符号字冗余，不必重发——
    # 诚实 FX 基线 = 1 头拍 + 23 数据平面 = 24 拍/组（原 25 双计了一拍）；
    # FX+证书 = 1 + max(23−j*, 1)；BF+证书 = 1 + max(planes, 1)（不变，BF 头已含指数）。
    cyc_base_fx = 24
    cyc_cert_fx = 1 + np.maximum(23 - j_star, 1)
    cyc_cert_bf = 1 + np.maximum(planes_bf, 1)
    group_supply = float(cyc_cert_fx.mean() / cyc_base_fx)
    group_bf_cycles = float(cyc_cert_bf.mean() / cyc_base_fx)

    # ---- 写 RTL 激励（readmemh 元素位宽对齐：a=16bit, pn=48bit, tau=64bit） ----
    outdir.mkdir(parents=True, exist_ok=True)
    wlines = lambda arr, bits: '\n'.join(
        format(int(x) & ((1 << bits) - 1), '0%dx' % (bits // 4)) for x in arr.ravel()) + '\n'
    (outdir / 'a.hex').write_text(wlines(A_q, 16))
    (outdir / 'pn.hex').write_text(wlines(np.stack([P_t, N_t], 1).ravel(), 48))
    for t in range(T):
        word = (Dflag << 63) | (thr[t] & ((1 << 48) - 1))
        (outdir / f'tau_t{t}.hex').write_text(wlines(word, 64))
    lines = []
    for g in range(G_PER_TRACE):
        signw = 0
        for s in range(T):
            if int(Yq[g, s]) < 0:
                signw |= 1 << s
        lines.append('G %d %03x' % (hs[g], signw))
        for j in range(22, -1, -1):                    # bit23=符号字，不重发
            plane = 0
            for s in range(T):
                plane |= int((int(Yq[g, s]) >> j) & 1) << s
            lines.append('B %03x' % plane)
    (outdir / 'stim_fx.txt').write_text('\n'.join(lines) + '\n')
    # 块浮点激励：头部带符号字+指数，数据平面从 e-1 往下（e=组内最大|Y|的 bit_length）
    lines = []
    for g in range(G_PER_TRACE):
        e = int(msb_g[g])
        signw = 0
        for s in range(T):
            if int(Yq[g, s]) < 0:
                signw |= 1 << s
        lines.append('G %d %03x %d' % (hs[g], signw, e))
        for j in range(e - 1, -1, -1):
            plane = 0
            for s in range(T):
                plane |= int((int(Yq[g, s]) >> j) & 1) << s
            lines.append('B %03x' % plane)
    (outdir / 'stim_bf.txt').write_text('\n'.join(lines) + '\n')
    np.savez(outdir / 'expected.npz', dec=dec_full.astype(np.uint8),
             j_star=j_star.astype(np.int16),
             cyc_fx=cyc_cert_fx.astype(np.int32),
             cyc_bf=cyc_cert_bf.astype(np.int32))

    return {
        'trace': str(trace.relative_to(HW)),
        'groups': G_PER_TRACE, 'H': H, 'P': P,
        'y_saturation_count': sat,
        'cert_vs_full_mismatches': 0,
        'per_decision_oracle_supply': float(per_dec_supply),
        'group_fx_cert_cycle_ratio': group_supply,
        'group_bf_cert_cycle_ratio': group_bf_cycles,
        'bf_planes_mean': float(planes_bf.mean()),
        'bf_zero_data_plane_groups': int((planes_bf == 0).sum()),
        'msb_g_mean': float(msb_g.mean()),
        'j_star_mean': float(j_star.mean()),
        'j_star_hist': {str(int(j)): int((j_star == j).sum()) for j in np.unique(j_star)},
        'j_first_mean': float(j_first.mean()),
    }


def main():
    results = []
    for tr in TRACES:
        name = tr.stem.replace('trace_', '')
        r = one_trace(tr, ROOT / 'results' / 't5_rtl' / name)
        results.append(r)
        print('%-28s per_dec=%.4f fx_cert=%.4f bf_cert=%.4f planes=%.2f' %
              (r['trace'], r['per_decision_oracle_supply'],
               r['group_fx_cert_cycle_ratio'], r['group_bf_cert_cycle_ratio'],
               r['bf_planes_mean']))
    out = {
        'traces': results,
        'note': ('整数位平面证书模型（thr U侧方向折入、区间 [Vmin,Vmax] 判定、锁定单调冻结）。'
                 '拍数口径（0915修正，bit23与符号字冗余不重发）：FX基线=1+23=24拍/组；'
                 'FX+证书=1+max(23−j*,1)；BF+证书=1+max(planes,1)。'
                 '重要修正：T1b 的 9.24% 是相对精度位数记账（逐词归一化浮点供数才成立）；'
                 '定点 MSB 供数的真实组级供数为 60-62%（≈静态 per-lane 57.9% 量级，无动态增益）；'
                 '组级块浮点（共享指数）+证书把供数位降到 ~3.2 位/组（拍比 ~17.5%）。'
                 'RTL 激励（FX 与 BF 两套）已生成。'),
    }
    (ROOT / 'results' / 't5_c1_cert_model.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
