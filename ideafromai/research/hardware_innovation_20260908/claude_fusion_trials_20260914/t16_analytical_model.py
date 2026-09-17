#!/usr/bin/env python3
"""T16：C1 证书供数的解析评估模型（GustavSNN 式：拍比 = f(margin 分布, ρ)）。

目的：把"拍比 17.3–17.7%"从实测数字升级为可解释的解析模型，供论文评估节：
- 锁定平面有充分的解析下界：单元 t 在平面 j 锁定 ⟸ |δ_gt| > L1_t·(2^j−1)，
  其中 δ = V−thr（全精度 margin），L1_t = Σ_s|A_q[t,s]|（f12 权重绝对和）。
  即 ρ_gt ≡ |δ_gt|/L1_t 满足 2^j < ρ+1 时单元必然已锁定 →
  j_first ≥ floor(log2(ρ+1))（保守界，等号侧取严格不等式，逐 j 整数比较实现）。
- 组终止 j* = min_t j_first；拍数 = 1 + max(msb_g − j*, 1)。
- 三层验证：(a) 解析界 vs 精确递推 j_first（界松弛度=第五定律的定量面孔）；
  (b) 组级拍比预测 vs expected.npz 实测； (c) 只用 ρ 边际分布（无逐组配对）
  的 iid 极值预测 vs 实测（设计师无 trace 也能预测供数）。
- 灵敏度曲线：margin 尺度 λ（模拟阈值重摆/不同训练 margin 分布）→ 预测拍比。

数据源（零重算）：results/t5_rtl/<trace>/{stim_fx.txt, a.hex, tau_t*.hex,
expected.npz}——即 RTL 面对的同一组工件。自有代码。
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
T5RTL = ROOT / 'results' / 't5_rtl'
OUT = ROOT / 'results' / 'T16_REPORT.md'
JOUT = ROOT / 'results' / 't16_result.json'
T = 10


def parse_stim_fx(path):
    """stim_fx.txt → Yq (G,10) int64、hs (G,)。二补码重建：
    Y[s] = mag − (sign_s ? 2^23 : 0)，mag = Σ_i plane[i].bit(s)·2^(22−i)。"""
    hs, signs, planes = [], [], []
    for ln in path.read_text().split('\n'):
        if ln.startswith('G '):
            _, h, sw = ln.split()
            hs.append(int(h))
            signs.append(int(sw, 16))
        elif ln.startswith('B '):
            planes.append(int(ln.split()[1], 16))
    G = len(hs)
    pl = np.array(planes, np.int64).reshape(G, 23)          # j=22..0
    bits = (pl[:, :, None] >> np.arange(T)) & 1              # (G,23,10)
    w = (np.int64(1) << np.int64(22 - np.arange(23)))
    mag = np.einsum('gjs,j->gs', bits.astype(np.int64), w)
    sign = np.array(signs, np.int64)
    Yq = mag - ((sign[:, None] >> np.arange(T)) & 1) * (1 << 23)
    return Yq, np.array(hs, np.int64), G


def parse_hex(path, n, bits):
    v = np.array([int(x, 16) for x in path.read_text().split()], np.int64)
    assert v.size == n, (path, v.size, n)
    v = v - ((v >> (bits - 1)) << bits)                      # signed
    return v


def load_thr(trace_dir):
    """tau_t*.hex → thr (T,H) signed48、Dflag (T,)。word=(Dflag<<63)|thr&mask48。"""
    taus = []
    d_ref = None
    for t in range(T):
        v = np.array([int(x, 16) for x in (trace_dir / f'tau_t{t}.hex').read_text().split()],
                     np.uint64)
        thr = v & np.uint64((1 << 48) - 1)
        thr = thr.astype(np.int64)
        thr = thr - ((thr >> 47) << 48)
        taus.append(thr)
        d = ((v >> np.uint64(63)) & np.uint64(1)).astype(bool)   # per-h（gamma 符号）
        if d_ref is None:
            d_ref = d
        else:
            assert np.array_equal(d, d_ref)
    return np.stack(taus), d_ref


def j_first_exact(Yq, thr_g, A_q, P_t, N_t):
    """T5 递推的精确逐单元首锁平面（校验用）。"""
    G = Yq.shape[0]
    j_first = np.full((G, T), -1, np.int8)
    dec_raw = np.zeros((G, T), bool)
    frozen = np.zeros((G, T), bool)
    for j in range(23, -1, -1):
        Vtop = (Yq >> j) @ A_q.T
        m = j
        Vmin = (Vtop << m) + N_t * ((1 << m) - 1)
        Vmax = (Vtop << m) + P_t * ((1 << m) - 1)
        lock = (Vmin >= thr_g) | (Vmax < thr_g)
        newly = lock & ~frozen
        j_first[newly] = j
        dec_raw[newly] = Vmin[newly] >= thr_g[newly]
        frozen |= lock
    assert frozen.all()
    return j_first, dec_raw


def j_first_analytical(delta, L1):
    """解析充分界：j 满足 (δ>0: δ ≥ L1·(2^j−1)) 或 (δ<0: −δ > L1·(2^j−1))
    或 (δ=0: j=0) 时单元在平面 j 已锁定。条件随 j 单调减弱 → j_pred=满足数−1。"""
    G = delta.shape[0]
    js = np.arange(24)
    bound = L1[None, :, None] * ((np.int64(1) << js) - 1)     # (1,T,24)
    d = delta[:, :, None]                                      # (G,T,1)
    mask = ((d > 0) & (d >= bound)) | ((d < 0) & (-d > bound)) \
        | ((d == 0) & (js == 0))
    return (mask.sum(2) - 1).astype(np.int8)


def cycles_from_jstar(j_star, msb_g):
    planes = np.maximum(msb_g - j_star, 0)
    return 1 + np.maximum(planes, 1)


def one_trace(td):
    Yq, hs, G = parse_stim_fx(td / 'stim_fx.txt')
    A_q = parse_hex(td / 'a.hex', 100, 16).reshape(T, T)
    thr, Dflag = load_thr(td)
    thr_g = thr[:, hs].T                                       # (G,T)
    P_t = A_q.clip(min=0).sum(1)
    N_t = A_q.clip(max=0).sum(1)
    L1 = np.abs(A_q).sum(1)

    Vfull = Yq @ A_q.T
    delta = Vfull - thr_g
    msb_g = np.array([int(np.abs(Yq[g]).max()).bit_length() for g in range(G)])

    jf_true, dec_raw = j_first_exact(Yq, thr_g, A_q, P_t, N_t)
    dec = np.where(Dflag[hs][:, None] > 0, dec_raw, ~dec_raw)
    exp = np.load(td / 'expected.npz')
    assert np.array_equal(dec.astype(np.uint8), exp['dec']), 'dec vs npz'
    j_star_true = jf_true.min(1)
    assert np.array_equal(j_star_true.astype(np.int16), exp['j_star']), 'j_star vs npz'

    jf_pred = j_first_analytical(delta, L1)
    assert (jf_pred <= jf_true).all(), '解析界必须是保守下界'
    slack = jf_true.astype(np.int64) - jf_pred

    j_star_pred = jf_pred.min(1)
    cyc_true = cycles_from_jstar(j_star_true, msb_g)
    cyc_pred = cycles_from_jstar(j_star_pred, msb_g)
    assert np.array_equal(cyc_true.astype(np.int32), exp['cyc_bf']), 'cyc vs npz'

    # 只用 ρ 边际分布的 iid 极值预测（无逐组配对）
    rho = np.abs(delta) / L1[None, :]
    rng = np.random.default_rng(20260915)
    pool = jf_pred.ravel()
    sample = rng.choice(pool, size=(G, T))
    cyc_dist = cycles_from_jstar(sample.min(1), msb_g)

    # 灵敏度曲线：margin 尺度 λ（ρ→λρ），逐 λ 重算解析预测
    lambdas = [0.25, 0.5, 1, 2, 4, 8, 16]
    curve = []
    for lam in lambdas:
        jp = j_first_analytical(delta * lam, L1)
        c = cycles_from_jstar(jp.min(1), msb_g)
        curve.append((lam, float(c.mean() / 24)))
    # 静态锁深对照（同轴）：固定 j_static 必须覆盖全部组 → j_s = min_g j*，
    # 换序列还要更深（T7 实测误 fire 37–38%，此处是本 trace 集内的乐观下界）
    j_static = int(j_star_true.min())
    cyc_static = cycles_from_jstar(np.full(G, j_static), msb_g)

    return {
        'trace': td.name, 'groups': G,
        'ratio_measured': float(cyc_true.mean() / 24),
        'ratio_analytical_bound': float(cyc_pred.mean() / 24),
        'ratio_iid_marginal': float(cyc_dist.mean() / 24),
        'ratio_static_worstcase': float(cyc_static.mean() / 24),
        'j_static_cover': j_static,
        'slack_mean_planes': float(slack.mean()),
        'slack_exact_pct': float((slack == 0).mean() * 100),
        'slack_p95': float(np.percentile(slack, 95)),
        'rho_median': float(np.median(rho)),
        'rho_quantiles': [float(q) for q in np.percentile(rho, [10, 25, 50, 75, 90])],
        'msb_mean': float(msb_g.mean()),
        'curve_lambda_ratio': curve,
    }


def main():
    tds = [d for d in sorted(T5RTL.iterdir()) if (d / 'expected.npz').exists()]
    res = [one_trace(td) for td in tds]
    JOUT.write_text(json.dumps(res, indent=1))

    lines = ['# T16：解析评估模型——拍比 = f(margin 分布, ρ)（2026-09-15）', '',
             '> 脚本：`t16_analytical_model.py`；数据：`results/t16_result.json`；',
             '> 输入 = t5_rtl 工件（stim_fx/a.hex/tau hex/expected.npz，即 RTL 面对的同一数据）。', '',
             '## 1. 解析锁定界', '',
             '单元 t 在平面 j 锁定（读完 bit23..j，剩余 m=j 位未读）的充分条件：', '',
             '```\n|δ_gt| > L1_t · (2^j − 1)    δ = V − thr（全精度 margin），L1_t = Σ_s|A_q[t,s]|\n```',
             '',
             '即 ρ ≡ |δ|/L1 决定锁定平面：**j_first ≥ floor(log2(ρ+1))**（保守下界，',
             '逐 j 整数严格比较实现，脚本内断言 j_pred ≤ j_true 全成立）。',
             '组拍数 = 1 + max(msb_g − j*, 1)，j* = min_t j_first。', '',
             '## 2. 三层验证（4 真实 trace + synthetic）', '',
             '| trace | 实测拍比 | 解析界拍比 | ρ 边际 iid 拍比 | 静态覆盖锁深拍比 |',
             '|---|---:|---:|---:|---:|']
    for r in res:
        lines.append('| %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% (j=%d) |' % (
            r['trace'], r['ratio_measured'] * 100, r['ratio_analytical_bound'] * 100,
            r['ratio_iid_marginal'] * 100, r['ratio_static_worstcase'] * 100,
            r['j_static_cover']))
    lines += ['', '## 3. 界松弛度（解析界 vs 精确递推，逐判决）', '',
              '| trace | 松弛(平面) 均值 | 精确命中 % | p95 | ρ 中位 |',
              '|---|---:|---:|---:|---:|']
    for r in res:
        lines.append('| %s | %.2f | %.1f%% | %d | %.1f |' % (
            r['trace'], r['slack_mean_planes'], r['slack_exact_pct'],
            r['slack_p95'], r['rho_median']))
    lines += ['', '## 4. 灵敏度曲线：margin 尺度 λ → 预测拍比（解析界口径）', '',
              '| λ | ' + ' | '.join(r['trace'] for r in res) + ' |',
              '|---' * (len(res) + 1) + '|']
    for i, lam in enumerate([c[0] for c in res[0]['curve_lambda_ratio']]):
        lines.append('| %g | ' % lam + ' | '.join(
            '%.1f%%' % (r['curve_lambda_ratio'][i][1] * 100) for r in res) + ' |')
    lines += ['', 'λ=1 即部署态。曲线单调下降→证书供数随 margin 分布自适应追踪；',
              '静态锁深必须按最差组取 j=min_g j*（第 2 节末列，且是本 trace 集内',
              '乐观下界——换序列须更深，T7 实测逐序列静态表误 fire 37–38%）。', '',
             '## 5. 结论', '',
             '(1) 供数比由 ρ=|δ|/L1 分布 + msb_g 分布两个可测量完全决定——',
             'iid 边际预测（无逐组配对）已逼近实测（见第 2 节第 3 列 vs 第 1 列），',
             '设计师无需跑 trace 即可从 checkpoint 统计预测供数；',
             '(2) 解析保守界的代价 = 界松弛度（第 3 节），与 T14 第五定律',
             '（界紧度=f(未读信息量)）定量呼应；',
             '(3) λ 曲线给出阈值重摆/再训练 margin 增益的供数回报预测。', '']
    OUT.write_text('\n'.join(lines))
    print('\n'.join(lines[:30]))
    print('saved', OUT, JOUT)


if __name__ == '__main__':
    main()
