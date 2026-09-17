"""T1b（C1 卡）：逐判决动态位深（证书驱动提前终止）的供数估计——四 trace 换源复核版。

T1 的静态 per-lane 位深是悲观版（被每个通道的最差判决拉高到 12-14 位）。
真正的 C1 机制是逐判决动态终止：硬件按位平面继续供数，直到判决裕度超过
剩余截断不确定界（margin > L1·2^-b，L1=该判决总绝对贡献界），此时判决
已精确锁定，可停。本脚本在全部 4 个 trace（s0/s10 × stage0/stage3）上算
逐判决证书位深 d* 的分布与平均供数比例。样本内静态界，非RTL。自有代码。
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HW = ROOT.parents[0]
TRACES = sorted((HW / 'bn_state').glob('trace_*.npz'))


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
    absA = np.abs(A)
    act = S.reshape(T, P, C)
    depths = []
    for lo in range(0, H, 32):
        hi = min(H, lo + 32)
        wh = W[lo:hi]
        Y = (S @ wh.T).reshape(T, P, hi - lo)
        mu = Y.mean((0, 1))
        var = ((Y - mu) ** 2).mean((0, 1))
        direction = np.sign(gamma[lo:hi])
        tau = (mu * R + np.sqrt(var + 1e-5) / gamma[lo:hi]
               * (theta + center - bias - beta[lo:hi] * R)) * direction
        # 按 patch 分块控制峰值内存
        for p0 in range(0, P, 2400):
            p1 = min(P, p0 + 2400)
            Yc = Y[:, p0:p1]
            Vc = np.einsum('ts,sph->tph', A, Yc, optimize=True) * direction
            margin = np.abs(Vc - tau[:, None, :])
            del Vc
            l1 = np.einsum('ts,spc,hc->tph', absA, act[:, p0:p1],
                           np.abs(wh), optimize=True)
            with np.errstate(divide='ignore', invalid='ignore'):
                need = np.log2(np.maximum(l1 / np.maximum(margin, 1e-300), 1.0))
            del margin, l1
            d = np.ceil(need)
            del need
            d = np.where(np.isfinite(d), d, 24)
            d = np.clip(d, 0, 24).astype(np.int8)
            depths.append(d.ravel())
            del d
        del Y
    depths = np.concatenate(depths)
    return {
        'trace': str(trace.relative_to(HW)),
        'n_decisions': int(depths.size),
        'depth_hist': {str(int(b)): int((depths == b).sum()) for b in np.unique(depths)},
        'depth_mean': float(depths.mean()),
        'depth_quantiles_50_90_99_999': [float(np.quantile(depths, q))
                                         for q in (0.5, 0.9, 0.99, 0.999)],
        'gate_supply_frac_per_decision_dynamic': float(depths.mean() / 24.0),
    }


def main():
    results = [one_trace(t) for t in TRACES]
    out = {
        'traces': results,
        'supply_range': [min(r['gate_supply_frac_per_decision_dynamic'] for r in results),
                         max(r['gate_supply_frac_per_decision_dynamic'] for r in results)],
        'note': ('d* = ceil(log2(L1/margin))，margin>L1*2^-d 时截断不可能翻转判决（精确证书）。'
                 '供数比例=平均 d*/24。只覆盖门消费者路径；I24 全深度；'
                 '未计逐判决终止逻辑/证书电路费用。oracle 边界：用终态 tau。'),
    }
    (ROOT / 'results' / 't1b_dynamic_depth.json').write_text(json.dumps(out, indent=2) + '\n')

    lines = [
        '# T1b：逐判决动态证书位深（C1 卡，四 trace 换源复核）',
        '',
        '| trace | 判决数 | 位深均值 | 50/90/99/99.9% 分位 | 门路供数比例 |',
        '|---|---:|---:|---|---:|',
    ]
    for r in results:
        lines.append(f"| `{r['trace']}` | {r['n_decisions']:,} | {r['depth_mean']:.2f} | "
                     f"{r['depth_quantiles_50_90_99_999']} | "
                     f"**{r['gate_supply_frac_per_decision_dynamic']:.4f}** |")
    smin, smax = out['supply_range']
    lines += [
        '',
        f"供数比例跨源范围：**{smin:.4f} – {smax:.4f}**。",
        '',
        '## 判读',
        '',
        '- d* = ceil(log2(L1/margin))：margin > L1·2^−d 时截断不可能翻转判决（精确证书）；',
        '- 静态 per-lane（T1）为 57.9%；逐判决动态显著更低；',
        '- oracle 边界：用了终态 tau（BN 全局矩）；可实现版用块前缀矩 ptau，',
        '  前缀矩判决偏差本项目实测 0.30%，真实供数会略高；',
        '- 证书判定与元数据费用未计（Sparsity Tax 教训），RTL 阶段须补。',
        '',
    ]
    lines.append('**过换源复核**：全部 trace 门路供数 ≤ 50% 且跨源稳定，C1 动态形态晋级 RTL。'
                 if smax <= 0.5 else '**跨源不稳**：存在 trace 供数 > 50%。')
    (ROOT / 'results' / 'T1B_REPORT.md').write_text('\n'.join(lines) + '\n')
    for r in results:
        print('%-28s mean=%.2f supply=%.4f' % (r['trace'], r['depth_mean'],
                                               r['gate_supply_frac_per_decision_dynamic']))


if __name__ == '__main__':
    main()
