"""T1（C1 卡 A' 试验）：门消费者按 lane 分级的位深需求实测。

A 照抄对象：本地 K=4 条件完成框架（failure 判定）+ BISMO 位平面供数的位组织；
X 候选：门消费者（T10 判决）按 lane 联合位深证书提前结束供数，I24 消费者全深度。

本试验自己重算 V/tau，然后：
  - 逐通道 b*_h = 最小分数位深，使该通道全部 (t,p) 判决在贡献截断到 b 位后不变
    （Y 截断 = RNE 到 b 分数位，重新混合 A，与全精度判决比对，样本内精确）；
  - lane（8 通道）b* = max 成员；
  - 供数节省代理 = 1 - mean(b*/24)（门路状态/乘积宽度相对 24bit I24 全深度）。
静态数值试验（样本内精确），非 RTL。自有代码。
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HW = ROOT.parents[0]
TRACE = HW / 'bn_state/trace_s0_stage0.npz'

LANE = 8
B_CANDS = list(range(1, 15))  # 1..14 分数位（f14 状态格式上限）


def main():
    z = np.load(TRACE)
    S = np.unpackbits(z['source_packed'], axis=1, bitorder='little')[:, :96].astype(np.float64)
    N, C = S.shape
    T, P, H = 10, N // 10, 384
    W = z['W'].astype(np.float64)
    A = z['A']
    gamma, beta = z['gamma'], z['beta']
    bias, center = z['bias'], z['center']
    theta = 1.0
    R = A.sum(1).reshape(T, 1)
    b_star = np.full(H, -1, int)   # 每通道最小位深（-1=14 位内无解）
    flip_counts = np.zeros((len(B_CANDS), H), int)
    for lo in range(0, H, 96):
        hi = min(H, lo + 96)
        Y = (S @ W[lo:hi].T).reshape(T, P, hi - lo)  # theta_source=1
        mu = Y.mean((0, 1))
        var = ((Y - mu) ** 2).mean((0, 1))
        direction = np.sign(gamma[lo:hi])
        tau = (mu * R + np.sqrt(var + 1e-5) / gamma[lo:hi]
               * (theta + center - bias - beta[lo:hi] * R)) * direction
        truth = (np.einsum('ts,sph->tph', A, Y, optimize=True) * direction >= tau[:, None, :])
        for bi, b in enumerate(B_CANDS):
            Yq = np.rint(Y * (1 << b)) / (1 << b)
            Vq = np.einsum('ts,sph->tph', A, Yq, optimize=True) * direction
            flips = (Vq >= tau[:, None, :]) != truth
            n_flip = flips.sum((0, 1))
            flip_counts[bi, lo:hi] = n_flip
            solved = (b_star[lo:hi] < 0) & (n_flip == 0)
            b_star[lo:hi][solved] = b
        del Y
    # lane 位深 = lane 内最大 b*
    lane_b = b_star.reshape(H // LANE, LANE).max(1)
    supply_full = 24.0
    ch_frac = np.where(b_star > 0, b_star, 24) / supply_full
    lane_frac = np.where(lane_b > 0, lane_b, 24) / supply_full
    out = {
        'trace': str(TRACE.relative_to(HW)),
        'lane_width': LANE,
        'b_candidates': B_CANDS,
        'channel_b_star_hist': {str(int(b)): int((b_star == b).sum()) for b in set(b_star.tolist())},
        'channel_unsolved': int((b_star < 0).sum()),
        'lane_b_hist': {str(int(b)): int((lane_b == b).sum()) for b in set(lane_b.tolist())},
        'lane_unsolved': int((lane_b < 0).sum()),
        'gate_supply_frac_channel_granularity': float(ch_frac.mean()),
        'gate_supply_frac_lane8_granularity': float(lane_frac.mean()),
        'flip_counts_by_b': {
            str(b): int(flip_counts[bi].sum()) for bi, b in enumerate(B_CANDS)},
        'note': ('样本内精确：b* 为该 trace 上判决零翻转的最小分数位深；'
                 '外推到其他源不保证。供数节省=门路宽度相对 24bit 全深度的比例，'
                 '未计证书判定电路与元数据费用。'),
    }
    (ROOT / 'results' / 't1_lane_bitdepth.json').write_text(json.dumps(out, indent=2) + '\n')

    lines = [
        '# T1：门消费者 lane 分级位深实测（C1 卡 A-prime 试验）',
        '',
        f"数据源：`{TRACE.relative_to(HW)}`；lane 宽 {LANE}；候选位深 1–14（f14 状态上限）。",
        '',
        f"- 通道级 b* 直方图：{out['channel_b_star_hist']}（无解 {out['channel_unsolved']}）",
        f"- lane8 级 b* 直方图：{out['lane_b_hist']}（无解 {out['lane_unsolved']}）",
        f"- 门路供数比例（通道粒度）：**{out['gate_supply_frac_channel_granularity']:.4f}**",
        f"- 门路供数比例（lane8 粒度）：**{out['gate_supply_frac_lane8_granularity']:.4f}**",
        '',
        '## 各位深下全通道判决翻转总数（样本内）',
        '',
        '| b | 翻转数 |',
        '|---:|---:|',
    ]
    for b in B_CANDS:
        lines.append(f"| {b} | {out['flip_counts_by_b'][str(b)]} |")
    lines += [
        '',
        '## 判读（C1 杀门2）',
        '',
        f"- lane8 供数比例 {out['gate_supply_frac_lane8_granularity']:.2%}，"
        f"相对全深度节省 {1-out['gate_supply_frac_lane8_granularity']:.2%}；",
        '- 该节省只作用于门消费者路径（I24 仍全深度）；门路占完整链的供数份额',
        '  须在同端口 RTL 中确认后才能折算整链收益；',
        '- 样本内精确，换源复核与元数据计费是下一步。',
        '',
    ]
    g = out['gate_supply_frac_lane8_granularity']
    lines.append('**过静态门**：lane8 门路供数 ≤ 50%，C1 晋级 RTL。' if g <= 0.5
                 else '**不过门**：lane8 门路供数 > 50%，C1 位深分级收益不足。')
    (ROOT / 'results' / 'T1_REPORT.md').write_text('\n'.join(lines) + '\n')
    print('T1 done: channel_frac=%.4f lane8_frac=%.4f unsolved=%d'
          % (out['gate_supply_frac_channel_granularity'],
             out['gate_supply_frac_lane8_granularity'], out['lane_unsolved']))


if __name__ == '__main__':
    main()
