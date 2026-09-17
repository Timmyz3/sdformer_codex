"""T2（C4 卡照抄+统计）：运动对齐跨窗脉冲 delta 统计。

A 照抄对象：MotionDeltaCNN 的运动对齐差分传播思想——用光流把前一窗特征
warp 到当前视角，只传残差。本地网络是光流网络，粗层/参考流可作 warp 参考。

本脚本只做统计（不写 RTL）：
  - 未对齐 XOR toggle 率（相邻窗同位置直接异或）
  - 运动对齐 XOR toggle 率（按流场 warp 后异或，整数位移=硬件最近邻采样）
  - 基线：活动率（假设"不变"时的供给义务）
全部使用已有 motion/capture 捕获，不调网络、不训练。自有代码，不复用其他目录脚本。
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HW = ROOT.parents[0]
CAP = HW / 'motion' / 'capture'


def load_frame(stem):
    z = np.load(CAP / f'{stem}_source.npz')
    shape = tuple(int(x) for x in z['source_shape'])
    bits = np.unpackbits(z['source_bits'], bitorder='little')[:np.prod(shape)]
    g = bits.reshape(shape).astype(bool)  # (T,1,C,H,W)
    flow = np.load(CAP / f'{stem}_flow.npz')['flow'][0]  # (2,480,640)
    return g, flow


def warp(g, flow, direction):
    """按流场对 g 的空间维做最近邻 backward warp。direction=+1/-1 两个方向都试。"""
    T, _, C, H, W = g.shape
    # flow (2,480,640) -> g 分辨率 (240,320)，位移同比例缩半
    fl = flow[:, ::2, ::2].astype(np.float32) * 0.5  # (2,H,W)
    u, v = fl[0] * direction, fl[1] * direction
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    ny = np.clip(np.rint(ys + v).astype(int), 0, H - 1)
    nx = np.clip(np.rint(xs + u).astype(int), 0, W - 1)
    return g[..., ny, nx]


def main():
    frames = json.loads((CAP / 'frames.json').read_text())['frames']
    stems = [f['stem'] for f in frames]
    pairs = [(a, b) for a, b in zip(stems, stems[1:])
             if (CAP / f'{a}_source.npz').exists() and (CAP / f'{b}_source.npz').exists()
             and (CAP / f'{a}_flow.npz').exists() and (CAP / f'{b}_flow.npz').exists()]
    rows = []
    for a, b in pairs:
        g0, flow0 = load_frame(a)
        g1, flow1 = load_frame(b)
        act0, act1 = g0.mean(), g1.mean()
        unaligned = (g1 ^ g0).mean()
        # 两个 warp 方向与两份流场都试，报最好与全部（不做方向挑选的隐藏调参）
        cands = {}
        for fname, flow in (('flow0', flow0), ('flow1', flow1)):
            for d in (+1, -1):
                gw = warp(g1 if d == +1 else g0, flow, d)
                other = g0 if d == +1 else g1
                cands[f'{fname}_d{d:+d}'] = (gw ^ other).mean()
        best_name = min(cands, key=cands.get)
        rows.append({
            'pair': f'{a}->{b}',
            'activity_g0': float(act0), 'activity_g1': float(act1),
            'toggle_unaligned': float(unaligned),
            'toggle_aligned_all': {k: float(v) for k, v in cands.items()},
            'toggle_aligned_best': float(cands[best_name]),
            'best_warp': best_name,
        })
        print(rows[-1]['pair'], 'unaligned=%.4f best_aligned=%.4f (%s)'
              % (unaligned, cands[best_name], best_name))
    agg = {
        'mean_toggle_unaligned': float(np.mean([r['toggle_unaligned'] for r in rows])),
        'mean_toggle_aligned_best': float(np.mean([r['toggle_aligned_best'] for r in rows])),
        'mean_activity': float(np.mean([r['activity_g1'] for r in rows])),
        'pairs': rows,
        'note': ('整数位移最近邻 warp=硬件可实现采样；toggle 率=需新供给的位比例。'
                 '静态统计，非RTL周期；未计 warp 寻址与 misalignment 回退费用。'),
    }
    (ROOT / 'results' / 't2_motion_toggle.json').write_text(json.dumps(agg, indent=2) + '\n')

    mu, ma = agg['mean_toggle_unaligned'], agg['mean_toggle_aligned_best']
    lines = [
        '# T2：运动对齐跨窗脉冲 delta 统计（C4 卡）',
        '',
        f"数据源：`motion/capture/` 相邻帧对 {len(rows)} 组；A=MotionDeltaCNN 运动对齐差分传播。",
        '',
        f"- 平均活动率（供给基线）：{agg['mean_activity']:.4%}",
        f"- 未对齐 XOR toggle：**{mu:.4%}**",
        f"- 运动对齐 XOR toggle（最好方向/流场）：**{ma:.4%}**",
        '',
        '## 判读（C4 杀门1：对齐须 ≥2× 下降且绝对值 <50%）',
        '',
        f"- 相对未对齐下降：{(1 - ma / mu):.2%}（要求 ≥50% 即 2× 下降）",
        f"- 绝对 toggle：{ma:.2%}（要求 <50%）",
        f"- 相对活动率基线：对齐后 toggle {'<' if ma < agg['mean_activity'] else '>'} 活动率"
        f"（{ma:.2%} vs {agg['mean_activity']:.2%}）",
        '',
    ]
    if ma < 0.5 and ma < mu / 2:
        lines.append('**过统计门**：对齐 delta 有 ≥2× 稀疏化，C4 保留进入 warp 计费阶段。')
    else:
        lines.append('**不过门**：对齐收益不足 2× 或绝对 toggle 过高，C4 按 25% 净服务门继续评估前先降权。')
    (ROOT / 'results' / 'T2_REPORT.md').write_text('\n'.join(lines) + '\n')
    print('T2 done: unaligned=%.4f aligned_best=%.4f' % (mu, ma))


if __name__ == '__main__':
    main()
