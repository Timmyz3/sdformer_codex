"""T3（C2 卡照抄+实测）：真实 anchor 占比 + R-Sparse 式幅值两路代理。

A 照抄对象：R-Sparse (ICLR 2025) §3.2–3.5——大输入分量走原 W 精细路、小分量走
低秩近似路。本地连续量只有内部消费者（PSN/残差），本脚本：
  1) 实测 PED 捕获里的 anchor_mask 占比（替换 kill C 的 25% 假设参数）；
  2) 在真实连续内部张量上原样套用幅值分流 + 低秩近似两路，测重构误差，
     对照同预算纯低秩（无分流）。
静态数值代理，不是 RTL 周期；不训练。自有代码。
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HW = ROOT.parents[0]
PED = HW / 'algorithm/patch_probe/residual_consumer_probe/capture_train4_ped'


def anchor_stats():
    rows = []
    for f in sorted(PED.glob('*.npz')):
        z = np.load(f)
        if 'anchor_mask' not in z.files or 'frame_name' not in z.files:
            continue
        m = z['anchor_mask']
        rows.append({'file': f.name, 'anchor_frac': float(np.asarray(m, bool).mean())})
    return rows


def rsparse_proxy(tensor, keep_fracs, ranks):
    """幅值分流：top-p 幅值位置走精确路，其余位置走 rank-r 近似路。

    行=空间位置（T×P），特征=96 通道。返回逐 (keep, rank) 的重构误差。
    对照：纯低秩（keep=0）同 rank。
    """
    T, C, H, W = tensor.shape
    # (T,C,H,W) -> (T*H*W, C)：每行=某时刻某位置的全部通道
    X = tensor.transpose(0, 2, 3, 1).reshape(-1, C).astype(np.float64)
    n, d = X.shape
    out = []
    for keep in keep_fracs:
        mag = np.abs(X).sum(1)  # 每行幅值（行=空间位置的所有特征）
        thr = np.quantile(mag, 1 - keep) if keep > 0 else np.inf
        big = mag >= thr if keep > 0 else np.zeros(n, bool)
        small = ~big
        for r in ranks:
            # 小分量路：只对小行做 rank-r SVD 近似
            Xs = X[small]
            if Xs.shape[0] <= r or r >= min(Xs.shape):
                out.append({'keep_frac': keep, 'rank': r, 'rel_err': None})
                continue
            U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
            approx_small = (U[:, :r] * S[:r]) @ Vt[:r]
            # 精确路直通
            rec = np.empty_like(X)
            rec[big] = X[big]
            rec[small] = approx_small
            rel_err = np.linalg.norm(rec - X) / np.linalg.norm(X)
            small_share = np.linalg.norm(Xs) / np.linalg.norm(X)
            out.append({
                'keep_frac': keep, 'rank': int(r),
                'rel_err_fro': float(rel_err),
                'small_path_norm_share': float(small_share),
            })
    return out


def main():
    anchors = anchor_stats()
    files = [f for f in sorted(PED.glob('*.npz'))
             if 'frame_name' in np.load(f, allow_pickle=False).files]
    z = np.load(files[0])
    # 连续内部张量：r1_sn1_output（PSN 输出）与 proj_input（投影输入）
    results = {}
    for key in ('r1_sn1_output', 'proj_input'):
        t = z[key]
        results[key] = {
            'shape': list(t.shape),
            'grid': rsparse_proxy(t, keep_fracs=(0.0, 0.05, 0.1, 0.25, 0.5),
                                  ranks=(4, 8, 16)),
        }
    out = {
        'anchor': anchors,
        'anchor_mean': float(np.mean([r['anchor_frac'] for r in anchors])),
        'rsparse_proxy': results,
        'note': ('幅值分流按空间位置行幅值；rel_err 为 Frobenius 相对误差。'
                 '纯低秩对照=keep_frac 0 行。静态代理，无训练、无RTL。'),
    }
    (ROOT / 'results' / 't3_anchor_rsparse.json').write_text(json.dumps(out, indent=2) + '\n')

    am = out['anchor_mean']
    lines = [
        '# T3：anchor 实测 + R-Sparse 幅值两路代理（C2 卡）',
        '',
        f"数据源：`{PED.relative_to(HW)}`（4 帧 PED 捕获）。",
        '',
        '## 1. anchor 占比实测（替换 kill C 假设参数）',
        '',
        '| 文件 | anchor 占比 |',
        '|---|---:|',
    ]
    for r in anchors:
        lines.append(f"| {r['file']} | {r['anchor_frac']:.4%} |")
    lines += [
        '',
        f"**实测均值：{am:.4%}**（kill C 假设 25%）。",
        '',
        '## 2. R-Sparse 幅值分流代理（r1_sn1_output / proj_input）',
        '',
        'keep=0 行即纯低秩对照（同 rank，无分流）。',
        '',
    ]
    for key, res in results.items():
        lines.append(f"### {key} {res['shape']}")
        lines.append('')
        lines.append('| keep 精确路 | rank | 相对误差 | 小路范数占比 |')
        lines.append('|---:|---:|---:|---:|')
        for g in res['grid']:
            if g.get('rel_err_fro') is None:
                continue
            lines.append(f"| {g['keep_frac']:.0%} | {g['rank']} | "
                         f"{g['rel_err_fro']:.4f} | {g['small_path_norm_share']:.4f} |")
        lines.append('')
    (ROOT / 'results' / 'T3_REPORT.md').write_text('\n'.join(lines) + '\n')
    print('T3 done: anchor_mean=%.4f' % am)
    for key, res in results.items():
        base = [g for g in res['grid'] if g['keep_frac'] == 0.0 and g['rank'] == 8
                and 'rel_err_fro' in g]
        split = [g for g in res['grid'] if g['keep_frac'] == 0.1 and g['rank'] == 8
                 and 'rel_err_fro' in g]
        if base and split:
            print(key, 'pure_rank8 err=%.4f | 10%%split+rank8 err=%.4f'
                  % (base[0]['rel_err_fro'], split[0]['rel_err_fro']))


if __name__ == '__main__':
    main()
