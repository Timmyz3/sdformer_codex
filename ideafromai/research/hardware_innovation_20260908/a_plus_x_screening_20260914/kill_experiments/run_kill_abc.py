"""A+X 筛选杀实验 A/B/C（2026-09-14）。

全部用已有捕获数据，不新开训练、不调网络、不改生产树。
A: lifting 系数 |q|<Q/2 静态证书上限 (constant_compilation_result.json)
B: 门失败率 / SIMD lane 联合接受率 / 判决裕度分布 (bn_state/trace_s0_stage0.npz)
C: 门保持率 x anchor 结构的并集占用代理 (同 trace)
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
HW = ROOT.parents[1]

LIFT = HW / ('algorithm/patch_probe/residual_consumer_probe/projection_chain/'
             'fast_temporal_recovery_lifting40/constant_compilation_result.json')
TRACE = HW / 'bn_state/trace_s0_stage0.npz'


def kill_a():
    d = json.loads(LIFT.read_text())
    Q = 4096
    rows = {}
    for name, m in d['models'].items():
        c = np.asarray(m['coefficients_q12'], dtype=np.int64)
        total = c.size
        stats = {
            'shape': list(c.shape),
            'n': int(total),
            'range': [int(c.min()), int(c.max())],
            'frac_absq_lt_Q_over_2': float(np.mean(np.abs(c) < Q / 2)),
            'frac_absq_lt_Q_over_4': float(np.mean(np.abs(c) < Q / 4)),
            'frac_absq_lt_Q_over_8': float(np.mean(np.abs(c) < Q / 8)),
            'frac_zero': float(np.mean(c == 0)),
            'max_absq': int(np.abs(c).max()),
            'coefficients_clips': m.get('coefficient_clips'),
            'deployed_q12_mismatches': m.get('deployed_q12_mismatches'),
        }
        # 逐系数可跳过增量上限 d_max（r0=0 最宽条件 |q|*d < Q/2）
        nz = np.abs(c)
        nz = nz[nz > 0]
        if nz.size:
            with np.errstate(divide='ignore', invalid='ignore'):
                dmax = (Q / 2 - 1) // nz
            stats['dmax_ge1_frac'] = float(np.mean(dmax >= 1))
            stats['dmax_ge2_frac'] = float(np.mean(dmax >= 2))
            stats['dmax_ge4_frac'] = float(np.mean(dmax >= 4))
        rows[name] = stats
    out = {'Q': Q, 'state_format': d['state_format'],
           'coefficient_format': d['coefficient_format'], 'models': rows}
    (ROOT / 'kill_a_raw.json').write_text(json.dumps(out, indent=2) + '\n')

    lines = [
        '# Kill A：lifting 系数舍入裕度证书静态上限',
        '',
        f"数据源：`{LIFT.relative_to(HW)}`（fast_temporal_recovery_lifting40 常量编译结果）。",
        f"系数格式：{d['coefficient_format']}；状态格式：{d['state_format']}；Q={Q}。",
        '',
        '证书条件：|q|·d < m₀ = Q/2 − |r₀|。静态上限 = |q|<Q/2 的系数比例',
        '（只有这些系数在某个状态下才可能被精确跳过）。',
        '',
        '| 模型 | n | 系数范围 | \\|q\\|<Q/2 | \\|q\\|<Q/4 | \\|q\\|<Q/8 | 零系数 |',
        '|---|---:|---|---:|---:|---:|---:|',
    ]
    for name, s in rows.items():
        lines.append(f"| {name} | {s['n']} | {s['range']} | "
                     f"{s['frac_absq_lt_Q_over_2']:.4f} | {s['frac_absq_lt_Q_over_4']:.4f} | "
                     f"{s['frac_absq_lt_Q_over_8']:.4f} | {s['frac_zero']:.4f} |")
    lines += [
        '',
        '## 判读（C3 卡杀门1：|q|<Q/2 比例 < 30% 即停）',
        '',
        '见下结论节。注意：该比例是**静态资格上限**，动态可达率还受 |r₀| 分布限制，',
        '只能比它低；报告中不把静态上限当作实测跳过率。',
        '',
    ]
    fracs = [s['frac_absq_lt_Q_over_2'] for s in rows.values()]
    verdict = ('**过静态门**：全部模型 |q|<Q/2 比例 ≥ 30%，C3 卡保留，进入动态测量。'
               if min(fracs) >= 0.30 else
               '**杀**：存在模型 |q|<Q/2 比例 < 30%，静态资格不足，C3 卡停止。')
    lines.append(verdict)
    (ROOT / 'RESULT_A.md').write_text('\n'.join(lines) + '\n')
    print('KILL A done:', {k: round(v['frac_absq_lt_Q_over_2'], 4) for k, v in rows.items()})


def layer_moments_and_margins(z):
    """按 build_traces.py 口径重算 V/tau 的裕度，分块限制内存。"""
    S = np.unpackbits(z['source_packed'], axis=1, bitorder='little')[:, :96].astype(np.float64)
    N, C = S.shape
    T, P, H = 10, N // 10, 384
    W = z['W'].astype(np.float64)
    A = z['A']
    gamma = z['gamma']
    beta = z['beta']
    bias = z['bias']
    center = z['center']
    theta = 1.0
    theta_source = 1.0
    R = A.sum(1).reshape(T, 1)
    margins = []
    l1s = []
    for lo in range(0, H, 96):
        hi = min(H, lo + 96)
        wh = W[lo:hi]
        Y = (S @ wh.T * theta_source).reshape(T, P, hi - lo)
        mu = Y.mean((0, 1))
        var = ((Y - mu) ** 2).mean((0, 1))
        U = np.einsum('ts,sph->tph', A, Y, optimize=True)
        direction = np.sign(gamma[lo:hi])
        V = U * direction
        final_tau = (mu * R + np.sqrt(var + 1e-5) / gamma[lo:hi]
                     * (theta + center - bias - beta[lo:hi] * R)) * direction
        margins.append(np.abs(V - final_tau[:, None, :]).ravel())
        # 每判决 L1 供数规模：门消费者对该判决的总绝对贡献界
        absA = np.abs(A)
        absW = np.abs(wh)
        act = S.reshape(T, P, C)
        contrib = np.einsum('ts,spc,hc->tph', absA, act, absW, optimize=True) * theta_source
        l1s.append(contrib.ravel())
        del Y, U, V, contrib
    return np.concatenate(margins), np.concatenate(l1s)


def kill_b():
    z = np.load(TRACE)
    fail = z['failure']  # (600, 384)
    nb, H = fail.shape
    per_bc = float(fail.mean())
    # block 级门保持率
    block_fail = fail.any(1)
    block_fail_rate = float(block_fail.mean())
    # SIMD lane 联合接受：块内连续通道分组
    lane_stats = {}
    for L in (2, 4, 8, 16, 32, 96, 384):
        lanes = fail.reshape(nb, H // L, L)
        lane_fail = lanes.any(2)
        lane_stats[L] = {
            'lane_fail_rate': float(lane_fail.mean()),
            'joint_accept_rate': float(1 - lane_fail.mean()),
        }
    # 通道偏斜
    chan_fail = fail.mean(0)
    margins, l1s = layer_moments_and_margins(z)
    ratio = margins / np.maximum(l1s, 1e-30)
    # 位深锁定代理：margin > L1 * 2^-b
    lock = {}
    for b in (4, 8, 12, 16, 20, 24):
        lock[b] = float(np.mean(ratio > 2.0 ** -b))
    qs = [float(np.quantile(ratio, q)) for q in (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9)]
    out = {
        'trace': str(TRACE.relative_to(HW)),
        'n_block': nb, 'n_channel': H,
        'block_channel_fail_rate': per_bc,
        'block_any_fail_rate': block_fail_rate,
        'lane': lane_stats,
        'channel_fail_top10': {
            int(i): float(chan_fail[i]) for i in np.argsort(chan_fail)[-10:]},
        'margin_over_l1_quantiles_1_5_10_25_50_75_90': qs,
        'lock_frac_by_bitdepth': {str(k): v for k, v in lock.items()},
        'note': 'margin/L1 为裕度相对该判决总绝对贡献界的比值；lock 为 margin>L1*2^-b 的判决比例，是静态代理，不是RTL实测',
    }
    (ROOT / 'kill_b_raw.json').write_text(json.dumps(out, indent=2) + '\n')

    lines = [
        '# Kill B：消费者分级条件完成的静态天花板',
        '',
        f"数据源：`{TRACE.relative_to(HW)}`（真实捕获源 + ep34 权重，K=4 秩条件完成口径）。",
        '',
        f"- (block,channel) 门失败率：**{per_bc:.4%}**（逐通道接受率 {1-per_bc:.4%}）",
        f"- block 级门保持率（任一通道失败）：**{block_fail_rate:.4%}**",
        '',
        '| SIMD lane 宽 L | lane 联合失败率 | lane 联合接受率 |',
        '|---:|---:|---:|',
    ]
    for L, s in lane_stats.items():
        lines.append(f"| {L} | {s['lane_fail_rate']:.4%} | {s['joint_accept_rate']:.4%} |")
    lines += [
        '',
        '## 判决裕度 → 位深锁定代理（margin > L1·2^−b）',
        '',
        '| 位深 b | 锁定判决比例 |',
        '|---:|---:|',
    ]
    for b, v in lock.items():
        lines.append(f"| {b} | {v:.4%} |")
    lines += [
        '',
        f"margin/L1 分位数（1/5/10/25/50/75/90%）：{['%.3e' % q for q in qs]}",
        '',
        '## 判读（C1 卡杀门1/2）',
        '',
        '- lane 宽度联合接受相对逐通道的粒度税看 L=8 一档：',
        f"  L=8 联合接受率 {lane_stats[8]['joint_accept_rate']:.4%} vs 逐通道 {1-per_bc:.4%}。",
        '- 位深锁定比例（b=4/8 两档）决定"浅位深锁定"是否有物理空间。',
        '- 全部为静态代理（单trace、旧学生链、块粒度B=32），不是RTL周期结论。',
        '',
    ]
    l8 = lane_stats[8]['joint_accept_rate']
    lock4, lock8 = lock[4], lock[8]
    if lock8 >= 0.90:
        lines.append(f'**过静态门**：b=8 锁定比例 {lock8:.2%}（b=4 即 {lock4:.2%}），'
                     '门判决对贡献精度极不敏感，C1 消费者分级位深方向空间很大，'
                     '进入逐通道/lane 证书 RTL 阶段。')
    elif lock8 >= 0.15:
        lines.append(f'**部分保留**：b=8 锁定比例 {lock8:.2%} ≥ 15% 但 < 90%，C1 保留，位深分级按实测分布设计。')
    else:
        lines.append(f'**弱**：b=8 锁定比例 {lock8:.2%} < 15%，浅位深锁定空间不足，C1 的位深分级子方向降权。')
    (ROOT / 'RESULT_B.md').write_text('\n'.join(lines) + '\n')
    print('KILL B done: per_bc=%.4f block_any=%.4f L8_joint=%.4f lock4=%.4f lock8=%.4f'
          % (per_bc, block_fail_rate, l8, lock[4], lock[8]))


def kill_c():
    z = np.load(TRACE)
    fail = z['failure']
    nb, H = fail.shape
    K, B = 4, 32
    block_fail = fail.any(1)
    # 两种粒度的门侧占用模型：pass 单元只处理前 K 秩，fail 单元全处理
    gate_occ_block = float((1 - block_fail.mean()) * (K / B) + block_fail.mean() * 1.0)
    L = 8
    lanes = fail.reshape(nb, H // L, L)
    lane_fail = lanes.any(2)
    gate_occ_lane = float((1 - lane_fail.mean()) * (K / B) + lane_fail.mean() * 1.0)
    rows = []
    for f_anchor in (0.0, 0.1, 0.25, 0.5, 1.0):
        ub = f_anchor * 1.0 + (1 - f_anchor) * gate_occ_block
        ul = f_anchor * 1.0 + (1 - f_anchor) * gate_occ_lane
        rows.append((f_anchor, ub, ul))
    out = {
        'trace': str(TRACE.relative_to(HW)),
        'block_any_fail_rate': float(block_fail.mean()),
        'lane8_any_fail_rate': float(lane_fail.mean()),
        'K_over_B': K / B,
        'gate_occupancy_block_granularity': gate_occ_block,
        'gate_occupancy_lane8_granularity': gate_occ_lane,
        'union_by_anchor_frac': [
            {'anchor_frac': f, 'union_block': ub, 'union_lane8': ul,
             'saving_block': 1.0 - ub, 'saving_lane8': 1.0 - ul}
            for f, ub, ul in rows],
        'note': ('并集占用代理：anchor 位由连续消费者全程持有，非 anchor 位仅门消费者持有；'
                 '门占用 = pass 单元 K/B + fail 单元全量。anchor 占比未从本 trace 实测，'
                 '按 stride2 投影结构给出参数化曲线。单trace、旧学生链，非完整链实测。'),
    }
    (ROOT / 'kill_c_raw.json').write_text(json.dumps(out, indent=2) + '\n')

    lines = [
        '# Kill C：双消费者并集占用代理（C2/F1）',
        '',
        f"数据源：`{TRACE.relative_to(HW)}`；块数 {nb}，B=32，K=4。",
        '',
        f"- block 级门保持率（384 通道任一失败须全处理）：{block_fail.mean():.4%}",
        f"- lane8 级联合失败率：{lane_fail.mean():.4%}",
        f"- 门侧占用（block 粒度）：**{gate_occ_block:.2%}**",
        f"- 门侧占用（lane8 粒度）：**{gate_occ_lane:.2%}**",
        '',
        '## 并集占用 vs anchor 占比（连续消费者全量持有 anchor 位）',
        '',
        '| anchor 占比 | 并集占用(block粒度) | 节省 | 并集占用(lane8粒度) | 节省 |',
        '|---:|---:|---:|---:|---:|',
    ]
    for f, ub, ul in rows:
        lines.append(f"| {f:.0%} | {ub:.2%} | {1-ub:.2%} | {ul:.2%} | {1-ul:.2%} |")
    lines += [
        '',
        '## 判读（C2 卡杀门1：并集须有 ≥15% 供给义务下降空间）',
        '',
        f"- block 粒度：即便 anchor=0%，节省上限也只有 {1-gate_occ_block:.2%}；",
        f"  anchor=25% 时仅 {1-(0.25+0.75*gate_occ_block):.2%}。**块粒度退役空间不足**。",
        f"- lane8 粒度：anchor=25% 时节省 {1-(0.25+0.75*gate_occ_lane):.2%}，",
        '  若该值 ≥ 15%，C2 的源字退役必须落在 lane 粒度上，块粒度版本直接杀。',
        '- 该代理不含：连续消费者提前完成、bank 端口冲突、证书判定费用；',
        '  也不覆盖 anchor 占比实测——下一步应从真实 stride2 投影布局量 anchor 占比。',
        '',
    ]
    best_lane = 1 - (0.25 + 0.75 * gate_occ_lane)
    if best_lane >= 0.15:
        lines.append('**有条件保留**（按 anchor≤25%% 假设，lane8 粒度）：节省空间 %.2f%% ≥ 15%%，'
                     'C2 保留但 X 必须做在 lane 粒度；块粒度版本杀。' % (best_lane * 100))
    else:
        lines.append('**杀**：即便 lane8 粒度、anchor=25%%，节省空间 %.2f%% < 15%%。' % (best_lane * 100))
    (ROOT / 'RESULT_C.md').write_text('\n'.join(lines) + '\n')
    print('KILL C done: gate_occ_block=%.4f gate_occ_lane8=%.4f saving_lane8@25%%anchor=%.4f'
          % (gate_occ_block, gate_occ_lane, best_lane))


if __name__ == '__main__':
    kill_a()
    kill_b()
    kill_c()
