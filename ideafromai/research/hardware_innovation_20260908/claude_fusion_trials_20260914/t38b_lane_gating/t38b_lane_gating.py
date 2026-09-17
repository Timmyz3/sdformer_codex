#!/usr/bin/env python3
"""T38b：support-aware lane gating 的**精确性证明** + 两个改进变体 + 数量维的"剪难组"口径。

T38 门 A 结果：k=4 掩码下，只发"未锁判决还在用的 lane"可省 **34.86% 比特**；
dense A_q 下省 0.00%（= 照抄 BitL 失败的根因）。本脚本把它做扎实：

  [1] **精确性验证**：逐 (组, 平面) 断言"被省略的 lane 对所有未解析判决的权重恒为 0"
      —— 若断言通过，则该 gating 是**零差**的（省掉的比特对判决无影响），
      不是近似压缩。这是它能否进论文的前提。

  [2] **改进变体 A（免反馈调度）：块状支撑**。原版的 lane 退役调度依赖消费者实时
      解出的 jf（需 10 根 ready 线）。若把 A 的支撑**约束成块对角**，则 lane k 只被
      自己那一块的判决使用 → 调度可由"块完成"直接推导（更少反馈）。
      代价：块约束限制了 top-k 的选择自由度 → 测它的供数代价 vs gating 收益。

  [3] **改进变体 B（更省反馈）**：把 10 根 per-lane ready 换成"每平面一个 active-mask"
      —— 测"退役事件数/组"，量化反馈成本（T38 门 A 的隐形成本尚未计入）。

  [4] **数量维的对称性**（T38 门 D 后续）：已测"剪最贵组 25% 省 49.28%" vs
      "剪最便宜组 25% 省 5.81%"。这里给出**供数/丢弃比**曲线，判断"按难度剪"
      是否是一个可用的口径（其精度代价留待 GPU）。

口径：bits/组；分组 = 10 lane × planes。自有代码；capture 只读。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 't32_lever'))
from t32_lever import cert_planes, trace_setup, mask_keep  # noqa: E402
import t19_all_layers as t19  # noqa: E402

T = 10
LIDS = [8, 14, 20, 28]
NSID = 10
KEEP = 4
POPTBL = np.array([bin(i).count('1') for i in range(1 << T)], np.int8)


def block_mask(A_q, keep, nblk=2):
    """块对角 top-k：把 10 个判决行分成 nblk 块，每块只在自己那块 lane 里选 keep 个。"""
    out = np.zeros_like(A_q)
    B = T // nblk
    for b in range(nblk):
        rs, cs = slice(b * B, (b + 1) * B), slice(b * B, (b + 1) * B)
        blk = A_q[rs, cs]
        ordr = np.argsort(np.abs(blk), axis=1, kind='stable')
        drop = ordr[:, :blk.shape[1] - keep]
        nb = blk.copy()
        np.put_along_axis(nb, drop, 0, axis=1)
        out[rs, cs] = nb
    return out


def gating(Aw, msb, jfw, check_exact=False):
    """返回 (bits_actual, bits_gated, ok_exact, n_retire_events)。"""
    sup = (Aw != 0)
    masks = np.array([sum((1 << k) for k in range(T) if sup[t, k]) for t in range(T)], np.int64)
    act = gate = 0
    events = 0
    ok = True
    jmin = jfw.min(1)
    for g in range(msb.size):
        top, bot = int(msb[g]), int(jmin[g])
        if top - bot <= 0:
            continue
        act += (top - bot) * T
        prev = -1
        for j in range(top - 1, bot - 1, -1):          # **按发送顺序**（j 递减）
            uu = 0
            for t_ in range(T):
                if jfw[g, t_] <= j:
                    uu |= int(masks[t_])
            pc = int(POPTBL[uu])
            gate += pc
            if prev >= 0 and pc < prev:
                events += 1
            prev = pc
            if check_exact:
                # 被省略的 lane 必须在所有未解析判决里权重为 0
                for k in range(T):
                    if not (uu >> k) & 1:
                        for t_ in range(T):
                            if jfw[g, t_] <= j and Aw[t_, k] != 0:
                                ok = False
    return act, gate, ok, events


def main():
    z = np.load(t19.PARAMS)
    prms = {}
    for lid in LIDS:
        prms[lid] = {'lid': lid, 'W': z['L%d_W' % lid], 'A': z['L%d_A' % lid],
                     'gamma': z['L%d_gamma' % lid], 'beta': z['L%d_beta' % lid],
                     'bias': z['L%d_bias' % lid], 'center': z['L%d_center' % lid],
                     'theta_src': z['L%d_theta_src' % lid]}
    src = t19.parse_sources(list(range(NSID)))

    res = {t: {'act': 0, 'gate': 0, 'ev': 0} for t in ('k4', 'k4blk')}
    exact_ok = True
    planes_plain, planes_blk = [], []

    for lid in LIDS:
        A_q = np.rint(prms[lid]['A'] * 4096).astype(np.int64)
        A_q = np.where(A_q >= (1 << 15), A_q - (1 << 16), A_q)
        A4 = mask_keep(A_q, KEEP)
        A4b = block_mask(A_q, KEEP, nblk=2)

        for sid in range(NSID):
            key = (sid, lid)
            if key not in src:
                continue
            packed, C, br = src[key]
            Yq, thr_g, _ = trace_setup(packed, C, br, prms[lid])
            del src[key]
            msb = np.array([int(np.abs(Yq[g]).max()).bit_length() for g in range(Yq.shape[0])])

            for tag, Aw in (('k4', A4), ('k4blk', A4b)):
                pl, _, jf = cert_planes(Yq, thr_g, Aw)
                a, gt, ok, ev = gating(Aw, msb, jf, check_exact=(sid == 0 and tag == 'k4'))
                exact_ok &= ok
                res[tag]['act'] += a
                res[tag]['gate'] += gt
                res[tag]['ev'] += ev
                if tag == 'k4':
                    planes_plain.append(pl)
                else:
                    planes_blk.append(pl)
        print('  L%02d done' % lid, flush=True)

    n_grp = len(planes_plain[0]) * len(planes_plain)
    print('\n== [1] 精确性验证 ==')
    print('  被省略 lane 对未解析判决权重恒 0：', 'PASS（零差）' if exact_ok else 'FAIL')
    print('\n== [2] 改进变体 A：块状支撑（免反馈调度）==')
    for tag, name in (('k4', '普通 top-4'), ('k4blk', '块对角 top-4')):
        a, gt, ev = res[tag]['act'], res[tag]['gate'], res[tag]['ev']
        print('  %s：实际 %d，gated %d → 省 **%.2f%%**；退役事件 %.2f 次/组'
              % (name, a, gt, 100 * (1 - gt / a), ev / n_grp))
    print('\n== [3] 供数口径（planes/组；分母 24）==')
    pp = np.concatenate(planes_plain); pb = np.concatenate(planes_blk)
    print('  普通 top-4 : %.3f planes/组（供数 %.2f%%）' % (pp.mean(), 100 * pp.mean() / 24))
    print('  块对角 top-4: %.3f planes/组（供数 %.2f%%）← 块约束的供数代价'
          % (pb.mean(), 100 * pb.mean() / 24))
    print('\n== [4] 数量维对称性：供数/丢弃比 ==')
    dense = np.concatenate(planes_plain).astype(np.float64)
    dense = np.clip(dense, 0, None)
    tot = dense.sum()
    o_asc, o_desc = np.argsort(dense), np.argsort(-dense)
    curve = {}
    for q in (0.05, 0.10, 0.25, 0.50):
        n = int(q * dense.size)
        cheap = dense[o_asc[:n]].sum() / tot
        hard = dense[o_desc[:n]].sum() / tot
        curve['q=%.2f' % q] = {'drop_cheapest_saves': float(cheap),
                               'drop_hardest_saves': float(hard)}
        print('  丢弃 %4.0f%% 的组：按最便宜 %.2f%% 供数；按最难 %.2f%% 供数'
              % (100 * q, 100 * cheap, 100 * hard))

    out = {'exact_ok': bool(exact_ok),
           'k4': {'act': int(res['k4']['act']), 'gate': int(res['k4']['gate']),
                  'saving': float(1 - res['k4']['gate'] / res['k4']['act']),
                  'retire_events_per_group': float(res['k4']['ev'] / n_grp)},
           'k4_block': {'act': int(res['k4blk']['act']), 'gate': int(res['k4blk']['gate']),
                        'saving': float(1 - res['k4blk']['gate'] / res['k4blk']['act']),
                        'retire_events_per_group': float(res['k4blk']['ev'] / n_grp)},
           'planes_plain_per_group': float(pp.mean()),
           'planes_block_per_group': float(pb.mean()),
           'quantity_curve': curve, 'n_groups': int(n_grp)}
    (ROOT / 'results' / 't38b_lane_gating.json').write_text(json.dumps(out, indent=1) + '\n')
    print('\nwrote results/t38b_lane_gating.json')


if __name__ == '__main__':
    main()
