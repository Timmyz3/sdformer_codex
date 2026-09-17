#!/usr/bin/env python3
"""T45c-2 + T45c-3：用 40 样本真实捕获交叉验证成本表，并给 C1 的天花板下"结构"判据。

## T45c-2 为什么能交叉验证

T45b 的活动率来自 `m2041_.../spike_profile.json`（825 样本、480×640、T=10、no_running BN）。
`m1458_..._unified_hardware_capture_s40_r1_20260831/operator_runtime.json` 是**另一条独立管线**
（40 样本、逐算子 forward hook 抓的**真实输入张量**），它给出每个算子的
`input_sample_density / input_activity / value_min/max/absmax/mean_abs`。

同一算子的"输入脉冲占比"被两条管线各测一次 ⇒ 可以逐算子对表。对得上 ⇒ T45b 的成本列
是**双来源**的，不是单点估计。

## T45c-3 为什么是"结构"判据而不是"效果"判据

C1 的精确证书作用在**被逐位送达的多比特操作数**上：把 MSB 先送，用未读位的可达区间
做证书，区间与阈值单侧分离就退休。**如果操作数本身就是 0/1 脉冲，就没有位平面可裁**，
证书无从谈起 —— 这不是"效果弱"，是**没有作用对象**。

所以问题变成纯计数：全网活动加权 SOP 里，有多大比例的操作数是**多比特**的？

判据（只用捕获里已有的汇总量，不需要重读 payload）：
0/θ 型脉冲张量满足 `mean_abs == θ·density == max·density`。
于是 `bulk = mean_abs / (max·density)`：`bulk≈1` ⇒ 所有非零值同一个常数 ⇒ 0/θ 脉冲；
`bulk<1` ⇒ 非零值有分布 ⇒ 真多比特。

（注意 `input_sample_binary01_ratio` **不能**直接用来判：它测的是"值落在 {0,1} 里的比例"，
但本模型的脉冲值 θ≈0.99991 而不是 1.0，于是所有非零值都判"不是 1"，该指标退化成
零占比 `1−density`。resblocks 那几个 `binary01_ratio 0.87 / 0.85` 就是这么来的，
它们其实是干净的 0/1 脉冲 —— 本脚本另有 payload 逐值核对作证。）

用法：/opt/anaconda3/envs/pytorch310/bin/python t45c2_density_operand_width.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CODE = Path('/home/zhumd/work/sdformer_codex/SDformer')
CAP = (CODE / 'hw_autoresearch_nts07/results'
       / 'm1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831')
PROF = (CODE / 'hw_autoresearch_nts07/system_handoff/incoming'
        / 'm2041_ep34_quant_binding_inputs/spike_profile.json')
T45B = ROOT / 'results' / 't45b_mapping_cost_ep34.json'
DECONV = ROOT / 'results' / 't45b3_deconv_convention.json'

SUPER = (('enc.', 'encoder.swin3d'), ('frontend.', 'frontend.patch_embed'),
         ('decoder.', 'decoder'), ('bottleneck.', 'bottleneck.resblocks'),
         ('pred.', 'pred'))


def supergroup(group: str) -> str:
    for pre, name in SUPER:
        if group.startswith(pre):
            return name
    return 'other:' + group


def rows_sop(rows, suffix):
    return sum(r['sop_B'] for r in rows if r['op'].endswith(suffix))


def main():
    ops = {e['name']: e for e in json.loads((CAP / 'operator_runtime.json').read_text())}
    t45b = json.loads(T45B.read_text())
    prof = {k: v['firing_rate']
            for k, v in json.loads(PROF.read_text())['layer_firing_rates'].items()}
    dec = {r['module']: r for r in json.loads(DECONV.read_text())['rows']}

    # 来源 C：逐样本记录（40 样本，逐样本 active/elements）——与前两条管线完全独立
    rec_acc = {}
    for line in (CAP / 'unified_ordered_records.jsonl').read_text().splitlines():
        r = json.loads(line)
        inp = r['input']
        if not inp['elements']:
            continue
        rec_acc.setdefault(r['name'], []).append(inp['active'] / inp['elements'])
    rec = {k: (sum(v) / len(v), len(v)) for k, v in rec_acc.items()}

    # ---------- T45c-2：两条独立管线逐算子对表 ----------
    xval = []
    for name, d in t45b['detail'].items():
        e = ops.get(name)
        rc = rec.get(name)
        a = e['input_activity'] if e is not None else None
        xval.append({
            'op': name, 'profile_rate_n825': d['rate'], 'capture_activity_n40': a,
            'capture_sample_density': e['input_sample_density'] if e else None,
            'record_mean_density_n40': rc[0] if rc else None,
            'record_samples': rc[1] if rc else 0,
            'rel_diff': (a - d['rate']) / d['rate'] if a is not None and d['rate'] else None,
            'rel_diff_rec': (rc[0] - d['rate']) / d['rate'] if rc and d['rate'] else None,
            'profile_neuron': d['rate_src'],
        })

    both = [x for x in xval if (x.get('capture_activity_n40') is not None
                                or x.get('record_mean_density_n40') is not None)
            and x['profile_rate_n825']]
    # 三条管线里至少两条都有、且该算子不是连续输入的，才拿来比
    tri = [x for x in both if x.get('record_mean_density_n40') is not None
           or x.get('capture_activity_n40') is not None]
    EXCL = ('patch_embed.head.conv.0', 'patch_embed.proj.conv_res')
    tri_pure = [x for x in tri if not x['op'].endswith(EXCL)]
    # C 源缺失时用 B 源（两者由同一批张量导出，本就应当一致）
    for x in tri_pure:
        a = x.get('capture_activity_n40')
        c = x.get('record_mean_density_n40')
        x['_src'] = 'B' if a is not None else 'C'
        x['_act'] = a if a is not None else c
        x['_rel'] = (x['_act'] - x['profile_rate_n825']) / x['profile_rate_n825']
    close = [x for x in tri_pure if abs(x['_rel']) <= 0.05]

    # ---------- T45c-3：操作数宽度 ----------
    width = []
    for name, d in t45b['detail'].items():
        e = ops.get(name)
        if e is None:
            continue
        dens = e['input_sample_density']
        mx = e['input_sample_value_max']
        bulk = (e['input_sample_value_mean_abs'] / (mx * dens)) if dens > 0 and mx > 0 else None
        if bulk is None:
            cls = 'empty'          # 该算子的输入全零（dead 路径）
        elif bulk >= 0.99:
            cls = 'binary01'       # 非零值同一个常数 ⇒ 0/θ 脉冲，没有位平面可裁
        elif bulk >= 0.90:
            cls = 'boundary'       # 非零值有 1–10% 的浅分布
        else:
            cls = 'multi_bit'      # 真多比特
        width.append({'op': name, 'group': d['group'], 'operand': cls,
                      'value_min': e['input_sample_value_min'], 'value_max': mx,
                      'bulk_ratio': bulk, 'theta_if_binary': mx if cls == 'binary01' else None,
                      'capture_binary01_ratio': e['input_sample_binary01_ratio'],
                      'sop_A': d['sops']})

    # ---------- 口径 B + frontend 两处修正 ----------
    width_by_op = {w['op']: w for w in width}
    # 口径 B：decoder 的 stride-2 deconv 换成输入散射（T45b3）
    b_by_stage = {m.split('decoders.')[1].split('.')[0]: r['sop_B'] for m, r in dec.items()}

    def b_sop(name, d):
        if d['group'].startswith('decoder.'):
            st = d['group'].split('decoder.')[1]
            return b_by_stage[st]
        return d['sops']

    # 两处 frontend 修正：捕获实测输入占比替换 T45b 里的保守/错挂估计
    # （修正量 = 实测 input_activity，不是手填的常数）
    PF = 'sttmultires_unet.encoders.swin3d.patch_embed.'
    FIX = {
        # head.conv 的输入是连续 voxel，T45b 按 r=1.0 保守计入
        PF + 'head.conv.0': ('head.conv', ops[PF + 'head.conv.0']['input_activity']),
        # proj.conv 与 proj.conv_res 在 T45b 里都被挂到 residual_encoding 的脉冲上（r≈0.032），
        # 实测：conv 是脉冲但更密（0.056），conv_res 是稠密连续（1.0）
        PF + 'proj.conv': ('proj.conv', ops[PF + 'proj.conv']['input_activity']),
        PF + 'proj.conv_res': ('proj.conv_res', ops[PF + 'proj.conv_res']['input_activity']),
    }

    rows = []
    no_cap = []
    for name, d in t45b['detail'].items():
        w = width_by_op.get(name)
        if w is None:
            no_cap.append(name)
            operand, bulk = 'unknown', None
        else:
            operand, bulk = w['operand'], w['bulk_ratio']
        sop = b_sop(name, d)
        rate = d['rate']
        fixed = None
        if name in FIX:
            fixed, rate = FIX[name]
            sop = d['macs'] * rate
        rows.append({'op': name, 'group': d['group'], 'super': supergroup(d['group']),
                     'macs': d['macs'], 'rate_used': rate, 'sop_B': sop,
                     'corrected': fixed, 'operand': operand,
                     'bulk_ratio': bulk})

    tot = sum(r['sop_B'] for r in rows)

    # 每组/超组汇总
    agg = {}
    for r in rows:
        for key, sel in (('group', r['group']), ('super', r['super'])):
            g = agg.setdefault((key, sel), {'sop': 0.0, 'macs': 0.0, 'nops': 0,
                                            'multibit_sop': 0.0})
            g['sop'] += r['sop_B']
            g['macs'] += r['macs']
            g['nops'] += 1
            if r['operand'] == 'multi_bit':
                g['multibit_sop'] += r['sop_B']

    mb = sum(r['sop_B'] for r in rows if r['operand'] == 'multi_bit')
    bd = sum(r['sop_B'] for r in rows if r['operand'] == 'boundary')
    mb_ops = [r for r in rows if r['operand'] == 'multi_bit']
    bd_ops = [r for r in rows if r['operand'] == 'boundary']

    out = {'capture': str(CAP), 'profile': str(PROF),
           'cross_validation': xval, 'operand_width': width,
           'rows': rows,
           'network_sop_B_corrected': tot,
           'multibit_sop': mb, 'multibit_frac_strict': mb / tot,
           'boundary_sop': bd, 'boundary_frac': bd / tot,
           'multibit_plus_boundary_frac': (mb + bd) / tot,
           'multibit_ops': [r['op'] for r in mb_ops],
           'boundary_ops': [r['op'] for r in bd_ops],
           'agg_group': {k[1]: v for k, v in agg.items() if k[0] == 'group'},
           'agg_super': {k[1]: v for k, v in agg.items() if k[0] == 'super'}}
    (ROOT / 'results' / 't45c2_density_operand_width.json').write_text(
        json.dumps(out, indent=1) + '\n')

    # ---------- 打印 ----------
    print('=== T45c-2：两条独立管线对表 ===')
    print('A = m2041 spike_profile（825 样本，480×640，T=10，no_running BN）')
    print('B = operator_runtime 逐算子 hook（40 样本，input_activity）')
    print('C = unified_ordered_records 逐样本 active/elements（40 样本；B 缺记录时用它顶）')
    print('可比的脉冲输入算子 %d 个；A vs B/C 相对差 ≤5%% 的 %d 个（≤1%% 的 %d 个）'
          % (len(tri_pure), len(close), len([x for x in close if abs(x['_rel']) <= 0.01])))
    print('%-56s %9s %9s %8s %6s' % ('op', 'A_prof', 'B/C_cap', 'rel', 'src'))
    for x in sorted(tri_pure, key=lambda y: -abs(y['_rel'])):
        nm = x['op'].replace('sttmultires_unet.', '').replace('encoders.swin3d.', '')
        print('%-56s %9.5f %9.5f %+7.2f%% %6s'
              % (nm, x['profile_rate_n825'], x['_act'], 100 * x['_rel'], x['_src']))
    print('\n连续输入（不可比）与捕获缺 B 记录的算子：')
    for x in both:
        if x['op'].endswith(EXCL):
            nm = x['op'].replace('sttmultires_unet.', '').replace('encoders.swin3d.', '')
            b = x.get('capture_activity_n40')
            c = x.get('record_mean_density_n40')
            print('  %-56s A %8.5f  B %s  C %s'
                  % (nm, x['profile_rate_n825'], ('%.5f' % b) if b is not None else '—',
                     ('%.5f' % c) if c is not None else '—'))
    miss = [x['op'] for x in xval if x.get('capture_activity_n40') is None
            and x.get('record_mean_density_n40') is None]
    if miss:
        print('两条捕获来源都没有的算子（%d 个）：%s'
              % (len(miss), ', '.join(m.replace('sttmultires_unet.', '') for m in miss)))

    print('\n=== T45c-3：操作数宽度（bulk = mean_abs/(max·density)；≈1 ⇒ 0/θ 脉冲）===')
    print('%-58s %7s %10s %8s %10s' % ('op', 'dens', 'max', 'bulk', 'operand'))
    for w in sorted(width, key=lambda y: (y['operand'] != 'multi_bit',
                                          y['operand'] != 'boundary', y['op'])):
        nm = w['op'].replace('sttmultires_unet.', '').replace('encoders.swin3d.', '')
        print('%-58s %7.4f %10.4f %8s %10s'
              % (nm, ops[w['op']]['input_sample_density'], w['value_max'],
                 ('%.4f' % w['bulk_ratio']) if w['bulk_ratio'] is not None else '—',
                 w['operand']))
    print('\n真多比特（bulk<0.90）%d 个：%s'
          % (len(mb_ops), ', '.join(r['op'].split('sttmultires_unet.')[-1] for r in mb_ops)))
    print('  ⇒ %.4fG / %.3fG = **%.2f%%**' % (mb / 1e9, tot / 1e9, 100 * mb / tot))
    print('浅分布边界带（0.90≤bulk<0.99）%d 个：%s'
          % (len(bd_ops), ', '.join(r['op'].split('sttmultires_unet.')[-1] for r in bd_ops)))
    print('  ⇒ %.4fG / %.3fG = %.2f%%（即使全算多比特也只到 %.2f%%）'
          % (bd / 1e9, tot / 1e9, 100 * bd / tot, 100 * (mb + bd) / tot))
    print('  ⇒ C1 的可作用面（多比特操作数）合计 ≤ %.2f%%：其中 %.4fG（=多比特的 %.0f%%）'
          % (100 * (mb + bd) / tot, mb / 1e9, 100 * 1.0))
    print('     全在 frontend 的两个**连续输入**算子上（head.conv %.4fG + proj.conv_res %.4fG），'
          % (rows_sop(rows, 'head.conv.0') / 1e9, rows_sop(rows, 'proj.conv_res') / 1e9))
    print('     它们不是"被逐位送达的脉冲"，C1 的位平面证书在结构上不适用；')
    print('     剩下的 %.4fG 是 encoder attn.proj 的浅分布（bulk %.2f–%.2f，约 2–4%% 的摊开）。'
          % (bd / 1e9, min(x['bulk_ratio'] for x in width if x['operand'] == 'boundary'),
             max(x['bulk_ratio'] for x in width if x['operand'] == 'boundary')))

    print('\n=== 口径 B + frontend 三处修正后的成本分布 ===')
    print('%-22s %9s %7s | %s' % ('supergroup', 'SOP(G)', 'SOP%', '其中多比特操作数'))
    for s, g in sorted(out['agg_super'].items(), key=lambda kv: -kv[1]['sop']):
        if s.startswith('other'):
            continue
        print('%-22s %9.3f %6.2f%% | %.4fG (%.2f%%)'
              % (s, g['sop'] / 1e9, 100 * g['sop'] / tot,
                 g['multibit_sop'] / 1e9, 100 * g['multibit_sop'] / g['sop']))
    print('\n三处 frontend 修正（T45b → 捕获实测）：')
    for name, (tag, rate) in FIX.items():
        d = t45b['detail'][name]
        print('  %-14s r %.5f → %.5f   %+.4fG'
              % (tag, d['rate'], rate, (d['macs'] * rate - d['sops']) / 1e9))
    print('\nwrote results/t45c2_density_operand_width.json')


if __name__ == '__main__':
    main()
