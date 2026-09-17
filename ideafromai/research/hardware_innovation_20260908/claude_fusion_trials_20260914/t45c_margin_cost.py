#!/usr/bin/env python3
"""T45c-1：把 C1 的"证书残差"概念搬到**全网**，做一张 成本 × 判决软度 交叉表。

## 为什么这是 T45c 要做的第一件事

T45b 修正后的靶点分布是三足鼎立（encoder 32.1% / frontend 29.0% / decoder 27.4%），
而 C1（组级 BFP + 位平面证书 + 逐 t 退休）只覆盖 12 个 swin-MLP 的 `fc1→sn2` 门路 = 17.1%。
"把 C1 搬到整机"这件事能不能成立，取决于一件可测的事：

> **全网每一个 ATLIF neuron 的判决有多"软"**，即膜电位离阈值多近。

C1 在门路上做的事本质是：**用未读比特的可达集给出剩余不确定度，再拿来早停**。
任何这种机制的成本都由"判决有多接近阈值"决定 —— 判决越接近阈值，
精确证书就越不能早停（T27 已证 MSB-first 在精确证书下最优，且界已紧）。

## 数据来源（都是我们自己 ep34 的实测，不是借的）

`results/m1458_..._unified_hardware_capture_s40_r1_20260831/atlif_activity.json`
（93 个 ATLIF neuron，40 样本，与 ep34 checkpoint 同身份）：

- `activity`               = spikes / elements（发放率）
- `quant_sample_events`   = 膜电位序列的样本总数（= elements）
- `margin_abs_le_1_K`     = |h − threshold| ≤ 1/K 的样本数
  定义源码：`runtime_profile_nts11_hardware_p0_de881269.py:818-821`
  `(h_seq.sub(threshold).abs() <= 1.0/128).sum()`
- `deployment_dead_result` = 生产树已定论的 dead_debug（12 个 `attn_sn`）

**注意口径**：`margin_abs_le_1_K / elements` 是"**所有**膜电位样本"里近阈值的比例，
不是"发放样本里"的比例。所以它**不是**"可省 SOP 的比例"。
本脚本把它当**判决软度信号**用，并同时给出 `f_K / activity` 这个比值：
比值 ≫1 ⇒ 近阈值样本比脉冲本身还多，说明"精确证书"在这层会被迫保留大量分辨率；
比值 ≪1 ⇒ 判决几乎都是干脆的，证书便宜。

用法：/opt/anaconda3/envs/pytorch310/bin/python t45c_margin_cost.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t45_mapping_cost import bucket, rate_key  # noqa: E402

ROOT = Path(__file__).resolve().parent
CODE = Path('/home/zhumd/work/sdformer_codex/SDformer')
CAP = (CODE / 'hw_autoresearch_nts07/results'
       / 'm1458_m1434_motion_ep34_live93_unified_hardware_capture_s40_r1_20260831'
       / 'atlif_activity.json')
T45B = ROOT / 'results' / 't45b_mapping_cost_ep34.json'

# decoder 用口径 B（真实工作量）——见 T45B_REPORT.md §3
DECONV_B = ROOT / 'results' / 't45b3_deconv_convention.json'
KS = (16, 32, 64, 128)
GROUPS = {
    'encoder.swin3d': lambda k: 'encoders.swin3d.layers.' in k,
    'frontend.patch_embed': lambda k: '.patch_embed' in k,
    'decoder': lambda k: 'sttmultires_unet.decoders.' in k,
    'bottleneck': lambda k: 'sttmultires_unet.resblocks' in k,
    'pred': lambda k: 'sttmultires_unet.preds.' in k,
}


def group_of(name):
    for g, p in GROUPS.items():
        if p(name):
            return g
    return 'other'


def main():
    cap = {e['name'].replace('.spiking_neuron', ''): e
           for e in json.loads(CAP.read_text())}
    t45b = json.loads(T45B.read_text())
    conv = json.loads(DECONV_B.read_text())
    # T45c-2 用 40 样本真实捕获查出三处 frontend 挂载错误（head.conv / proj.conv /
    # proj.conv_res），沿用它的修正值，保证本报告与 T45c-2/3 同一口径
    corr = {r['op']: r['sop_B'] for r in json.loads(
        (ROOT / 'results' / 't45c2_density_operand_width.json').read_text())['rows']
        if r['corrected']}

    # 口径 B：把 decoder 的 SOP 换掉（按 stage 匹配）
    b_sop = {}
    for r in conv['rows']:
        st = r['module'].split('decoders.')[1].split('.')[0]
        b_sop['sttmultires_unet.decoders.%s' % st] = r['sop_B']

    rows = []
    for opname, d in t45b['detail'].items():
        grp = group_of(opname)
        sop = d['sops'] if d['sops'] is not None else d['macs']
        if grp == 'decoder':
            for st, v in b_sop.items():
                if opname.startswith(st):
                    sop = v
        if opname in corr:
            sop = corr[opname]
        k = rate_key(opname)
        neuron = None
        if isinstance(k, str) and k not in ('@dense_input',):
            neuron = k
        elif isinstance(k, tuple):
            neuron = k[-1] if k[0] != '@UB' else k[1]
        rows.append({'op': opname, 'group': grp, 'macs': d['macs'], 'sop': sop,
                     'neuron': neuron,
                     'dense_input': k == '@dense_input'})

    # 逐 neuron 的软度
    def soft(neuron):
        e = cap.get(neuron)
        if e is None:
            return None
        n = e['quant_sample_events'] or 1
        out = {'activity': e['activity'], 'dead': e['deployment_dead_result']}
        for K in KS:
            m = e.get('margin_abs_le_1_%d' % K, 0)
            out['f%d' % K] = m / n
            out['r%d' % K] = (m / n) / e['activity'] if e['activity'] > 0 else None
        return out

    agg = {}
    for r in rows:
        g = agg.setdefault(r['group'], {'sop': 0.0, 'macs': 0.0, 'nops': 0,
                                        'sop_soft': {K: 0.0 for K in KS},
                                        'sop_noneur': 0.0, 'dead_sop': 0.0})
        g['sop'] += r['sop']
        g['macs'] += r['macs']
        g['nops'] += 1
        s = soft(r['neuron']) if r['neuron'] else None
        r['soft'] = s
        if s is None:
            g['sop_noneur'] += r['sop']
            continue
        if s['dead']:
            g['dead_sop'] += r['sop']
        for K in KS:
            g['sop_soft'][K] += r['sop'] * s['f%d' % K]

    out = {'source': str(CAP), 'rows': rows, 'groups': agg}
    (ROOT / 'results' / 't45c_margin_cost.json').write_text(
        json.dumps(out, indent=1) + '\n')

    TS = sum(g['sop'] for g in agg.values())
    print('=== 成本 × 判决软度（口径 B，SOP 合计 %.3fG）===' % (TS / 1e9))
    print('%-24s %9s %7s | %s' % ('group', 'SOP(G)', 'SOP%',
                                  '  '.join('f1/%-3d' % K for K in KS)))
    for g, d in sorted(agg.items(), key=lambda kv: -kv[1]['sop']):
        print('%-24s %9.3f %6.2f%% | %s'
              % (g, d['sop'] / 1e9, 100 * d['sop'] / TS,
                 '  '.join('%5.3f ' % (d['sop_soft'][K] / d['sop']) for K in KS)))
    print('\n（f1/K = 该组 SOP 加权后的"膜电位落在阈值 ±1/K 内"的样本占比）')

    print('\n=== 逐 neuron：软度最极端者（按 f1/16 排序，只列活跃 neuron）===')
    print('%-56s %7s %8s %8s %8s %8s' % ('neuron', 'act', 'f1/16', 'f1/64',
                                          'r1/16', 'r1/64'))
    live = [r for r in rows if r['soft'] and not r['soft']['dead']]
    seen = set()
    for r in sorted(live, key=lambda x: -x['soft']['f16']):
        nm = r['neuron']
        if nm in seen:
            continue
        seen.add(nm)
        s = r['soft']
        print('%-56s %7.4f %8.4f %8.5f %8s %8s'
              % (nm.replace('sttmultires_unet.', '').replace('encoders.swin3d.', 'enc.'),
                 s['activity'], s['f16'], s['f64'],
                 ('%.2f' % s['r16']) if s['r16'] is not None else '-',
                 ('%.2f' % s['r64']) if s['r64'] is not None else '-'))
        if len(seen) >= 14:
            break

    print('\n=== 无 neuron 可挂 / dead 的 SOP ===')
    for g, d in sorted(agg.items(), key=lambda kv: -kv[1]['sop']):
        if d['sop_noneur'] or d['dead_sop']:
            print('%-24s 无 neuron %.4fG   dead %.4fG'
                  % (g, d['sop_noneur'] / 1e9, d['dead_sop'] / 1e9))
    print('\nwrote results/t45c_margin_cost.json')


if __name__ == '__main__':
    main()
