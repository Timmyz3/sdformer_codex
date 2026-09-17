#!/usr/bin/env python3
"""T41：把 C1 侧与融合基线侧的 OOC 综合报告收成一张可比表 → results/t41_ppa.json。

口径对齐的三件事（否则表不可比）：
  1. **同器件同流程**：全部 xczu5ev-sfvc784-2-e / OOC / 4ns 目标 / Vivado 2025.1。
  2. **同一 A**：两侧都用 results/t21b_k4_gate_params.npz: L8_A（t21f k=4 微调）。
  3. **同一时序语义**：t41_fused_gate 无 reg→reg 路径，默认报告给 WNS=NA；本表用
     `t41_if.xdc`（input/output delay 0）把它变成「端口→组合→FF」，与 T25/T39 的
     「reg→组合→reg」拿到同一条 4ns 预算。两套数都记（`*_util.rpt` 不带延时）。
  4. 融合侧的 40 个乘法 Vivado 默认推成 DSP48E2（241 LUT / 40 DSP）；T25/T39 的加法树
     是 0 DSP。故融合侧补跑 `-max_dsp 0` 给出 LUT-only 面积/时序，才是与 C1 同实现的对照。

用法：python t41_fused/t41_ppa.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'

# (标签, 报告路径, 说明)
UTIL = [
    ('t41_dsp', ROOT / 't41_fused/dsp_util.rpt'),
    ('t41_lut', ROOT / 't41_fused/lut_util.rpt'),
    ('plane_ser', ROOT / 't24_fpga/plane_ser_util.rpt'),
    ('t25_k4_gate', ROOT / 't25_k4_gate/treek4_util.rpt'),
    ('t39_va_gate', ROOT / 't39_sglr_rtl/pp_util.rpt'),
]
TIM = [
    ('t41_dsp', ROOT / 't41_fused/dsp_timing.rpt'),
    ('t41_lut', ROOT / 't41_fused/lut_timing.rpt'),
    ('plane_ser', ROOT / 't24_fpga/plane_ser_timing.rpt'),
    ('t25_k4_gate', ROOT / 't25_k4_gate/treek4_timing.rpt'),
    ('t39_va_gate', ROOT / 't39_sglr_rtl/pp_timing.rpt'),
]


def grab_util(p):
    t = p.read_text()
    out = {}
    for key, pat in (('lut', r'\|\s*CLB LUTs\*\s*\|\s*(\d+)\s*\|'),
                     ('ff', r'\|\s*CLB Registers\s*\|\s*(\d+)\s*\|'),
                     ('dsp', r'\|\s*DSPs\s*\|\s*(\d+)\s*\|'),
                     ('carry8', r'\|\s*CARRY8\s*\|\s*(\d+)\s*\|')):
        m = re.search(pat, t)
        out[key] = int(m.group(1)) if m else None
    return out


def grab_timing(p):
    t = p.read_text()
    out = {}
    m = re.search(r'Worst Slack\s+(-?[\d.]+)ns', t)
    out['wns'] = float(m.group(1)) if m else None
    m = re.search(r'Data Path Delay:\s+([\d.]+)ns', t)
    out['datapath_ns'] = float(m.group(1)) if m else None
    m = re.search(r'Source:\s+(\S+)', t)
    out['src'] = m.group(1) if m else None
    m = re.search(r'Destination:\s+(\S+)', t)
    out['dst'] = m.group(1) if m else None
    if out['datapath_ns']:
        out['fmax_mhz_est'] = round(1000.0 / out['datapath_ns'], 1)
    return out


def main():
    tab = {}
    for (tag, pu), (_, pt) in zip(UTIL, TIM):
        if not pu.exists() or not pt.exists():
            continue
        row = grab_util(pu)
        row.update(grab_timing(pt))
        tab[tag] = row

    c1_a = {k: (tab['plane_ser'][k] + tab['t39_va_gate'][k]) for k in ('lut', 'ff', 'dsp')}
    c1_t25 = {k: (tab['plane_ser'][k] + tab['t25_k4_gate'][k]) for k in ('lut', 'ff', 'dsp')}

    out = {
        'device': 'xczu5ev-sfvc784-2-e', 'ooc': True, 'clk_ns': 4.0,
        'modules': tab,
        'c1_composite': {
            'plane_ser + t39_A_gate': c1_a,
            'plane_ser + t25_k4_gate': c1_t25,
            'note': 'C1 侧 = 生产者串行器 + 门核（融合模块同时替代这两者）。',
        },
        'payload_bits_per_group': {'c1_t39_A': 13.6516, 'fused_decisions': 10.0},
        'cycles_per_group': {'c1_t39_A': 3.0310, 'fused_decisions': 1.0},
        'aee': 'identical（证书递归终态判决 == A_q·Y ≥ thr，0 失配）',
        'caveats': [
            'OOC 综合估计，未 place/route；+0.077ns 是零裕量，不能当作已收敛。',
            't41_lut（-max_dsp 0）才是与 T25/T39 同实现（0 DSP）的对照；t41_dsp 是默认推断。',
            '融合侧不含 fc1 的 W×S→Y MAC（两侧都吃 Y 作为输入，故可比）。',
            '吞吐粗算：C1 3.0310 拍/组 @ 332MHz ≈ 109.6 M 组/s；融合 1 拍/组 @ 254MHz ≈ 254 M 组/s。',
        ],
    }
    (R / 't41_ppa.json').write_text(json.dumps(out, indent=1) + '\n')
    for tag, r in tab.items():
        print('%-14s LUT %-6s FF %-6s DSP %-3s WNS %-8s datapath %-7s @%.1fMHz'
              % (tag, r['lut'], r['ff'], r['dsp'], r['wns'], r['datapath_ns'],
                 r.get('fmax_mhz_est') or 0))
    print('C1 (plane_ser + t39_A) :', c1_a)
    print('C1 (plane_ser + t25_k4):', c1_t25)
    print('wrote results/t41_ppa.json')


if __name__ == '__main__':
    main()
