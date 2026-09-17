#!/usr/bin/env python3
"""T47：从 Vivado 报告里抽 LUT/FF/DSP/BRAM/WNS，与槽位普查一起落成结果 JSON。

用法：/opt/anaconda3/envs/pytorch310/bin/python t47_ppa.py
（先跑 t47_fused/t47_verify.sh 与 t47_fused/t47_synth.tcl）
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D = ROOT / 't47_fused'
TAGS = {'z': 't47_deconv_z', 'p': 't47_deconv_p'}

UTIL_PAT = {
    'lut_logic': r'^\|\s*LUT as Logic\s*\|\s*(\d+)',
    'lut_mem': r'^\|\s*LUT as Memory\s*\|\s*(\d+)',
    'ff': r'^\|\s*CLB Registers\s*\|\s*(\d+)',
    'bram': r'^\|\s*Block RAM Tile\s*\|\s*(\d+)',
    'dsp': r'^\|\s*DSPs\s*\|\s*(\d+)',
}


def parse_util(path: Path) -> dict:
    out = {}
    text = path.read_text(errors='ignore')
    for k, pat in UTIL_PAT.items():
        m = re.search(pat, text, re.M)
        out[k] = int(m.group(1)) if m else None
    return out


def parse_wns(path: Path):
    """report_timing_summary 的汇总行：WNS 那一行是纯数字行，紧跟在表头之后。"""
    text = path.read_text(errors='ignore')
    block = re.search(r'WNS\(ns\).*?\n[- ]+\n\s*(-?\d+\.\d+)', text)
    return float(block.group(1)) if block else None


def main():
    meta = json.loads((ROOT / 'results' / 't47_stim_meta.json').read_text())
    res = {'stim': meta, 'variants': {}}
    for tag, top in TAGS.items():
        u = parse_util(D / f'{tag}_util.rpt')
        res['variants'][tag] = {
            'top': top, **u,
            'wns_ns': parse_wns(D / f'{tag}_timing.rpt'),
            'clk_period_ns': 4.0,
        }

    z, p = res['variants']['z'], res['variants']['p']
    zl = z['lut_logic'] or 0
    pl = p['lut_logic'] or 0
    zs = meta['slots_expected']['z_slots']
    ps = meta['slots_expected']['p_slots_exact']
    nelem = meta['H'] * meta['W'] * meta['CIN']
    ny = 4 * meta['H'] * meta['W']
    res['comparison'] = {
        'lut_ratio_z_over_p': zl / pl if pl else None,
        'slot_ratio_z_over_p': zs / ps,
        'z_slots_per_input_elem': zs / nelem,
        'p_slots_per_input_elem': ps / nelem,
        'z_cycles_per_out_elem': zs / ny,
        'p_cycles_per_out_elem': ps / ny,
        'z_ns_per_out_elem_4ns': zs / ny * 4.0,
        'p_ns_per_out_elem_4ns': ps / ny * 4.0,
        'dsp_z': z['dsp'], 'dsp_p': p['dsp'],
        'both_zero_dsp': z['dsp'] == 0 and p['dsp'] == 0,
    }
    (ROOT / 'results' / 't47_ppa.json').write_text(json.dumps(res, indent=1) + '\n')

    c = res['comparison']
    print('%-34s %10s %10s' % ('', 'Z 零插值', 'P 相位分解'))
    print('%-34s %10s %10s' % ('LUT as Logic', zl, pl))
    print('%-34s %10s %10s' % ('CLB Registers', z['ff'], p['ff']))
    print('%-34s %10s %10s' % ('DSP / BRAM', f"{z['dsp']} / {z['bram']}",
                               f"{p['dsp']} / {p['bram']}"))
    print('%-34s %10s %10s' % ('WNS @4ns (ns)', z['wns_ns'], p['wns_ns']))
    print('%-34s %10s %10s' % ('抽头槽位 / 输入元素', c['z_slots_per_input_elem'],
                               c['p_slots_per_input_elem']))
    print('%-34s %10s %10s' % ('周期 / 输出元素', c['z_cycles_per_out_elem'],
                               c['p_cycles_per_out_elem']))
    print('\nLUT 比 %.3f×；槽位比 %.3f×；两者 DSP 都是 0：%s'
          % (c['lut_ratio_z_over_p'], c['slot_ratio_z_over_p'], c['both_zero_dsp']))
    print('wrote results/t47_ppa.json')


if __name__ == '__main__':
    main()
