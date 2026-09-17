"""T10 RTL 校验：BitL 优化版（cert_gate_bitl）vs 整数模型期望（T5 存档）。

判决逐组逐单元 + 逐组拍数零差；另与 T5 原 RTL 输出逐行比对（RTL↔RTL 交叉）。
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RES = ROOT.parent / 'results'
TRACES = ['s0_stage0', 's0_stage3', 's10_stage0', 's10_stage3']


def read_rtl(path):
    dec, cyc = [], []
    for ln in path.read_text().splitlines():
        if ln.startswith('#'):
            continue
        _, d, p = ln.split()
        dec.append(int(d, 16))
        cyc.append(1 + int(p))
    return np.array(dec, np.uint16), np.array(cyc, np.int32)


def main():
    rows = []
    for tr in TRACES:
        d5, d10 = RES / 't5_rtl' / tr, RES / 't10_rtl' / tr
        exp = np.load(d5 / 'expected.npz')
        dec_exp = exp['dec']
        row = {'trace': tr}
        for mode in ('fx_full', 'fx_cert', 'bf_full', 'bf_cert'):
            dec, cyc = read_rtl(d10 / f'rtl_{mode}.txt')
            dec_rtl = ((dec[:, None] >> np.arange(10)[None, :]) & 1).astype(np.uint8)
            assert dec_rtl.shape == dec_exp.shape
            mism = int((dec_rtl != dec_exp).sum())
            if mode == 'fx_cert':
                cyc_mism = int((cyc != exp['cyc_fx']).sum())
            elif mode == 'bf_cert':
                cyc_mism = int((cyc != exp['cyc_bf']).sum())
            else:
                cyc_mism = 0
                if mode == 'fx_full':
                    assert (cyc == 24).all()
            # RTL↔RTL 交叉：与 T5 原 RTL 输出逐行一致
            cross = int((read_rtl(d5 / f'rtl_{mode}.txt')[0] != dec).sum())
            row[mode] = {'dec_mismatches': mism, 'cycle_mismatches': cyc_mism,
                         'cross_rtl_dec_mismatches': cross,
                         'mean_cycles': float(cyc.mean())}
        rows.append(row)
        print('%-12s dec_mism fx=%d bf=%d, cyc_mism fx=%d bf=%d, cross_rtl=%d' %
              (tr, row['fx_cert']['dec_mismatches'], row['bf_cert']['dec_mismatches'],
               row['fx_cert']['cycle_mismatches'], row['bf_cert']['cycle_mismatches'],
               row['fx_cert']['cross_rtl_dec_mismatches'] +
               row['bf_cert']['cross_rtl_dec_mismatches']))
    ok = all(r[m]['dec_mismatches'] == 0 and r[m]['cycle_mismatches'] == 0
             and r[m]['cross_rtl_dec_mismatches'] == 0
             for r in rows for m in ('fx_full', 'fx_cert', 'bf_full', 'bf_cert'))
    out = {'traces': rows, 'all_match': bool(ok),
           'note': 'T10 BitL 优化版 RTL（Verilator 4.028）零差校验：'
                   'vs 整数模型（判决+cert 拍数逐组）+ vs T5 原 RTL（逐组判决）。'
                   '数据通路改动：dot10→2×(5bit 子集和 LUT)；pn·msk 乘法→'
                   '增量递推 (pmsk−P)>>>1。'}
    (RES / 't10_rtl_verify.json').write_text(json.dumps(out, indent=1) + '\n')
    print('ALL MATCH' if ok else 'MISMATCH!')


if __name__ == '__main__':
    main()
