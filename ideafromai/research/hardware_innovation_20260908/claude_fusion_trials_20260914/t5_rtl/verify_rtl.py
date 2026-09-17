"""T5 RTL 校验：RTL 仿真结果 vs 整数模型期望（判决逐组逐单元 + 逐组拍数）。"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
R5 = ROOT.parent / 'results' / 't5_rtl'
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
        d = R5 / tr
        exp = np.load(d / 'expected.npz')
        dec_exp = exp['dec']                      # (G,10) uint8，全深度整数判决
        row = {'trace': tr}
        for mode in ('fx_full', 'fx_cert', 'bf_full', 'bf_cert'):
            dec, cyc = read_rtl(d / f'rtl_{mode}.txt')
            dec_rtl = ((dec[:, None] >> np.arange(10)[None, :]) & 1).astype(np.uint8)
            assert dec_rtl.shape == dec_exp.shape
            mism = int((dec_rtl != dec_exp).sum())
            # 拍数：cert 模式与模型逐组比对；full 模式仅检查常数口径
            if mode == 'fx_cert':
                cyc_mism = int((cyc != exp['cyc_fx']).sum())
            elif mode == 'bf_cert':
                cyc_mism = int((cyc != exp['cyc_bf']).sum())
            else:
                cyc_mism = 0
                if mode == 'fx_full':
                    assert (cyc == 24).all()      # 1头拍+23数据平面（bit23冗余不重发）
            row[mode] = {'dec_mismatches': mism, 'cycle_mismatches': cyc_mism,
                         'mean_cycles': float(cyc.mean())}
        rows.append(row)
        print('%-12s dec_mism fx=%d bf=%d, cyc_mism fx=%d bf=%d' %
              (tr, row['fx_cert']['dec_mismatches'], row['bf_cert']['dec_mismatches'],
               row['fx_cert']['cycle_mismatches'], row['bf_cert']['cycle_mismatches']))
    ok = all(r[m]['dec_mismatches'] == 0 and r[m]['cycle_mismatches'] == 0
             for r in rows for m in ('fx_full', 'fx_cert', 'bf_full', 'bf_cert'))
    out = {'traces': rows, 'all_match': bool(ok),
           'note': 'RTL（Verilator 4.028）与整数模型逐组逐判决零差校验；'
                   '拍数逐组比对（fx_cert vs 1+max(23-j*,1), bf_cert vs 1+max(planes,1)；'
                   'FX 基线 24 拍/组，bit23 与符号字冗余不重发）。'}
    (ROOT.parent / 'results' / 't5_rtl_verify.json').write_text(json.dumps(out, indent=1) + '\n')
    print('ALL MATCH' if ok else 'MISMATCH!')


if __name__ == '__main__':
    main()
