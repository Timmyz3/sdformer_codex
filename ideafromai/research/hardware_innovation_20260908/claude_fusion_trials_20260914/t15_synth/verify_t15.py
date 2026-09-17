#!/usr/bin/env python3
"""T15 RTL 校验：综合口径变体（cert_gate_bitl_synth / fx_gate_synth）零差验证。

- 常数按 trace 烘焙：t15_synth/<trace>/{a,pn,lut_lo,lut_hi}.hex（a/pn 复制自
  results/t5_rtl/<trace>，LUT 从 signed a 计算），SV 经 sed 替换 readmemh 路径
  生成 per-trace 副本后 verilator 构建——验证的即"烘焙常数"形态本身。
- cert_gate_bitl_synth：4 模式（fx_full/fx_cert/bf_full/bf_cert），期望 = T5 存档
  expected.npz（判决 + 拍数），另与 T10 RTL 输出逐行交叉。thr 从 tau_t*.hex 由 TB
  经 640b 端口注入（验证外置 SRAM 口径不改变功能）。
- fx_gate_synth：字串行基线，从平面激励按二补码重建 24b 词
  （Y[s] = mag − (sign_s ? 2^23 : 0)），期望 = expected.npz 判决（与 fx_full 同）。
- 综合面积口径用 s0_stage0 常数（结构相同，报告披露）。
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
T15 = ROOT / 't15_synth'
RES = ROOT / 'results'
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


def hexlines(arr, bits):
    return '\n'.join(format(int(x) & ((1 << bits) - 1), '0%dx' % (bits // 4))
                     for x in np.asarray(arr).ravel()) + '\n'


def gen_trace_hex(tr):
    d = T15 / tr
    d.mkdir(exist_ok=True)
    src = RES / 't5_rtl' / tr
    a = [int(x, 16) for x in (src / 'a.hex').read_text().split()]
    a = [x - (1 << 16) if x >= (1 << 15) else x for x in a]   # signed16
    pn = [int(x, 16) for x in (src / 'pn.hex').read_text().split()]
    (d / 'a.hex').write_text(hexlines(a, 16))
    (d / 'pn.hex').write_text(hexlines(pn, 48))
    lut_lo = np.zeros(320, np.int64)
    lut_hi = np.zeros(320, np.int64)
    for t in range(10):
        for cc in range(32):
            for ss in range(5):
                if cc >> ss & 1:
                    lut_lo[t * 32 + cc] += a[t * 10 + ss]
                    lut_hi[t * 32 + cc] += a[t * 10 + 5 + ss]
    (d / 'lut_lo.hex').write_text(hexlines(lut_lo, 20))
    (d / 'lut_hi.hex').write_text(hexlines(lut_hi, 20))


def gen_trace_sv(tr):
    """per-trace SV 副本：readmemh 路径指向 t15_synth/<trace>/。"""
    out = {}
    for top, tmpl in (('cert_gate_bitl_synth', 'cert_gate_bitl_synth.sv'),
                      ('fx_gate_synth', 'fx_gate_synth.sv')):
        s = (T15 / tmpl).read_text()
        s = s.replace('$readmemh("t15_synth/', f'$readmemh("t15_synth/{tr}/')
        dst = T15 / tr / f'{top}.sv'
        dst.write_text(s)
        out[top] = dst
    return out


def build(top, sv, tb):
    obj = T15 / f'obj_{top}'
    obj.mkdir(exist_ok=True)
    r = subprocess.run(
        ['verilator', '--cc', '--exe', '-Wno-fatal',
         '--top-module', top, str(sv), str(T15 / tb),
         '-o', 'sim', '--Mdir', str(obj)],
        capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        sys.exit(f'verilator failed for {top}')
    r = subprocess.run(['make', '-C', str(obj), '-f', f'V{top}.mk', '-j', '4'],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        sys.exit(f'make failed for {top}')
    return obj / 'sim'


def run(sim, tr, mode, out):
    stim = 'stim_bf.txt' if mode.startswith('bf') else 'stim_fx.txt'
    r = subprocess.run(
        [str(sim), f'+stim={RES}/t5_rtl/{tr}/{stim}', f'+mode={mode}',
         f'+taudir={RES}/t5_rtl/{tr}', f'+out={out}'],
        capture_output=True, text=True, timeout=1800, cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        sys.exit(f'sim failed {tr}/{mode}')


def main():
    for tr in TRACES:
        gen_trace_hex(tr)
    rows = []
    for tr in TRACES:
        svs = gen_trace_sv(tr)
        sim_cert = build('cert_gate_bitl_synth', svs['cert_gate_bitl_synth'],
                         'tb_cert_synth.cpp')
        sim_fx = build('fx_gate_synth', svs['fx_gate_synth'], 'tb_fx_synth.cpp')
        d5 = RES / 't5_rtl' / tr
        d10 = RES / 't10_rtl' / tr
        d15 = T15 / tr
        exp = np.load(d5 / 'expected.npz')
        dec_exp = exp['dec']
        row = {'trace': tr}
        for mode in ('fx_full', 'fx_cert', 'bf_full', 'bf_cert'):
            run(sim_cert, tr, mode, d15 / f'rtl_{mode}.txt')
            dec, cyc = read_rtl(d15 / f'rtl_{mode}.txt')
            dec_rtl = ((dec[:, None] >> np.arange(10)[None, :]) & 1).astype(np.uint8)
            assert dec_rtl.shape == dec_exp.shape, (tr, mode, dec_rtl.shape)
            mism = int((dec_rtl != dec_exp).sum())
            cross = int((read_rtl(d10 / f'rtl_{mode}.txt')[0] != dec).sum())
            if mode == 'fx_cert':
                cyc_mism = int((cyc != exp['cyc_fx']).sum())
            elif mode == 'bf_cert':
                cyc_mism = int((cyc != exp['cyc_bf']).sum())
            else:
                cyc_mism = 0
                if mode == 'fx_full':
                    assert (cyc == 24).all()
            row[mode] = {'dec_mismatches': mism, 'cycle_mismatches': cyc_mism,
                         'cross_t10_mismatches': cross,
                         'mean_cycles': float(cyc.mean())}
        # 字串行 FX 基线：判决应与 fx_full 同（= expected）
        run(sim_fx, tr, 'fx_word', d15 / 'rtl_fx_word.txt')
        dec, _ = read_rtl(d15 / 'rtl_fx_word.txt')
        dec_rtl = ((dec[:, None] >> np.arange(10)[None, :]) & 1).astype(np.uint8)
        row['fx_word'] = {'dec_mismatches': int((dec_rtl != dec_exp).sum()),
                          'cross_t10_mismatches':
                              int((read_rtl(d10 / 'rtl_fx_full.txt')[0] != dec).sum())}
        rows.append(row)
        print('%-12s dec_mism: fx=%d bf=%d word=%d | cyc_mism: fx=%d bf=%d | cross_t10: %d/%d/%d'
              % (tr, row['fx_cert']['dec_mismatches'], row['bf_cert']['dec_mismatches'],
                 row['fx_word']['dec_mismatches'], row['fx_cert']['cycle_mismatches'],
                 row['bf_cert']['cycle_mismatches'],
                 row['fx_cert']['cross_t10_mismatches'], row['bf_cert']['cross_t10_mismatches'],
                 row['fx_word']['cross_t10_mismatches']))
    ok = all(r[m]['dec_mismatches'] == 0 and r[m]['cycle_mismatches'] == 0
             and r[m]['cross_t10_mismatches'] == 0
             for r in rows for m in ('fx_full', 'fx_cert', 'bf_full', 'bf_cert')) \
        and all(r['fx_word']['dec_mismatches'] == 0
                and r['fx_word']['cross_t10_mismatches'] == 0 for r in rows)
    out = {'traces': rows, 'all_match': bool(ok),
           'note': 'synth variants (thr as 640b port, per-trace baked hex) '
                   'functionally identical to T10-verified RTL and T5 integer model'}
    (T15 / 't15_rtl_verify.json').write_text(json.dumps(out, indent=1))
    print('ALL MATCH' if ok else 'MISMATCH!', '- saved t15_synth/t15_rtl_verify.json')


if __name__ == '__main__':
    main()
