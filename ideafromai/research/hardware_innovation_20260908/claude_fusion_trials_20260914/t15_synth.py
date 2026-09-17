#!/usr/bin/env python3
"""T15：生成综合用常数 hex（a/pn 从 T5 激励复制，LUT 从 a 计算）+ 运行 yosys 综合。

工具链：yowasp-yosys（WASM）+ Nangate45 typical liberty。
已知限制（写进报告）：
- yowasp 的 ABC 子进程 stdout 丢失（含 "Delay =" 行）→ 无延迟数字，无 STA 工具；
  面积/单元数从 yosys 内部日志（-l）解析，两设计同库同流程，面积与周期为结论轴。
- 两口径（对两设计对称施加）：
    nomap：memory -nomap，ROM（lut/pn/a）保留为 $mem_v2 → 面积=数据通路逻辑，
           ROM 按位单列（C1: lut 12.8Kb + pn 960b；FX: a 1.6Kb）；
    logic：memory 全映射，ROM 常数折叠进逻辑 → 面积含烘焙常数，无 ROM。
  实测 C1 logic < nomap（LUT 常数高度可折叠），两口径都报告。
- thr（10×64b/组）两设计对称外置为 640b 宽 SRAM 行口（T13a：层级静态常数）。
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
T15 = ROOT / 't15_synth'
SRC = ROOT / 'results' / 't5_rtl' / 's0_stage0'
YOSYS = 'yowasp-yosys'
LIB = Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07'
           '/third_party/openroad_nangate45/lib/NangateOpenCellLibrary_typical.lib')


def hexlines(arr, bits):
    return '\n'.join(format(int(x) & ((1 << bits) - 1), '0%dx' % (bits // 4))
                     for x in np.asarray(arr).ravel()) + '\n'


def gen_hex():
    a = [int(x, 16) for x in (SRC / 'a.hex').read_text().split()]
    a = [x - (1 << 16) if x >= (1 << 15) else x for x in a]   # signed16
    pn = [int(x, 16) for x in (SRC / 'pn.hex').read_text().split()]
    (T15 / 'a.hex').write_text(hexlines(a, 16))
    (T15 / 'pn.hex').write_text(hexlines(pn, 48))
    lut_lo = np.zeros(320, np.int64)
    lut_hi = np.zeros(320, np.int64)
    for t in range(10):
        for cc in range(32):
            for ss in range(5):
                if cc >> ss & 1:
                    lut_lo[t * 32 + cc] += a[t * 10 + ss]
                    lut_hi[t * 32 + cc] += a[t * 10 + 5 + ss]
    (T15 / 'lut_lo.hex').write_text(hexlines(lut_lo, 20))
    (T15 / 'lut_hi.hex').write_text(hexlines(lut_hi, 20))
    print('hex done: a=100 pn=20 lut=640 entries')


def run_yosys(top, variant, timeout=7200):
    mem = 'memory -nomap' if variant == 'nomap' else 'memory'
    script = f"""
read_verilog -sv t15_synth/{top}.sv
hierarchy -top {top}
proc
{mem}
opt
techmap
opt
dfflibmap -liberty {LIB}
abc -liberty {LIB}
opt_clean
stat -liberty {LIB}
write_json t15_synth/{top}_{variant}.json
"""
    ys = T15 / f'{top}_{variant}.ys'
    log = T15 / f'{top}_{variant}.log'
    ys.write_text(script)
    # 幂等：已有完整日志（含 Chip area）则直接解析，不重跑
    if log.exists() and 'Chip area' in log.read_text():
        return log.read_text()
    log.touch()
    # 注意：$readmemh 路径相对 ROOT，需在 ROOT 下运行
    r = subprocess.run([YOSYS, '-l', str(log), '-s', str(ys)],
                       capture_output=True, text=True, timeout=timeout, cwd=ROOT)
    if r.returncode != 0:
        print(log.read_text()[-3000:])
        sys.exit(f'yosys failed for {top}/{variant}')
    return log.read_text()


def parse_stat(text):
    area = re.search(r'Chip area for module.*?:\s*([\d.]+)', text)
    cells = re.search(r'Number of cells:\s*(\d+)', text)
    dff = re.search(r'\s+(\d+)\s+[\d.+E+]+\s+DFF_X1\b', text)
    return {'cells': int(cells.group(1)) if cells else None,
            'area_um2': float(area.group(1)) if area else None,
            'dff': int(dff.group(1)) if dff else None}


def main():
    gen_hex()
    out = {}
    for top in ('cert_gate_bitl_synth', 'fx_gate_synth'):
        for variant in ('nomap', 'logic'):
            log = run_yosys(top, variant)
            st = parse_stat(log)
            types = re.findall(r'^\s+(\S+)\s+(\d+)\s+[\d.+E+]+\s*$', log, re.M)
            st['top_cells'] = sorted(((t, int(n)) for t, n in types),
                                     key=lambda x: -x[1])[:8]
            out[f'{top}/{variant}'] = st
            print(top, variant, st['cells'], st['area_um2'])
    # plane_ser（生产者端，无 ROM，单口径）
    ps_log = T15 / 'plane_ser_synth.log'
    if ps_log.exists() and 'Chip area' in ps_log.read_text():
        out['plane_ser/nomap'] = parse_stat(ps_log.read_text())
        print('plane_ser', out['plane_ser/nomap'])
    (T15 / 't15_synth_result.json').write_text(json.dumps(out, indent=1))
    print('saved t15_synth/t15_synth_result.json')


if __name__ == '__main__':
    main()
