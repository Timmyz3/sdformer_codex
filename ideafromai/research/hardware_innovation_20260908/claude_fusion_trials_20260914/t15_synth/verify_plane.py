#!/usr/bin/env python3
"""T15 生产者端校验：plane_ser RTL 由 fx 平面激励重建的 Yq 词再生 bf 激励，
与 T5 生成的 stim_bf.txt 逐行比对（生产者→门核供数回路闭合）。"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T15 = ROOT / 't15_synth'
RES = ROOT / 'results'
TRACES = ['s0_stage0', 's0_stage3', 's10_stage0', 's10_stage3']


def build():
    obj = T15 / 'obj_plane_ser'
    obj.mkdir(exist_ok=True)
    r = subprocess.run(
        ['verilator', '--cc', '--exe', '-Wno-fatal',
         '--top-module', 'plane_ser', str(T15 / 'plane_ser.sv'),
         str(T15 / 'tb_plane_ser.cpp'), '-o', 'sim', '--Mdir', str(obj)],
        capture_output=True, text=True, timeout=1200)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        sys.exit('verilator failed for plane_ser')
    r = subprocess.run(['make', '-C', str(obj), '-f', 'Vplane_ser.mk', '-j', '4'],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        sys.exit('make failed for plane_ser')
    return obj / 'sim'


def main():
    sim = build()
    for tr in TRACES:
        out = T15 / tr / 'rtl_plane_ser_bf.txt'
        r = subprocess.run(
            [str(sim), f'+stim={RES}/t5_rtl/{tr}/stim_fx.txt', f'+out={out}'],
            capture_output=True, text=True, timeout=1800, cwd=ROOT)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            sys.exit(f'sim failed {tr}')
        ref = (RES / 't5_rtl' / tr / 'stim_bf.txt').read_text()
        got = out.read_text()
        n_ref, n_got = len(ref.splitlines()), len(got.splitlines())
        match = ref == got
        print('%-12s lines ref=%d got=%d %s'
              % (tr, n_ref, n_got, 'EXACT MATCH' if match else 'DIFF'))
        if not match:
            sys.exit(f'plane_ser replay mismatch for {tr}')
    print('plane_ser: all 4 traces regenerate stim_bf exactly')


if __name__ == '__main__':
    main()
