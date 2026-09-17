#!/usr/bin/env python3
"""T39 参考比对：T39 输出 vs T25 k=4 已验证参考。

T39 相对 T25 增加了 sop 拍粗区间判定 + SGLR lane 退役，二者都只应**减少供数**，
不应改变任何判决。故判据是：
  - 判决列 `R <dec>` 逐字节相同（正确性）；
  - 拍数 `fed_T39 <= fed_T25` 逐组成立（供数只减不增）。
用法：python t39_cmp_ref.py <t39_out> <ref_out> <mode>
"""
import sys
from pathlib import Path


def rows(path):
    out = []
    for ln in Path(path).read_text().splitlines():
        if not ln or ln[0] == '#':
            continue
        p = ln.split()
        out.append((int(p[1], 16), int(p[2])))
    return out


def main():
    a = rows(sys.argv[1])
    b = rows(sys.argv[2])
    mode = sys.argv[3]
    if len(a) != len(b):
        print('  %s: **组数不符** %d vs %d' % (mode, len(a), len(b)))
        return 1
    dec_bad = [i for i, (x, y) in enumerate(zip(a, b)) if x[0] != y[0]]
    fed_up = [i for i, (x, y) in enumerate(zip(a, b)) if x[1] > y[1]]
    n = len(a)
    fa = sum(x[1] for x in a) / n
    fb = sum(x[1] for x in b) / n
    print('  %s: dec 不符 %d/%d  拍数上升 %d/%d  |  planes/组 %.4f -> %.4f (%.1f%%)'
          % (mode, len(dec_bad), n, len(fed_up), n, fb, fa, 100 * (fa / fb - 1)))
    ok = not dec_bad and not fed_up
    print('  %s: 判决 == T25 参考 且 拍数只减不增 %s' % (mode, '✓' if ok else '✗'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
