#!/usr/bin/env python3
"""VCD 名称规范化：Verilator 写成 "name [9:0]"，Vivado 只认 "name[9:0]"。

只改 $var 声明行（其余字节原样流式复制，含值变化段）。
用法：python fix_vcd.py in.vcd out.vcd
"""
import re
import sys


def main():
    src, dst = sys.argv[1], sys.argv[2]
    n_fixed = 0
    with open(src, 'r', errors='replace') as f, open(dst, 'w') as g:
        for ln in f:
            if ln.lstrip().startswith('$var'):
                tk = ln.split()
                name = ' '.join(tk[4:-1])          # 去掉尾部 $end；range 与名字间有空格
                name2 = re.sub(r'\s+\[', '[', name)
                if name2 != name:
                    n_fixed += 1
                    ln = f"$var {tk[1]} {tk[2]} {tk[3]} {name2} $end\n"
            g.write(ln)
    print(f'fixed {n_fixed} $var names -> {dst}')


if __name__ == '__main__':
    main()
