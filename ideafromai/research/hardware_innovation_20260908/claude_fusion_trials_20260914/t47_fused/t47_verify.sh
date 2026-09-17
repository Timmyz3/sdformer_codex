#!/usr/bin/env bash
# T47：两版 RTL（零插值行走器 Z / 相位分解行走器 P）的零差回放。
# 判据：三个用例（真实权重+真实脉冲 / 全 1 / 随机 30%）下，
#       Z 与 P 写出的 ymem 都必须与 Python 金标**逐元素相同**，且 Z 与 P 互相同。
# 用法：bash t47_fused/t47_verify.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
D="$ROOT/t47_fused"
STIM="$ROOT/t47_stim"
VERILATOR=${VERILATOR:-verilator}
PY=/opt/anaconda3/envs/pytorch310/bin/python

"$PY" "$ROOT/t47_polyphase_rtl.py" || exit 1

rm -rf "$D/obj"
"$VERILATOR" --cc --exe --Mdir "$D/obj" -o sim -Wno-fatal \
    --top-module t47_walk_top \
    "$D/t47_deconv_walker.sv" "$D/t47_tb.cpp" > "$D/t47_build.log" 2>&1 || {
    echo "VERILATOR FAIL"; tail -30 "$D/t47_build.log"; exit 1; }
make -C "$D/obj" -f Vt47_walk_top.mk -j4 >> "$D/t47_build.log" 2>&1 || {
    echo "MAKE FAIL"; tail -30 "$D/t47_build.log"; exit 1; }

fail=0
for C in real ones rand; do
    "$D/obj/sim" +xm="$STIM/x_$C.txt" +wm="$STIM/w_$C.txt" \
        +exp="$STIM/exp_$C.txt" +out="$D/t47_out_$C.txt" > "$D/t47_run_$C.log" 2>&1
    rc=$?
    cat "$D/t47_run_$C.log"
    if [ "$rc" -ne 0 ]; then echo "$C ZERO-DIFF FAIL"; fail=1; else echo "$C ZERO-DIFF PASS"; fi
done

[ "$fail" -eq 0 ] && echo "T47 RTL ZERO-DIFF PASS (3 cases, Z & P)" || echo "T47 RTL FAIL"
exit "$fail"
