#!/usr/bin/env bash
# T41：融合基线 RTL 的**逐层**零差回放。判据：
#   [1] 每层各自生成 RTL（gates/L%d_gate.sv）与激励（stim_L%d.txt），Verilator 回放，
#       out_dec 与期望判决**逐组逐比特相同**。
#   [2] 真值来源：t41_equiv.py 已证「证书递归终态判决 == A_q·Y ≥ thr」在 1200 万判决上
#       零失配；本步只证各层 SV 生成物确实实现了该式。
#   ⚠ 层必须配对：拿 L8 的门去回放 L14 的组必然失配（A 逐层不同），故激励按层切开。
# 用法：bash t41_fused/t41_verify.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
D="$ROOT/t41_fused"
VERILATOR=${VERILATOR:-verilator}

/opt/anaconda3/envs/python310/bin/python "$D/t41_gen.py" || exit 1
/opt/anaconda3/envs/python310/bin/python "$D/t41_stim.py" || exit 1

fail=0
for L in 8 14 20 28; do
    rm -rf "$D/obj_L$L"
    "$VERILATOR" --cc --exe --Mdir "$D/obj_L$L" -o sim -Wno-fatal \
        --top-module t41_fused_gate \
        "$D/gates/L${L}_gate.sv" "$D/tb_fused.cpp" >> "$D/build.log" 2>&1 || {
        echo "L$L VERILATOR FAIL"; tail -20 "$D/build.log"; fail=1; continue; }
    make -C "$D/obj_L$L" -f Vt41_fused_gate.mk -j4 >> "$D/build.log" 2>&1 || {
        echo "L$L MAKE FAIL"; tail -20 "$D/build.log"; fail=1; continue; }

    "$D/obj_L$L/sim" "+stim=$D/stim_L$L.txt" "+out=$D/rtl_dec_L$L.txt" || fail=1
    awk '{print $NF}' "$D/stim_L$L.txt" > "$D/exp_dec_L$L.txt"
    if cmp -s "$D/rtl_dec_L$L.txt" "$D/exp_dec_L$L.txt"; then
        echo "L$L ZERO-DIFF PASS"
    else
        echo "L$L DIFF FAIL"; cmp "$D/rtl_dec_L$L.txt" "$D/exp_dec_L$L.txt" | head -3; fail=1
    fi
done

[ "$fail" -eq 0 ] && echo "T41 RTL ZERO-DIFF PASS (4 layers)" || echo "T41 RTL FAIL"
exit "$fail"
