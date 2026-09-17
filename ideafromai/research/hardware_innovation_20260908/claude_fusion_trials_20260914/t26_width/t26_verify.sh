#!/usr/bin/env bash
# T26b：位宽变体的 Verilator 零差回放（对比 T25 k4 参考输出，逐字节 cmp）。
# 用法：bash t26_width/t26_verify.sh [W ...]     默认 48 40 38 32
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STIM=results/t5_rtl/s0_stage0/stim_bf.txt
REF=t25_k4_gate/k4
WS="${*:-48 40 38 32}"
TB="$ROOT/t15_synth/tb_cert_synth.cpp"

for W in $WS; do
  D="$ROOT/t26_width/w${W}_k4"
  OBJ="$D/obj"
  mkdir -p "$OBJ"
  verilator --cc --exe --Mdir "$OBJ" -o sim -Wno-fatal \
      "$D/cert_gate_bitl_synth.sv" "$TB" > "$D/build.log" 2>&1 || {
      echo "W=$W VERILATOR FAIL"; tail -5 "$D/build.log"; continue; }
  make -C "$OBJ" -f Vcert_gate_bitl_synth.mk -j4 >> "$D/build.log" 2>&1 || {
      echo "W=$W MAKE FAIL"; tail -5 "$D/build.log"; continue; }

  ok=1
  for mode in bf_cert bf_full; do
    "$OBJ/sim" "+stim=$STIM" "+mode=$mode" "+taudir=$D" \
        "+out=$D/rtl_${mode}.txt" > "$D/run_${mode}.log" 2>&1 || ok=0
    if cmp -s "$D/rtl_${mode}.txt" "$REF/rtl_${mode}.txt"; then
      echo "  W=$W $mode: IDENTICAL to T25 k4"
    else
      echo "  W=$W $mode: **DIFF**"; ok=0
    fi
  done
  /opt/anaconda3/bin/python t25_k4_gate/t25_check.py \
      "../t26_width/w${W}_k4" "$STIM" bf_cert bf_full 2>&1 | sed 's/^/  /' \
      | grep -v '^  $'
done
