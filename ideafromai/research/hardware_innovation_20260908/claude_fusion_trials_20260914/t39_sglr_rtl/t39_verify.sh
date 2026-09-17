#!/usr/bin/env bash
# T39：SGLR per-lane ready + sop 拍粗区间判定的 Verilator 零差回放（含消融）。
# 两个变体：
#   A = cert_gate_sglr.sv    （SGLR + sop 粗区间判定，T39 设计点）
#   B = cert_gate_sglr_np.sv （仅 SGLR，消融）
# 判据：
#   [1] gated（只送 need 里的 lane）与非 gated（全 lane）输出**逐字节相同**
#       —— 两个变体都须成立：SGLR 不改变任何判决、也不改变拍数。
#   [2] A/B 的判决列与 T25 k=4 已验证参考逐字节相同（T25 参考本身已与全深度整数模型零差）。
#   [3] 拍数只减不增：A 的 fed ≤ T25 参考；B 的 fed == T25 参考（只退役 lane 不改拍）。
#   [4] 逐组整数模型复现 A（--preplane）与 B（--prefix-np）的 dec/fed/bits 逐组逐比特。
# 用法：bash t39_sglr_rtl/t39_verify.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
D="$ROOT/t39_sglr_rtl"
STIM=results/t5_rtl/s0_stage0/stim_bf.txt
REF=t25_k4_gate/k4
export PYTHONPATH="$ROOT/t25_k4_gate"

/opt/anaconda3/bin/python t39_sglr_rtl/t39_gen.py || exit 1

build () {  # build <sv> <objdir>   （TB 固定 include Vcert_gate_sglr.h，故统一文件名）
  local stage="$D/stage_$2"
  rm -rf "$stage"; mkdir -p "$stage" "$D/$2"
  cp "$1" "$stage/cert_gate_sglr.sv"
  verilator --cc --exe --Mdir "$D/$2" -o sim -Wno-fatal \
      "$stage/cert_gate_sglr.sv" "$D/tb_sglr.cpp" \
      >> "$D/build.log" 2>&1 || { echo "VERILATOR FAIL ($1)"; tail -20 "$D/build.log"; return 1; }
  make -C "$D/$2" -f Vcert_gate_sglr.mk -j4 >> "$D/build.log" 2>&1 || {
      echo "MAKE FAIL ($1)"; tail -20 "$D/build.log"; return 1; }
}

build "$D/cert_gate_sglr.sv"    obj    || exit 1
build "$D/cert_gate_sglr_np.sv" obj_np || exit 1

fail=0
for mode in bf_cert bf_full; do
  for G in 0 1; do
    "$D/obj/sim" "+stim=$STIM" "+mode=$mode" "+taudir=$REF" "+gated=$G" \
        "+out=$D/rtl_${mode}_g${G}.txt" "+stats=$D/stats_${mode}_g${G}.txt" \
        2>&1 | sed 's/^/  [A] /'
  done
  if cmp -s "$D/rtl_${mode}_g0.txt" "$D/rtl_${mode}_g1.txt"; then
    echo "  [A] $mode: gated == ungated (逐字节) ✓"
  else
    echo "  [A] $mode: **gated != ungated**"; fail=1
  fi
  /opt/anaconda3/bin/python t39_sglr_rtl/t39_cmp_ref.py \
      "$D/rtl_${mode}_g1.txt" "$REF/rtl_${mode}.txt" "[A] $mode" || fail=1
done

# 变体 B（仅 SGLR）：cert 模式，拍数须与 T25 参考**完全相等**
for G in 0 1; do
  "$D/obj_np/sim" "+stim=$STIM" "+mode=bf_cert" "+taudir=$REF" "+gated=$G" \
      "+out=$D/rtl_np_bf_cert_g${G}.txt" "+stats=$D/stats_np_bf_cert_g${G}.txt" \
      2>&1 | sed 's/^/  [B] /'
done
if cmp -s "$D/rtl_np_bf_cert_g0.txt" "$D/rtl_np_bf_cert_g1.txt"; then
  echo "  [B] bf_cert: gated == ungated (逐字节) ✓"
else
  echo "  [B] bf_cert: **gated != ungated**"; fail=1
fi
if cmp -s "$D/rtl_np_bf_cert_g1.txt" "$REF/rtl_bf_cert.txt"; then
  echo "  [B] bf_cert: 与 T25 k4 参考逐字节相同（拍数不变，仅退役 lane）✓"
else
  echo "  [B] bf_cert: **与 T25 参考不符**"; fail=1
fi

# [4] 逐组整数模型
/opt/anaconda3/bin/python t39_sglr_rtl/t39_model.py --preplane --cert-only || fail=1
/opt/anaconda3/bin/python t39_sglr_rtl/t39_model.py --prefix-np --cert-only || fail=1
[ $fail -eq 0 ] && echo 'T39 ZERO-DIFF PASS' || echo 'T39 FAIL'
exit $fail
