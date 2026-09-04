#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/dc_handoff/runs/storage_ablation"
mkdir -p "$OUT"
cd "$ROOT"

run_one() {
  local design="$1"
  local filelist="$2"
  local active_depth="$3"
  local hist_depth="$4"
  local tag="$5"
  local rtl_files
  rtl_files="$(tr '\n' ' ' < "$filelist")"
  yosys -Q -p "read_verilog -sv ${rtl_files}; chparam -set ACTIVE_MEM_DEPTH ${active_depth} -set SCORE_CLASS_DEPTH ${hist_depth} ${design}; hierarchy -check -top ${design}; synth -flatten -top ${design}; opt_clean; check -assert; tee -o ${OUT}/${design}_${tag}.json stat -json" \
    > "$OUT/${design}_${tag}.log"
}

run_one h67_attention_top rtl_h67/filelist.f 162 35 exact
run_one h67_attention_top rtl_h67/filelist.f 256 64 padded
run_one h68_castling_deploy_top rtl_h68/filelist.f 162 3 exact
run_one h68_castling_deploy_top rtl_h68/filelist.f 256 4 padded
python3 dc_handoff/scripts/summarize_storage_ablation.py
