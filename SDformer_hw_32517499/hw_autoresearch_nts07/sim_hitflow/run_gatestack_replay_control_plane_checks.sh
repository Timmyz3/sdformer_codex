#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build_hitflow/gatestack_replay_control_plane"
LINTER="${ERIE_LINTER:-/root/.codex/skills/erie-verilog-generator/scripts/verilog_lint.py}"
mkdir -p "$BUILD"
cd "$ROOT"

RTL=(
  rtl_hitflow/gatestack_replay_plan_builder.sv
  rtl_hitflow/gatestack_replay_atomic_commit.sv
  rtl_hitflow/gatestack_dualtag_replay_lifecycle_manager.sv
  rtl_hitflow/gatestack_replay_control_plane_top.sv
)
SVA=(
  verif_hitflow/gatestack_replay_plan_builder_assertions.sv
  verif_hitflow/bind_gatestack_replay_plan_builder_assertions.sv
  verif_hitflow/gatestack_replay_atomic_commit_assertions.sv
  verif_hitflow/bind_gatestack_replay_atomic_commit_assertions.sv
  verif_hitflow/gatestack_dualtag_replay_lifecycle_assertions.sv
  verif_hitflow/bind_gatestack_dualtag_replay_lifecycle_assertions.sv
  verif_hitflow/gatestack_replay_control_plane_assertions.sv
  verif_hitflow/bind_gatestack_replay_control_plane_assertions.sv
)

iverilog -g2012 -Wall -s tb_gatestack_replay_control_plane_top \
  -o "$BUILD/tb.vvp" "${RTL[@]}" \
  tb_hitflow/tb_gatestack_replay_control_plane_top.sv
vvp "$BUILD/tb.vvp" | tee "$BUILD/iverilog.log"

verilator --binary --timing --assert -Wall \
  --top-module tb_gatestack_replay_control_plane_top \
  -Mdir "$BUILD/verilator_obj" \
  "${RTL[@]}" "${SVA[@]}" \
  tb_hitflow/tb_gatestack_replay_control_plane_top.sv \
  >"$BUILD/verilator_build.log" 2>&1
"$BUILD/verilator_obj/Vtb_gatestack_replay_control_plane_top" | \
  tee "$BUILD/verilator.log"

yosys -q -l "$BUILD/yosys.log" -p \
  "read_verilog -sv ${RTL[*]}; hierarchy -check -top gatestack_replay_control_plane_top; proc; opt; check; stat"
: >"$BUILD/erie_lint.log"
for rtl_file in "${RTL[@]}"; do
  python3 "$LINTER" "$rtl_file" >>"$BUILD/erie_lint.log" 2>&1
done

if grep -Eq '%Warning|%Error' "$BUILD/verilator_build.log"; then
  echo "FAIL: Verilator warning/error" >&2
  exit 1
fi
echo "PASS: GateStack replay control plane；Verilator/Erie 0 warning/error"
