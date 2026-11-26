#!/usr/bin/env bash
set -euo pipefail
r33_hw="$(cd "$(dirname "$0")" && pwd)"
for r33_kind in dynamic static none; do
 r33_dc="$r33_hw/results/dc_$r33_kind"
 mkdir -p "$r33_dc"
 cd "$r33_dc"
 if ! FLOW_TOP="bitmap_ctrl_$r33_kind" FLOW_SOURCE="$r33_hw/rtl/bitmap_controller.sv" /opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell -f "$r33_hw/rtl/synth_ctrl.tcl" > run.log 2>&1; then cp run.log "$r33_hw/results/failures/${r33_kind}_dc.log"; exit 1; fi
 if [ ! -f summary.json ]; then cp run.log "$r33_hw/results/failures/${r33_kind}_dc_missing_summary.log"; exit 1; fi
 cat summary.json
done
