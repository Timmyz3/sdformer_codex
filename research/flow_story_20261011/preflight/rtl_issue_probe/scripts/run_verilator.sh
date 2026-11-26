#!/usr/bin/env bash
set -euo pipefail
probe_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$probe_root"
for mode in dynamic forced; do
  force=0
  if [[ "$mode" == forced ]]; then force=1; fi
  mkdir -p "results/$mode"
  verilator --cc rtl/risk_group_issue.sv --exe "$probe_root/tb/issue_tb.cpp" \
    --top-module risk_group_issue -GFORCE_FULL="$force" \
    -Mdir "$probe_root/results/obj_$mode" \
    -CFLAGS '-std=c++11 -O2' >"results/$mode/build.log" 2>&1
  make -C "results/obj_$mode" -f Vrisk_group_issue.mk -j 2 \
    >>"results/$mode/build.log" 2>&1
  "results/obj_$mode/Vrisk_group_issue" "$mode" "$probe_root/results/$mode" \
    >"results/$mode/test.log" 2>&1
done
