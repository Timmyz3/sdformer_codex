#!/usr/bin/env bash
set -euo pipefail
probe_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
for mode in dynamic forced; do
  force=0
  if [[ "$mode" == forced ]]; then force=1; fi
  mkdir -p "$probe_root/results/dc_$mode"
  (
    cd "$probe_root/results/dc_$mode"
    FLOW_ISSUE_FORCE_FULL="$force" dc_shell -f "$probe_root/scripts/synth.tcl" >dc.log 2>&1
    test -s summary.json
  )
done
