#!/usr/bin/env bash
set -euo pipefail
hw_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
for lanes in 4 16; do
 mkdir -p "$hw_root/results/dc_lanes$lanes"
 (cd "$hw_root/results/dc_lanes$lanes"; FLOW_DOT_LANES=$lanes dc_shell -f "$hw_root/scripts/synth.tcl" >dc.log 2>&1; test -s summary.json)
done
