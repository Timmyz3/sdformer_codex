#!/usr/bin/env bash
set -euo pipefail
flow_k1_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$flow_k1_root/results/dc"
cd "$flow_k1_root/results/dc"
/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell -f "$flow_k1_root/scripts/synth.tcl" >dc.log 2>&1
test -s summary.json
