#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$ROOT/.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT/runs/m30b_atlif_t2_packed24_vcs_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_DIR"
if [[ -e "$RUN_DIR/simv" || -e "$RUN_DIR/compile.log" || -e "$RUN_DIR/sim.log" ]]; then
    echo "refusing to overwrite M30B VCS run: $RUN_DIR" >&2
    exit 2
fi
cd "$HW_ROOT"
sha256sum \
    rtl_m30/qfit_atlif_t2_packed24_stream_core.sv \
    verif_m30/qfit_atlif_t2_packed24_stream_assertions.sv \
    tb_m30/tb_qfit_atlif_t2_packed24_stream_core.sv \
    dc_handoff/filelists/date_m30b_atlif_t2_packed24_vcs.f \
    dc_handoff/scripts/run_vcs_m30b_atlif_t2_packed24_sva.sh \
    > "$RUN_DIR/input_sha256.txt"
vcs -full64 -sverilog -assert svaext -debug_access+pp \
    -Mdir="$RUN_DIR/csrc" \
    +define+SIMULATOR_VCS +define+SVA_RUNTIME_ENABLED \
    -timescale=1ns/1ps \
    -f dc_handoff/filelists/date_m30b_atlif_t2_packed24_vcs.f \
    -top tb_qfit_atlif_t2_packed24_stream_core \
    -o "$RUN_DIR/simv" 2>&1 | tee "$RUN_DIR/compile.log"
"$RUN_DIR/simv" 2>&1 | tee "$RUN_DIR/sim.log"
grep -q 'M30B_PASS' "$RUN_DIR/sim.log"
grep -q 'SIMULATOR=Synopsys VCS' "$RUN_DIR/sim.log"
grep -q 'ASSERTIONS=enabled' "$RUN_DIR/sim.log"
grep -q 'M30B_SVA_BOUND=1' "$RUN_DIR/sim.log"
if grep -Eiq 'assertion[^[:cntrl:]]*(fail|error)|offending.*assert' \
    "$RUN_DIR/sim.log"; then
    echo "M30B assertion failure signature found" >&2
    exit 3
fi
sha256sum "$RUN_DIR/compile.log" "$RUN_DIR/sim.log" \
    > "$RUN_DIR/output_sha256.txt"
