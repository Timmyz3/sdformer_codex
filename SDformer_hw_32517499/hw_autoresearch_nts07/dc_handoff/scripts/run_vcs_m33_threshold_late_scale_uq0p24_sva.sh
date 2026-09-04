#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$ROOT/.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT/runs/m33_threshold_late_scale_uq_vcs_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_DIR"
if [[ -e "$RUN_DIR/simv" || -e "$RUN_DIR/compile.log" \
        || -e "$RUN_DIR/sim.log" || -e "$RUN_DIR/vectors.txt" ]]; then
    echo "refusing to overwrite M33 UQ VCS run: $RUN_DIR" >&2
    exit 2
fi
cd "$HW_ROOT"
sha256sum \
    rtl_m31/qfit_signed_int8_mul96_pool.sv \
    rtl_m33/qfit_threshold_late_scale_uq0p24_radix20x4.sv \
    verif_m33/qfit_threshold_late_scale_uq0p24_radix20x4_assertions.sv \
    tb_m33/tb_qfit_threshold_late_scale_uq0p24_radix20x4.sv \
    dc_handoff/filelists/date_m33_threshold_late_scale_uq0p24_vcs.f \
    dc_handoff/scripts/run_vcs_m33_threshold_late_scale_uq0p24_sva.sh \
    > "$RUN_DIR/input_sha256.txt"
vcs -full64 -sverilog -assert svaext -debug_access+pp \
    -Mdir="$RUN_DIR/csrc" \
    +define+SIMULATOR_VCS +define+SVA_RUNTIME_ENABLED \
    -timescale=1ns/1ps \
    -f dc_handoff/filelists/date_m33_threshold_late_scale_uq0p24_vcs.f \
    -top tb_qfit_threshold_late_scale_uq0p24_radix20x4 \
    -o "$RUN_DIR/simv" 2>&1 | tee "$RUN_DIR/compile.log"
"$RUN_DIR/simv" \
    +M33_UQ_VECTOR_FILE="$RUN_DIR/vectors.txt" \
    2>&1 | tee "$RUN_DIR/sim.log"
grep -q 'M33_UQ_PASS' "$RUN_DIR/sim.log"
grep -q 'SIMULATOR=Synopsys VCS' "$RUN_DIR/sim.log"
grep -q 'ASSERTIONS=enabled' "$RUN_DIR/sim.log"
grep -q 'M33_UQ_SVA_BOUND=1' "$RUN_DIR/sim.log"
grep -q 'M33_UQ_RANDOM_SEED=0x4d333402' "$RUN_DIR/sim.log"
grep -q 'masks=ffff' "$RUN_DIR/sim.log"
if grep -Eiq 'assertion[^[:cntrl:]]*(fail|error)|offending.*assert' \
        "$RUN_DIR/sim.log"; then
    echo "M33 UQ assertion failure signature found" >&2
    exit 3
fi
task_vector_sha="$(sha256sum "$RUN_DIR/vectors.txt" | awk '{print $1}')"
if [[ "$task_vector_sha" != \
        e1a326a06904776c5f9da076ee4261b90bb6650d98a5ccb93c80e989afe15b53 ]]; then
    echo "M33 UQ deterministic vector digest drift: $task_vector_sha" >&2
    exit 4
fi
sha256sum "$RUN_DIR/compile.log" "$RUN_DIR/sim.log" \
    "$RUN_DIR/vectors.txt" > "$RUN_DIR/output_sha256.txt"
