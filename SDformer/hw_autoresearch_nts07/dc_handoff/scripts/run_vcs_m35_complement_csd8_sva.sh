#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$ROOT/.." && pwd)"
RUN_DIR="${RUN_DIR:-$ROOT/runs/m35_complement_csd8_vcs_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RUN_DIR"
if [[ -e "$RUN_DIR/simv" || -e "$RUN_DIR/compile.log" \
        || -e "$RUN_DIR/sim.log" || -e "$RUN_DIR/vectors.txt" ]]; then
    echo "refusing to overwrite M35 VCS run: $RUN_DIR" >&2
    exit 2
fi
cd "$HW_ROOT"
M35_MATH_CONTRACT_SHA256="2d453166e0ded2a7783699fb35877aeb3b2d19e8a3906d43700cace1839ca28e"
M35_MATH_RESULT_SHA256="c47121f7d9b9fef15f4f1d770c4944d0bef9f640e5c9e4d522e6529742687869"
test "$(sha256sum contracts/m35_complement_csd_input_contract_r3_20260822.json | awk '{print $1}')" \
    = "$M35_MATH_CONTRACT_SHA256"
test "$(sha256sum results/m35_complement_csd_r3_20260822/m35_complement_csd.json | awk '{print $1}')" \
    = "$M35_MATH_RESULT_SHA256"
sha256sum \
    rtl_m35/qfit_complement_csd8_late_scale.sv \
    verif_m35/qfit_complement_csd8_late_scale_assertions.sv \
    tb_m35/tb_qfit_complement_csd8_late_scale.sv \
    dc_handoff/filelists/date_m35_complement_csd8_vcs.f \
    dc_handoff/scripts/run_vcs_m35_complement_csd8_sva.sh \
    contracts/m35_complement_csd_input_contract_r3_20260822.json \
    results/m35_complement_csd_r3_20260822/m35_complement_csd.json \
    > "$RUN_DIR/input_sha256.txt"
vcs -full64 -sverilog -assert svaext -debug_access+pp \
    -Mdir="$RUN_DIR/csrc" \
    +define+SIMULATOR_VCS +define+SVA_RUNTIME_ENABLED \
    -timescale=1ns/1ps \
    -f dc_handoff/filelists/date_m35_complement_csd8_vcs.f \
    -top tb_qfit_complement_csd8_late_scale \
    -o "$RUN_DIR/simv" 2>&1 | tee "$RUN_DIR/compile.log"
"$RUN_DIR/simv" +M35_VECTOR_FILE="$RUN_DIR/vectors.txt" \
    2>&1 | tee "$RUN_DIR/sim.log"
grep -Eq '^M35_PASS packets=5120 valid_products=23680 config_loads=10 config_releases=10 stalls=[0-9]+ consecutive_full_rate=630 masks_all=1 illegal_accepts=2 illegal_rejections=2 illegal_sum_rejections=1 illegal_shift_rejections=1 busy_release_rejects=[1-9][0-9]* descriptor_pin_perturbations=105$' "$RUN_DIR/sim.log"
grep -q 'SIMULATOR=Synopsys VCS' "$RUN_DIR/sim.log"
grep -q 'ASSERTIONS=enabled' "$RUN_DIR/sim.log"
grep -q 'M35_SVA_BOUND=1' "$RUN_DIR/sim.log"
grep -q 'M35_RANDOM_SEED=0x4d350102' "$RUN_DIR/sim.log"
grep -q 'masks_all=1' "$RUN_DIR/sim.log"
if grep -Eq ', [0-9]+ attempts, 0 match$' "$RUN_DIR/sim.log"; then
    echo "M35 uncovered SVA cover property" >&2
    exit 5
fi
if grep -Eiq 'assertion[^[:cntrl:]]*(fail|error)|offending.*assert' \
        "$RUN_DIR/sim.log"; then
    echo "M35 assertion failure signature found" >&2
    exit 3
fi
M35_GOLDEN_VECTOR_SHA256="49dbb64c0757c4a6ee14fecffe8e9bacca5e84af7ee53bdc1cfd112a3c960565"
M35_VECTOR_SHA256="$(sha256sum "$RUN_DIR/vectors.txt" | awk '{print $1}')"
if [[ "$M35_VECTOR_SHA256" != "$M35_GOLDEN_VECTOR_SHA256" ]]; then
    echo "M35 vector SHA256 mismatch: expected $M35_GOLDEN_VECTOR_SHA256 got $M35_VECTOR_SHA256" >&2
    exit 4
fi
sha256sum "$RUN_DIR/compile.log" "$RUN_DIR/sim.log" \
    "$RUN_DIR/vectors.txt" > "$RUN_DIR/output_sha256.txt"
