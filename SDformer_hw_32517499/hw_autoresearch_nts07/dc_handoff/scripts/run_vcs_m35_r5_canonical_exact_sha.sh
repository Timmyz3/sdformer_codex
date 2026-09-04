#!/usr/bin/env bash
set -euo pipefail

DC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HW_ROOT="$(cd "$DC_ROOT/.." && pwd)"
DEFAULT_RUN_ROOT="/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/runs"
RUN_ROOT="${M35_R5_RUN_ROOT:-$DEFAULT_RUN_ROOT}"
RUN_DIR="${RUN_DIR:-$RUN_ROOT/m35_r5_canonical_exact_sha_vcs_$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ -e "$RUN_DIR" ]]; then
    echo "refusing to overwrite M35-r5 VCS run: $RUN_DIR" >&2
    exit 2
fi
mkdir -p "$RUN_DIR/snapshot/inputs"
cd "$HW_ROOT"

RTL_SHA="84b1f3cb6344863ecfdbac2af8abcfdd15b1f16571979588badbc3e2e0dd1854"
REVIEW_SHA="8b0978b3158d780a0d5acee4ac0a780c32349e1dd45c1722f1421cb01b86fb6f"
VALIDATOR_SHA="305f7ff80090fcd6fd2a957e4a3f07d8b0c53219c392719e243973942674d2e8"
CONTRACT_SHA="28f4c9a8b6b9c28d1e10bf47397fb6da104e6d64f14d53a3410cd09c034b5ac6"
TB_SHA="8a6c32c770d37de53076d39b21b0dc08de7d8b3762879252fc8818e92694454b"
SVA_SHA="f561add1deadefa01b4ee984c45098f09bfca5defdbfb903d105548aa08c12d8"
FILELIST_SHA="529a11620761cfb5b02419032c0bff07fabb1f70491467a7e6e0ae99106b0192"
MITER_SHA="da2518cc0512f3a0864546bb2d68f27c6d1a82592f8dc4c68711d13bbc6ce992"
RECEIPT_BUILDER_SHA="620874ef3918fa5d21090e0f3745483a003e9607366f38e76789817b89894222"

declare -a INPUTS=(
    "rtl_m35_r4/qfit_complement_csd8_canonical.sv"
    "verif_m35_r5/qfit_complement_csd8_canonical_assertions.sv"
    "tb_m35_r5/tb_qfit_complement_csd8_canonical.sv"
    "dc_handoff/filelists/date_m35_r5_canonical_vcs.f"
    "dc_handoff/scripts/run_vcs_m35_r5_canonical_exact_sha.sh"
    "dc_handoff/scripts/miter_m35_r5_vcs_trace.py"
    "dc_handoff/scripts/build_m35_r5_vcs_receipt.py"
    "contracts/m35_canonical_descriptor_contract_r4_20260822.json"
    "results/m35_r4_canonical_descriptor_model_20260822/m35_r4_canonical_descriptor_audit.json"
    "results/m35_r4_independent_hammer_review_20260822/m35_r4_independent_hammer_review.json"
    "results/m35_r4_independent_hammer_review_20260822/validate_m35_r4_independent_hammer_review.py"
)

test "$(sha256sum "${INPUTS[0]}" | awk '{print $1}')" = "$RTL_SHA"
test "$(sha256sum "${INPUTS[1]}" | awk '{print $1}')" = "$SVA_SHA"
test "$(sha256sum "${INPUTS[2]}" | awk '{print $1}')" = "$TB_SHA"
test "$(sha256sum "${INPUTS[3]}" | awk '{print $1}')" = "$FILELIST_SHA"
test "$(sha256sum "${INPUTS[5]}" | awk '{print $1}')" = "$MITER_SHA"
test "$(sha256sum "${INPUTS[6]}" | awk '{print $1}')" = "$RECEIPT_BUILDER_SHA"
test "$(sha256sum "${INPUTS[7]}" | awk '{print $1}')" = "$CONTRACT_SHA"
test "$(sha256sum "${INPUTS[9]}" | awk '{print $1}')" = "$REVIEW_SHA"
test "$(sha256sum "${INPUTS[10]}" | awk '{print $1}')" = "$VALIDATOR_SHA"

for relative in "${INPUTS[@]}"; do
    mkdir -p "$RUN_DIR/snapshot/inputs/$(dirname "$relative")"
    cp -p "$relative" "$RUN_DIR/snapshot/inputs/$relative"
done
(
    cd "$RUN_DIR/snapshot/inputs"
    sha256sum "${INPUTS[@]}"
) > "$RUN_DIR/input_sha256.txt"

set +e
/usr/bin/python3.6 \
    results/m35_r4_independent_hammer_review_20260822/validate_m35_r4_independent_hammer_review.py \
    >"$RUN_DIR/review_validation.stdout.log" \
    2>"$RUN_DIR/review_validation.stderr.log"
review_rc=$?
set -e
printf '%d\n' "$review_rc" > "$RUN_DIR/review_validation.exit_status"
if [[ "$review_rc" -ne 0 ]]; then
    echo "M35-r5 independent review validation failed: rc=$review_rc" >&2
    exit "$review_rc"
fi

set +e
vcs -full64 -ID >"$RUN_DIR/tool_version.stdout.log" \
    2>"$RUN_DIR/tool_version.stderr.log"
version_rc=$?
set -e
if [[ "$version_rc" -ne 0 ]]; then
    echo "M35-r5 VCS version query failed: rc=$version_rc" >&2
    exit "$version_rc"
fi

set +e
vcs -full64 -sverilog -assert svaext -debug_access+pp \
    -Mdir="$RUN_DIR/csrc" \
    +define+SIMULATOR_VCS +define+SVA_RUNTIME_ENABLED \
    -timescale=1ns/1ps \
    -f dc_handoff/filelists/date_m35_r5_canonical_vcs.f \
    -top tb_qfit_complement_csd8_canonical \
    -o "$RUN_DIR/simv" \
    >"$RUN_DIR/compile.stdout.log" \
    2>"$RUN_DIR/compile.stderr.log"
compile_rc=$?
set -e
printf '%d\n' "$compile_rc" > "$RUN_DIR/compile.exit_status"
if [[ "$compile_rc" -ne 0 ]]; then
    echo "M35-r5 VCS compile failed: rc=$compile_rc run=$RUN_DIR" >&2
    exit "$compile_rc"
fi

set +e
"$RUN_DIR/simv" -no_save \
    +M35_R5_TRACE_FILE="$RUN_DIR/handshake_trace.csv" \
    >"$RUN_DIR/sim.stdout.log" \
    2>"$RUN_DIR/sim.stderr.log"
sim_rc=$?
set -e
printf '%d\n' "$sim_rc" > "$RUN_DIR/sim.exit_status"
if [[ "$sim_rc" -ne 0 ]]; then
    echo "M35-r5 VCS simulation failed: rc=$sim_rc run=$RUN_DIR" >&2
    exit "$sim_rc"
fi

grep -q '^M35_R5_PASS .*mismatches=0$' "$RUN_DIR/sim.stdout.log"
grep -q '^M35_R5_SIMULATOR=Synopsys VCS$' "$RUN_DIR/sim.stdout.log"
grep -q '^M35_R5_ASSERTIONS=enabled$' "$RUN_DIR/sim.stdout.log"
grep -q '^M35_R5_SVA_BOUND=1$' "$RUN_DIR/sim.stdout.log"
grep -q '^M35_R5_RANDOM_SEED=0x4d350105$' "$RUN_DIR/sim.stdout.log"
if grep -Eq ', [0-9]+ attempts, 0 match$' "$RUN_DIR/sim.stdout.log"; then
    echo "M35-r5 uncovered SVA cover property" >&2
    exit 5
fi
if grep -Eiq 'assertion[^[:cntrl:]]*(fail|error)|offending.*assert' \
        "$RUN_DIR/sim.stdout.log" "$RUN_DIR/sim.stderr.log"; then
    echo "M35-r5 assertion failure signature found" >&2
    exit 3
fi

set +e
/usr/bin/python3.6 dc_handoff/scripts/miter_m35_r5_vcs_trace.py \
    --trace "$RUN_DIR/handshake_trace.csv" \
    --output "$RUN_DIR/m35_r5_trace_miter.json" \
    >"$RUN_DIR/miter.stdout.log" \
    2>"$RUN_DIR/miter.stderr.log"
miter_rc=$?
set -e
printf '%d\n' "$miter_rc" > "$RUN_DIR/miter.exit_status"
if [[ "$miter_rc" -ne 0 ]]; then
    echo "M35-r5 independent trace miter failed: rc=$miter_rc run=$RUN_DIR" >&2
    exit "$miter_rc"
fi

/usr/bin/python3.6 dc_handoff/scripts/build_m35_r5_vcs_receipt.py \
    --run-dir "$RUN_DIR" \
    --output "$RUN_DIR/m35_r5_vcs_receipt.json" \
    >"$RUN_DIR/receipt_builder.stdout.log" \
    2>"$RUN_DIR/receipt_builder.stderr.log"

(
    cd "$RUN_DIR"
    sha256sum \
        input_sha256.txt \
        review_validation.stdout.log review_validation.stderr.log \
        review_validation.exit_status \
        tool_version.stdout.log tool_version.stderr.log \
        compile.stdout.log compile.stderr.log compile.exit_status \
        sim.stdout.log sim.stderr.log sim.exit_status simv \
        handshake_trace.csv \
        miter.stdout.log miter.stderr.log miter.exit_status \
        m35_r5_trace_miter.json \
        receipt_builder.stdout.log receipt_builder.stderr.log \
        m35_r5_vcs_receipt.json \
        > output_sha256.txt
    sha256sum input_sha256.txt output_sha256.txt \
        m35_r5_vcs_receipt.json > seal_sha256.txt
)
chmod -R a-w "$RUN_DIR"
printf 'PASS M35-r5 exact-SHA VCS run: %s\n' "$RUN_DIR"
