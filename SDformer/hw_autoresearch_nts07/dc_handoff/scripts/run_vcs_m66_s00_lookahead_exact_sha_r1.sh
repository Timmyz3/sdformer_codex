#!/usr/bin/env bash
set -euo pipefail

REPO=/home/zhumd/work/sdformer_codex/SDformer
HW="$REPO/hw_autoresearch_nts07"
COMPILE="$HW/dc_handoff/runs/m66_s00_lookahead_exact_sha_compile_r3_20260823"
RUN="$HW/results/m66_h67_k4c16_temporal_vcs_s00_lookahead_exact_sha_r1_20260823"
CONTRACT="$HW/contracts/m66_s00_lookahead_exact_sha_vcs_contract_r1_20260823.json"
CORE="$HW/rtl_m66/qfit_k4_parent_delta_p8_l96_ctx16_lookahead.sv"
BRIDGE="$HW/rtl_m66/qfit_m66_m53_schedule_bridge_lookahead.sv"
M54_SVA="$HW/verif_m54/qfit_k4_parent_delta_p8_l96_ctx16_assertions.sv"
M66_SVA="$HW/verif_m66/qfit_k4_parent_delta_lookahead_assertions.sv"
TB="$HW/tb_m66/tb_m66_m53_schedule_bridge_lookahead.sv"
FILELIST="$HW/dc_handoff/filelists/date_m66_m53_schedule_bridge_lookahead_vcs.f"
STREAM="$HW/dc_handoff/runs/m57_diagnostics_20260823/s00_sim_r2/input.bin"
SCHEDULE="$HW/results/m57_h67_k4c16_temporal_vcs_r1_20260823/m57_s00_schedule_manifest.json"
REPLAYER="$HW/verif_m66/replay_m66_handshake_ledger.py"
BUILDER="$HW/verif_m66/build_m66_s00_exact_sha_receipt.py"
VALIDATOR="$HW/verif_m66/validate_m66_s00_exact_sha_receipt.py"
M57_REPLAY="$HW/results/m57_h67_k4c16_temporal_vcs_s00_phase_safe_full_compact_r3_20260823/m57_s00_ledger_replay.json"
M57_RECEIPT="$HW/results/m57_h67_k4c16_temporal_vcs_s00_phase_safe_full_compact_r3_20260823/m57_s00_phase_safe_exact_sha_vcs_receipt.json"
VCS=/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs
FIFO="$RUN/ledger.fifo"

if [[ -e "$COMPILE" || -e "$RUN" ]]; then
    echo "FAIL M66 refusing exact-run overwrite" >&2
    exit 1
fi

check_sha() {
    local expected="$1"
    local path="$2"
    local observed
    observed=$(sha256sum "$path" | awk '{print $1}')
    if [[ "$observed" != "$expected" ]]; then
        echo "FAIL M66 SHA drift path=$path expected=$expected observed=$observed" >&2
        exit 1
    fi
}

check_sha b9a2064ab73764534415f2dc54aa134807a147c6b8528f0fb041e3afc5d13f4d "$CORE"
check_sha d1020823c328c528c5e9693cc85bd973667e143a335de2fa7a1f081f19e7c7af "$BRIDGE"
check_sha 1338421c3ee3d12f70fb2b2299e76d6651c297500920b1ffb70989c90cc2a267 "$M54_SVA"
check_sha e522c849411ab89e59037825764410e617cc642a158d3a488472272131fb3973 "$M66_SVA"
check_sha 67d6f76182c1566ffbda9274cdbc0f01cbca19a34668290f3c086e4730c32771 "$TB"
check_sha 1a6bea2c3bc7b9a83fa69b875739f21bcb896021bc4cddcbd4089dbea311af03 "$FILELIST"
check_sha 496706ce20dd685bbb913523d8da6e44eee6ed2c836c557d634f5f75bc45a63a "$STREAM"
check_sha 7e93928600e0ceeddf2e2103de66c7d065260e98a5845d44c0618d26c3c4c125 "$SCHEDULE"
check_sha bb7d2e3b600226e1ec09498ce64d035db6f1d4ad92fb641edfe04331362fdd1e "$REPLAYER"
check_sha 1860e0f5e58d397da5ed88ebf35ab89acf11f704e2a13c6fe7e64e80338524f5 "$BUILDER"
check_sha 2076618dfad618bcc9682c76e63b7a52b8f7a10c1e23794ba6f941e04393cc46 "$VALIDATOR"
check_sha 6ff1f3101ae9d0c1a2331e428d133e17397005294ff54b2b16fc1caa31afec9b "$M57_REPLAY"
check_sha ad65e91ed45f171870ef6718079f4d25806111ef7004622c673f6fdbad9cbbf7 "$M57_RECEIPT"
check_sha 0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287 "$VCS"

mkdir -p "$COMPILE"
sha256sum "$CONTRACT" "$CORE" "$BRIDGE" "$M54_SVA" "$M66_SVA" "$TB" \
    "$FILELIST" "$STREAM" "$SCHEDULE" "$REPLAYER" "$BUILDER" "$VALIDATOR" \
    "$M57_REPLAY" "$M57_RECEIPT" "$VCS" > "$COMPILE/precompile_input.sha256"

COMPILE_COMMAND="$VCS -full64 -sverilog -assert svaext -f $FILELIST -top tb_m66_m53_schedule_bridge_lookahead -Mdir=$COMPILE/csrc -o $COMPILE/simv -l $COMPILE/compile.raw.log"
printf '%s\n' "$COMPILE_COMMAND" > "$COMPILE/compile.command.txt"
set +e
(
    cd "$HW"
    "$VCS" -full64 -sverilog -assert svaext -f "$FILELIST" \
        -top tb_m66_m53_schedule_bridge_lookahead \
        -Mdir="$COMPILE/csrc" -o "$COMPILE/simv" \
        -l "$COMPILE/compile.raw.log"
) > "$COMPILE/compile.console.log" 2>&1
compile_rc=$?
set -e
printf '%s\n' "$compile_rc" > "$COMPILE/compile.rc"
if [[ "$compile_rc" -ne 0 || ! -x "$COMPILE/simv" ]]; then
    printf '%s\n' FAILED_OR_INCOMPLETE_DO_NOT_CITE > "$COMPILE/FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt"
    echo "FAIL M66 VCS compile rc=$compile_rc" >&2
    exit 1
fi
sha256sum "$COMPILE/simv" "$COMPILE/compile.raw.log" "$COMPILE/compile.command.txt" \
    "$COMPILE/precompile_input.sha256" > "$COMPILE/compile_output.sha256"

mkdir -p "$RUN"
SIM_COMMAND="$COMPILE/simv +STREAM=$STREAM +LEDGER=$FIFO -assert report"
printf '%s\n' "$SIM_COMMAND" > "$RUN/sim.command.txt"
sha256sum "$CONTRACT" "$COMPILE/precompile_input.sha256" "$COMPILE/simv" \
    "$COMPILE/compile.raw.log" "$COMPILE/compile.command.txt" "$STREAM" "$SCHEDULE" \
    "$REPLAYER" "$BUILDER" "$VALIDATOR" > "$RUN/prelaunch_input.sha256"
date +%s > "$RUN/start_epoch.txt"
mkfifo "$FIFO"
exec 9<> "$FIFO"

cleanup_fifo() {
    exec 9>&- 2>/dev/null || true
    if [[ -p "$FIFO" ]]; then rm -f "$FIFO"; fi
}
trap cleanup_fifo EXIT

gzip -c < "$FIFO" 9>&- > "$RUN/m66_s00_handshake_ledger.compact.log.gz" &
gzip_pid=$!
set +e
"$COMPILE/simv" +STREAM="$STREAM" +LEDGER="$FIFO" -assert report 9>&- \
    2>&1 | tee "$RUN/sim.raw.log"
sim_rc=${PIPESTATUS[0]}
exec 9>&-
wait "$gzip_pid"
gzip_rc=$?
set -e
printf '%s\n' "$sim_rc" > "$RUN/sim.rc"
printf '%s\n' "$gzip_rc" > "$RUN/gzip.rc"
cleanup_fifo
trap - EXIT
date +%s > "$RUN/end_epoch.txt"
if [[ "$sim_rc" -ne 0 || "$gzip_rc" -ne 0 ]]; then
    printf '%s\n' FAILED_OR_INCOMPLETE_DO_NOT_CITE > "$RUN/FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt"
    echo "FAIL M66 simulation/gzip rc=$sim_rc/$gzip_rc" >&2
    exit 1
fi

set +e
python3 "$REPLAYER" --ledger "$RUN/m66_s00_handshake_ledger.compact.log.gz" \
    --schedule-manifest "$SCHEDULE" --output "$RUN/m66_s00_ledger_replay.json" \
    > "$RUN/replay.raw.log" 2>&1
replay_rc=$?
set -e
printf '%s\n' "$replay_rc" > "$RUN/replay.rc"
if [[ "$replay_rc" -ne 0 ]]; then
    printf '%s\n' FAILED_OR_INCOMPLETE_DO_NOT_CITE > "$RUN/FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt"
    echo "FAIL M66 ledger replay" >&2
    exit 1
fi

python3 "$BUILDER" --repo "$REPO" --output "$RUN/m66_s00_exact_sha_vcs_receipt.json" \
    > "$RUN/receipt_builder.raw.log" 2>&1
set +e
python3 "$VALIDATOR" --repo "$REPO" --receipt "$RUN/m66_s00_exact_sha_vcs_receipt.json" \
    > "$RUN/validator.raw.log" 2>&1
validator_rc=$?
set -e
printf '%s\n' "$validator_rc" > "$RUN/validator.rc"
if [[ "$validator_rc" -ne 0 ]]; then
    printf '%s\n' FAILED_OR_INCOMPLETE_DO_NOT_CITE > "$RUN/FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt"
    cat "$RUN/validator.raw.log" >&2
    exit 1
fi

sha256sum "$RUN/start_epoch.txt" "$RUN/end_epoch.txt" "$RUN/sim.command.txt" \
    "$RUN/prelaunch_input.sha256" "$RUN/sim.raw.log" "$RUN/sim.rc" "$RUN/gzip.rc" \
    "$RUN/m66_s00_handshake_ledger.compact.log.gz" "$RUN/replay.raw.log" "$RUN/replay.rc" \
    "$RUN/m66_s00_ledger_replay.json" "$RUN/receipt_builder.raw.log" \
    "$RUN/m66_s00_exact_sha_vcs_receipt.json" "$RUN/validator.raw.log" "$RUN/validator.rc" \
    > "$RUN/output_manifest.sha256"
printf '%s\n' \
    'PASS_M66_EXACT_SHA_FULL_S00_LOOKAHEAD_VCS_REPLAY' \
    'M57_RTL_CYCLES=8791654' \
    'M66_RTL_CYCLES=8117392' \
    'CYCLES_SAVED=674262' \
    'SAME_TRACE_KERNEL_SPEEDUP=1.083063870761' \
    'SYSTEM_SPEEDUP_ADMITTED=false' \
    'ONLINE_SCHEDULER_ADMITTED=false' \
    'PAPER_PPA_READY=false' > "$RUN/RUN_COMPLETE.txt"
chmod 0444 "$COMPILE/precompile_input.sha256" "$COMPILE/compile.command.txt" \
    "$COMPILE/compile.console.log" "$COMPILE/compile.raw.log" "$COMPILE/compile.rc" \
    "$COMPILE/compile_output.sha256" "$COMPILE/simv"
find "$RUN" -type f -exec chmod 0444 {} +
chmod 0555 "$RUN"
echo "PASS M66 official exact-SHA run=$RUN"
