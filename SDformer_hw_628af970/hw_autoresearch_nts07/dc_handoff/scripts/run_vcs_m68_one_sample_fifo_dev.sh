#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "usage: $0 SAMPLE_ID RUN_TAG" >&2
    exit 2
fi

SAMPLE_ID="$1"
RUN_TAG="$2"
if [[ ! "$SAMPLE_ID" =~ ^[0-9]+$ ]] || (( SAMPLE_ID < 0 || SAMPLE_ID > 9 )); then
    echo "FAIL M68 invalid sample id: $SAMPLE_ID" >&2
    exit 2
fi
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "FAIL M68 invalid run tag: $RUN_TAG" >&2
    exit 2
fi

REPO=/home/zhumd/work/sdformer_codex/SDformer
HW="$REPO/hw_autoresearch_nts07"
COMPILE="$HW/dc_handoff/runs/m66_s00_lookahead_exact_sha_compile_r3_20260823"
SIMV="$COMPILE/simv"
INPUT="$HW/results/m68_m66_all10_schedule_streams_dev_r1_20260823"
SID=$(printf '%02d' "$SAMPLE_ID")
STREAM_GZ="$INPUT/m68_s${SID}_schedule.bin.gz"
SCHEDULE="$INPUT/m68_s${SID}_schedule_manifest.json"
REPLAYER="$HW/verif_m66/replay_m66_handshake_ledger.py"
RUN="$HW/results/m68_m66_all10_vcs_dev_${RUN_TAG}_20260823/s${SID}"
STREAM_FIFO="$RUN/schedule_stream.fifo"
LEDGER_FIFO="$RUN/handshake_ledger.fifo"

fail() {
    local message="$1"
    printf '%s\n' FAILED_OR_INCOMPLETE_DO_NOT_CITE > "$RUN/FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt" 2>/dev/null || true
    echo "FAIL M68 $message" >&2
    exit 1
}

[[ -x "$SIMV" ]] || { echo "FAIL M68 missing compiled simv" >&2; exit 1; }
[[ -f "$STREAM_GZ" && -f "$SCHEDULE" ]] || {
    echo "FAIL M68 missing sample-$SID stream or manifest" >&2
    exit 1
}
[[ ! -e "$RUN" ]] || { echo "FAIL M68 refusing run overwrite: $RUN" >&2; exit 1; }

mkdir -p "$RUN"
sha256sum "$SIMV" "$STREAM_GZ" "$SCHEDULE" "$REPLAYER" > "$RUN/prelaunch_input.sha256"
printf '%q ' "$SIMV" "+STREAM=$STREAM_FIFO" "+LEDGER=$LEDGER_FIFO" -assert report > "$RUN/sim.command.txt"
printf '\n' >> "$RUN/sim.command.txt"
date +%s > "$RUN/start_epoch.txt"
mkfifo "$STREAM_FIFO" "$LEDGER_FIFO"

stream_pid=
gzip_pid=
cleanup() {
    if [[ -n "$stream_pid" ]]; then kill "$stream_pid" 2>/dev/null || true; fi
    if [[ -n "$gzip_pid" ]]; then kill "$gzip_pid" 2>/dev/null || true; fi
    rm -f "$STREAM_FIFO" "$LEDGER_FIFO"
}
trap cleanup EXIT

gzip -dc "$STREAM_GZ" > "$STREAM_FIFO" &
stream_pid=$!
gzip -c < "$LEDGER_FIFO" > "$RUN/m68_s${SID}_handshake_ledger.compact.log.gz" &
gzip_pid=$!

set +e
"$SIMV" +STREAM="$STREAM_FIFO" +LEDGER="$LEDGER_FIFO" -assert report \
    > "$RUN/sim.raw.log" 2>&1
sim_rc=$?
wait "$stream_pid"
stream_rc=$?
stream_pid=
wait "$gzip_pid"
gzip_rc=$?
gzip_pid=
set -e
printf '%s\n' "$sim_rc" > "$RUN/sim.rc"
printf '%s\n' "$stream_rc" > "$RUN/stream_gzip.rc"
printf '%s\n' "$gzip_rc" > "$RUN/ledger_gzip.rc"
cleanup
trap - EXIT
date +%s > "$RUN/end_epoch.txt"

[[ "$sim_rc" -eq 0 && "$stream_rc" -eq 0 && "$gzip_rc" -eq 0 ]] || \
    fail "sample-$SID simulation/stream/ledger rc=$sim_rc/$stream_rc/$gzip_rc"
grep -q '^PASS M66 ' "$RUN/sim.raw.log" || fail "sample-$SID missing unique VCS PASS marker"
if grep -Eq '(^|[^A-Za-z])(Fatal:|Error:|Assertion failure|FAIL M66)' "$RUN/sim.raw.log"; then
    fail "sample-$SID failure marker in VCS log"
fi

set +e
python3 "$REPLAYER" \
    --ledger "$RUN/m68_s${SID}_handshake_ledger.compact.log.gz" \
    --schedule-manifest "$SCHEDULE" \
    --output "$RUN/m68_s${SID}_ledger_replay.json" \
    > "$RUN/replay.raw.log" 2>&1
replay_rc=$?
set -e
printf '%s\n' "$replay_rc" > "$RUN/replay.rc"
[[ "$replay_rc" -eq 0 ]] || fail "sample-$SID exact accepted-handshake replay"

sha256sum "$RUN/start_epoch.txt" "$RUN/end_epoch.txt" "$RUN/sim.command.txt" \
    "$RUN/prelaunch_input.sha256" "$RUN/sim.raw.log" "$RUN/sim.rc" \
    "$RUN/stream_gzip.rc" "$RUN/ledger_gzip.rc" \
    "$RUN/m68_s${SID}_handshake_ledger.compact.log.gz" \
    "$RUN/replay.raw.log" "$RUN/replay.rc" "$RUN/m68_s${SID}_ledger_replay.json" \
    > "$RUN/output_manifest.sha256"
printf '%s\n' \
    "PASS_M68_SAMPLE_${SID}_M66_LOOKAHEAD_VCS_REPLAY" \
    'DEVELOPMENT_RECEIPT_ONLY=true' \
    'SYSTEM_SPEEDUP_ADMITTED=false' \
    'PAPER_PPA_READY=false' > "$RUN/RUN_COMPLETE.txt"
echo "PASS M68 sample=$SID run=$RUN"
