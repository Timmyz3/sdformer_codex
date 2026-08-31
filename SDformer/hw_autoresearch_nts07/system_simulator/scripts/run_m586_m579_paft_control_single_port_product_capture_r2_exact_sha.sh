#!/usr/bin/env bash
set -euo pipefail

# M586 source-only runner.  With no argument it performs only the lightweight
# import/spawn probe.  --execute additionally requires a separately reviewed
# execution contract with launch_now=true and max_attempts=1.  This source
# contract does not provide that authorization.

PYTHON_BIN="/opt/anaconda3/envs/python310/bin/python3.10"
PYTHON_SHA="4cd88f501216f7553ce8b80cc4c85c72ca09b0c6f03d62debfa16e8726546b0f"
ANALYZER_REL="system_simulator/scripts/analyze_m579_paft_control_single_port_product_capture_r2.py"
ANALYZER_SHA="70eb07465bb008569967f69ae0ea0d51057d64dd0d51669b604a8f1cd4d4b471"
RESULT_REL="results/m579_paft_control_single_port_product_capture_r2_20260828"
ATTEMPT_REL="results/m579_paft_control_single_port_product_capture_r2_20260828.attempt"

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
HW_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
ANALYZER="$HW_ROOT/$ANALYZER_REL"
RESULT_DIR="$HW_ROOT/$RESULT_REL"
ATTEMPT_DIR="$HW_ROOT/$ATTEMPT_REL"

MODE="preflight"
CONTRACT=""
WORKERS="3"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --preflight-only)
      MODE="preflight"
      shift
      ;;
    --execute)
      MODE="execute"
      shift
      ;;
    --contract)
      [ "$#" -ge 2 ] || { echo "missing --contract value" >&2; exit 2; }
      CONTRACT="$2"
      shift 2
      ;;
    --workers)
      [ "$#" -ge 2 ] || { echo "missing --workers value" >&2; exit 2; }
      WORKERS="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

[ -x "$PYTHON_BIN" ] || { echo "missing frozen Python" >&2; exit 1; }
printf '%s  %s\n' "$PYTHON_SHA" "$PYTHON_BIN" | sha256sum -c - >/dev/null
[ -f "$ANALYZER" ] || { echo "missing M579 r2 analyzer" >&2; exit 1; }
printf '%s  %s\n' "$ANALYZER_SHA" "$ANALYZER" | sha256sum -c - >/dev/null

# Always prove exact interpreter, NumPy, spawn import and a tiny eight-row
# M505 recurrence before considering a production attempt.
"$PYTHON_BIN" "$ANALYZER" --preflight-only
if [ "$MODE" = "preflight" ]; then
  [ -z "$CONTRACT" ] || { echo "preflight must not receive a contract" >&2; exit 2; }
  echo "PASS_M586_M579_R2_SOURCE_PREFLIGHT_ONLY__NO_RESULT_OR_ATTEMPT"
  exit 0
fi

[ -n "$CONTRACT" ] || { echo "--execute requires --contract" >&2; exit 2; }
case "$WORKERS" in
  1|2|3) ;;
  *) echo "workers must be 1..3" >&2; exit 2 ;;
esac
CONTRACT="$(realpath -e -- "$CONTRACT")"

# The analyzer strictly checks launch authorization, every frozen input, both
# manifests, all 80 packed payloads, M504, M255 and M528.  This runs before an
# attempt directory is consumed and processes zero formal trace records.
"$PYTHON_BIN" "$ANALYZER" --contract "$CONTRACT" --validate-contract-only

# Runner/output coordinates and this exact runner SHA are checked separately.
"$PYTHON_BIN" - "$CONTRACT" "$0" "$RESULT_REL" "$ATTEMPT_REL" <<'PY'
import hashlib, json, pathlib, sys

contract_path = pathlib.Path(sys.argv[1])
runner_path = pathlib.Path(sys.argv[2]).resolve()
result_rel = sys.argv[3]
attempt_rel = sys.argv[4]

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise RuntimeError("duplicate JSON key: " + key)
        out[key] = value
    return out

contract = json.loads(contract_path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda raw: (_ for _ in ()).throw(RuntimeError(raw)))
if contract["runner_sha256"] != sha(runner_path):
    raise RuntimeError("runner SHA drift")
if contract["output"]["result_dir"] != result_rel:
    raise RuntimeError("result coordinate drift")
if contract["output"]["attempt_dir"] != attempt_rel:
    raise RuntimeError("attempt coordinate drift")
PY

[ ! -e "$RESULT_DIR" ] || { echo "formal result already exists" >&2; exit 1; }
[ ! -e "$ATTEMPT_DIR" ] || { echo "formal attempt already exists/consumed" >&2; exit 1; }
[ ! -e "${ATTEMPT_DIR}.consumed" ] || { echo "formal attempt was already consumed" >&2; exit 1; }

STAGING_DIR="${RESULT_DIR}.staging.$$"
[ ! -e "$STAGING_DIR" ] || { echo "staging collision" >&2; exit 1; }
mkdir "$ATTEMPT_DIR"
printf 'M586_M579_R2_ATTEMPT_CONSUMED pid=%s contract=%s\n' "$$" "$CONTRACT" \
  > "$ATTEMPT_DIR/ATTEMPT_CONSUMED.txt"

SUCCESS=0
cleanup() {
  rc=$?
  trap - EXIT INT TERM HUP
  if [ "$SUCCESS" -ne 1 ]; then
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    if [ -e "$STAGING_DIR" ]; then
      mv -- "$STAGING_DIR" "${RESULT_DIR}.failed_or_incomplete.${stamp}.$$"
    fi
    if [ -e "$ATTEMPT_DIR" ]; then
      mv -- "$ATTEMPT_DIR" "${ATTEMPT_DIR}.failed_or_incomplete.${stamp}.$$"
    fi
  fi
  exit "$rc"
}
trap cleanup EXIT INT TERM HUP

"$PYTHON_BIN" "$ANALYZER" \
  --contract "$CONTRACT" \
  --output-dir "$STAGING_DIR" \
  --workers "$WORKERS" \
  > "$ATTEMPT_DIR/production_stdout.log" \
  2> "$ATTEMPT_DIR/production_stderr.log"

[ -f "$STAGING_DIR/m579_paft_control_single_port_product_capture_r2.json" ]
[ -f "$STAGING_DIR/m579_per_sample_cycles_r2.csv" ]
cp -- "$ATTEMPT_DIR/production_stdout.log" "$STAGING_DIR/production_stdout.log"
cp -- "$ATTEMPT_DIR/production_stderr.log" "$STAGING_DIR/production_stderr.log"

# Terminally rehash the execution contract, analyzer, every declared input,
# both manifests, all 80 packed payloads and docs359 before sealing.
"$PYTHON_BIN" "$ANALYZER" \
  --contract "$CONTRACT" \
  --output-dir "$STAGING_DIR" \
  --terminal-rehash \
  > "$STAGING_DIR/terminal_rehash_receipt.json.tmp" \
  2> "$ATTEMPT_DIR/terminal_rehash_stderr.log"
mv -- "$STAGING_DIR/terminal_rehash_receipt.json.tmp" \
  "$STAGING_DIR/terminal_rehash_receipt.json"

(
  cd "$STAGING_DIR"
  sha256sum \
    m579_paft_control_single_port_product_capture_r2.json \
    m579_per_sample_cycles_r2.csv \
    production_stdout.log \
    production_stderr.log \
    terminal_rehash_receipt.json \
    > SHA256SUMS
  sha256sum SHA256SUMS > SHA256SUMS.seal.sha256
  sha256sum -c SHA256SUMS >/dev/null
  sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
)

# Recheck immutable executables after the long run and refuse a racing target.
printf '%s  %s\n' "$PYTHON_SHA" "$PYTHON_BIN" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$ANALYZER_SHA" "$ANALYZER" | sha256sum -c - >/dev/null

# Linux renameat2(RENAME_NOREPLACE) closes the final-target TOCTOU window: an
# empty or nonempty racing target both fail without overwriting or nesting.
"$PYTHON_BIN" - "$STAGING_DIR" "$RESULT_DIR" <<'PY'
import ctypes, os, sys

source = os.fsencode(sys.argv[1])
target = os.fsencode(sys.argv[2])
libc = ctypes.CDLL(None, use_errno=True)
renameat2 = libc.renameat2
renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                      ctypes.c_uint]
renameat2.restype = ctypes.c_int
if renameat2(-100, source, -100, target, 1) != 0:  # AT_FDCWD, RENAME_NOREPLACE
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error), sys.argv[2])
PY
SUCCESS=1
trap - EXIT INT TERM HUP

# The sealed result is now authoritative.  Consume the attempt with the same
# no-replace primitive.  If bookkeeping collides, leave the original attempt
# in place (which still blocks a rerun) and report a warning; never disturb the
# already atomically published result.
if ! "$PYTHON_BIN" - "$ATTEMPT_DIR" "${ATTEMPT_DIR}.consumed" <<'PY'
import ctypes, os, sys

source = os.fsencode(sys.argv[1])
target = os.fsencode(sys.argv[2])
libc = ctypes.CDLL(None, use_errno=True)
renameat2 = libc.renameat2
renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
                      ctypes.c_uint]
renameat2.restype = ctypes.c_int
if renameat2(-100, source, -100, target, 1) != 0:
    error = ctypes.get_errno()
    raise OSError(error, os.strerror(error), sys.argv[2])
PY
then
  echo "WARNING_RESULT_PUBLISHED_ATTEMPT_REMAINS_CONSUMED_IN_PLACE $ATTEMPT_DIR" >&2
fi
echo "PASS_M586_M579_R2_ATOMIC_RESULT $RESULT_DIR"
