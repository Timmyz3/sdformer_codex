#!/usr/bin/env bash
set -euo pipefail

# M601/M579 r4 source-only runner.  The default is a lightweight import/spawn
# probe.  --execute requires a separately reviewed v4 execution contract.  R4
# changes only canonical-path handling: lexists, explicit no-symlink, and the
# same RENAME_NOREPLACE publication used by M594 r3.

PYTHON_BIN="/opt/anaconda3/envs/python310/bin/python3.10"
PYTHON_SHA="4cd88f501216f7553ce8b80cc4c85c72ca09b0c6f03d62debfa16e8726546b0f"
ANALYZER_REL="system_simulator/scripts/analyze_m579_paft_control_single_port_product_capture_r4.py"
ANALYZER_SHA="ba8fc0326b4d17f45d6db156d89b29da0894560d70d82f65ea5ae5f40b115195"
RESULT_REL="results/m579_paft_control_single_port_product_capture_r4_20260828"
ATTEMPT_REL="results/m579_paft_control_single_port_product_capture_r4_20260828.attempt"

path_lexists() {
  [ -e "$1" ] || [ -L "$1" ]
}

require_absent() {
  if path_lexists "$1"; then
    echo "$2 already exists, including dangling symlink" >&2
    return 1
  fi
}

require_regular_nosymlink() {
  path_lexists "$1" || { echo "missing $2" >&2; return 1; }
  [ ! -L "$1" ] || { echo "$2 must not be a symlink" >&2; return 1; }
  [ -f "$1" ] || { echo "$2 must be a regular file" >&2; return 1; }
}

require_directory_nosymlink() {
  path_lexists "$1" || { echo "missing $2" >&2; return 1; }
  [ ! -L "$1" ] || { echo "$2 must not be a symlink" >&2; return 1; }
  [ -d "$1" ] || { echo "$2 must be a directory" >&2; return 1; }
}

require_tree_nosymlink() {
  [ -z "$(find "$1" -type l -print -quit)" ] || {
    echo "$2 contains a symlink" >&2
    return 1
  }
}

[ ! -L "$0" ] || { echo "runner must not be invoked through a symlink" >&2; exit 1; }
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
HW_ROOT="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
RUNNER="$(realpath -e -- "$0")"
ANALYZER="$HW_ROOT/$ANALYZER_REL"
RESULT_DIR="$HW_ROOT/$RESULT_REL"
ATTEMPT_DIR="$HW_ROOT/$ATTEMPT_REL"

sha_file() {
  sha256sum -- "$1" | awk '{print $1}'
}

rename_noreplace() {
  "$PYTHON_BIN" - "$1" "$2" <<'PY'
import ctypes
import os
import sys

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
}

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

require_regular_nosymlink "$PYTHON_BIN" "frozen Python"
printf '%s  %s\n' "$PYTHON_SHA" "$PYTHON_BIN" | sha256sum -c - >/dev/null
require_regular_nosymlink "$ANALYZER" "M579 r4 analyzer"
printf '%s  %s\n' "$ANALYZER_SHA" "$ANALYZER" | sha256sum -c - >/dev/null
require_regular_nosymlink "$RUNNER" "M579 r4 runner"

# Lightweight preflight creates no execution contract, result or attempt.
"$PYTHON_BIN" "$ANALYZER" --preflight-only
if [ "$MODE" = "preflight" ]; then
  [ -z "$CONTRACT" ] || { echo "preflight must not receive a contract" >&2; exit 2; }
  echo "PASS_M601_M579_R4_SOURCE_PREFLIGHT_ONLY__NO_RESULT_OR_ATTEMPT"
  exit 0
fi

[ -n "$CONTRACT" ] || { echo "--execute requires --contract" >&2; exit 2; }
case "$WORKERS" in
  1|2|3) ;;
  *) echo "workers must be 1..3" >&2; exit 2 ;;
esac
require_regular_nosymlink "$CONTRACT" "M601 r4 execution contract"
CONTRACT="$(realpath -e -- "$CONTRACT")"
require_regular_nosymlink "$CONTRACT" "resolved M601 r4 execution contract"
CONTRACT_SHA_START="$(sha_file "$CONTRACT")"
RUNNER_SHA_START="$(sha_file "$RUNNER")"

# Full validation remains pre-attempt and processes zero formal records.
"$PYTHON_BIN" "$ANALYZER" \
  --contract "$CONTRACT" \
  --expected-contract-sha256 "$CONTRACT_SHA_START" \
  --expected-runner-sha256 "$RUNNER_SHA_START" \
  --validate-contract-only

[ "$(sha_file "$CONTRACT")" = "$CONTRACT_SHA_START" ] || { echo "contract changed after validation" >&2; exit 1; }
[ "$(sha_file "$RUNNER")" = "$RUNNER_SHA_START" ] || { echo "runner changed after validation" >&2; exit 1; }
printf '%s  %s\n' "$PYTHON_SHA" "$PYTHON_BIN" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$ANALYZER_SHA" "$ANALYZER" | sha256sum -c - >/dev/null

# Freeze output coordinates before attempt consumption.
"$PYTHON_BIN" - "$CONTRACT" "$RUNNER_SHA_START" "$RESULT_REL" "$ATTEMPT_REL" <<'PY'
import json
import pathlib
import sys

contract_path = pathlib.Path(sys.argv[1])
runner_sha = sys.argv[2]
result_rel = sys.argv[3]
attempt_rel = sys.argv[4]

def pairs(items):
    out = {}
    for key, value in items:
        if key in out:
            raise RuntimeError("duplicate JSON key: " + key)
        out[key] = value
    return out

contract = json.loads(
    contract_path.read_text(encoding="utf-8"),
    object_pairs_hook=pairs,
    parse_constant=lambda raw: (_ for _ in ()).throw(RuntimeError(raw)),
)
if contract["schema"] != "m579_paft_control_single_port_product_capture_execution_contract_v4":
    raise RuntimeError("execution schema drift")
if contract["runner_sha256"] != runner_sha:
    raise RuntimeError("runner SHA drift")
if contract["output"]["result_dir"] != result_rel:
    raise RuntimeError("result coordinate drift")
if contract["output"]["attempt_dir"] != attempt_rel:
    raise RuntimeError("attempt coordinate drift")
PY

require_absent "$RESULT_DIR" "formal result"
require_absent "$ATTEMPT_DIR" "formal attempt"
require_absent "${ATTEMPT_DIR}.consumed" "consumed formal attempt"
if compgen -G "${ATTEMPT_DIR}.quarantine.staging.*" >/dev/null \
   || compgen -G "${ATTEMPT_DIR}.failed_or_incomplete.*" >/dev/null; then
  echo "pre-existing quarantine coordinate blocks the one-attempt execution" >&2
  exit 1
fi
STAGING_DIR="${RESULT_DIR}.staging.$$"
require_absent "$STAGING_DIR" "formal staging"

# Install handlers before the first canonical mutation.
SUCCESS=0
STAGE="before_attempt_mkdir"
SIGNAL_CAUGHT="none"
cleanup() {
  rc=$?
  trap - EXIT INT TERM HUP
  if [ "$SUCCESS" -ne 1 ] && { path_lexists "$ATTEMPT_DIR" || path_lexists "$STAGING_DIR"; }; then
    stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    quarantine_stage="${ATTEMPT_DIR}.quarantine.staging.${stamp}.$$"
    quarantine_final="${ATTEMPT_DIR}.failed_or_incomplete.${stamp}.$$"
    require_absent "$quarantine_stage" "quarantine staging"
    require_absent "$quarantine_final" "quarantine final"
    mkdir -- "$quarantine_stage"
    if path_lexists "$ATTEMPT_DIR"; then
      rename_noreplace "$ATTEMPT_DIR" "$quarantine_stage/attempt"
    fi
    if path_lexists "$STAGING_DIR"; then
      rename_noreplace "$STAGING_DIR" "$quarantine_stage/staging"
    fi
    "$PYTHON_BIN" - \
      "$quarantine_stage" "$rc" "$STAGE" "$SIGNAL_CAUGHT" \
      "$CONTRACT" "$CONTRACT_SHA_START" "$RUNNER" "$RUNNER_SHA_START" \
      "$ANALYZER" "$ANALYZER_SHA" "$RESULT_DIR" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
contract = pathlib.Path(sys.argv[5])
runner = pathlib.Path(sys.argv[7])
analyzer = pathlib.Path(sys.argv[9])
result = pathlib.Path(sys.argv[11])

def observed(path):
    if not os.path.lexists(os.fspath(path)) or path.is_symlink() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

receipt = {
    "schema": "m601_m579_r4_failed_attempt_quarantine_v1",
    "status": "FAILED_OR_INTERRUPTED_ATTEMPT_QUARANTINED",
    "exit_code": int(sys.argv[2]),
    "failure_stage": sys.argv[3],
    "signal": sys.argv[4],
    "contract_path": str(contract),
    "contract_sha256_start": sys.argv[6],
    "contract_sha256_at_quarantine": observed(contract),
    "contract_lexists_at_quarantine": os.path.lexists(os.fspath(contract)),
    "contract_is_symlink_at_quarantine": contract.is_symlink(),
    "contract_bytes_unchanged": observed(contract) == sys.argv[6],
    "runner_path": str(runner),
    "runner_sha256_start": sys.argv[8],
    "runner_sha256_at_quarantine": observed(runner),
    "runner_lexists_at_quarantine": os.path.lexists(os.fspath(runner)),
    "runner_is_symlink_at_quarantine": runner.is_symlink(),
    "runner_bytes_unchanged": observed(runner) == sys.argv[8],
    "analyzer_path": str(analyzer),
    "analyzer_sha256_expected": sys.argv[10],
    "analyzer_sha256_at_quarantine": observed(analyzer),
    "analyzer_lexists_at_quarantine": os.path.lexists(os.fspath(analyzer)),
    "analyzer_is_symlink_at_quarantine": analyzer.is_symlink(),
    "analyzer_bytes_unchanged": observed(analyzer) == sys.argv[10],
    "final_result_lexists": os.path.lexists(os.fspath(result)),
    "final_result_is_symlink": result.is_symlink(),
    "final_result_is_directory": result.is_dir() and not result.is_symlink(),
}
temporary = directory / ".failure_receipt.tmp"
temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(directory / "failure_receipt.json")
PY
    require_tree_nosymlink "$quarantine_stage" "failure quarantine"
    (
      cd "$quarantine_stage"
      find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 -print0 \
        | LC_ALL=C sort -z \
        | xargs -0 sha256sum > SHA256SUMS
      sha256sum SHA256SUMS > SHA256SUMS.seal.sha256
      sha256sum -c SHA256SUMS >/dev/null
      sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
    )
    rename_noreplace "$quarantine_stage" "$quarantine_final"
    require_directory_nosymlink "$quarantine_final" "published failure quarantine"
  fi
  exit "$rc"
}
on_signal() {
  SIGNAL_CAUGHT="$1"
  exit "$2"
}
trap cleanup EXIT
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap 'on_signal HUP 129' HUP

STAGE="attempt_mkdir"
mkdir -- "$ATTEMPT_DIR"
require_directory_nosymlink "$ATTEMPT_DIR" "new formal attempt"
STAGE="attempt_marker"
"$PYTHON_BIN" - "$ATTEMPT_DIR" "$$" "$CONTRACT" "$CONTRACT_SHA_START" "$RUNNER_SHA_START" <<'PY'
import json
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
payload = {
    "schema": "m601_m579_r4_attempt_v1",
    "status": "ATTEMPT_CONSUMED",
    "pid": int(sys.argv[2]),
    "contract_path": sys.argv[3],
    "contract_sha256_start": sys.argv[4],
    "runner_sha256_start": sys.argv[5],
    "canonical_path_policy": "LEXISTS_AND_REJECT_SYMLINK_THEN_RENAME_NOREPLACE",
}
temporary = directory / ".attempt.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(directory / "ATTEMPT_CONSUMED.json")
PY

STAGE="production"
"$PYTHON_BIN" "$ANALYZER" \
  --contract "$CONTRACT" \
  --expected-contract-sha256 "$CONTRACT_SHA_START" \
  --expected-runner-sha256 "$RUNNER_SHA_START" \
  --output-dir "$STAGING_DIR" \
  --workers "$WORKERS" \
  > "$ATTEMPT_DIR/production_stdout.log" \
  2> "$ATTEMPT_DIR/production_stderr.log"

require_directory_nosymlink "$STAGING_DIR" "r4 staging after production"
require_regular_nosymlink "$STAGING_DIR/m579_paft_control_single_port_product_capture_r4.json" "r4 staging result"
require_regular_nosymlink "$STAGING_DIR/m579_per_sample_cycles_r4.csv" "r4 staging sample CSV"
[ "$(sha_file "$CONTRACT")" = "$CONTRACT_SHA_START" ]
[ "$(sha_file "$RUNNER")" = "$RUNNER_SHA_START" ]

cp -- "$ATTEMPT_DIR/production_stdout.log" "$STAGING_DIR/production_stdout.log"
cp -- "$ATTEMPT_DIR/production_stderr.log" "$STAGING_DIR/production_stderr.log"

STAGE="terminal_rehash"
"$PYTHON_BIN" "$ANALYZER" \
  --contract "$CONTRACT" \
  --expected-contract-sha256 "$CONTRACT_SHA_START" \
  --expected-runner-sha256 "$RUNNER_SHA_START" \
  --output-dir "$STAGING_DIR" \
  --terminal-rehash \
  > "$STAGING_DIR/terminal_rehash_receipt.json.tmp" \
  2> "$ATTEMPT_DIR/terminal_rehash_stderr.log"
mv -- "$STAGING_DIR/terminal_rehash_receipt.json.tmp" \
  "$STAGING_DIR/terminal_rehash_receipt.json"
cp -- "$ATTEMPT_DIR/terminal_rehash_stderr.log" \
  "$STAGING_DIR/terminal_rehash_stderr.log"

STAGE="seal_result"
require_tree_nosymlink "$STAGING_DIR" "result staging"
(
  cd "$STAGING_DIR"
  sha256sum \
    m579_paft_control_single_port_product_capture_r4.json \
    m579_per_sample_cycles_r4.csv \
    production_stdout.log \
    production_stderr.log \
    terminal_rehash_receipt.json \
    terminal_rehash_stderr.log \
    > SHA256SUMS
  sha256sum SHA256SUMS > SHA256SUMS.seal.sha256
  sha256sum -c SHA256SUMS >/dev/null
  sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
)

STAGE="pre_publish_identity"
require_regular_nosymlink "$CONTRACT" "execution contract at pre-publish"
require_regular_nosymlink "$RUNNER" "runner at pre-publish"
require_regular_nosymlink "$ANALYZER" "analyzer at pre-publish"
[ "$(sha_file "$CONTRACT")" = "$CONTRACT_SHA_START" ]
[ "$(sha_file "$RUNNER")" = "$RUNNER_SHA_START" ]
printf '%s  %s\n' "$PYTHON_SHA" "$PYTHON_BIN" | sha256sum -c - >/dev/null
printf '%s  %s\n' "$ANALYZER_SHA" "$ANALYZER" | sha256sum -c - >/dev/null
require_absent "$RESULT_DIR" "formal result immediately before publish"

STAGE="publish_result_noreplace"
rename_noreplace "$STAGING_DIR" "$RESULT_DIR"
require_directory_nosymlink "$RESULT_DIR" "published formal result"

STAGE="seal_success_attempt"
require_tree_nosymlink "$ATTEMPT_DIR" "successful attempt"
"$PYTHON_BIN" - "$ATTEMPT_DIR" "$CONTRACT_SHA_START" "$RUNNER_SHA_START" <<'PY'
import json
import pathlib
import sys

directory = pathlib.Path(sys.argv[1])
payload = {
    "schema": "m601_m579_r4_attempt_completion_v1",
    "status": "RESULT_PUBLISHED_ATTEMPT_READY_TO_CONSUME",
    "contract_sha256_start": sys.argv[2],
    "runner_sha256_start": sys.argv[3],
    "canonical_path_policy": "LEXISTS_AND_REJECT_SYMLINK_THEN_RENAME_NOREPLACE",
}
temporary = directory / ".completion.tmp"
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(directory / "ATTEMPT_COMPLETION.json")
PY
(
  cd "$ATTEMPT_DIR"
  find . -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > SHA256SUMS
  sha256sum SHA256SUMS > SHA256SUMS.seal.sha256
  sha256sum -c SHA256SUMS >/dev/null
  sha256sum -c SHA256SUMS.seal.sha256 >/dev/null
)

STAGE="consume_attempt_noreplace"
require_absent "${ATTEMPT_DIR}.consumed" "consumed attempt immediately before publish"
rename_noreplace "$ATTEMPT_DIR" "${ATTEMPT_DIR}.consumed"
require_directory_nosymlink "${ATTEMPT_DIR}.consumed" "consumed successful attempt"
SUCCESS=1
trap - EXIT INT TERM HUP
echo "PASS_M601_M579_R4_ATOMIC_LEXISTS_NOSYMLINK_RESULT $RESULT_DIR"
