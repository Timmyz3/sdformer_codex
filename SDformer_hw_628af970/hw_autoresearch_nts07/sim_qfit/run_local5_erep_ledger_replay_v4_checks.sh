#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-results/local5_erep_ledger_replay_v4_20260810}"
PYTHON="/opt/conda/bin/python3.11"
mkdir -p "$OUT_DIR"

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  "$PYTHON" --version
  "$PYTHON" -c 'import numpy; print("numpy=" + numpy.__version__)'
} >"$OUT_DIR/tool_versions.txt"

"$PYTHON" -m py_compile \
  scripts/local5_erep_archive_replay_v4.py \
  scripts/local5_erep_ledger_replay_v4.py \
  scripts/local5_erep_statistics_v4.py \
  tests/test_local5_erep_archive_replay_v4.py \
  tests/test_local5_erep_ledger_replay_v4.py \
  tests/test_local5_erep_statistics_v4.py \
  >"$OUT_DIR/py_compile.log" 2>&1

"$PYTHON" -m unittest \
  tests/test_local5_erep_archive_replay_v4.py \
  tests/test_local5_erep_ledger_replay_v4.py \
  tests/test_local5_erep_statistics_v4.py \
  tests/test_local5_erep_command_schedule_v4.py \
  tests/test_local5_erep_capacity_baselines_v4.py \
  >"$OUT_DIR/unittest.log" 2>&1

if "$PYTHON" scripts/local5_erep_statistics_v4.py \
  >"$OUT_DIR/formal_entry.stdout.log" 2>"$OUT_DIR/formal_entry.stderr.log"; then
  echo "formal statistics unexpectedly passed without admission artifacts" >&2
  exit 1
fi
grep -F "required frozen artifact is absent:" "$OUT_DIR/formal_entry.stderr.log" \
  >"$OUT_DIR/formal_deny_check.log"

"$PYTHON" - "$OUT_DIR/synthetic_replay_summary.json" <<'PY'
import json
import sys
from pathlib import Path

from scripts.local5_erep_archive_replay_v4 import (
    encode_miter_fixture,
    encode_trace_fixture,
    validate_archive_contents,
)
from scripts.local5_erep_ledger_replay_v4 import replay_ledger_document
from tests.test_local5_erep_ledger_replay_v4 import synthetic_head_ledger

head = synthetic_head_ledger()
miter, head = encode_miter_fixture(head)
trace = encode_trace_fixture(head)
archive = validate_archive_contents(trace, miter, head)
windows, commands = replay_ledger_document(head)
row = commands["rows"][0]
value = {
    "schema": "local5_erep_ledger_replay_v4_synthetic_summary",
    "evidence": "[synthetic-contract]",
    "head_ledger_contains_candidate_scalars": any(
        key in json.dumps(head, sort_keys=True)
        for key in ('"c0"', '"c1"', '"c2"', '"c3"', '"c4"')
    ),
    "window_count": len(windows["rows"]),
    "command_count": len(commands["rows"]),
    "cycles": {
        candidate: row[candidate]
        for candidate in ("c0", "c1", "c2", "c3", "c4")
    },
    "window_schedule_sha256": row["window_schedule_sha256"],
    "command_ledger_sha256": row["command_ledger_sha256"],
    "formal_g0": "DENY",
    "archive_content_replay": archive,
}
Path(sys.argv[1]).write_text(
    json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

sha256sum \
  scripts/local5_erep_archive_replay_v4.py \
  scripts/local5_erep_ledger_replay_v4.py \
  scripts/local5_erep_statistics_v4.py \
  scripts/local5_erep_command_schedule_v4.py \
  scripts/local5_erep_capacity_baselines_v4.py \
  tests/test_local5_erep_archive_replay_v4.py \
  tests/test_local5_erep_ledger_replay_v4.py \
  tests/test_local5_erep_statistics_v4.py \
  tests/test_local5_erep_command_schedule_v4.py \
  tests/test_local5_erep_capacity_baselines_v4.py \
  sim_qfit/run_local5_erep_ledger_replay_v4_checks.sh \
  contracts/local5_erep_g0_runtime_v4_20260810.json \
  docs/310_Local5_EREP正式Archive内容重放合同_20260810.md \
  >"$OUT_DIR/source_input_sha256.txt"

tar --sort=name --mtime='UTC 2026-08-10' --owner=0 --group=0 --numeric-owner \
  -cf - \
  scripts/local5_erep_archive_replay_v4.py \
  scripts/local5_erep_ledger_replay_v4.py \
  scripts/local5_erep_statistics_v4.py \
  scripts/local5_erep_command_schedule_v4.py \
  scripts/local5_erep_capacity_baselines_v4.py \
  tests/test_local5_erep_archive_replay_v4.py \
  tests/test_local5_erep_ledger_replay_v4.py \
  tests/test_local5_erep_statistics_v4.py \
  tests/test_local5_erep_command_schedule_v4.py \
  tests/test_local5_erep_capacity_baselines_v4.py \
  sim_qfit/run_local5_erep_ledger_replay_v4_checks.sh \
  contracts/local5_erep_g0_runtime_v4_20260810.json \
  docs/310_Local5_EREP正式Archive内容重放合同_20260810.md \
  | gzip -n >"$OUT_DIR/source_bundle.tar.gz"

{
  git rev-parse HEAD
  git status --short -- \
    scripts/local5_erep_ledger_replay_v4.py \
    scripts/local5_erep_archive_replay_v4.py \
    scripts/local5_erep_statistics_v4.py \
    tests/test_local5_erep_ledger_replay_v4.py \
    tests/test_local5_erep_archive_replay_v4.py \
    tests/test_local5_erep_statistics_v4.py \
    sim_qfit/run_local5_erep_ledger_replay_v4_checks.sh
} >"$OUT_DIR/source_git_state.txt"

sha256sum \
  "$OUT_DIR"/{tool_versions.txt,py_compile.log,unittest.log,formal_entry.stdout.log,formal_entry.stderr.log,formal_deny_check.log,synthetic_replay_summary.json,source_input_sha256.txt,source_bundle.tar.gz,source_git_state.txt} \
  >"$OUT_DIR/result_sha256.txt"

"$PYTHON" - "$OUT_DIR/complete.json" "$OUT_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[2]).resolve()
result_hashes = output / "result_sha256.txt"
value = {
    "schema": "local5_erep_ledger_replay_v4_complete",
    "status": "SYNTHETIC_CONTRACT_PASS_FORMAL_G0_DENY",
    "evidence": "[synthetic-contract]+[代码审计]",
    "formal_g0": "DENY",
    "output_directory": str(output),
    "result_sha256_file_sha256": hashlib.sha256(result_hashes.read_bytes()).hexdigest(),
}
Path(sys.argv[1]).write_text(
    json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

sha256sum "$OUT_DIR/result_sha256.txt" "$OUT_DIR/complete.json" \
  >"$OUT_DIR/receipt_sha256.txt"

printf 'PASS Local5 EREP ledger replay v4 synthetic_contract=PASS formal_g0=DENY output=%s\n' \
  "$OUT_DIR"
