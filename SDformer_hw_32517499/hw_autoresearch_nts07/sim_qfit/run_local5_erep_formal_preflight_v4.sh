#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-results/local5_erep_formal_preflight_v4_20260810}"
mkdir -p "$OUT_DIR"

SCRIPT="scripts/local5_erep_formal_preflight_v4.py"
TEST="tests/test_local5_erep_formal_preflight_v4.py"

{
  date -u +%Y-%m-%dT%H:%M:%SZ
  python3 --version
} >"$OUT_DIR/tool_versions.txt"

python3 -m py_compile "$SCRIPT" "$TEST" \
  >"$OUT_DIR/py_compile.log" 2>&1
python3 -m unittest "$TEST" \
  >"$OUT_DIR/unittest.log" 2>&1
python3 "$SCRIPT" --output "$OUT_DIR/preflight.json" \
  >"$OUT_DIR/preflight.log" 2>&1
python3 - "$OUT_DIR/preflight.json" "$OUT_DIR/source_input_sha256.txt" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path.cwd().resolve()
sys.path.insert(0, str(root / "scripts"))
import local5_erep_formal_preflight_v4 as preflight

report_path = Path(sys.argv[1])
value = json.loads(report_path.read_text(encoding="utf-8"))
preflight.validate_report_for_packaging(value)

paths = [
    root / "scripts/local5_erep_formal_preflight_v4.py",
    root / "tests/test_local5_erep_formal_preflight_v4.py",
    root / "sim_qfit/run_local5_erep_formal_preflight_v4.sh",
    preflight.SELECTION_PLAN,
    preflight.PROJECTION_CONTRACT,
]
projection = json.loads(preflight.PROJECTION_CONTRACT.read_text(encoding="utf-8"))
paths.append(
    preflight.safe_profile_artifact(
        preflight.PROFILE_DIR,
        projection["payload_file"],
        "runner projection payload",
    )
)
if value["formal_artifact_bindings"] is not None:
    for binding in value["formal_artifact_bindings"].values():
        path = (root / binding["path"]).resolve()
        if preflight.sha256_file(path) != binding["sha256"]:
            raise SystemExit(f"formal artifact changed after preflight: {path}")
        paths.append(path)

seen = set()
lines = []
for path in paths:
    path = path.resolve()
    if path in seen:
        continue
    seen.add(path)
    if not path.is_file():
        raise SystemExit(f"source/input artifact absent: {path}")
    relative = path.relative_to(root).as_posix()
    lines.append(f"{preflight.sha256_file(path)}  {relative}")
Path(sys.argv[2]).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

{
  git rev-parse HEAD
  git status --short -- \
    "$SCRIPT" \
    "$TEST" \
    sim_qfit/run_local5_erep_formal_preflight_v4.sh
} >"$OUT_DIR/source_git_state.txt"

sha256sum \
  "$OUT_DIR"/{tool_versions.txt,py_compile.log,unittest.log,preflight.log,preflight.json,source_input_sha256.txt,source_git_state.txt} \
  >"$OUT_DIR/result_sha256.txt"

python3 - "$OUT_DIR/complete.json" "$OUT_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

output = Path(sys.argv[2]).resolve()
result_hashes = output / "result_sha256.txt"
preflight = json.loads((output / "preflight.json").read_text(encoding="utf-8"))
value = {
    "schema": "local5_erep_formal_preflight_complete_v4",
    "status": preflight["status"],
    "formal_g0_status": "DENY",
    "admission_generated": False,
    "output_directory": str(output),
    "result_sha256_file_sha256": hashlib.sha256(result_hashes.read_bytes()).hexdigest(),
}
Path(sys.argv[1]).write_text(
    json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

sha256sum \
  "$OUT_DIR/result_sha256.txt" \
  "$OUT_DIR/complete.json" \
  >"$OUT_DIR/receipt_sha256.txt"

PREFLIGHT_STATUS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["status"])' "$OUT_DIR/preflight.json")"
printf 'PASS Local5 EREP formal preflight v4 preflight_status=%s formal_g0=DENY output=%s\n' \
  "$PREFLIGHT_STATUS" "$OUT_DIR"
