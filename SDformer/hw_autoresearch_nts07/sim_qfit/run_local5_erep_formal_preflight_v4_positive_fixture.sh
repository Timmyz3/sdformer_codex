#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-results/local5_erep_formal_preflight_v4_positive_fixture_20260810}"
FIXTURE="$OUT_DIR/profile_fixture"
RUNNER_OUT="$OUT_DIR/runner"
BUILDER="tests/build_local5_erep_formal_preflight_v4_positive_fixture.py"
SELF="sim_qfit/run_local5_erep_formal_preflight_v4_positive_fixture.sh"

mkdir -p "$OUT_DIR"
python3 "$BUILDER" --output "$FIXTURE" >"$OUT_DIR/fixture_build.log" 2>&1

LOCAL5_EREP_PROFILE_DIR="$FIXTURE" \
OUT_DIR="$RUNNER_OUT" \
bash sim_qfit/run_local5_erep_formal_preflight_v4.sh \
  >"$OUT_DIR/runner.log" 2>&1

sha256sum -c "$RUNNER_OUT/result_sha256.txt" >"$OUT_DIR/runner_result_check.log"
sha256sum -c "$RUNNER_OUT/receipt_sha256.txt" >"$OUT_DIR/runner_receipt_check.log"

python3 - "$RUNNER_OUT/preflight.json" "$RUNNER_OUT/source_input_sha256.txt" <<'PY'
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report["status"] != "PREFLIGHT_PASS_NOT_G0":
    raise SystemExit("positive fixture did not reach PREFLIGHT_PASS_NOT_G0")
if report["admission_generated"] is not False:
    raise SystemExit("positive fixture must not generate G0 admission")
if set(report["formal_artifact_bindings"]) != {
    "formal_manifest", "ordered_payload", "cohort",
    "projection_contract", "projection_payload",
}:
    raise SystemExit("positive fixture artifact binding set is incomplete")
hash_text = Path(sys.argv[2]).read_text(encoding="utf-8")
for name in (
    "ordered_term_manifest.json", "ordered_term_items.npz", "ordered_cohort.json",
    "checkpoint_projection_contract.json", "checkpoint_projection_contract.npz",
):
    if name not in hash_text:
        raise SystemExit(f"positive source-input receipt misses {name}")
PY

sha256sum \
  "$BUILDER" \
  "$SELF" \
  "$OUT_DIR/fixture_build.log" \
  "$OUT_DIR/runner.log" \
  "$OUT_DIR/runner_result_check.log" \
  "$OUT_DIR/runner_receipt_check.log" \
  "$RUNNER_OUT/preflight.json" \
  "$RUNNER_OUT/source_input_sha256.txt" \
  "$RUNNER_OUT/receipt_sha256.txt" \
  >"$OUT_DIR/result_sha256.txt"

sha256sum "$OUT_DIR/result_sha256.txt" >"$OUT_DIR/receipt_sha256.txt"
printf 'PASS Local5 EREP manifest-present runner fixture output=%s\n' "$OUT_DIR"
