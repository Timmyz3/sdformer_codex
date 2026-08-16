#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_DIR="${RELEASE_DIR:-$ROOT/results/local5_erep_numeric_rtl_release_v9_phasepatch_20260811}"
OUT_DIR="${OUT_DIR:-$ROOT/results/local5_h3_phase_template_patch_canary_v1_20260811}"
TABLE_DIR="${TABLE_DIR:-$ROOT/results/local5_identity_service_tables_sample2_h3_v4_reviewfix3_20260811}"
VECTOR_DIR="${VECTOR_DIR:-$ROOT/results/local5_erep_numeric_sample2_h3_canary_v5_release_20260811/windows/s0_b0}"
BASELINE_DIR="${BASELINE_DIR:-$ROOT/results/local5_identity_service_h3_canary_v8_sealed_20260811}"

for name in RELEASE_DIR OUT_DIR TABLE_DIR VECTOR_DIR BASELINE_DIR; do
  value="${!name}"
  printf -v "$name" '%s' "$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$value")"
done
if [[ -e "$OUT_DIR" ]]; then
  echo "OUT_DIR 已存在，拒绝覆盖：$OUT_DIR" >&2
  exit 2
fi

if [[ ! -f "$RELEASE_DIR/release_complete.json" ]]; then
  RELEASE_ONLY=1 RELEASE_SERVICE_MODE=identity RELEASE_WEIGHT_HOLD_CYCLES=2 \
    RELEASE_DIR="$RELEASE_DIR" \
    bash "$ROOT/sim_qfit/run_local5_erep_numeric_sample_shard.sh"
fi

RELEASE_MANIFEST="$RELEASE_DIR/release_manifest.json"
RELEASE_SCRIPT="$RELEASE_DIR/source/scripts/local5_erep_numeric_release.py"
GENERATOR="$RELEASE_DIR/source/scripts/generate_local5_h3_phase_template_patch_v1.py"
VERIFIER="$RELEASE_DIR/source/scripts/verify_local5_h3_phase_template_patch_v1.py"
RUNNER_SOURCE="$RELEASE_DIR/source/sim_qfit/run_local5_h3_phase_template_patch_canary_v1.sh"
TABLE_VERIFY="$TABLE_DIR/source/verify_local5_identity_service_tables_v4.py"
IDENTITY_MANIFEST="$TABLE_DIR/manifest.json"
IDENTITY_RECEIPT="$TABLE_DIR/verification_receipt.json"
IDENTITY_TASK_PLAN="$TABLE_DIR/task_plan.json"
COMBINED_INPUT="$VECTOR_DIR/vectors/combined_head_inputs.txt"
WEIGHTS="$VECTOR_DIR/vectors/projection_weights.txt"
VECTOR_MANIFEST="$VECTOR_DIR/vectors/manifest.json"
EXPECTED="$VECTOR_DIR/software_expected/software_expected.npz"
BASELINE_TRACE="$BASELINE_DIR/identity_trace.csv"
BASELINE_ACTUAL="$BASELINE_DIR/actual.memh"

python3 - "$RELEASE_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
build = manifest["builds"]["3"]
argv = build["compile_argv"]
if (
    build["service_mode"] != "identity_derived"
    or "-GFORCE_WEIGHT_RESPONSE_HOLD_CYCLES=2" not in argv
):
    raise SystemExit("v9 H3 release 未冻结两周期 weight response hold")
PY

for path in \
  "$GENERATOR" "$VERIFIER" "$RUNNER_SOURCE" "$TABLE_VERIFY" \
  "$IDENTITY_MANIFEST" "$IDENTITY_RECEIPT" "$IDENTITY_TASK_PLAN" \
  "$COMBINED_INPUT" "$WEIGHTS" "$VECTOR_MANIFEST" "$EXPECTED" \
  "$BASELINE_TRACE" "$BASELINE_ACTUAL"; do
  if [[ ! -f "$path" ]]; then
    echo "输入文件缺失：$path" >&2
    exit 2
  fi
done

mkdir -p "$OUT_DIR"
(cd /tmp && python3 "$RELEASE_SCRIPT" verify --release-dir "$RELEASE_DIR") \
  >"$OUT_DIR/release_preverify.json"
(cd /tmp && python3 "$TABLE_VERIFY" --package-dir "$TABLE_DIR" verify) \
  >"$OUT_DIR/table_preverify.json"

EXECUTABLE="$(python3 - "$RELEASE_MANIFEST" "$RELEASE_DIR" <<'PY'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(Path(sys.argv[2]) / manifest["builds"]["3"]["executable_path"])
PY
)"
COMPILE_ARGV="$(python3 - "$RELEASE_MANIFEST" "$RELEASE_DIR" <<'PY'
import json
import sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(Path(sys.argv[2]) / manifest["builds"]["3"]["compile_argv_path"])
PY
)"
MANIFEST_SHA="$(sha256sum "$IDENTITY_MANIFEST" | cut -d' ' -f1)"
RECEIPT_SHA="$(sha256sum "$IDENTITY_RECEIPT" | cut -d' ' -f1)"
RUN_ARGV=(
  "$EXECUTABLE"
  "+INPUTS=$COMBINED_INPUT"
  "+WEIGHTS=$WEIGHTS"
  "+STAGE_ID=0" "+BLOCK_ID=0" "+WINDOW_ID=249" "+NO_ACC_CHECK"
  "+SERVICE_SEED=20260810"
  "+RELATION_DELAY_MEMH=$TABLE_DIR/relation_delay.memh"
  "+WEIGHT_DELAY_MEMH=$TABLE_DIR/weight_delay.memh"
  "+FINAL_DELAY_MEMH=$TABLE_DIR/final_delay.memh"
  "+IDENTITY_MANIFEST_SHA=$MANIFEST_SHA"
  "+IDENTITY_RECEIPT_SHA=$RECEIPT_SHA"
  "+IDENTITY_TRACE=$OUT_DIR/candidate_trace.csv"
  "+ACTUAL_ACC_FILE=$OUT_DIR/candidate_actual.memh"
)
python3 - "$OUT_DIR/run_argv.json" "${RUN_ARGV[@]}" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(
    json.dumps(sys.argv[2:], ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

/usr/bin/time -f 'wall_seconds=%e\nmax_rss_kb=%M' \
  -o "$OUT_DIR/verilator_time.txt" \
  "${RUN_ARGV[@]}" >"$OUT_DIR/verilator.log" 2>&1

python3 "$GENERATOR" \
  --trace "$OUT_DIR/candidate_trace.csv" \
  --heads 3 \
  --output-dir "$OUT_DIR/template_patch" \
  >"$OUT_DIR/generator_stdout.json"

python3 "$VERIFIER" \
  --archive "$OUT_DIR/template_patch/phase_template_patch.npz" \
  --manifest "$OUT_DIR/template_patch/manifest.json" \
  --candidate-trace "$OUT_DIR/candidate_trace.csv" \
  --baseline-trace "$BASELINE_TRACE" \
  --candidate-actual "$OUT_DIR/candidate_actual.memh" \
  --baseline-actual "$BASELINE_ACTUAL" \
  --expected "$EXPECTED" \
  --inputs "$COMBINED_INPUT" \
  --weights "$WEIGHTS" \
  --identity-manifest "$IDENTITY_MANIFEST" \
  --identity-receipt "$IDENTITY_RECEIPT" \
  --verilator-log "$OUT_DIR/verilator.log" \
  --output "$OUT_DIR/verification.json" \
  >"$OUT_DIR/verification_stdout.json"

(cd /tmp && python3 "$RELEASE_SCRIPT" verify --release-dir "$RELEASE_DIR") \
  >"$OUT_DIR/release_postverify.json"

python3 - \
  "$ROOT" "$OUT_DIR" "$RELEASE_DIR" "$RELEASE_MANIFEST" "$EXECUTABLE" \
  "$COMPILE_ARGV" "$GENERATOR" "$VERIFIER" "$RUNNER_SOURCE" \
  "$COMBINED_INPUT" "$WEIGHTS" "$VECTOR_MANIFEST" "$EXPECTED" \
  "$IDENTITY_TASK_PLAN" "$IDENTITY_MANIFEST" "$IDENTITY_RECEIPT" \
  "$BASELINE_TRACE" "$BASELINE_ACTUAL" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

values = list(map(Path, sys.argv[1:]))
(
    root, out, release, release_manifest, executable, compile_argv,
    generator, verifier, runner, inputs, weights, vector_manifest, expected,
    task_plan, identity_manifest, identity_receipt, baseline_trace,
    baseline_actual,
) = values
paths = {
    "release_manifest": release_manifest,
    "h3_executable": executable,
    "h3_compile_argv": compile_argv,
    "template_generator": generator,
    "independent_template_verifier": verifier,
    "sealed_runner_source": runner,
    "combined_head_inputs": inputs,
    "projection_weights": weights,
    "vector_manifest": vector_manifest,
    "software_expected_npz": expected,
    "identity_task_plan": task_plan,
    "identity_manifest": identity_manifest,
    "identity_receipt": identity_receipt,
    "baseline_v8_trace": baseline_trace,
    "baseline_v8_actual": baseline_actual,
    "release_preverify": out / "release_preverify.json",
    "table_preverify": out / "table_preverify.json",
    "run_argv": out / "run_argv.json",
    "verilator_time": out / "verilator_time.txt",
    "verilator_log": out / "verilator.log",
    "candidate_trace": out / "candidate_trace.csv",
    "candidate_actual": out / "candidate_actual.memh",
    "template_archive": out / "template_patch/phase_template_patch.npz",
    "template_manifest": out / "template_patch/manifest.json",
    "generator_stdout": out / "generator_stdout.json",
    "verification": out / "verification.json",
    "verification_stdout": out / "verification_stdout.json",
    "release_postverify": out / "release_postverify.json",
}
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
verification = json.loads(paths["verification"].read_text(encoding="utf-8"))
if (
    verification.get("status") != "PASS_PHASE_TEMPLATE_TILE_PATCH_H3_NOT_G0"
    or verification.get("formal_g0") != "DENY"
    or verification.get("acc32", {}).get("mismatch") != 0
    or verification.get("payload_and_backpressure", {}).get("weight_held_valid_pairs")
       != 9216
    or verification.get("payload_and_backpressure", {}).get("weight_valid1_ready0_cycles")
       != 18432
):
    raise SystemExit("phase-template/tile-patch 独立验证未通过")

def locator(path):
    resolved = path.resolve()
    for scope, base in (("run", out), ("release", release), ("workspace", root)):
        try:
            return {"scope": scope, "relative_path": resolved.relative_to(base).as_posix()}
        except ValueError:
            pass
    return {"scope": "external", "absolute_path": str(resolved)}

complete = {
    "schema": "local5_h3_phase_template_patch_canary_complete_v1",
    "status": "PASS_SEALED_PHASE_TEMPLATE_TILE_PATCH_H3_NOT_G0",
    "evidence": "[rtl]+[独立软件展开验证]+[rtl-build-provenance]",
    "formal_g0": "DENY",
    "identity": {"sample": 2, "stage": 0, "block": 0, "window": 249, "heads": 3},
    "direct_bindings": {
        name: {**locator(path), "sha256": digest(path)} for name, path in paths.items()
    },
    "verified_metrics": {
        "trace_rows": verification["expansion"]["rows"],
        "expanded_trace_sha256": verification["expansion"]["expanded_trace_sha256"],
        "acc32_scalars": verification["acc32"]["scalars"],
        "acc32_mismatch": verification["acc32"]["mismatch"],
        "weight_held_valid_pairs": verification["payload_and_backpressure"]["weight_held_valid_pairs"],
        "weight_valid1_ready0_cycles": verification["payload_and_backpressure"]["weight_valid1_ready0_cycles"],
        "base_event_reuse_factor": verification["archive"]["base_event_reuse_factor"],
        "archive_file_size_reduction": verification["archive"]["file_size_reduction"],
    },
    "boundary": [
        "仅 H3 单窗 phase-template + typed tile-patch canary",
        "archive 文件压缩率不是片上 SRAM、周期或能耗收益",
        "formal G0、full encoder 与 ASIC PPA 保持未完成",
    ],
}
temporary = out / f"complete.json.tmp.{os.getpid()}"
temporary.write_text(json.dumps(complete, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, out / "complete.json")
PY

chmod -R a-w "$OUT_DIR"
printf 'PASS Local5 H3 phase-template tile-patch canary formal_g0=DENY output=%s\n' \
  "$OUT_DIR"
