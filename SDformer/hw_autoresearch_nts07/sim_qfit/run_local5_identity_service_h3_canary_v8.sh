#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_DIR="${RELEASE_DIR:-$ROOT/results/local5_erep_numeric_rtl_release_v8_identity_20260811}"
OUT_DIR="${OUT_DIR:-$ROOT/results/local5_identity_service_h3_canary_v8_sealed_20260811}"
TABLE_DIR="${TABLE_DIR:-$ROOT/results/local5_identity_service_tables_sample2_h3_v4_reviewfix3_20260811}"
VECTOR_DIR="${VECTOR_DIR:-$ROOT/results/local5_erep_numeric_sample2_h3_canary_v5_release_20260811/windows/s0_b0}"

for name in RELEASE_DIR OUT_DIR TABLE_DIR VECTOR_DIR; do
  value="${!name}"
  printf -v "$name" '%s' "$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$value")"
done

if [[ -e "$OUT_DIR" ]]; then
  echo "OUT_DIR 已存在，拒绝覆盖：$OUT_DIR" >&2
  exit 2
fi

if [[ ! -f "$RELEASE_DIR/release_complete.json" ]]; then
  RELEASE_ONLY=1 RELEASE_SERVICE_MODE=identity RELEASE_DIR="$RELEASE_DIR" \
    bash "$ROOT/sim_qfit/run_local5_erep_numeric_sample_shard.sh"
fi

RELEASE_SCRIPT="$RELEASE_DIR/source/scripts/local5_erep_numeric_release.py"
TABLE_VERIFY="$TABLE_DIR/source/verify_local5_identity_service_tables_v4.py"
TRACE_VERIFY="$RELEASE_DIR/source/scripts/verify_local5_identity_service_rtl_trace_v2.py"
STATE_REFERENCE="$RELEASE_DIR/source/contracts/local5_identity_service_h3_state_reference_v1.json"
RELEASE_MANIFEST="$RELEASE_DIR/release_manifest.json"
IDENTITY_MANIFEST="$TABLE_DIR/manifest.json"
IDENTITY_RECEIPT="$TABLE_DIR/verification_receipt.json"
IDENTITY_TASK_PLAN="$TABLE_DIR/task_plan.json"
COMBINED_INPUT="$VECTOR_DIR/vectors/combined_head_inputs.txt"
WEIGHTS="$VECTOR_DIR/vectors/projection_weights.txt"
VECTOR_MANIFEST="$VECTOR_DIR/vectors/manifest.json"
EXPECTED="$VECTOR_DIR/software_expected/software_expected.npz"

python3 - "$RELEASE_MANIFEST" "$TRACE_VERIFY" "$STATE_REFERENCE" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest_path, verifier_path, state_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
build = manifest["builds"]["3"]
if build["service_mode"] != "identity_derived":
    raise SystemExit("H3 release 不是 identity-derived service mode")
bindings = {row["path"]: row["sha256"] for row in manifest["source_bindings"]}
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
for path in (verifier_path, state_path):
    relative = path.relative_to(manifest_path.parent / "source").as_posix()
    if bindings.get(relative) != digest(path):
        raise SystemExit(f"release source binding 失配: {relative}")
PY

for path in \
  "$TABLE_VERIFY" "$IDENTITY_MANIFEST" "$IDENTITY_RECEIPT" \
  "$IDENTITY_TASK_PLAN" "$COMBINED_INPUT" "$WEIGHTS" \
  "$VECTOR_MANIFEST" "$EXPECTED" "$TRACE_VERIFY" "$STATE_REFERENCE"; do
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
  "+STAGE_ID=0"
  "+BLOCK_ID=0"
  "+WINDOW_ID=249"
  "+NO_ACC_CHECK"
  "+SERVICE_SEED=20260810"
  "+RELATION_DELAY_MEMH=$TABLE_DIR/relation_delay.memh"
  "+WEIGHT_DELAY_MEMH=$TABLE_DIR/weight_delay.memh"
  "+FINAL_DELAY_MEMH=$TABLE_DIR/final_delay.memh"
  "+IDENTITY_MANIFEST_SHA=$MANIFEST_SHA"
  "+IDENTITY_RECEIPT_SHA=$RECEIPT_SHA"
  "+IDENTITY_TRACE=$OUT_DIR/identity_trace.csv"
  "+ACTUAL_ACC_FILE=$OUT_DIR/actual.memh"
)
python3 - "$OUT_DIR/run_argv.json" "${RUN_ARGV[@]}" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(
    json.dumps(sys.argv[2:], ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

/usr/bin/time -f 'wall_seconds=%e\nmax_rss_kb=%M' \
  -o "$OUT_DIR/verilator_time.txt" \
  "${RUN_ARGV[@]}" >"$OUT_DIR/verilator.log" 2>&1

python3 "$TRACE_VERIFY" \
  --trace "$OUT_DIR/identity_trace.csv" \
  --package-dir "$TABLE_DIR" \
  --state-reference "$STATE_REFERENCE" \
  --actual "$OUT_DIR/actual.memh" \
  --expected "$EXPECTED" \
  --verilator-log "$OUT_DIR/verilator.log" \
  --output "$OUT_DIR/verification.json" \
  >"$OUT_DIR/verification_stdout.json"

(cd /tmp && python3 "$RELEASE_SCRIPT" verify --release-dir "$RELEASE_DIR") \
  >"$OUT_DIR/release_postverify.json"

python3 - \
  "$OUT_DIR" "$RELEASE_MANIFEST" "$EXECUTABLE" "$COMPILE_ARGV" \
  "$TRACE_VERIFY" "$STATE_REFERENCE" "$COMBINED_INPUT" "$WEIGHTS" \
  "$VECTOR_MANIFEST" "$EXPECTED" "$IDENTITY_TASK_PLAN" \
  "$IDENTITY_MANIFEST" "$IDENTITY_RECEIPT" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

(
    out_text, release_text, executable_text, compile_argv_text,
    verifier_text, state_text, input_text, weights_text, vector_manifest_text,
    expected_text, task_plan_text, identity_manifest_text, identity_receipt_text,
) = sys.argv[1:]
out = Path(out_text)
paths = {
    "release_manifest": Path(release_text),
    "h3_executable": Path(executable_text),
    "h3_compile_argv": Path(compile_argv_text),
    "trace_verifier_v2": Path(verifier_text),
    "state_reference_v1": Path(state_text),
    "combined_head_inputs": Path(input_text),
    "projection_weights": Path(weights_text),
    "vector_manifest": Path(vector_manifest_text),
    "software_expected_npz": Path(expected_text),
    "identity_task_plan": Path(task_plan_text),
    "identity_manifest": Path(identity_manifest_text),
    "identity_verification_receipt": Path(identity_receipt_text),
    "release_preverify": out / "release_preverify.json",
    "table_preverify": out / "table_preverify.json",
    "run_argv": out / "run_argv.json",
    "verilator_time": out / "verilator_time.txt",
    "verilator_log": out / "verilator.log",
    "identity_trace": out / "identity_trace.csv",
    "actual_acc32": out / "actual.memh",
    "verification": out / "verification.json",
    "verification_stdout": out / "verification_stdout.json",
    "release_postverify": out / "release_postverify.json",
}
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
verification = json.loads(paths["verification"].read_text(encoding="utf-8"))
if (
    verification.get("status") != "PASS_IDENTITY_SERVICE_RTL_TRACE_V2_NOT_G0"
    or verification.get("formal_g0") != "DENY"
    or verification.get("acc32", {}).get("mismatch") != 0
    or verification.get("payload_stability", {}).get("status")
       != "PASS_EXACT_AVAILABLE_ACCEPT_PAYLOAD"
):
    raise SystemExit("trace-v2/Acc32 验证未通过")
value = {
    "schema": "local5_identity_service_h3_canary_complete_v2",
    "status": "PASS_SEALED_H3_IDENTITY_SERVICE_TRACE_V2_NOT_G0",
    "evidence": "[rtl]+[软件整数金参考]+[rtl-build-provenance]",
    "formal_g0": "DENY",
    "identity": {
        "sample": 2, "stage": 0, "block": 0, "window": 249, "heads": 3,
    },
    "direct_bindings": {
        name: {"path": str(path.resolve()), "sha256": digest(path)}
        for name, path in paths.items()
    },
    "verified_metrics": {
        "trace_rows": verification["trace_rows"],
        "rtl_cycles_validation_only": verification["rtl_cycles"],
        "acc32_scalars": verification["acc32"]["scalars"],
        "acc32_mismatch": verification["acc32"]["mismatch"],
        "relation_payload_pairs": verification["payload_stability"]["relation_pairs"],
        "weight_payload_pairs": verification["payload_stability"]["weight_pairs"],
        "final_payload_pairs": verification["payload_stability"]["final_pairs"],
        "state_event_count": verification["state_reference"]["all_state_count"],
    },
    "boundary": [
        "仅 sample2/stage0/block0/window249/H3 单窗 canary",
        "周期为验证环境延迟，不是架构性能",
        "formal G0 仍为 DENY；无 full encoder、ASIC PPA、吞吐或能耗结论",
    ],
}
temporary = out / f"complete.json.tmp.{os.getpid()}"
temporary.write_text(
    json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
os.replace(temporary, out / "complete.json")
PY

chmod -R a-w "$OUT_DIR"
printf 'PASS Local5 sealed H3 identity-service trace-v2 formal_g0=DENY output=%s\n' \
  "$OUT_DIR"
