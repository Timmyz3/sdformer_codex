#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RELEASE_DIR="${RELEASE_DIR:-$ROOT/results/local5_erep_numeric_rtl_release_v10_phasepatch_20260811}"
TABLE_DIR="${TABLE_DIR:-$ROOT/results/local5_identity_service_tables_sample2_h3_v4_reviewfix3_20260811}"
VECTOR_DIR="${VECTOR_DIR:-$ROOT/results/local5_erep_numeric_sample2_h3_canary_v5_release_20260811/windows/s0_b0}"
PROFILE_DIR="${PROFILE_DIR:-$ROOT/results/local5_fullres_bb1e4_joint_heads_profile100_20260809}"
OUT_DIR="${OUT_DIR:-$ROOT/results/local5_phase_telemetry_pilot_h3_sample2_w249_v3_canonical_20260812}"
REQUESTED_SAMPLE="${REQUESTED_SAMPLE:-}"
REQUESTED_STAGE="${REQUESTED_STAGE:-}"
REQUESTED_BLOCK="${REQUESTED_BLOCK:-}"
REQUESTED_WINDOW="${REQUESTED_WINDOW:-}"

for name in RELEASE_DIR TABLE_DIR VECTOR_DIR PROFILE_DIR OUT_DIR; do
  value="${!name}"
  printf -v "$name" '%s' "$(python3 -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$value")"
done

if [[ -e "$OUT_DIR" ]]; then
  echo "OUT_DIR 已存在，拒绝覆盖：$OUT_DIR" >&2
  exit 2
fi

RELEASE_MANIFEST="$RELEASE_DIR/release_manifest.json"
RELEASE_COMPLETE="$RELEASE_DIR/release_complete.json"
TASK_PLAN="$TABLE_DIR/task_plan.json"
TABLE_MANIFEST="$TABLE_DIR/manifest.json"
TABLE_RECEIPT="$TABLE_DIR/verification_receipt.json"
TABLE_VERIFY="$TABLE_DIR/source/verify_local5_identity_service_tables_v4.py"
VECTOR_MANIFEST="$VECTOR_DIR/vectors/manifest.json"
SELECTION_PLAN="$PROFILE_DIR/joint_window_selection_plan.json"
PROFILE_MANIFEST="$PROFILE_DIR/ordered_term_manifest.json"
COMBINED_INPUT="$VECTOR_DIR/vectors/combined_head_inputs.txt"
WEIGHTS="$VECTOR_DIR/vectors/projection_weights.txt"
EXPECTED="$VECTOR_DIR/software_expected/software_expected.npz"
MONITOR="$ROOT/verif_qfit/local5_phase_semantic_monitor_v1.sv"
BIND="$ROOT/verif_qfit/bind_local5_phase_semantic_monitor_v1.sv"
VERIFIER="$ROOT/scripts/verify_local5_phase_telemetry_pilot_v1.py"
TEST="$ROOT/scripts/test_verify_local5_phase_telemetry_pilot_v1.py"
RUNNER="$ROOT/sim_qfit/run_local5_phase_telemetry_pilot_v1.sh"

for path in \
  "$RELEASE_MANIFEST" "$RELEASE_COMPLETE" "$TASK_PLAN" \
  "$TABLE_MANIFEST" "$TABLE_RECEIPT" "$TABLE_VERIFY" \
  "$VECTOR_MANIFEST" "$SELECTION_PLAN" "$PROFILE_MANIFEST" \
  "$COMBINED_INPUT" "$WEIGHTS" "$EXPECTED" \
  "$MONITOR" "$BIND" "$VERIFIER" "$TEST" "$RUNNER"; do
  if [[ ! -f "$path" ]]; then
    echo "pilot 输入缺失：$path" >&2
    exit 2
  fi
done

if ! IDENTITY_OUTPUT="$(python3 - \
  "$RELEASE_MANIFEST" "$RELEASE_COMPLETE" "$TASK_PLAN" \
  "$TABLE_MANIFEST" "$TABLE_RECEIPT" "$VECTOR_MANIFEST" \
  "$SELECTION_PLAN" "$PROFILE_MANIFEST" \
  "$REQUESTED_SAMPLE" "$REQUESTED_STAGE" "$REQUESTED_BLOCK" \
  "$REQUESTED_WINDOW" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

(
    release_path, complete_path, plan_path, table_path, table_receipt_path,
    vector_path, selection_path, profile_path,
    req_sample, req_stage, req_block, req_window,
) = sys.argv[1:]
release_file = Path(release_path)
complete_file = Path(complete_path)
plan_file = Path(plan_path)
table_file = Path(table_path)
table_receipt_file = Path(table_receipt_path)
vector_file = Path(vector_path)
selection_file = Path(selection_path)
profile_file = Path(profile_path)
release = json.loads(release_file.read_text(encoding="utf-8"))
complete = json.loads(complete_file.read_text(encoding="utf-8"))
plan = json.loads(plan_file.read_text(encoding="utf-8"))
table = json.loads(table_file.read_text(encoding="utf-8"))
table_receipt = json.loads(table_receipt_file.read_text(encoding="utf-8"))
vector = json.loads(vector_file.read_text(encoding="utf-8"))
selection = json.loads(selection_file.read_text(encoding="utf-8"))
profile = json.loads(profile_file.read_text(encoding="utf-8"))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if (
    release.get("formal_g0") != "DENY"
    or complete.get("formal_g0") != "DENY"
    or complete.get("release_manifest_sha256") != sha(release_file)
    or release.get("builds", {}).get("3", {}).get("service_mode")
       != "identity_derived"
):
    raise SystemExit("v10 H3 release provenance 不合法")
if (
    plan.get("schema") != "local5_projection_task_plan_v1"
    or table.get("identity") != {
        key: int(plan[key]) for key in ("sample", "stage", "block", "window", "heads")
    }
    or vector.get("identity") != {
        **{key: int(plan[key]) for key in ("sample", "stage", "block", "window", "heads")},
        "tokens": 450,
        "out_dim": 32,
    }
    or table_receipt.get("manifest_sha256") != sha(table_file)
    or int(plan.get("heads", 0)) != 3
):
    raise SystemExit("task/table/vector identity binding 不一致")
actual = {key: int(plan[key]) for key in ("sample", "stage", "block", "window", "heads")}
requested = {
    "sample": actual["sample"] if req_sample == "" else int(req_sample),
    "stage": actual["stage"] if req_stage == "" else int(req_stage),
    "block": actual["block"] if req_block == "" else int(req_block),
    "window": actual["window"] if req_window == "" else int(req_window),
    "heads": 3,
}
records = [
    row for row in selection.get("records", [])
    if all(int(row.get(key, -1)) == actual[key]
           for key in ("sample", "stage", "block", "window", "heads"))
]
group_indices = sorted({int(row["input_group_index"]) for row in plan.get("tasks", [])})
groups = profile.get("groups", [])
if (
    selection.get("schema") != "local5_uniform_joint_window_plan_v1"
    or len(records) != 1
    or plan.get("source_manifest_sha256") != sha(profile_file)
    or len(group_indices) != 3
    or any(index < 0 or index >= len(groups) for index in group_indices)
):
    raise SystemExit("selection/profile/task canonical provenance 不一致")
selected_groups = [groups[index] for index in group_indices]
if sorted(int(row.get("head", -1)) for row in selected_groups) != [0, 1, 2] or any(
    any(int(row.get(key, -1)) != actual[key]
        for key in ("sample", "stage", "block", "window", "heads"))
    for row in selected_groups
):
    raise SystemExit("task group 不是 canonical H3 tuple")
if requested != actual:
    fields = [key for key in actual if requested[key] != actual[key]]
    raise SystemExit(
        "身份 P0 fail-closed：requested != actual；"
        f"requested={requested} actual={actual} mismatch={fields}"
    )
print(actual["sample"])
print(actual["stage"])
print(actual["block"])
print(actual["window"])
print(actual["heads"])
PY
)"; then
  printf '%s\n' "$IDENTITY_OUTPUT" >&2
  exit 3
fi
readarray -t ACTUAL_IDENTITY <<<"$IDENTITY_OUTPUT"
if [[ "${#ACTUAL_IDENTITY[@]}" -ne 5 ]]; then
  echo "canonical identity preflight 输出损坏" >&2
  exit 3
fi

ACTUAL_SAMPLE="${ACTUAL_IDENTITY[0]}"
ACTUAL_STAGE="${ACTUAL_IDENTITY[1]}"
ACTUAL_BLOCK="${ACTUAL_IDENTITY[2]}"
ACTUAL_WINDOW="${ACTUAL_IDENTITY[3]}"
ACTUAL_HEADS="${ACTUAL_IDENTITY[4]}"
REQUESTED_SAMPLE="${REQUESTED_SAMPLE:-$ACTUAL_SAMPLE}"
REQUESTED_STAGE="${REQUESTED_STAGE:-$ACTUAL_STAGE}"
REQUESTED_BLOCK="${REQUESTED_BLOCK:-$ACTUAL_BLOCK}"
REQUESTED_WINDOW="${REQUESTED_WINDOW:-$ACTUAL_WINDOW}"
REQUEST_STATUS="MATCH"

mkdir -p "$OUT_DIR/build" "$OUT_DIR/source"
cp --reflink=auto "$MONITOR" "$OUT_DIR/source/local5_phase_semantic_monitor_v1.sv"
cp --reflink=auto "$BIND" "$OUT_DIR/source/bind_local5_phase_semantic_monitor_v1.sv"
cp --reflink=auto "$VERIFIER" "$OUT_DIR/source/verify_local5_phase_telemetry_pilot_v1.py"
cp --reflink=auto "$TEST" "$OUT_DIR/source/test_verify_local5_phase_telemetry_pilot_v1.py"
cp --reflink=auto "$RUNNER" "$OUT_DIR/source/run_local5_phase_telemetry_pilot_v1.sh"
MONITOR_RUN="$OUT_DIR/source/local5_phase_semantic_monitor_v1.sv"
BIND_RUN="$OUT_DIR/source/bind_local5_phase_semantic_monitor_v1.sv"
VERIFIER_RUN="$OUT_DIR/source/verify_local5_phase_telemetry_pilot_v1.py"
TEST_RUN="$OUT_DIR/source/test_verify_local5_phase_telemetry_pilot_v1.py"
RUNNER_RUN="$OUT_DIR/source/run_local5_phase_telemetry_pilot_v1.sh"

python3 -m py_compile "$VERIFIER_RUN" "$TEST_RUN" >"$OUT_DIR/py_compile.log" 2>&1
(cd "$OUT_DIR/source" && python3 -m unittest \
  test_verify_local5_phase_telemetry_pilot_v1 -v) \
  >"$OUT_DIR/unittest.log" 2>&1
(cd /tmp && python3 "$TABLE_VERIFY" --package-dir "$TABLE_DIR" verify) \
  >"$OUT_DIR/table_preverify.json"

python3 - "$RELEASE_MANIFEST" "$RELEASE_DIR" "$OUT_DIR/build/compile_argv.json" \
  "$OUT_DIR/build/obj" "$MONITOR_RUN" "$BIND_RUN" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, release_text, output_text, obj_text, monitor_text, bind_text = sys.argv[1:]
release = Path(release_text)
argv = json.loads(Path(manifest_path).read_text(encoding="utf-8"))["builds"]["3"]["compile_argv"]
argv = [str((release / arg).resolve()) if arg.startswith("source/") else arg for arg in argv]
argv[argv.index("build/h3/obj")] = str(Path(obj_text).resolve())
argv.extend([str(Path(monitor_text).resolve()), str(Path(bind_text).resolve())])
Path(output_text).write_text(json.dumps(argv, indent=2) + "\n", encoding="utf-8")
PY

python3 - "$OUT_DIR/build/compile_argv.json" "$OUT_DIR/build/compile.log" <<'PY'
import json
import subprocess
import sys
from pathlib import Path
argv = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
with Path(sys.argv[2]).open("w", encoding="utf-8") as log:
    subprocess.run(argv, check=True, stdout=log, stderr=subprocess.STDOUT)
PY

EXECUTABLE="$OUT_DIR/build/obj/Vtb_qfit_local5_memo_multitile_cross_head"
if [[ ! -x "$EXECUTABLE" ]]; then
  echo "pilot executable 未生成" >&2
  exit 2
fi

MANIFEST_SHA="$(sha256sum "$TABLE_MANIFEST" | cut -d' ' -f1)"
RECEIPT_SHA="$(sha256sum "$TABLE_RECEIPT" | cut -d' ' -f1)"
RUN_ARGV=(
  "$EXECUTABLE"
  "+INPUTS=$COMBINED_INPUT"
  "+WEIGHTS=$WEIGHTS"
  "+STAGE_ID=$ACTUAL_STAGE"
  "+BLOCK_ID=$ACTUAL_BLOCK"
  "+WINDOW_ID=$ACTUAL_WINDOW"
  "+NO_ACC_CHECK"
  "+SERVICE_SEED=20260810"
  "+RELATION_DELAY_MEMH=$TABLE_DIR/relation_delay.memh"
  "+WEIGHT_DELAY_MEMH=$TABLE_DIR/weight_delay.memh"
  "+FINAL_DELAY_MEMH=$TABLE_DIR/final_delay.memh"
  "+IDENTITY_MANIFEST_SHA=$MANIFEST_SHA"
  "+IDENTITY_RECEIPT_SHA=$RECEIPT_SHA"
  "+IDENTITY_TRACE=$OUT_DIR/identity_trace.csv"
  "+ACTUAL_ACC_FILE=$OUT_DIR/actual.memh"
  "+PHASE_TELEMETRY=$OUT_DIR/phase_telemetry.csv"
  "+TELEMETRY_STAGE=$ACTUAL_STAGE"
  "+TELEMETRY_BLOCK=$ACTUAL_BLOCK"
  "+TELEMETRY_WINDOW=$ACTUAL_WINDOW"
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

python3 - \
  "$ROOT" "$OUT_DIR" "$REQUESTED_SAMPLE" "$REQUESTED_STAGE" \
  "$REQUESTED_BLOCK" "$REQUESTED_WINDOW" "$ACTUAL_SAMPLE" \
  "$ACTUAL_STAGE" "$ACTUAL_BLOCK" "$ACTUAL_WINDOW" "$ACTUAL_HEADS" \
  "$REQUEST_STATUS" "$TASK_PLAN" "$TABLE_MANIFEST" "$TABLE_RECEIPT" \
  "$VECTOR_MANIFEST" "$SELECTION_PLAN" "$PROFILE_MANIFEST" \
  "$COMBINED_INPUT" "$WEIGHTS" "$EXPECTED" \
  "$RELEASE_MANIFEST" "$RELEASE_COMPLETE" "$MONITOR_RUN" "$BIND_RUN" \
  "$VERIFIER_RUN" "$TEST_RUN" "$RUNNER_RUN" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

(
    root_text, out_text, req_sample, req_stage, req_block, req_window,
    act_sample, act_stage, act_block, act_window, act_heads, request_status,
    task_text, table_text, table_receipt_text, vector_text,
    selection_text, profile_text, input_text,
    weights_text, expected_text, release_text, release_complete_text,
    monitor_text, bind_text, verifier_text, test_text, runner_text,
) = sys.argv[1:]
root, out = Path(root_text), Path(out_text)
paths = {
    "telemetry": out / "phase_telemetry.csv",
    "identity_trace": out / "identity_trace.csv",
    "actual_acc32": out / "actual.memh",
    "task_plan": Path(task_text),
    "table_manifest": Path(table_text),
    "table_receipt": Path(table_receipt_text),
    "vector_manifest": Path(vector_text),
    "selection_plan": Path(selection_text),
    "profile_manifest": Path(profile_text),
    "combined_inputs": Path(input_text),
    "projection_weights": Path(weights_text),
    "software_expected": Path(expected_text),
    "release_manifest": Path(release_text),
    "release_complete": Path(release_complete_text),
    "compile_argv": out / "build/compile_argv.json",
    "compile_log": out / "build/compile.log",
    "executable": out / "build/obj/Vtb_qfit_local5_memo_multitile_cross_head",
    "run_argv": out / "run_argv.json",
    "verilator_log": out / "verilator.log",
    "verilator_time": out / "verilator_time.txt",
    "table_preverify": out / "table_preverify.json",
    "unittest_log": out / "unittest.log",
    "monitor_source": Path(monitor_text),
    "bind_source": Path(bind_text),
    "verifier_source": Path(verifier_text),
    "test_source": Path(test_text),
    "runner_source": Path(runner_text),
}
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
for name, path in paths.items():
    if not path.is_file():
        raise SystemExit(f"pilot artifact 缺失: {name}")

def locator(path: Path):
    resolved = path.resolve()
    try:
        return {"path": resolved.relative_to(out.resolve()).as_posix()}
    except ValueError:
        return {"path": str(resolved)}

receipt = {
    "schema": "local5_phase_telemetry_pilot_run_receipt_v1",
    "status": "RUN_COMPLETE_PENDING_INDEPENDENT_VERIFY_NOT_G0",
    "evidence": "[rtl-direct-run-provenance]",
    "formal_g0": "DENY",
    "requested_identity": {
        "sample": int(req_sample), "stage": int(req_stage),
        "block": int(req_block), "window": int(req_window), "heads": 3,
    },
    "actual_identity": {
        "sample": int(act_sample), "stage": int(act_stage),
        "block": int(act_block), "window": int(act_window),
        "heads": int(act_heads),
    },
    "requested_tuple_status": request_status,
    "bindings": {
        name: {**locator(path), "sha256": sha(path)}
        for name, path in paths.items()
    },
    "boundary": [
        "sample 不是 RTL signal，只能通过冻结 task/vector/table 来源绑定",
        "不存在的 requested tuple 必须 fail closed，不得重标 payload",
        "phase_count=52 仅为 H3 Direct pilot 语义 phase 记录，不是 formal 462600 phase schema",
        "验证周期不是架构性能；formal G0 保持 DENY",
    ],
}
temporary = out / f"run_receipt.json.tmp.{os.getpid()}"
temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, out / "run_receipt.json")
PY

python3 "$VERIFIER_RUN" \
  --package-dir "$OUT_DIR" \
  --output "$OUT_DIR/verification.json" \
  --machine-report-md "$OUT_DIR/machine_report.md" \
  >"$OUT_DIR/verification_stdout.json"

python3 - "$OUT_DIR" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

out = Path(sys.argv[1])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
verification = json.loads((out / "verification.json").read_text(encoding="utf-8"))
if (
    verification.get("status") != "PASS_H3_PHASE_TELEMETRY_PILOT_NOT_G0"
    or verification.get("formal_g0") != "DENY"
):
    raise SystemExit("pilot 独立验证未通过")
paths = {
    "run_receipt": out / "run_receipt.json",
    "verification": out / "verification.json",
    "verification_stdout": out / "verification_stdout.json",
    "machine_report": out / "machine_report.md",
    "monitor_source": out / "source/local5_phase_semantic_monitor_v1.sv",
    "bind_source": out / "source/bind_local5_phase_semantic_monitor_v1.sv",
    "verifier_source": out / "source/verify_local5_phase_telemetry_pilot_v1.py",
    "test_source": out / "source/test_verify_local5_phase_telemetry_pilot_v1.py",
    "runner_source": out / "source/run_local5_phase_telemetry_pilot_v1.sh",
}
for name, path in paths.items():
    if not path.is_file():
        raise SystemExit(f"final evidence binding 缺失: {name}")
pilot_receipt = {
    "schema": "local5_phase_telemetry_pilot_evidence_receipt_v1",
    "status": "PASS_H3_PHASE_TELEMETRY_PILOT_NOT_G0",
    "formal_g0": "DENY",
    "identity_audit": verification["identity_audit"],
    "bindings": {
        name: {"path": path.relative_to(out).as_posix(), "sha256": sha(path)}
        for name, path in paths.items()
    },
    "phase_schema_scope": {
        "pilot_phase_records": verification["telemetry"]["phase_count"],
        "formal_phase_schema_records": 462600,
        "equivalent": False,
    },
}
temporary = out / f"pilot_receipt.json.tmp.{os.getpid()}"
temporary.write_text(json.dumps(pilot_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, out / "pilot_receipt.json")
complete_paths = {"pilot_receipt": out / "pilot_receipt.json", **paths}
complete = {
    "schema": "local5_phase_telemetry_pilot_complete_v1",
    "status": "PASS_H3_PHASE_TELEMETRY_PILOT_NOT_G0",
    "evidence": verification["evidence"],
    "formal_g0": "DENY",
    "identity_audit": verification["identity_audit"],
    "verified_metrics": {
        "phase_count": verification["telemetry"]["phase_count"],
        "resource_event_count": verification["telemetry"]["resource_event_count"],
        "identity_trace_rows": verification["identity_trace"]["rows"],
        "acc32_scalars": verification["acc32_miter"]["scalars"],
        "acc32_mismatch": verification["acc32_miter"]["mismatch"],
        "telemetry_to_trace_ratio": verification["archive_size"]["telemetry_to_trace_ratio"],
    },
    "bindings": {
        name: {"path": path.relative_to(out).as_posix(), "sha256": sha(path)}
        for name, path in complete_paths.items()
    },
    "phase_schema_scope": pilot_receipt["phase_schema_scope"],
    "formal_boundary": "DENY_UNCHANGED",
}
temporary = out / f"complete.json.tmp.{os.getpid()}"
temporary.write_text(json.dumps(complete, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, out / "complete.json")
PY

(cd "$OUT_DIR" && sha256sum \
  complete.json pilot_receipt.json run_receipt.json verification.json verification_stdout.json \
  machine_report.md phase_telemetry.csv identity_trace.csv actual.memh \
  run_argv.json verilator.log verilator_time.txt unittest.log \
  >result_sha256.txt)

printf 'PASS Local5 H3 phase telemetry pilot actual=sample%s/stage%s/block%s/window%s requested_status=%s formal_g0=DENY output=%s\n' \
  "$ACTUAL_SAMPLE" "$ACTUAL_STAGE" "$ACTUAL_BLOCK" "$ACTUAL_WINDOW" \
  "$REQUEST_STATUS" "$OUT_DIR"
