#!/usr/bin/env bash
set -euo pipefail

task_sim_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_hw_root="$(cd "$task_sim_root/.." && pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="$task_hw_root/results/m480_fc1_fullwidth_resource_matched_context_factorized_dse_r2_exact_20260826"
[[ ! -e "$task_run" ]] || exit 2
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?; if [[ $task_complete -ne 1 ]]; then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc" > "$task_run/RUN_FAILED_OR_INCOMPLETE.txt"; fi' EXIT

cd "$task_hw_root"
declare -A task_expected=(
  ["system_simulator/scripts/analyze_m480_fc1_fullwidth_resource_matched_context_factorized_dse.py"]="2d3c7158136d4b9d8a30aacc8d8c4fe129feba265441269114f06183a14c4cc3"
  ["contracts/m480_fc1_fullwidth_resource_matched_context_factorized_dse_contract_r1_20260826.json"]="0f5be255a8bd8c92b2f4b3c8b2c6f3b738b63f799be409bc01f23f7791f0a037"
  ["results/m51_h67_ep35_binary_input_trace_r2_gpu_receipt_20260823/manifest.json"]="2a5e6e472b3897ea508f61c3727bddd97818c921b0729c0f2731b150c6d7a76e"
  ["results/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1_20260825/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1.json"]="6110dff1cac748ca934e05033ddabe39f06e8b54286699a7843c209ddfe4a6ca"
  ["results/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1_20260825/SHA256SUMS"]="133c32c37d6ff61d19ca119634b5604d8a9fe12dd510cd4d9425e59e967247e5"
  ["results/m262_fc1_descriptor_lifecycle_trace_r2_exact_20260825/trace_payload/m262_fc1_descriptor_lifecycle_trace_r1.json"]="9aa24e2ef8889e6e697121817e5e27ca028db81e9e0dee4206fbc34394ec103a"
  ["results/m262_fc1_descriptor_lifecycle_trace_r2_exact_20260825/SHA256SUMS"]="23f10ed13167d6dc8c6b5c9dbeba42a0777e995becfc7241396746608066b16e"
  ["results/m292_m287_scope_corrected_amdahl_overlay_r1_20260825/m292_m287_scope_corrected_amdahl_overlay_r1.json"]="02ce52761729dc842ea27a7419879fece9ba4c9e31c6ba44b4fc5c004da09242"
  ["results/m292_m287_scope_corrected_amdahl_overlay_r1_20260825/SHA256SUMS"]="5910278c824ce8cb9e78a4506df0b01167f8b74324e88c1e7d92fef21df82c2a"
  ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
: > "$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}"; do
  task_observed="$(sha256sum "$task_path" | awk '{print $1}')"
  printf 'path=%s expected=%s observed=%s\n' "$task_path" \
    "${task_expected[$task_path]}" "$task_observed" >> "$task_run/preflight_sha_checks.txt"
  [[ "$task_observed" == "${task_expected[$task_path]}" ]] || exit 10
done
sha256sum "${!task_expected[@]}" > "$task_run/input_sha256.txt"

cp contracts/m480_fc1_fullwidth_resource_matched_context_factorized_dse_contract_r1_20260826.json \
  "$task_run/wrong_sha_contract.json"
printf '\n' >> "$task_run/wrong_sha_contract.json"
set +e
python3 system_simulator/scripts/analyze_m480_fc1_fullwidth_resource_matched_context_factorized_dse.py \
  --manifest results/m51_h67_ep35_binary_input_trace_r2_gpu_receipt_20260823/manifest.json \
  --payload-root system_handoff/incoming/m51_capture_bundle_r2_20260823 \
  --m230-result results/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1_20260825/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1.json \
  --m230-seal results/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1_20260825/SHA256SUMS \
  --m262-result results/m262_fc1_descriptor_lifecycle_trace_r2_exact_20260825/trace_payload/m262_fc1_descriptor_lifecycle_trace_r1.json \
  --m262-seal results/m262_fc1_descriptor_lifecycle_trace_r2_exact_20260825/SHA256SUMS \
  --m292-result results/m292_m287_scope_corrected_amdahl_overlay_r1_20260825/m292_m287_scope_corrected_amdahl_overlay_r1.json \
  --m292-seal results/m292_m287_scope_corrected_amdahl_overlay_r1_20260825/SHA256SUMS \
  --contract "$task_run/wrong_sha_contract.json" \
  --docs359 docs/359_DATE终局冻结_20260813.md \
  --output-dir "$task_run/wrong_sha_output" \
  > "$task_run/wrong_sha.stdout.log" 2> "$task_run/wrong_sha.stderr.log"
task_wrong_rc=$?
set -e
printf '%s\n' "$task_wrong_rc" > "$task_run/wrong_sha.rc"
[[ $task_wrong_rc -ne 0 && ! -e "$task_run/wrong_sha_output" ]] || exit 20
grep -q 'frozen input identity drift' "$task_run/wrong_sha.stderr.log" || exit 21

python3 system_simulator/scripts/analyze_m480_fc1_fullwidth_resource_matched_context_factorized_dse.py \
  --manifest results/m51_h67_ep35_binary_input_trace_r2_gpu_receipt_20260823/manifest.json \
  --payload-root system_handoff/incoming/m51_capture_bundle_r2_20260823 \
  --m230-result results/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1_20260825/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1.json \
  --m230-seal results/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1_20260825/SHA256SUMS \
  --m262-result results/m262_fc1_descriptor_lifecycle_trace_r2_exact_20260825/trace_payload/m262_fc1_descriptor_lifecycle_trace_r1.json \
  --m262-seal results/m262_fc1_descriptor_lifecycle_trace_r2_exact_20260825/SHA256SUMS \
  --m292-result results/m292_m287_scope_corrected_amdahl_overlay_r1_20260825/m292_m287_scope_corrected_amdahl_overlay_r1.json \
  --m292-seal results/m292_m287_scope_corrected_amdahl_overlay_r1_20260825/SHA256SUMS \
  --contract contracts/m480_fc1_fullwidth_resource_matched_context_factorized_dse_contract_r1_20260826.json \
  --docs359 docs/359_DATE终局冻结_20260813.md \
  --output-dir "$task_run/payload" \
  > "$task_run/run.stdout.log" 2> "$task_run/run.stderr.log"
[[ ! -s "$task_run/run.stderr.log" ]] || exit 30
grep -q '^PASS M480 points=108 ' "$task_run/run.stdout.log" || exit 31

python3 - "$task_run/payload/m480_fc1_fullwidth_resource_matched_context_factorized_dse_r2.json" \
  "$task_run/payload/m480_fc1_fullwidth_resource_matched_context_factorized_dse_r2.csv" <<'PY'
import csv,json,sys
row=json.load(open(sys.argv[1]))
assert row["status"]=="PASS_EXACT_MASK_108_POINT_CPU_DSE_NO_PERFORMANCE_ADMISSION"
assert len(row["points"])==108
assert all(row["reconciliations"].values())
assert row["decision"]["m230_m262_ratios_multiplied"] is False
assert row["decision"]["rtl_promotion"] is False
assert row["decision"]["compact_fullwidth_gate_point"]=="L96_F2_C16_B2"
assert row["scope_partition"]["excluded_stage3_nonbinary_fc1_cycles"]==17474490
assert row["scope_partition"]["partition_conserves"] is True
for point in row["points"]:
    assert point["resource"]["resource_identical_between_modes"] is True
    assert point["admission"]["system_speedup"] is False
    assert point["admission"]["headline"] is False
assert row["admission"]["system_speedup"] is False
assert row["admission"]["headline"] is False
with open(sys.argv[2],newline="") as handle:
    rows=list(csv.DictReader(handle))
assert len(rows)==108
assert all(value["system_speedup"]=="False" for value in rows)
assert all(value["headline"]=="False" for value in rows)
PY

sha256sum "$task_runner" > "$task_run/runner_sha256.txt"
printf '%s\n' 'PASS_M480_EXACT_MASK_108_POINT_CPU_DSE_NO_PERFORMANCE_ADMISSION' \
  > "$task_run/RUN_COMPLETE.txt"
find "$task_run" -type f ! -name SHA256SUMS ! -name SHA256SUMS.seal.sha256 \
  -print0 | sort -z | xargs -0 sha256sum > "$task_run/SHA256SUMS"
(
  cd "$task_run"
  sha256sum -c SHA256SUMS
  sha256sum SHA256SUMS > SHA256SUMS.seal.sha256
  sha256sum -c SHA256SUMS.seal.sha256
)
task_complete=1
echo "PASS M480 exact-mask DSE sealed at $task_run"
