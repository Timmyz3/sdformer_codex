#!/usr/bin/env bash
set -euo pipefail
task_sim_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."&&pwd)"
task_hw_root="$(cd "$task_sim_root/.."&&pwd)"
task_runner="$(realpath "${BASH_SOURCE[0]}")"
task_run="$task_hw_root/results/m262_fc1_descriptor_lifecycle_trace_r2_exact_20260825"
[[ ! -e "$task_run" ]]||exit 2
mkdir -p "$(dirname "$task_run")"
mkdir "$task_run"
task_complete=0
trap 'task_rc=$?;if [[ $task_complete -ne 1 ]];then printf "status=FAILED_OR_INCOMPLETE_DO_NOT_CITE\nrunner_exit_code=%s\n" "$task_rc">"$task_run/RUN_FAILED_OR_INCOMPLETE.txt";fi' EXIT
cd "$task_hw_root"
declare -A task_expected=(
 ["system_simulator/scripts/analyze_m262_fc1_descriptor_lifecycle_trace.py"]="f6229fe94844b17ff994b4bee9f3ae2c7a73d18a4624adee63507ba252f2b2d9"
 ["contracts/m262_fc1_descriptor_lifecycle_vcs_trace_contract_r1_20260825.json"]="b1bbdee8d0b151af094eef9378b50936358695eced1553398d13a908dc824415"
 ["results/m262_fc1_descriptor_lifecycle_directed_vcs_r3_exact_20260825/SHA256SUMS"]="f60f3fa5639d7e9410a081afd6e285a7d83443867f2f0f4b110e4c4956450245"
 ["results/m262_fc1_descriptor_lifecycle_directed_vcs_r3_exact_20260825/m262_vcs_receipt_r1.txt"]="8509ce1f3ff8b9993934adf733455a4cded3e60b533b5db63bf590c8d091a960"
 ["results/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1_20260825/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1.json"]="6110dff1cac748ca934e05033ddabe39f06e8b54286699a7843c209ddfe4a6ca"
 ["results/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1_20260825/SHA256SUMS"]="133c32c37d6ff61d19ca119634b5604d8a9fe12dd510cd4d9425e59e967247e5"
 ["results/m230_independent_hammer_review_r1_20260825/SHA256SUMS"]="7b8e904a873d2b2abf95667a3b6dcff100400f2127db661cd59074905eddadc4"
 ["docs/359_DATE终局冻结_20260813.md"]="dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
)
:>"$task_run/preflight_sha_checks.txt"
for task_path in "${!task_expected[@]}";do
 task_observed="$(sha256sum "$task_path"|awk '{print $1}')"
 printf 'path=%s expected=%s observed=%s\n' "$task_path" \
  "${task_expected[$task_path]}" "$task_observed">>"$task_run/preflight_sha_checks.txt"
 [[ "$task_observed" == "${task_expected[$task_path]}" ]]||exit 10
done
sha256sum "${!task_expected[@]}">"$task_run/input_sha256.txt"

cp contracts/m262_fc1_descriptor_lifecycle_vcs_trace_contract_r1_20260825.json \
 "$task_run/wrong_sha_contract.json"
printf '\n'>>"$task_run/wrong_sha_contract.json"
set +e
python3 system_simulator/scripts/analyze_m262_fc1_descriptor_lifecycle_trace.py \
 --m230-result results/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1_20260825/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1.json \
 --m230-seal results/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1_20260825/SHA256SUMS \
 --m230-review-seal results/m230_independent_hammer_review_r1_20260825/SHA256SUMS \
 --m262-vcs-seal results/m262_fc1_descriptor_lifecycle_directed_vcs_r3_exact_20260825/SHA256SUMS \
 --m262-vcs-receipt results/m262_fc1_descriptor_lifecycle_directed_vcs_r3_exact_20260825/m262_vcs_receipt_r1.txt \
 --m262-contract "$task_run/wrong_sha_contract.json" \
 --docs359 docs/359_DATE终局冻结_20260813.md \
 --output-dir "$task_run/wrong_sha_output" \
 >"$task_run/wrong_sha.stdout.log" 2>"$task_run/wrong_sha.stderr.log"
task_wrong_rc=$?
set -e
printf '%s\n' "$task_wrong_rc">"$task_run/wrong_sha.rc"
[[ $task_wrong_rc -ne 0&&! -e "$task_run/wrong_sha_output" ]]||exit 20
grep -q 'frozen input identity drift' "$task_run/wrong_sha.stderr.log"||exit 21

python3 system_simulator/scripts/analyze_m262_fc1_descriptor_lifecycle_trace.py \
 --m230-result results/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1_20260825/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1.json \
 --m230-seal results/m230_h67_fc1_m229_fixed_latency_trace_recurrence_r1_20260825/SHA256SUMS \
 --m230-review-seal results/m230_independent_hammer_review_r1_20260825/SHA256SUMS \
 --m262-vcs-seal results/m262_fc1_descriptor_lifecycle_directed_vcs_r3_exact_20260825/SHA256SUMS \
 --m262-vcs-receipt results/m262_fc1_descriptor_lifecycle_directed_vcs_r3_exact_20260825/m262_vcs_receipt_r1.txt \
 --m262-contract contracts/m262_fc1_descriptor_lifecycle_vcs_trace_contract_r1_20260825.json \
 --docs359 docs/359_DATE终局冻结_20260813.md \
 --output-dir "$task_run/trace_payload" \
 >"$task_run/trace.stdout.log" 2>"$task_run/trace.stderr.log"
grep -qx 'PASS M262 trace bit/dense=7.199783 factor/dense=12.039764 factor/bit=1.672240' \
 "$task_run/trace.stdout.log"||exit 30
[[ ! -s "$task_run/trace.stderr.log" ]]||exit 31
python3 - "$task_run/trace_payload/m262_fc1_descriptor_lifecycle_trace_r1.json" <<'PY'
import json,sys
row=json.load(open(sys.argv[1]))
assert row["status"]=="PASS_M262_EXACT_AGGREGATE_SMALL_WIDTH_LIFECYCLE_MAPPING"
assert row["aggregate_input"]["group_streams"]==4320000
assert row["aggregate_input"]["empty_group_streams"]==148932
assert row["aggregate_input"]["source_context_updates"]==1010523752
assert row["aggregate_input"]["unique_source_weight_reads"]==391666724
assert len(row["per_record"])==100
assert row["admission"]["module_lifecycle_cycle_mapping"] is True
for key in ("full_96_lane_rtl","full_trace_rtl_replay","physical_sram",
            "macro_ppa","complete_fc1","complete_ffn","system_speedup","headline"):
    assert row["admission"][key] is False
PY
{
 echo status=PASS_M262_EXACT_SMALL_WIDTH_TRACE_MAPPING
 echo exact_sha=true
 echo wrong_sha_fail_closed=true
 echo frozen_records=100
 echo serialized_8lane_slices_per_96lane_block=12
 echo bit_sparse_vs_dense_module_lifecycle_cycles=7.199782509882893
 echo context_factorized_vs_dense_module_lifecycle_cycles=12.039763991479225
 echo context_factorized_vs_bit_sparse_module_lifecycle_cycles=1.6722399565476673
 echo context_factorized_weight_read_reduction_vs_bit_sparse=2.580060265727348
 echo full_96_lane_rtl=false
 echo full_trace_rtl=false
 echo physical_sram=false
 echo macro_ppa=false
 echo complete_fc1=false
 echo complete_ffn=false
 echo system_speedup=false
 echo headline=false
}>"$task_run/m262_trace_receipt_r1.txt"
sha256sum "$task_runner">"$task_run/runner_sha256.txt"
find "$task_run" -type f ! -name SHA256SUMS -print0|sort -z|xargs -0 sha256sum \
 >"$task_run/SHA256SUMS"
echo PASS_M262_EXACT_SMALL_WIDTH_TRACE_MAPPING>"$task_run/RUN_COMPLETE.txt"
task_complete=1
echo "PASS M262 exact trace mapping sealed at $task_run"
