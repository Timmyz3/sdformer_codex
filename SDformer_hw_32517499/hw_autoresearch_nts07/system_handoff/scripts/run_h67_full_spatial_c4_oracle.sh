#!/usr/bin/env bash
set -euo pipefail

task_repo_root=$(pwd)
task_python=/opt/conda/envs/sdformerflow/bin/python
task_run_dir=hw_autoresearch_nts07/results/h67_ep35_full_spatial_c4_s1_v2_20260822
task_config=neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml
task_checkpoint=neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth
task_dependency=hw_autoresearch_nts07/results/m13_inputs_20260822/dependency_audit_r8.json
task_dependency_root=hw_autoresearch_nts07/results/m18_inputs_20260822/h67_ep35_dependency_dag_s1_r7_v2_persistent_20260822/dependency
task_dependency_manifest=${task_dependency_root}/manifest.json
task_dependency_events=${task_dependency_root}/dependency_events.jsonl
task_profiler=neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py
task_analyzer=hw_autoresearch_nts07/system_simulator/scripts/analyze_m17_full_spatial_c4_oracle.py
task_sealer=hw_autoresearch_nts07/system_simulator/scripts/seal_m17_full_spatial_c4_evidence.py
task_archive=hw_autoresearch_nts07/results/h67_ep35_full_spatial_c4_s1_v2_20260822.m17-evidence.tar

if [[ -e "${task_run_dir}/m17_reconciliation.json" ]]; then
    echo "refusing to overwrite completed M17 evidence: ${task_run_dir}" >&2
    exit 2
fi
mkdir -p "${task_run_dir}"
task_command=(
    "${task_python}" -u "${task_profiler}"
    --config "${task_config}"
    --checkpoint "${task_checkpoint}"
    --output-dir "${task_run_dir}"
    --samples 1
    --num-workers 0
    --ordered-trace
    --dual-line-trace
    --full-spatial-c4-dir "${task_run_dir}/full_spatial_c4"
    --full-spatial-c4-dependency-audit "${task_dependency}"
    --full-spatial-c4-dependency-manifest "${task_dependency_manifest}"
    --full-spatial-c4-dependency-events "${task_dependency_events}"
)
printf '%s\n' "${task_command[@]}" > "${task_run_dir}/profile_cmdline.txt"
"${task_python}" -c \
    "from pathlib import Path; import platform, torch; Path('${task_run_dir}/profile_environment.txt').write_text('python='+platform.python_version()+'\ntorch='+torch.__version__+'\ncuda='+str(torch.version.cuda)+'\n')"
"${task_command[@]}" > "${task_run_dir}/console.log" 2>&1
task_analyzer_command=(
    "${task_python}" "${task_analyzer}"
    --oracle-dir "${task_run_dir}/full_spatial_c4" \
    --same-sample-source-ledger "${task_run_dir}/dual_line_operator_trace.csv" \
    --output "${task_run_dir}/m17_reconciliation.json"
)
printf '%s\n' "${task_analyzer_command[@]}" > "${task_run_dir}/analyzer_cmdline.txt"
"${task_analyzer_command[@]}"
sha256sum \
    "${task_run_dir}/full_spatial_c4/manifest.json" \
    "${task_run_dir}/full_spatial_c4/prototypes.json" \
    "${task_run_dir}/full_spatial_c4/ordered_stream.npz" \
    "${task_run_dir}/dual_line_operator_trace.csv" \
    "${task_run_dir}/m17_reconciliation.json" \
    > "${task_run_dir}/sha256sums.txt"
"${task_python}" "${task_sealer}" \
    --repo-root "${task_repo_root}" \
    --run-dir "${task_run_dir}" \
    --output "${task_archive}" \
    --input config "${task_config}" \
    --input dependency_audit "${task_dependency}" \
    --input dependency_manifest "${task_dependency_manifest}" \
    --input dependency_events "${task_dependency_events}"
echo "PASS H67 exact full-spatial C4 oracle"
