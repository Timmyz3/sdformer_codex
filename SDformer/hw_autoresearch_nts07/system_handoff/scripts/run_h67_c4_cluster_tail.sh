#!/usr/bin/env bash
set -euo pipefail

task_repo_root=$(pwd)
task_python=/opt/conda/envs/sdformerflow/bin/python
task_config=neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml
task_checkpoint=neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth
task_profiler=neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py
task_sealer=hw_autoresearch_nts07/system_simulator/scripts/seal_h67_real_tile_evidence.py

for task_pairs in 256 512; do
    task_run_dir=hw_autoresearch_nts07/results/h67_ep35_real_tile_cluster_s1_p${task_pairs}_c4_v2_20260822
    if [[ -e "${task_run_dir}/evidence_manifest.json" ]]; then
        echo "refusing to overwrite sealed evidence: ${task_run_dir}" >&2
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
        --dual-line-tile-dir "${task_run_dir}/real_tiles"
        --dual-line-tile-pairs-per-call "${task_pairs}"
    )
    printf '%s\n' "${task_command[@]}" > "${task_run_dir}/profile_cmdline.txt"
    "${task_python}" -c \
        "from pathlib import Path; import platform, torch; Path('${task_run_dir}/profile_environment.txt').write_text('python='+platform.python_version()+'\ntorch='+torch.__version__+'\ncuda='+str(torch.version.cuda)+'\n')"
    "${task_command[@]}" > "${task_run_dir}/console.log" 2>&1
    "${task_python}" "${task_sealer}" \
        --run-dir "${task_run_dir}" \
        --repo-root "${task_repo_root}"
    echo "PASS cluster convergence P${task_pairs}"
done
