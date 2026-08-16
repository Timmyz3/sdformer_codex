#!/usr/bin/env bash
set -euo pipefail

REPO="/root/private_data/work/sdformer_codex/SDformer"
EXP="${REPO}/neuron_experiments/H9_bipolar_self_attention"
RESULTS="${EXP}/results"
PY="/opt/conda/envs/sdformerflow/bin/python"
LOG="${RESULTS}/algorithm_queue_resume_20260718.log"

export SDFORMER_USE_MLFLOW=0
export SDFORMER_MLFLOW_MODEL_LOGGING=0
export SDFORMER_SNN_BACKEND=cupy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${REPO}"
exec >>"${LOG}" 2>&1

record() {
    printf '[%s] %s\n' "$(date -Is)" "$*"
}

prune_ranked() {
    local pattern="$1"
    local dirs=()
    while IFS= read -r path; do
        dirs+=("${path}")
    done < <(find "${RESULTS}" -maxdepth 1 -type d -name "${pattern}" | sort)
    if ((${#dirs[@]})); then
        "${PY}" "${EXP}/entrypoints/prune_ranked_checkpoints.py" "${dirs[@]}"
    fi
}

H66D_RUN="${RESULTS}/h66d_allbinary_all12_lr_ttx_w720_fastlr_full30_bs8_full30_20260712_setsid"
H66D_CFG="${EXP}/configs/generated/h66d_allbinary_all12_lr_ttx_w720_fastlr_full30.yml"

record "RESUME algorithm queue: H66d -> H66e -> H81 -> H73-H80"
prune_ranked 'h66[abc]_*full30*setsid'

if [[ ! -f "${H66D_RUN}/checkpoint_epoch29.pth" ]]; then
    record "TRUE RESUME H66d from epoch24 model plus optimizer/scheduler/scaler state"
    "${PY}" -u "${EXP}/entrypoints/train.py" \
        --config "${H66D_CFG}" \
        --prev_runid "${H66D_RUN}/checkpoint_epoch24.pth" \
        --resume 1 \
        --save_path "${H66D_RUN}/checkpoint_epoch{}.pth"
fi

"${PY}" -u "${EXP}/entrypoints/run_h66_full30_after_h71.py"
prune_ranked 'h66[abcde]_*full30*setsid'

"${PY}" -u "${EXP}/entrypoints/run_h81_equal_budget_after_h66.py"
prune_ranked 'h81_*full30*setsid'

"${PY}" -u "${EXP}/entrypoints/run_match_code_after_h66.py"
prune_ranked 'h7[345]_*full30*setsid'

"${PY}" -u "${EXP}/entrypoints/run_round3_match_after_h75.py"
prune_ranked 'h7[678]_*full30*setsid'

"${PY}" -u "${EXP}/entrypoints/run_round4_assignment_after_h78.py"
prune_ranked 'h79_*full30*setsid'
prune_ranked 'h80_*full30*setsid'

record "ALL COMPLETE resumed algorithm queue"
