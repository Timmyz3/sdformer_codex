#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/root/private_data/work/sdformer_codex/SDformer"
PYTHON_BIN="/opt/conda/envs/sdformerflow/bin/python"
BASE_REL="neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml"
CONFIG_REL="neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_ep35_M29_rank3_factor_atlif_ft5_20260822.yml"
CHECKPOINT_REL="neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth"
RESULT_REL="neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_ep35_M29_rank3_factor_atlif_ft5_20260822"
WATCHER_LOCK="/tmp/sdformer_m29_h67_rank3_queue_20260822.lock"
ALGORITHM_LOCK="/tmp/sdformer_date_algorithm_evidence_queue_20260821.lock"
FACTORIAL_LOCK="/tmp/sdformer_date_fullres_factorial_controls_20260821.lock"
LOCAL5_LOCK="/tmp/sdformer_date_local5_same_parent_control_20260821.lock"
A800_LOCK="/tmp/sdformer_a800_training_global.lock"
M162_RECEIPT_REL="neuron_experiments/H9_bipolar_self_attention/results/m162_paft_ep4_bn_policy_ab_valid825_20260824/M162_COMPLETE.txt"

cd "$REPO_ROOT"
exec 209>"$WATCHER_LOCK"
if ! flock -n 209; then
    echo "M29 watcher already owns $WATCHER_LOCK" >&2
    exit 2
fi

verify_frozen_inputs() {
sha256sum --strict -c <<'EOF'
d9ee7e172f941a53ad1c031b0d5cdbbf7819f521c807e5bc54001a80c41b57f3  neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/atlif_ternary_psn.py
5873063b98eb4a267afa6513d03b86621f3fb6a885b310b4c5569ef5448ae657  neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/installer.py
f0e408c6bd136d7ce36b779881ca37a04de6f0cb6220701431b0a05b338f6d6b  neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/atlif_ternary_psn/__init__.py
172b3b8086cfe5c43bf9627fe92f947ca63148f9bbe8c50bca729b23c6273e68  neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN/h9_load_audit.py
49c77538f2de2c54b709b05ae246da4cf7f36a147da990a03acb9e94a917446b  neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py
04f692c5bda6d1f88cdc932ce48f012767f22a2bb1ca161378971232f99c0684  neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py
5dbe838cabca7a1b47f7c9e3abde54b6a947bbbb39677fa432ef5dc936e475a6  neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_sops.py
ba555e897bf915319bc9976ce40b1b47abd5cd341472e3bfba0a6e68777a222a  neuron_experiments/H9_bipolar_self_attention/entrypoints/make_m29_h67_rank3_factor_config.py
55cabb82c64f59c6d30e83ac2b07e395f6c3162eb4f065314391ad21bc12621a  neuron_experiments/H9_bipolar_self_attention/entrypoints/verify_m29_h67_rank3_launch.py
331ec9b6ad62193ebe693bf930875b1af8db43ca1e4afac4e77793a567cfd714  neuron_experiments/H9_bipolar_self_attention/entrypoints/test_m29_atlif_temporal_factorization.py
b5aa4245c7237399ea49c65c2daae827e05120f760f4a72cf224dd7525dfdc29  neuron_experiments/H9_bipolar_self_attention/entrypoints/make_m29_h67_rank3_run_receipt.py
8be3f7bbffd75c4356d3abf5935679d80e15c1caefd307c19a727729659e6c49  neuron_experiments/H9_bipolar_self_attention/configs/generated/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_hardware_order_q7q17_deploy.yml
4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158  neuron_experiments/H9_bipolar_self_attention/results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805/checkpoint_epoch35.pth
EOF
}

verify_frozen_inputs

"$PYTHON_BIN" -m unittest -q \
    neuron_experiments/H9_bipolar_self_attention/entrypoints/test_m29_atlif_temporal_factorization.py

"$PYTHON_BIN" \
    neuron_experiments/H9_bipolar_self_attention/entrypoints/make_m29_h67_rank3_factor_config.py \
    --force

# M87 is the Motion bottleneck-pattern PAFT line.  Its M162 valid825
# no-running/running A/B must finish before the independent temporal-rank M29
# line consumes the A800.  This is only a queue dependency: M87 does not supply
# M29 factors or admit the FFN Q8/rank-3 path.
while [[ ! -f "$M162_RECEIPT_REL" ]]; do
    sleep 60
done
grep -qx 'status=PASS_M162_PAFT_EP4_BN_POLICY_AB_VALID825' "$M162_RECEIPT_REL"

while pgrep -f 'run_date_algorithm_evidence_queue_20260821.py' >/dev/null \
    || pgrep -f 'run_date_fullres_factorial_controls_20260821.py' >/dev/null \
    || pgrep -f 'run_date_local5_same_parent_after_factorial_20260821.py' >/dev/null; do
    sleep 180
done

exec 210>"$ALGORITHM_LOCK"
exec 211>"$FACTORIAL_LOCK"
exec 212>"$LOCAL5_LOCK"
exec 213>"$A800_LOCK"
while ! flock -n 210 || ! flock -n 211 || ! flock -n 212 || ! flock -n 213; do
    flock -u 210 || true
    flock -u 211 || true
    flock -u 212 || true
    flock -u 213 || true
    sleep 180
done

while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]')" ]]; do
    sleep 180
done

verify_frozen_inputs
"$PYTHON_BIN" \
    neuron_experiments/H9_bipolar_self_attention/entrypoints/make_m29_h67_rank3_factor_config.py \
    --base "$BASE_REL" \
    --checkpoint "$CHECKPOINT_REL" \
    --output "$CONFIG_REL" \
    --force

if [[ -e "$RESULT_REL/checkpoint_epoch40.pth" ]]; then
    echo "M29 epoch40 already exists; refusing duplicate launch" >&2
    exit 3
fi
if compgen -G "$RESULT_REL/checkpoint_epoch*.pth" >/dev/null; then
    echo "M29 partial checkpoints exist; refusing unaudited overwrite/resume" >&2
    exit 4
fi

ATTEMPT_TAG="$(date -u +%Y%m%dT%H%M%SZ)_pid$$"
ATTEMPT_DIR="hw_autoresearch_nts07/system_handoff/m29_receipts/$ATTEMPT_TAG"
PREFLIGHT_REL="$ATTEMPT_DIR/preflight.json"
LAUNCH_RECEIPT_REL="$ATTEMPT_DIR/launch_receipt.json"
POSTFLIGHT_RECEIPT_REL="$ATTEMPT_DIR/postflight_receipt.json"
TRAIN_LOG_REL="$ATTEMPT_DIR/train.log"
mkdir -p "$ATTEMPT_DIR" "$RESULT_REL"

CUDA_VISIBLE_DEVICES='' "$PYTHON_BIN" \
    neuron_experiments/H9_bipolar_self_attention/entrypoints/verify_m29_h67_rank3_launch.py \
    --config "$CONFIG_REL" \
    --checkpoint "$CHECKPOINT_REL" \
    --receipt "${CONFIG_REL%.yml}.receipt.json" \
    --output "$PREFLIGHT_REL"

"$PYTHON_BIN" \
    neuron_experiments/H9_bipolar_self_attention/entrypoints/make_m29_h67_rank3_run_receipt.py \
    --phase launch \
    --config "$CONFIG_REL" \
    --source-checkpoint "$CHECKPOINT_REL" \
    --preflight "$PREFLIGHT_REL" \
    --result-dir "$RESULT_REL" \
    --train-log "$TRAIN_LOG_REL" \
    --watcher-pid "$$" \
    --output "$LAUNCH_RECEIPT_REL"

TRAIN_EXIT=0
"$PYTHON_BIN" -u \
    neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py \
    --config "$REPO_ROOT/$CONFIG_REL" \
    --prev_runid "$REPO_ROOT/$CHECKPOINT_REL" \
    --save_path "$REPO_ROOT/$RESULT_REL/checkpoint_epoch{}.pth" \
    --finetune 1 >"$TRAIN_LOG_REL" 2>&1 || TRAIN_EXIT=$?

"$PYTHON_BIN" \
    neuron_experiments/H9_bipolar_self_attention/entrypoints/make_m29_h67_rank3_run_receipt.py \
    --phase postflight \
    --config "$CONFIG_REL" \
    --source-checkpoint "$CHECKPOINT_REL" \
    --preflight "$PREFLIGHT_REL" \
    --result-dir "$RESULT_REL" \
    --train-log "$TRAIN_LOG_REL" \
    --launch-receipt "$LAUNCH_RECEIPT_REL" \
    --exit-code "$TRAIN_EXIT" \
    --watcher-pid "$$" \
    --output "$POSTFLIGHT_RECEIPT_REL"
exit "$TRAIN_EXIT"
