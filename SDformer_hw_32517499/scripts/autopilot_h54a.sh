#!/usr/bin/env bash
# H54a autopilot: wait for H56a → profile → H54a sweep → full30
# Run from: /root/private_data/work/sdformer_codex/SDformer
set -euo pipefail

export SDFORMER_USE_MLFLOW=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source /opt/conda/etc/profile.d/conda.sh
conda activate sdformerflow

REPO="/root/private_data/work/sdformer_codex/SDformer"
RESULTS="$REPO/neuron_experiments/H9_bipolar_self_attention/results"
BASELINE="$REPO/experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"
NOW=$(date +%Y%m%d_%H%M%S)
AUTOPILOT_DIR="$RESULTS/h54a_autopilot_$NOW"
mkdir -p "$AUTOPILOT_DIR"

log() { echo "[autopilot $(date +%H:%M:%S)] $*" | tee -a "$AUTOPILOT_DIR/autopilot.log"; }

# ── Phase 0: wait for H56a to finish ──────────────────────────────
H56A_DIR="$RESULTS/h56a_best_full30_20260528"
log "Phase 0: waiting for H56a full30 to finish..."
while true; do
    last_epoch=$(grep -E "^Epoch [0-9]+$" "$H56A_DIR/train.log" 2>/dev/null | tail -1 | awk '{print $2}')
    if [ "$last_epoch" = "29" ]; then
        # Check if validation for epoch 29 is done
        val_done=$(grep -c "Epoch loss (Validation)" "$H56A_DIR/train.log" 2>/dev/null || echo 0)
        if [ "$val_done" -ge 30 ]; then
            log "H56a epoch 29 validation done. Proceeding."
            break
        fi
    fi
    log "  H56a at epoch ${last_epoch:-?}/30, waiting 5min..."
    sleep 300
done

# ── Phase 1: profile H56a + J62a checkpoints (valid816) ──────────
log "Phase 1: profiling completed experiments..."

PROFILE_DIR="$AUTOPILOT_DIR/profiles"
mkdir -p "$PROFILE_DIR"

profile_checkpoint() {
    local name="$1" config="$2" ckpt="$3" samples="${4:-816}"
    local out="$PROFILE_DIR/${name}_valid${samples}"
    mkdir -p "$out"
    log "  profiling $name (samples=$samples)..."
    python -u entrypoints/profile_sops.py \
        --config "$config" \
        --checkpoint "$ckpt" \
        --output-dir "$out" \
        --split valid \
        --num-samples "$samples" \
        --batch-size 1 \
        --num-workers 4 \
        --metric AEE --metric AAE \
        > "$out/profile.log" 2>&1
    grep -E "AEE:|AAE:|sops:|firing:|global_firing|estimated_total" "$out/profile.log" | tail -5
}

# H56a config
H56A_CFG="$H56A_DIR/config.yml"
# J62a config
J62A_DIR="$RESULTS/j62a_full30_20260527_191500"
J62A_CFG="$J62A_DIR/config.yml"

# Profile H56a checkpoints (short valid40 first to rank, then valid816 on best)
for ep in 4 9 14 19 24 29; do
    ckpt="$H56A_DIR/checkpoint_epoch${ep}.pth"
    [ -f "$ckpt" ] || continue
    profile_checkpoint "h56a_ep${ep}" "$H56A_CFG" "$ckpt" 40
done

# Profile J62a checkpoints
for ep in 4 9 14 19 24 29; do
    ckpt="$J62A_DIR/checkpoint_epoch${ep}.pth"
    [ -f "$ckpt" ] || continue
    profile_checkpoint "j62a_ep${ep}" "$J62A_CFG" "$ckpt" 40
done

log "Phase 1 done. Top valid40 results:"
for f in "$PROFILE_DIR"/*valid40/profile.log; do
    name=$(basename "$(dirname "$f")")
    metrics=$(grep -E "AEE:|AAE:|estimated_total_sops:" "$f" | tr '\n' ' ')
    echo "  $name: $metrics"
done | sort -t: -k2 -n | head -20 | tee -a "$AUTOPILOT_DIR/autopilot.log"

# Pick best H56a epoch for valid816 profiling
BEST_H56A_EP=$(for f in "$PROFILE_DIR"/h56a_ep*valid40/profile.log; do
    name=$(basename "$(dirname "$f")")
    aee=$(grep "AEE:" "$f" | head -1 | awk '{print $2}')
    echo "$aee $name"
done | sort -n | head -1 | awk '{print $2}' | sed 's/h56a_ep//' | sed 's/_valid40//')
log "Best H56a epoch: $BEST_H56A_EP (valid816 profiling...)"

profile_checkpoint "h56a_ep${BEST_H56A_EP}" "$H56A_CFG" "$H56A_DIR/checkpoint_epoch${BEST_H56A_EP}.pth" 816

# ── Phase 2: H54a lambda sweep short tests (360 steps) ───────────
log "Phase 2: H54a lambda sweep..."

SWEEP_CONFIGS=(
    "h54a_swp_lam0p3_fast_warm_s360"
    "h54a_swp_lam0p3_slowbb_s360"
    "h54a_swp_lam0p5_fast_warm_s360"
    "h54a_swp_lam0p5_slowbb_s360"
    "h54a_swp_lam1p0_fast_warm_s360"
    "h54a_swp_lam1p0_slowbb_s360"
    "h54a_swp_lam2p0_fast_warm_s360"
    "h54a_swp_lam2p0_slowbb_s360"
)

SWEEP_DIR="$AUTOPILOT_DIR/sweep_runs"
mkdir -p "$SWEEP_DIR"

for cfg_name in "${SWEEP_CONFIGS[@]}"; do
    CFG_PATH="neuron_experiments/H9_bipolar_self_attention/configs/generated/${cfg_name}.yml"
    RUN_DIR="$SWEEP_DIR/$cfg_name"
    mkdir -p "$RUN_DIR"
    log "  running $cfg_name..."
    python -u entrypoints/train.py \
        --config "$CFG_PATH" \
        --prev_runid "$BASELINE" \
        --save_path "${RUN_DIR}/checkpoint_epoch{}.pth" \
        > "${RUN_DIR}/train.log" 2>&1

    # Profile epoch 0 with valid40
    [ -f "${RUN_DIR}/checkpoint_epoch0.pth" ] || { log "  SKIP $cfg_name: no checkpoint"; continue; }
    profile_checkpoint "${cfg_name}" "$CFG_PATH" "${RUN_DIR}/checkpoint_epoch0.pth" 40
done

log "Phase 2 done. Sweep results:"
for f in "$PROFILE_DIR"/h54a_swp_lam*valid40/profile.log; do
    name=$(basename "$(dirname "$f")")
    aee=$(grep "AEE:" "$f" | head -1 | awk '{print $2}')
    aae=$(grep "AAE:" "$f" | head -1 | awk '{print $2}')
    sops=$(grep "estimated_total_sops:" "$f" | awk '{print $2}')
    echo "  $name: AEE=$aee AAE=$aae SOPs=$sops"
done | tee -a "$AUTOPILOT_DIR/autopilot.log"

# ── Phase 3: pick best config, generate full30, launch ───────────
log "Phase 3: selecting best H54a config for full30..."

BEST_SWEEP=$(for f in "$PROFILE_DIR"/h54a_swp_lam*valid40/profile.log; do
    name=$(basename "$(dirname "$f")")
    aee=$(grep "AEE:" "$f" | head -1 | awk '{print $2}')
    # Score: lower AEE is better
    echo "$aee $name"
done | sort -n | head -1 | awk '{print $2}' | sed 's/_valid40//')

log "Best sweep config: $BEST_SWEEP"

# Extract lambda and LR from config name
BEST_LAM=$(echo "$BEST_SWEEP" | grep -oP 'lam\K[0-9p]+' | sed 's/p/./')
BEST_LR=$(echo "$BEST_SWEEP" | grep -oP 'lam[0-9p]+_\K[a-z_]+(?=_s)')

log "Best: lambda=$BEST_LAM, LR=$BEST_LR"

# Generate full30 config from short config + H49 neuron stability recipe
FULL30_NAME="h54a_lam${BEST_LAM/./p}_${BEST_LR}_full30"
FULL30_CFG="neuron_experiments/H9_bipolar_self_attention/configs/generated/${FULL30_NAME}.yml"

log "Generating full30 config: $FULL30_CFG"
python - <<PYEOF
import yaml
from copy import deepcopy
from pathlib import Path

base_path = Path("neuron_experiments/H9_bipolar_self_attention/configs/generated/${BEST_SWEEP}.yml")
with open(base_path) as f:
    cfg = yaml.safe_load(f)

# Switch to full30 settings
cfg["experiment"] = "${FULL30_NAME}"
cfg["runtime"]["max_train_steps"] = 0  # no step limit
cfg["loader"]["n_epochs"] = 30
cfg["runtime"]["force_save_epochs"] = list(range(30))
cfg["note"] = (
    f"H54a full30: best sweep config (lambda=${BEST_LAM}, LR=${BEST_LR}). "
    + "Neuron: symmetric_target_rate + tr=0.07 for full30 stability. "
    + cfg.get("note", "")
)

# Switch neuron to symmetric_target_rate for full30 stability
atlif = cfg.setdefault("atlif_ternary_psn", {})
atlif["threshold_mode"] = "symmetric_target_rate"
atlif["target_rate"] = 0.07
atlif["target_rate_eta"] = 0.08
# Keep the per-stage threshold_eta values from H54a (they work well)
# but add target_rate feedback to prevent threshold drift

for grp in atlif.get("target_groups", []):
    if grp.get("name", "").startswith("qk"):
        grp["threshold_mode"] = "symmetric_target_rate"
        grp["target_rate"] = 0.07
        grp["target_rate_eta"] = 0.06

out_path = Path("${FULL30_CFG}")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
print(f"wrote {out_path}")
PYEOF

# Launch full30
FULL30_DIR="$RESULTS/${FULL30_NAME}_$NOW"
mkdir -p "$FULL30_DIR"
log "Launching H54a full30: $FULL30_NAME"
nohup python -u entrypoints/train.py \
    --config "$FULL30_CFG" \
    --prev_runid "$BASELINE" \
    --save_path "${FULL30_DIR}/checkpoint_epoch{}.pth" \
    > "${FULL30_DIR}/train.log" 2>&1 &

FULL30_PID=$!
log "H54a full30 launched (PID=$FULL30_PID) → $FULL30_DIR"

# ── Phase 4: wait for full30 → profile ───────────────────────────
log "Phase 4: waiting for full30 to finish..."
sleep 30  # let it start
while true; do
    last_epoch=$(grep -E "^Epoch [0-9]+$" "$FULL30_DIR/train.log" 2>/dev/null | tail -1 | awk '{print $2}')
    if [ "$last_epoch" = "29" ]; then
        val_done=$(grep -c "Epoch loss (Validation)" "$FULL30_DIR/train.log" 2>/dev/null || echo 0)
        if [ "$val_done" -ge 30 ]; then
            log "Full30 epoch 29 validation done."
            break
        fi
    fi
    log "  Full30 at epoch ${last_epoch:-?}/30, waiting 10min..."
    sleep 600
done

# Profile all checkpoints with valid40, then best with valid816
log "Profiling full30 checkpoints..."
for ep in 4 9 14 19 24 29; do
    ckpt="$FULL30_DIR/checkpoint_epoch${ep}.pth"
    [ -f "$ckpt" ] || continue
    profile_checkpoint "${FULL30_NAME}_ep${ep}" "$FULL30_CFG" "$ckpt" 40
done

# Find best epoch and do valid816
BEST_EP=$(for f in "$PROFILE_DIR"/${FULL30_NAME}_ep*valid40/profile.log; do
    name=$(basename "$(dirname "$f")")
    aee=$(grep "AEE:" "$f" | head -1 | awk '{print $2}')
    echo "$aee $name"
done | sort -n | head -1 | awk '{print $2}' | grep -oP 'ep\K[0-9]+')

log "Best full30 epoch: $BEST_EP"
profile_checkpoint "${FULL30_NAME}_ep${BEST_EP}" "$FULL30_CFG" \
    "$FULL30_DIR/checkpoint_epoch${BEST_EP}.pth" 816

# ── Final summary ─────────────────────────────────────────────────
log "============================================"
log "AUTOPILOT COMPLETE"
log "============================================"
log ""
log "=== H56a (SC agree/disagree) best valid816 ==="
grep -E "AEE:|AAE:|sops:" "$PROFILE_DIR/h56a_ep${BEST_H56A_EP}_valid816/profile.log" | tail -3 | tee -a "$AUTOPILOT_DIR/autopilot.log"

log ""
log "=== H54a sweep best valid40 ==="
grep -E "AEE:|AAE:|sops:" "$PROFILE_DIR/${BEST_SWEEP}_valid40/profile.log" | tail -3 | tee -a "$AUTOPILOT_DIR/autopilot.log"

log ""
log "=== H54a full30 best valid816 ==="
grep -E "AEE:|AAE:|sops:" "$PROFILE_DIR/${FULL30_NAME}_ep${BEST_EP}_valid816/profile.log" | tail -3 | tee -a "$AUTOPILOT_DIR/autopilot.log"

log ""
log "All results in: $AUTOPILOT_DIR"
log "Full30 training log: $FULL30_DIR/train.log"
