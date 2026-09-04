#!/usr/bin/env bash
# Queue monitor: auto-launch I13 when current training finishes.
set -euo pipefail

REPO="/root/private_data/work/sdformer_codex/SDformer"
STATE_FILE="$REPO/neuron_autoresearch/.queue_state"
LOG="$REPO/neuron_autoresearch/queue_monitor.log"
BASELINE_CKPT="experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"
I13_CONFIG="neuron_autoresearch/experiments/i13_soc_angular/configs/full.yml"
I13_ENTRYPOINT="neuron_autoresearch/experiments/i13_soc_angular/entrypoints/train.py"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# Check if H9h is running
H9H_PID=$(pgrep -f "h9h_stage0_stage2" 2>/dev/null | head -1 || true)
CURRENT_STATE=$(cat "$STATE_FILE" 2>/dev/null || echo "unknown")

if [ -n "$H9H_PID" ]; then
    if [ "$CURRENT_STATE" != "h9h_running" ]; then
        echo "h9h_running" > "$STATE_FILE"
        log "H9h running (PID=$H9H_PID), waiting..."
    fi
    exit 0
fi

# H9h is NOT running
if [ "$CURRENT_STATE" = "h9h_running" ]; then
    log "H9h COMPLETED! Launching I13 SOC+angular full training..."
    echo "launching_i13" > "$STATE_FILE"
fi

CURRENT_STATE=$(cat "$STATE_FILE" 2>/dev/null || echo "unknown")

if [ "$CURRENT_STATE" = "launching_i13" ]; then
    cd /root/private_data/work/SDformer
    nohup bash -c "
        SDFORMER_USE_MLFLOW=0 python $I13_ENTRYPOINT \
            --config $I13_CONFIG \
            --prev_runid $BASELINE_CKPT \
            > neuron_autoresearch/experiments/i13_soc_angular/results/i13_full_$(date +%Y%m%d_%H%M%S).log 2>&1
        echo 'i13_done' > neuron_autoresearch/.queue_state
    " &
    I13_PID=$!
    log "I13 launched (PID=$I13_PID)"
    echo "i13_running" > "$STATE_FILE"
elif [ "$CURRENT_STATE" = "i13_running" ]; then
    I13_PID=$(pgrep -f "i13_soc_angular" 2>/dev/null | head -1 || true)
    if [ -z "$I13_PID" ]; then
        log "I13 process not found, may have completed or crashed"
    else
        log "I13 still running (PID=$I13_PID)"
    fi
fi
