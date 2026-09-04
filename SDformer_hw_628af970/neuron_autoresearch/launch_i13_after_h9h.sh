#!/usr/bin/env bash
# Persistent background monitor: wait for H9h to finish, then launch I13.
# Runs independently of Claude Code — survives terminal close.
set -euo pipefail

REPO="/root/private_data/work/sdformer_codex/SDformer"
LOG="$REPO/neuron_autoresearch/launcher.log"
BASELINE_CKPT="experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Queue launcher started ==="
log "Waiting for H9h to finish..."

# Wait for H9h python training process to exit (match python, not tail or ourselves)
while pgrep -f "python.*h9h_stage0_stage2.*train.py" > /dev/null 2>&1; do
    sleep 60
done

log "H9h finished! Launching I13 SOC+angular full training..."

cd /root/private_data/work/SDformer

source /opt/conda/etc/profile.d/conda.sh && conda activate sdformerflow
SDFORMER_USE_MLFLOW=0 python neuron_autoresearch/experiments/i13_soc_angular/entrypoints/train.py \
    --config neuron_autoresearch/experiments/i13_soc_angular/configs/full.yml \
    --prev_runid "$BASELINE_CKPT" \
    > "neuron_autoresearch/experiments/i13_soc_angular/results/i13_full_${TIMESTAMP}.log" 2>&1

I13_EXIT=$?
if [ $I13_EXIT -eq 0 ]; then
    log "I13 training completed successfully (exit=$I13_EXIT)"
else
    log "I13 training exited with code $I13_EXIT"
fi
