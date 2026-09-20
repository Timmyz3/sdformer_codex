#!/bin/bash
# H12 queue: waits for a mostly-free GPU (does NOT touch the t5x queue owned by
# the other session), then launches calibrate+train (theta calibration + 5-epoch
# ternary warm-start retrain on the H12 FIXED backward).
set -u

H12=/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H12_atlif_gradfix
PY=/opt/conda/envs/sdformerflow/bin/python
H9=/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention
SRC_CFG=$H9/configs/generated/t49_ternary_ws4_ep5.yml
WS=$H9/results/t49_ternary_ws4_ep5/warmstart/checkpoint_epoch34.pth
CFG=$H12/configs/generated/h12cal_ternary_ws4_ep5.yml
RUN=$H12/results/h12cal_ternary_ws4_ep5

mkdir -p "$RUN" "$H12/configs/generated"
if [ ! -f "$CFG" ]; then
  cp "$SRC_CFG" "$CFG"
fi

echo "$(date '+%F %T') [h12queue] waiting for GPU < 2000 MiB (3 consecutive checks, 30s apart)" >> "$RUN/queue.log"
ok=0
while [ "$ok" -lt 3 ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${used:-99999}" -lt 2000 ]; then
    ok=$((ok + 1))
  else
    ok=0
  fi
  echo "$(date '+%F %T') used=${used}MiB ok=$ok" >> "$RUN/queue.log"
  if [ "$ok" -lt 3 ]; then
    sleep 30
  fi
done

echo "$(date '+%F %T') [h12queue] GPU free, launching h12cal_ternary_ws4_ep5" >> "$RUN/queue.log"
setsid nohup env SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 \
  SDFORMER_SNN_BACKEND=cupy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  H12_CALIB_BATCHES=60 H12_CALIB_RTARGET=0.25 H12_CALIB_OUT=$RUN/theta_table.json \
  "$PY" -u "$H12/entrypoints/calibrate_and_train.py" \
  --config "$CFG" \
  --prev-runid "$WS" \
  --save-path "$RUN/checkpoint_epoch{}.pth" \
  --calib-batches 60 --r-target 0.25 --theta-out "$RUN/theta_table.json" \
  >> "$RUN/train.log" 2>&1 &
echo "$(date '+%F %T') [h12queue] launched pid=$!" >> "$RUN/queue.log"
