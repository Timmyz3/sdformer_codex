#!/bin/bash
# H12 queue #4: h12base probe —— θ=1.0 原样（不做任何标定），只隔离 H12 修复 backward。
# 依据：ep34 权重与 θ=1.0 共同训练适配，per-layer θ 重标定打碎协同（v1/v2 AEE≈8 实测）。
# 600 步 probe → 自动 profile，验证 AEE 恢复正常量级后再排 5-epoch 全量。
set -u

H12=/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H12_atlif_gradfix
H9=/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention
PY=/opt/conda/envs/sdformerflow/bin/python
WS=$H9/results/t49_ternary_ws4_ep5/warmstart/checkpoint_epoch34.pth
CFG=$H12/configs/generated/h12cal2_probe.yml   # 同一 probe 配置（600 步/1 epoch），但不跑标定
RUN=$H12/results/h12base_probe

mkdir -p "$RUN"

echo "$(date '+%F %T') [h12basequeue] waiting for GPU < 2000 MiB (3 consecutive checks, 60s apart)" >> "$RUN/queue.log"
ok=0
while [ "$ok" -lt 3 ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${used:-99999}" -lt 2000 ]; then ok=$((ok+1)); else ok=0; fi
  echo "$(date '+%F %T') used=${used}MiB ok=$ok" >> "$RUN/queue.log"
  if [ "$ok" -lt 3 ]; then sleep 60; fi
done

echo "$(date '+%F %T') [h12basequeue] launching h12base probe (no calibration, theta as loaded)" >> "$RUN/queue.log"
setsid nohup env SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 \
  SDFORMER_SNN_BACKEND=cupy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u "$H12/entrypoints/train.py" \
  --config "$CFG" \
  --prev_runid "$WS" \
  --save_path "$RUN/checkpoint_epoch{}.pth" \
  --finetune 1 \
  >> "$RUN/train.log" 2>&1 &
TRAIN_PID=$!
echo "$(date '+%F %T') [h12basequeue] launched pid=$TRAIN_PID" >> "$RUN/queue.log"

wait "$TRAIN_PID" 2>/dev/null
echo "$(date '+%F %T') [h12basequeue] probe exited, profiling" >> "$RUN/queue.log"
( cd /root/private_data/work/sdformer_codex/SDformer && \
  env SDFORMER_USE_MLFLOW=0 SDFORMER_SNN_BACKEND=cupy \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u "$H9/entrypoints/profile_checkpoints.py" \
    --config "$CFG" --run-dir "$RUN" --epoch 35 \
    >> "$RUN/profile.log" 2>&1 )
echo "$(date '+%F %T') [h12basequeue] profile done rc=$?" >> "$RUN/queue.log"
grep -h "AEE=" "$RUN/profile.log" >> "$RUN/queue.log" 2>/dev/null
echo "$(date '+%F %T') [h12basequeue] all done" >> "$RUN/queue.log"
