#!/bin/bash
# H12 queue #3: v2 迭代标定 probe（h12cal2）。
# 600 步 probe：验证 v2 闭环标定能把 firing 压回 r_target 附近、AEE 恢复正常量级，
# 通过后再排 5-epoch 全量。GPU 一空闲即启动。
set -u

H12=/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H12_atlif_gradfix
H9=/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention
PY=/opt/conda/envs/sdformerflow/bin/python
WS=$H9/results/t49_ternary_ws4_ep5/warmstart/checkpoint_epoch34.pth
CFG=$H12/configs/generated/h12cal2_probe.yml
RUN=$H12/results/h12cal2_probe

mkdir -p "$RUN"

echo "$(date '+%F %T') [h12cal2queue] waiting for GPU < 2000 MiB (3 consecutive checks, 60s apart)" >> "$RUN/queue.log"
ok=0
while [ "$ok" -lt 3 ]; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "${used:-99999}" -lt 2000 ]; then ok=$((ok+1)); else ok=0; fi
  echo "$(date '+%F %T') used=${used}MiB ok=$ok" >> "$RUN/queue.log"
  if [ "$ok" -lt 3 ]; then sleep 60; fi
done

echo "$(date '+%F %T') [h12cal2queue] launching v2 probe" >> "$RUN/queue.log"
setsid nohup env SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 \
  SDFORMER_SNN_BACKEND=cupy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  H12_CALIB_ROUNDS=5 H12_CALIB_BATCHES=30 H12_CALIB_RTARGET=0.25 \
  H12_CALIB_OUT=$RUN/theta_table.json \
  "$PY" -u "$H12/entrypoints/calibrate_and_train.py" \
  --config "$CFG" \
  --prev-runid "$WS" \
  --save-path "$RUN/checkpoint_epoch{}.pth" \
  --calib-batches 30 --r-target 0.25 --calib-rounds 5 --theta-out "$RUN/theta_table.json" \
  >> "$RUN/train.log" 2>&1 &
PROBE_PID=$!
echo "$(date '+%F %T') [h12cal2queue] launched pid=$PROBE_PID" >> "$RUN/queue.log"

# 等 probe 训练进程退出后立刻做 profile（valid40，快）
wait "$PROBE_PID" 2>/dev/null
echo "$(date '+%F %T') [h12cal2queue] probe exited, profiling" >> "$RUN/queue.log"
( cd /root/private_data/work/sdformer_codex/SDformer && \
  env SDFORMER_USE_MLFLOW=0 SDFORMER_SNN_BACKEND=cupy \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$PY" -u "$H9/entrypoints/profile_checkpoints.py" \
    --config "$CFG" --run-dir "$RUN" --epoch 35 \
    >> "$RUN/profile.log" 2>&1 )
echo "$(date '+%F %T') [h12cal2queue] profile done rc=$?" >> "$RUN/queue.log"
grep -h "AEE=" "$RUN/profile.log" >> "$RUN/queue.log" 2>/dev/null
echo "$(date '+%F %T') [h12cal2queue] all done" >> "$RUN/queue.log"
