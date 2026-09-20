#!/bin/bash
# H12 queue #2: stage0-local5 experiment (h12s0_local5_ternary_ep5).
# 单变量 = stage0 attention 换 local5 stencil；其余与 h12cal 完全一致
#（三元神经元、θ 标定、H12 修复 backward、t49 ep34 warm start、5 epoch）。
# 排在 h12cal 之后：等 GPU 空闲（<2000MiB 连续 3 次）再启动。
set -u

H12=/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H12_atlif_gradfix
PY=/opt/conda/envs/sdformerflow/bin/python
H9=/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention
WS=$H9/results/t49_ternary_ws4_ep5/warmstart/checkpoint_epoch34.pth
CFG=$H12/configs/generated/h12s0_local5_ternary_ep5.yml
RUN=$H12/results/h12s0_local5_ternary_ep5

mkdir -p "$RUN"

# 生成配置（幂等）
"$PY" "$H12/entrypoints/build_stage0_config.py" >> "$RUN/queue.log" 2>&1

echo "$(date '+%F %T') [h12s0queue] waiting for GPU < 2000 MiB (3 consecutive checks, 60s apart)" >> "$RUN/queue.log"
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
    sleep 60
  fi
done

echo "$(date '+%F %T') [h12s0queue] GPU free, launching h12s0_local5_ternary_ep5" >> "$RUN/queue.log"
setsid nohup env SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 \
  SDFORMER_SNN_BACKEND=cupy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  H12_CALIB_BATCHES=60 H12_CALIB_RTARGET=0.25 H12_CALIB_OUT=$RUN/theta_table.json \
  "$PY" -u "$H12/entrypoints/calibrate_and_train.py" \
  --config "$CFG" \
  --prev-runid "$WS" \
  --save-path "$RUN/checkpoint_epoch{}.pth" \
  --calib-batches 60 --r-target 0.25 --theta-out "$RUN/theta_table.json" \
  >> "$RUN/train.log" 2>&1 &
echo "$(date '+%F %T') [h12s0queue] launched pid=$!" >> "$RUN/queue.log"
