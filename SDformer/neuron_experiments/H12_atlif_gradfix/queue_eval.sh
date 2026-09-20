#!/bin/bash
# H12 eval queue: profile_checkpoints (valid825 AEE) for h12cal and h12s0.
# 顺序：等 GPU 空 → 评 h12cal ep35-39 → 等 h12s0 训练进程退出 → 评 h12s0 ep35-39。
set -u

H12=/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H12_atlif_gradfix
H9=/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention
PY=/opt/conda/envs/sdformerflow/bin/python
LOG=$H12/results/eval_queue.log

wait_gpu_free() {
  local ok=0 used
  while [ "$ok" -lt 3 ]; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "${used:-99999}" -lt 2000 ]; then ok=$((ok+1)); else ok=0; fi
    echo "$(date '+%F %T') gpu_used=${used}MiB ok=$ok" >> "$LOG"
    [ "$ok" -lt 3 ] && sleep 60
  done
}

run_profile() {  # $1=tag $2=run_dir $3=config
  local tag=$1 rd=$2 cfg=$3
  echo "$(date '+%F %T') [evalqueue] profiling $tag" >> "$LOG"
  ( cd /root/private_data/work/sdformer_codex/SDformer && \
    env SDFORMER_USE_MLFLOW=0 SDFORMER_SNN_BACKEND=cupy \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$PY" -u "$H9/entrypoints/profile_checkpoints.py" \
      --config "$cfg" --run-dir "$rd" \
      --epoch 35 --epoch 36 --epoch 37 --epoch 38 --epoch 39 \
      >> "$H12/results/${tag}_profile.log" 2>&1 )
  echo "$(date '+%F %T') [evalqueue] $tag done rc=$?" >> "$LOG"
}

echo "$(date '+%F %T') [evalqueue] start" >> "$LOG"

# 1) h12cal（训练已完成）
wait_gpu_free
run_profile h12cal_ternary_ws4_ep5 "$H12/results/h12cal_ternary_ws4_ep5" "$H12/configs/generated/h12cal_ternary_ws4_ep5.yml"

# 2) 等 h12s0 训练退出
echo "$(date '+%F %T') [evalqueue] waiting for h12s0 training to exit" >> "$LOG"
while pgrep -f "h12s0_local5_ternary_ep5" > /dev/null 2>&1; do
  sleep 120
done
echo "$(date '+%F %T') [evalqueue] h12s0 exited" >> "$LOG"
wait_gpu_free
run_profile h12s0_local5_ternary_ep5 "$H12/results/h12s0_local5_ternary_ep5" "$H12/configs/generated/h12s0_local5_ternary_ep5.yml"

echo "$(date '+%F %T') [evalqueue] all done" >> "$LOG"
