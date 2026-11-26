#!/usr/bin/env bash
# T57: 官方 θ 梯度 vs 修正 θ 梯度的短训 A/B（单变量：只有 backward 的 grad_thre 不同）。
#
# 目的：T56 是在 6 个样本上离线测的（无 AMP）。这里在**真实训练回路**里跑同样步数，
# 直接读 `threshold_mean` 漂移，确认「修正后 θ 真的会动、官方冻住」不是离线假象。
#
# 两条臂除了 monkeypatch 完全同参：
#   binary + official_atlif + threshold_eta=0（源配置）⇒ 手动项①恒为 0，
#   阈值冻结 `threshold_freeze_after_step: 1224` 也只冻结手动项①，
#   ⇒ **θ 的唯一来源就是优化器梯度路径③**，归因干净。
set -u
PY=/opt/conda/envs/sdformerflow/bin/python
LR=2.5e-5
STEPS=1200
R=/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention
D_FIX=$R/results/t57_grad_fix

echo "=== T57 gradient smoke start $(date -Is) ==="

echo "--- t57_grad_ctrl (official gradient) $(date -Is) ---"
"${PY}" -u /root/t49_ternary_retrain.py --name t57_grad_ctrl --epochs 1 --binary-official \
    --backbone-lr "${LR}" --max-steps "${STEPS}"
echo "--- t57_grad_ctrl rc=$? ---"

echo "--- t57_grad_fix (corrected gradient via monkeypatch) $(date -Is) ---"
"${PY}" -u /root/t49_ternary_retrain.py --name t57_grad_fix --epochs 1 --binary-official \
    --backbone-lr "${LR}" --max-steps "${STEPS}" --dry-run
echo "--- t57_grad_fix dry-run rc=$? ---"
env SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0 SDFORMER_SNN_BACKEND=cupy \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "${PY}" -u /root/t57_gradfix_train.py \
        --config "$R/configs/generated/t57_grad_fix.yml" \
        --prev_runid "${D_FIX}/warmstart/checkpoint_epoch34.pth" \
        --save_path "${D_FIX}/checkpoint_epoch{}.pth" \
        --finetune 1 > "${D_FIX}/train.log" 2>&1
echo "--- t57_grad_fix rc=$? ---"

echo "=== threshold_mean 漂移 ==="
"${PY}" - <<'EOF'
import re, ast
from pathlib import Path
R = Path("/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/"
         "H9_bipolar_self_attention/results")
for name in ("t57_grad_ctrl", "t57_grad_fix"):
    log = R / name / "train.log"
    if not log.exists():
        print(f"{name}: NO LOG"); continue
    txt = log.read_text(errors="replace").replace("\r", "\n")
    steps = []
    for m in re.finditer(r"\[H9\] step (\d+) update: (\{.*\})", txt):
        try:
            d = ast.literal_eval(m.group(2))
        except Exception:
            continue
        steps.append((int(m.group(1)), d))
    if not steps:
        tail = "\n".join(txt.splitlines()[-6:])
        print(f"{name}: no step lines; tail=\n{tail}"); continue
    s0, d0 = steps[0]
    s1, d1 = steps[-1]
    print(f"{name}: n_logged={len(steps)}")
    print(f"  step {s0}: mean={d0['threshold_mean']:.10f} min={d0['threshold_min']} "
          f"max={d0['threshold_max']:.6f} raw_update={d0.get('raw_update_mean')}")
    print(f"  step {s1}: mean={d1['threshold_mean']:.10f} min={d1['threshold_min']} "
          f"max={d1['threshold_max']:.6f} raw_update={d1.get('raw_update_mean')}")
    print(f"  Δmean = {d1['threshold_mean'] - d0['threshold_mean']:+.10f}")
EOF
echo "=== T57 COMPLETE $(date -Is) ==="
