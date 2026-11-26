#!/usr/bin/env bash
# T57-N1b / N1c：把 BN 的 running 统计**重校准**到数据集上，再按 running 口径评测。
#
#   N1a（已完成）ep34 + running（直接用训练 EMA 统计）      AEE 1.30079 / Fl 6.61965
#   N1b           ep34 + running（**训练集**重校准统计）     ← 本脚本
#   N1c           ep34 + running（**valid** 重校准统计）     ← 直推式上界，仅诊断
#
# 动机：训练期 BN 一直走 batch 统计并更新 running，但 running 统计本身是 EMA
# （momentum 0.1）且只见过训练分布 ⇒ 直接用它对 valid 归一化是有偏的。
# 重校准把 running 统计变成目标数据集上的**精确均值/方差**（momentum=None 累积），
# 用来把"BN 统计量不准"和"固定统计这个部署口径本身不好"两件事分开。
set -u
PY=/opt/conda/envs/sdformerflow/bin/python
S=/root/private_data/work/sdformer_codex/SDformer
R=$S/neuron_experiments/H9_bipolar_self_attention
CFG=$R/configs/generated/dsec_c12_alpha0125_ep29_resume5_20260830.yml
CKPT=$R/results/dsec_c12_alpha0125_ep29_resume5_20260830/checkpoint_epoch34.pth
OUT=$R/results/t57_bn_probe

export SDFORMER_USE_MLFLOW=0 SDFORMER_MLFLOW_MODEL_LOGGING=0
export SDFORMER_SNN_BACKEND=cupy PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== T57 BN recal chain start $(date -Is) ==="

for SPLIT in train valid; do
    TAG=$([ "$SPLIT" = train ] && echo N1b || echo N1c)
    echo "--- [$TAG] recal on $SPLIT $(date -Is) ---"
    "${PY}" -u /root/t57_bn_recal.py \
        --config "$CFG" --checkpoint "$CKPT" \
        --mode recal --split "$SPLIT" --batches 300 \
        --out "$OUT/recal_${SPLIT}.pth" 2>&1 | tail -8
    echo "--- recal $SPLIT rc=${PIPESTATUS[0]} ---"

    echo "--- [$TAG] eval running @ recal_$SPLIT $(date -Is) ---"
    cd "$S" && "${PY}" -u third_party/SDformerFlow/eval_DSEC_flow_SNN.py \
        --config "$CFG" --checkpoint "$OUT/recal_${SPLIT}.pth" \
        --path_results "$OUT/ep34_running_recal_${SPLIT}" \
        --mode valid --bn-policy running > "$OUT/eval_recal_${SPLIT}.log" 2>&1
    echo "--- eval $SPLIT rc=$? ---"
done

echo "=== 汇总 ==="
"${PY}" - <<'EOF'
import json
from pathlib import Path
base = Path("/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/"
            "H9_bipolar_self_attention/results/t57_bn_probe")
rows = [("N0  anchor  no_running", base.parent / "dsec_c12_alpha0125_ep29_resume5_20260830"),
        ("N1a running raw        ", base / "ep34_running_raw"),
        ("N1b running recal train", base / "ep34_running_recal_train"),
        ("N1c running recal valid", base / "ep34_running_recal_valid")]
for name, d in rows:
    cands = list(d.glob("**/spike_profile.json")) if d.exists() else []
    if not cands:
        print(f"{name}: (no profile)"); continue
    j = json.load(open(sorted(cands)[0])); m = j.get("metrics", {})
    print(f"{name}: AEE={m.get('AEE')} Fl={m.get('DSEC_Fl')} "
          f"FR={j.get('global_firing_rate')} spikes={j.get('total_spikes')} "
          f"E={j.get('energy_uj')}")
EOF
echo "=== T57 BN RECAL COMPLETE $(date -Is) ==="
