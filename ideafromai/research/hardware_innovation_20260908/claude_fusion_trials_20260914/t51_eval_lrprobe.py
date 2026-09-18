#!/usr/bin/env python3
"""T51: 对三个 LR 探针的 checkpoint 做 **valid825 正式评测**。

为什么不用训练内那个 `Epoch loss (Validation)` 选 LR：训练配置是
`test: n_valid: 1`，即只用 **1 个序列**的 loss，噪声太大；而且它是 600 步的
终点快照，跨 LR 可比性有限。`eval_DSEC_flow_SNN.py --mode valid` 走的是
`DSECDatasetLite(file_list="valid")` = **整个 valid split**，口径与锚点
（ep34：AEE 1.19951 / DSEC_Fl 5.31336）一致，判据才是真的数。

用法（远端 /root 下）：
  python -u /root/t51_eval_lrprobe.py            # 三个 arm 全评
  python -u /root/t51_eval_lrprobe.py 1e4 3e4    # 只评指定 arm
汇总：python t49_collect_eval.py <results/t51_lrprobe_*>
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
PY = "/opt/conda/envs/sdformerflow/bin/python"
EPOCH = 35          # EPOCH_OFFSET=35；探针原始 epoch0 ⇒ checkpoint_epoch35.pth


def main() -> int:
    tags = sys.argv[1:] or ["1e4", "3e4", "1e3"]
    env = {**os.environ, "SDFORMER_USE_MLFLOW": "0", "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
           "SDFORMER_SNN_BACKEND": "cupy", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    missing = []
    for tag in tags:
        run_dir = EXP / "results" / ("t51_lrprobe_%s" % tag)
        cfg = EXP / "configs/generated" / ("t51_lrprobe_%s.yml" % tag)
        ckpt = run_dir / ("checkpoint_epoch%d.pth" % EPOCH)
        if not ckpt.exists() or not cfg.exists():
            missing.append(tag)
            continue
        out_dir = run_dir / "standard_valid825" / ("epoch%d" % EPOCH)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [PY, "-u", "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
               "--config", str(cfg), "--checkpoint", str(ckpt),
               "--path_results", str(out_dir), "--mode", "valid"]
        print("[EVAL] launching %s" % tag, flush=True)
        with open(out_dir / "eval.log", "w", encoding="utf-8") as log:
            rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                env=env).returncode
        print("[EVAL] %s rc=%d" % (tag, rc), flush=True)
    if missing:
        print("[EVAL] skipped (missing cfg/ckpt): %s" % ", ".join(missing), flush=True)
    print("[EVAL] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
