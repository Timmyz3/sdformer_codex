#!/usr/bin/env python3
"""T52: 对**任意 run 名**的 checkpoint 做 valid825 正式评测（自适应格用）。

与 t51_eval_lrprobe.py 的区别：那个把 tag 拼成 `t51_lrprobe_<tag>`，这个直接吃
完整 run 名，于是同一份驱动可以评自适应格（`t52_eta_*`）、三值格、以及任何以后的 run。
除了 run 名，评测口径与 t51_eval_lrprobe.py **逐字相同**（`--mode valid` =
`DSECDatasetLite(file_list="valid")` = 整个 valid split，与锚点
ep34 AEE 1.19951 / DSEC_Fl 5.31336 同口径）。

SOPS 不需要另算：评测落盘的 `spike_profile.json` 里 `total_spikes` 就是
`synops_total`（72.89G ↔ 7.289e10，一一对应），`global_firing_rate` 是全局发放率。

用法（远端 /root 下）：
  python -u /root/t52_eval_runs.py t52_eta_etaON --epoch 35
汇总：python t49_collect_eval.py <results/t52_eta_*>
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
PY = "/opt/conda/envs/sdformerflow/bin/python"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="完整 run 名（results/<name> 与 configs/generated/<name>.yml）")
    ap.add_argument("--epoch", type=int, default=35, help="checkpoint 编号（EPOCH_OFFSET=35 ⇒ 首轮=35）")
    args = ap.parse_args()

    env = {**os.environ, "SDFORMER_USE_MLFLOW": "0", "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
           "SDFORMER_SNN_BACKEND": "cupy", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    missing = []
    for name in args.runs:
        run_dir = EXP / "results" / name
        cfg = EXP / "configs/generated" / (name + ".yml")
        ckpt = run_dir / ("checkpoint_epoch%d.pth" % args.epoch)
        if not ckpt.exists() or not cfg.exists():
            print("[EVAL] skip %s (missing %s)" % (
                name, "cfg" if not cfg.exists() else "ckpt %s" % ckpt.name), flush=True)
            missing.append(name)
            continue
        out_dir = run_dir / "standard_valid825" / ("epoch%d" % args.epoch)
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [PY, "-u", "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
               "--config", str(cfg), "--checkpoint", str(ckpt),
               "--path_results", str(out_dir), "--mode", "valid"]
        print("[EVAL] launching %s" % name, flush=True)
        with open(out_dir / "eval.log", "w", encoding="utf-8") as log:
            rc = subprocess.run(cmd, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                env=env).returncode
        print("[EVAL] %s rc=%d" % (name, rc), flush=True)
    if missing:
        print("[EVAL] skipped: %s" % ", ".join(missing), flush=True)
    print("[EVAL] ALL DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
