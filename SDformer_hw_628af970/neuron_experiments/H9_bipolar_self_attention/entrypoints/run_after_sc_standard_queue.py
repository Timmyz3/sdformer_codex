#!/usr/bin/env python3
"""Wait for the current SC run, evaluate it, then launch TX token selector.

This is intentionally narrow: it is for the 2026-05-31 stride SC run and the
next corrected H49/H53-style TX qkselector full run.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
H9_ROOT = REPO_ROOT / "neuron_experiments" / "H9_bipolar_self_attention"
CONFIG_DIR = H9_ROOT / "configs" / "generated"
RESULTS_DIR = H9_ROOT / "results"
REDESIGN_MD = REPO_ROOT / "neuron_autoresearch" / "EXPERIMENT_REDESIGN_PLAN.md"
BASELINE_CKPT = REPO_ROOT / "experiments" / "baseline_stride_upstream" / "checkpoint_epoch59.pth"
SC_CONFIG = CONFIG_DIR / "stride_h41_sc_s012c.yml"
SC_RUN_DIR = RESULTS_DIR / "stride_h41_sc_s012c_20260531_170553"
H53_SOURCE_CONFIG = CONFIG_DIR / "h53b_h49_clean_no_stage3_s02_full30.yml"
H53_STRIDE_CONFIG = CONFIG_DIR / "stride_h53b_h49_clean_no_stage3_s02_full30.yml"
DSEC_DATA = Path("/root/private_data/SothisAI/dataset/Console/DSEC/main/DSEC/saved_flow_data")
PYTHON = Path("/opt/conda/envs/sdformerflow/bin/python")


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def run(cmd: list[str], log_path: Path, env: dict[str, str] | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log("RUN " + " ".join(cmd))
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(f"command failed ({process.returncode}); see {log_path}")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def make_stride_txselector_config() -> Path:
    cfg = deepcopy(read_yaml(H53_SOURCE_CONFIG))
    cfg["experiment"] = "stride_h53b_h49_clean_no_stage3_s02_full30"
    cfg.setdefault("data", {})["path"] = str(DSEC_DATA)
    cfg["data"].pop("sequence_list_overrides", None)
    cfg.setdefault("test", {})["sample"] = 825
    cfg.setdefault("runtime", {})["snn_backend"] = "cupy"
    cfg["runtime"]["use_mlflow_model_logging"] = False
    cfg.setdefault("optimizer", {})["lr_warmup"] = {
        "enabled": True,
        "steps": 300,
        "start_factor": 0.1,
    }
    cfg["note"] = (
        "Stride rerun of corrected H53b/H49 token selector. Uses the official "
        "stride train/valid split and standard upstream baseline epoch59. "
        "Stage3 remains untouched; Q/K PSN+ATLIF ternary has no target-rate feedback; "
        "FFN uses official binary ATLIF on S0+S2-half. Uses 300-step LR warmup."
    )
    write_yaml(H53_STRIDE_CONFIG, cfg)
    return H53_STRIDE_CONFIG


def as_float(value: Any) -> float:
    return float(value)


def fmt_human(value: float) -> str:
    for suffix, scale in (("P", 1e15), ("T", 1e12), ("G", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(value) >= scale:
            return f"{value / scale:.4f}{suffix}"
    return f"{value:.4f}"


def evaluate_checkpoint(epoch: int) -> dict[str, Any]:
    ckpt = SC_RUN_DIR / f"checkpoint_epoch{epoch}.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")
    out_dir = SC_RUN_DIR / "standard_valid825" / f"epoch{epoch}"
    env = os.environ.copy()
    env.update(
        {
            "SDFORMER_USE_MLFLOW": "0",
            "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
            "SDFORMER_SNN_BACKEND": "cupy",
        }
    )
    cmd = [
        str(PYTHON),
        "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
        "--config",
        str(SC_CONFIG),
        "--checkpoint",
        str(ckpt),
        "--path_results",
        str(out_dir),
        "--mode",
        "valid",
    ]
    profile_path = out_dir / "spike_profile.json"
    if profile_path.exists():
        profile_path.unlink()
    run(cmd, out_dir / "eval.log", env=env)
    if not profile_path.exists():
        raise FileNotFoundError(f"eval did not write {profile_path}")
    with profile_path.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    metrics = profile.get("metrics", {})
    return {
        "epoch": epoch,
        "samples": int(profile.get("samples", 0)),
        "AEE": as_float(metrics.get("AEE", "nan")),
        "AAE": as_float(metrics.get("AAE", "nan")),
        "PE1": as_float(metrics.get("AEE_PE1", "nan")),
        "PE2": as_float(metrics.get("AEE_PE2", "nan")),
        "PE3": as_float(metrics.get("AEE_PE3", "nan")),
        "outliers": as_float(metrics.get("AEE_outliers", "nan")),
        "total_spikes": float(profile["total_spikes"]),
        "global_firing_rate": float(profile["global_firing_rate"]),
        "dense_flops": float(profile["dense_flops"]),
        "effective_flops": float(profile["effective_flops"]),
        "sparsity_ratio": float(profile["sparsity_ratio"]),
        "synops_mac": float(profile["synops_mac"]),
        "synops_logic": float(profile["synops_logic"]),
        "synops_total": float(profile["synops_total"]),
        "energy_uj": float(profile["energy_uj"]),
        "profiled_layers": int(profile["profiled_layers"]),
        "profile_dir": str(out_dir),
    }


def append_results_to_md(rows: list[dict[str, Any]], tx_config: Path) -> None:
    lines = [
        "",
        "### 31.9 stride_h41_sc_s012c 标准 valid825 结果（自动队列）",
        "",
        f"- 评估配置：`{SC_CONFIG.relative_to(REPO_ROOT)}`",
        f"- 训练目录：`{SC_RUN_DIR.relative_to(REPO_ROOT)}`",
        "- 推理口径：valid825，`SDFORMER_USE_MLFLOW=0`，`SDFORMER_SNN_BACKEND=cupy`，H9 config 加载审计开启。",
        "",
        "| epoch | samples | AEE | AAE | PE1 | PE2 | PE3/outlier | total_spikes | firing | dense_flops | effective_flops | sparsity | synops_mac | synops_logic | energy_uj |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {epoch} | {samples} | {AEE:.4f} | {AAE:.4f} | {PE1:.4f} | {PE2:.4f} | "
            "{outliers:.4f} | {total_spikes} | {firing:.4%} | {dense_flops} | "
            "{effective_flops} | {sparsity:.2%} | {synops_mac} | {synops_logic} | {energy:.2f} |".format(
                epoch=row["epoch"],
                samples=row["samples"],
                AEE=row["AEE"],
                AAE=row["AAE"],
                PE1=row["PE1"],
                PE2=row["PE2"],
                outliers=row["outliers"],
                total_spikes=fmt_human(row["total_spikes"]),
                firing=row["global_firing_rate"],
                dense_flops=fmt_human(row["dense_flops"]),
                effective_flops=fmt_human(row["effective_flops"]),
                sparsity=row["sparsity_ratio"],
                synops_mac=fmt_human(row["synops_mac"]),
                synops_logic=fmt_human(row["synops_logic"]),
                energy=row["energy_uj"],
            )
        )
    best = min(rows, key=lambda row: (row["AEE"], row["AAE"]))
    lines += [
        "",
        f"当前 SC 标准 valid825 最优点按 AEE 排序为 epoch{best['epoch']}。完整推理日志和 `spike_profile.json` 保存在各 `standard_valid825/epoch*/` 目录。",
        "",
        "后续已接入 TX 逐 token selector 全量：",
        "",
        f"- 配置：`{tx_config.relative_to(REPO_ROOT)}`",
        f"- 续训起点：`{BASELINE_CKPT.relative_to(REPO_ROOT)}`",
        "- 方案：corrected H53b/H49 qkselector，`ternary_alpha_xnor_qkselector_shiftmax`，stage3 不替换，Q/K 无 target-rate，FFN official binary ATLIF。",
        "",
    ]
    with REDESIGN_MD.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def launch_txselector(tx_config: Path) -> None:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"stride_h53b_h49_clean_no_stage3_s02_full30_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "SDFORMER_USE_MLFLOW": "0",
            "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
            "SDFORMER_SNN_BACKEND": "cupy",
        }
    )
    cmd = [
        str(PYTHON),
        "-u",
        "neuron_experiments/H9_bipolar_self_attention/entrypoints/train.py",
        "--config",
        str(tx_config),
        "--prev_runid",
        str(BASELINE_CKPT),
        "--save_path",
        str(out_dir / "checkpoint_epoch{}.pth"),
    ]
    run(cmd, out_dir / "train.log", env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, default=12109)
    parser.add_argument("--epochs", type=int, nargs="+", default=[20, 24, 27, 29])
    args = parser.parse_args()

    log(f"waiting for SC pid {args.wait_pid}")
    while pid_exists(args.wait_pid):
        time.sleep(60)
    log("SC process ended")

    final_ckpt = SC_RUN_DIR / f"checkpoint_epoch{max(args.epochs)}.pth"
    while not final_ckpt.exists():
        log(f"waiting for {final_ckpt}")
        time.sleep(60)

    rows = [evaluate_checkpoint(epoch) for epoch in args.epochs]
    tx_config = make_stride_txselector_config()
    append_results_to_md(rows, tx_config)
    launch_txselector(tx_config)


if __name__ == "__main__":
    main()
