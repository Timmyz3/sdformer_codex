"""Select a reviewed candidate and launch full fine-tuning.

This is the handoff script for overnight/autonomous runs:
1. read rapid-screen summaries;
2. prefer a reviewed H37 attention if it is close to the H36 fallback;
3. write a Chinese full-run review;
4. generate a full config and run training;
5. profile the final checkpoint for AEE/AAE/SOPs/firing.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = EXP_ROOT / "configs"
RESULTS_DIR = EXP_ROOT / "results"
REVIEWS_DIR = EXP_ROOT / "reviews"
DEFAULT_BASELINE_CKPT = REPO_ROOT / "experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def dump_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


def read_rows(summary_csv: Path | None) -> list[dict[str, Any]]:
    if summary_csv is None or not summary_csv.exists():
        return []
    rows: list[dict[str, Any]] = []
    with summary_csv.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted = dict(row)
            for key in ("AEE", "AAE", "SOPs_G", "firing", "score"):
                converted[key] = float(converted[key])
            converted["samples"] = int(converted["samples"])
            rows.append(converted)
    return rows


def candidate_base_name(row_name: str) -> str:
    name = row_name
    if name.endswith("_valid40"):
        name = name.removesuffix("_valid40")
    for suffix in ("_steps120", "_steps360"):
        if name.endswith(suffix):
            name = name.removesuffix(suffix)
    return name


def best_valid40(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [
        row
        for row in rows
        if row.get("samples") == 40 and row.get("stage") == "confirm" and row.get("gate") == "pass"
    ]
    if not valid:
        return None
    return min(valid, key=lambda row: row["score"])


def choose_candidate(h37_rows: list[dict[str, Any]], fallback_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    h37 = best_valid40(h37_rows)
    fallback = best_valid40(fallback_rows)
    if fallback is None and h37 is None:
        raise RuntimeError(
            "No confirm/pass valid40 candidate is available. "
            "Refusing to launch a full run from valid10, 120-step, or failed-gate rows."
        )
    if fallback is None:
        return h37, "没有 H36 fallback valid40，选择 H37 最佳 valid40。"
    if h37 is None:
        return fallback, "H37 尚无 promoted valid40，回退到 H36 最佳 valid40。"

    close_enough = (
        h37["AEE"] <= fallback["AEE"] + 0.03
        and h37["AAE"] <= fallback["AAE"] + 0.20
        and h37["SOPs_G"] <= fallback["SOPs_G"] + 0.10
    )
    if h37["score"] <= fallback["score"] or close_enough:
        return h37, "H37 修正版注意力达到或接近 H36 fallback，优先选择论文范式更干净的 H37。"
    return fallback, "H37 修正版注意力未达到 H36 fallback 指标，回退到 H36 最佳候选。"


def make_full_config(base_config: Path, selected: dict[str, Any], reason: str, run_stamp: str) -> Path:
    cfg = deepcopy(load_yaml(base_config))
    base_name = candidate_base_name(selected["name"])
    full_name = f"{base_name}_reviewed_auto_full_{run_stamp}"
    cfg["experiment"] = full_name
    runtime = cfg.setdefault("runtime", {})
    runtime["max_train_steps"] = 0
    runtime["skip_state_save"] = True
    runtime["force_save_epochs"] = [9, 19, 29]
    runtime["use_mlflow_model_logging"] = False
    loader = cfg.setdefault("loader", {})
    loader["n_epochs"] = 30
    loader["batch_size"] = 8
    loader["n_workers"] = 8
    loader["persistent_workers"] = True
    loader["prefetch_factor"] = 4
    loader["pin_memory"] = True
    cfg.setdefault("optimizer", {})["use_amp"] = True
    cfg.setdefault("test", {})["sample"] = 10
    cfg.setdefault("metrics", {})["name"] = ["AEE", "AAE"]
    cfg["note"] = (
        "自动全量续训。选择依据："
        f"{reason} 短测指标：AEE={selected['AEE']:.4f}, AAE={selected['AAE']:.4f}, "
        f"SOPs={selected['SOPs_G']:.4f}G, firing={selected['firing']:.5f}。"
    )
    out = CONFIG_DIR / f"{full_name}.yml"
    dump_yaml(out, cfg)
    return out


def write_review(selected: dict[str, Any], base_config: Path, full_config: Path, reason: str, run_stamp: str) -> Path:
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    review = REVIEWS_DIR / f"FULL_REVIEW_{full_config.stem}_{run_stamp}.md"
    text = f"""# 全量训练前 Review：{full_config.stem}

## 选择结论

- 选择方案：`{candidate_base_name(selected["name"])}`
- 选择原因：{reason}
- 短测指标：AEE={selected["AEE"]:.4f}, AAE={selected["AAE"]:.4f}, SOPs={selected["SOPs_G"]:.4f}G, firing={selected["firing"]:.5f}
- 源配置：`{base_config.relative_to(REPO_ROOT)}`
- 全量配置：`{full_config.relative_to(REPO_ROOT)}`

## 范式检查

- 神经元主线：Q/K 使用三值 PSN+ATLIF；高 SOP 替换层使用二值 official ATLIF。
- baseline 完整性：入口仍走 baseline 训练逻辑，改动通过 `neuron_experiments/H9_bipolar_self_attention/overlay` 注入。
- 外部 review 处理：H37 已新增严格 QKV-BSA、二元 alpha-XNOR、QKV-A2OS2A；旧 strict-BSA/alpha-XNOR/A2OS2A 只按 adapted/inspired 表述。
- 学习率策略：采用 H36/H37 短测中对应配置的 differential LR，backbone 小 LR，新神经元/阈值较大 LR。

## 风险

- 如果选择的是 H36 fallback，则注意力仍是 SDFormerFlow 适配范式，不应在论文中写作原版 alpha-XNOR/BSA/A2OS2A。
- 如果选择的是 H37 QKV 分支，则 V 是从 K copy 初始化后独立训练，属于 baseline QKFormer 的结构扩展，需在实验表里单独标注参数量增加。
"""
    review.write_text(text, encoding="utf-8")
    return review


def run_command(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\n[select-full] exit_code={proc.returncode}\n")
    return int(proc.returncode)


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(run_dir.glob("checkpoint_epoch*.pth"))
    return checkpoints[-1] if checkpoints else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h37-summary", type=Path, required=True)
    parser.add_argument("--fallback-summary", type=Path, required=True)
    parser.add_argument("--prev-runid", type=Path, default=DEFAULT_BASELINE_CKPT)
    args = parser.parse_args(argv)

    h37_rows = read_rows(args.h37_summary)
    fallback_rows = read_rows(args.fallback_summary)
    selected, reason = choose_candidate(h37_rows, fallback_rows)
    base_name = candidate_base_name(selected["name"])
    base_config = CONFIG_DIR / f"{base_name}.yml"
    if not base_config.exists():
        raise FileNotFoundError(base_config)

    run_stamp = stamp()
    full_config = make_full_config(base_config, selected, reason, run_stamp)
    review = write_review(selected, base_config, full_config, reason, run_stamp)
    run_dir = RESULTS_DIR / f"{full_config.stem}_bs8_{run_stamp}_setsid"
    train_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/train.py"),
        "--config",
        str(full_config),
        "--prev_runid",
        str(args.prev_runid.resolve()),
        "--save_path",
        str(run_dir / "checkpoint_epoch{}.pth"),
    ]
    print(f"selected={base_name}")
    print(f"reason={reason}")
    print(f"review={review}")
    print(f"full_config={full_config}")
    print(f"run_dir={run_dir}")
    train_exit = run_command(train_cmd, run_dir / "train.log")
    if train_exit != 0:
        return train_exit

    checkpoint = latest_checkpoint(run_dir)
    if checkpoint is None:
        return 2
    profile_dir = run_dir / "profile_valid40"
    profile_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/profile_sops.py"),
        "--config",
        str(full_config),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(profile_dir),
        "--split",
        "valid",
        "--num-samples",
        "40",
        "--batch-size",
        "1",
        "--num-workers",
        "4",
        "--metric",
        "AEE",
        "--metric",
        "AAE",
    ]
    return run_command(profile_cmd, profile_dir / "profile.log")


if __name__ == "__main__":
    raise SystemExit(main())
