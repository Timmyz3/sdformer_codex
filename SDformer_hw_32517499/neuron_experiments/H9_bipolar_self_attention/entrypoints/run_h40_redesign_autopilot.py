"""Autopilot for the H40 redesign plan.

The queue is intentionally serial on one GPU. Short runs still use the full
model and bs8, so parallel jobs are more likely to waste time through OOM or
contention than to speed up the search.
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
GENERATED_DIR = CONFIG_DIR / "generated"
RESULTS_DIR = EXP_ROOT / "results"
REVIEWS_DIR = EXP_ROOT / "reviews"
BASELINE_CKPT = REPO_ROOT / "experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"


SCREEN_SOURCES = {
    "h40_p4_SNS02_ang05": "h40_p3_SNS02_ang05.yml",
    "h40_p4_TXS02_ang05": "h40_p3_TXS02_ang05.yml",
    "h40_p4_HTS02_ang02": "h40_p3_HTS02_ang02.yml",
    "h40_p4_SCS02_ang05": "h40_p2_SCS02_F.yml",
    "h40_p4_SCS012_ang05": "h40_p2_SCS012_F.yml",
    "h40_p4_SNS012_ang05": "h40_p2_SNS012_F.yml",
    "h40_p4_TXS012_ang05": "h40_p2_TXS012_F.yml",
    "h40_p4_HTS012_ang05": "h40_p2_HTS012_F.yml",
    "h40_p4_HTS02_ang05": "h40_p2_HTS02_F.yml",
    "h40_p4_SLS02_ang05": "h40_p2_SLS02_F.yml",
}

LR_STRATEGIES = [
    {
        "suffix": "dlr",
        "description": "baseline differential LR",
        "optimizer": {},
        "atlif": {},
    },
    {
        "suffix": "warm",
        "description": "differential LR with 80-step linear warmup from 0.2x",
        "optimizer": {
            "lr_warmup": {"enabled": True, "steps": 80, "start_factor": 0.2},
        },
        "atlif": {},
    },
    {
        "suffix": "slowbb",
        "description": "more conservative backbone/norm/threshold LR",
        "optimizer": {
            "param_groups": {
                "backbone_lr": 2.0e-7,
                "norm_lr": 2.0e-7,
                "neuron_lr": 1.2e-5,
                "threshold_lr": 3.0e-6,
            },
        },
        "atlif": {
            "threshold_base_lr": 3.0e-6,
        },
    },
    {
        "suffix": "warm_slowbb",
        "description": "conservative differential LR with 100-step warmup",
        "optimizer": {
            "lr_warmup": {"enabled": True, "steps": 100, "start_factor": 0.2},
            "param_groups": {
                "backbone_lr": 2.0e-7,
                "norm_lr": 2.0e-7,
                "neuron_lr": 1.2e-5,
                "threshold_lr": 3.0e-6,
            },
        },
        "atlif": {
            "threshold_base_lr": 3.0e-6,
        },
    },
]


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)


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
        log.write(f"\n[h40-autopilot] exit_code={proc.returncode}\n")
    return int(proc.returncode)


def generate_screen_configs() -> list[str]:
    generated: list[str] = []
    for experiment, source_name in SCREEN_SOURCES.items():
        source = GENERATED_DIR / source_name
        if not source.exists():
            raise FileNotFoundError(source)
        for strategy in LR_STRATEGIES:
            cfg = deepcopy(read_yaml(source))
            suffix = str(strategy["suffix"])
            variant = f"{experiment}_{suffix}"
            cfg["experiment"] = variant
            cfg.setdefault("loss", {})["use_angular_loss"] = True
            cfg.setdefault("loss", {})["lambda_ang"] = float(cfg["loss"].get("lambda_ang", 0.5) or 0.5)
            opt_cfg = cfg.setdefault("optimizer", {})
            for key, value in strategy.get("optimizer", {}).items():
                if key == "param_groups":
                    opt_cfg.setdefault("param_groups", {}).update(value)
                else:
                    opt_cfg[key] = value
            cfg.setdefault("atlif_ternary_psn", {}).update(strategy.get("atlif", {}))
            cfg["note"] = (
                "H40 redesign P4: structure/lr combined sweep; source="
                f"{source_name}; lr_strategy={suffix} ({strategy['description']}); "
                "generated for SCREEN -> CONFIRM -> FULL queue."
            )
            out = GENERATED_DIR / f"{variant}.yml"
            write_yaml(out, cfg)
            generated.append(f"generated/{out.name}")
    return generated


def read_rows(summary_csv: Path) -> list[dict[str, Any]]:
    if not summary_csv.exists():
        return []
    rows: list[dict[str, Any]] = []
    with summary_csv.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted = dict(row)
            for key in ("AEE", "AAE", "SOPs_G", "firing", "score"):
                if converted.get(key) not in {None, ""}:
                    converted[key] = float(converted[key])
            converted["samples"] = int(converted.get("samples") or 0)
            rows.append(converted)
    return rows


def config_from_row_name(row_name: str) -> str:
    name = row_name
    if name.endswith("_valid40"):
        name = name.removesuffix("_valid40")
    for suffix in ("_steps360", "_steps160", "_steps120", "_steps80"):
        if name.endswith(suffix):
            name = name.removesuffix(suffix)
    return f"generated/{name}.yml"


def pass_screen_configs(summary_csv: Path) -> list[str]:
    configs: list[str] = []
    for row in read_rows(summary_csv):
        if row.get("gate") == "pass" and int(row.get("samples") or 0) >= 5:
            cfg = config_from_row_name(row["name"])
            if cfg not in configs:
                configs.append(cfg)
    return configs


def pass_confirm_rows(summary_csv: Path) -> list[dict[str, Any]]:
    rows = [
        row
        for row in read_rows(summary_csv)
        if row.get("gate") == "pass"
        and row.get("stage") == "confirm"
        and int(row.get("samples") or 0) >= 40
    ]
    rows.sort(key=lambda row: (float(row["score"]), float(row["SOPs_G"])))
    return rows


def make_full_config(base_config: Path, row: dict[str, Any], run_stamp: str) -> Path:
    cfg = deepcopy(read_yaml(base_config))
    base_name = base_config.stem
    full_name = f"{base_name}_redesign_full_{run_stamp}"
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
        "H40 redesign autopilot full run. Promoted from valid40 confirm: "
        f"AEE={row['AEE']:.4f}, AAE={row['AAE']:.4f}, "
        f"SOPs={row['SOPs_G']:.4f}G, firing={row['firing']:.5f}."
    )
    out = CONFIG_DIR / f"{full_name}.yml"
    write_yaml(out, cfg)
    return out


def write_full_review(base_config: Path, full_config: Path, row: dict[str, Any], run_stamp: str) -> Path:
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    review = REVIEWS_DIR / f"FULL_REVIEW_{full_config.stem}_{run_stamp}.md"
    text = f"""# 全量训练前 Review：{full_config.stem}

## 选择结论

- 选择方案：`{base_config.stem}`
- 源配置：`{base_config.relative_to(REPO_ROOT)}`
- 全量配置：`{full_config.relative_to(REPO_ROOT)}`
- valid40 短测：AEE={row['AEE']:.4f}, AAE={row['AAE']:.4f}, SOPs={row['SOPs_G']:.4f}G, firing={row['firing']:.5f}
- promotion gate：`{row['gate']}`，score={row['score']:.4f}

## 范式检查

- 神经元主线仍是 PSN + ATLIF；Q/K 为三值输出，高 SOP FFN target groups 为二值 official ATLIF。
- 注意力使用 H40 redesign 中的 signed/ternary/hamming 系列，不再把失败的 QKFormer+gate 作为主线。
- 训练从 baseline `checkpoint_epoch59.pth` 续训，全量参数参与训练，使用 differential LR。
- 入口仍走 baseline 训练逻辑，H9 overlay 只在实验目录注入模块，不改 `third_party/SDformerFlow`。

## 风险和观察点

- valid40 通过才进入全量，但 full 仍可能因 30 epoch 阈值继续上升导致 AEE/AAE 变差；最终必须 profile epoch9/19/29 或 latest。
- 如果三值负发放重新塌缩，应优先看 `ternary_worst_pos_neg_ratio` 和 `zero_neg`，不要只看 SOPs。
- 如果 SOPs 明显低但 AAE 变差，优先降低稀疏强度，而不是扩大替换范围。
"""
    review.write_text(text, encoding="utf-8")
    return review


def latest_checkpoint(run_dir: Path) -> Path | None:
    checkpoints = sorted(run_dir.glob("checkpoint_epoch*.pth"))
    return checkpoints[-1] if checkpoints else None


def run_rapid_one(
    tag: str,
    config: str,
    steps: int,
    valid_samples: int,
    log_dir: Path,
    *,
    batch_size: int,
    workers: int,
) -> Path:
    command = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/rapid_screen.py"),
        "--tag",
        tag,
        "--steps",
        str(steps),
        "--valid-samples",
        str(valid_samples),
        "--batch-size",
        str(batch_size),
        "--workers",
        str(workers),
        "--amp",
        "--config",
        config,
    ]
    if steps < 360 or valid_samples >= 40:
        command.append("--no-promote-valid40")
    code = run_command(command, log_dir / f"{tag}.log")
    if code != 0:
        raise RuntimeError(f"rapid_screen failed for {tag}; see {log_dir / (tag + '.log')}")
    candidates = sorted(RESULTS_DIR.glob(f"{tag}_*/summary.csv"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No summary found for tag {tag}")
    return candidates[-1]


def run_rapid(
    tag: str,
    configs: list[str],
    steps: int,
    valid_samples: int,
    log_dir: Path,
    *,
    batch_size: int = 8,
    workers: int = 8,
) -> Path:
    command = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/rapid_screen.py"),
        "--tag",
        tag,
        "--steps",
        str(steps),
        "--valid-samples",
        str(valid_samples),
        "--batch-size",
        str(batch_size),
        "--workers",
        str(workers),
        "--amp",
    ]
    for config in configs:
        command.extend(["--config", config])
    if steps < 360 or valid_samples >= 40:
        command.append("--no-promote-valid40")
    code = run_command(command, log_dir / f"{tag}.log")
    if code != 0:
        raise RuntimeError(f"rapid_screen failed for {tag}; see {log_dir / (tag + '.log')}")
    candidates = sorted(RESULTS_DIR.glob(f"{tag}_*/summary.csv"), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No summary found for tag {tag}")
    return candidates[-1]


def run_parallel_screen(
    tag: str,
    configs: list[str],
    steps: int,
    valid_samples: int,
    log_dir: Path,
    *,
    parallel: int = 2,
    batch_size: int = 4,
    workers: int = 4,
) -> list[Path]:
    summaries: list[Path] = []
    active: list[tuple[subprocess.Popen[Any], str, Path]] = []

    def launch(index: int, config: str) -> None:
        run_tag = f"{tag}_{index:02d}_{Path(config).stem}"
        log_path = log_dir / f"{run_tag}.log"
        command = [
            sys.executable,
            "-u",
            str(EXP_ROOT / "entrypoints/rapid_screen.py"),
            "--tag",
            run_tag,
            "--config",
            config,
            "--steps",
            str(steps),
            "--valid-samples",
            str(valid_samples),
            "--batch-size",
            str(batch_size),
            "--workers",
            str(workers),
            "--amp",
            "--no-promote-valid40",
        ]
        env = os.environ.copy()
        env["SDFORMER_USE_MLFLOW"] = "0"
        env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w", encoding="utf-8")
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        active.append((proc, run_tag, log_path))

    cursor = 0
    while cursor < len(configs) or active:
        while cursor < len(configs) and len(active) < parallel:
            launch(cursor, configs[cursor])
            cursor += 1
        time.sleep(10)
        still_active: list[tuple[subprocess.Popen[Any], str, Path]] = []
        for proc, run_tag, log_path in active:
            code = proc.poll()
            if code is None:
                still_active.append((proc, run_tag, log_path))
                continue
            if code != 0:
                running_children = [
                    running
                    for running, _, _ in active
                    if running is not proc and running.poll() is None
                ]
                for running in running_children:
                    running.terminate()
                for running in running_children:
                    try:
                        running.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        running.kill()
                raise RuntimeError(f"parallel rapid_screen failed for {run_tag}; see {log_path}")
            candidates = sorted(RESULTS_DIR.glob(f"{run_tag}_*/summary.csv"), key=lambda path: path.stat().st_mtime)
            if not candidates:
                running_children = [
                    running
                    for running, _, _ in active
                    if running is not proc and running.poll() is None
                ]
                for running in running_children:
                    running.terminate()
                for running in running_children:
                    try:
                        running.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        running.kill()
                raise FileNotFoundError(f"No summary found for tag {run_tag}")
            summaries.append(candidates[-1])
        active = still_active
    return summaries


def pass_screen_configs_from_many(summary_csvs: list[Path]) -> list[str]:
    configs: list[str] = []
    for summary_csv in summary_csvs:
        for config in pass_screen_configs(summary_csv):
            if config not in configs:
                configs.append(config)
    return configs


def run_full(full_config: Path, run_stamp: str, log_dir: Path) -> int:
    run_dir = RESULTS_DIR / f"{full_config.stem}_bs8_{run_stamp}_setsid"
    train_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/train.py"),
        "--config",
        str(full_config),
        "--prev_runid",
        str(BASELINE_CKPT),
        "--save_path",
        str(run_dir / "checkpoint_epoch{}.pth"),
    ]
    train_code = run_command(train_cmd, run_dir / "train.log")
    if train_code != 0:
        return train_code
    checkpoint = latest_checkpoint(run_dir)
    if checkpoint is None:
        return 2
    profile_cmd = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/profile_sops.py"),
        "--config",
        str(full_config),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(run_dir / "profile_latest_valid40"),
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
    return run_command(profile_cmd, run_dir / "profile_latest_valid40" / "profile.log")


def append_plan_note(text: str) -> None:
    plan = REPO_ROOT / "neuron_autoresearch" / "EXPERIMENT_REDESIGN_PLAN.md"
    with plan.open("a", encoding="utf-8") as handle:
        handle.write("\n" + text.rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-full", type=int, default=2)
    parser.add_argument(
        "--reuse-screen-glob",
        default=None,
        help="Reuse existing screen summary CSVs instead of running the parallel screen stage.",
    )
    args = parser.parse_args()

    run_stamp = stamp()
    log_dir = RESULTS_DIR / f"h40_redesign_autopilot_{run_stamp}"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_screen_glob:
        generated = []
        screen_summaries = sorted(Path(path) for path in RESULTS_DIR.glob(args.reuse_screen_glob))
        if not screen_summaries:
            raise FileNotFoundError(f"No screen summaries matched {args.reuse_screen_glob!r}")
        append_plan_note(
            f"""
## 七、H40 redesign autopilot 续跑记录（{run_stamp}）

- 复用早筛 summary：{', '.join(f'`{item.relative_to(REPO_ROOT)}`' for item in screen_summaries)}
- 从 confirm 阶段继续：360-step valid40 -> valid40 pass 后串行 full。
"""
        )
    else:
        generated = generate_screen_configs()
        append_plan_note(
            f"""
## 七、H40 redesign autopilot 接管记录（{run_stamp}）

- 自动生成补测配置：{', '.join(f'`{item}`' for item in generated)}
- 执行策略：160-step valid5 早筛 -> 360-step valid40 确认 -> valid40 pass 后串行 full。
- 并行策略：早筛使用 `parallel=2, bs4, workers4`。实测两个 bs4 并发显存约 42GB，稳定；confirm/full 仍使用 bs8 串行。
"""
        )

        screen_summaries = run_parallel_screen(
            tag="h40_p4_ang05_screen160_bs4x2",
            configs=generated,
            steps=160,
            valid_samples=5,
            log_dir=log_dir,
            parallel=2,
            batch_size=4,
            workers=4,
        )
    promoted = pass_screen_configs_from_many(screen_summaries)
    confirm_configs = []
    for config in promoted:
        if config not in confirm_configs:
            confirm_configs.append(config)
    append_plan_note(
        f"""
### H40 P4 早筛完成

- screen summaries：{', '.join(f'`{item.relative_to(REPO_ROOT)}`' for item in screen_summaries)}
- 进入 confirm 的配置：{', '.join(f'`{item}`' for item in confirm_configs)}
"""
    )
    if not confirm_configs:
        append_plan_note("- 早筛没有 pass 配置，未启动 confirm/full。")
        return 0

    confirm_summary = run_rapid(
        tag="h40_p4_confirm360_valid40",
        configs=confirm_configs,
        steps=360,
        valid_samples=40,
        log_dir=log_dir,
    )
    confirm_pass = pass_confirm_rows(confirm_summary)
    append_plan_note(
        f"""
### H40 P4 valid40 确认完成

- confirm summary：`{confirm_summary.relative_to(REPO_ROOT)}`
- valid40 pass 数量：{len(confirm_pass)}
"""
    )

    if not confirm_pass:
        append_plan_note("- 没有 valid40 pass，未启动 full，等待下一轮超参调整。")
        return 0

    for row in confirm_pass[: args.max_full]:
        base_config = (CONFIG_DIR / config_from_row_name(row["name"])).resolve()
        full_config = make_full_config(base_config, row, run_stamp)
        review = write_full_review(base_config, full_config, row, run_stamp)
        append_plan_note(
            f"""
### 启动 full：`{full_config.stem}`

- review：`{review.relative_to(REPO_ROOT)}`
- full config：`{full_config.relative_to(REPO_ROOT)}`
"""
        )
        code = run_full(full_config, run_stamp, log_dir)
        append_plan_note(f"- full `{full_config.stem}` exit_code={code}")
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
