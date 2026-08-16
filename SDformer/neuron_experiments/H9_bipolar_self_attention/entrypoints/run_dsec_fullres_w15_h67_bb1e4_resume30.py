"""Continue H67 full-resolution training from ep15 to the equal-budget ep30."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
CONFIG = EXP / "configs/generated/dsec_fullres_w15_H67_crop_bb1e4_resume_ep30.yml"
SOURCE_ROOT = EXP / "results/dsec_fullres_w15_H67_crop_bb1e4_resume15_20260803"
SOURCE_CHECKPOINT = SOURCE_ROOT / "checkpoint_epoch15.pth"
SOURCE_STATE = SOURCE_ROOT / "checkpoint_epoch15_state_dict.pth"
ROOT = EXP / "results/dsec_fullres_w15_H67_crop_bb1e4_resume30_20260804"
STAGING = ROOT / "resume_source_ep15_scheduler_aligned"
STAGED_CHECKPOINT = STAGING / "checkpoint_epoch15.pth"
STAGED_STATE = STAGING / "checkpoint_epoch15_state_dict.pth"
STATUS = ROOT / "status.log"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
EVAL_EPOCHS = (20, 25, 30)
PYTHON = Path(sys.executable)


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "SDFORMER_USE_MLFLOW": "0",
            "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
            "SDFORMER_SNN_BACKEND": "cupy",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return env


def run(command: list[str], log: Path, label: str) -> None:
    record(f"START {label}: {' '.join(command)}")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=REPO,
            env=environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    record(f"END {label}: exit_code={result.returncode}")
    if result.returncode:
        raise RuntimeError(f"{label} failed; see {log}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_scheduler_aligned_resume() -> None:
    """Repair only scheduler counters that lag because state is saved pre-step."""

    STAGING.mkdir(parents=True, exist_ok=True)
    if STAGED_CHECKPOINT.exists() or STAGED_CHECKPOINT.is_symlink():
        STAGED_CHECKPOINT.unlink()
    # The training entrypoint resolves symlinks in CLI paths. A hard link keeps
    # the staged pathname, so resume_model finds the audited staged state beside it.
    os.link(SOURCE_CHECKPOINT, STAGED_CHECKPOINT)

    state = torch.load(SOURCE_STATE, map_location="cpu", weights_only=False)
    scheduler = state.get("scheduler")
    if not isinstance(scheduler, dict):
        raise RuntimeError("source training state has no scheduler dictionary")
    if int(state.get("epoch", -1)) != 14:
        raise RuntimeError(f"expected internal source epoch 14, got {state.get('epoch')}")

    before = {
        "last_epoch": int(scheduler.get("last_epoch", -1)),
        "step_count": int(scheduler.get("_step_count", -1)),
        "last_lr": list(scheduler.get("_last_lr", [])),
        "optimizer_lrs": [float(group["lr"]) for group in state["optimizer"]["param_groups"]],
    }
    if before["last_epoch"] != 12:
        raise RuntimeError(f"unexpected source scheduler last_epoch: {before['last_epoch']}")

    scheduler["last_epoch"] = 15
    scheduler["_step_count"] = before["step_count"] + 3
    scheduler["milestones"] = Counter({10: 1, 20: 1})
    after = {
        "last_epoch": int(scheduler["last_epoch"]),
        "step_count": int(scheduler["_step_count"]),
        "last_lr": list(scheduler.get("_last_lr", [])),
        "optimizer_lrs": [float(group["lr"]) for group in state["optimizer"]["param_groups"]],
    }
    if after["last_lr"] != before["last_lr"] or after["optimizer_lrs"] != before["optimizer_lrs"]:
        raise RuntimeError("scheduler alignment must not alter the current learning rates")

    torch.save(state, STAGED_STATE)
    audit = {
        "reason": "upstream checkpoint state is serialized before end-of-epoch scheduler.step",
        "source_model": str(SOURCE_CHECKPOINT),
        "source_state": str(SOURCE_STATE),
        "source_model_sha256": sha256(SOURCE_CHECKPOINT),
        "staged_model_is_symlink": STAGED_CHECKPOINT.is_symlink(),
        "staged_model_is_hardlink": STAGED_CHECKPOINT.stat().st_ino == SOURCE_CHECKPOINT.stat().st_ino,
        "state_epoch_unchanged": int(state["epoch"]),
        "scheduler_before": before,
        "scheduler_after": after,
        "optimizer_and_scaler_unchanged": True,
    }
    (STAGING / "scheduler_alignment_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )


def append_protocol() -> None:
    marker = "DSEC_FULLRES_W15_H67_BB1E4_RESUME30_20260804"
    if marker in REDESIGN.read_text(encoding="utf-8"):
        return
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### H67 fullres bb1e4 ep15-to-ep30 等预算续训（2026-08-04）\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(
            "- ep15 已达到 AEE `1.4757`、AAE-Benchmark `6.1599`、spikes `78.2806G`，"
            "满足 NB0+5% 与 spikes 至少下降20%的门槛；继续到 ep30 是为了与 NB0 fullres 30 epochs 等预算。\n"
        )
        handle.write(
            "- 审计发现上游 state 在每轮 `scheduler.step()` 前保存；此前两次分段使 saved scheduler "
            "在 ep15 时为 `last_epoch=12`。实际已执行 LR 为 ep1--12 `1e-4`、ep13--15 `5e-5`。\n"
        )
        handle.write(
            "- 不修改历史 checkpoint/state；新增 staged state，仅将 scheduler "
            "`last_epoch 12->15`、`_step_count +3`，model/optimizer/scaler/current LR 均不变，"
            "之后按 milestone20 正常降 LR。\n"
        )
        handle.write(
            "- 保存并 standard-valid825 评估 ep20/25/30；其余 480x640、window2x15x15、"
            "batch2、BN no_running、H67 Motion-XOR 结构不变。\n"
        )
        handle.write(f"- config：`{CONFIG.relative_to(REPO)}`。\n")
        handle.write(f"- scheduler audit：`{(STAGING / 'scheduler_alignment_audit.json').relative_to(REPO)}`。\n")
        handle.write(f"- status：`{STATUS.relative_to(REPO)}`。\n")


def append_results() -> None:
    marker = "DSEC_FULLRES_W15_H67_BB1E4_RESUME30_RESULT_20260804"
    if marker in REDESIGN.read_text(encoding="utf-8"):
        return
    ranking = ROOT / "profile_ranking_valid825.md"
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### H67 fullres bb1e4 ep20/25/30 结果\n\n")
        handle.write(f"<!-- {marker} -->\n\n")
        handle.write(ranking.read_text(encoding="utf-8"))


def main() -> int:
    for required in (SOURCE_CHECKPOINT, SOURCE_STATE):
        if not required.is_file():
            raise FileNotFoundError(required)

    run(
        [str(PYTHON), str(EXP / "entrypoints/make_dsec_fullres_w15_h67_bb1e4_resume30_config.py")],
        ROOT / "config.log",
        "generate config",
    )
    stage_scheduler_aligned_resume()
    append_protocol()

    if not (ROOT / "checkpoint_epoch30.pth").is_file():
        run(
            [
                str(PYTHON),
                "-u",
                str(EXP / "entrypoints/train.py"),
                "--config",
                str(CONFIG),
                "--prev_runid",
                str(STAGED_CHECKPOINT),
                "--save_path",
                str(ROOT / "checkpoint_epoch{}.pth"),
                "--finetune",
                "1",
                "--resume",
                "1",
            ],
            ROOT / "train.log",
            "H67 bb1e4 scheduler-aligned resume ep15-to-ep30",
        )

    ranking = ROOT / "profile_ranking_valid825.md"
    if not ranking.is_file():
        epochs = ",".join(str(epoch) for epoch in EVAL_EPOCHS)
        epoch_args = [item for epoch in EVAL_EPOCHS for item in ("--epoch", str(epoch))]
        run(
            [
                str(PYTHON),
                "-u",
                str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
                "--config",
                str(CONFIG),
                "--run-dir",
                str(ROOT),
                "--ranking-mode",
                "aee",
                *epoch_args,
            ],
            ROOT / "valid825.log",
            f"H67 bb1e4 epochs {epochs} valid825",
        )
    append_results()
    record("ALL COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
