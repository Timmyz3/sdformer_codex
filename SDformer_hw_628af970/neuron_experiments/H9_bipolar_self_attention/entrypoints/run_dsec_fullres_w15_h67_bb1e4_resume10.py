"""Strictly resume the best H67 full-resolution rescue from ep5 to ep10."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
CONFIG = EXP / "configs/generated/dsec_fullres_w15_H67_crop_bb1e4_resume_ep10.yml"
SOURCE_ROOT = EXP / "results/dsec_fullres_w15_rescue_short5_20260801/H67_crop_bb1e4"
SOURCE_CHECKPOINT = SOURCE_ROOT / "checkpoint_epoch5.pth"
SOURCE_STATE = SOURCE_ROOT / "checkpoint_epoch5_state_dict.pth"
ROOT = EXP / "results/dsec_fullres_w15_H67_crop_bb1e4_resume10_20260802"
STATUS = ROOT / "status.log"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
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


def append_protocol() -> None:
    marker = "DSEC_FULLRES_W15_H67_BB1E4_STRICT_RESUME10_20260802"
    if marker in REDESIGN.read_text(encoding="utf-8"):
        return
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### H67 fullres bb1e4 strict ep5-to-ep10 resume（2026-08-02）\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(
            "- short5 的 ep5 仍在改善，故只延长排名第一的 `H67_crop_bb1e4`；"
            "不继续较差的 `bb2e5`。\n"
        )
        handle.write(
            "- 同时加载 `checkpoint_epoch5.pth` 与同目录 `checkpoint_epoch5_state_dict.pth`，"
            "使用 `--resume 1` 严格恢复 optimizer/scheduler/AMP scaler；目标为 ep10。\n"
        )
        handle.write(
            "- 其余协议保持 480x640、window 2x15x15、batch2、LR bb1e4、"
            "BN no_running、valid825 不变。\n"
        )
        handle.write(f"- config：`{CONFIG.relative_to(REPO)}`。\n")
        handle.write(f"- status：`{STATUS.relative_to(REPO)}`。\n")


def append_result() -> None:
    marker = "DSEC_FULLRES_W15_H67_BB1E4_STRICT_RESUME10_RESULT_20260802"
    if marker in REDESIGN.read_text(encoding="utf-8"):
        return
    ranking = ROOT / "profile_ranking_valid825.md"
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### H67 fullres bb1e4 strict ep10 结果\n\n")
        handle.write(f"<!-- {marker} -->\n\n")
        handle.write(ranking.read_text(encoding="utf-8"))


def main() -> int:
    for required in (SOURCE_CHECKPOINT, SOURCE_STATE):
        if not required.is_file():
            raise FileNotFoundError(required)

    run(
        [str(PYTHON), str(EXP / "entrypoints/make_dsec_fullres_w15_h67_bb1e4_resume10_config.py")],
        ROOT / "config.log",
        "generate config",
    )
    append_protocol()

    checkpoint = ROOT / "checkpoint_epoch10.pth"
    if not checkpoint.is_file():
        run(
            [
                str(PYTHON),
                "-u",
                str(EXP / "entrypoints/train.py"),
                "--config",
                str(CONFIG),
                "--prev_runid",
                str(SOURCE_CHECKPOINT),
                "--save_path",
                str(ROOT / "checkpoint_epoch{}.pth"),
                "--finetune",
                "1",
                "--resume",
                "1",
            ],
            ROOT / "train.log",
            "H67 bb1e4 strict resume ep5-to-ep10",
        )

    ranking = ROOT / "profile_ranking_valid825.md"
    if not ranking.is_file():
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
                "--epoch",
                "10",
            ],
            ROOT / "valid825.log",
            "H67 bb1e4 ep10 valid825",
        )
    append_result()
    record("ALL COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
