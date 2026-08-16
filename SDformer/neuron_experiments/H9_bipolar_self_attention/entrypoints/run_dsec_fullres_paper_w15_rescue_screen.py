"""Run the one-epoch/full-valid825 H67 full-resolution rescue screen."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
TAG = "20260801"
ROOT = RESULTS / f"dsec_fullres_w15_rescue_screen_{TAG}"
STATUS = ROOT / "status.log"
MANIFEST = GEN / "dsec_fullres_w15_rescue_screen_manifest.json"
PY = Path(sys.executable)


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
    log.parent.mkdir(parents=True, exist_ok=True)
    record(f"START {label}: {' '.join(command)}")
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(command) + "\n")
        handle.flush()
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


def parse_rank1(path: Path) -> dict[str, Any]:
    headers: list[str] | None = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and cells[0] == "rank":
            headers = cells
            continue
        if headers and cells and cells[0] == "1":
            return dict(zip(headers, cells))
    raise RuntimeError(f"rank-1 row missing from {path}")


def append_protocol(rows: list[dict[str, Any]]) -> None:
    marker = "DSEC_FULLRES_W15_LR_RESCUE_SCREEN_20260801"
    if marker in REDESIGN.read_text(encoding="utf-8"):
        return
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### DSEC fullres LR rescue 短筛（2026-08-01）\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(
            "- 失败诊断：旧 H67/H66d fullres backbone/norm LR 为 `2e-6/1e-6`，"
            "NB0 为 `1e-4`；旧候选不是等强度 fullres adaptation。\n"
        )
        handle.write(
            "- 固定结构、480x640、2x15x15、remap=v1、BN=no_running、batch2；"
            "每项完整训练 1 epoch 后跑 standard valid825。\n"
        )
        handle.write(
            "- 两条 own-crop LR 仅改变优化强度；NB0-fullres conversion 只用于判别"
            "初始化问题，不自动作为论文主协议。\n"
        )
        for row in rows:
            handle.write(
                f"- `{row['id']}`：profile `{row['profile']}`，init `{row['init']}`，"
                f"config `{Path(row['config']).relative_to(REPO)}`。\n"
            )
        handle.write(f"- status：`{STATUS.relative_to(REPO)}`。\n")


def append_results(summary: Path) -> None:
    marker = "DSEC_FULLRES_W15_LR_RESCUE_SCREEN_RESULT_20260801"
    text = REDESIGN.read_text(encoding="utf-8")
    if marker in text:
        return
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### DSEC fullres LR rescue 短筛结果\n\n")
        handle.write(f"<!-- {marker} -->\n\n")
        handle.write(summary.read_text(encoding="utf-8"))


def main() -> int:
    run(
        [
            str(PY),
            str(EXP / "entrypoints/make_dsec_fullres_paper_w15_rescue_configs.py"),
            "--mode",
            "screen",
            "--batch-size",
            "2",
        ],
        STATUS,
        "generate rescue screen configs",
    )
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    append_protocol(rows)
    results = []
    for row in rows:
        run_dir = ROOT / row["id"]
        checkpoint = run_dir / "checkpoint_epoch0.pth"
        train_log = run_dir / "train.log"
        if not checkpoint.is_file():
            run(
                [
                    str(PY),
                    "-u",
                    str(EXP / "entrypoints/train.py"),
                    "--config",
                    row["config"],
                    "--prev_runid",
                    row["checkpoint"],
                    "--save_path",
                    str(run_dir / "checkpoint_epoch{}.pth"),
                    "--finetune",
                    "1",
                ],
                train_log,
                f"{row['id']} train1",
            )
        ranking = run_dir / "profile_ranking_valid825.md"
        if not ranking.is_file():
            run(
                [
                    str(PY),
                    "-u",
                    str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
                    "--config",
                    row["config"],
                    "--run-dir",
                    str(run_dir),
                    "--ranking-mode",
                    "aee",
                    "--epoch",
                    "0",
                ],
                run_dir / "valid825.log",
                f"{row['id']} valid825",
            )
        results.append({**row, "metrics": parse_rank1(ranking)})

    results.sort(key=lambda item: float(item["metrics"]["AEE"]))
    summary_json = ROOT / "summary.json"
    summary_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    summary_md = ROOT / "summary.md"
    lines = [
        "# DSEC fullres window15 rescue screen",
        "",
        "| rank | candidate | init | LR profile | AEE | AAE benchmark | spikes | energy |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(results, start=1):
        metric = row["metrics"]
        lines.append(
            f"| {rank} | {row['id']} | {row['init']} | {row['profile']} | "
            f"{metric.get('AEE', '')} | {metric.get('AAE benchmark', '')} | "
            f"{metric.get('total_spikes', '')} | {metric.get('spike_energy_proxy_uj', '')} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    append_results(summary_md)
    record(f"ALL COMPLETE summary={summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
