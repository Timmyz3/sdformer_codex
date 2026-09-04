"""Run two five-epoch H67 full-resolution rescue continuations serially."""

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
ROOT = EXP / "results/dsec_fullres_w15_rescue_short5_20260801"
STATUS = ROOT / "status.log"
MANIFEST = GEN / "dsec_fullres_w15_rescue_short5_manifest.json"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
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
    marker = "DSEC_FULLRES_W15_LR_RESCUE_SHORT5_20260801"
    if marker in REDESIGN.read_text(encoding="utf-8"):
        return
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### DSEC fullres LR rescue short5 续跑（2026-08-01）\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(
            "- 只续 own-crop `bb1e4` 与 `bb2e5`，各从 screen epoch0 模型补 5 epochs；"
            "NB0-fullres conversion 不续。\n"
        )
        handle.write(
            "- screen 未保存 optimizer/scaler state，因此两条线都从已训练模型重建 AdamW；"
            "这是 model continuation，不写成 strict optimizer resume。\n"
        )
        handle.write(
            "- 结构、480x640、window2x15x15、remap=v1、BN=no_running、batch2、LR profile "
            "保持不变；checkpoint epoch offset=1，最终为 epoch5。\n"
        )
        for row in rows:
            handle.write(
                f"- `{row['id']}`：config `{Path(row['config']).relative_to(REPO)}`，"
                f"source `{Path(row['checkpoint']).relative_to(REPO)}`。\n"
            )
        handle.write(f"- status：`{STATUS.relative_to(REPO)}`。\n")


def write_summary(results: list[dict[str, Any]]) -> Path:
    results.sort(key=lambda item: float(item["metrics"]["AEE"]))
    (ROOT / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    summary = ROOT / "summary.md"
    lines = [
        "# DSEC fullres window15 rescue short5",
        "",
        "| rank | candidate | LR | epoch | AEE | AAE benchmark | outlier | spikes | energy |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(results, start=1):
        metric = row["metrics"]
        lines.append(
            f"| {rank} | {row['id']} | {row['profile']} | {row['final_epoch']} | "
            f"{metric.get('AEE', '')} | {metric.get('AAE benchmark', '')} | "
            f"{metric.get('outlier', '')} | {metric.get('total_spikes', '')} | "
            f"{metric.get('spike_energy_proxy_uj', '')} |"
        )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def append_results(summary: Path) -> None:
    marker = "DSEC_FULLRES_W15_LR_RESCUE_SHORT5_RESULT_20260801"
    if marker in REDESIGN.read_text(encoding="utf-8"):
        return
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### DSEC fullres LR rescue short5 结果\n\n")
        handle.write(f"<!-- {marker} -->\n\n")
        handle.write(summary.read_text(encoding="utf-8"))


def main() -> int:
    run(
        [str(PY), str(EXP / "entrypoints/make_dsec_fullres_w15_rescue_short5_configs.py")],
        ROOT / "launcher.log",
        "generate rescue short5 configs",
    )
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    append_protocol(rows)
    results = []
    for row in rows:
        run_dir = ROOT / row["id"]
        final_checkpoint = run_dir / f"checkpoint_epoch{row['final_epoch']}.pth"
        if not final_checkpoint.is_file():
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
                run_dir / "train.log",
                f"{row['id']} continue5",
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
                    str(row["final_epoch"]),
                ],
                run_dir / "valid825.log",
                f"{row['id']} epoch{row['final_epoch']} valid825",
            )
        results.append({**row, "metrics": parse_rank1(ranking)})

    summary = write_summary(results)
    append_results(summary)
    record(f"ALL COMPLETE summary={summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
