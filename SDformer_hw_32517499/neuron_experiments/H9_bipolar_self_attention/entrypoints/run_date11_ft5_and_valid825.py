"""Run a DATE11 FT5 experiment from a checkpoint and evaluate epochs 0..4."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
REDESIGN_MD = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
PY = Path(sys.executable)


def experiment_name(config: Path) -> str:
    data = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    return str(data.get("experiment") or config.stem)


def run(command: list[str], log: Path, label: str) -> None:
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["SDFORMER_SNN_BACKEND"] = "cupy"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== {label} {datetime.now().isoformat(timespec='seconds')} ===\n")
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        proc = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[date11-ft5] {label} exit_code={proc.returncode}\n")
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed; log={log}")


def ranking_rows(ranking: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in ranking.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 10 or cells[0] in {"rank", "---:"}:
            continue
        rows.append(
            {
                "rank": cells[0],
                "epoch": cells[1],
                "AEE": cells[2],
                "AAE": cells[3],
                "PE1": cells[4],
                "PE2": cells[5],
                "outlier": cells[6],
                "spikes": cells[7],
                "firing": cells[8],
                "energy": cells[9],
            }
        )
    return rows


def append_md(label: str, config: Path, resume: Path, run_dir: Path) -> None:
    marker = f"DATE11_FT_APPEND::{run_dir.name}"
    text = REDESIGN_MD.read_text(encoding="utf-8")
    if marker in text:
        return
    ranking = run_dir / "profile_ranking_valid825.md"
    rows = ranking_rows(ranking)
    with REDESIGN_MD.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n### DATE11 自动结果追加：{label}（{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}）\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(f"- 配置：`{config.relative_to(REPO)}`\n")
        handle.write(f"- 起点：`{resume.relative_to(REPO)}`\n")
        handle.write(f"- 运行目录：`{run_dir.relative_to(REPO)}`\n")
        handle.write(f"- 标准 valid825 ranking：`{ranking.relative_to(REPO)}`\n")
        if rows:
            best = rows[0]
            handle.write(
                f"- best：epoch `{best['epoch']}`，AEE `{best['AEE']}`，AAE `{best['AAE']}`，"
                f"total_spikes `{best['spikes']}`，firing `{best['firing']}`，energy `{best['energy']}uJ`。\n\n"
            )
            handle.write("| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |\n")
            handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in rows:
                handle.write(
                    f"| {row['rank']} | {row['epoch']} | {row['AEE']} | {row['AAE']} | "
                    f"{row['PE1']} | {row['PE2']} | {row['outlier']} | {row['spikes']} | "
                    f"{row['firing']} | {row['energy']} |\n"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    config = args.config.resolve()
    resume = args.resume.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{experiment_name(config)}_bs8_{stamp}_setsid"
    run_dir = EXP / "results" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "pipeline.log"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"=== DATE11 FT5 start {datetime.now().isoformat(timespec='seconds')} ===\n")
        handle.write(f"label={args.label}\nconfig={config}\nresume={resume}\nrun_dir={run_dir}\n")

    run(
        [
            str(PY),
            "-u",
            str(EXP / "entrypoints/train.py"),
            "--config",
            str(config),
            "--prev_runid",
            str(resume),
            "--save_path",
            str(run_dir / "checkpoint_epoch{}.pth"),
        ],
        log,
        "train",
    )
    run(
        [
            str(PY),
            "-u",
            str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
            "--config",
            str(config),
            "--run-dir",
            str(run_dir),
            "--epoch",
            "0",
            "--epoch",
            "1",
            "--epoch",
            "2",
            "--epoch",
            "3",
            "--epoch",
            "4",
        ],
        log,
        "standard_valid825",
    )
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"=== DATE11 FT5 complete {datetime.now().isoformat(timespec='seconds')} ===\n")
    append_md(args.label or experiment_name(config), config, resume, run_dir)
    print(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
