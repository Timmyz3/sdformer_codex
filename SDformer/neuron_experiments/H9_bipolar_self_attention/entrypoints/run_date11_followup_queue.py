"""DATE11 follow-up queue for full factorial ablations.

Waits for an already launched DATE11 run, appends its standard-valid825 result
to the redesign markdown, then runs the next DATE11 config and appends that
result as well.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
REDESIGN_MD = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
PY = Path(sys.executable)
BASELINE = REPO / "experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
EPOCHS = [9, 14, 19, 24, 28, 29]


def append_status(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{datetime.now().isoformat(timespec='seconds')}] {message}\n")


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_process(pid: int, status: Path, label: str) -> None:
    append_status(status, f"wait {label} pid={pid}")
    while process_alive(pid):
        time.sleep(300)
        append_status(status, f"still waiting {label} pid={pid}")
    append_status(status, f"done waiting {label} pid={pid}")


def run(command: list[str], log_path: Path, status: Path, label: str) -> None:
    append_status(status, f"start {label}: {' '.join(command)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SDFORMER_USE_MLFLOW"] = "0"
    env["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    env["SDFORMER_SNN_BACKEND"] = "cupy"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(command)}\n")
        handle.flush()
        proc = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"\n[date11-queue] exit_code={proc.returncode}\n")
    append_status(status, f"finish {label} exit={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed; log={log_path}")


def wait_gpu_mem(status: Path, limit_mib: int = 8000) -> None:
    while True:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            cwd=REPO,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        first = (proc.stdout.strip().splitlines() or ["0"])[0].strip()
        try:
            used = int(first)
        except ValueError:
            used = 0
        if used <= limit_mib:
            return
        append_status(status, f"wait gpu memory used={used}MiB")
        time.sleep(120)


def ranking_rows(ranking: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not ranking.exists():
        return rows
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


def first_match(path: Path, pattern: str) -> str | None:
    if not path.exists():
        return None
    regex = re.compile(pattern)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = regex.search(line)
        if match:
            return match.group(1)
    return None


def experiment_name(config: Path) -> str:
    for line in config.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^experiment:\s*['\"]?([^'\"]+?)['\"]?\s*$", line)
        if match:
            return match.group(1)
    return config.stem


def append_md_once(label: str, config: Path, run_dir: Path, status: Path) -> None:
    marker = f"DATE11_APPEND::{run_dir.name}"
    text = REDESIGN_MD.read_text(encoding="utf-8")
    if marker in text:
        append_status(status, f"md already contains {marker}")
        return

    ranking = run_dir / "profile_ranking_valid825.md"
    rows = ranking_rows(ranking)
    pipe_log = run_dir / "pipeline.log"
    train_overlay = first_match(pipe_log, r"\[H9\] installed ATLIFTernaryPSN before load: (\d+) modules")
    train_attn = first_match(pipe_log, r"\[H9\] attention summary after install: \{'num_modules': (\d+)\}")
    load_audit = first_match(pipe_log, r"\[H9\] load audit: (checkpoint_overlay_keys=\d+, missing=\d+, unexpected=\d+)")

    with REDESIGN_MD.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n### DATE11 自动结果追加：{label}（{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}）\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(f"- 配置：`{config.relative_to(REPO)}`\n")
        handle.write(f"- 运行目录：`{run_dir.relative_to(REPO)}`\n")
        handle.write(f"- 标准 valid825 ranking：`{ranking.relative_to(REPO)}`\n")
        if train_overlay or train_attn or load_audit:
            handle.write(
                f"- 加载审计：ATLIF `{train_overlay or 'NA'}`，Shiftmax `{train_attn or '0'}`，"
                f"`{load_audit or 'NA'}`\n"
            )
        if rows:
            best = rows[0]
            handle.write(
                f"- best：epoch `{best['epoch']}`，AEE `{best['AEE']}`，AAE `{best['AAE']}`，"
                f"total_spikes `{best['spikes']}`，firing `{best['firing']}`。\n\n"
            )
            handle.write("| rank | epoch | AEE | AAE | PE1 | PE2 | outlier | total_spikes | firing | energy_uj |\n")
            handle.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
            for row in rows:
                handle.write(
                    f"| {row['rank']} | {row['epoch']} | {row['AEE']} | {row['AAE']} | "
                    f"{row['PE1']} | {row['PE2']} | {row['outlier']} | {row['spikes']} | "
                    f"{row['firing']} | {row['energy']} |\n"
                )
        else:
            handle.write("- 状态：未找到 ranking 表，需人工检查 pipeline/eval 日志。\n")
    append_status(status, f"md appended for {label}")


def run_full_pipeline(label: str, config: Path, run_dir: Path, status: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log = run_dir / "pipeline.log"
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"=== {label} full30+valid825 start {datetime.now().isoformat(timespec='seconds')} ===\n")
        handle.write(f"python={PY}\nconfig={config}\nresume={BASELINE}\nrun_dir={run_dir}\n")

    run([str(PY), "-u", str(EXP / "entrypoints/verify_nts11_chain.py"), str(config)], log, status, f"{label} verify")
    wait_gpu_mem(status)
    run(
        [
            str(PY),
            "-u",
            str(EXP / "entrypoints/train.py"),
            "--config",
            str(config),
            "--prev_runid",
            str(BASELINE),
            "--save_path",
            str(run_dir / "checkpoint_epoch{}.pth"),
        ],
        log,
        status,
        f"{label} train",
    )
    wait_gpu_mem(status)
    run(
        [
            str(PY),
            "-u",
            str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
            "--config",
            str(config),
            "--run-dir",
            str(run_dir),
            *sum((["--epoch", str(epoch)] for epoch in EPOCHS), []),
        ],
        log,
        status,
        f"{label} standard_valid825",
    )
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"=== {label} pipeline complete {datetime.now().isoformat(timespec='seconds')} ===\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--first-label", required=True)
    parser.add_argument("--first-config", type=Path, required=True)
    parser.add_argument("--first-run-dir", type=Path, required=True)
    parser.add_argument("--next-label", default="ternary + SC")
    parser.add_argument(
        "--next-config",
        type=Path,
        default=EXP / "configs/generated/date11full_all_ternary_atlif_sc_w720_fastlr_full30.yml",
    )
    args = parser.parse_args()

    driver = EXP / "results" / f"date11_followup_queue_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    status = driver / "status.log"
    driver.mkdir(parents=True, exist_ok=True)
    append_status(status, f"driver={driver}")

    wait_process(args.wait_pid, status, args.first_label)
    first_ranking = args.first_run_dir / "profile_ranking_valid825.md"
    for _ in range(120):
        if first_ranking.exists():
            break
        append_status(status, f"wait ranking for {args.first_label}: {first_ranking}")
        time.sleep(60)
    append_md_once(args.first_label, args.first_config.resolve(), args.first_run_dir.resolve(), status)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    next_run_dir = EXP / "results" / f"{experiment_name(args.next_config)}_bs8_{stamp}_setsid"
    run_full_pipeline(args.next_label, args.next_config.resolve(), next_run_dir, status)
    append_md_once(args.next_label, args.next_config.resolve(), next_run_dir.resolve(), status)
    append_status(status, "queue complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
