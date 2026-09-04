"""Finish and evaluate H73 without launching the rejected H74-H80 queue."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
RESULTS = EXP / "results"
CONFIG = EXP / "configs/generated/h73_allbinary_all12_de9_match_code_w720_fastlr_full30_bs4acc2.yml"
RUN_DIR = RESULTS / "h73_allbinary_all12_de9_match_code_w720_fastlr_full30_bs4acc2_20260720_setsid"
FINAL = RUN_DIR / "checkpoint_epoch29.pth"
RANKING = RUN_DIR / "profile_ranking_valid825.md"
STATUS = RESULTS / "h73_only_after_date_review_status.log"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
PY = Path("/opt/conda/envs/sdformerflow/bin/python")
EPOCHS = (0, 4, 9, 14, 19, 24, 28, 29)


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def run(command: list[str], log: Path, label: str) -> None:
    env = os.environ.copy()
    env.update({
        "SDFORMER_USE_MLFLOW": "0",
        "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
        "SDFORMER_SNN_BACKEND": "cupy",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    record(f"START {label}: {' '.join(command)}")
    with log.open("a", encoding="utf-8") as handle:
        proc = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    record(f"END {label}: exit_code={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed; log={log}")


def audit_training_log() -> None:
    text = (RUN_DIR / "train.log").read_text(encoding="utf-8", errors="ignore")
    required = (
        r"installed ATLIFTernaryPSN before load: 105 modules",
        r"installed attention before load: 12 modules",
        r"load audit: checkpoint_overlay_keys=210, missing=12, unexpected=0",
        r"initialized new Match-Code weights: 12",
    )
    if any(re.search(pattern, text) is None for pattern in required):
        raise RuntimeError("H73 warm-start audit is incomplete")


def append_result() -> None:
    marker = "H73_ONLY_AFTER_DATE_REVIEW_20260720"
    if marker in REDESIGN.read_text(encoding="utf-8"):
        return
    rows = [line for line in RANKING.read_text(encoding="utf-8").splitlines() if line.startswith("| ")]
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### 43.41 H73 DATE novelty-gated full30 结果\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write("- H73 是 Match-Code 基础机制的 full30 代表；H74-H78 经 DATE novelty review 后未训练，H79/H80 按作者决定作为 assignment 增强候选另行补跑。\n")
        handle.write(f"- config: `{CONFIG.relative_to(REPO)}`\n")
        handle.write(f"- run dir: `{RUN_DIR.relative_to(REPO)}`\n")
        handle.write("- load: ATLIF105, attention12, warm-start overlay210/missing12/unexpected0; trained checkpoint strict-load missing0/unexpected0.\n\n")
        for row in rows:
            handle.write(row + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-pid", type=int, required=True)
    args = parser.parse_args()
    record(f"WAIT H73 train pid={args.train_pid}; H74-H80 launch disabled")
    while alive(args.train_pid):
        time.sleep(300)
    if not FINAL.exists():
        raise RuntimeError(f"H73 training exited without {FINAL}")
    audit_training_log()

    strict = RUN_DIR / "trained_strict_load_audit.json"
    if not strict.exists():
        run(
            [
                str(PY), str(EXP / "entrypoints/verify_round3_match_chain.py"),
                "--trained", str(CONFIG), str(FINAL), "--output", str(strict),
            ],
            RUN_DIR / "trained_strict_load_audit.log",
            "H73 trained strict-load audit",
        )
    if not RANKING.exists():
        command = [
            str(PY), "-u", str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
            "--config", str(CONFIG), "--run-dir", str(RUN_DIR),
        ]
        for epoch in EPOCHS:
            command.extend(["--epoch", str(epoch)])
        run(command, RUN_DIR / "valid825_queue.log", "H73 standard valid825")
    append_result()
    run(
        [str(PY), str(EXP / "entrypoints/prune_ranked_checkpoints.py"), str(RUN_DIR)],
        RUN_DIR / "checkpoint_prune.log",
        "H73 ranked checkpoint pruning",
    )
    record(f"ALL COMPLETE H73 only: {RANKING}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
