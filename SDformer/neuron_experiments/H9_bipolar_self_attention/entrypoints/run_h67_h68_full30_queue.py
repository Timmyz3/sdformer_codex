"""Run H67 then H68 full30 and standard valid825 evaluation."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
PY = Path(sys.executable)
TTX = (
    RESULTS
    / "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid"
    / "checkpoint_epoch2.pth"
)
EPOCHS = [0, 4, 9, 14, 19, 24, 28, 29]
STATUS = RESULTS / "h67_h68_full30_queue_status.log"


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(command: list[str], log: Path, label: str) -> None:
    env = os.environ.copy()
    env.update({
        "SDFORMER_USE_MLFLOW": "0",
        "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
        "SDFORMER_SNN_BACKEND": "cupy",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    log.parent.mkdir(parents=True, exist_ok=True)
    record(f"START {label}: {' '.join(command)}")
    with log.open("a", encoding="utf-8") as handle:
        proc = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    record(f"END {label}: exit_code={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed; log={log}")


def append_result(name: str, train_cfg: Path, eval_cfg: Path, run_dir: Path) -> None:
    marker = f"H67_H68_FULL30::{name}"
    current = REDESIGN.read_text(encoding="utf-8")
    if marker in current:
        return
    ranking = run_dir / "profile_ranking_valid825.md"
    rows = [line for line in ranking.read_text(encoding="utf-8").splitlines() if line.startswith("| ")]
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n### H67/H68 full30 自动结果：{name}\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(f"- train config: `{train_cfg.relative_to(REPO)}`\n")
        handle.write(f"- eval config: `{eval_cfg.relative_to(REPO)}`\n")
        handle.write(f"- start checkpoint: `{TTX.relative_to(REPO)}`\n")
        handle.write(f"- run dir: `{run_dir.relative_to(REPO)}`\n")
        handle.write(f"- ranking: `{ranking.relative_to(REPO)}`\n\n")
        for line in rows:
            handle.write(line + "\n")


def run_one(name: str, train_cfg: Path, eval_cfg: Path) -> None:
    run_dir = RESULTS / f"{name}_bs8_full30_20260711_setsid"
    run_dir.mkdir(parents=True, exist_ok=True)
    ranking = run_dir / "profile_ranking_valid825.md"
    if not ranking.exists():
        last = run_dir / "checkpoint_epoch29.pth"
        if not last.exists():
            run(
                [
                    str(PY), "-u", str(EXP / "entrypoints/train.py"),
                    "--config", str(train_cfg),
                    "--prev_runid", str(TTX),
                    "--save_path", str(run_dir / "checkpoint_epoch{}.pth"),
                ],
                run_dir / "train.log",
                f"{name} train full30",
            )
        eval_command = [
            str(PY), "-u", str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
            "--config", str(eval_cfg), "--run-dir", str(run_dir),
        ]
        for epoch in EPOCHS:
            eval_command.extend(["--epoch", str(epoch)])
        run(eval_command, run_dir / "valid825_queue.log", f"{name} valid825")
    append_result(name, train_cfg, eval_cfg, run_dir)
    record(f"COMPLETE {name}: ranking={ranking}")


def main() -> int:
    run([str(PY), str(EXP / "entrypoints/make_h67_h68_full30_configs.py")], STATUS, "generate configs")
    queue = [
        (
            "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30",
            GEN / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30.yml",
            GEN / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30.yml",
        ),
        (
            "h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30",
            GEN / "h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30.yml",
            GEN / "h68_allbinary_all12_castling_ttx_deploy_full30.yml",
        ),
    ]
    for name, train_cfg, eval_cfg in queue:
        run_one(name, train_cfg, eval_cfg)
    record("ALL COMPLETE H67 -> H68 full30 queue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

