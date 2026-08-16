"""Run H66f Local5+TP then H66g Local5+Motion full30 + valid825 from TTX ep2."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
PY = Path(sys.executable)
TTX = RESULTS / (
    "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/"
    "checkpoint_epoch2.pth"
)
STATUS = RESULTS / "h66f_h66g_local5_combo_status.log"
MANIFEST = GEN / "h66f_h66g_local5_combo_full30_manifest.json"
EPOCHS = (0, 4, 9, 14, 19, 24, 28, 29)
RUN_TAG = "20260723_setsid"


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


def append_result(name: str, config: Path, run_dir: Path) -> None:
    marker = f"H66_COMBO_FULL30::{name}"
    text = REDESIGN.read_text(encoding="utf-8") if REDESIGN.exists() else ""
    if marker in text:
        return
    ranking = run_dir / "profile_ranking_valid825.md"
    table = [line for line in ranking.read_text(encoding="utf-8").splitlines() if line.startswith("| ")]
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n### H66 combo full30 自动结果：{name}\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(f"- config: `{config.relative_to(REPO)}`\n")
        handle.write(f"- start checkpoint: `{TTX.relative_to(REPO)}`\n")
        handle.write(f"- run dir: `{run_dir.relative_to(REPO)}`\n\n")
        for line in table:
            handle.write(line + "\n")


def main() -> int:
    run(
        [str(PY), str(EXP / "entrypoints/make_h66f_h66g_local5_combo_configs.py")],
        STATUS,
        "generate H66f/H66g configs",
    )
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for row in rows:
        name = row["name"]
        config = Path(row["config"])
        run_dir = RESULTS / f"{name}_bs8_full30_{RUN_TAG}"
        run_dir.mkdir(parents=True, exist_ok=True)
        final = run_dir / "checkpoint_epoch29.pth"
        ranking = run_dir / "profile_ranking_valid825.md"
        if not final.exists():
            run(
                [
                    str(PY), "-u", str(EXP / "entrypoints/train.py"),
                    "--config", str(config),
                    "--prev_runid", str(TTX),
                    "--save_path", str(run_dir / "checkpoint_epoch{}.pth"),
                ],
                run_dir / "train.log",
                f"{name} train full30",
            )
        else:
            record(f"REUSE completed {name} full30: {final}")
        if not ranking.exists():
            command = [
                str(PY), "-u", str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
                "--config", str(config),
                "--run-dir", str(run_dir),
            ]
            for epoch in EPOCHS:
                command.extend(["--epoch", str(epoch)])
            run(command, run_dir / "valid825_queue.log", f"{name} valid825")
        else:
            record(f"REUSE completed {name} valid825: {ranking}")
        append_result(name, config, run_dir)
        record(f"COMPLETE {name}: {ranking}")
    record(f"ALL COMPLETE H66F/H66G LOCAL5 COMBO: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
