"""Formal multi-checkpoint valid825 for DSEC full-res window9 FT runs.

Policy (hardware-consistent 480x640 / window[2,9,9], NOT paper window15):
  - Default eval epochs = force_save set used in training:
      0, 4, 9, 14, 19, 24, 28, 29
    Same selection as H67/H68 crop full30 formal valid825 queue.
  - Missing checkpoints are skipped; at least one epoch must exist.
  - Uses `run_h9_standard_valid825_eval.py` + the matching fullres yml so geometry
    (resolution/crop/window/remap) stays identical to fine-tune.
  - Artifacts land under:
      <run_dir>/standard_valid825/epoch{N}/
      <run_dir>/profile_ranking_valid825.md
      <run_dir>/fullres_formal_eval_summary.json

This entrypoint does not train and does not modify old crop experiments.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
MANIFEST = GEN / "dsec_fullres_window9_manifest.json"
STATUS = RESULTS / "dsec_fullres_window9_formal_eval_status.log"
PY = Path(sys.executable)

# Must match make_dsec_fullres_window9_configs.SAVE_EPOCHS / H67-H68 full30 eval set.
FORMAL_EVAL_EPOCHS = [0, 4, 9, 14, 19, 24, 28, 29]
RUN_TAG = "20260726"


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
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


def run(command: list[str], log: Path, label: str) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    record(f"START {label}: {' '.join(command)}")
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n$ {' '.join(command)}\n")
        handle.flush()
        proc = subprocess.run(
            command,
            cwd=REPO,
            env=environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    record(f"END {label}: exit_code={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed; log={log}")
    return int(proc.returncode)


def gpu_memory_used_mib() -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        return int(float(out.strip().splitlines()[0].strip()))
    except Exception:
        return None


def fullres_train_running() -> bool:
    """True if a DSEC fullres window9 train.py is holding the GPU."""
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
    except Exception:
        return False
    for line in out.splitlines():
        if "entrypoints/train.py" not in line:
            continue
        if "dsec_fullres_w9_" in line or "dsec_fullres_window9" in line:
            return True
    return False


def wait_for_gpu(*, max_used_mib: int, poll_seconds: int) -> None:
    while True:
        used = gpu_memory_used_mib()
        busy_train = fullres_train_running()
        if not busy_train and used is not None and used <= max_used_mib:
            record(f"GPU free for formal eval: used={used}MiB train_busy=0")
            return
        record(
            f"WAIT GPU for formal eval: used={used}MiB "
            f"train_busy={int(busy_train)} max_used={max_used_mib}MiB"
        )
        time.sleep(max(30, poll_seconds))


def load_manifest() -> list[dict[str, Any]]:
    if not MANIFEST.is_file():
        raise FileNotFoundError(MANIFEST)
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def run_dir_for(row: dict[str, Any], batch_size: int) -> Path:
    return RESULTS / f"{row['name']}_bs{batch_size}_{RUN_TAG}"


def parse_best_train_val_epoch(train_log: Path) -> int | None:
    """Best training-time validation loss epoch among FORMAL_EVAL_EPOCHS (if logged)."""
    if not train_log.is_file():
        return None
    text = train_log.read_text(encoding="utf-8", errors="ignore")
    # Pattern: "Epoch N" then later "Epoch loss (Validation): X"
    epoch = None
    best_epoch = None
    best_loss = None
    for line in text.splitlines():
        m_ep = re.match(r"^Epoch\s+(\d+)\s*$", line.strip())
        if m_ep:
            epoch = int(m_ep.group(1))
            continue
        m_val = re.search(r"Epoch loss \(Validation\):\s*([0-9.eE+-]+)", line)
        if m_val and epoch is not None:
            loss = float(m_val.group(1))
            if best_loss is None or loss < best_loss:
                best_loss = loss
                best_epoch = epoch
    if best_epoch is None:
        return None
    # Map to nearest saved formal epoch <= best_epoch, prefer exact if saved.
    if best_epoch in FORMAL_EVAL_EPOCHS:
        return best_epoch
    candidates = [e for e in FORMAL_EVAL_EPOCHS if e <= best_epoch]
    return candidates[-1] if candidates else None


def select_epochs(run_dir: Path, requested: list[int] | None) -> list[int]:
    epochs = list(requested) if requested else list(FORMAL_EVAL_EPOCHS)
    # Ensure train-val best mapped epoch is included when available.
    mapped = parse_best_train_val_epoch(run_dir / "train.log")
    if mapped is not None and mapped not in epochs:
        epochs.append(mapped)
    epochs = sorted(set(epochs))
    existing = [e for e in epochs if (run_dir / f"checkpoint_epoch{e}.pth").is_file()]
    return existing


def ranking_rows(ranking: Path) -> list[str]:
    if not ranking.is_file():
        return []
    return [line for line in ranking.read_text(encoding="utf-8").splitlines() if line.startswith("| ")]


def parse_ranking_best(ranking: Path) -> dict[str, Any] | None:
    rows = ranking_rows(ranking)
    # header then separator then data; rank 1 is first data row after sort in eval script
    data = []
    for line in rows:
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 4:
            continue
        if parts[0] in {"rank", "---:", "---"} or parts[0].startswith("---"):
            continue
        try:
            data.append(
                {
                    "rank": int(parts[0]),
                    "epoch": int(parts[1]),
                    "AEE": float(parts[2]),
                    "AAE": float(parts[3]),
                }
            )
        except ValueError:
            continue
    if not data:
        return None
    data.sort(key=lambda r: r["rank"])
    return data[0]


def append_redesign(model_id: str, config: Path, run_dir: Path, epochs: list[int]) -> None:
    marker = f"DSEC_FULLRES_W9_FORMAL_EVAL::{model_id}::{run_dir.name}"
    text = REDESIGN.read_text(encoding="utf-8") if REDESIGN.is_file() else ""
    if marker in text:
        return
    ranking = run_dir / "profile_ranking_valid825.md"
    best = parse_ranking_best(ranking)
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n### DSEC fullres window9 正式 valid825：{model_id}\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(
            "- protocol：`480x640` / `window=[2,9,9]` / `crop=null` / `remap=v1`；"
            "standard valid825 via `eval_DSEC_flow_SNN.py --mode valid`；"
            "**不是** paper window15。\n"
        )
        handle.write(
            f"- eval epochs policy：`{FORMAL_EVAL_EPOCHS}`（= force_save set；"
            "missing skipped；train-val best mapped epoch auto-included if missing）。\n"
        )
        handle.write(f"- evaluated epochs：`{epochs}`\n")
        handle.write(f"- config：`{config.relative_to(REPO)}`\n")
        handle.write(f"- run dir：`{run_dir.relative_to(REPO)}`\n")
        handle.write(f"- ranking：`{(run_dir / 'profile_ranking_valid825.md').relative_to(REPO)}`\n")
        if best:
            handle.write(
                f"- best rank1：epoch{best['epoch']} AEE={best['AEE']:.4f} AAE={best['AAE']:.4f}\n"
            )
        handle.write("\n")
        for line in ranking_rows(ranking):
            handle.write(line + "\n")


def formal_eval_one(
    *,
    model_id: str,
    config: Path,
    run_dir: Path,
    epochs: list[int] | None,
) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    config = config.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    if not (run_dir / "checkpoint_epoch29.pth").is_file():
        # allow partial if user forces epochs, but default requires completed FT30
        if epochs is None:
            raise FileNotFoundError(f"missing final checkpoint: {run_dir / 'checkpoint_epoch29.pth'}")

    selected = select_epochs(run_dir, epochs)
    if not selected:
        raise RuntimeError(f"no evaluable checkpoints under {run_dir} for epochs={epochs or FORMAL_EVAL_EPOCHS}")

    summary_path = run_dir / "fullres_formal_eval_summary.json"
    ranking = run_dir / "profile_ranking_valid825.md"
    if ranking.is_file() and summary_path.is_file():
        prev = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            prev.get("epochs") == selected
            and prev.get("status") == "complete"
            and prev.get("ranking_mode") == "aee"
        ):
            record(f"REUSE formal eval {model_id}: {ranking}")
            return prev

    command = [
        str(PY),
        "-u",
        str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
        "--config",
        str(config),
        "--run-dir",
        str(run_dir),
        "--ranking-mode",
        "aee",
    ]
    for epoch in selected:
        command.extend(["--epoch", str(epoch)])

    run(command, run_dir / "fullres_formal_valid825_queue.log", f"{model_id} fullres formal valid825")
    best = parse_ranking_best(ranking)
    summary = {
        "status": "complete",
        "model_id": model_id,
        "config": str(config),
        "run_dir": str(run_dir),
        "epochs_policy": FORMAL_EVAL_EPOCHS,
        "epochs": selected,
        "ranking": str(ranking),
        "ranking_mode": "aee",
        "best": best,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "protocol": "480x640_window9_hardware_consistent_valid825",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    append_redesign(model_id, config, run_dir, selected)
    record(f"COMPLETE formal eval {model_id}: best={best} ranking={ranking}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ids",
        nargs="+",
        choices=("NB0", "H67", "H66d"),
        help="Which fullres models to evaluate (default: all in manifest order).",
    )
    parser.add_argument("--batch-size", type=int, default=2, choices=(1, 2))
    parser.add_argument(
        "--epoch",
        action="append",
        type=int,
        default=[],
        help="Override formal epochs (repeatable). Default uses FORMAL_EVAL_EPOCHS.",
    )
    parser.add_argument(
        "--wait-gpu",
        action="store_true",
        help="Wait until no fullres train is running and GPU memory is below threshold.",
    )
    parser.add_argument("--max-used-mib", type=int, default=8192)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument(
        "--require-final",
        action="store_true",
        default=True,
        help="Require checkpoint_epoch29 before evaluating a model (default on).",
    )
    parser.add_argument(
        "--no-require-final",
        action="store_false",
        dest="require_final",
    )
    parser.add_argument(
        "--wait-ready",
        action="store_true",
        help="If a model is not finished, wait until checkpoint_epoch29 appears.",
    )
    parser.add_argument("--wait-timeout-hours", type=float, default=96.0)
    args = parser.parse_args()

    rows = load_manifest()
    if args.ids:
        wanted = set(args.ids)
        rows = [row for row in rows if row["id"] in wanted]
    epochs = args.epoch or None

    record(
        f"FORMAL EVAL START ids={[r['id'] for r in rows]} "
        f"epochs={epochs or FORMAL_EVAL_EPOCHS} wait_gpu={args.wait_gpu}"
    )

    summaries = []
    for row in rows:
        run_dir = run_dir_for(row, args.batch_size)
        final_ckpt = run_dir / "checkpoint_epoch29.pth"
        if args.wait_ready and not final_ckpt.is_file():
            deadline = time.time() + args.wait_timeout_hours * 3600.0
            while not final_ckpt.is_file():
                if time.time() > deadline:
                    raise TimeoutError(f"timeout waiting for {final_ckpt}")
                record(f"WAIT train finish {row['id']}: missing {final_ckpt.name}")
                time.sleep(max(30, args.poll_seconds))
        if args.require_final and not final_ckpt.is_file():
            record(f"SKIP {row['id']}: final checkpoint missing ({final_ckpt})")
            continue
        if args.wait_gpu:
            wait_for_gpu(max_used_mib=args.max_used_mib, poll_seconds=args.poll_seconds)
        summary = formal_eval_one(
            model_id=row["id"],
            config=Path(row["config"]),
            run_dir=run_dir,
            epochs=epochs,
        )
        summaries.append(summary)

    record(f"ALL COMPLETE fullres formal eval: n={len(summaries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
