"""Run the memory-safe H73-H80 full30 queue after the H73 batch8 OOM."""

from __future__ import annotations

import argparse
import json
import os
import re
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
    "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_"
    "20260629_154937_setsid/checkpoint_epoch2.pth"
)
MANIFEST = GEN / "h73_h80_bs4acc2_full30_manifest.json"
STATUS = RESULTS / "h73_h80_bs4acc2_queue_status.log"
EPOCHS = (0, 4, 9, 14, 19, 24, 28, 29)


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
        proc = subprocess.run(
            command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT
        )
    record(f"END {label}: exit_code={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed; log={log}")


def audit_warmstart(log: Path, row: dict) -> None:
    text = log.read_text(encoding="utf-8", errors="ignore")
    expected_missing = int(row["expected_missing"])
    required = {
        "ATLIF105": r"installed ATLIFTernaryPSN before load: 105 modules",
        "attention12": r"installed attention before load: 12 modules",
        "codebook12": r"initialized new Match-Code weights: 12",
        "load audit": (
            rf"load audit: checkpoint_overlay_keys=210, "
            rf"missing={expected_missing}, unexpected=0"
        ),
    }
    missing = [name for name, pattern in required.items() if not re.search(pattern, text)]
    if missing:
        raise RuntimeError(f"{row['id']} warm-start audit failed: {missing}; log={log}")
    record(
        f"PASS {row['id']} load: ATLIF=105 attention=12 overlay=210 "
        f"missing={expected_missing} unexpected=0"
    )


def strict_audit(row: dict, config: Path, checkpoint: Path, run_dir: Path) -> None:
    candidate = row["id"]
    if candidate in {"H76", "H77", "H78"}:
        verifier = EXP / "entrypoints/verify_round3_match_chain.py"
    elif candidate in {"H79", "H80"}:
        verifier = EXP / "entrypoints/verify_round4_assignment_chain.py"
    else:
        verifier = EXP / "entrypoints/verify_match_code_chain.py"
    command = [str(PY), str(verifier)]
    if candidate in {"H73", "H74", "H75"}:
        command.extend(["--config", str(config), "--checkpoint", str(checkpoint)])
    else:
        command.extend(["--trained", str(config), str(checkpoint)])
    command.extend(["--output", str(run_dir / "trained_strict_load_audit.json")])
    run(command, run_dir / "trained_strict_load_audit.log", f"{candidate} strict load audit")


def append_result(row: dict, config: Path, run_dir: Path) -> None:
    marker = f"MATCH_BS4ACC2_FULL30::{row['id']}::{row['name']}"
    if marker in REDESIGN.read_text(encoding="utf-8"):
        return
    ranking = run_dir / "profile_ranking_valid825.md"
    table = [line for line in ranking.read_text(encoding="utf-8").splitlines() if line.startswith("| ")]
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n### {row['id']} Match-Code full30 显存安全结果\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(f"- config: `{config.relative_to(REPO)}`\n")
        handle.write(f"- start checkpoint: `{TTX.relative_to(REPO)}`\n")
        handle.write(f"- run dir: `{run_dir.relative_to(REPO)}`\n")
        handle.write(
            "- protocol: batch4, accumulation2, effective batch8, warmup1440 micro-steps "
            "= 720 optimizer updates = 5760 samples; full30; standard valid825.\n"
        )
        handle.write(
            f"- load: ATLIF105, attention12, overlay210, warm-start "
            f"missing{row['expected_missing']}/unexpected0; trained strict-load audited.\n\n"
        )
        for line in table:
            handle.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ids",
        nargs="+",
        choices=("H73", "H74", "H75", "H76", "H77", "H78", "H79", "H80"),
        help="Run only the selected candidates, preserving manifest order.",
    )
    args = parser.parse_args()
    run(
        [str(PY), str(EXP / "entrypoints/make_h73_h80_bs4acc2_configs.py")],
        STATUS,
        "generate H73-H80 batch4-acc2 configs",
    )
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if args.ids:
        selected = set(args.ids)
        rows = [row for row in rows if row["id"] in selected]
        if {row["id"] for row in rows} != selected:
            raise RuntimeError(f"manifest does not contain every requested candidate: {args.ids}")
    record(f"SELECTED candidates: {[row['id'] for row in rows]}")

    round4_rows = [row for row in rows if row["id"] in {"H79", "H80"}]
    if round4_rows:
        preflight_log = RESULTS / "h79_h80_bs4acc2_preflight.log"
        run(
            [
                str(PY), "-m", "unittest",
                str(EXP / "tests/test_bsa_attention.py"),
                str(EXP / "tests/test_h9_load_audit.py"),
            ],
            preflight_log,
            "H79-H80 formula/load health tests",
        )
        command = [str(PY), str(EXP / "entrypoints/verify_round4_assignment_chain.py")]
        for row in round4_rows:
            command.extend(["--config", row["config"]])
        command.extend([
            "--checkpoint", str(TTX),
            "--output", str(
                REPO / "neuron_autoresearch/experiments/h79_h80_round4_assignment/"
                "load_chain_audit_bs4acc2.json"
            ),
        ])
        run(command, preflight_log, "H79-H80 frozen-TTX warm-start preflight")
    for row in rows:
        candidate = row["id"]
        config = Path(row["config"])
        run_dir = RESULTS / f"{row['name']}_20260720_setsid"
        run_dir.mkdir(parents=True, exist_ok=True)
        final = run_dir / "checkpoint_epoch29.pth"
        ranking = run_dir / "profile_ranking_valid825.md"
        train_log = run_dir / "train.log"

        if not final.exists():
            run(
                [
                    str(PY), "-u", str(EXP / "entrypoints/train.py"),
                    "--config", str(config), "--prev_runid", str(TTX),
                    "--save_path", str(run_dir / "checkpoint_epoch{}.pth"),
                ],
                train_log,
                f"{candidate} batch4-acc2 full30",
            )
        audit_warmstart(train_log, row)
        strict_audit(row, config, final, run_dir)

        if not ranking.exists():
            command = [
                str(PY), "-u", str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
                "--config", str(config), "--run-dir", str(run_dir),
            ]
            for epoch in EPOCHS:
                command.extend(["--epoch", str(epoch)])
            run(command, run_dir / "valid825_queue.log", f"{candidate} standard valid825")
        append_result(row, config, run_dir)
        run(
            [str(PY), str(EXP / "entrypoints/prune_ranked_checkpoints.py"), str(run_dir)],
            run_dir / "checkpoint_prune.log",
            f"{candidate} ranked checkpoint pruning",
        )
        record(f"COMPLETE {candidate}: {ranking}")
    record(f"ALL COMPLETE selected batch4-acc2 queue: {[row['id'] for row in rows]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
