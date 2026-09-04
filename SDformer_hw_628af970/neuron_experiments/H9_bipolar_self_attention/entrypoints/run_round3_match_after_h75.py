"""Wait for H73-H75, then run H76, H77, and H78 full30 plus valid825."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
PY = Path(os.environ.get("ROUND3_MATCH_PYTHON", "/opt/conda/envs/sdformerflow/bin/python"))
TTX = RESULTS / "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth"
PREV_STATUS = RESULTS / "match_code_after_h66_status.log"
STATUS = RESULTS / "round3_match_after_h75_status.log"
MANIFEST = GEN / "h76_h78_round3_match_full30_manifest.json"
AUDIT = REPO / "neuron_autoresearch/experiments/h76_h78_round3_match/load_chain_audit.json"
EPOCHS = (0, 4, 9, 14, 19, 24, 28, 29)


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
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


def wait_h75() -> None:
    while True:
        if PREV_STATUS.exists():
            text = PREV_STATUS.read_text(encoding="utf-8", errors="ignore")
            if "ALL COMPLETE MATCH-CODE:" in text:
                return
        record(f"WAIT H73-H75 completion marker: {PREV_STATUS}")
        time.sleep(600)


def audit_warmstart(log: Path, row: dict) -> None:
    text = log.read_text(encoding="utf-8", errors="ignore")
    expected_missing = int(row["expected_new_keys"])
    required = {
        "ATLIFTernaryPSN preload count": r"installed ATLIFTernaryPSN before load: 105 modules",
        "attention preload count": r"installed attention before load: 12 modules",
        "new codebook count": r"initialized new Match-Code weights: 12",
        "clean TTX warm-start audit": (
            rf"load audit: checkpoint_overlay_keys=210, missing={expected_missing}, unexpected=0"
        ),
    }
    if row["id"] == "H77":
        required["new LC4 coefficient count"] = r"initialized new Round3 auxiliary parameters: 12"
    missing = [label for label, pattern in required.items() if re.search(pattern, text) is None]
    if missing:
        raise RuntimeError(f"Round3 warm-start audit failed ({', '.join(missing)}): {log}")
    record(
        f"PASS {row['id']} warm-start: ATLIF=105 attention=12 candidate=12 "
        f"overlay=210 missing={expected_missing} unexpected=0 ({log})"
    )


def append_result(row: dict, config: Path, run_dir: Path) -> None:
    marker = f"ROUND3_MATCH_FULL30::{row['id']}::{row['name']}"
    current = REDESIGN.read_text(encoding="utf-8")
    if marker in current:
        return
    ranking = run_dir / "profile_ranking_valid825.md"
    table = [line for line in ranking.read_text(encoding="utf-8").splitlines() if line.startswith("| ")]
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n### Round3 Match-Code full30 自动结果：{row['id']} {row['name']}\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(f"- formula: {row['formula']}\n")
        handle.write(f"- config: `{config.relative_to(REPO)}`\n")
        handle.write(f"- frozen independent start: `{TTX.relative_to(REPO)}`\n")
        handle.write(f"- run dir: `{run_dir.relative_to(REPO)}`\n")
        handle.write(
            "- audited chain: ATLIF105, attention12, candidate12, TTX overlay210, "
            f"warm-start missing{row['expected_new_keys']}/unexpected0, trained strict missing0/unexpected0.\n\n"
        )
        for line in table:
            handle.write(line + "\n")


def main() -> int:
    wait_h75()
    queue_log = RESULTS / "round3_match_generation_and_audit.log"
    run(
        [str(PY), str(EXP / "entrypoints/make_h76_h78_round3_match_configs.py")],
        queue_log,
        "generate H76-H78 full30 configs",
    )
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))

    # This is an implementation health check only. It never ranks or eliminates a candidate.
    run(
        [str(PY), "-m", "unittest", str(EXP / "tests/test_bsa_attention.py")],
        queue_log,
        "Round3 formula health tests (non-selective)",
    )
    warm_command = [str(PY), str(EXP / "entrypoints/verify_round3_match_chain.py")]
    for row in rows:
        warm_command.extend(["--config", row["config"]])
    warm_command.extend(["--checkpoint", str(TTX), "--output", str(AUDIT)])
    run(warm_command, queue_log, "H76-H78 frozen TTX warm-start audit")

    for row in rows:
        name = row["name"]
        config = Path(row["config"])
        run_dir = RESULTS / f"{name}_bs8_full30_20260713_setsid"
        run_dir.mkdir(parents=True, exist_ok=True)
        final = run_dir / "checkpoint_epoch29.pth"
        ranking = run_dir / "profile_ranking_valid825.md"
        train_log = run_dir / "train.log"
        if not final.exists():
            run([
                str(PY), "-u", str(EXP / "entrypoints/train.py"),
                "--config", str(config), "--prev_runid", str(TTX),
                "--save_path", str(run_dir / "checkpoint_epoch{}.pth"),
            ], train_log, f"{row['id']} train full30")
        audit_warmstart(train_log, row)

        strict_audit = run_dir / "trained_strict_load_audit.json"
        run([
            str(PY), str(EXP / "entrypoints/verify_round3_match_chain.py"),
            "--trained", str(config), str(final), "--output", str(strict_audit),
        ], run_dir / "trained_strict_load_audit.log", f"{row['id']} trained strict-load audit")

        if not ranking.exists():
            command = [
                str(PY), "-u", str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
                "--config", str(config), "--run-dir", str(run_dir),
            ]
            for epoch in EPOCHS:
                command.extend(["--epoch", str(epoch)])
            run(command, run_dir / "valid825_queue.log", f"{row['id']} standard valid825")
        append_result(row, config, run_dir)
        record(f"COMPLETE {row['id']} {name}: {ranking}")
    record(f"ALL COMPLETE ROUND3 MATCH-CODE: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
