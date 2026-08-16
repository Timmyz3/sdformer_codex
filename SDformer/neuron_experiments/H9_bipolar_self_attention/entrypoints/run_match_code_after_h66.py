"""Run H73 DE9, H74 MC49, and H75 AX17 after H66 and H81 control."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
PY = Path(sys.executable)
TTX = RESULTS / "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth"
PREV_STATUS = RESULTS / "h81_equal_budget_after_h66_status.log"
STATUS = RESULTS / "match_code_after_h66_status.log"
MANIFEST = GEN / "h73_h74_match_code_full30_manifest.json"
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
        proc = subprocess.run(command, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    record(f"END {label}: exit_code={proc.returncode}")
    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed; log={log}")


def wait_previous() -> None:
    while True:
        if PREV_STATUS.exists() and "ALL COMPLETE H81 EQUAL-BUDGET + AAE AUDIT:" in PREV_STATUS.read_text(encoding="utf-8", errors="ignore"):
            return
        record(f"WAIT H81 equal-budget control and AAE audit: {PREV_STATUS}")
        time.sleep(600)


def audit_warmstart(log: Path) -> None:
    text = log.read_text(encoding="utf-8", errors="ignore")
    required = {
        "ATLIFTernaryPSN preload count": r"installed ATLIFTernaryPSN before load: 105 modules",
        "attention preload count": r"installed attention before load: 12 modules",
        "new codebook count": r"initialized new Match-Code weights: 12",
        "clean TTX warm-start audit": r"load audit: checkpoint_overlay_keys=210, missing=12, unexpected=0",
    }
    missing = [label for label, pattern in required.items() if re.search(pattern, text) is None]
    if missing:
        raise RuntimeError(f"Match-Code warm-start audit failed ({', '.join(missing)}): {log}")
    record(f"PASS warm-start audit: ATLIF=105 attention=12 new_codebooks=12 overlay=210 missing=12 unexpected=0 ({log})")


def append_result(name: str, config: Path, run_dir: Path) -> None:
    marker = f"MATCH_CODE_FULL30::{name}"
    if marker in REDESIGN.read_text(encoding="utf-8"):
        return
    ranking = run_dir / "profile_ranking_valid825.md"
    table = [line for line in ranking.read_text(encoding="utf-8").splitlines() if line.startswith("| ")]
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n### Match-Code full30 自动结果：{name}\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(f"- config: `{config.relative_to(REPO)}`\n")
        handle.write(f"- start checkpoint: `{TTX.relative_to(REPO)}`\n")
        handle.write(f"- run dir: `{run_dir.relative_to(REPO)}`\n")
        handle.write("- audited chain: ATLIF105, attention12, new codebooks12, TTX overlay210, missing12, unexpected0.\n\n")
        for line in table:
            handle.write(line + "\n")


def main() -> int:
    wait_previous()
    run([str(PY), str(EXP / "entrypoints/make_h73_h74_match_code_configs.py")], STATUS, "generate Match-Code full30 configs")
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for row in rows:
        name = row["name"]
        config = Path(row["config"])
        run_dir = RESULTS / f"{name}_bs8_full30_20260712_setsid"
        run_dir.mkdir(parents=True, exist_ok=True)
        final = run_dir / "checkpoint_epoch29.pth"
        ranking = run_dir / "profile_ranking_valid825.md"
        train_log = run_dir / "train.log"
        if not final.exists():
            run([
                str(PY), "-u", str(EXP / "entrypoints/train.py"),
                "--config", str(config), "--prev_runid", str(TTX),
                "--save_path", str(run_dir / "checkpoint_epoch{}.pth"),
            ], train_log, f"{name} train full30")
        audit_warmstart(train_log)
        if not ranking.exists():
            command = [
                str(PY), "-u", str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
                "--config", str(config), "--run-dir", str(run_dir),
            ]
            for epoch in EPOCHS:
                command.extend(["--epoch", str(epoch)])
            run(command, run_dir / "valid825_queue.log", f"{name} valid825")
        append_result(name, config, run_dir)
        record(f"COMPLETE {name}: {ranking}")
    record(f"ALL COMPLETE MATCH-CODE: {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
