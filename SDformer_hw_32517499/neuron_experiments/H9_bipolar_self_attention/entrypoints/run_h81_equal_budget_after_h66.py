"""Run the reviewer-mandated H60/no-motion control after the H66 queue."""

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
AAE_REPORT = REPO / "neuron_autoresearch/AAE_BASELINE_DIAGNOSTIC_20260717.md"
PY = Path(sys.executable)
PREV_STATUS = RESULTS / "h66_full30_after_h71_status.log"
STATUS = RESULTS / "h81_equal_budget_after_h66_status.log"
TTX = RESULTS / "date11full_all_binary_atlif_h60_mu0_txonly_slowlr_cont_ep2_ft8_bs8_20260629_154937_setsid/checkpoint_epoch2.pth"
H67_RUN = RESULTS / "h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_full30_20260711_setsid"
H81_NAME = "h81_allbinary_all12_h60_nomotion_equalbudget_w720_fastlr_full30"
H81_RUN = RESULTS / f"{H81_NAME}_bs8_full30_20260717_setsid"
NB0_CHECKPOINT = REPO / "experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
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


def wait_h66() -> None:
    while True:
        if PREV_STATUS.exists() and "ALL COMPLETE H66 FULL30:" in PREV_STATUS.read_text(encoding="utf-8", errors="ignore"):
            return
        record(f"WAIT H66 full30 queue: {PREV_STATUS}")
        time.sleep(600)


def audit_warmstart(log: Path) -> None:
    text = log.read_text(encoding="utf-8", errors="ignore")
    required = (
        "installed ATLIFTernaryPSN before load: 105 modules",
        "installed attention before load: 12 modules",
        "load audit: checkpoint_overlay_keys=210, missing=0, unexpected=0",
    )
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"H81 warm-start audit failed: {missing}; log={log}")
    record("PASS H81 warm-start audit: ATLIF=105 attention=12 overlay=210 missing=0 unexpected=0")


def best_epoch(ranking: Path) -> int:
    for line in ranking.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\|\s*1\s*\|\s*(\d+)\s*\|", line)
        if match:
            return int(match.group(1))
    raise RuntimeError(f"cannot parse best epoch from {ranking}")


def eval_benchmark(config: Path, checkpoint: Path, out_dir: Path, label: str) -> dict:
    profile = out_dir / "spike_profile.json"
    if not profile.exists():
        run([
            str(PY), "-u", "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
            "--config", str(config), "--checkpoint", str(checkpoint),
            "--path_results", str(out_dir), "--mode", "valid",
        ], out_dir / "eval.log", label)
    return json.loads(profile.read_text(encoding="utf-8"))


def write_aae_report(rows: list[dict]) -> None:
    lines = [
        "# AAE Baseline Diagnostic (2026-07-17)",
        "",
        "## Verified Definition Mismatch",
        "",
        "- Legacy local `AAE` is the 2-D direction angle between `(u,v)` vectors.",
        "- DSEC benchmark `AE` follows Barron and uses normalized space-time vectors `(u,v,1)`.",
        "- The paper's `4.871` is official DSEC test AE; local valid825 is a center-cropped training split.",
        "- Therefore legacy valid825 `AAE` and paper/test `AE` are not directly comparable.",
        "",
        "## Same-Checkpoint Valid825 Audit",
        "",
        "| model | epoch | AEE | legacy AAE-2D | DSEC/Barron AE-3D |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        metrics = row["profile"]["metrics"]
        lines.append(
            f"| {row['model']} | {row['epoch']} | {float(metrics['AEE']):.4f} | "
            f"{float(metrics['AAE']):.4f} | {float(metrics['AAE_Benchmark']):.4f} |"
        )
    lines.extend([
        "",
        "## Reporting Rule",
        "",
        "Use legacy AAE-2D only to compare historical local runs. Report DSEC/Barron AE-3D for benchmark-facing tables, and label valid825 separately from official test.",
    ])
    AAE_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_redesign(rows: list[dict], h81_epoch: int) -> None:
    marker = "H81_EQUAL_BUDGET_AAE_AUDIT_20260717"
    if marker in REDESIGN.read_text(encoding="utf-8"):
        return
    ranking = (H81_RUN / "profile_ranking_valid825.md").read_text(encoding="utf-8")
    table = [line for line in ranking.splitlines() if line.startswith("| ")]
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### H81 等预算 no-motion 控制与 AAE 口径审计\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write("H81 与 H67 的训练预算、起点、all12 H60 和 all-binary ATLIF 完全一致，只关闭 Motion-XOR。\n\n")
        for line in table:
            handle.write(line + "\n")
        handle.write("\n同 checkpoint 双口径 AAE：\n\n")
        handle.write("| model | epoch | AEE | legacy AAE-2D | DSEC/Barron AE-3D |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for row in rows:
            metrics = row["profile"]["metrics"]
            handle.write(
                f"| {row['model']} | {row['epoch']} | {float(metrics['AEE']):.4f} | "
                f"{float(metrics['AAE']):.4f} | {float(metrics['AAE_Benchmark']):.4f} |\n"
            )
        handle.write(f"\nH81 best epoch: `{h81_epoch}`。论文 `4.871` 来自 official test，不再标注为 valid825。\n")


def main() -> int:
    wait_h66()
    run([str(PY), str(EXP / "entrypoints/make_h81_equal_budget_control.py")], STATUS, "generate H81 and AAE audit configs")

    H81_RUN.mkdir(parents=True, exist_ok=True)
    config = GEN / f"{H81_NAME}.yml"
    train_log = H81_RUN / "train.log"
    if not (H81_RUN / "checkpoint_epoch29.pth").exists():
        run([
            str(PY), "-u", str(EXP / "entrypoints/train.py"),
            "--config", str(config), "--prev_runid", str(TTX),
            "--save_path", str(H81_RUN / "checkpoint_epoch{}.pth"),
        ], train_log, "H81 equal-budget no-motion full30")
    audit_warmstart(train_log)

    ranking = H81_RUN / "profile_ranking_valid825.md"
    if not ranking.exists():
        command = [
            str(PY), "-u", str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
            "--config", str(config), "--run-dir", str(H81_RUN),
        ]
        for epoch in EPOCHS:
            command.extend(["--epoch", str(epoch)])
        run(command, H81_RUN / "valid825_queue.log", "H81 standard valid825")

    h81_epoch = best_epoch(ranking)
    audit_rows = [
        {
            "model": "NB0",
            "epoch": 59,
            "profile": eval_benchmark(
                GEN / "nb0_benchmark_aae_valid825.yml", NB0_CHECKPOINT,
                REPO / "results_inference/nb0_epoch59_benchmark_aae_valid825_20260717",
                "NB0 benchmark-AAE valid825",
            ),
        },
        {
            "model": "H67 Motion-XOR",
            "epoch": 19,
            "profile": eval_benchmark(
                GEN / "h67_motionxor_benchmark_aae_valid825.yml", H67_RUN / "checkpoint_epoch19.pth",
                H67_RUN / "benchmark_aae_valid825/epoch19", "H67 benchmark-AAE valid825",
            ),
        },
        {
            "model": "H81 no-motion",
            "epoch": h81_epoch,
            "profile": eval_benchmark(
                GEN / "h81_nomotion_benchmark_aae_valid825.yml", H81_RUN / f"checkpoint_epoch{h81_epoch}.pth",
                H81_RUN / f"benchmark_aae_valid825/epoch{h81_epoch}", "H81 benchmark-AAE valid825",
            ),
        },
    ]
    write_aae_report(audit_rows)
    append_redesign(audit_rows, h81_epoch)
    record(f"ALL COMPLETE H81 EQUAL-BUDGET + AAE AUDIT: {AAE_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
