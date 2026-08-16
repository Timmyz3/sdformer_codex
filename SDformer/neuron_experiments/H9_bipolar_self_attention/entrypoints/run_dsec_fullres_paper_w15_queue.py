"""Run the paper-protocol DSEC full-resolution NB0/H67/H66d queue.

Each candidate uses 480x640, window 2x15x15, 30 fine-tuning epochs, physical
batch 1 or 2 with no accumulation, bicubic relative-position remapping, and
evaluation with BatchNorm running statistics disabled.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
PY = Path(sys.executable)
MANIFEST = GEN / "dsec_fullres_paper_w15_manifest.json"
STATUS = RESULTS / "dsec_fullres_paper_w15_queue_status.log"
RUN_TAG = "20260728"
EVAL_EPOCHS = [0, 4, 9, 14, 19, 24, 29]


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


def run(command: list[str], log: Path, label: str, *, check: bool = True) -> int:
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
    if check and proc.returncode:
        raise RuntimeError(f"{label} failed; log={log}")
    return int(proc.returncode)


def generate(batch_size: int, *, smoke: bool) -> Path:
    command = [
        str(PY),
        str(EXP / "entrypoints/make_dsec_fullres_paper_w15_configs.py"),
        "--batch-size",
        str(batch_size),
    ]
    if smoke:
        command.append("--smoke")
    run(command, STATUS, f"generate paper-w15 configs batch{batch_size} smoke={smoke}")
    return GEN / (
        "dsec_fullres_paper_w15_smoke_manifest.json"
        if smoke
        else "dsec_fullres_paper_w15_manifest.json"
    )


def selected_rows(manifest: Path, ids: list[str] | None) -> list[dict[str, Any]]:
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if ids:
        wanted = set(ids)
        rows = [row for row in rows if row["id"] in wanted]
    return rows


def assert_smoke_audit(row: dict[str, Any], log: Path) -> None:
    text = log.read_text(encoding="utf-8", errors="ignore")
    required = [
        r"remap=v1 interpolation complete; applying interpolated state dict",
        rf"checkpoint_overlay_keys={row['expected_overlay']}",
    ]
    if row["expected_atlif"]:
        required.append(r"installed ATLIFTernaryPSN before load: 105 modules")
    if row["expected_attention"]:
        required.append(r"installed attention before load: 12 modules")
    missing = [pattern for pattern in required if not re.search(pattern, text)]
    if missing:
        raise RuntimeError(f"{row['id']} smoke load audit failed {missing}; log={log}")
    if re.search(r"out of memory|CUDNN_STATUS_NOT_SUPPORTED", text, re.I):
        raise RuntimeError(f"{row['id']} smoke hit a memory/backend error; log={log}")


def smoke(rows: list[dict[str, Any]], batch_size: int) -> None:
    root = RESULTS / f"dsec_fullres_paper_w15_smoke_bs{batch_size}_{RUN_TAG}"
    for row in rows:
        log = root / row["id"] / "train.log"
        command = [
            str(PY),
            "-u",
            str(EXP / "entrypoints/train.py"),
            "--config",
            row["config"],
            "--prev_runid",
            row["checkpoint"],
            "--save_path",
            str(root / row["id"] / "checkpoint_epoch{}.pth"),
            "--finetune",
            "1",
        ]
        run(command, log, f"{row['id']} paper-w15 two-step smoke")
        assert_smoke_audit(row, log)
        record(f"SMOKE PASS {row['id']} paper-w15 batch{batch_size}")


def run_chain_audit(manifest: Path) -> None:
    run(
        [
            str(PY),
            str(EXP / "entrypoints/verify_dsec_fullres_window9_chain.py"),
            "--manifest",
            str(manifest),
            "--expected-window",
            "15",
            "--output",
            str(
                REPO
                / "neuron_autoresearch/experiments/dsec_fullres_paper_w15/load_chain_audit.json"
            ),
        ],
        RESULTS / "dsec_fullres_paper_w15_load_chain_audit.log",
        "paper-w15 strict load-chain audit",
    )


def parse_best_epoch(ranking: Path) -> int:
    for line in ranking.read_text(encoding="utf-8").splitlines():
        parts = [part.strip() for part in line.strip("|").split("|")]
        if len(parts) < 3:
            continue
        try:
            if int(parts[0]) == 1:
                return int(parts[1])
        except ValueError:
            continue
    raise RuntimeError(f"rank1 row missing from {ranking}")


def verify_eval_protocol(run_dir: Path, epochs: list[int]) -> None:
    for epoch in epochs:
        profile_path = run_dir / "standard_valid825" / f"epoch{epoch}" / "spike_profile.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        protocol = profile.get("eval_protocol", {})
        expected = {
            "resolution": [480, 640],
            "crop": None,
            "window_size": [2, 15, 15],
            "remap": "v1",
            "bn_policy": "no_running",
            "eval_batch_size": 1,
        }
        for key, value in expected.items():
            if key == "eval_batch_size" and key not in protocol:
                # Profiles produced before 2026-07-30 did not serialize this
                # field, although the evaluator already forced batch size 1.
                continue
            if protocol.get(key) != value:
                raise RuntimeError(
                    f"epoch{epoch} eval protocol mismatch for {key}: "
                    f"{protocol.get(key)!r} != {value!r}"
                )


def formal_eval(row: dict[str, Any], run_dir: Path) -> int:
    ranking = run_dir / "profile_ranking_valid825.md"
    summary = run_dir / "paper_w15_formal_eval_summary.json"
    epochs = [
        epoch
        for epoch in EVAL_EPOCHS
        if (run_dir / f"checkpoint_epoch{epoch}.pth").is_file()
    ]
    if summary.is_file() and ranking.is_file():
        previous = json.loads(summary.read_text(encoding="utf-8"))
        if previous.get("status") == "complete" and previous.get("epochs") == epochs:
            verify_eval_protocol(run_dir, epochs)
            return int(previous["best_epoch"])

    command = [
        str(PY),
        "-u",
        str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
        "--config",
        row["config"],
        "--run-dir",
        str(run_dir),
        "--ranking-mode",
        "aee",
    ]
    for epoch in epochs:
        command.extend(["--epoch", str(epoch)])
    run(command, run_dir / "paper_w15_formal_valid825.log", f"{row['id']} paper-w15 valid825")
    verify_eval_protocol(run_dir, epochs)
    best_epoch = parse_best_epoch(ranking)
    summary.write_text(
        json.dumps(
            {
                "status": "complete",
                "model_id": row["id"],
                "protocol": row["protocol"],
                "source_crop_epochs": row["source_crop_epochs"],
                "epochs": epochs,
                "best_epoch": best_epoch,
                "ranking": str(ranking),
                "finished_at": datetime.now().isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return best_epoch


def latest_resumable_checkpoint(run_dir: Path) -> Path | None:
    """Return the newest checkpoint that has a matching optimizer state."""
    candidates: list[tuple[int, Path]] = []
    for state_path in run_dir.glob("checkpoint_epoch*_state_dict.pth"):
        match = re.fullmatch(r"checkpoint_epoch(\d+)_state_dict\.pth", state_path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        checkpoint = run_dir / f"checkpoint_epoch{epoch}.pth"
        if checkpoint.is_file():
            candidates.append((epoch, checkpoint))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def prune_after_eval(run_dir: Path, best_epoch: int) -> int:
    """Keep curve start, AEE best, final model, and final resumable state."""
    keep_models = {0, best_epoch, 29}
    reclaimed = 0
    for path in run_dir.glob("checkpoint_epoch*.pth"):
        match = re.fullmatch(r"checkpoint_epoch(\d+)(_state_dict)?\.pth", path.name)
        if not match:
            continue
        epoch = int(match.group(1))
        is_state = bool(match.group(2))
        keep = epoch in keep_models if not is_state else epoch == 29
        if keep:
            continue
        reclaimed += path.stat().st_size
        path.unlink()
    record(
        f"PRUNE {run_dir.name}: kept model epochs={sorted(keep_models)}, "
        f"kept state epoch=29, reclaimed={reclaimed / 2**30:.3f}GiB"
    )
    return reclaimed


def append_protocol(rows: list[dict[str, Any]], batch_size: int) -> None:
    marker = "DSEC_FULLRES_PAPER_W15_QUEUE_20260728"
    text = REDESIGN.read_text(encoding="utf-8")
    if marker in text:
        return
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### DSEC 论文全分辨率 window15 重跑协议（2026-07-28）\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(
            "- 论文公开协议：crop 阶段 `2x9x9`；full-resolution 阶段 "
            "`480x640`、`crop=null`、`2x15x15`、额外 30 epochs、physical "
            "batch 1 或 2、相对位置偏置 bicubic remap；测试关闭 BN running-state，"
            "并固定 released validation batch size `1`。\n"
        )
        handle.write(
            f"- 本队列统一使用 physical batch `{batch_size}`、`num_acc=1`、"
            "AMP、CuPy、workers=8、MLflow off；formal valid825 按 AEE 排序。\n"
        )
        handle.write(
            "- 重要限制：本地没有论文 80-epoch crop checkpoint。NB0/H67/H66d "
            "起点分别只有 60/20/30 crop epochs，因此可称为 paper full-resolution "
            "protocol，不可称为论文 checkpoint 的逐点复现。\n"
        )
        handle.write(
            "- 旧 `480x640/window9` 结果保留为协议失败审计，不再用于论文对比；"
            "其 checkpoint/state 已删除，日志、ranking、profile 保留，回收 9.814 GiB。\n"
        )
        handle.write(f"- status：`{STATUS.relative_to(REPO)}`。\n")
        for row in rows:
            handle.write(
                f"- {row['id']}：config `{Path(row['config']).relative_to(REPO)}`；"
                f"start `{Path(row['checkpoint']).relative_to(REPO)}`；"
                f"source crop epochs `{row['source_crop_epochs']}`。\n"
            )


def append_result(row: dict[str, Any], run_dir: Path, best_epoch: int) -> None:
    marker = f"DSEC_FULLRES_PAPER_W15_RESULT::{row['id']}::{run_dir.name}"
    text = REDESIGN.read_text(encoding="utf-8")
    if marker in text:
        return
    ranking = run_dir / "profile_ranking_valid825.md"
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n### DSEC paper-window15 valid825：{row['id']}\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(
            f"- best epoch：`{best_epoch}`；run：`{run_dir.relative_to(REPO)}`；"
            f"source crop epochs：`{row['source_crop_epochs']}`。\n"
        )
        handle.write(
            "- protocol：`480x640 / 2x15x15 / remap=v1 / BN=no_running / "
            "eval_batch=1 / standard valid825 / ranking=AEE`。\n\n"
        )
        handle.write(ranking.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--ids", nargs="+", choices=("NB0", "H67", "H66d"))
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--no-prune", action="store_true")
    parser.add_argument("--skip-deploy", action="store_true")
    args = parser.parse_args()

    formal_manifest = generate(args.batch_size, smoke=False)
    rows = selected_rows(formal_manifest, args.ids)
    append_protocol(rows, args.batch_size)
    run_chain_audit(formal_manifest)

    if not args.skip_smoke:
        smoke_manifest = generate(args.batch_size, smoke=True)
        smoke(selected_rows(smoke_manifest, args.ids), args.batch_size)
    if args.smoke_only:
        record("SMOKE-ONLY COMPLETE")
        return 0

    for row in rows:
        run_dir = RESULTS / f"{row['name']}_bs{args.batch_size}_{RUN_TAG}"
        final = run_dir / "checkpoint_epoch29.pth"
        if not final.is_file():
            resume_checkpoint = latest_resumable_checkpoint(run_dir)
            command = [
                str(PY),
                "-u",
                str(EXP / "entrypoints/train.py"),
                "--config",
                row["config"],
                "--prev_runid",
                str(resume_checkpoint or row["checkpoint"]),
                "--save_path",
                str(run_dir / "checkpoint_epoch{}.pth"),
                "--finetune",
                "1",
            ]
            if resume_checkpoint is not None:
                command.extend(["--resume", "1"])
                record(f"RESUME {row['id']} from {resume_checkpoint.name}")
            run(
                command,
                run_dir / "train.log",
                f"{row['id']} paper-w15 FT30",
            )
        best_epoch = formal_eval(row, run_dir)
        append_result(row, run_dir, best_epoch)
        if not args.no_prune:
            prune_after_eval(run_dir, best_epoch)
        record(f"COMPLETE {row['id']} paper-w15 best_epoch={best_epoch}")

    deploy_ids = [row["id"] for row in rows if row["id"] in {"H67", "H66d"}]
    if deploy_ids and not args.skip_deploy:
        run(
            [
                str(PY),
                "-u",
                str(EXP / "entrypoints/run_dsec_fullres_paper_w15_deploy_followup.py"),
                "--batch-size",
                str(args.batch_size),
                "--ids",
                *deploy_ids,
            ],
            RESULTS / "dsec_fullres_paper_w15_deploy_followup.log",
            f"paper-w15 deploy followup ids={deploy_ids}",
        )

    record("ALL COMPLETE DSEC PAPER-W15 QUEUE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
