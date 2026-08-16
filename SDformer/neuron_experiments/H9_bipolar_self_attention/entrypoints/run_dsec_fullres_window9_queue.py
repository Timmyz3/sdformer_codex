"""Run audited NB0, H67, and H66d 480x640/window9 DSEC fine-tuning.

After each model finishes FT30, runs formal multi-checkpoint valid825 using the
same force_save epoch set as crop full30 (0/4/9/14/19/24/28/29). See
`run_dsec_fullres_window9_formal_eval.py` for the selection policy.
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


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
PY = Path(sys.executable)
MANIFEST = GEN / "dsec_fullres_window9_manifest.json"
STATUS = RESULTS / "dsec_fullres_window9_queue_status.log"
RUN_TAG = "20260726"
# Keep in sync with make_dsec_fullres_window9_configs.SAVE_EPOCHS and formal eval.
FORMAL_EVAL_EPOCHS = [0, 4, 9, 14, 19, 24, 28, 29]


def record(message: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
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
        str(EXP / "entrypoints/make_dsec_fullres_window9_configs.py"),
        "--batch-size",
        str(batch_size),
        "--effective-batch",
        "8",
    ]
    if smoke:
        command.append("--smoke")
    run(command, STATUS, f"generate fullres configs batch{batch_size} smoke={smoke}")
    return (
        GEN / "dsec_fullres_window9_smoke_manifest.json"
        if smoke
        else MANIFEST
    )


def smoke_batch(batch_size: int) -> bool:
    manifest = generate(batch_size, smoke=True)
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    smoke_root = RESULTS / f"dsec_fullres_window9_smoke_bs{batch_size}_{RUN_TAG}"
    for row in rows:
        log = smoke_root / row["id"] / "train.log"
        command = [
            str(PY),
            "-u",
            str(EXP / "entrypoints/train.py"),
            "--config",
            row["config"],
            "--prev_runid",
            row["checkpoint"],
            "--save_path",
            str(smoke_root / row["id"] / "checkpoint_epoch{}.pth"),
            "--finetune",
            "1",
        ]
        exit_code = run(
            command,
            log,
            f"{row['id']} fullres smoke batch{batch_size}",
            check=False,
        )
        text = log.read_text(encoding="utf-8", errors="ignore")
        if exit_code or re.search(r"out of memory|CUDNN_STATUS_NOT_SUPPORTED", text, re.I):
            record(f"SMOKE REJECT batch{batch_size} at {row['id']}; log={log}")
            return False
        if "remap=v1 interpolation complete; applying interpolated state dict" not in text:
            raise RuntimeError(f"{row['id']} smoke did not use audited v1 load: {log}")
        if row["expected_atlif"] and not re.search(
            r"installed ATLIFTernaryPSN before load: 105 modules", text
        ):
            raise RuntimeError(f"{row['id']} ATLIF count audit failed: {log}")
        if row["expected_attention"] and not re.search(
            r"installed attention before load: 12 modules", text
        ):
            raise RuntimeError(f"{row['id']} attention count audit failed: {log}")
        record(f"SMOKE PASS {row['id']} batch{batch_size}")
    return True


def append_launch(rows: list[dict], batch_size: int) -> None:
    marker = "DSEC_FULLRES_WINDOW9_QUEUE_20260726"
    text = REDESIGN.read_text(encoding="utf-8")
    if marker in text:
        return
    with REDESIGN.open("a", encoding="utf-8") as handle:
        handle.write("\n\n### DSEC 480x640/window9 三模型全分辨率队列（2026-07-26）\n\n")
        handle.write(f"<!-- {marker} -->\n")
        handle.write(
            "- 顺序：NB0 ep59 -> H67 Motion-XOR ep19 -> H66d Local-5 ep29；"
            "三者均为 30 epoch full-resolution fine-tune。\n"
        )
        handle.write(
            f"- geometry：`480x640`、`crop=null`、`window=[2,9,9]`；"
            f"physical batch `{batch_size}`、effective batch `8`。\n"
        )
        handle.write(
            "- 加载：`--finetune 1` 触发 audited `remap=v1`；插值后必须执行 "
            "`load_state_dict`，并核对 ATLIF/attention/overlay/missing/unexpected。\n"
        )
        handle.write(
            "- 定位：这是保持 N=162 硬件 tile 的 full-resolution 对照，"
            "不是论文 `[2,15,15]` protocol；window15 只在最终 winner 冻结后补跑。\n"
        )
        handle.write(
            f"- 正式推理选点：每个模型 FT30 结束后立刻 valid825，"
            f"默认 epoch=`{FORMAL_EVAL_EPOCHS}`（= force_save 全集，与 H67/H68 crop full30 一致）；"
            "缺 ckpt 跳过；train-val best 映射 epoch 会自动补进集合。\n"
        )
        handle.write(f"- status：`{STATUS.relative_to(REPO)}`。\n")
        for row in rows:
            handle.write(
                f"- {row['id']} config：`{Path(row['config']).relative_to(REPO)}`；"
                f"start：`{Path(row['checkpoint']).relative_to(REPO)}`。\n"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=None)
    parser.add_argument("--ids", nargs="+", choices=("NB0", "H67", "H66d"))
    args = parser.parse_args()

    run(
        [
            str(PY),
            "-m",
            "unittest",
            "neuron_experiments/H9_bipolar_self_attention/tests/test_h9_load_audit.py",
        ],
        RESULTS / "dsec_fullres_window9_preflight_tests.log",
        "fullres load regression tests",
    )

    if args.batch_size is not None:
        batch_size = args.batch_size
        if not args.skip_smoke and not smoke_batch(batch_size):
            raise RuntimeError(f"requested batch{batch_size} failed smoke")
    elif args.skip_smoke:
        batch_size = 1
    else:
        batch_size = 2 if smoke_batch(2) else 1
        if batch_size == 1 and not smoke_batch(1):
            raise RuntimeError("both batch2 and batch1 failed full-resolution smoke")

    manifest = generate(batch_size, smoke=False)
    run(
        [
            str(PY),
            str(EXP / "entrypoints/verify_dsec_fullres_window9_chain.py"),
            "--manifest",
            str(manifest),
        ],
        RESULTS / "dsec_fullres_window9_load_chain_audit.log",
        "fullres strict load-chain audit",
    )
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if args.ids:
        selected = set(args.ids)
        rows = [row for row in rows if row["id"] in selected]
    append_launch(rows, batch_size)
    record(
        f"FORMAL START batch{batch_size}, accumulation={8 // batch_size}, "
        f"selected={[row['id'] for row in rows]}"
    )

    formal_eval = EXP / "entrypoints/run_dsec_fullres_window9_formal_eval.py"
    for row in rows:
        run_dir = RESULTS / f"{row['name']}_bs{batch_size}_{RUN_TAG}"
        final = run_dir / "checkpoint_epoch29.pth"
        ranking = run_dir / "profile_ranking_valid825.md"
        if not final.exists():
            command = [
                str(PY),
                "-u",
                str(EXP / "entrypoints/train.py"),
                "--config",
                row["config"],
                "--prev_runid",
                row["checkpoint"],
                "--save_path",
                str(run_dir / "checkpoint_epoch{}.pth"),
                "--finetune",
                "1",
            ]
            run(command, run_dir / "train.log", f"{row['id']} fullres formal30")
            record(f"COMPLETE {row['id']} fullres formal30: {final}")
        else:
            record(f"REUSE completed {row['id']} fullres train: {final}")

        # Formal multi-ckpt inference immediately after each model finishes FT30.
        if ranking.exists() and (run_dir / "fullres_formal_eval_summary.json").exists():
            record(f"REUSE completed {row['id']} fullres formal valid825: {ranking}")
            continue
        eval_cmd = [
            str(PY),
            "-u",
            str(formal_eval),
            "--ids",
            row["id"],
            "--batch-size",
            str(batch_size),
            # Safe if an older sibling model is still training on a concurrent queue.
            "--wait-gpu",
            "--max-used-mib",
            "8192",
            "--poll-seconds",
            "120",
        ]
        for epoch in FORMAL_EVAL_EPOCHS:
            eval_cmd.extend(["--epoch", str(epoch)])
        run(
            eval_cmd,
            run_dir / "fullres_formal_valid825_queue.log",
            f"{row['id']} fullres formal valid825 after train",
        )
        record(f"COMPLETE {row['id']} fullres formal valid825: {ranking}")

    record("ALL COMPLETE DSEC FULLRES WINDOW9 QUEUE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
