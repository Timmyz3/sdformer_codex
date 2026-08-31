#!/usr/bin/env python3
"""Queue ATLIF-only MVSEC training, standard evaluation, and paper receipt."""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
ENTRY = EXP / "entrypoints"
RESULTS = EXP / "results"
CONFIG = EXP / "configs/generated/mvsec_cicc_atlif_only_w8_seed0_20260825.yml"
MANIFEST = EXP / "manifests/mvsec_cicc_dt1_v1.json"
NB0_CKPT = RESULTS / "mvsec_cicc_nb0_w8_seed0_v4_20260811/checkpoint_epoch11.pth"
TRAIN = RESULTS / "mvsec_cicc_atlif_only_w8_seed0_20260825"
SMOKE = RESULTS / "mvsec_cicc_atlif_only_w8_seed0_smoke_20260825"
FIXED = RESULTS / "mvsec_cicc_atlif_only_w8_seed0_fixed800_20260825"
FULL = RESULTS / "mvsec_cicc_atlif_only_w8_seed0_full_20260825"
NB0_FIXED = RESULTS / "mvsec_cicc_nb0_w8_seed0_v4_fixed800_20260811/mvsec_summary.json"
NB0_FULL = RESULTS / "mvsec_cicc_nb0_w8_seed0_v4_full_20260811/mvsec_summary.json"
H67_FIXED = RESULTS / "mvsec_cicc_h67_motion_w8_seed0_v4_fixed800_20260811/mvsec_summary.json"
H67_FULL = RESULTS / "mvsec_cicc_h67_motion_w8_seed0_v4_full_20260811/mvsec_summary.json"
OUTPUT = REPO / "neuron_autoresearch/MVSEC_TWO_CONTRIBUTION_ABLATION_20260825.json"
OUTPUT_MD = OUTPUT.with_suffix(".md")
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
STATUS = RESULTS / "mvsec_cicc_atlif_only_queue_20260825.log"
LOCK = Path("/tmp/sdformer_mvsec_atlif_only_ablation_20260825.lock")
PY = sys.executable
EXIT_RE = re.compile(r"\[mvsec-cicc-train\] exit_code=(\d+)")
EPOCH_RE = re.compile(r"^Epoch (\d+)\s*$")
VALID_RE = re.compile(r"Epoch loss \(Validation\): ([0-9.eE+-]+)")
RESULT_MARKER = "<!-- MVSEC_ATLIF_ONLY_ABLATION_RESULT_20260825 -->"


def record(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    record("START " + " ".join(command))
    result = subprocess.run(command, cwd=REPO, env=env)
    record(f"END exit_code={result.returncode}")
    if result.returncode:
        raise RuntimeError(f"command failed: {' '.join(command)}")


def gpu_compute_pids() -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    return [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]


def wait_for_idle_gpu() -> None:
    stable = 0
    while stable < 2:
        pids = gpu_compute_pids()
        if pids:
            stable = 0
            record(f"WAIT GPU compute pids={pids}")
        else:
            stable += 1
            record(f"GPU idle confirmation {stable}/2")
        if stable < 2:
            time.sleep(30)


def completed_train(output_dir: Path) -> int | None:
    log = output_dir / "train.log"
    if not log.is_file():
        return None
    matches = EXIT_RE.findall(log.read_text(encoding="utf-8", errors="replace"))
    return int(matches[-1]) if matches else None


def smoke() -> None:
    receipt = SMOKE / "load_audit.json"
    if receipt.is_file():
        record("SKIP completed smoke")
        return
    code = completed_train(SMOKE)
    if code != 0:
        if code is not None:
            raise RuntimeError(f"previous ATLIF-only smoke failed with exit_code={code}")
        data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        data["loader"]["n_epochs"] = 1
        SMOKE.mkdir(parents=True, exist_ok=True)
        smoke_config = SMOKE / "smoke_config.yml"
        smoke_config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        env = os.environ.copy()
        env["SDFORMER_MDR_MAX_TRAIN_BATCHES"] = "1"
        env["SDFORMER_MDR_MAX_VALID_BATCHES"] = "1"
        run(
            [
                PY,
                "-u",
                str(ENTRY / "run_mvsec_cicc_train.py"),
                "--config",
                str(smoke_config),
                "--output-dir",
                str(SMOKE),
                "--prev-runid",
                str(NB0_CKPT),
            ],
            env=env,
        )
    log_text = (SMOKE / "train.log").read_text(encoding="utf-8", errors="replace")
    required = (
        "installed ATLIFTernaryPSN before load: 105 modules",
        "checkpoint_overlay_keys=0, missing=210, unexpected=0",
    )
    missing = [marker for marker in required if marker not in log_text]
    if missing:
        raise RuntimeError(f"ATLIF-only smoke load audit failed: {missing}")
    if "installed Shiftmax attention" in log_text:
        raise RuntimeError("ATLIF-only smoke unexpectedly installed Shiftmax attention")
    for checkpoint in SMOKE.glob("checkpoint_epoch*.pth"):
        checkpoint.unlink()
    receipt.write_text(
        json.dumps({"status": "PASS", "required": required}, indent=2) + "\n",
        encoding="utf-8",
    )


def train() -> None:
    code = completed_train(TRAIN)
    if code == 0:
        record("SKIP completed ATLIF-only training")
        return
    if code is not None:
        raise RuntimeError(f"previous ATLIF-only training failed with exit_code={code}")
    if (TRAIN / "train.log").is_file():
        raise RuntimeError("incomplete prior train.log exists; refusing to overwrite")
    run(
        [
            PY,
            "-u",
            str(ENTRY / "run_mvsec_cicc_train.py"),
            "--config",
            str(CONFIG),
            "--output-dir",
            str(TRAIN),
            "--prev-runid",
            str(NB0_CKPT),
        ]
    )


def validation_losses() -> dict[int, float]:
    current = None
    losses: dict[int, float] = {}
    for line in (TRAIN / "train.log").read_text(encoding="utf-8", errors="replace").splitlines():
        epoch_match = EPOCH_RE.match(line.strip())
        if epoch_match:
            current = int(epoch_match.group(1))
        valid_match = VALID_RE.search(line)
        if valid_match and current is not None:
            losses[current] = float(valid_match.group(1))
    return losses


def select_best() -> Path:
    losses = validation_losses()
    candidates: list[tuple[float, int, Path]] = []
    for checkpoint in TRAIN.glob("checkpoint_epoch*.pth"):
        if checkpoint.name.endswith("_state_dict.pth"):
            continue
        match = re.fullmatch(r"checkpoint_epoch(\d+)\.pth", checkpoint.name)
        if match and int(match.group(1)) in losses:
            epoch = int(match.group(1))
            candidates.append((losses[epoch], epoch, checkpoint))
    if not candidates:
        raise RuntimeError("no validation-bound ATLIF-only checkpoint")
    loss, epoch, checkpoint = min(candidates)
    receipt = {
        "schema": "mvsec_best_valid_checkpoint_v1",
        "checkpoint": str(checkpoint.resolve()),
        "epoch": epoch,
        "validation_loss": loss,
        "available_validation_losses": losses,
    }
    (TRAIN / "best_checkpoint.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record(f"BEST ep{epoch} validation_loss={loss:.6f}")
    return checkpoint.resolve()


def evaluate(checkpoint: Path, output_dir: Path, fixed800: bool) -> None:
    if (output_dir / "mvsec_summary.json").is_file():
        record(f"SKIP completed eval {output_dir.name}")
        return
    command = [
        PY,
        "-u",
        str(ENTRY / "run_h9_standard_mvsec_eval.py"),
        "--config",
        str(CONFIG),
        "--checkpoint",
        str(checkpoint),
        "--out-dir",
        str(output_dir),
    ]
    if fixed800:
        command.extend(["--fixed800-manifest", str(MANIFEST)])
    run(command)


def route_metrics(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw["sequences"]
    return {
        "summary": str(path.resolve()),
        "mean_aee": raw["mean_aee"],
        "weighted_aee": raw["valid_pixel_weighted_aee"],
        "mean_fl_percent": sum(float(row["gt_fl_percent"]) for row in rows) / len(rows),
        "total_spikes_g": sum(float(row["spikes_g"]) for row in rows),
        "total_energy_uj": sum(float(row["energy_uj"]) for row in rows),
        "per_sequence": rows,
    }


def write_receipt(checkpoint: Path) -> None:
    routes = {
        "nb0": {
            "fixed800": route_metrics(NB0_FIXED),
            "full_sequence": route_metrics(NB0_FULL),
        },
        "atlif_only": {
            "fixed800": route_metrics(FIXED / "mvsec_summary.json"),
            "full_sequence": route_metrics(FULL / "mvsec_summary.json"),
        },
        "atlif_ttx": {
            "fixed800": route_metrics(H67_FIXED),
            "full_sequence": route_metrics(H67_FULL),
        },
    }
    payload = {
        "schema": "mvsec_two_contribution_ablation_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "outdoor_day2_dt1_train; OD1/IF1/IF2/IF3 test; all replacements full-network",
        "config": str(CONFIG.resolve()),
        "parent_checkpoint": str(NB0_CKPT.resolve()),
        "selected_checkpoint": str(checkpoint),
        "routes": routes,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# MVSEC two-contribution full-replacement ablation",
        "",
        "| route | macro AEE | weighted AEE | macro Fl (%) | spikes (G) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("nb0", "atlif_only", "atlif_ttx"):
        row = routes[name]["full_sequence"]
        lines.append(
            f"| {name} | {row['mean_aee']:.6f} | {row['weighted_aee']:.6f} | "
            f"{row['mean_fl_percent']:.4f} | {row['total_spikes_g']:.4f} |"
        )
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if RESULT_MARKER not in REDESIGN.read_text(encoding="utf-8"):
        with REDESIGN.open("a", encoding="utf-8") as handle:
            handle.write("\n" + RESULT_MARKER + "\n\n")
            handle.write("### MVSEC 全量 ATLIF-only 两贡献消融结果（2026-08-25）\n\n")
            handle.write("三条路线均为全网定义，不含部分 stage 或混合注意力。\n\n")
            handle.write("\n".join(lines[2:]) + "\n")
            handle.write(
                "\n权威收据：`neuron_autoresearch/MVSEC_TWO_CONTRIBUTION_ABLATION_20260825.{json,md}`。\n"
            )
    record(f"WROTE {OUTPUT}")


def main() -> int:
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("MVSEC ATLIF-only ablation queue already active", flush=True)
            return 0
        for path in (
            CONFIG,
            MANIFEST,
            NB0_CKPT,
            NB0_FIXED,
            NB0_FULL,
            H67_FIXED,
            H67_FULL,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        wait_for_idle_gpu()
        smoke()
        train()
        checkpoint = select_best()
        evaluate(checkpoint, FIXED, True)
        evaluate(checkpoint, FULL, False)
        write_receipt(checkpoint)
        record("ALL COMPLETE MVSEC ATLIF-only two-contribution ablation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
