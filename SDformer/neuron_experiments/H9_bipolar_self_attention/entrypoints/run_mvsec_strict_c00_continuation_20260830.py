#!/usr/bin/env python3
"""Train and evaluate strict same-parent MVSEC C00, then write a four-route receipt."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import yaml


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
ENTRY = EXP / "entrypoints"
RESULTS = EXP / "results"
CONFIG = EXP / "configs/generated/mvsec_cicc_strict_c00_psn_original_w8_seed0_full30_20260830.yml"
MANIFEST = EXP / "configs/generated/mvsec_strict_same_parent_c00_20260830.json"
SPLIT_MANIFEST = EXP / "manifests/mvsec_cicc_dt1_v1.json"
PARENT = RESULTS / "mvsec_cicc_nb0_w8_seed0_v4_20260811/checkpoint_epoch11.pth"
SMOKE = RESULTS / "mvsec_cicc_strict_c00_psn_original_smoke_20260830"
TRAIN = RESULTS / "mvsec_cicc_strict_c00_psn_original_full30_20260830"
FIXED = RESULTS / "mvsec_cicc_strict_c00_psn_original_fixed800_20260830"
FULL = RESULTS / "mvsec_cicc_strict_c00_psn_original_full_20260830"
EXISTING = REPO / "neuron_autoresearch/MVSEC_TWO_CONTRIBUTION_ABLATION_20260825.json"
OUTPUT = REPO / "neuron_autoresearch/MVSEC_STRICT_SAME_PARENT_ABLATION_20260830.json"
OUTPUT_MD = OUTPUT.with_suffix(".md")
REDESIGN = REPO / "neuron_autoresearch/EXPERIMENT_REDESIGN_PLAN.md"
STATUS = RESULTS / "mvsec_strict_c00_queue_20260830.log"
LOCK = Path("/tmp/sdformer_mvsec_strict_c00_20260830.lock")
PY = sys.executable
EXIT_RE = re.compile(r"\[mvsec-cicc-train\] exit_code=(\d+)")
EPOCH_RE = re.compile(r"^Epoch (\d+)\s*$")
VALID_RE = re.compile(r"Epoch loss \(Validation\): ([0-9.eE+-]+)")
RESULT_MARKER = "<!-- MVSEC_STRICT_SAME_PARENT_C00_RESULT_20260830 -->"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "SDFORMER_MDR_USE_MLFLOW": "0",
            "SDFORMER_MDR_SKIP_MLFLOW_MODEL_LOG": "1",
            "SDFORMER_MDR_SKIP_MLFLOW_STATE_LOG": "1",
            "SDFORMER_SNN_BACKEND": "cupy",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    return env


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    record("START " + " ".join(command))
    result = subprocess.run(command, cwd=REPO, env=env or environment())
    record(f"END exit_code={result.returncode}")
    if result.returncode:
        raise RuntimeError(f"command failed: {' '.join(command)}")


def wait_for_idle_gpu() -> None:
    stable = 0
    while stable < 2:
        pid_result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        memory_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if pid_result.returncode or memory_result.returncode:
            raise RuntimeError((pid_result.stderr + memory_result.stderr).strip())
        pids = [line.strip() for line in pid_result.stdout.splitlines() if line.strip().isdigit()]
        memory_mib = max(
            [int(line.strip()) for line in memory_result.stdout.splitlines() if line.strip().isdigit()],
            default=0,
        )
        if pids or memory_mib > 1024:
            stable = 0
            record(f"WAIT GPU compute pids={pids}, memory_mib={memory_mib}")
        else:
            stable += 1
            record(f"GPU idle confirmation {stable}/2 memory_mib={memory_mib}")
        if stable < 2:
            time.sleep(30)


def completed_train(output_dir: Path) -> int | None:
    log = output_dir / "train.log"
    if not log.is_file():
        return None
    matches = EXIT_RE.findall(log.read_text(encoding="utf-8", errors="replace"))
    return int(matches[-1]) if matches else None


def checkpoint_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    import torch

    payload = torch.load(str(path), map_location="cpu", weights_only=False, mmap=True)
    if hasattr(payload, "state_dict"):
        state = payload.state_dict()
    elif isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    elif isinstance(payload, dict):
        state = payload
    else:
        raise TypeError(f"unsupported checkpoint payload: {type(payload)}")
    return {key: tuple(value.shape) for key, value in state.items()}


def smoke() -> None:
    receipt = SMOKE / "load_audit.json"
    if receipt.is_file():
        record("SKIP completed strict-C00 smoke")
        return
    code = completed_train(SMOKE)
    if code is not None and code != 0:
        raise RuntimeError(f"previous strict-C00 smoke failed: {code}")
    if code is None:
        config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        config["loader"]["n_epochs"] = 1
        SMOKE.mkdir(parents=True, exist_ok=True)
        smoke_config = SMOKE / "smoke_config.yml"
        smoke_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        env = environment()
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
                str(PARENT),
            ],
            env=env,
        )
    text = (SMOKE / "train.log").read_text(encoding="utf-8", errors="replace")
    restored_marker = f"Model restored from local checkpoint {PARENT}"
    if restored_marker not in text:
        raise RuntimeError("strict-C00 smoke lacks baseline restore marker")
    if "installed ATLIFTernaryPSN" in text or "installed Shiftmax attention" in text:
        raise RuntimeError("strict-C00 smoke unexpectedly installed overlay modules")
    smoke_models = [
        path
        for path in SMOKE.glob("checkpoint_epoch*.pth")
        if not path.name.endswith("_state_dict.pth")
    ]
    if not smoke_models:
        raise RuntimeError("strict-C00 smoke did not save a model checkpoint")
    parent_shapes = checkpoint_shapes(PARENT)
    smoke_shapes = checkpoint_shapes(smoke_models[-1])
    missing = sorted(set(parent_shapes) - set(smoke_shapes))
    unexpected = sorted(set(smoke_shapes) - set(parent_shapes))
    shape_mismatch = sorted(
        key for key in set(parent_shapes) & set(smoke_shapes)
        if parent_shapes[key] != smoke_shapes[key]
    )
    if missing or unexpected or shape_mismatch:
        raise RuntimeError(
            "strict-C00 baseline checkpoint structure audit failed: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
            f"shape_mismatch={shape_mismatch[:8]}"
        )
    for checkpoint in SMOKE.glob("checkpoint_epoch*.pth"):
        checkpoint.unlink()
    receipt.write_text(
        json.dumps(
            {
                "status": "PASS",
                "load_path": "baseline load_model strict=False branch",
                "restore_marker": restored_marker,
                "parent_model_keys": len(parent_shapes),
                "smoke_model_keys": len(smoke_shapes),
                "missing_count": 0,
                "unexpected_count": 0,
                "shape_mismatch_count": 0,
                "ATLIFTernaryPSN": 0,
                "ShiftmaxAttention": 0,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def train() -> None:
    code = completed_train(TRAIN)
    if code == 0:
        record("SKIP completed strict-C00 full30")
        return
    if code is not None:
        raise RuntimeError(f"previous strict-C00 training failed: {code}")
    if (TRAIN / "train.log").is_file():
        raise RuntimeError("incomplete strict-C00 train.log exists; refusing overwrite")
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
            str(PARENT),
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
    candidates = []
    for checkpoint in TRAIN.glob("checkpoint_epoch*.pth"):
        if checkpoint.name.endswith("_state_dict.pth"):
            continue
        match = re.fullmatch(r"checkpoint_epoch(\d+)\.pth", checkpoint.name)
        if match and int(match.group(1)) in losses:
            epoch = int(match.group(1))
            candidates.append((losses[epoch], epoch, checkpoint.resolve()))
    if not candidates:
        raise RuntimeError("no validation-bound strict-C00 checkpoint")
    loss, epoch, checkpoint = min(candidates)
    receipt = {
        "schema": "mvsec_strict_c00_best_checkpoint_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "epoch": epoch,
        "validation_loss": loss,
        "all_validation_losses": losses,
    }
    (TRAIN / "best_checkpoint.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record(f"BEST strict-C00 ep{epoch} validation_loss={loss:.6f}")
    return checkpoint


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
        command.extend(["--fixed800-manifest", str(SPLIT_MANIFEST)])
    run(command)


def route_metrics(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw["sequences"]
    return {
        "summary": str(path.resolve()),
        "mean_aee": float(raw["mean_aee"]),
        "weighted_aee": float(raw["valid_pixel_weighted_aee"]),
        "mean_fl_percent": sum(float(row["gt_fl_percent"]) for row in rows) / len(rows),
        "total_spikes_g": sum(float(row["spikes_g"]) for row in rows),
        "total_energy_uj": sum(float(row["energy_uj"]) for row in rows),
        "per_sequence": rows,
    }


def write_receipt(checkpoint: Path) -> None:
    existing = json.loads(EXISTING.read_text(encoding="utf-8"))
    routes = {
        "c00_strict": {
            "fixed800": route_metrics(FIXED / "mvsec_summary.json"),
            "full_sequence": route_metrics(FULL / "mvsec_summary.json"),
        },
        "c10_atlif_only": existing["routes"]["atlif_only"],
        "c12_atlif_ttx": existing["routes"]["atlif_ttx"],
        "legacy_nb0_parent": existing["routes"]["nb0"],
    }
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload = {
        "schema": "mvsec_strict_same_parent_ablation_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha256(MANIFEST),
        "parent_checkpoint": str(PARENT.resolve()),
        "parent_checkpoint_sha256": sha256(PARENT),
        "selected_c00_checkpoint": str(checkpoint),
        "selected_c00_checkpoint_sha256": sha256(checkpoint),
        "protocol": manifest["protocol"],
        "routes": routes,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# MVSEC strict same-parent full30 ablation",
        "",
        "| route | macro AEE | weighted AEE | macro Fl (%) | spikes (G) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("c00_strict", "c10_atlif_only", "c12_atlif_ttx"):
        row = routes[name]["full_sequence"]
        lines.append(
            f"| {name} | {row['mean_aee']:.6f} | {row['weighted_aee']:.6f} | "
            f"{row['mean_fl_percent']:.4f} | {row['total_spikes_g']:.4f} |"
        )
    lines += [
        "",
        "All three rows start from the same MVSEC NB0 ep11 model and use the same fresh-optimizer full30 recipe.",
        "",
    ]
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")
    if RESULT_MARKER not in REDESIGN.read_text(encoding="utf-8"):
        with REDESIGN.open("a", encoding="utf-8") as handle:
            handle.write("\n" + RESULT_MARKER + "\n\n")
            handle.write("### MVSEC strict 同父 C00 full30 结果（2026-08-30）\n\n")
            handle.write("\n".join(lines[2:]) + "\n")
            handle.write(
                "\n权威收据：`neuron_autoresearch/MVSEC_STRICT_SAME_PARENT_ABLATION_20260830.{json,md}`。\n"
            )
    record(f"WROTE strict receipt {OUTPUT}")


def main() -> int:
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("MVSEC strict-C00 queue already active", flush=True)
            return 0
        for path in (CONFIG, MANIFEST, SPLIT_MANIFEST, PARENT, EXISTING):
            if not path.is_file():
                raise FileNotFoundError(path)
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if sha256(CONFIG) != manifest["config_sha256"]:
            raise RuntimeError("strict-C00 config SHA mismatch")
        wait_for_idle_gpu()
        smoke()
        train()
        checkpoint = select_best()
        evaluate(checkpoint, FIXED, True)
        evaluate(checkpoint, FULL, False)
        write_receipt(checkpoint)
        record("ALL COMPLETE MVSEC strict same-parent C00")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
