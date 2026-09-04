#!/usr/bin/env python3
"""Complete NB0 -> H67/Local5 direct-MVSEC training and fixed800 evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
ENTRYPOINT_ROOT = REPO_ROOT / "neuron_experiments/H9_bipolar_self_attention/entrypoints"
CONFIG_ROOT = REPO_ROOT / "neuron_experiments/H9_bipolar_self_attention/configs/generated"
RESULT_ROOT = REPO_ROOT / "neuron_experiments/H9_bipolar_self_attention/results"
MANIFEST = REPO_ROOT / "neuron_experiments/H9_bipolar_self_attention/manifests/mvsec_cicc_dt1_v1.json"
TRAINER = ENTRYPOINT_ROOT / "run_mvsec_cicc_train.py"
EVALUATOR = ENTRYPOINT_ROOT / "run_h9_standard_mvsec_eval.py"
EXIT_RE = re.compile(r"\[mvsec-cicc-train\] exit_code=(\d+)")
EPOCH_RE = re.compile(r"^Epoch (\d+)\s*$")
VALID_RE = re.compile(r"Epoch loss \(Validation\): ([0-9.eE+-]+)")
CHECKPOINT_RE = re.compile(r"checkpoint_epoch(\d+)\.pth$")


ROUTES = {
    "nb0": {
        "config": CONFIG_ROOT / "mvsec_cicc_nb0_w8_seed0.yml",
        "train": RESULT_ROOT / "mvsec_cicc_nb0_w8_seed0_v4_20260811",
        "eval_fixed800": RESULT_ROOT / "mvsec_cicc_nb0_w8_seed0_v4_fixed800_20260811",
        "eval_full": RESULT_ROOT / "mvsec_cicc_nb0_w8_seed0_v4_full_20260811",
    },
    "h67": {
        "config": CONFIG_ROOT / "mvsec_cicc_h67_motion_w8_seed0.yml",
        "train": RESULT_ROOT / "mvsec_cicc_h67_motion_w8_seed0_v4_20260811",
        "smoke": RESULT_ROOT / "mvsec_cicc_h67_motion_w8_seed0_v4_load_smoke_20260811",
        "eval_fixed800": RESULT_ROOT / "mvsec_cicc_h67_motion_w8_seed0_v4_fixed800_20260811",
        "eval_full": RESULT_ROOT / "mvsec_cicc_h67_motion_w8_seed0_v4_full_20260811",
    },
    "local5": {
        "config": CONFIG_ROOT / "mvsec_cicc_local5_w8_seed0.yml",
        "train": RESULT_ROOT / "mvsec_cicc_local5_w8_seed0_v4_20260811",
        "smoke": RESULT_ROOT / "mvsec_cicc_local5_w8_seed0_v4_load_smoke_20260811",
        "eval_fixed800": RESULT_ROOT / "mvsec_cicc_local5_w8_seed0_v4_fixed800_20260811",
        "eval_full": RESULT_ROOT / "mvsec_cicc_local5_w8_seed0_v4_full_20260811",
    },
}


def log(message: str, handle) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    handle.write(line + "\n")
    handle.flush()


def completed_exit_code(output_dir: Path) -> int | None:
    train_log = output_dir / "train.log"
    if not train_log.is_file():
        return None
    matches = EXIT_RE.findall(train_log.read_text(encoding="utf-8", errors="replace"))
    return int(matches[-1]) if matches else None


def make_smoke_config(config: Path, output_dir: Path) -> Path:
    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["loader"]["n_epochs"] = 1
    output_dir.mkdir(parents=True, exist_ok=True)
    smoke_config = output_dir / "smoke_config.yml"
    smoke_config.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return smoke_config


def remove_smoke_checkpoints(output_dir: Path) -> None:
    removed: list[dict[str, object]] = []
    for checkpoint in sorted(output_dir.glob("checkpoint_epoch*.pth")):
        removed.append({"path": str(checkpoint.resolve()), "bytes": checkpoint.stat().st_size})
        checkpoint.unlink()
    receipt = {
        "schema": "mvsec_load_smoke_checkpoint_cleanup_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "removed": removed,
        "removed_bytes": sum(int(row["bytes"]) for row in removed),
        "retained": ["train.log", "launch_provenance.json", "smoke_config.yml"],
    }
    (output_dir / "checkpoint_cleanup.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )


def run_training(config: Path, output_dir: Path, checkpoint: Path | None = None, smoke: bool = False) -> None:
    existing = completed_exit_code(output_dir)
    if existing is not None:
        if existing != 0:
            raise RuntimeError(f"existing training failed with exit code {existing}: {output_dir}")
        return
    run_config = config
    if smoke:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_config = make_smoke_config(config, output_dir)
    command = [
        sys.executable,
        "-u",
        str(TRAINER),
        "--config",
        str(run_config),
        "--output-dir",
        str(output_dir),
    ]
    if checkpoint is not None:
        command.extend(["--prev-runid", str(checkpoint)])
    env = os.environ.copy()
    if smoke:
        env.update(
            {
                "SDFORMER_MDR_MAX_TRAIN_BATCHES": "1",
                "SDFORMER_MDR_MAX_VALID_BATCHES": "1",
            }
        )
    result = subprocess.run(command, cwd=REPO_ROOT, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"training failed with exit code {result.returncode}: {output_dir}")


def wait_for_external_training(output_dir: Path, handle, poll_seconds: int) -> None:
    while True:
        code = completed_exit_code(output_dir)
        if code is not None:
            if code != 0:
                raise RuntimeError(f"joined training failed with exit code {code}: {output_dir}")
            log(f"joined completed training: {output_dir}", handle)
            return
        log(f"WAIT active NB0 training: {output_dir}", handle)
        time.sleep(poll_seconds)


def validation_losses(train_log: Path) -> dict[int, float]:
    current_epoch: int | None = None
    losses: dict[int, float] = {}
    for line in train_log.read_text(encoding="utf-8", errors="replace").splitlines():
        epoch_match = EPOCH_RE.match(line.strip())
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
        valid_match = VALID_RE.search(line)
        if valid_match and current_epoch is not None:
            losses[current_epoch] = float(valid_match.group(1))
    return losses


def select_best_checkpoint(output_dir: Path) -> Path:
    losses = validation_losses(output_dir / "train.log")
    candidates: list[tuple[float, int, Path]] = []
    for checkpoint in output_dir.glob("checkpoint_epoch*.pth"):
        if checkpoint.name.endswith("_state_dict.pth"):
            continue
        match = CHECKPOINT_RE.match(checkpoint.name)
        if match and int(match.group(1)) in losses:
            epoch = int(match.group(1))
            candidates.append((losses[epoch], epoch, checkpoint.resolve()))
    if not candidates:
        raise RuntimeError(f"no validation-bound checkpoint found in {output_dir}")
    loss, epoch, checkpoint = min(candidates)
    receipt = {
        "schema": "mvsec_best_valid_checkpoint_v1",
        "output_dir": str(output_dir.resolve()),
        "checkpoint": str(checkpoint),
        "epoch": epoch,
        "validation_loss": loss,
        "available_validation_losses": losses,
    }
    (output_dir / "best_checkpoint.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return checkpoint


def audit_candidate_load(smoke_dir: Path) -> None:
    text = (smoke_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    required = (
        "installed ATLIFTernaryPSN before load: 105 modules",
        "installed Shiftmax attention before load: 12 modules",
        "checkpoint_overlay_keys=0",
        "missing=210",
        "unexpected=0",
        "[mvsec-cicc-train] exit_code=0",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise RuntimeError(f"candidate load audit failed for {smoke_dir}: missing {missing}")


def run_evaluation(route: dict[str, Path], checkpoint: Path, fixed800: bool) -> None:
    output_dir = route["eval_fixed800" if fixed800 else "eval_full"]
    summary = output_dir / "mvsec_summary.json"
    if summary.is_file():
        return
    command = [
        sys.executable,
        "-u",
        str(EVALUATOR),
        "--config",
        str(route["config"]),
        "--checkpoint",
        str(checkpoint),
        "--out-dir",
        str(output_dir),
    ]
    if fixed800:
        command.extend(["--fixed800-manifest", str(MANIFEST)])
    result = subprocess.run(command, cwd=REPO_ROOT)
    if result.returncode != 0:
        protocol = "fixed800" if fixed800 else "full-sequence"
        raise RuntimeError(f"{protocol} evaluation failed: {output_dir}")


def write_comparison(checkpoints: dict[str, Path], output_path: Path) -> None:
    routes = []
    for name, route in ROUTES.items():
        fixed_summary = json.loads(
            (route["eval_fixed800"] / "mvsec_summary.json").read_text(encoding="utf-8")
        )
        full_summary = json.loads(
            (route["eval_full"] / "mvsec_summary.json").read_text(encoding="utf-8")
        )
        routes.append(
            {
                "route": name,
                "checkpoint": str(checkpoints[name]),
                "fixed800": fixed_summary,
                "full_sequence": full_summary,
            }
        )
    payload = {
        "schema": "mvsec_cicc_nb0_h67_local5_comparison_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(MANIFEST),
        "routes": routes,
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown = output_path.with_suffix(".md")
    with markdown.open("w", encoding="utf-8") as handle:
        handle.write("# Direct MVSEC Fixed800 Comparison\n\n")
        handle.write("| route | protocol | best checkpoint | mean AEE | valid-pixel-weighted AEE | valid pixels |\n")
        handle.write("|---|---|---|---:|---:|---:|\n")
        for row in routes:
            for protocol, key in (("fixed800", "fixed800"), ("full", "full_sequence")):
                summary = row[key]
                handle.write(
                    f"| {row['route']} | {protocol} | `{Path(row['checkpoint']).name}` | "
                    f"{summary['mean_aee']:.4f} | "
                    f"{summary['valid_pixel_weighted_aee']:.4f} | "
                    f"{summary['total_valid_pixels']} |\n"
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--join-active-nb0", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    pipeline_log = RESULT_ROOT / "mvsec_cicc_pipeline_20260811.log"
    with pipeline_log.open("a", encoding="utf-8") as handle:
        if args.join_active_nb0 and completed_exit_code(ROUTES["nb0"]["train"]) is None:
            wait_for_external_training(ROUTES["nb0"]["train"], handle, args.poll_seconds)
        else:
            run_training(ROUTES["nb0"]["config"], ROUTES["nb0"]["train"])

        checkpoints = {"nb0": select_best_checkpoint(ROUTES["nb0"]["train"])}
        log(f"NB0 best checkpoint: {checkpoints['nb0']}", handle)
        run_evaluation(ROUTES["nb0"], checkpoints["nb0"], fixed800=True)
        run_evaluation(ROUTES["nb0"], checkpoints["nb0"], fixed800=False)

        for name in ("h67", "local5"):
            route = ROUTES[name]
            run_training(route["config"], route["smoke"], checkpoints["nb0"], smoke=True)
            audit_candidate_load(route["smoke"])
            remove_smoke_checkpoints(route["smoke"])
            log(f"{name} NB0-load smoke PASS", handle)
            run_training(route["config"], route["train"], checkpoints["nb0"])
            checkpoints[name] = select_best_checkpoint(route["train"])
            log(f"{name} best checkpoint: {checkpoints[name]}", handle)
            run_evaluation(route, checkpoints[name], fixed800=True)
            run_evaluation(route, checkpoints[name], fixed800=False)

        comparison = RESULT_ROOT / "mvsec_cicc_nb0_h67_local5_comparison_20260811.json"
        write_comparison(checkpoints, comparison)
        log(f"ALL COMPLETE direct-MVSEC CICC pipeline: {comparison}", handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
