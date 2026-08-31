#!/usr/bin/env python3
"""Run held-out-validation-gated MVSEC Complete-TTX alpha screening."""

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
from typing import Any


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
MANIFEST = EXP / "configs/generated/mvsec_ttx_alpha_screen_20260826.json"
ROOT = EXP / "results/mvsec_ttx_alpha_screen_20260826"
STATUS = ROOT / "status.log"
LOCK = Path("/tmp/sdformer_mvsec_ttx_alpha_screen_20260826.lock")
TRAINER = EXP / "entrypoints/run_mvsec_cicc_train.py"
EVALUATOR = EXP / "entrypoints/run_h9_standard_mvsec_eval.py"
FIXED800 = EXP / "manifests/mvsec_cicc_dt1_v1.json"
EPOCH_RE = re.compile(r"^Epoch (\d+)\s*$")
VALID_RE = re.compile(r"Epoch loss \(Validation\): ([0-9.eE+-]+)")
CHECKPOINT_RE = re.compile(r"checkpoint_epoch(\d+)\.pth$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
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


def run(command: list[str], log_path: Path, label: str) -> None:
    record(f"START {label}: {' '.join(command)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            command,
            cwd=REPO,
            env=environment(),
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    record(f"END {label}: exit_code={result.returncode}")
    if result.returncode:
        raise RuntimeError(f"{label} failed; see {log_path}")


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


def select_best(run_dir: Path) -> dict[str, Any]:
    losses = validation_losses(run_dir / "train.log")
    candidates = []
    for checkpoint in run_dir.glob("checkpoint_epoch*.pth"):
        if checkpoint.name.endswith("_state_dict.pth"):
            continue
        match = CHECKPOINT_RE.match(checkpoint.name)
        if match and int(match.group(1)) in losses:
            epoch = int(match.group(1))
            candidates.append((losses[epoch], epoch, checkpoint.resolve()))
    if not candidates:
        raise RuntimeError(f"no validation-bound checkpoint found in {run_dir}")
    loss, epoch, checkpoint = min(candidates)
    receipt = {
        "epoch": epoch,
        "validation_loss": loss,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "all_validation_losses": losses,
    }
    (run_dir / "best_checkpoint.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def audit_initial_load(train_log: Path) -> None:
    text = train_log.read_text(encoding="utf-8", errors="replace")
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
        raise RuntimeError(f"MVSEC alpha initial-load audit failed: missing {missing}")


def train_candidate(row: dict[str, Any], parent: Path) -> dict[str, Any]:
    run_dir = ROOT / row["name"] / "train"
    completion = run_dir / "best_checkpoint.json"
    if not completion.is_file():
        run(
            [
                sys.executable,
                "-u",
                str(TRAINER),
                "--config",
                row["config"],
                "--output-dir",
                str(run_dir),
                "--prev-runid",
                str(parent),
            ],
            ROOT / row["name"] / "runner_train.log",
            f"MVSEC train {row['name']}",
        )
        audit_initial_load(run_dir / "train.log")
        return select_best(run_dir)
    audit_initial_load(run_dir / "train.log")
    return json.loads(completion.read_text(encoding="utf-8"))


def evaluate_candidate(row: dict[str, Any], best: dict[str, Any]) -> dict[str, Any]:
    outputs = {}
    for protocol in ("fixed800", "full"):
        out_dir = ROOT / row["name"] / f"eval_{protocol}"
        summary = out_dir / "mvsec_summary.json"
        if not summary.is_file():
            command = [
                sys.executable,
                "-u",
                str(EVALUATOR),
                "--config",
                row["config"],
                "--checkpoint",
                best["checkpoint"],
                "--out-dir",
                str(out_dir),
            ]
            if protocol == "fixed800":
                command.extend(["--fixed800-manifest", str(FIXED800)])
            run(
                command,
                ROOT / row["name"] / f"runner_eval_{protocol}.log",
                f"MVSEC eval {row['name']} {protocol}",
            )
        outputs[protocol] = json.loads(summary.read_text(encoding="utf-8"))
    return outputs


def write_summary(
    manifest: dict[str, Any],
    screened: list[dict[str, Any]],
    promoted: dict[str, Any] | None,
) -> None:
    payload = {
        "schema": "mvsec_ttx_alpha_screen_result_v1",
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": sha256(MANIFEST),
        "screened": screened,
        "promoted": promoted,
        "selection_rule": manifest["selection_rule"],
        "claim_boundary": manifest["claim_boundary"],
    }
    (ROOT / "summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# MVSEC Complete-TTX dyadic-alpha screen",
        "",
        "| alpha | validation rank1 epoch | held-out validation loss | promoted to test |",
        "|---:|---:|---:|---:|",
    ]
    promoted_name = promoted["name"] if promoted else None
    for row in screened:
        lines.append(
            f"| {row['alpha']} | {row['best']['epoch']} | "
            f"{row['best']['validation_loss']:.7f} | "
            f"{'yes' if row['name'] == promoted_name else 'no'} |"
        )
    if promoted:
        lines += [
            "",
            f"Promoted `{promoted['name']}` using held-out day2 validation only. "
            f"Full-sequence macro AEE: `{promoted['evaluation']['full']['mean_aee']:.6f}`; "
            f"valid-pixel-weighted AEE: "
            f"`{promoted['evaluation']['full']['valid_pixel_weighted_aee']:.6f}`.",
        ]
    else:
        lines += [
            "",
            "No new alpha passed the preregistered held-out validation margin; the existing "
            "alpha=0.25 H67 checkpoint remains frozen and no new test-sequence evaluation ran.",
        ]
    lines += [
        "",
        "OD1/IF1/IF2/IF3 are not used for alpha or checkpoint selection.",
    ]
    (ROOT / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("MVSEC alpha screen already active", flush=True)
            return 0
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        parent = Path(manifest["parent_checkpoint"])
        if sha256(parent) != manifest["parent_checkpoint_sha256"]:
            raise RuntimeError("MVSEC alpha-screen parent checkpoint SHA changed")
        threshold = manifest["baseline_h67_best_validation_loss"] * (
            1.0 - manifest["promotion_relative_margin"]
        )
        candidates = {row["name"]: row for row in manifest["candidates"]}
        screened = []
        first = candidates["alpha050"]
        best = train_candidate(first, parent)
        screened.append({**first, "best": best})
        if best["validation_loss"] >= threshold:
            record(
                f"alpha=0.5 did not pass validation threshold {threshold:.7f}; "
                "running alpha=0.125 fallback"
            )
            second = candidates["alpha0125"]
            second_best = train_candidate(second, parent)
            screened.append({**second, "best": second_best})
        eligible = [
            row for row in screened if row["best"]["validation_loss"] < threshold
        ]
        promoted = None
        if eligible:
            winner = min(eligible, key=lambda row: row["best"]["validation_loss"])
            evaluation = evaluate_candidate(winner, winner["best"])
            promoted = {**winner, "evaluation": evaluation}
            record(
                f"PROMOTE {winner['name']} by held-out validation; "
                f"loss={winner['best']['validation_loss']:.7f}"
            )
        write_summary(manifest, screened, promoted)
        record("ALL COMPLETE MVSEC Complete-TTX dyadic-alpha screen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
