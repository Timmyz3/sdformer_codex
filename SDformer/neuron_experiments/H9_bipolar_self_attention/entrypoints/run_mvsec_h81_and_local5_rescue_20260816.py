#!/usr/bin/env python3
"""Queue H81 same-protocol MVSEC, then Local5 DSEC-ep44 day2 fine-tune."""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
ENTRY = EXP / "entrypoints"
GEN = EXP / "configs/generated"
RESULTS = EXP / "results"
MANIFEST = EXP / "manifests/mvsec_cicc_dt1_v1.json"
NB0_CKPT = RESULTS / "mvsec_cicc_nb0_w8_seed0_v4_20260811/checkpoint_epoch11.pth"
LOCAL5_DSEC = (
    RESULTS
    / "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/checkpoint_epoch44.pth"
)
H81_CFG = GEN / "mvsec_cicc_h81_nomotion_w8_seed0.yml"
L5_CFG = GEN / "mvsec_cicc_local5_dsec_ep44_ft15_w8_seed0.yml"
H81_TRAIN = RESULTS / "mvsec_cicc_h81_nomotion_w8_seed0_20260816"
H81_SMOKE = RESULTS / "mvsec_cicc_h81_nomotion_w8_seed0_smoke_20260816"
H81_FIXED = RESULTS / "mvsec_cicc_h81_nomotion_w8_seed0_fixed800_20260816"
H81_FULL = RESULTS / "mvsec_cicc_h81_nomotion_w8_seed0_full_20260816"
L5_TRAIN = RESULTS / "mvsec_cicc_local5_dsec_ep44_ft15_20260816"
L5_SMOKE = RESULTS / "mvsec_cicc_local5_dsec_ep44_ft15_smoke_20260816"
L5_FIXED = RESULTS / "mvsec_cicc_local5_dsec_ep44_ft15_fixed800_20260816"
L5_FULL = RESULTS / "mvsec_cicc_local5_dsec_ep44_ft15_full_20260816"
STATUS = RESULTS / "mvsec_h81_local5_rescue_watcher_20260816.log"
OUTPUT = REPO / "neuron_autoresearch/MVSEC_H81_LOCAL5_RESCUE_20260816.json"
OUTPUT_MD = OUTPUT.with_suffix(".md")
LOCK = Path("/tmp/sdformer_mvsec_h81_local5_rescue.lock")
EXIT_RE = re.compile(r"\[mvsec-cicc-train\] exit_code=(\d+)")
EPOCH_RE = re.compile(r"^Epoch (\d+)\s*$")
VALID_RE = re.compile(r"Epoch loss \(Validation\): ([0-9.eE+-]+)")
OLD_AUDIT = RESULTS / "mvsec_cicc_nb0_h67_local5_audit_20260812.json"
PY = sys.executable


def record(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def completed_train(output_dir: Path) -> int | None:
    log = output_dir / "train.log"
    if not log.is_file():
        return None
    matches = EXIT_RE.findall(log.read_text(encoding="utf-8", errors="replace"))
    return int(matches[-1]) if matches else None


def run(command: list[str], cwd: Path = REPO) -> None:
    record("START " + " ".join(command))
    result = subprocess.run(command, cwd=cwd)
    record(f"END exit_code={result.returncode}")
    if result.returncode:
        raise RuntimeError(f"command failed: {command}")


def validation_losses(train_log: Path) -> dict[int, float]:
    current = None
    losses: dict[int, float] = {}
    for line in train_log.read_text(encoding="utf-8", errors="replace").splitlines():
        epoch_match = EPOCH_RE.match(line.strip())
        if epoch_match:
            current = int(epoch_match.group(1))
        valid_match = VALID_RE.search(line)
        if valid_match and current is not None:
            losses[current] = float(valid_match.group(1))
    return losses


def select_best(output_dir: Path) -> Path:
    losses = validation_losses(output_dir / "train.log")
    candidates = []
    for checkpoint in output_dir.glob("checkpoint_epoch*.pth"):
        if checkpoint.name.endswith("_state_dict.pth"):
            continue
        match = re.match(r"checkpoint_epoch(\d+)\.pth$", checkpoint.name)
        if match and int(match.group(1)) in losses:
            candidates.append((losses[int(match.group(1))], int(match.group(1)), checkpoint))
    if not candidates:
        raise RuntimeError(f"no val-bound checkpoint in {output_dir}")
    loss, epoch, checkpoint = min(candidates)
    receipt = {
        "schema": "mvsec_best_valid_checkpoint_v1",
        "output_dir": str(output_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "epoch": epoch,
        "validation_loss": loss,
        "available_validation_losses": losses,
    }
    (output_dir / "best_checkpoint.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    record(f"BEST {output_dir.name} ep{epoch} val={loss:.4f}")
    return checkpoint.resolve()


def train_if_needed(config: Path, output_dir: Path, prev: Path) -> None:
    code = completed_train(output_dir)
    if code == 0:
        record(f"SKIP completed training {output_dir}")
        return
    if code not in (None, 0):
        raise RuntimeError(f"previous training failed: {output_dir}")
    run(
        [
            PY,
            "-u",
            str(ENTRY / "run_mvsec_cicc_train.py"),
            "--config",
            str(config),
            "--output-dir",
            str(output_dir),
            "--prev-runid",
            str(prev),
        ]
    )


def smoke(config: Path, smoke_dir: Path, prev: Path, required: tuple[str, ...]) -> None:
    if (smoke_dir / "load_audit.json").is_file():
        record(f"SKIP smoke {smoke_dir}")
        return
    smoke_cfg = yaml_smoke(config, smoke_dir)
    env = os.environ.copy()
    env["SDFORMER_MDR_MAX_TRAIN_BATCHES"] = "1"
    env["SDFORMER_MDR_MAX_VALID_BATCHES"] = "1"
    record("START smoke " + str(smoke_cfg))
    result = subprocess.run(
        [
            PY,
            "-u",
            str(ENTRY / "run_mvsec_cicc_train.py"),
            "--config",
            str(smoke_cfg),
            "--output-dir",
            str(smoke_dir),
            "--prev-runid",
            str(prev),
        ],
        cwd=REPO,
        env=env,
    )
    record(f"END smoke exit_code={result.returncode}")
    if result.returncode:
        raise RuntimeError(f"smoke failed: {smoke_dir}")
    text = (smoke_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    missing = [item for item in required if item not in text]
    if missing:
        raise RuntimeError(f"smoke audit failed {smoke_dir}: {missing}")
    for checkpoint in smoke_dir.glob("checkpoint_epoch*.pth"):
        checkpoint.unlink()
    (smoke_dir / "load_audit.json").write_text(
        json.dumps({"status": "PASS", "required": required}, indent=2) + "\n",
        encoding="utf-8",
    )


def yaml_smoke(config: Path, smoke_dir: Path) -> Path:
    import yaml

    data = yaml.safe_load(config.read_text(encoding="utf-8"))
    data["loader"]["n_epochs"] = 1
    smoke_dir.mkdir(parents=True, exist_ok=True)
    path = smoke_dir / "smoke_config.yml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def evaluate(config: Path, checkpoint: Path, out_dir: Path, fixed800: bool) -> None:
    summary = out_dir / "mvsec_summary.json"
    if summary.is_file():
        record(f"SKIP eval {out_dir}")
        return
    command = [
        PY,
        "-u",
        str(ENTRY / "run_h9_standard_mvsec_eval.py"),
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--out-dir",
        str(out_dir),
    ]
    if fixed800:
        command.extend(["--fixed800-manifest", str(MANIFEST)])
    run(command)


def load_summary(path: Path) -> dict:
    return json.loads((path / "mvsec_summary.json").read_text(encoding="utf-8"))


def write_receipt(h81_ckpt: Path, l5_ckpt: Path | None) -> None:
    old = json.loads(OLD_AUDIT.read_text(encoding="utf-8"))
    h81_full = load_summary(H81_FULL)
    h81_fixed = load_summary(H81_FIXED)
    nb0 = {row["sequence"]: row["AEE"] for row in old["routes"]["nb0"]["full_sequence"]["per_sequence"]}
    h81_seq = {row["sequence"]: row["AEE"] for row in h81_full.get("sequences") or h81_full.get("per_sequence") or []}
    if not h81_seq and "per_sequence" not in h81_full:
        # evaluator summary uses "sequences"
        rows = h81_full.get("sequences") or []
        h81_seq = {row["sequence"]: float(row["AEE"]) for row in rows}
    h81_all = all(h81_seq[name] < nb0[name] for name in nb0 if name in h81_seq)
    payload = {
        "schema": "mvsec_h81_local5_rescue_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "h81": {
            "protocol_family": "day2_scratch_same_as_h67_local5",
            "checkpoint": str(h81_ckpt),
            "fixed800": h81_fixed,
            "full_sequence": h81_full,
            "all_sequence_better_than_NB0": h81_all,
        },
    }
    if l5_ckpt is not None and (L5_FULL / "mvsec_summary.json").is_file():
        l5_full = load_summary(L5_FULL)
        l5_fixed = load_summary(L5_FIXED)
        l5_seq = {row["sequence"]: float(row["AEE"]) for row in (l5_full.get("sequences") or [])}
        l5_all = all(l5_seq[name] < nb0[name] for name in nb0 if name in l5_seq)
        payload["local5_dsec_ft"] = {
            "protocol_family": "dsec_pretrain_day2_ft",
            "checkpoint": str(l5_ckpt),
            "fixed800": l5_fixed,
            "full_sequence": l5_full,
            "all_sequence_better_than_NB0": l5_all,
            "old_scratch_IF1": 1.6282,
        }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# MVSEC H81 completion and Local5 rescue",
        "",
        f"H81 same-protocol full-sequence all-four-better-than-NB0: `{h81_all}`.",
        "",
        "Local5 rescue is DSEC-ep44 day2 fine-tune and is labeled separately from the scratch table.",
    ]
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    record(f"WROTE {OUTPUT}")


def main() -> int:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("MVSEC H81/Local5 rescue already active", flush=True)
            return 0
        for path in (H81_CFG, L5_CFG, NB0_CKPT, LOCAL5_DSEC, MANIFEST, OLD_AUDIT):
            if not path.is_file():
                raise FileNotFoundError(path)

        smoke(
            H81_CFG,
            H81_SMOKE,
            NB0_CKPT,
            (
                "installed ATLIFTernaryPSN before load: 105 modules",
                "installed Shiftmax attention before load: 12 modules",
                "checkpoint_overlay_keys=0",
                "missing=210",
                "unexpected=0",
            ),
        )
        train_if_needed(H81_CFG, H81_TRAIN, NB0_CKPT)
        h81_ckpt = select_best(H81_TRAIN)
        evaluate(H81_CFG, h81_ckpt, H81_FIXED, True)
        evaluate(H81_CFG, h81_ckpt, H81_FULL, False)

        l5_ckpt = None
        try:
            smoke(
                L5_CFG,
                L5_SMOKE,
                LOCAL5_DSEC,
                (
                    "installed ATLIFTernaryPSN before load: 105 modules",
                    "installed Shiftmax attention before load: 12 modules",
                    "checkpoint_overlay_keys=210",
                    "missing=12",
                    "unexpected=0",
                    "attn.positional_encoding",
                ),
            )
            train_if_needed(L5_CFG, L5_TRAIN, LOCAL5_DSEC)
            l5_ckpt = select_best(L5_TRAIN)
            evaluate(L5_CFG, l5_ckpt, L5_FIXED, True)
            evaluate(L5_CFG, l5_ckpt, L5_FULL, False)
        except Exception as exc:
            record(f"LOCAL5_TRANSFER_FAILED {exc}")

        write_receipt(h81_ckpt, l5_ckpt)
        record("ALL COMPLETE MVSEC H81 + Local5 rescue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
