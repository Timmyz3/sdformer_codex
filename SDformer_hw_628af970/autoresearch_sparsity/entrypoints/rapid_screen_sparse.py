"""Short-train sparse/voxel candidates and profile them consistently."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CKPT = REPO_ROOT / "experiments/checkpoints/bs4_resume_epoch15_to60_20260424_163657/checkpoint_epoch59.pth"


def deep_set(tree: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = tree
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def run_cmd(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\n[rapid-sparse] exit_code={proc.returncode}\n")
        return int(proc.returncode)


def read_profile(profile_dir: Path) -> dict[str, Any]:
    summary_path = profile_dir / "sops_summary.json"
    if not summary_path.exists():
        return {"summary": str(summary_path), "status": "missing_profile"}
    data = json.loads(summary_path.read_text())
    metrics = data.get("metrics", {})
    return {
        "status": "ok",
        "summary": str(summary_path),
        "AEE": metrics.get("AEE"),
        "AAE": metrics.get("AAE"),
        "PE1": metrics.get("AEE_PE1"),
        "PE2": metrics.get("AEE_PE2"),
        "PE3": metrics.get("AEE_PE3"),
        "SOPs_G": data.get("estimated_total_sops", 0.0) / 1e9,
        "firing": data.get("global_firing_rate"),
    }


def gate(row: dict[str, Any], max_aee: float, max_aae: float, max_sops: float) -> str:
    failures = []
    if row.get("profile_exit") not in {None, 0}:
        failures.append("profile_failed")
    if row.get("status") != "ok":
        failures.append(str(row.get("status")))
    if row.get("AEE") is None or float(row["AEE"]) > max_aee:
        failures.append(f"AEE>{max_aee}")
    if row.get("AAE") is None or float(row["AAE"]) > max_aae:
        failures.append(f"AAE>{max_aae}")
    if row.get("SOPs_G") is None or float(row["SOPs_G"]) > max_sops:
        failures.append(f"SOPs>{max_sops}G")
    return "pass" if not failures else ";".join(failures)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="sparse_voxel_short")
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--valid-samples", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--profile-batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--max-aee", type=float, default=1.90)
    parser.add_argument("--max-aae", type=float, default=9.00)
    parser.add_argument("--max-sops", type=float, default=3.35)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = REPO_ROOT / "autoresearch_sparsity" / "results" / f"{args.tag}_{stamp}"
    cfg_root = out_root / "configs"
    run_root = out_root / "runs"
    profile_root = out_root / "profiles"
    cfg_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for cfg_name in args.config:
        cfg_path = Path(cfg_name)
        if not cfg_path.is_absolute():
            cfg_path = REPO_ROOT / cfg_path
        cfg = yaml.safe_load(cfg_path.read_text())
        name = cfg.get("experiment") or cfg_path.stem
        name = f"{name}_steps{args.steps}"
        cfg["experiment"] = name
        deep_set(cfg, "runtime.max_train_steps", args.steps)
        deep_set(cfg, "runtime.use_mlflow_model_logging", False)
        deep_set(cfg, "runtime.skip_train_validation", True)
        deep_set(cfg, "loader.n_epochs", 1)
        deep_set(cfg, "loader.batch_size", args.batch_size)
        deep_set(cfg, "loader.n_workers", args.workers)
        deep_set(cfg, "loader.pin_memory", False)
        if args.workers > 0:
            deep_set(cfg, "loader.persistent_workers", True)
            deep_set(cfg, "loader.prefetch_factor", 4)
        else:
            deep_set(cfg, "loader.persistent_workers", False)
            cfg.get("loader", {}).pop("prefetch_factor", None)
        deep_set(cfg, "optimizer.use_amp", bool(args.amp))
        deep_set(cfg, "test.sample", min(args.valid_samples, 40))
        deep_set(cfg, "test.n_valid", 1)
        local_cfg = cfg_root / f"{name}.yml"
        local_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))

        run_dir = run_root / name
        ckpt_template = run_dir / "checkpoint_epoch{}.pth"
        train_cmd = [
            sys.executable, "-u", "-m", "autoresearch_sparsity.entrypoints.train",
            "--config", str(local_cfg),
            "--prev_runid", str(Path(args.checkpoint).resolve()),
            "--save_path", str(ckpt_template),
        ]
        start = time.time()
        train_code = run_cmd(train_cmd, run_dir / "train.log")
        train_seconds = time.time() - start
        ckpt = run_dir / "checkpoint_epoch0.pth"
        row: dict[str, Any] = {
            "name": name,
            "source_config": str(cfg_path),
            "steps": args.steps,
            "samples": args.valid_samples,
            "train_exit": train_code,
            "train_seconds": train_seconds,
        }
        if train_code != 0 or not ckpt.exists():
            row.update({"status": "train_failed", "gate": "train_failed"})
        else:
            profile_dir = profile_root / f"{name}_valid{args.valid_samples}"
            profile_cmd = [
                sys.executable, "-u", "-m", "autoresearch_sparsity.entrypoints.profile_upstream_sparse",
                "--config", str(local_cfg),
                "--checkpoint", str(ckpt),
                "--output-dir", str(profile_dir),
                "--split", "valid",
                "--num-samples", str(args.valid_samples),
                "--batch-size", str(args.profile_batch_size or args.batch_size),
                "--num-workers", "4",
                "--snn-backend", "torch",
                "--metric", "AEE",
                "--metric", "AAE",
            ]
            pstart = time.time()
            profile_code = run_cmd(profile_cmd, profile_dir / "profile.log")
            row["profile_exit"] = profile_code
            row["profile_seconds"] = time.time() - pstart
            row.update(read_profile(profile_dir))
            if profile_code != 0:
                row["status"] = "profile_failed"
            row["gate"] = gate(row, args.max_aee, args.max_aae, args.max_sops)
        rows.append(row)

        summary_csv = out_root / "summary.csv"
        fields = [
            "name", "gate", "status", "steps", "samples", "AEE", "AAE", "PE1", "PE2", "PE3",
            "SOPs_G", "firing", "train_exit", "profile_exit", "train_seconds", "profile_seconds",
            "source_config", "summary",
        ]
        with summary_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        lines = ["# 稀疏/体素化短测结果", ""]
        for item in rows:
            lines.append(
                f"- `{item['name']}`: gate={item.get('gate')}, "
                f"AEE={item.get('AEE')}, AAE={item.get('AAE')}, "
                f"SOPs_G={item.get('SOPs_G')}, firing={item.get('firing')}"
            )
        (out_root / "summary.md").write_text("\n".join(lines) + "\n")

    print(f"summary: {out_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
