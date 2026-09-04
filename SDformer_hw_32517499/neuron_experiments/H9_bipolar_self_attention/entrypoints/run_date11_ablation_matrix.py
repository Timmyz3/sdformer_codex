"""Run DATE11 full-factorial ablation configs with standard valid825 eval.

This is the portable runner for a second server. It generates/uses the DATE11
manifest, trains selected configs from NB0, evaluates saved checkpoints on the
standard valid825 split, and leaves a compact status file per launch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
GENERATOR = EXP / "entrypoints/make_date11_full_factorial_configs.py"
MANIFEST = EXP / "configs/generated/date11_full_factorial_manifest.json"
BASELINE = REPO / "experiments/baseline_stride_upstream/checkpoint_epoch59.pth"
DEFAULT_EPOCHS = [9, 14, 19, 24, 28, 29]


PRESETS: dict[str, list[str]] = {
    "full": [
        "date11full_all_binary_atlif_original_w720_fastlr",
        "date11full_all_binary_atlif_tx_w720_fastlr",
        "date11full_all_binary_atlif_sc_w720_fastlr",
        "date11full_all_binary_atlif_nts_w720_fastlr",
        "date11full_all_ternary_atlif_original_w720_fastlr",
        "date11full_all_ternary_atlif_tx_w720_fastlr",
        "date11full_all_ternary_atlif_sc_w720_fastlr",
        "date11full_all_ternary_atlif_nts_w720_fastlr",
        "date11full_psn_tx_w720_fastlr",
        "date11full_psn_sc_w720_fastlr",
        "date11full_psn_nts_w720_fastlr",
    ],
    "date-paper-core": [
        "date11full_all_binary_atlif_original_w720_fastlr",
        "date11full_all_binary_atlif_tx_w720_fastlr",
        "date11full_all_binary_atlif_sc_w720_fastlr",
        "date11full_all_binary_atlif_nts_w720_fastlr",
        "date11full_all_ternary_atlif_original_w720_fastlr",
        "date11full_all_ternary_atlif_tx_w720_fastlr",
        "date11full_all_ternary_atlif_sc_w720_fastlr",
        "date11full_all_ternary_atlif_nts_w720_fastlr",
    ],
    "psn-attention-only": [
        "date11full_psn_tx_w720_fastlr",
        "date11full_psn_sc_w720_fastlr",
        "date11full_psn_nts_w720_fastlr",
    ],
}


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def env() -> dict[str, str]:
    out = os.environ.copy()
    out["SDFORMER_USE_MLFLOW"] = "0"
    out["SDFORMER_MLFLOW_MODEL_LOGGING"] = "0"
    out["SDFORMER_SNN_BACKEND"] = out.get("SDFORMER_SNN_BACKEND", "cupy")
    out["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    return out


def run(command: list[str], log: Path, *, dry_run: bool = False) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n=== {now()} ===\n")
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        if dry_run:
            handle.write("[dry-run] skipped\n")
            return
        proc = subprocess.run(command, cwd=REPO, env=env(), stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"[exit_code] {proc.returncode}\n")
    if not dry_run and proc.returncode != 0:
        raise RuntimeError(f"command failed; log={log}")


def load_manifest(generate: bool) -> list[dict[str, Any]]:
    if generate or not MANIFEST.exists():
        subprocess.run([sys.executable, str(GENERATOR)], cwd=REPO, check=True)
    rows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return [row for row in rows if row.get("name") != "NB0"]


def selected_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    names = set(PRESETS[args.preset])
    if args.name:
        names &= set(args.name)
    out = [row for row in rows if row["name"] in names]
    if args.priority:
        wanted = set(args.priority)
        out = [row for row in out if row.get("priority") in wanted]
    order = {"P0": 0, "P1": 1, "P2": 2}
    return sorted(out, key=lambda row: (order.get(str(row.get("priority")), 99), str(row["name"])))


def experiment_name(config: Path) -> str:
    for line in config.read_text(encoding="utf-8").splitlines():
        if line.startswith("experiment:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return config.stem


def existing_complete(run_dir: Path) -> bool:
    return (run_dir / "profile_ranking_valid825.md").exists()


def run_one(row: dict[str, Any], args: argparse.Namespace, driver: Path) -> None:
    config = Path(row["config"])
    if not config.is_absolute():
        config = REPO / config
    exp_name = experiment_name(config)
    run_dir = EXP / "results" / f"{exp_name}_bs8_{args.stamp}_setsid"
    log = run_dir / "pipeline.log"
    status = {
        "time": now(),
        "name": row["name"],
        "priority": row.get("priority"),
        "config": str(config.relative_to(REPO)),
        "run_dir": str(run_dir.relative_to(REPO)),
        "expected": {
            "atlif": row.get("atlif_expected"),
            "ternary": row.get("ternary_expected"),
            "binary": row.get("binary_expected"),
            "shiftmax": row.get("shiftmax_expected"),
        },
    }
    with (driver / "status.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "start", **status}, ensure_ascii=False) + "\n")

    if args.skip_existing and existing_complete(run_dir):
        with (driver / "status.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "skip_existing", **status}, ensure_ascii=False) + "\n")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-u", str(EXP / "entrypoints/verify_nts11_chain.py"), str(config)], log, dry_run=args.dry_run)
    run(
        [
            sys.executable,
            "-u",
            str(EXP / "entrypoints/train.py"),
            "--config",
            str(config),
            "--prev_runid",
            str(args.resume),
            "--save_path",
            str(run_dir / "checkpoint_epoch{}.pth"),
        ],
        log,
        dry_run=args.dry_run,
    )
    eval_cmd = [
        sys.executable,
        "-u",
        str(EXP / "entrypoints/run_h9_standard_valid825_eval.py"),
        "--config",
        str(config),
        "--run-dir",
        str(run_dir),
    ]
    for epoch in args.epoch:
        eval_cmd.extend(["--epoch", str(epoch)])
    run(eval_cmd, log, dry_run=args.dry_run)

    with (driver / "status.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "complete", **status}, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="date-paper-core")
    parser.add_argument("--name", action="append", default=[], help="Run only this manifest name; repeatable.")
    parser.add_argument("--priority", action="append", default=[], choices=["P0", "P1", "P2"])
    parser.add_argument("--epoch", action="append", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--resume", type=Path, default=BASELINE)
    parser.add_argument("--stamp", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--generate", action="store_true", help="Regenerate configs/manifest before running.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.resume.exists():
        raise FileNotFoundError(args.resume)
    driver = EXP / "results" / f"date11_ablation_matrix_driver_{args.stamp}"
    driver.mkdir(parents=True, exist_ok=True)
    rows = selected_rows(load_manifest(args.generate), args)
    plan = driver / "plan.json"
    plan.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"driver={driver}")
    print(f"plan={plan}")
    print("selected:")
    for row in rows:
        print(f"  {row.get('priority')} {row['name']} -> {row['config']}")
    for row in rows:
        run_one(row, args, driver)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
