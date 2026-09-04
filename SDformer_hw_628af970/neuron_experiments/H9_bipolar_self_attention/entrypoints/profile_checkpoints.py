"""Profile multiple saved checkpoints for one H9 experiment run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


EXP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]


def run_profile(config: Path, checkpoint: Path, output_dir: Path, samples: int) -> dict | None:
    command = [
        sys.executable,
        "-u",
        str(EXP_ROOT / "entrypoints/profile_sops.py"),
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(output_dir),
        "--split",
        "valid",
        "--num-samples",
        str(samples),
        "--batch-size",
        "1",
        "--num-workers",
        "4",
        "--metric",
        "AEE",
        "--metric",
        "AAE",
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "profile.log").open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(command) + "\n")
        log.flush()
        proc = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
    summary_path = output_dir / "sops_summary.json"
    if proc.returncode != 0 or not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def compact(summary: dict) -> dict[str, float]:
    metrics = summary.get("metrics", {})
    return {
        "AEE": float(metrics.get("AEE", float("inf"))),
        "AAE": float(metrics.get("AAE", float("inf"))),
        "SOPs_G": float(summary.get("estimated_total_sops", float("inf"))) / 1.0e9,
        "firing": float(summary.get("global_firing_rate", float("inf"))),
    }


def score(metrics: dict[str, float]) -> float:
    return metrics["AEE"] + 0.025 * metrics["AAE"] + 0.35 * max(0.0, metrics["SOPs_G"] - 3.0847)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=EXP_ROOT / "results")
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--epoch", action="append", type=int, default=[])
    args = parser.parse_args()

    checkpoints: list[Path] = []
    if args.epoch:
        checkpoints = [args.run_dir / f"checkpoint_epoch{epoch}.pth" for epoch in args.epoch]
    else:
        checkpoints = sorted(args.run_dir.glob("checkpoint_epoch*.pth"))
    checkpoints = [path for path in checkpoints if path.exists()]
    if not checkpoints:
        print("No checkpoints found.", file=sys.stderr)
        return 1

    rows: list[tuple[Path, dict[str, float]]] = []
    for checkpoint in checkpoints:
        output_dir = args.output_root / f"profile_{args.run_dir.name}_{checkpoint.stem}_valid{args.samples}"
        summary = run_profile(args.config, checkpoint, output_dir, args.samples)
        if summary is None:
            print(f"profile failed: {checkpoint}", file=sys.stderr)
            continue
        metrics = compact(summary)
        rows.append((checkpoint, metrics))
        print(
            f"{checkpoint.name}: AEE={metrics['AEE']:.4f} AAE={metrics['AAE']:.4f} "
            f"SOPs={metrics['SOPs_G']:.4f}G firing={metrics['firing']:.5f}",
            flush=True,
        )

    if not rows:
        return 1
    rows.sort(key=lambda item: score(item[1]))
    report = args.run_dir / f"profile_ranking_valid{args.samples}.md"
    with report.open("w", encoding="utf-8") as handle:
        handle.write("# Checkpoint Profile Ranking\n\n")
        handle.write("| rank | checkpoint | AEE | AAE | SOPs(G) | firing | score |\n")
        handle.write("|---:|---|---:|---:|---:|---:|---:|\n")
        for rank, (checkpoint, metrics) in enumerate(rows, 1):
            handle.write(
                f"| {rank} | `{checkpoint.name}` | {metrics['AEE']:.4f} | {metrics['AAE']:.4f} | "
                f"{metrics['SOPs_G']:.4f} | {metrics['firing']:.5f} | {score(metrics):.4f} |\n"
            )
    print(f"ranking: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
