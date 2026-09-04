"""Summarize full-valid profile result directories into a Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-missing-checkpoint",
        action="store_true",
        help="Skip summaries whose recorded checkpoint path no longer exists.",
    )
    args = parser.parse_args()

    rows = []
    for summary_path in sorted(args.root.glob("*/sops_summary.json")):
        summary = load_summary(summary_path)
        metrics = summary.get("metrics", {})
        checkpoint_path = Path(str(summary.get("checkpoint", "")))
        checkpoint_ok = checkpoint_path.exists()
        if args.skip_missing_checkpoint and not checkpoint_ok:
            continue
        rows.append(
            {
                "name": summary_path.parent.name,
                "samples": int(summary.get("samples", 0)),
                "AEE": float(metrics.get("AEE", float("nan"))),
                "AAE": float(metrics.get("AAE", float("nan"))),
                "PE1": float(metrics.get("AEE_PE1", float("nan"))),
                "PE2": float(metrics.get("AEE_PE2", float("nan"))),
                "PE3": float(metrics.get("AEE_PE3", float("nan"))),
                "outlier": float(metrics.get("AEE_outliers", float("nan"))),
                "SOPs_G": float(summary.get("estimated_total_sops", float("nan"))) / 1.0e9,
                "firing": float(summary.get("global_firing_rate", float("nan"))),
                "checkpoint": checkpoint_path.name,
                "status": "ok" if checkpoint_ok else "missing_checkpoint",
            }
        )

    rows.sort(key=lambda row: (row["AEE"], row["AAE"], row["SOPs_G"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write("# Full Valid Profile Results\n\n")
        handle.write(
            "| rank | experiment | checkpoint | status | samples | AEE | AAE | "
            "PE1 | PE2 | PE3/outlier | SOPs(G) | firing |\n"
        )
        handle.write("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for rank, row in enumerate(rows, 1):
            handle.write(
                f"| {rank} | `{row['name']}` | `{row['checkpoint']}` | `{row['status']}` | {row['samples']} | "
                f"{fmt(row['AEE'])} | {fmt(row['AAE'])} | {fmt(row['PE1'])} | {fmt(row['PE2'])} | "
                f"{fmt(row['outlier'])} | {fmt(row['SOPs_G'])} | {fmt(row['firing'], 5)} |\n"
            )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
