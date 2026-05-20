"""汇总 rapid_screen 训练日志里的 ATLIF 三值发放是否正常。"""

from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path
from typing import Any


SUMMARY_RE = re.compile(r"\[H9\] ATLIFTernaryPSN summary: (\{.*\})")


def parse_summary(log_path: Path) -> dict[str, Any] | None:
    last: dict[str, Any] | None = None
    if not log_path.exists():
        return None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = SUMMARY_RE.search(line)
        if match:
            last = ast.literal_eval(match.group(1))
    return last


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rapid_dir", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for train_log in sorted((args.rapid_dir / "runs").glob("*/train.log")):
        summary = parse_summary(train_log)
        if not summary:
            continue
        row = {"run": train_log.parent.name}
        for key in (
            "num_modules",
            "threshold_mean",
            "threshold_min",
            "threshold_max",
            "activity_mean",
            "pos_mean",
            "neg_mean",
            "ternary_activity_mean",
            "ternary_pos_mean",
            "ternary_neg_mean",
            "binary_activity_mean",
            "target_rate_mean",
            "negative_scale_mean",
            "symmetric_target_rate_modules",
            "center_bias_modules",
        ):
            row[key] = summary.get(key, "")
        rows.append(row)

    if not rows:
        print("没有找到 ATLIFTernaryPSN summary。")
        return 1

    out_csv = args.rapid_dir / "neuron_balance_summary.csv"
    fields = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"写入: {out_csv}")
    for row in rows:
        print(
            f"{row['run']}: ternary={row.get('ternary_activity_mean')} "
            f"pos={row.get('ternary_pos_mean')} neg={row.get('ternary_neg_mean')} "
            f"binary={row.get('binary_activity_mean')} thresh={row.get('threshold_mean')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
