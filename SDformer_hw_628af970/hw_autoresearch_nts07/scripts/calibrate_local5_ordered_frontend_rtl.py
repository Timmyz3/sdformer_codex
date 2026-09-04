#!/usr/bin/env python3
"""用旧 Local5 五 bank RTL 日志校准 ordered frontend 周期边界。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


GROUP_RE = re.compile(r"^GROUP (?P<body>.+)$")
FIELD_RE = re.compile(r"(?P<key>[a-zA-Z0-9_]+)=(?P<value>-?\d+)")
V2_CONTROL = 4
V2_RELATION = 450
V2_BACKEND_FIXED = 1 + 15 + 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(path: Path) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        match = GROUP_RE.match(line)
        if match:
            row = {
                item.group("key"): int(item.group("value"))
                for item in FIELD_RE.finditer(match.group("body"))
            }
            required = {"group", "cycles", "active", "terms", "term_stall"}
            if not required.issubset(row):
                raise ValueError(f"GROUP 行缺少字段: {path}")
            rows.append(row)
    if len(rows) != 100 or len({row["group"] for row in rows}) != 100:
        raise ValueError(f"RTL 校准日志必须覆盖100个唯一group: {path}")
    return rows


def work(row: dict[str, int]) -> int:
    return row["active"] + row["terms"] + row["term_stall"]


def residual(row: dict[str, int]) -> int:
    return row["cycles"] - work(row)


def summary(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def model_metrics(
    rows: list[dict[str, int]], *, fixed: int, mode: str
) -> dict[str, object]:
    actual = np.asarray([row["cycles"] for row in rows], dtype=np.float64)
    ordered_work = np.asarray([work(row) for row in rows], dtype=np.float64)
    if mode == "sequential":
        predicted = fixed + ordered_work
    elif mode == "v2_max_overlap":
        predicted = V2_CONTROL + np.maximum(
            V2_RELATION, V2_BACKEND_FIXED + ordered_work
        )
    else:
        raise ValueError(mode)
    error = predicted - actual
    absolute = np.abs(error)
    return {
        "actual_cycles": int(actual.sum()),
        "predicted_cycles": int(predicted.sum()),
        "mean_signed_error": float(error.mean()),
        "mae": float(absolute.mean()),
        "p95_absolute_error": float(np.percentile(absolute, 95)),
        "max_absolute_error": float(absolute.max()),
        "mean_relative_error": float(np.mean(error / actual)),
    }


def build_report(paths: dict[str, Path]) -> dict[str, object]:
    rows = {name: parse_log(path) for name, path in paths.items()}
    calibration_rows = rows["cal_direct"] + rows["cal_gasr"]
    calibration_residual = np.asarray(
        [residual(row) for row in calibration_rows], dtype=np.int64
    )
    fixed = int(np.median(calibration_residual))
    heldout = {}
    for name in ("heldout_direct", "heldout_gasr"):
        heldout[name] = {
            "residual": summary(
                np.asarray([residual(row) for row in rows[name]], dtype=np.float64)
            ),
            "sequential": model_metrics(rows[name], fixed=fixed, mode="sequential"),
            "v2_max_overlap": model_metrics(
                rows[name], fixed=fixed, mode="v2_max_overlap"
            ),
        }
    actual_ratio = (
        heldout["heldout_direct"]["sequential"]["actual_cycles"]
        / heldout["heldout_gasr"]["sequential"]["actual_cycles"]
    )
    sequential_ratio = (
        heldout["heldout_direct"]["sequential"]["predicted_cycles"]
        / heldout["heldout_gasr"]["sequential"]["predicted_cycles"]
    )
    overlap_ratio = (
        heldout["heldout_direct"]["v2_max_overlap"]["predicted_cycles"]
        / heldout["heldout_gasr"]["v2_max_overlap"]["predicted_cycles"]
    )
    return {
        "schema": "local5_ordered_frontend_rtl_calibration_v1",
        "status": "V2_MAX_OVERLAP_REQUIRES_REJECTION_OR_SEPARATE_RTL",
        "evidence": "[rtl校准]，旧post-G0 profile100，不是新joint-head正式profile",
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "calibration": {
            "rows": len(calibration_rows),
            "fixed_cycles_median": fixed,
            "residual": summary(calibration_residual),
            "definition": "cycles-active_sources-terms-term_stall",
        },
        "heldout": heldout,
        "heldout_speedup": {
            "actual_direct_over_gasr": actual_ratio,
            "sequential_model": sequential_ratio,
            "v2_max_overlap_model": overlap_ratio,
        },
        "decision": {
            "v2_prereg": "INVALIDATE_BEFORE_FORMAL_PROFILE",
            "v3_primary_boundary": "calibrated sequential fixed + active + terms + term_stall",
            "ideal_overlap": "only a separate optimistic sensitivity; cannot drive promotion",
        },
        "limits": [
            "校准日志是旧post-G0 100组，不是同窗全head正式profile。",
            "固定项由20260804日志拟合，20260805 bb1e4日志只作held-out。",
            "该校准证明当前集成相序，不证明未来FCSR重排绝不可能实现重叠。",
            "GASR2C-P跨head preserve、最终readout和scalar serializer仍未在此校准。",
        ],
    }


def render_markdown(report: dict[str, object]) -> str:
    cal = report["calibration"]
    heldout = report["heldout"]
    speed = report["heldout_speedup"]
    lines = [
        "# Local5 Ordered 前端 RTL 周期校准",
        "",
        "## 结论",
        "",
        "旧五 bank RTL 的当前集成相序符合 `固定relation/frontier成本 + active descriptor + term + stall`，",
        "不支持 v2 用于晋级的 `max(450, backend)` 理想重叠。v2 必须在正式 joint-head profile 前作废；",
        "v3 以 held-out 校准过的串行边界为主，理想重叠只能作为敏感性上界。",
        "",
        "## 校准",
        "",
        f"20260804 Direct+GASR 共 `{cal['rows']}` 组，`cycles-active-terms-term_stall` 的",
        f"中位固定项为 `{cal['fixed_cycles_median']}` 拍。",
        "",
        "## Held-out 结果",
        "",
        "| 路径 | 模型 | MAE | p95绝对误差 | 均值有符号误差 |",
        "|---|---|---:|---:|---:|",
    ]
    for path_name, label in (
        ("heldout_direct", "Direct"),
        ("heldout_gasr", "GASR"),
    ):
        for model_name, model_label in (
            ("sequential", "串行校准"),
            ("v2_max_overlap", "v2 max重叠"),
        ):
            row = heldout[path_name][model_name]
            lines.append(
                f"| {label} | {model_label} | {row['mae']:.2f} | "
                f"{row['p95_absolute_error']:.2f} | {row['mean_signed_error']:.2f} |"
            )
    lines += [
        "",
        "聚合 Direct/GASR 周期比：",
        "",
        f"- RTL：`{speed['actual_direct_over_gasr']:.6f}x`；",
        f"- 串行校准模型：`{speed['sequential_model']:.6f}x`；",
        f"- v2 max重叠模型：`{speed['v2_max_overlap_model']:.6f}x`。",
        "",
        "## 证据边界",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limits"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = {
        "cal_direct": root / "results/local5_qgasr2c_fivebank_postg0_rtl_20260804/direct_profile100.log",
        "cal_gasr": root / "results/local5_qgasr2c_fivebank_postg0_rtl_20260804/qgasr_profile100.log",
        "heldout_direct": root / "results/local5_bb1e4_qgasr2c_fivebank_postg0_rtl_20260805/direct_profile100.log",
        "heldout_gasr": root / "results/local5_bb1e4_qgasr2c_fivebank_postg0_rtl_20260805/qgasr_profile100.log",
    }
    report = build_report(paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    print("PASS Local5 ordered frontend RTL calibration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
