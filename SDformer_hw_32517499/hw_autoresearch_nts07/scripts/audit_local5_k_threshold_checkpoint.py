#!/usr/bin/env python3
"""CPU审计Local5 fullres checkpoint中12个attention K-ATLIF标量阈值。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
BASELINE = REPO / "third_party/SDformerFlow"
DEFAULT_CHECKPOINT = (
    EXP
    / "results/dsec_fullres_paper_w15_h66d_local5_ep29_ft30_bs2_20260728"
    / "checkpoint_epoch29.pth"
)
DEFAULT_OUT = ROOT / "results/local5_k_threshold_audit_20260801"


def load_checkpoint(path: Path) -> torch.nn.Module:
    sys.path.insert(0, str(BASELINE))
    import models
    import models.STSwinNet_SNN as stsnn

    overlay_models = str(EXP / "overlay/models")
    overlay_stsnn = str(EXP / "overlay/models/STSwinNet_SNN")
    if overlay_models not in list(models.__path__):
        models.__path__.append(overlay_models)
    if overlay_stsnn not in list(stsnn.__path__):
        stsnn.__path__.append(overlay_stsnn)
    from models.STSwinNet_SNN.bsa_attention import (
        register_shiftmax_pickle_compat,
    )

    register_shiftmax_pickle_compat()
    model = torch.load(path, map_location="cpu")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("checkpoint不是完整torch.nn.Module")
    return model


def audit(path: Path) -> dict[str, object]:
    model = load_checkpoint(path)
    entries = []
    suffix = ".attn.sn_k.spiking_neuron.thresh"
    for name, value in model.state_dict().items():
        if not name.endswith(suffix):
            continue
        flat = value.detach().cpu().reshape(-1)
        if flat.numel() != 1:
            raise ValueError(f"K-ATLIF阈值不是标量: {name} shape={tuple(value.shape)}")
        threshold = float(flat.item())
        entries.append(
            {
                "name": name,
                "shape": list(value.shape),
                "threshold": threshold,
                "abs_deviation_from_one": abs(threshold - 1.0),
            }
        )
    if len(entries) != 12:
        raise RuntimeError(f"期望12个attention K阈值，实际{len(entries)}")
    return {
        "schema": "local5_k_threshold_checkpoint_audit_v1",
        "checkpoint": str(path.resolve()),
        "count": len(entries),
        "all_scalar": True,
        "minimum": min(row["threshold"] for row in entries),
        "maximum": max(row["threshold"] for row in entries),
        "max_abs_deviation_from_one": max(
            row["abs_deviation_from_one"] for row in entries
        ),
        "exactly_one_count": sum(row["threshold"] == 1.0 for row in entries),
        "entries": entries,
        "contract": (
            "K=value_support*theta_block；theta是每attention block标量，"
            "可预折叠到projection weight"
        ),
    }


def markdown(report: dict[str, object]) -> str:
    lines = [
        "# Local5 K-ATLIF 阈值 Checkpoint 审计",
        "",
        "> 日期：2026-08-01  ",
        "> 证据等级：`[checkpoint-static]`；不等于运行时 trace 或定点等价。",
        "",
        "## 结论",
        "",
        f"- attention block 数：{report['count']}；",
        f"- 全部阈值为标量：{'是' if report['all_scalar'] else '否'}；",
        f"- 阈值范围：{report['minimum']:.10f} 至 {report['maximum']:.10f}；",
        f"- 与1.0最大绝对偏差：{report['max_abs_deviation_from_one']:.10e}；",
        f"- 精确等于1.0：{report['exactly_one_count']}/12。",
        "",
        "软件 value 路径使用 `K = event * theta_block`。因此 projection 可重写为",
        "`gate * event * (theta_block * weight)`，无需为每个事件传输多位 K 幅度。",
        "但只要 theta 不严格等于1，旧的 `k_orig == k_event` bit-exact 声明就不成立。",
        "",
        "## 逐 Block 数值",
        "",
        "| 参数 | theta | |theta-1| |",
        "|---|---:|---:|",
    ]
    for row in report["entries"]:
        lines.append(
            f"| `{row['name']}` | {row['threshold']:.10f} | "
            f"{row['abs_deviation_from_one']:.10e} |"
        )
    lines += [
        "",
        "## 后续门槛",
        "",
        "1. post-G0 runtime profile 验证每次 callback 只有一个非零幅度；",
        "2. 比较 theta 保留、theta 定点折叠和 theta=1 三种 valid825；",
        "3. RTL weight loader 加入每 block theta descriptor 或离线预折叠；",
        "4. 在上述完成前，Local5 projection 只具事件支持集等价，不具数值 exact。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = audit(args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
