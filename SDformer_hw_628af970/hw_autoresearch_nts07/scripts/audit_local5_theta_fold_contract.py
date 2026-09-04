#!/usr/bin/env python3
"""CPU审计Local5 K-ATLIF theta是否改变dyadic INT8投影合同。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from audit_local5_k_threshold_checkpoint import load_checkpoint
from profile_local5_hardware_features import quantize_projection_weight_dyadic


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
DEFAULT_CHECKPOINT = (
    EXP
    / "results/dsec_fullres_paper_w15_h66d_local5_ep29_ft30_bs2_20260728"
    / "checkpoint_epoch29.pth"
)
DEFAULT_OUTPUT = ROOT / "results/local5_theta_fold_contract_audit_20260805"
ATTN_SUFFIX = ".attn.proj.weight"
THRESH_SUFFIX = ".attn.sn_k.spiking_neuron.thresh"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(checkpoint: Path) -> dict[str, Any]:
    model = load_checkpoint(checkpoint)
    state = model.state_dict()
    weights = {
        name[: -len(ATTN_SUFFIX)]: value.detach().cpu()
        for name, value in state.items()
        if name.endswith(ATTN_SUFFIX)
    }
    thresholds = {
        name[: -len(THRESH_SUFFIX)]: value.detach().cpu().reshape(-1)
        for name, value in state.items()
        if name.endswith(THRESH_SUFFIX)
    }
    prefixes = sorted(set(weights) & set(thresholds))
    if len(prefixes) != 12:
        raise RuntimeError(
            "期望12个同时具备K阈值与proj.weight的attention block，"
            f"实际{len(prefixes)}"
        )
    if set(weights) != set(thresholds):
        raise RuntimeError(
            "K阈值与projection权重block集合不一致: "
            f"weight_only={sorted(set(weights) - set(thresholds))}, "
            f"threshold_only={sorted(set(thresholds) - set(weights))}"
        )

    rows: list[dict[str, Any]] = []
    total_int8 = 0
    total_int8_mismatch = 0
    total_scale = 0
    total_scale_mismatch = 0
    for prefix in prefixes:
        threshold_flat = thresholds[prefix]
        if threshold_flat.numel() != 1:
            raise RuntimeError(f"K阈值不是标量: {prefix}")
        theta = float(threshold_flat.item())
        weight = weights[prefix].float()
        raw_int8, raw_scale = quantize_projection_weight_dyadic(weight)
        folded_int8, folded_scale = quantize_projection_weight_dyadic(
            weight * theta
        )
        int8_mismatch = int(np.count_nonzero(raw_int8 != folded_int8))
        scale_mismatch = int(np.count_nonzero(raw_scale != folded_scale))
        int8_count = int(raw_int8.size)
        scale_count = int(raw_scale.size)
        total_int8 += int8_count
        total_int8_mismatch += int8_mismatch
        total_scale += scale_count
        total_scale_mismatch += scale_mismatch
        rows.append(
            {
                "block": prefix,
                "theta": theta,
                "abs_theta_minus_one": abs(theta - 1.0),
                "weight_shape": list(weight.shape),
                "float_effective_weight_max_abs_delta": float(
                    (weight * theta - weight).abs().max().item()
                ),
                "int8_entries": int8_count,
                "int8_mismatch": int8_mismatch,
                "scale_entries": scale_count,
                "scale_exp2_mismatch": scale_mismatch,
                "integer_contract_equal": (
                    int8_mismatch == 0 and scale_mismatch == 0
                ),
            }
        )

    return {
        "schema": "local5_theta_fold_contract_audit_v1",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "blocks": len(rows),
        "quantization": (
            "per-output-channel symmetric INT8; scale=2^e; "
            "e=ceil(log2(max_abs/127)); RNE; clamp[-127,127]"
        ),
        "comparison": "quantize(W) versus quantize(theta_block * W)",
        "theta_min": min(row["theta"] for row in rows),
        "theta_max": max(row["theta"] for row in rows),
        "theta_max_abs_deviation_from_one": max(
            row["abs_theta_minus_one"] for row in rows
        ),
        "int8_entries": total_int8,
        "int8_mismatch": total_int8_mismatch,
        "scale_entries": total_scale,
        "scale_exp2_mismatch": total_scale_mismatch,
        "all_integer_contract_equal": (
            total_int8_mismatch == 0 and total_scale_mismatch == 0
        ),
        "float_contract_equal": all(row["theta"] == 1.0 for row in rows),
        "rows": rows,
        "interpretation": {
            "integer_equal": (
                "仅证明当前checkpoint在指定dyadic INT8量化合同下，"
                "省略theta不改变整数权重与指数；不证明浮点网络等价"
            ),
            "integer_not_equal": (
                "必须将theta先折叠到W再量化，或把theta编码进block descriptor"
            ),
        },
    }


def markdown(report: dict[str, Any]) -> str:
    integer_equal = bool(report["all_integer_contract_equal"])
    float_equal = bool(report["float_contract_equal"])
    lines = [
        "# Local5 theta 折叠与 Dyadic INT8 投影合同审计",
        "",
        "## 结论",
        "",
        f"- checkpoint：`{report['checkpoint']}`；",
        f"- checkpoint SHA256：`{report['checkpoint_sha256']}`；",
        f"- attention block：{report['blocks']}；",
        f"- theta 范围：{report['theta_min']:.10f} 至 {report['theta_max']:.10f}；",
        "- 与 1.0 最大绝对偏差："
        f"{report['theta_max_abs_deviation_from_one']:.10e}；",
        f"- INT8 权重不一致：{report['int8_mismatch']}/"
        f"{report['int8_entries']}；",
        f"- dyadic 指数不一致：{report['scale_exp2_mismatch']}/"
        f"{report['scale_entries']}；",
        f"- 指定整数部署合同等价：{'是' if integer_equal else '否'}；",
        f"- 浮点合同等价：{'是' if float_equal else '否'}。",
        "",
    ]
    if integer_equal:
        lines += [
            "本 checkpoint 下，`quantize(W)` 与 "
            "`quantize(theta_block × W)` 逐项相同。因此在当前指定的 "
            "dyadic INT8 部署合同中，可以不单独传输 theta。该结论只对当前 "
            "checkpoint SHA 和量化规则成立，不能外推到新 checkpoint，也不能称为 "
            "浮点 bit-exact。",
            "",
        ]
    else:
        lines += [
            "theta 已改变整数权重或 dyadic 指数。部署合同必须改为先折叠 theta "
            "再量化，或显式携带 theta descriptor；现有 `gate × event × W` "
            "投影向量不能作为软件数值闭环证据。",
            "",
        ]
    lines += [
        "## 逐 Block 结果",
        "",
        "| Block | theta | max |theta×W-W| | INT8 mismatch | scale mismatch | 整数合同相同 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| `{row['block']}` | {row['theta']:.10f} | "
            f"{row['float_effective_weight_max_abs_delta']:.6e} | "
            f"{row['int8_mismatch']}/{row['int8_entries']} | "
            f"{row['scale_exp2_mismatch']}/{row['scale_entries']} | "
            f"{'是' if row['integer_contract_equal'] else '否'} |"
        )
    lines += [
        "",
        "## 证据边界",
        "",
        "1. 结果绑定 checkpoint SHA，不允许跨 checkpoint 复用；",
        "2. 只覆盖 attention projection 的 dyadic INT8 权重和指数；",
        "3. 不覆盖 cross-head reduction、bias、no-running BN、requant、残差和 decoder；",
        "4. 新 Local5 rank-1 释放后必须原样复跑，才能冻结正式部署合同。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
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
    print(
        json.dumps(
            {
                "blocks": report["blocks"],
                "int8_mismatch": report["int8_mismatch"],
                "scale_exp2_mismatch": report["scale_exp2_mismatch"],
                "all_integer_contract_equal": report[
                    "all_integer_contract_equal"
                ],
                "float_contract_equal": report["float_contract_equal"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
