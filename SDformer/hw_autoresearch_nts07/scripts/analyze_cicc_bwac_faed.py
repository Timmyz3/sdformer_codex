#!/usr/bin/env python3
"""评估 CICC'26 BWAC 在 H67/Local5 FAED 权重流上的适用性。"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
DEFAULT_MOTION = ROOT / "results/gatestack_dctf96_real_trace_20260720/vectors"
DEFAULT_LOCAL = (
    EXP
    / "results/dsec_fullres_paper_w15_h66d_local5_ep29_ft30_bs2_20260728"
    / "checkpoint_epoch29.pth"
)
DEFAULT_OUT = ROOT / "results/cicc2026_bwac_faed_dse_20260801"
BLOCK_RE = re.compile(r"layers\.(\d+)\.swin_blocks\.(\d+)\.attn\.proj\.weight$")


def signed_bits(value: int) -> int:
    for width in range(1, 9):
        if -(1 << (width - 1)) <= value <= (1 << (width - 1)) - 1:
            return width
    raise ValueError(f"值超出INT8: {value}")


def load_memh_int8(path: Path) -> list[int]:
    values = []
    for item in path.read_text(encoding="utf-8").split():
        raw = int(item, 16)
        values.append(raw - 256 if raw >= 128 else raw)
    return values


def encode_groups(values: list[int], group: int) -> dict[str, object]:
    totals = {"raw8": 0, "minbw": 0, "bwac_bitmap": 0, "adaptive": 0}
    modes = {"raw8": 0, "minbw": 0, "bwac_bitmap": 0}
    width_hist: dict[int, int] = {}
    for offset in range(0, len(values), group):
        chunk = values[offset : offset + group]
        if not chunk:
            continue
        nonzero = [value for value in chunk if value != 0]
        width = max((signed_bits(value) for value in nonzero), default=1)
        width_hist[width] = width_hist.get(width, 0) + 1
        raw = 2 + 8 * len(chunk)
        minbw = 2 + 3 + width * len(chunk)
        bitmap = 2 + 3 + len(chunk) + width * len(nonzero)
        totals["raw8"] += raw
        totals["minbw"] += minbw
        totals["bwac_bitmap"] += bitmap
        options = {"raw8": raw, "minbw": minbw, "bwac_bitmap": bitmap}
        mode = min(options, key=options.get)
        modes[mode] += 1
        totals["adaptive"] += options[mode]
    baseline = 8 * len(values)
    return {
        "group": group,
        "weights": len(values),
        "zero_ratio": sum(value == 0 for value in values) / max(1, len(values)),
        "baseline_bits_without_mode": baseline,
        "bits": totals,
        "ratios_vs_fixed8": {
            key: value / baseline for key, value in totals.items()
        },
        "adaptive_mode_groups": modes,
        "required_width_histogram": {str(k): v for k, v in sorted(width_hist.items())},
    }


def motion_weights(root: Path) -> tuple[list[int], list[dict[str, object]]]:
    all_values = []
    stages = []
    for stage in range(4):
        path = root / f"s{stage}/projection_weights_int8.memh"
        values = load_memh_int8(path)
        all_values.extend(values)
        stages.append(
            {
                "stage": stage,
                "weights": len(values),
                "zero_ratio": sum(value == 0 for value in values) / len(values),
                "minimum": min(values),
                "maximum": max(values),
            }
        )
    return all_values, stages


def load_model(path: Path) -> torch.nn.Module:
    sys.path.insert(0, str(REPO / "third_party/SDformerFlow"))
    import models
    import models.STSwinNet_SNN as stsnn

    overlay = EXP / "overlay/models"
    overlay_stsnn = overlay / "STSwinNet_SNN"
    if str(overlay) not in list(models.__path__):
        models.__path__.append(str(overlay))
    if str(overlay_stsnn) not in list(stsnn.__path__):
        stsnn.__path__.append(str(overlay_stsnn))
    from models.STSwinNet_SNN.bsa_attention import register_shiftmax_pickle_compat

    register_shiftmax_pickle_compat()
    model = torch.load(path, map_location="cpu")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("checkpoint不是完整torch.nn.Module")
    return model


def quantize_per_output(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight = weight.detach().double()
    absmax = weight.abs().amax(dim=1).clamp_min(torch.finfo(torch.float64).tiny)
    scale = absmax / 127.0
    code = torch.round(weight / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return code, scale


def quantize_pow2_per_output(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight = weight.detach().double()
    absmax = weight.abs().amax(dim=1).clamp_min(torch.finfo(torch.float64).tiny)
    exponent = torch.ceil(torch.log2(absmax / 127.0))
    scale = torch.pow(torch.tensor(2.0, dtype=torch.float64), exponent)
    code = torch.round(weight / scale[:, None]).clamp(-127, 127).to(torch.int8)
    return code, exponent.to(torch.int16)


def local_weights_and_theta(path: Path) -> dict[str, object]:
    model = load_model(path)
    state = model.state_dict()
    blocks = []
    all_codes: list[int] = []
    float_scale_code_mismatch = 0
    pow2_code_mismatch = 0
    pow2_exp_mismatch = 0
    for name, weight in state.items():
        match = BLOCK_RE.search(name)
        if not match:
            continue
        prefix = name[: -len("proj.weight")]
        theta_name = prefix + "sn_k.spiking_neuron.thresh"
        theta = float(state[theta_name].reshape(-1)[0].item())
        base_code, base_scale = quantize_per_output(weight)
        folded_code, folded_scale = quantize_per_output(weight * theta)
        base_pow2, base_exp = quantize_pow2_per_output(weight)
        folded_pow2, folded_exp = quantize_pow2_per_output(weight * theta)
        float_mismatch = int((base_code != folded_code).sum().item())
        pow2_mismatch = int((base_pow2 != folded_pow2).sum().item())
        exp_mismatch = int((base_exp != folded_exp).sum().item())
        float_scale_ratio_error = float(
            ((folded_scale / base_scale) - theta).abs().amax().item()
        )
        all_codes.extend(int(value) for value in base_pow2.reshape(-1).tolist())
        blocks.append(
            {
                "stage": int(match.group(1)),
                "block": int(match.group(2)),
                "shape": list(weight.shape),
                "theta": theta,
                "float_scale_code_mismatch": float_mismatch,
                "float_scale_ratio_error": float_scale_ratio_error,
                "pow2_code_mismatch": pow2_mismatch,
                "pow2_exponent_mismatch": exp_mismatch,
            }
        )
        float_scale_code_mismatch += float_mismatch
        pow2_code_mismatch += pow2_mismatch
        pow2_exp_mismatch += exp_mismatch
    if len(blocks) != 12:
        raise RuntimeError(f"期望12个attention projection，实际{len(blocks)}")
    return {
        "blocks": blocks,
        "codes": all_codes,
        "code_count": len(all_codes),
        "float_scale_code_mismatch": float_scale_code_mismatch,
        "pow2_code_mismatch": pow2_code_mismatch,
        "pow2_exponent_mismatch": pow2_exp_mismatch,
    }


def evaluate(motion_root: Path, local_checkpoint: Path) -> dict[str, object]:
    motion, motion_stages = motion_weights(motion_root)
    local = local_weights_and_theta(local_checkpoint)
    groups = (4, 8, 16, 32, 64)
    return {
        "schema": "cicc2026_bwac_faed_dse_v1",
        "evidence": "checkpoint-static/open-vector/model",
        "motion": {
            "source": str(motion_root.resolve()),
            "stages": motion_stages,
            "group_dse": [encode_groups(motion, group) for group in groups],
        },
        "local5": {
            "source": str(local_checkpoint.resolve()),
            "blocks": local["blocks"],
            "theta_fold": {
                "code_count": local["code_count"],
                "per_output_float_scale_code_mismatch": local[
                    "float_scale_code_mismatch"
                ],
                "per_output_float_scale_code_mismatch_ratio": local[
                    "float_scale_code_mismatch"
                ]
                / local["code_count"],
                "per_output_pow2_code_mismatch": local["pow2_code_mismatch"],
                "per_output_pow2_code_mismatch_ratio": local[
                    "pow2_code_mismatch"
                ]
                / local["code_count"],
                "per_output_pow2_exponent_mismatch": local[
                    "pow2_exponent_mismatch"
                ],
            },
            "group_dse": [
                encode_groups(local["codes"], group) for group in groups
            ],
        },
    }


def best_adaptive(rows: list[dict[str, object]]) -> dict[str, object]:
    return min(rows, key=lambda row: row["ratios_vs_fixed8"]["adaptive"])


def markdown(report: dict[str, object]) -> str:
    motion_best = best_adaptive(report["motion"]["group_dse"])
    local_best = best_adaptive(report["local5"]["group_dse"])
    fold = report["local5"]["theta_fold"]
    lines = [
        "# CICC 2026 BWAC 与 FAED 权重流 DSE",
        "",
        "> 日期：2026-08-01  ",
        "> 证据等级：`[open-vector]`、`[checkpoint-static]`、`[模型]`；不是部署精度或ASIC PPA。",
        "",
        "## 结论",
        "",
        f"- Motion 最优自适应分组为 G={motion_best['group']}，位数为固定INT8的 "
        f"{motion_best['ratios_vs_fixed8']['adaptive']:.3%}；",
        f"- Local5 最优自适应分组为 G={local_best['group']}，位数为固定INT8的 "
        f"{local_best['ratios_vs_fixed8']['adaptive']:.3%}；",
        f"- Local5 若先折叠theta再重新量化，逐输出浮点scale码差 "
        f"{fold['per_output_float_scale_code_mismatch']}/{fold['code_count']}，"
        f"逐输出2幂scale码差 {fold['per_output_pow2_code_mismatch']}/{fold['code_count']}，"
        f"但2幂scale指数差 {fold['per_output_pow2_exponent_mismatch']}。",
        "",
        "CICC 原版 `MinBW + nonzero bitmap` 不适合当前接近稠密、逐输出通道量化的投影权重；"
        "bitmap 开销会抵消变位宽收益。可保留的是三模式自适应格式：raw8、MinBW、"
        "MinBW+bitmap 每组择优。其收益较小，只能作为内存子机制，不应单列主贡献。",
        "",
        "## 分组结果",
        "",
        "| 线 | G | 零值率 | MinBW/INT8 | CICC bitmap/INT8 | adaptive/INT8 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for line in ("motion", "local5"):
        for row in report[line]["group_dse"]:
            ratio = row["ratios_vs_fixed8"]
            lines.append(
                f"| {line} | {row['group']} | {row['zero_ratio']:.3%} | "
                f"{ratio['minbw']:.3%} | {ratio['bwac_bitmap']:.3%} | "
                f"{ratio['adaptive']:.3%} |"
            )
    lines += [
        "",
        "## FAED 位宽决策",
        "",
        "正标量 theta 可并入逐输出通道 scale：theta 不需要进入每个事件，也不需要扩大"
        "INT8权重码。若选择“先折叠再重新量化”，有限精度舍入会造成极少量码差；因此"
        "bit-exact FAED 应固定为“INT8 W码不变，theta乘入scale descriptor”，而不是重新"
        "生成 `W_theta`。本次2幂scale的指数没有变化，但仍观测到少量舍入码差，进一步"
        "说明两种运算次序不能混称等价。",
        "",
        "该结论仍不等于 valid825 数值签核。最终必须冻结 theta、权重scale和舍入顺序。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion-root", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--local-checkpoint", type=Path, default=DEFAULT_LOCAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = evaluate(args.motion_root, args.local_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
