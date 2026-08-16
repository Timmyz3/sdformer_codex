#!/usr/bin/env python3
"""审计H67/H68的GCM-P投影、BN折叠与最小int8部署合同。"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
sys.path[:0] = [
    str(EXP / "overlay"),
    str(REPO / "third_party/SDformerFlow"),
    str(REPO),
]

from models.STSwinNet_SNN.bsa_attention import register_shiftmax_pickle_compat  # noqa: E402


DEFAULT_CASES = {
    "H67": EXP / (
        "results/h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_"
        "full30_20260711_setsid/checkpoint_epoch19.pth"
    ),
    "H68": EXP / (
        "results/h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30_"
        "bs8_full30_20260711_setsid/checkpoint_epoch19.pth"
    ),
}
EXPECTED_DIMS = {0: 96, 1: 192, 2: 384, 3: 768}
EXPECTED_BLOCKS = {0: 2, 1: 2, 2: 6, 3: 2}
BLOCK_PATTERN = re.compile(r"layers\.(\d+)\.swin_blocks\.(\d+)\.attn$")
GATE_FRAC_BITS = 7
GATE_MAX_CODE = 256
INT8_MAX = 127
INT32_MAX = (1 << 31) - 1


def fold_linear_batch_norm(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """将eval态BatchNorm精确折叠进前置Linear。"""

    if bias is None:
        bias = torch.zeros_like(running_mean)
    alpha = gamma / torch.sqrt(running_var + float(eps))
    return weight * alpha[:, None], (bias - running_mean) * alpha + beta


def symmetric_int8_quantize(
    weight: torch.Tensor, *, per_output_channel: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回int8码和每个输出通道或全tensor的浮点scale。"""

    if per_output_channel:
        absmax = weight.abs().amax(dim=1).clamp_min(torch.finfo(torch.float32).tiny)
        scale = absmax / INT8_MAX
        code = torch.round(weight / scale[:, None]).clamp(-INT8_MAX, INT8_MAX)
    else:
        scale = weight.abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
        scale = scale / INT8_MAX
        code = torch.round(weight / scale).clamp(-INT8_MAX, INT8_MAX)
    return code.to(torch.int8), scale.float()


def quantized_projection(
    gate_code: torch.Tensor,
    weight_code: torch.Tensor,
    weight_scale: torch.Tensor,
    folded_bias: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """以Q1.7输入、int8权重和int32偏置执行投影。"""

    accumulator_scale = weight_scale / float(1 << GATE_FRAC_BITS)
    if accumulator_scale.ndim == 0:
        bias_code = torch.round(folded_bias / accumulator_scale)
        restored_scale = accumulator_scale
    else:
        bias_code = torch.round(folded_bias / accumulator_scale)
        restored_scale = accumulator_scale[None, :]
    bias_code = bias_code.clamp(-INT32_MAX, INT32_MAX).to(torch.int64)
    accumulator = gate_code.to(torch.int64) @ weight_code.to(torch.int64).T
    accumulator = accumulator + bias_code[None, :]
    restored = accumulator.float() * restored_scale
    return restored, {
        "bias_int32_clip_count": int((bias_code.abs() >= INT32_MAX).sum().item()),
        "accumulator_absmax": int(accumulator.abs().amax().item()),
        "accumulator_int32_margin": float(INT32_MAX / max(1, int(accumulator.abs().amax().item()))),
    }


def tensor_error(reference: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    error = actual - reference
    reference_energy = torch.sum(reference.double().square()).item()
    error_energy = torch.sum(error.double().square()).item()
    return {
        "mae": float(error.abs().mean().item()),
        "rmse": float(torch.sqrt(error.float().square().mean()).item()),
        "max_abs": float(error.abs().amax().item()),
        "relative_l2": float(math.sqrt(error_energy / reference_energy)) if reference_energy else 0.0,
    }


def quantization_audit(
    folded_weight: torch.Tensor,
    folded_bias: torch.Tensor,
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    channels = int(folded_weight.shape[1])
    densities = (0.02, 0.10, 0.50)
    result: dict[str, Any] = {}
    for name, per_channel in (("per_tensor_int8", False), ("per_output_channel_int8", True)):
        weight_code, scale = symmetric_int8_quantize(
            folded_weight, per_output_channel=per_channel
        )
        restored_weight = weight_code.float() * (
            scale[:, None] if scale.ndim else scale
        )
        accumulator_scale = scale / float(1 << GATE_FRAC_BITS)
        bias_code = torch.round(folded_bias / accumulator_scale).to(torch.int64)
        theoretical_abs_bound = (
            GATE_MAX_CODE * weight_code.to(torch.int64).abs().sum(dim=1)
            + bias_code.abs()
        )
        output_cases = []
        max_accumulator = 0
        min_margin = float("inf")
        bias_clip_count = 0
        for density in densities:
            event = torch.rand((samples, channels), generator=generator) < density
            gate = torch.randint(
                0,
                GATE_MAX_CODE + 1,
                (samples, 1),
                generator=generator,
                dtype=torch.int64,
            )
            gate_code = event.to(torch.int64) * gate
            reference = (
                gate_code.float() / float(1 << GATE_FRAC_BITS)
            ) @ folded_weight.T + folded_bias[None, :]
            actual, bounds = quantized_projection(
                gate_code, weight_code, scale, folded_bias
            )
            output_cases.append({"event_density": density, **tensor_error(reference, actual)})
            max_accumulator = max(max_accumulator, int(bounds["accumulator_absmax"]))
            min_margin = min(min_margin, float(bounds["accumulator_int32_margin"]))
            bias_clip_count += int(bounds["bias_int32_clip_count"])
        result[name] = {
            "weight": tensor_error(folded_weight, restored_weight),
            "scale_min": float(scale.amin().item()),
            "scale_max": float(scale.amax().item()),
            "weight_code_min": int(weight_code.min().item()),
            "weight_code_max": int(weight_code.max().item()),
            "scale_entries": int(scale.numel()),
            "scale_storage_bytes_fp32": int(scale.numel() * 4),
            "theoretical_accumulator_abs_bound": int(theoretical_abs_bound.max().item()),
            "theoretical_accumulator_int32_margin": float(
                INT32_MAX / max(1, int(theoretical_abs_bound.max().item()))
            ),
            "synthetic_outputs": output_cases,
            "synthetic_accumulator_absmax": max_accumulator,
            "synthetic_accumulator_int32_margin_min": min_margin,
            "bias_int32_clip_count": bias_clip_count,
        }
    return result


def analyze_case(name: str, checkpoint: Path, *, seed: int, samples: int) -> dict[str, Any]:
    register_shiftmax_pickle_compat()
    model = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.eval()
    rows = []
    for module_name, module in model.named_modules():
        match = BLOCK_PATTERN.search(module_name)
        if not match or module.__class__.__name__ != "Spiking_QK_WindowAttention3D":
            continue
        stage, block = map(int, match.groups())
        threshold = module.sn_k.spiking_neuron.thresh.detach().float().reshape(-1)
        norm = module.proj_bn.norm_layer
        folded_weight, folded_bias = fold_linear_batch_norm(
            module.proj.weight.detach().float(),
            module.proj.bias.detach().float() if module.proj.bias is not None else None,
            norm.weight.detach().float(),
            norm.bias.detach().float(),
            norm.running_mean.detach().float(),
            norm.running_var.detach().float(),
            float(norm.eps),
        )
        threshold_scalar = float(threshold.item()) if threshold.numel() == 1 else None
        effective_weight = folded_weight * (
            threshold_scalar if threshold_scalar is not None else 1.0
        )
        rows.append(
            {
                "stage": stage,
                "block": block,
                "module": module_name,
                "channels": int(module.dim),
                "heads": int(module.num_heads),
                "head_dim": int(module.dim // module.num_heads),
                "k_threshold_entries": int(threshold.numel()),
                "k_threshold_min": float(threshold.min().item()),
                "k_threshold_max": float(threshold.max().item()),
                "k_threshold_is_identity": bool(torch.equal(threshold, torch.ones_like(threshold))),
                "projection_bias": module.proj.bias is not None,
                "projection_bn_type": norm.__class__.__name__,
                "projection_bn_eps": float(norm.eps),
                "folded_weight_min": float(effective_weight.min().item()),
                "folded_weight_max": float(effective_weight.max().item()),
                "folded_bias_min": float(folded_bias.min().item()),
                "folded_bias_max": float(folded_bias.max().item()),
                "quantization": quantization_audit(
                    effective_weight,
                    folded_bias,
                    seed=seed + stage * 100 + block,
                    samples=samples,
                ),
            }
        )
    del model
    gc.collect()
    rows.sort(key=lambda row: (row["stage"], row["block"]))

    stage_counts = {stage: sum(row["stage"] == stage for row in rows) for stage in EXPECTED_BLOCKS}
    structural_checks = {
        "attention_blocks_12": len(rows) == 12,
        "stage_block_counts": stage_counts == EXPECTED_BLOCKS,
        "stage_dimensions": all(row["channels"] == EXPECTED_DIMS[row["stage"]] for row in rows),
        "head_dim_32": all(row["head_dim"] == 32 for row in rows),
        "scalar_k_threshold": all(row["k_threshold_entries"] == 1 for row in rows),
        "identity_k_threshold": all(row["k_threshold_is_identity"] for row in rows),
        "batch_norm_2d": all(row["projection_bn_type"] == "BatchNorm2d" for row in rows),
        "int32_bias_no_clip": all(
            q["bias_int32_clip_count"] == 0
            for row in rows
            for q in row["quantization"].values()
        ),
        "int32_theoretical_bound_safe": all(
            q["theoretical_accumulator_abs_bound"] <= INT32_MAX
            for row in rows
            for q in row["quantization"].values()
        ),
    }
    aggregate = {}
    for mode in ("per_tensor_int8", "per_output_channel_int8"):
        aggregate[mode] = {
            "weight_relative_l2_max": max(
                row["quantization"][mode]["weight"]["relative_l2"] for row in rows
            ),
            "synthetic_output_relative_l2_max": max(
                case["relative_l2"]
                for row in rows
                for case in row["quantization"][mode]["synthetic_outputs"]
            ),
            "synthetic_accumulator_absmax": max(
                row["quantization"][mode]["synthetic_accumulator_absmax"] for row in rows
            ),
            "synthetic_accumulator_int32_margin_min": min(
                row["quantization"][mode]["synthetic_accumulator_int32_margin_min"] for row in rows
            ),
            "theoretical_accumulator_abs_bound_max": max(
                row["quantization"][mode]["theoretical_accumulator_abs_bound"] for row in rows
            ),
            "theoretical_accumulator_int32_margin_min": min(
                row["quantization"][mode]["theoretical_accumulator_int32_margin"] for row in rows
            ),
            "scale_entries": sum(
                row["quantization"][mode]["scale_entries"] for row in rows
            ),
            "scale_storage_bytes_fp32": sum(
                row["quantization"][mode]["scale_storage_bytes_fp32"] for row in rows
            ),
        }
    return {
        "model": name,
        "checkpoint": str(checkpoint),
        "structural_checks": structural_checks,
        "structural_pass": all(structural_checks.values()),
        "stage_block_counts": stage_counts,
        "aggregate": aggregate,
        "blocks": rows,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# GCM-P投影阈值与最小定点合同",
        "",
        f"**日期**：2026-07-13  ",
        f"**静态结构审计**：{'通过' if payload['structural_pass'] else '失败'}  ",
        "**网络精度签核**：未完成，仍需真实valid825投影量化推理",
        "",
        "## 1. 结论",
        "",
        "1. H67与H68的12个attention block均使用标量K事件阈值，且checkpoint中全部精确为1.0。GCM-P无需为token、时间或channel携带事件幅值；该阈值折叠在当前两条线中是恒等操作。",
        "2. 每个投影都是带bias的C乘C Linear，后接eval态BatchNorm2d。BN可静态折叠为一组有效权重和有效偏置，运行时不需要独立BN除法或平方根。",
        "3. 本审计只证明checkpoint结构、BN折叠公式、int8权重与int32累加的数值代理。它不等于valid825网络精度，也不允许据此冻结最终位宽。",
        "4. GCM-P和direct基线必须使用同一组折叠后权重码、偏置码、累加位宽和末端舍入；这样架构消融只比较数据流，不混入量化差异。",
        "",
        "## 2. 静态合同",
        "",
        "| 设计 | block数 | K阈值 | head维 | BN | 结构结果 |",
        "|---|---:|---|---:|---|---|",
    ]
    for case in payload["cases"]:
        thresholds = {
            (row["k_threshold_min"], row["k_threshold_max"]) for row in case["blocks"]
        }
        threshold_text = ", ".join(f"{lo:g}..{hi:g}" for lo, hi in sorted(thresholds))
        lines.append(
            f"| {case['model']} | {len(case['blocks'])} | {threshold_text} | 32 | BatchNorm2d | "
            f"{'通过' if case['structural_pass'] else '失败'} |"
        )
    lines += [
        "",
        "折叠公式为：",
        "",
        "```text",
        "alpha[o] = gamma[o] / sqrt(running_var[o] + eps)",
        "W_fold[o,i] = alpha[o] * W[o,i]",
        "b_fold[o] = alpha[o] * (b[o] - running_mean[o]) + beta[o]",
        "```",
        "",
        "GCM-P对每个`(head, gate_class, input_channel)`生成一次`gate_code乘W_fold[:,i]`，但每个token-output仍独立累加。不同head不能仅凭class编号合并。",
        "",
        "## 3. CPU最小int8代理",
        "",
        "gate固定为无符号Q1.7码0到256，权重分别采用整tensor对称int8和逐输出通道对称int8，bias换算到对应累加尺度并使用int32。合成输入覆盖2%、10%和50%事件密度；这些输入只用于检查算术范围与量化误差，不代表真实光流分布。",
        "",
        "| 设计 | 权重量化 | 最大权重相对L2 | 最大合成输出相对L2 | 理论累加上界 | int32理论余量 | scale条目 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for case in payload["cases"]:
        for mode, label in (
            ("per_tensor_int8", "整tensor int8"),
            ("per_output_channel_int8", "逐输出通道 int8"),
        ):
            row = case["aggregate"][mode]
            lines.append(
                f"| {case['model']} | {label} | {row['weight_relative_l2_max']:.6f} | "
                f"{row['synthetic_output_relative_l2_max']:.6f} | "
                f"{row['theoretical_accumulator_abs_bound_max']:,} | "
                f"{row['theoretical_accumulator_int32_margin_min']:.1f}倍 | "
                f"{row['scale_entries']:,} |"
            )
    lines += [
        "",
        "## 4. RTL冻结边界",
        "",
        "首版RTL可以冻结：",
        "",
        "- K事件payload为1 bit，当前H67/H68的K阈值1.0不占运行时接口；",
        "- gate为9 bit无符号Q1.7码，范围0到256；",
        "- 投影先做BN静态折叠，再将同一份权重镜像供direct和GCM-P模式使用；",
        "- 乘积为9乘8 bit有符号结果，token-output使用至少32 bit有符号累加；",
        "- bias在所有活动输入累加后加一次，不能随class或K事件重复加入；",
        "- int32安全性由所有输入channel取最大gate码时的逐输出通道绝对和上界证明，不依赖随机样本；",
        "- 最终重标定、舍入、饱和和后续ATLIF输入格式在网络量化验证后再冻结。",
        "",
        "当前不能冻结：",
        "",
        "- 最终选择整tensor还是逐输出通道int8；",
        "- 输出截位位宽、逐层scale SRAM格式和bias码；",
        "- int8投影后的valid825 AEE、AAE、spikes变化；",
        "- SRAM宏、乘法器和多播网络的真实DC/SAIF PPA。",
        "",
        "## 5. 最小补跑清单",
        "",
        "1. 在H67 epoch19与H68 epoch19的冻结dyadic部署图中加入12层BN折叠投影int8仿真，跑valid825。",
        "2. 同时记录逐block输出相对L2、最大误差、下一ATLIF事件翻转率和最终AEE/AAE/spikes。",
        "3. 优先比较FP32投影、整tensorint8、逐输出通道int8三档；若逐通道scale控制开销过高，再评估按stage或按16输出通道组共享scale。",
        "4. 接受门槛暂定为AEE相对当前dyadic部署退化不超过0.5%，AAE退化不超过0.1度，spikes变化不超过1%，且无NaN/Inf。超过门槛时不进入DC主线。",
        "",
        "## 6. 证据边界",
        "",
        "本报告的静态结构数据来自真实checkpoint；量化误差来自确定性CPU合成输入；网络精度与真实workload的最终结论仍等待valid825和ordered profile。论文中必须分开标注，不能将本报告的合成误差写成任务精度。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h67", type=Path, default=DEFAULT_CASES["H67"])
    parser.add_argument("--h68", type=Path, default=DEFAULT_CASES["H68"])
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--json", type=Path, default=ROOT / "results/gcmp_projection_quant_contract.json"
    )
    parser.add_argument(
        "--md", type=Path, default=ROOT / "docs/67_GCM-P投影阈值与最小定点合同.md"
    )
    args = parser.parse_args()
    if args.samples <= 0:
        raise ValueError("samples必须为正数")
    cases = [
        analyze_case("H67", args.h67.resolve(), seed=args.seed, samples=args.samples),
        analyze_case("H68", args.h68.resolve(), seed=args.seed + 1000, samples=args.samples),
    ]
    payload = {
        "schema_version": 1,
        "gate_format": {"unsigned_bits": 9, "fractional_bits": 7, "code_range": [0, 256]},
        "weight_candidate": "symmetric_int8",
        "accumulator_candidate": "signed_int32",
        "samples_per_synthetic_density": args.samples,
        "synthetic_densities": [0.02, 0.10, 0.50],
        "cases": cases,
        "structural_pass": all(case["structural_pass"] for case in cases),
        "evidence_boundary": "checkpoint结构为真实证据；合成量化误差不是valid825网络精度。",
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.md.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(payload, args.md)
    print(args.md)
    return 0 if payload["structural_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
