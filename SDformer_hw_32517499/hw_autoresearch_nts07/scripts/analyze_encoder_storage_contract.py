#!/usr/bin/env python3
"""生成H67/H68全Encoder存储、ATLIF执行图与统一瓦片合同。"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
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


CASES = {
    "H67": {
        "checkpoint": EXP / (
            "results/h67_allbinary_all12_motionxor_ttx_w025_w720_fastlr_full30_bs8_"
            "full30_20260711_setsid/checkpoint_epoch19.pth"
        ),
        "profile": EXP / "results/h67_ep19_true_ttb_profile100_20260712/nts11_hardware_p0_profile.json",
        "activity": EXP / "results/h67_ep19_true_ttb_profile100_20260712/atlif_activity.csv",
    },
    "H68": {
        "checkpoint": EXP / (
            "results/h68_allbinary_all12_castling_ttx_aux050_toep20_w720_fastlr_full30_"
            "bs8_full30_20260711_setsid/checkpoint_epoch19.pth"
        ),
        "profile": EXP / "results/h68_ep19_true_ttb_profile100_20260713/nts11_hardware_p0_profile.json",
        "activity": EXP / "results/h68_ep19_true_ttb_profile100_20260713/atlif_activity.csv",
    },
}
STAGES = (
    {"stage": 0, "blocks": 2, "channels": 96, "heads": 3, "height": 72, "width": 96},
    {"stage": 1, "blocks": 2, "channels": 192, "heads": 6, "height": 36, "width": 48},
    {"stage": 2, "blocks": 6, "channels": 384, "heads": 12, "height": 18, "width": 24},
    {"stage": 3, "blocks": 2, "channels": 768, "heads": 24, "height": 9, "width": 12},
)
TIME_FULL = 10
TIME_ATTN = 2
WINDOW_TOKENS = 81
HEAD_DIM = 32
CLOCK_HZ = 500_000_000
TARGET_FPS = 30


def byte_capacity(elements: int, bits: int) -> dict[str, float | int]:
    size = (elements * bits + 7) // 8
    return {"bits": bits, "bytes": size, "KiB": size / 1024, "MiB": size / (1024**2)}


def first_activation(records: list[dict[str, Any]], name: str) -> dict[str, Any]:
    return next(record for record in records if record["name"] == name)


def quantization_error(values: torch.Tensor, total_bits: int, fractional_bits: int) -> dict[str, float | int]:
    scale = float(1 << fractional_bits)
    qmin = -(1 << (total_bits - 1))
    qmax = (1 << (total_bits - 1)) - 1
    scaled = torch.round(values * scale)
    clipped = scaled.clamp(qmin, qmax)
    restored = clipped / scale
    error = (restored - values).abs()
    return {
        "total_bits": total_bits,
        "fractional_bits": fractional_bits,
        "step": 1.0 / scale,
        "clip_count": int(scaled.ne(clipped).sum().item()),
        "mae": float(error.mean().item()),
        "max_abs_error": float(error.max().item()),
    }


def load_called(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["name"]: row for row in csv.DictReader(handle)}


def analyze_case(name: str, paths: dict[str, Path]) -> dict[str, Any]:
    profile = json.loads(paths["profile"].read_text(encoding="utf-8"))
    activation_records = profile["summary"]["activation_records"]
    called = load_called(paths["activity"])

    register_shiftmax_pickle_compat()
    model = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
    modules = {
        module_name: module
        for module_name, module in model.named_modules()
        if module.__class__.__name__ == "ATLIFTernaryPSN"
    }

    dead_suffix = ".attn.attn_sn.spiking_neuron"
    installed_by_t = Counter(int(module.T) for module in modules.values())
    called_by_t = Counter(int(modules[module_name].T) for module_name in called)
    dead_names = sorted(module_name for module_name in called if module_name.endswith(dead_suffix))
    live_names = sorted(set(called) - set(dead_names))
    live_by_t = Counter(int(modules[module_name].T) for module_name in live_names)

    called_outputs = 0
    dead_outputs = 0
    live_outputs = 0
    called_temporal_macs = 0
    live_temporal_macs = 0
    live_temporal_macs_by_t: Counter[int] = Counter()
    for module_name, row in called.items():
        calls = int(row["calls"])
        outputs = int(row["elements"]) // calls
        temporal_macs = outputs * int(modules[module_name].T)
        called_outputs += outputs
        called_temporal_macs += temporal_macs
        if module_name in dead_names:
            dead_outputs += outputs
        else:
            live_outputs += outputs
            live_temporal_macs += temporal_macs
            live_temporal_macs_by_t[int(modules[module_name].T)] += temporal_macs

    parameter_values = []
    parameter_entries = 0
    parameter_entries_by_t: Counter[int] = Counter()
    for module_name in live_names:
        module = modules[module_name]
        values = torch.cat(
            [module.weight.detach().float().flatten(), module.bias.detach().float().flatten(), module.thresh.detach().float().flatten()]
        )
        parameter_values.append(values)
        parameter_entries += int(values.numel())
        parameter_entries_by_t[int(module.T)] += int(values.numel())
    all_parameters = torch.cat(parameter_values)

    skips = [first_activation(activation_records, f"S{stage}.skip") for stage in range(4)]
    long_skip_elements = sum(int(record["elements"]) for record in skips[:3])
    stage3_local_elements = int(skips[3]["elements"])
    stage_rows = []
    for stage, record in enumerate(skips):
        stage_rows.append({
            "stage": stage,
            "shape": record["shape"],
            "elements": int(record["elements"]),
            "density": float(record["density"]),
            "lifetime": "跨Encoder/Decoder长跳连" if stage < 3 else "瓶颈局部",
        })

    del model
    return {
        "model": name,
        "source_profile": str(paths["profile"]),
        "source_checkpoint": str(paths["checkpoint"]),
        "activation_evidence": {
            "stage_boundaries": stage_rows,
            "long_skip_elements_s0_s2": long_skip_elements,
            "stage3_local_elements": stage3_local_elements,
            "long_skip_capacity": [byte_capacity(long_skip_elements, bits) for bits in (1, 2, 4, 8, 16)],
            "warning": "density仅表示非零率，旧profile未证明张量可按binary或ternary存储。",
        },
        "atlif_execution_graph": {
            "installed": len(modules),
            "called": len(called),
            "dead_called": len(dead_names),
            "functionally_live": len(live_names),
            "installed_by_T": dict(sorted(installed_by_t.items())),
            "called_by_T": dict(sorted(called_by_t.items())),
            "live_by_T": dict(sorted(live_by_t.items())),
            "dead_names": dead_names,
            "called_output_elements_per_frame": called_outputs,
            "dead_output_elements_per_frame": dead_outputs,
            "live_output_elements_per_frame": live_outputs,
            "called_temporal_macs_per_frame": called_temporal_macs,
            "live_temporal_macs_per_frame": live_temporal_macs,
            "live_temporal_macs_by_T": dict(sorted(live_temporal_macs_by_t.items())),
            "dead_output_fraction": dead_outputs / called_outputs if called_outputs else 0.0,
            "dead_temporal_mac_fraction": (
                (called_temporal_macs - live_temporal_macs) / called_temporal_macs
                if called_temporal_macs else 0.0
            ),
        },
        "atlif_parameter_contract": {
            "live_parameter_entries": parameter_entries,
            "entries_by_T": dict(sorted(parameter_entries_by_t.items())),
            "range": {
                "min": float(all_parameters.min().item()),
                "max": float(all_parameters.max().item()),
                "absmax": float(all_parameters.abs().max().item()),
            },
            "capacity": [byte_capacity(parameter_entries, bits) for bits in (4, 6, 8, 16)],
            "parameter_only_quantization": [
                quantization_error(all_parameters, 4, 2),
                quantization_error(all_parameters, 6, 4),
                quantization_error(all_parameters, 8, 6),
            ],
            "warning": "参数误差不等于网络精度；任何位宽降级仍需逐层事件翻转率和valid825验证。",
        },
    }


def build_tile_contract() -> dict[str, Any]:
    full_elements = TIME_FULL * WINDOW_TOKENS * HEAD_DIM
    attn_elements = TIME_ATTN * WINDOW_TOKENS * HEAD_DIM
    stages = []
    for row in STAGES:
        item = dict(row)
        item["head_dim"] = row["channels"] // row["heads"]
        item["full_window_all_channel_elements"] = TIME_FULL * WINDOW_TOKENS * row["channels"]
        item["head_tiles_per_window"] = row["heads"]
        stages.append(item)
    return {
        "network_invariants": {
            "T_full": TIME_FULL,
            "T_attention": TIME_ATTN,
            "window_spatial_tokens": WINDOW_TOKENS,
            "head_dim": HEAD_DIM,
            "head_dim_constant_across_stages": all(
                row["channels"] // row["heads"] == HEAD_DIM for row in STAGES
            ),
        },
        "stage_geometry": stages,
        "head_time_tile": {
            "T10_elements": full_elements,
            "T2_elements": attn_elements,
            "T10_input_capacity": [byte_capacity(full_elements, bits) for bits in (4, 8, 16)],
            "T2_input_capacity": [byte_capacity(attn_elements, bits) for bits in (4, 8, 16)],
            "binary_output_capacity": byte_capacity(full_elements, 1),
            "TESSA_pair_row_capacity": byte_capacity(WINDOW_TOKENS * 128, 1),
        },
        "divisor_packed_temporal_array": {
            "geometry": "32 channel lanes x 10 temporal-output slots",
            "macs_per_cycle_per_array": HEAD_DIM * TIME_FULL,
            "T10_mapping": "一个空间位置的32通道并行，10个输出时刻，输入时间维串行10拍",
            "T2_mapping": "10列拆成5组，每组2个输出时刻，同时处理5个空间位置，输入时间维串行2拍",
        },
    }


def write_markdown(result: dict[str, Any], path: Path) -> None:
    h67 = result["models"]["H67"]
    graph = h67["atlif_execution_graph"]
    params = h67["atlif_parameter_contract"]
    skips = h67["activation_evidence"]
    tile = result["tile_contract"]["head_time_tile"]
    array = result["tile_contract"]["divisor_packed_temporal_array"]
    live_macs = graph["live_temporal_macs_per_frame"]
    array_rows = []
    for arrays in (1, 2, 4):
        macs_per_cycle = arrays * int(array["macs_per_cycle_per_array"])
        cycles = (live_macs + macs_per_cycle - 1) // macs_per_cycle
        array_rows.append((arrays, macs_per_cycle, cycles, cycles / CLOCK_HZ * 1000, CLOCK_HZ / cycles))
    lines = [
        "# H67/H68全Encoder存储与ATLIF执行合同",
        "",
        "**结论**：全二值只成立于ATLIF输出事件和Q/K事件，不能外推为所有残差、长跳连和卷积激活均为1 bit。当前部署ATLIF是`T×T` PSN时间矩阵加阈值发放，不是递归LIF膜状态。",
        "",
        "## 1. 可直接引用的统计结果",
        "",
        f"- S0-S2三条长跳连合计 `{skips['long_skip_elements_s0_s2']:,}` 元素；旧profile非零率接近100%。",
        f"- 若分别按1/2/4/8/16 bit保存，容量为 " + "、".join(
            f"{row['bits']}bit={row['MiB']:.3f} MiB" for row in skips["long_skip_capacity"]
        ) + "。其中1 bit只是容量下界，不是已验证表示。",
        f"- ATLIF安装`{graph['installed']}`点、forward调用`{graph['called']}`点；其中12个`attn_sn`结果不进入正常推理输出，固定部署功能活跃候选为`{graph['functionally_live']}`点。",
        f"- 活跃点构成为 `T=10`的{graph['live_by_T'][10]}点和`T=2`的{graph['live_by_T'][2]}点；不应实例化81套物理单元。",
        f"- 现有调用每帧产生 `{graph['called_output_elements_per_frame']:,}` 个ATLIF输出元素；删除12个死调用后为 `{graph['live_output_elements_per_frame']:,}`，输出事务下降 `{graph['dead_output_fraction']:.2%}`。",
        f"- 按当前稠密时间矩阵计，活跃ATLIF约 `{graph['live_temporal_macs_per_frame']:,}` 次标量MAC/帧；这说明完整芯片不能只优化attention行核。",
        f"- 81个活跃点的时间矩阵、bias和threshold仅 `{params['live_parameter_entries']:,}` 个标量；8 bit容量约 `{params['capacity'][2]['KiB']:.2f} KiB`，适合片上descriptor/ROM，而不是为每点配大状态SRAM。",
        "",
        "## 2. 统一Head-Time Tile",
        "",
        "四个stage均满足`channels/heads=32`，空间窗口均为`9×9`。因此可冻结一个跨stage的物理瓦片：",
        "",
        "```text",
        "HTT = {一个head, 9×9空间, 时间T}",
        "T=10：PSN temporal mixer / MLP / projection前后的ATLIF",
        "T=2 ：Q/K、attention投影ATLIF和TESSA pair数据流",
        "```",
        "",
        f"- `T=10`瓦片为 `{tile['T10_elements']:,}` 元素：8 bit输入缓冲 `{tile['T10_input_capacity'][1]['KiB']:.3f} KiB`，16 bit为 `{tile['T10_input_capacity'][2]['KiB']:.3f} KiB`。",
        f"- `T=2`瓦片为 `{tile['T2_elements']:,}` 元素：8 bit输入缓冲 `{tile['T2_input_capacity'][1]['KiB']:.3f} KiB`。",
        f"- 二值`T=10`输出瓦片仅 `{tile['binary_output_capacity']['KiB']:.3f} KiB`；TESSA一行128-bit时间对bank为 `{tile['TESSA_pair_row_capacity']['KiB']:.3f} KiB`。",
        "- 地址发生器、PSN时间矩阵阵列、阈值发放器和TESSA窗口读取器可共享同一瓦片坐标，不等于所有算术单元必须合并成一个模块。",
        "",
        "## 3. 除数打包时间矩阵阵列的吞吐下界",
        "",
        "一个物理阵列采用`32通道×10时间输出槽=320 MAC/拍`。`T=10`模式一组32通道用10拍完成；`T=2`模式将10列拆成5个二列组，同时处理5个空间位置，用2拍完成。这里的满利用率依赖输入、权重和累加位宽已量化并能每拍供数。",
        "",
        "| 阵列数 | 时间MAC/拍 | 仅活跃ATLIF周期/帧 | 500MHz延迟 | ATLIF-only FPS |",
        "|---:|---:|---:|---:|---:|",
    ]
    for arrays, macs_per_cycle, cycles, latency_ms, fps in array_rows:
        lines.append(f"| {arrays} | {macs_per_cycle} | {cycles:,} | {latency_ms:.3f} ms | {fps:.2f} |")
    lines.extend([
        "",
        f"30 FPS在500MHz下每帧只有`{CLOCK_HZ // TARGET_FPS:,}`拍。单阵列虽有约36 FPS的ATLIF-only上界，但几乎没有给projection/MLP/TESSA/存储留下余量；平衡候选至少从双阵列开始，最终仍需全Encoder周期模型淘汰。",
        "",
        "## 4. 死计算边界",
        "",
        "正常Swin推理使用attention返回值的第一个分量`x`，第二个分量`attn_sn(x)`只在显式`return_attention`调试路径中有意义。DATE固定部署可以删除12个`attn_sn`调用，但兼容调试/可视化的通用核不能直接删除。论文应分列“软件动态调用93”和“固定部署活跃81”，不能把81写成软件安装数。",
        "",
        "## 5. 量化结论",
        "",
        "ATLIF参数范围约为 "
        f"`[{params['range']['min']:.4f}, {params['range']['max']:.4f}]`。参数本身的定点误差扫描为：",
        "",
        "| 格式 | 步长 | clip数 | 参数MAE | 最大绝对误差 |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in params["parameter_only_quantization"]:
        lines.append(
            f"| Q{row['total_bits'] - row['fractional_bits'] - 1}.{row['fractional_bits']} ({row['total_bits']}bit signed) "
            f"| {row['step']:.6f} | {row['clip_count']} | {row['mae']:.6f} | {row['max_abs_error']:.6f} |"
        )
    lines.extend([
        "",
        "该表只证明参数存储可做候选，不证明事件输出不翻转。最小量化验证必须导出每ATLIF点的`h_seq-threshold` margin、量化前后事件翻转率，并重跑valid825；在此之前，RTL不得把8 bit ATLIF写成冻结事实。",
        "",
        "## 6. 下一轮profile新增项",
        "",
        "- 四个stage边界的min/max、近整数率、binary01率、ternary率，用来冻结残差和skip位宽。",
        "- ATLIF输入张量的稀疏度与数值格式，区分“二值输出”与“稠密时间矩阵输入”。",
        "- `h_seq-threshold` margin直方图和按site事件翻转敏感度，服务4/6/8 bit最小量化。",
        "- 分列93个动态调用、12个死结果和81个固定部署活跃点的周期/能量。",
        "",
        "## 7. 当前架构约束",
        "",
        "1. attention事件bank可以1 bit，残差/skip SRAM位宽尚未冻结。",
        "2. 三条长跳连若不能全部片上，应建模共享SRAM/DRAM生命周期与压缩，而不是假定1.45 MB。",
        "3. ATLIF物理核必须支持`T=10/T=2`，参数按site切换；主算力尺寸由每帧约44亿时间MAC而不是5247个参数决定。",
        "4. TESSA仍是attention子系统；只有加入PSN temporal mixer、projection/MLP、残差和三条长跳连的周期/存储模型后，才能称full-encoder accelerator。",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    result = {
        "说明": "旧profile提供形状/非零率；checkpoint提供ATLIF参数；死调用由固定H67/H68正常推理数据流静态判定。",
        "models": {name: analyze_case(name, paths) for name, paths in CASES.items()},
        "tile_contract": build_tile_contract(),
    }
    output = ROOT / "results/h67_h68_encoder_storage_contract.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(result, output.with_suffix(".md"))
    print(output.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
