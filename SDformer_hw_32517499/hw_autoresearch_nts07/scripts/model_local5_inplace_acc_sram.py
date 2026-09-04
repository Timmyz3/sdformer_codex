#!/usr/bin/env python3
"""建立 Local5 原位跨头累加的同口径 SRAM 组织模型。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArrayContract:
    name: str
    copies: int
    depth: int
    width: int
    port: str

    @property
    def logical_bits(self) -> int:
        return self.copies * self.depth * self.width


@dataclass(frozen=True)
class Macro:
    name: str
    depth: int
    width: int
    port: str
    evidence: str


def map_array(array: ArrayContract, macro: Macro) -> dict[str, object]:
    """按每个逻辑副本独立 depth/width tiling，禁止跨 bank 偷合并端口。"""
    if array.port != macro.port:
        return {
            "supported": False,
            "reason": f"端口不匹配：{array.port} 不能无损映射到 {macro.port}",
        }
    depth_tiles = math.ceil(array.depth / macro.depth)
    width_tiles = math.ceil(array.width / macro.width)
    macro_count = array.copies * depth_tiles * width_tiles
    allocated_bits = macro_count * macro.depth * macro.width
    return {
        "supported": True,
        "depth_tiles_per_copy": depth_tiles,
        "width_tiles_per_copy": width_tiles,
        "macro_count": macro_count,
        "allocated_bits": allocated_bits,
        "logical_bits": array.logical_bits,
        "waste_bits": allocated_bits - array.logical_bits,
        "waste_ratio": (allocated_bits - array.logical_bits) / allocated_bits,
    }


def build_model() -> dict[str, object]:
    tcfm = ArrayContract(
        name="TCFM5 五色向量 Acc", copies=5, depth=90, width=1024,
        port="1R1W",
    )
    scalar = ArrayContract(
        name="旧标量跨头 Acc", copies=1, depth=14400, width=32,
        port="1RW",
    )
    target_tcfm = Macro(
        name="目标1R1W_128x256", depth=128, width=256, port="1R1W",
        evidence="目标宏合同假设；本机无对应 liberty/LEF",
    )
    local_wide = Macro(
        name="本机fakeram45_128x256", depth=128, width=256, port="1RW",
        evidence="本机开放宏；仅1RW",
    )
    local_scalar = Macro(
        name="本机fakeram45_256x32", depth=256, width=32, port="1RW",
        evidence="本机开放宏",
    )

    tcfm_target = map_array(tcfm, target_tcfm)
    tcfm_local = map_array(tcfm, local_wide)
    scalar_local = map_array(scalar, local_scalar)
    assert bool(tcfm_target["supported"])
    assert not bool(tcfm_local["supported"])
    assert bool(scalar_local["supported"])

    common_alloc = int(tcfm_target["allocated_bits"])
    scalar_alloc = int(scalar_local["allocated_bits"])
    baseline_alloc = common_alloc + scalar_alloc
    inplace_alloc = common_alloc
    candidates = {
        "B0_scalar_recompute": {
            "logical_acc_bits": tcfm.logical_bits + scalar.logical_bits,
            "allocated_acc_bits": baseline_alloc,
            "extra_scalar_reads": 129600,
            "extra_scalar_writes": 129600,
        },
        "B1_scalar_memo": {
            "logical_acc_bits": tcfm.logical_bits + scalar.logical_bits,
            "allocated_acc_bits": baseline_alloc,
            "extra_scalar_reads": 129600,
            "extra_scalar_writes": 129600,
        },
        "B2_inplace_recompute": {
            "logical_acc_bits": tcfm.logical_bits,
            "allocated_acc_bits": inplace_alloc,
            "extra_scalar_reads": 0,
            "extra_scalar_writes": 0,
        },
        "B3_inplace_memo": {
            "logical_acc_bits": tcfm.logical_bits,
            "allocated_acc_bits": inplace_alloc,
            "extra_scalar_reads": 0,
            "extra_scalar_writes": 0,
        },
    }
    return {
        "evidence": "模型",
        "arrays": {"tcfm5": asdict(tcfm), "scalar_cross_head": asdict(scalar)},
        "macros": {
            item.name: asdict(item)
            for item in (target_tcfm, local_wide, local_scalar)
        },
        "mappings": {
            "tcfm5_to_target_1r1w": tcfm_target,
            "tcfm5_to_local_1rw": tcfm_local,
            "scalar_to_local_1rw": scalar_local,
        },
        "candidates": candidates,
        "deltas": {
            "logical_acc_bit_reduction": 1.0
            - candidates["B2_inplace_recompute"]["logical_acc_bits"]
            / candidates["B0_scalar_recompute"]["logical_acc_bits"],
            "allocated_acc_bit_reduction": 1.0
            - inplace_alloc / baseline_alloc,
            "deleted_scalar_transactions": 259200,
            "deleted_scalar_command_bits": 259200 * 32,
        },
        "boundaries": [
            "allocated bit 不是宏面积、功耗或 PPA",
            "目标1R1W宏只有组织假设，本机尚无 liberty/LEF",
            "本机1RW宽宏不能保持 TCFM5 每拍更新合同",
            "所有候选的 relation、weight、控制和输出存储均未计入此局部模型",
        ],
    }


def render_markdown(model: dict[str, object]) -> str:
    mappings = model["mappings"]
    candidates = model["candidates"]
    deltas = model["deltas"]
    tcfm = mappings["tcfm5_to_target_1r1w"]
    scalar = mappings["scalar_to_local_1rw"]
    local = mappings["tcfm5_to_local_1rw"]
    lines = [
        "# Local5 原位跨头累加 SRAM 组织模型",
        "",
        "## 结论",
        "",
        (
            "在统一宏组织假设下，旧标量路径需要公共 TCFM5 Acc 与额外跨头 Acc，"
            f"原位路径把逻辑 Acc 状态减少 {deltas['logical_acc_bit_reduction']:.2%}，"
            f"把向上取整后的 Acc 分配 bit 减少 "
            f"{deltas['allocated_acc_bit_reduction']:.2%} `[模型]`。"
        ),
        (
            f"它同时删除 {deltas['deleted_scalar_transactions']:,} 次额外标量 SRAM "
            "读写事务；这不是 SRAM 能量或功耗结论。"
        ),
        "",
        "## 映射合同",
        "",
        "| 逻辑数组 | 端口 | 物理组织 | 宏数 | 分配 bit | 浪费率 |",
        "|---|---|---|---:|---:|---:|",
        (
            f"| TCFM5 5x90x1024 | 1R1W | 128x256 1R1W | "
            f"{tcfm['macro_count']} | {tcfm['allocated_bits']} | "
            f"{tcfm['waste_ratio']:.2%} |"
        ),
        (
            f"| 旧跨头 14400x32 | 1RW | 256x32 1RW | "
            f"{scalar['macro_count']} | {scalar['allocated_bits']} | "
            f"{scalar['waste_ratio']:.2%} |"
        ),
        "",
        f"本机 128x256 开放宏映射 TCFM5：不支持；原因是 `{local['reason']}`。",
        "因此当前不能给原位 TCFM5 跑同吞吐 OpenROAD 宏 PPA；需要目标服务器的"
        "1R1W compiler 宏，或另做 1RW 降吞吐 RTL 公平基线。",
        "",
        "## 四候选局部存储",
        "",
        "| 候选 | 逻辑 Acc bit | 分配 Acc bit | 额外标量读 | 额外标量写 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in candidates.items():
        lines.append(
            f"| {name} | {item['logical_acc_bits']} | "
            f"{item['allocated_acc_bits']} | {item['extra_scalar_reads']} | "
            f"{item['extra_scalar_writes']} |"
        )
    lines.extend([
        "",
        "## 证据边界",
        "",
        "- `[模型]` 分配 bit 只反映 depth/width 向上取整，不含宏外围、布线和面积。",
        "- `[待验证]` 目标 1R1W 宏的面积、时序、读写能量与漏电。",
        "- `[待验证]` DC/STA/SAIF/PTPX 下 B0/B1/B2/B3 同 SDC、同宏规则对照。",
        "- `[rtl]` 删除标量跨头 memory 实例和事务由四候选回归另行证明。",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/qfit_local5_inplace_acc_20260809"),
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model = build_model()
    (args.out_dir / "storage_macro_model.json").write_text(
        json.dumps(model, indent=2, ensure_ascii=False) + "\n"
    )
    (args.out_dir / "storage_macro_model.md").write_text(
        render_markdown(model)
    )


if __name__ == "__main__":
    main()
