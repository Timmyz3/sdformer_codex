#!/usr/bin/env python3
"""生成 crop/full-resolution 下的 tile/term 硬件规模账本。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


BLOCKS = (2, 2, 6, 2)
HEADS = (3, 6, 12, 24)
HEAD_DIM = 32
PATCH_STRIDE = 4
ATTENTION_T = 2
NUM_STEPS = 10
SCORE_W = 8
GATE_W = 9
SCS_CLASSES = 35


@dataclass(frozen=True)
class StageLedger:
    stage: int
    feature_h: int
    feature_w: int
    blocks: int
    heads: int
    spatial_windows: int
    temporal_window_groups: int
    windows_per_block: int
    rows_per_frame: int
    tokens_per_row: int
    scheduled_token_slots_per_frame: int


@dataclass(frozen=True)
class CaseLedger:
    name: str
    input_h: int
    input_w: int
    window_h: int
    window_w: int
    tokens_per_row: int
    rows_per_frame: int
    scheduled_token_slots_per_frame: int
    geometry_valid_token_slots_per_frame: int
    padding_token_slots_per_frame: int
    padding_token_slot_ratio: float
    spatial_pixels: int
    token_bitmap_bytes_per_row: int
    q_or_k_tile_bytes_per_head: int
    score_materialization_bytes_per_row: int
    gate_materialization_bytes_per_row: int
    scs_counter_bits: int
    scs_histogram_bytes_per_row: int
    local5_three_row_k_bytes_per_head: int
    local5_candidate_slots_per_window: int
    local5_valid_edges_per_window: int
    local5_invalid_candidates_per_window: int
    local5_invalid_candidate_ratio: float
    ttb4_bundles_per_row: int
    stages: list[StageLedger]


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def stage_geometry(input_h: int, input_w: int, stage: int) -> tuple[int, int]:
    divisor = PATCH_STRIDE * (2**stage)
    return ceil_div(input_h, divisor), ceil_div(input_w, divisor)


def build_case(name: str, input_h: int, input_w: int, window: int) -> CaseLedger:
    if NUM_STEPS % ATTENTION_T:
        raise ValueError("NUM_STEPS 必须能被 ATTENTION_T 整除")
    temporal_groups = NUM_STEPS // ATTENTION_T
    tokens_per_row = ATTENTION_T * window * window
    stages: list[StageLedger] = []

    for stage, (blocks, heads) in enumerate(zip(BLOCKS, HEADS, strict=True)):
        feature_h, feature_w = stage_geometry(input_h, input_w, stage)
        spatial_windows = ceil_div(feature_h, window) * ceil_div(feature_w, window)
        windows_per_block = spatial_windows * temporal_groups
        rows_per_frame = windows_per_block * blocks * heads
        stages.append(
            StageLedger(
                stage=stage,
                feature_h=feature_h,
                feature_w=feature_w,
                blocks=blocks,
                heads=heads,
                spatial_windows=spatial_windows,
                temporal_window_groups=temporal_groups,
                windows_per_block=windows_per_block,
                rows_per_frame=rows_per_frame,
                tokens_per_row=tokens_per_row,
                scheduled_token_slots_per_frame=rows_per_frame * tokens_per_row,
            )
        )

    counter_bits = math.ceil(math.log2(tokens_per_row + 1))
    local5_candidate_slots = tokens_per_row * 5
    # Each time slice has window invalid requests on each of N/S/E/W.
    local5_invalid_candidates = ATTENTION_T * 4 * window
    local5_valid_edges = local5_candidate_slots - local5_invalid_candidates
    scheduled_slots = sum(s.scheduled_token_slots_per_frame for s in stages)
    valid_slots = sum(
        s.feature_h * s.feature_w * NUM_STEPS * s.blocks * s.heads for s in stages
    )
    return CaseLedger(
        name=name,
        input_h=input_h,
        input_w=input_w,
        window_h=window,
        window_w=window,
        tokens_per_row=tokens_per_row,
        rows_per_frame=sum(s.rows_per_frame for s in stages),
        scheduled_token_slots_per_frame=scheduled_slots,
        geometry_valid_token_slots_per_frame=valid_slots,
        padding_token_slots_per_frame=scheduled_slots - valid_slots,
        padding_token_slot_ratio=(scheduled_slots - valid_slots) / scheduled_slots,
        spatial_pixels=input_h * input_w,
        token_bitmap_bytes_per_row=ceil_div(tokens_per_row, 8),
        q_or_k_tile_bytes_per_head=ceil_div(tokens_per_row * HEAD_DIM, 8),
        score_materialization_bytes_per_row=ceil_div(tokens_per_row * SCORE_W, 8),
        gate_materialization_bytes_per_row=ceil_div(tokens_per_row * GATE_W, 8),
        scs_counter_bits=counter_bits,
        scs_histogram_bytes_per_row=ceil_div(SCS_CLASSES * counter_bits, 8),
        local5_three_row_k_bytes_per_head=ceil_div(
            3 * ATTENTION_T * window * HEAD_DIM, 8
        ),
        local5_candidate_slots_per_window=local5_candidate_slots,
        local5_valid_edges_per_window=local5_valid_edges,
        local5_invalid_candidates_per_window=local5_invalid_candidates,
        local5_invalid_candidate_ratio=(
            local5_invalid_candidates / local5_candidate_slots
        ),
        ttb4_bundles_per_row=ceil_div(tokens_per_row, 4),
        stages=stages,
    )


def render_markdown(cases: list[CaseLedger]) -> str:
    base = cases[0]
    lines = [
        "# Crop 与全分辨率 Tile-Term 硬件规模账本",
        "",
        "本账本是静态几何/位宽模型，不是 RTL 周期、DC 面积或 SAIF 功耗结果。",
        "统一假设：`T=10`，attention 每次处理 `T=2`，head dim 为 32，",
        "encoder blocks 为 `[2,2,6,2]`，heads 为 `[3,6,12,24]`。",
        "",
        "## 1. 全局对照",
        "",
        "| 配置 | 输入 | 窗口 | token/row | rows/frame | scheduled token slots/frame | "
        "相对 crop slots |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        ratio = (
            case.scheduled_token_slots_per_frame
            / base.scheduled_token_slots_per_frame
        )
        lines.append(
            f"| {case.name} | {case.input_h}x{case.input_w} | "
            f"T2x{case.window_h}x{case.window_w} | {case.tokens_per_row:,} | "
            f"{case.rows_per_frame:,} | "
            f"{case.scheduled_token_slots_per_frame:,} | "
            f"{ratio:.4f}x |"
        )

    lines += [
        "",
        "## 2. 几何有效槽与 padding",
        "",
        "| 配置 | scheduled slots | geometry-valid slots | padding slots | padding ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case.name} | {case.scheduled_token_slots_per_frame:,} | "
            f"{case.geometry_valid_token_slots_per_frame:,} | "
            f"{case.padding_token_slots_per_frame:,} | "
            f"{case.padding_token_slot_ratio:.4%} |"
        )

    lines += [
        "",
        "这里的 geometry-valid 只按 feature-map 边界计算。软件是否让 padding token "
        "参与 score/Shiftmax、Local5 是否跳过 invalid center，必须由真实 trace 再定。",
        "",
        "## 3. 每 row 理想 bit-packed logical lower bound",
        "",
        "| 配置 | token bitmap | 单个 Q/K tile/head | score 物化 | gate 物化 | "
        "SCS histogram | Local5 三行 K/head | TTB4 bundle |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case.name} | {case.token_bitmap_bytes_per_row} B | "
            f"{case.q_or_k_tile_bytes_per_head} B | "
            f"{case.score_materialization_bytes_per_row} B | "
            f"{case.gate_materialization_bytes_per_row} B | "
            f"{case.scs_histogram_bytes_per_row} B "
            f"({SCS_CLASSES}x{case.scs_counter_bits}b) | "
            f"{case.local5_three_row_k_bytes_per_head} B | "
            f"{case.ttb4_bundles_per_row} |"
        )

    lines += [
        "",
        "这些字节数不含 SRAM bank 对齐、端口复制、ECC、读改写端口和 macro 粒度。",
        "",
        "## 4. Local5 标称满窗口固定拓扑",
        "",
        "| 配置 | candidate slots/window | valid edges/window | "
        "invalid boundary candidates | invalid ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in cases:
        lines.append(
            f"| {case.name} | {case.local5_candidate_slots_per_window:,} | "
            f"{case.local5_valid_edges_per_window:,} | "
            f"{case.local5_invalid_candidates_per_window:,} | "
            f"{case.local5_invalid_candidate_ratio:.4%} |"
        )

    lines += [
        "",
        "该表是无 padding 的标称满窗口。部分窗口和软件 padding 语义需真实 trace。",
        "",
        "## 5. 分 stage 调度",
        "",
    ]
    for case in cases:
        lines += [
            f"### {case.name}",
            "",
            "| stage | feature HxW | blocks | heads | spatial windows | "
            "T2 windows/block | rows/frame | scheduled token slots/frame |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for stage in case.stages:
            lines.append(
                f"| S{stage.stage} | {stage.feature_h}x{stage.feature_w} | "
                f"{stage.blocks} | {stage.heads} | {stage.spatial_windows} | "
                f"{stage.windows_per_block} | {stage.rows_per_frame:,} | "
                f"{stage.scheduled_token_slots_per_frame:,} |"
            )
        lines.append("")

    lines += [
        "## 6. 可直接使用的硬件结论",
        "",
        "1. `full-w9` 保持 162-token 叶核，但 rows/frame 从 6,720 增到 "
        f"{cases[1].rows_per_frame:,}；必须扩 descriptor、带宽和端到端周期账本。",
        "2. `full-w15` 与 `crop-w9` 的 rows/frame 相同，但每 row 从 162 增到 "
        "450 token，scheduled token slots 增加到 2.7778x；这不是实际周期或能耗。",
        "3. Motion 的 per-token score/gate/bitmap 存储随 162→450 近线性增长；"
        "35 类 SCS histogram 只因计数位宽从 8b 增到 9b，增长很小。",
        "4. Local5 每个 destination 仍只有 3/4/5 个候选；w15 主要增加 destination "
        "数和三行驻留宽度，不改变 MFEP multiplicity 上界 5。",
        "5. 标称满窗口下，Local5 边界无效候选比例从 w9 的 8.89% 降到 "
        "w15 的 5.33%；部分窗口仍需按真实 padding 语义重算。",
        "6. 因而最终芯片应是分块/流式叶核，论文吞吐和能耗必须按 480x640 "
        "全帧累计，不能把 crop 单窗 RTL 周期直接写成 FPS。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/resolution_tile_term_ledger_20260728"),
    )
    args = parser.parse_args()

    cases = [
        build_case("crop-w9", 288, 384, 9),
        build_case("full-w9", 480, 640, 9),
        build_case("full-w15", 480, 640, 15),
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "evidence_level": "模型",
        "assumptions": {
            "num_steps": NUM_STEPS,
            "attention_t": ATTENTION_T,
            "head_dim": HEAD_DIM,
            "patch_stride": PATCH_STRIDE,
            "blocks": BLOCKS,
            "heads": HEADS,
            "score_w": SCORE_W,
            "gate_w": GATE_W,
            "scs_classes": SCS_CLASSES,
        },
        "cases": [asdict(case) for case in cases],
    }
    (args.out_dir / "ledger.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "ledger.md").write_text(
        render_markdown(cases) + "\n", encoding="utf-8"
    )
    print(f"写入 {args.out_dir / 'ledger.json'}")
    print(f"写入 {args.out_dir / 'ledger.md'}")


if __name__ == "__main__":
    main()
