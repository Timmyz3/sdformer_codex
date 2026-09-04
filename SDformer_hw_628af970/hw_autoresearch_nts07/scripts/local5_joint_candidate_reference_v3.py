#!/usr/bin/env python3
"""Local5 同窗全 head 候选的 RTL 校准串行周期模型。"""

from __future__ import annotations

from typing import Iterable

import local5_joint_candidate_reference_v2 as primitive


# v2 的 term 构造、五色地址和 1RW 后端模拟没有被 RTL 校准否决；v3 只替换
# relation/frontier 与 ordered backend 的组合相序。
HEIGHT = primitive.HEIGHT
WIDTH = primitive.WIDTH
PLANES = primitive.PLANES
TOKENS = primitive.TOKENS
HEAD_DIM = primitive.HEAD_DIM
BANKS = primitive.BANKS
BANK_DEPTH = primitive.BANK_DEPTH
ROLES = primitive.ROLES
ACTIVE_BITMAP_SCAN_CYCLES = primitive.ACTIVE_BITMAP_SCAN_CYCLES
VECTOR_READ_WORDS = primitive.VECTOR_READ_WORDS
SCALAR_SERIALIZER_CYCLES = primitive.SCALAR_SERIALIZER_CYCLES
RELATION_MEMO_CAPACITY_WORDS = primitive.RELATION_MEMO_CAPACITY_WORDS
REPLAY_CONTROL_CYCLES = primitive.REPLAY_CONTROL_CYCLES
REPLAY_MISS_FALLBACK_CYCLES = primitive.REPLAY_MISS_FALLBACK_CYCLES

Term = primitive.Term
SourceTerms = primitive.SourceTerms
HeadTrace = primitive.HeadTrace
BackendHeadResult = primitive.BackendHeadResult
BackendWindowResult = primitive.BackendWindowResult
destination_bank_address = primitive.destination_bank_address
build_source_terms = primitive.build_source_terms
build_head_trace = primitive.build_head_trace
simulate_direct_window = primitive.simulate_direct_window
simulate_srac2_window = primitive.simulate_srac2_window


CANDIDATES = primitive.CANDIDATES
FIXED_SCENARIOS = {
    "calibrated_median_459": 459,
    "calibration_max_475": 475,
}


def ordered_work_cycles(row: BackendHeadResult) -> int:
    """active descriptor capture + ordered term service + backend stall。"""
    value = row.descriptor_captures + row.term_issue_cycles + row.stall_cycles
    if value < 0:
        raise AssertionError("ordered work 不得为负")
    return value


def recompute_path_cycles(row: BackendHeadResult, *, fixed_cycles: int) -> int:
    """当前集成 RTL 校准的串行 recompute 边界。"""
    if fixed_cycles <= 0:
        raise ValueError("fixed_cycles 必须为正")
    return fixed_cycles + ordered_work_cycles(row)


def replay_path_cycles(
    row: BackendHeadResult,
    packet_words: int,
    *,
    forced_replay_miss: bool = False,
    fixed_cycles: int,
) -> int:
    """保守 replay：目录读、builder capture 和 ordered backend 均显式计账。"""
    if packet_words < 0 or packet_words != row.descriptor_captures:
        raise ValueError("memo packet words 必须等于 active descriptor 数")
    replay = (
        REPLAY_CONTROL_CYCLES
        + packet_words
        + row.descriptor_captures
        + row.term_issue_cycles
        + row.stall_cycles
        + 1  # directory end/commit
    )
    if not forced_replay_miss:
        return replay
    return (
        replay
        + REPLAY_MISS_FALLBACK_CYCLES
        + recompute_path_cycles(row, fixed_cycles=fixed_cycles)
    )


def memo_admission(heads: tuple[HeadTrace, ...]) -> list[bool]:
    """统一 head-order critical-only 7 KiB admission。"""
    admitted = [False] * len(heads)
    used = 0
    for index, trace in enumerate(heads):
        words = trace.active_sources
        critical = ACTIVE_BITMAP_SCAN_CYCLES + len(trace.terms) < TOKENS
        if critical and used + words <= RELATION_MEMO_CAPACITY_WORDS:
            admitted[index] = True
            used += words
    return admitted


def _candidate_window_cycles_fixed(
    heads: tuple[HeadTrace, ...], output_tiles: int, fixed_cycles: int
) -> dict[str, int]:
    if not heads or output_tiles <= 0:
        raise ValueError("joint-window head/output tile 必须为正")
    direct = simulate_direct_window(heads)
    gasr = simulate_srac2_window(heads)
    if direct.final_valid_sets != gasr.final_valid_sets:
        raise AssertionError("Direct 与 GASR2C-P 最终 valid-address 集合不一致")
    admitted = memo_admission(heads)

    def total(
        backend: BackendWindowResult,
        *,
        memo: bool,
    ) -> int:
        cycles = 0
        for output_tile in range(output_tiles):
            for head_index, row in enumerate(backend.heads):
                if memo and output_tile > 0 and admitted[head_index]:
                    cycles += replay_path_cycles(
                        row,
                        heads[head_index].active_sources,
                        fixed_cycles=fixed_cycles,
                    )
                else:
                    cycles += recompute_path_cycles(
                        row, fixed_cycles=fixed_cycles
                    )
            cycles += backend.final_readout_cycles
            cycles += backend.scalar_serializer_cycles
        return cycles

    return {
        "c0_direct_recompute": total(direct, memo=False),
        "c1_gasr2cp_recompute": total(gasr, memo=False),
        "c2_direct_erm7": total(direct, memo=True),
        "c3_gasr2cp_erm7": total(gasr, memo=True),
    }


def candidate_window_cycle_scenarios(
    heads: Iterable[HeadTrace], output_tiles: int
) -> dict[str, dict[str, int]]:
    """同时返回两个预提交固定项；正式晋级必须在两者下都成立。"""
    head_tuple = tuple(heads)
    return {
        scenario: _candidate_window_cycles_fixed(
            head_tuple, output_tiles, fixed_cycles
        )
        for scenario, fixed_cycles in FIXED_SCENARIOS.items()
    }
