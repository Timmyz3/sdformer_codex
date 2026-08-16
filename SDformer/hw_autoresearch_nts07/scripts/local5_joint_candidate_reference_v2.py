#!/usr/bin/env python3
"""Local5 同窗全 head 候选的公平 ordered-frontend/1RW v2 模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


HEIGHT = 15
WIDTH = 15
PLANES = 2
TOKENS = 450
HEAD_DIM = 32
BANKS = 5
BANK_DEPTH = 90
ROLES = 5
ROLE_DY = (0, 1, -1, 0, 0)
ROLE_DX = (0, 0, 0, 1, -1)
ACTIVE_BITMAP_SCAN_CYCLES = 15
RELATION_BUILD_CYCLES = 450
VECTOR_READ_WORDS = 450
SCALAR_SERIALIZER_CYCLES = 450 * 32
RELATION_MEMO_WORD_BITS = 112
RELATION_MEMO_CAPACITY_BITS = 7 * 8192
RELATION_MEMO_CAPACITY_WORDS = RELATION_MEMO_CAPACITY_BITS // RELATION_MEMO_WORD_BITS
GASR2CP_CONTEXTS_PER_BANK = 2
RELATION_READ_LATENCY = 1
RECOMPUTE_CONTROL_CYCLES = 4
REPLAY_CONTROL_CYCLES = 4
REPLAY_MISS_FALLBACK_CYCLES = 2

CANDIDATES = {
    "c0_direct_recompute": {
        "backend": "direct_1rw",
        "relation": "recompute_every_output_tile",
    },
    "c1_gasr2cp_recompute": {
        "backend": "gasr2c_cross_head_preserve_model",
        "relation": "recompute_every_output_tile",
    },
    "c2_direct_erm7": {
        "backend": "direct_1rw",
        "relation": "critical_only_exact_memo_7kib",
    },
    "c3_gasr2cp_erm7": {
        "backend": "gasr2c_cross_head_preserve_model",
        "relation": "critical_only_exact_memo_7kib",
    },
}


@dataclass(frozen=True)
class Term:
    source_id: int
    lane: int
    gate: int
    destinations: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class SourceTerms:
    source_id: int
    terms: tuple[Term, ...]


@dataclass(frozen=True)
class HeadTrace:
    sources: tuple[SourceTerms, ...]

    @property
    def terms(self) -> tuple[Term, ...]:
        return tuple(term for source in self.sources for term in source.terms)

    @property
    def active_sources(self) -> int:
        return sum(bool(source.terms) for source in self.sources)


@dataclass
class BackendHeadResult:
    service_cycles: int = 0
    term_issue_cycles: int = 0
    terms: int = 0
    destination_updates: int = 0
    first_touch_writes: int = 0
    rmw_terms: int = 0
    sram_reads: int = 0
    sram_writes: int = 0
    stall_cycles: int = 0
    flush_cycles: int = 0
    descriptor_captures: int = 0


@dataclass
class BackendWindowResult:
    heads: tuple[BackendHeadResult, ...]
    final_valid_addresses: int
    final_readout_cycles: int
    scalar_serializer_cycles: int = SCALAR_SERIALIZER_CYCLES
    final_valid_sets: tuple[tuple[int, ...], ...] = ()


def destination_bank_address(plane: int, y: int, x: int) -> tuple[int, int]:
    if not (0 <= plane < PLANES and 0 <= y < HEIGHT and 0 <= x < WIDTH):
        raise ValueError("Local5 destination 坐标越界")
    bank = (x + 2 * y) % BANKS
    address = plane * (HEIGHT * 3) + y * 3 + x // 5
    if not 0 <= address < BANK_DEPTH:
        raise AssertionError("五色 bank 地址越界")
    return bank, address


def build_source_terms(
    source_id: int,
    plane: int,
    source_y: int,
    source_x: int,
    k_bitmap: int,
    gates: Iterable[int],
    valid_mask: int,
) -> SourceTerms:
    """RTL 顺序：source 升序、lane 升序、gate 首次 role 出现顺序。"""
    gates_tuple = tuple(int(value) for value in gates)
    if len(gates_tuple) != ROLES or not 0 <= source_id < TOKENS:
        raise ValueError("source descriptor 形状或 id 非法")
    unique_gates: list[int] = []
    gate_roles: list[list[int]] = []
    for role, gate in enumerate(gates_tuple):
        if not ((valid_mask >> role) & 1) or gate == 0:
            continue
        if gate in unique_gates:
            gate_roles[unique_gates.index(gate)].append(role)
        else:
            unique_gates.append(gate)
            gate_roles.append([role])

    terms: list[Term] = []
    for lane in range(HEAD_DIM):
        if not ((int(k_bitmap) >> lane) & 1):
            continue
        for gate, roles in zip(unique_gates, gate_roles, strict=True):
            destinations: list[tuple[int, int]] = []
            occupied: set[int] = set()
            for role in roles:
                y = source_y + ROLE_DY[role]
                x = source_x + ROLE_DX[role]
                bank, address = destination_bank_address(plane, y, x)
                if bank in occupied:
                    raise AssertionError("五色映射在单 term 内冲突")
                occupied.add(bank)
                destinations.append((bank, address))
            terms.append(
                Term(
                    source_id=source_id,
                    lane=lane,
                    gate=gate,
                    destinations=tuple(destinations),
                )
            )
    return SourceTerms(source_id=source_id, terms=tuple(terms))


def build_head_trace(
    source_ids: np.ndarray,
    planes: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    k_bitmaps: np.ndarray,
    gates: np.ndarray,
    valid_masks: np.ndarray,
) -> HeadTrace:
    arrays = (source_ids, planes, ys, xs, k_bitmaps, valid_masks)
    if any(len(values) != TOKENS for values in arrays) or gates.shape != (TOKENS, ROLES):
        raise ValueError("一个 head trace 必须覆盖 450 个 source")
    expected = np.arange(TOKENS, dtype=np.int64)
    if not np.array_equal(np.asarray(source_ids, dtype=np.int64), expected):
        raise ValueError("source id 必须为 0..449")
    expected_plane = expected // (HEIGHT * WIDTH)
    expected_spatial = expected % (HEIGHT * WIDTH)
    if (
        not np.array_equal(np.asarray(planes, dtype=np.int64), expected_plane)
        or not np.array_equal(np.asarray(ys, dtype=np.int64), expected_spatial // WIDTH)
        or not np.array_equal(np.asarray(xs, dtype=np.int64), expected_spatial % WIDTH)
    ):
        raise ValueError("source id 与 plane/y/x 坐标不一致")
    sources = tuple(
        build_source_terms(
            int(source_ids[index]),
            int(planes[index]),
            int(ys[index]),
            int(xs[index]),
            int(k_bitmaps[index]),
            gates[index],
            int(valid_masks[index]),
        )
        for index in range(TOKENS)
    )
    return HeadTrace(sources=sources)


def _readout_cycles(valid: list[set[int]]) -> tuple[int, int]:
    valid_count = sum(len(addresses) for addresses in valid)
    # 无效地址一拍返回0；有效地址增加一拍同步read response。
    return valid_count, VECTOR_READ_WORDS + valid_count


def simulate_direct_window(heads: Iterable[HeadTrace]) -> BackendWindowResult:
    """B2v 公共边界：head0 reset，后续 head preserve，最后只读一次。"""
    valid: list[set[int]] = [set() for _ in range(BANKS)]
    results: list[BackendHeadResult] = []
    for trace in heads:
        result = BackendHeadResult()
        result.descriptor_captures = trace.active_sources
        for term in trace.terms:
            result.terms += 1
            result.destination_updates += len(term.destinations)
            repeated = any(address in valid[bank] for bank, address in term.destinations)
            result.term_issue_cycles += 1
            if repeated:
                result.term_issue_cycles += 1
                result.rmw_terms += 1
            for bank, address in term.destinations:
                if address in valid[bank]:
                    result.sram_reads += 1
                else:
                    result.first_touch_writes += 1
                    valid[bank].add(address)
                result.sram_writes += 1
        result.flush_cycles = 1
        result.service_cycles = (
            1
            + ACTIVE_BITMAP_SCAN_CYCLES
            + result.descriptor_captures
            + result.term_issue_cycles
            + result.flush_cycles
        )
        results.append(result)
    valid_count, readout = _readout_cycles(valid)
    return BackendWindowResult(
        heads=tuple(results),
        final_valid_addresses=valid_count,
        final_readout_cycles=readout,
        final_valid_sets=tuple(tuple(sorted(addresses)) for addresses in valid),
    )


@dataclass
class _Slot:
    address: int
    dirty: bool = False


def _source_targets(source: SourceTerms) -> list[int | None]:
    targets: list[int | None] = [None] * BANKS
    for term in source.terms:
        for bank, address in term.destinations:
            if targets[bank] is not None and targets[bank] != address:
                raise AssertionError("同一 source 在一个颜色 bank 命中多个地址")
            targets[bank] = address
    return targets


def _simulate_srac2_head(
    trace: HeadTrace, materialized: list[set[int]]
) -> BackendHeadResult:
    active_sources = [source for source in trace.sources if source.terms]
    slots: list[list[_Slot | None]] = [
        [None for _ in range(GASR2CP_CONTEXTS_PER_BANK)] for _ in range(BANKS)
    ]
    active_slot: list[int | None] = [None] * BANKS
    result = BackendHeadResult()
    result.descriptor_captures = len(active_sources)
    targets = [_source_targets(source) for source in active_sources]

    def prepare(bank: int, address: int) -> tuple[int, int]:
        for slot_index, slot in enumerate(slots[bank]):
            if slot is not None and slot.address == address:
                return slot_index, 0
        victim = 0 if active_slot[bank] != 0 else 1
        slot = slots[bank][victim]
        operations = 0
        if slot is not None and slot.dirty:
            result.sram_writes += 1
            operations += 1
            materialized[bank].add(slot.address)
        if address in materialized[bank]:
            result.sram_reads += 1
            operations += 1
        slots[bank][victim] = _Slot(address=address)
        return victim, operations

    if active_sources:
        first_operations = [0] * BANKS
        for bank, address in enumerate(targets[0]):
            if address is not None:
                active_slot[bank], first_operations[bank] = prepare(bank, address)
        result.stall_cycles += max(first_operations)

    for source_index, source in enumerate(active_sources):
        duration = len(source.terms)
        result.terms += duration
        result.destination_updates += sum(len(term.destinations) for term in source.terms)
        for bank, address in enumerate(targets[source_index]):
            if address is None:
                continue
            slot_index = active_slot[bank]
            slot = slots[bank][slot_index] if slot_index is not None else None
            if slot is None or slot.address != address:
                raise AssertionError("SRAC2 active slot 地址错误")
            slot.dirty = True
        if source_index + 1 == len(active_sources):
            continue
        next_active: list[int | None] = [None] * BANKS
        operations = [0] * BANKS
        for bank, address in enumerate(targets[source_index + 1]):
            if address is not None:
                next_active[bank], operations[bank] = prepare(bank, address)
        # Geometry is exposed one synchronous-read latency before descriptor
        # payload. Both candidates pay the same builder capture; only unhidden
        # 1RW writeback/refill work remains GASR-specific.
        result.stall_cycles += max(
            max(0, operation - (duration + RELATION_READ_LATENCY))
            for operation in operations
        )
        active_slot = next_active

    dirty_per_bank = [0] * BANKS
    for bank in range(BANKS):
        for slot in slots[bank]:
            if slot is not None and slot.dirty:
                result.sram_writes += 1
                dirty_per_bank[bank] += 1
                materialized[bank].add(slot.address)
    result.flush_cycles = max(dirty_per_bank, default=0) + 1
    result.term_issue_cycles = result.terms
    result.service_cycles = (
        1
        + ACTIVE_BITMAP_SCAN_CYCLES
        + result.descriptor_captures
        + result.terms
        + result.stall_cycles
        + result.flush_cycles
    )
    return result


def simulate_srac2_window(heads: Iterable[HeadTrace]) -> BackendWindowResult:
    materialized: list[set[int]] = [set() for _ in range(BANKS)]
    results = tuple(_simulate_srac2_head(trace, materialized) for trace in heads)
    valid_count, readout = _readout_cycles(materialized)
    return BackendWindowResult(
        heads=results,
        final_valid_addresses=valid_count,
        final_readout_cycles=readout,
        final_valid_sets=tuple(tuple(sorted(addresses)) for addresses in materialized),
    )


def _memo_admission(heads: tuple[HeadTrace, ...]) -> list[bool]:
    """严格复刻 vault 的 15+term<450 与 head-order 容量提交。"""
    admitted = [False] * len(heads)
    used = 0
    for index, trace in enumerate(heads):
        words = trace.active_sources
        critical = ACTIVE_BITMAP_SCAN_CYCLES + len(trace.terms) < RELATION_BUILD_CYCLES
        if critical and used + words <= RELATION_MEMO_CAPACITY_WORDS:
            admitted[index] = True
            used += words
    return admitted


def _path_cycles(
    backend_cycles: int,
    packet_words: int,
    *,
    replay: bool,
    first_tile_observe: bool = False,
    forced_replay_miss: bool = False,
) -> int:
    """Relation controller + producer + projection 的同边界并发上界。"""
    if replay:
        replay_backend = backend_cycles - ACTIVE_BITMAP_SCAN_CYCLES
        hit_cycles = REPLAY_CONTROL_CYCLES + max(packet_words + 1, replay_backend)
        if not forced_replay_miss:
            return hit_cycles
        recompute = RECOMPUTE_CONTROL_CYCLES + max(
            RELATION_BUILD_CYCLES, backend_cycles
        )
        return hit_cycles + REPLAY_MISS_FALLBACK_CYCLES + recompute
    memo_write = packet_words + 1 if first_tile_observe else 0
    return RECOMPUTE_CONTROL_CYCLES + max(
        RELATION_BUILD_CYCLES, backend_cycles, memo_write
    )


def candidate_window_cycles(
    heads: tuple[HeadTrace, ...], output_tiles: int
) -> dict[str, int]:
    """同一 sampled joint-window 的四候选配对周期。"""
    if not heads or output_tiles <= 0:
        raise ValueError("joint-window head/output tile 必须为正")
    direct = simulate_direct_window(heads)
    srac = simulate_srac2_window(heads)
    if direct.final_valid_sets != srac.final_valid_sets:
        raise AssertionError("Direct与GASR2C-P最终valid-address集合不一致")
    packet_words = [trace.active_sources for trace in heads]
    direct_service = [row.service_cycles for row in direct.heads]
    srac_service = [row.service_cycles for row in srac.heads]
    admitted = _memo_admission(heads)

    def total(
        backend: BackendWindowResult,
        service: list[int],
        memo: bool,
        admitted: list[bool],
    ) -> int:
        cycles = 0
        for output_tile in range(output_tiles):
            for head, backend_cycles in enumerate(service):
                replay = memo and output_tile > 0 and admitted[head]
                cycles += _path_cycles(
                    backend_cycles,
                    packet_words[head],
                    replay=replay,
                    first_tile_observe=memo and output_tile == 0,
                )
            cycles += backend.final_readout_cycles
            cycles += backend.scalar_serializer_cycles
        return cycles

    return {
        "c0_direct_recompute": total(
            direct, direct_service, False, [False] * len(heads)
        ),
        "c1_gasr2cp_recompute": total(
            srac, srac_service, False, [False] * len(heads)
        ),
        "c2_direct_erm7": total(
            direct, direct_service, True, admitted
        ),
        "c3_gasr2cp_erm7": total(
            srac, srac_service, True, admitted
        ),
    }
