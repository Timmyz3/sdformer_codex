#!/usr/bin/env python3
"""Deterministic C4/C5 exact-relation-memo capacity baselines.

C4 is deliberately a relaxed, payload-only comparison.  Its 561,600-bit
budget holds whole 112-bit records; tag and valid state is reported separately
and never deducted from that payload budget.  C5 admits the complete stage-3
working set and is therefore a larger, non-area-matched upper bound.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Sequence


RELATION_RECORD_BITS = 112

C4_PAYLOAD_CAPACITY_BITS = 561_600
C4_PAYLOAD_CAPACITY_RECORDS = (
    C4_PAYLOAD_CAPACITY_BITS // RELATION_RECORD_BITS
)
C4_PAYLOAD_STATIC_UNUSED_BITS = (
    C4_PAYLOAD_CAPACITY_BITS % RELATION_RECORD_BITS
)

C5_STAGE3_WORST_CASE_RECORDS = 10_800
C5_STAGE3_WORST_CASE_PAYLOAD_BITS = 1_209_600

ORACLE_TIE_BREAK = (
    "maximize saved cycles, then minimize admitted payload records, then "
    "choose the lexicographically smallest tuple of input head indices"
)


@dataclass(frozen=True)
class HeadEntry:
    """One indivisible head-sized memo entry.

    ``accesses`` is the number of output-tile accesses to this relation.  A
    resident entry has one initial miss/fill followed by ``accesses - 1``
    hit/read replays.  A nonresident entry takes the explicit miss path on
    every access.  Cycle costs are totals per entry event, not values inferred
    from the record count.

    Tag and valid widths are mandatory, separately accounted metadata.  They
    do not consume the relaxed C4 payload capacity.
    """

    head: int
    records: int
    accesses: int
    miss_cycles: int
    fill_cycles: int
    hit_cycles: int
    read_cycles: int
    tag_bits: int
    valid_bits: int = 1

    def __post_init__(self) -> None:
        if self.head < 0:
            raise ValueError("head must be nonnegative")
        if self.records < 0:
            raise ValueError("records must be nonnegative")
        if self.accesses <= 0:
            raise ValueError("accesses must be positive")
        cycle_values = (
            self.miss_cycles,
            self.fill_cycles,
            self.hit_cycles,
            self.read_cycles,
        )
        if any(value < 0 for value in cycle_values):
            raise ValueError("entry cycle costs must be nonnegative")
        if self.tag_bits <= 0 or self.valid_bits <= 0:
            raise ValueError("tag_bits and valid_bits must be positive")

    @property
    def nonresident_cycles(self) -> int:
        return self.accesses * self.miss_cycles

    @property
    def resident_cycles(self) -> int:
        replay_count = self.accesses - 1
        return (
            self.miss_cycles
            + self.fill_cycles
            + replay_count * (self.hit_cycles + self.read_cycles)
        )

    @property
    def saved_cycles(self) -> int:
        return self.nonresident_cycles - self.resident_cycles


@dataclass(frozen=True)
class MemoResult:
    baseline: Literal["c4_oracle", "c4_first_fit", "c5_full"]
    admission: tuple[bool, ...]
    admitted_heads: tuple[int, ...]
    cycles: int
    saved_cycles: int
    miss_path_cycles: int
    fill_path_cycles: int
    hit_path_cycles: int
    read_path_cycles: int
    record_reads: int
    record_writes: int
    entry_misses: int
    entry_fills: int
    entry_hits: int
    capacity_misses: int
    payload_capacity_bits: int
    payload_capacity_records: int
    payload_used_bits: int
    payload_used_records: int
    unused_bits: int
    tag_bits: int
    valid_bits: int
    metadata_bits: int
    metadata_in_payload_capacity: bool = False

    @property
    def misses(self) -> int:
        """Alias for the total compulsory and nonresident entry misses."""

        return self.entry_misses

    @property
    def allocated_state_bits(self) -> int:
        """Payload allocation plus separately reported tag/valid state."""

        return self.payload_capacity_bits + self.metadata_bits

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["misses"] = self.misses
        result["allocated_state_bits"] = self.allocated_state_bits
        return result


def _normalize_heads(heads: Iterable[HeadEntry]) -> tuple[HeadEntry, ...]:
    result = tuple(heads)
    identities = [entry.head for entry in result]
    if len(set(identities)) != len(identities):
        raise ValueError("head identities must be unique")
    if identities != sorted(identities):
        raise ValueError("heads must be supplied in strict input-head order")
    return result


def _capacity_records(capacity_bits: int) -> int:
    if capacity_bits < 0:
        raise ValueError("capacity_bits must be nonnegative")
    return capacity_bits // RELATION_RECORD_BITS


def _best_saved_by_exact_records(
    entries: Sequence[HeadEntry],
    capacity_records: int,
) -> dict[int, int]:
    best = {0: 0}
    for entry in entries:
        updated = dict(best)
        for used_records, saved_cycles in best.items():
            candidate_records = used_records + entry.records
            if candidate_records > capacity_records:
                continue
            candidate_saved = saved_cycles + entry.saved_cycles
            if candidate_saved > updated.get(candidate_records, candidate_saved - 1):
                updated[candidate_records] = candidate_saved
        best = updated
    return best


def offline_optimal_admission(
    heads: Iterable[HeadEntry],
    *,
    capacity_bits: int = C4_PAYLOAD_CAPACITY_BITS,
) -> tuple[bool, ...]:
    """Solve the whole-head 0/1 knapsack with the frozen deterministic tie-break."""

    entries = _normalize_heads(heads)
    capacity_records = _capacity_records(capacity_bits)
    all_best = _best_saved_by_exact_records(entries, capacity_records)
    best_saved = max(all_best.values())
    best_records = min(
        records for records, saved in all_best.items() if saved == best_saved
    )

    # Suffix tables allow exact lexicographic reconstruction without pruning a
    # zero-record/zero-value early index that may prefix a later optimal item.
    suffix = [dict() for _ in range(len(entries) + 1)]
    suffix[-1] = {0: 0}
    for index in range(len(entries) - 1, -1, -1):
        suffix[index] = _best_saved_by_exact_records(
            entries[index:], best_records
        )

    selected: set[int] = set()
    remaining_records = best_records
    remaining_saved = best_saved
    for index, entry in enumerate(entries):
        if remaining_records == 0 and remaining_saved == 0:
            break
        next_records = remaining_records - entry.records
        next_saved = remaining_saved - entry.saved_cycles
        if (
            next_records >= 0
            and suffix[index + 1].get(next_records) == next_saved
        ):
            selected.add(index)
            remaining_records = next_records
            remaining_saved = next_saved
    if remaining_records != 0 or remaining_saved != 0:
        raise AssertionError("failed to reconstruct deterministic knapsack result")
    return tuple(index in selected for index in range(len(entries)))


def head_order_first_fit_admission(
    heads: Iterable[HeadEntry],
    *,
    capacity_bits: int = C4_PAYLOAD_CAPACITY_BITS,
) -> tuple[bool, ...]:
    """Admit in fixed input-head order, skipping entries that do not fit.

    A skipped entry is never reconsidered and an admitted entry is never
    replaced.  Zero-record entries fit without consuming payload capacity.
    """

    entries = _normalize_heads(heads)
    capacity_records = _capacity_records(capacity_bits)
    used_records = 0
    admission: list[bool] = []
    for entry in entries:
        fits = (
            entry.saved_cycles > 0
            and used_records + entry.records <= capacity_records
        )
        admission.append(fits)
        if fits:
            used_records += entry.records
    return tuple(admission)


def evaluate_admission(
    heads: Iterable[HeadEntry],
    admission: Sequence[bool],
    *,
    baseline: Literal["c4_oracle", "c4_first_fit", "c5_full"],
    capacity_bits: int,
) -> MemoResult:
    """Apply an admission vector and produce one explicit cycle/traffic ledger."""

    entries = _normalize_heads(heads)
    selected = tuple(bool(value) for value in admission)
    if len(selected) != len(entries):
        raise ValueError("admission length must match heads")

    capacity_records = _capacity_records(capacity_bits)
    used_records = sum(
        entry.records for entry, admitted in zip(entries, selected, strict=True)
        if admitted
    )
    if used_records > capacity_records:
        raise ValueError("admission exceeds payload capacity")

    cycles = 0
    no_memo_cycles = 0
    miss_path_cycles = 0
    fill_path_cycles = 0
    hit_path_cycles = 0
    read_path_cycles = 0
    record_reads = 0
    record_writes = 0
    entry_misses = 0
    entry_fills = 0
    entry_hits = 0
    capacity_misses = 0
    tag_bits = 0
    valid_bits = 0

    for entry, admitted in zip(entries, selected, strict=True):
        no_memo_cycles += entry.nonresident_cycles
        if admitted:
            replays = entry.accesses - 1
            cycles += entry.resident_cycles
            miss_path_cycles += entry.miss_cycles
            fill_path_cycles += entry.fill_cycles
            hit_path_cycles += replays * entry.hit_cycles
            read_path_cycles += replays * entry.read_cycles
            record_writes += entry.records
            record_reads += entry.records * replays
            entry_misses += 1
            entry_fills += 1
            entry_hits += replays
            tag_bits += entry.tag_bits
            valid_bits += entry.valid_bits
        else:
            cycles += entry.nonresident_cycles
            miss_path_cycles += entry.nonresident_cycles
            entry_misses += entry.accesses
            capacity_misses += entry.accesses

    used_bits = used_records * RELATION_RECORD_BITS
    return MemoResult(
        baseline=baseline,
        admission=selected,
        admitted_heads=tuple(
            entry.head
            for entry, admitted in zip(entries, selected, strict=True)
            if admitted
        ),
        cycles=cycles,
        saved_cycles=no_memo_cycles - cycles,
        miss_path_cycles=miss_path_cycles,
        fill_path_cycles=fill_path_cycles,
        hit_path_cycles=hit_path_cycles,
        read_path_cycles=read_path_cycles,
        record_reads=record_reads,
        record_writes=record_writes,
        entry_misses=entry_misses,
        entry_fills=entry_fills,
        entry_hits=entry_hits,
        capacity_misses=capacity_misses,
        payload_capacity_bits=capacity_bits,
        payload_capacity_records=capacity_records,
        payload_used_bits=used_bits,
        payload_used_records=used_records,
        unused_bits=capacity_bits - used_bits,
        tag_bits=tag_bits,
        valid_bits=valid_bits,
        metadata_bits=tag_bits + valid_bits,
    )


def evaluate_c4_oracle(
    heads: Iterable[HeadEntry],
    *,
    capacity_bits: int = C4_PAYLOAD_CAPACITY_BITS,
) -> MemoResult:
    entries = _normalize_heads(heads)
    admission = offline_optimal_admission(entries, capacity_bits=capacity_bits)
    return evaluate_admission(
        entries,
        admission,
        baseline="c4_oracle",
        capacity_bits=capacity_bits,
    )


def evaluate_c4_first_fit(
    heads: Iterable[HeadEntry],
    *,
    capacity_bits: int = C4_PAYLOAD_CAPACITY_BITS,
) -> MemoResult:
    entries = _normalize_heads(heads)
    admission = head_order_first_fit_admission(entries, capacity_bits=capacity_bits)
    return evaluate_admission(
        entries,
        admission,
        baseline="c4_first_fit",
        capacity_bits=capacity_bits,
    )


def evaluate_c5_full(heads: Iterable[HeadEntry]) -> MemoResult:
    """Evaluate the full memo; stage-3's 1,209,600 payload bits are reported."""

    entries = _normalize_heads(heads)
    total_records = sum(entry.records for entry in entries)
    if total_records > C5_STAGE3_WORST_CASE_RECORDS:
        raise ValueError("heads exceed the frozen stage-3 full-memo record bound")
    return evaluate_admission(
        entries,
        (True,) * len(entries),
        baseline="c5_full",
        capacity_bits=C5_STAGE3_WORST_CASE_PAYLOAD_BITS,
    )


# Short names retained for callers that treat these as simulators.
simulate_c4_oracle = evaluate_c4_oracle
simulate_c4_first_fit = evaluate_c4_first_fit
simulate_c5_full = evaluate_c5_full
