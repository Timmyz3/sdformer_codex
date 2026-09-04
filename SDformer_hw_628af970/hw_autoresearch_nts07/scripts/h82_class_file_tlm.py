#!/usr/bin/env python3
"""Transaction-level Class File engine. Not synthesizable RTL.

Consumes a 450-token Q7 score row and emits (pair_id, k_mask, gate_c) in
ST_EXPAND order: occupied class id ascending, then pair_id ascending.
This is the schedule a later directory rewrite must match.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from scripts.h82_class_file_reference import (
    C_MAX_DEFAULT,
    ClassFile,
    build_class_file,
    integer_class_major_gates,
    q7_codes,
)


@dataclass(frozen=True)
class ExpandBeat:
    class_id: int
    pair_id: int
    k_mask: int
    gate_c_q17: int


@dataclass(frozen=True)
class EngineReport:
    n_classify: int
    n_shiftmax: int
    n_expand: int
    protocol_error: bool
    beats: tuple[ExpandBeat, ...]
    class_file: ClassFile


def run_row(scores: np.ndarray, *, c_max: int = C_MAX_DEFAULT) -> EngineReport:
    class_file = build_class_file(scores, preserve_mean=True)
    codes = q7_codes(scores)
    int_gates = integer_class_major_gates(codes, preserve_mean=True)
    gate_by_class = {}
    for index, code in enumerate(codes.tolist()):
        gate_by_class.setdefault(int(code), int(int_gates[index]))
    error = class_file.n_occupied > c_max
    beats: list[ExpandBeat] = []
    if not error:
        for record in sorted(class_file.records, key=lambda item: item.class_id):
            for member in sorted(record.members, key=lambda item: item.pair_id):
                beats.append(
                    ExpandBeat(
                        class_id=record.class_id,
                        pair_id=member.pair_id,
                        k_mask=member.k_mask,
                        gate_c_q17=gate_by_class[record.class_id],
                    )
                )
    return EngineReport(
        n_classify=450,
        n_shiftmax=class_file.n_occupied,
        n_expand=len(beats),
        protocol_error=error,
        beats=tuple(beats),
        class_file=class_file,
    )


def tokens_covered(report: EngineReport) -> int:
    return sum(beat.k_mask.bit_count() for beat in report.beats)
