#!/usr/bin/env python3
"""H85 row-delta Class File — hardware TLM, not production RTL.

Algorithm H85 (disk overlay, not the live H82 trainer) changed the operator
again: Shiftmax is per spatial row of 15 tokens, not over the 450-token
window. Adjacent rows expose class-set shared/insert/delete.

This file is the DATE object that claim would need:

- expand never allocates a length-450 token gate
- dest K stays dest-owned: attn[i] = K[i] * gate_c(class(i))
- adjacent-row directory can be a class-set delta
- the 41% store is the *member* CSR; class-set reuse does not shrink it

It also proves the identity split: H85 row-major gates are not H82
window-major gates. Do not bind this SHA to the live H82 ft15.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.h82_class_file_reference import (  # noqa: E402
    PAIRS,
    SCORE_STEP,
    TOKENS,
    build_class_file,
    class_center,
    jaccard,
    q7_codes,
    shiftmax_1d,
)

SPATIAL = 15
T_STEPS = 2


def as_grid(values: np.ndarray) -> np.ndarray:
    """[450] with T0-then-T1 layout -> [2, 15, 15]."""

    flat = np.asarray(values).reshape(-1)
    if flat.size != TOKENS:
        raise ValueError(f"expected {TOKENS} tokens, got {flat.size}")
    return flat.reshape(T_STEPS, SPATIAL, SPATIAL)


@dataclass(frozen=True)
class RowClass:
    class_id: int
    members: tuple[int, ...]  # columns in this spatial row
    gate_c: float


@dataclass(frozen=True)
class RowFile:
    time_idx: int
    row: int
    classes: tuple[RowClass, ...]

    @property
    def class_ids(self) -> frozenset[int]:
        return frozenset(item.class_id for item in self.classes)

    def members_of(self, class_id: int) -> frozenset[int]:
        for item in self.classes:
            if item.class_id == class_id:
                return frozenset(item.members)
        return frozenset()


@dataclass(frozen=True)
class RowDelta:
    time_idx: int
    prev_row: int
    curr_row: int
    shared: tuple[int, ...]
    insert: tuple[int, ...]
    delete: tuple[int, ...]
    member_jaccard_surviving: float
    class_set_jaccard: float
    member_insert: tuple[tuple[int, int], ...]  # (class_id, col)
    member_delete: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ExpandBeat:
    time_idx: int
    row: int
    col: int
    class_id: int
    gate_c: float


def build_row_files(scores: np.ndarray) -> tuple[RowFile, ...]:
    grid = as_grid(scores)
    files: list[RowFile] = []
    for time_idx in range(T_STEPS):
        for row in range(SPATIAL):
            row_scores = grid[time_idx, row]
            codes = q7_codes(row_scores)
            occupied = np.unique(codes)
            class_scores = np.asarray(
                [class_center(int(class_id)) for class_id in occupied], dtype=np.float64
            )
            gates = shiftmax_1d(class_scores)
            classes = []
            for class_id, gate in zip(occupied.tolist(), gates.tolist(), strict=True):
                members = tuple(int(col) for col in np.flatnonzero(codes == class_id))
                classes.append(
                    RowClass(class_id=int(class_id), members=members, gate_c=float(gate))
                )
            files.append(RowFile(time_idx=time_idx, row=row, classes=tuple(classes)))
    return tuple(files)


def row_deltas(files: tuple[RowFile, ...]) -> tuple[RowDelta, ...]:
    by_key = {(item.time_idx, item.row): item for item in files}
    deltas: list[RowDelta] = []
    for time_idx in range(T_STEPS):
        for row in range(1, SPATIAL):
            prev = by_key[(time_idx, row - 1)]
            curr = by_key[(time_idx, row)]
            shared = prev.class_ids & curr.class_ids
            insert = curr.class_ids - prev.class_ids
            delete = prev.class_ids - curr.class_ids
            member_scores = [
                jaccard(prev.members_of(class_id), curr.members_of(class_id))
                for class_id in shared
            ]
            member_insert = []
            member_delete = []
            for class_id in shared:
                now = curr.members_of(class_id)
                was = prev.members_of(class_id)
                member_insert.extend((class_id, col) for col in sorted(now - was))
                member_delete.extend((class_id, col) for col in sorted(was - now))
            deltas.append(
                RowDelta(
                    time_idx=time_idx,
                    prev_row=row - 1,
                    curr_row=row,
                    shared=tuple(sorted(shared)),
                    insert=tuple(sorted(insert)),
                    delete=tuple(sorted(delete)),
                    member_jaccard_surviving=(
                        float(np.mean(member_scores)) if member_scores else 1.0
                    ),
                    class_set_jaccard=jaccard(prev.class_ids, curr.class_ids),
                    member_insert=tuple(member_insert),
                    member_delete=tuple(member_delete),
                )
            )
    return tuple(deltas)


def expand_without_token_gate(
    scores: np.ndarray,
    k: np.ndarray,
    files: tuple[RowFile, ...] | None = None,
) -> tuple[np.ndarray, tuple[ExpandBeat, ...]]:
    """attn[i] = K[i] * gate_c(class(i)), written class-major per row.

    The public result is [450, D]. No length-450 gate vector is created.
    """

    files = files or build_row_files(scores)
    k_grid = as_grid(k) if k.ndim == 1 else np.asarray(k).reshape(T_STEPS, SPATIAL, SPATIAL, -1)
    if k_grid.ndim == 3:
        k_grid = k_grid[..., None]
    attn = np.zeros_like(k_grid, dtype=np.float64)
    beats: list[ExpandBeat] = []
    for row_file in files:
        for record in row_file.classes:
            for col in record.members:
                attn[row_file.time_idx, row_file.row, col] = (
                    k_grid[row_file.time_idx, row_file.row, col] * record.gate_c
                )
                beats.append(
                    ExpandBeat(
                        time_idx=row_file.time_idx,
                        row=row_file.row,
                        col=col,
                        class_id=record.class_id,
                        gate_c=record.gate_c,
                    )
                )
    return attn.reshape(TOKENS, k_grid.shape[-1]), tuple(beats)


def h82_window_gates(scores: np.ndarray) -> np.ndarray:
    return build_class_file(scores, preserve_mean=False).gate_tokens()


def h85_row_gates(files: tuple[RowFile, ...]) -> np.ndarray:
    gates = np.zeros(TOKENS, dtype=np.float64)
    for row_file in files:
        for record in row_file.classes:
            for col in record.members:
                token = row_file.time_idx * PAIRS + row_file.row * SPATIAL + col
                gates[token] = record.gate_c
    return gates


def storage_bits(files: tuple[RowFile, ...], deltas: tuple[RowDelta, ...]) -> dict:
    full = 0
    for row_file in files:
        for record in row_file.classes:
            full += 9 + 4 + 9 + 4 * len(record.members)  # id, n, gate, cols
    class_delta = 0
    member_delta = 0
    for delta in deltas:
        class_delta += 9 * (len(delta.shared) + len(delta.insert) + len(delta.delete))
        member_delta += 13 * (len(delta.member_insert) + len(delta.member_delete))
    first_rows = [row_file for row_file in files if row_file.row == 0]
    first_bits = 0
    for row_file in first_rows:
        for record in row_file.classes:
            first_bits += 9 + 4 + 9 + 4 * len(record.members)
    return {
        "full_row_files_bits": full,
        "class_set_delta_bits": first_bits + class_delta,
        "class_and_member_delta_bits": first_bits + class_delta + member_delta,
        "mean_class_set_jaccard": float(np.mean([d.class_set_jaccard for d in deltas])),
        "mean_member_jaccard": float(np.mean([d.member_jaccard_surviving for d in deltas])),
        "mean_member_edits": float(
            np.mean([len(d.member_insert) + len(d.member_delete) for d in deltas])
        ),
    }


def compare_operators(scores: np.ndarray) -> dict:
    files = build_row_files(scores)
    g82 = h82_window_gates(scores)
    g85 = h85_row_gates(files)
    return {
        "maxabs_h82_vs_h85": float(np.max(np.abs(g82 - g85))),
        "n_h82_occupied_window": int(build_class_file(scores, preserve_mean=False).n_occupied),
        "mean_h85_occupied_row": float(np.mean([len(item.classes) for item in files])),
    }


def demo(seed: int = 85) -> dict:
    rng = np.random.default_rng(seed)
    scores = rng.normal(0.0, 0.35, size=TOKENS).clip(-2, 2)
    # spatially smooth a copy
    grid = as_grid(scores)
    smooth = grid.copy()
    for _ in range(6):
        pad = np.pad(smooth, ((0, 0), (1, 1), (1, 1)), mode="edge")
        smooth = 0.6 * smooth + 0.4 * 0.25 * (
            pad[:, 0:-2, 1:-1] + pad[:, 2:, 1:-1] + pad[:, 1:-1, 0:-2] + pad[:, 1:-1, 2:]
        )
    reports = {}
    for name, field in (("raw", grid.reshape(-1)), ("spatial_smooth", smooth.reshape(-1))):
        files = build_row_files(field)
        deltas = row_deltas(files)
        k = np.ones((TOKENS, 2), dtype=np.float64)
        attn, beats = expand_without_token_gate(field, k, files)
        reports[name] = {
            "compare": compare_operators(field),
            "storage": storage_bits(files, deltas),
            "n_expand_beats": len(beats),
            "attn_row_sum_abs": float(np.abs(attn).sum()),
        }
    return reports


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "h82_class_file_isa_20260817"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "h85_row_delta_class_file_v1",
        "status": "HARDWARE_TLM_NOT_H82_IDENTITY",
        "live_h82_must_not_use_this_sha": True,
        "demo": demo(),
        "verdict": (
            "H85 is a new operator (row Shiftmax) plus a class-set delta. "
            "Class-set reuse does not cut the member CSR. Not 4.0, not H82."
        ),
    }
    (out / "h85_row_delta.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
