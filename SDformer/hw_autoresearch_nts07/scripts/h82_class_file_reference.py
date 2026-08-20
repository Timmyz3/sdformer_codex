#!/usr/bin/env python3
"""H82 Class File ISA reference. No RTL, no GPU, no docs/359 rewrite.

The live H67/H81 directory already streams occupied classes and then a 450-entry
active list, and the Shiftmax denominator is

    row_sum = sum_c exp2(s_c - s_max) * multiplicity_c

That is C7: multiplicity-weighted class Shiftmax, algebraically token Shiftmax
when every member of a class shares one Q7 code. H82 changes the operand:

    row_sum = sum_c exp2(s_c - s_max)          # one vote per occupied class
    gate_token = gather(gate_c, class_id)
    K is expanded after the class nonlinear

This file is the bit-level contract the hardware directory must match after an
H82 rank-1 checkpoint exists. It is not a DATE 4.0 claim by itself.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCORE_LO = -2.0
SCORE_HI = 2.0
SCORE_STEP = 1.0 / 128.0
N_BINS = int(round((SCORE_HI - SCORE_LO) / SCORE_STEP)) + 1  # 513
PAIRS = 225
TOKENS = 2 * PAIRS
C_MAX_DEFAULT = 64
LUT_Q8 = (256, 245, 234, 224, 215, 205, 196, 188,
          181, 173, 165, 158, 152, 145, 139, 133)
H67_MAX_SCORE = 162

# Bits for the storage-object model. Not DC/PTPX.
BITS_PAIR_ID = 8
BITS_SCORE_Q7 = 16
BITS_TMASK = 2
BITS_KMASK = 2
BITS_CLASS_ID = 9
BITS_MULT = 9
BITS_GATE_Q17 = 9
BITS_CSR_PTR = 9
BITS_COUNT = 10


@dataclass(frozen=True)
class MemberRecord:
    pair_id: int
    k_mask: int  # bit0 = T0, bit1 = T1


@dataclass(frozen=True)
class ClassRecord:
    class_id: int
    multiplicity: int
    temporal_mask: int
    class_score: float
    gate_c: float
    members: tuple[MemberRecord, ...]


@dataclass(frozen=True)
class ClassFile:
    records: tuple[ClassRecord, ...]
    codes: tuple[int, ...]
    n_tokens: int
    preserve_mean: bool

    @property
    def n_occupied(self) -> int:
        return len(self.records)

    def gate_tokens(self) -> np.ndarray:
        gate_by_id = {record.class_id: record.gate_c for record in self.records}
        return np.asarray([gate_by_id[code] for code in self.codes], dtype=np.float64)


def q7_codes(scores: np.ndarray) -> np.ndarray:
    """Software H82 class_id: round((s - (-2)) / (1/128)) clipped to [0, 512]."""

    flat = np.asarray(scores, dtype=np.float64).reshape(-1)
    codes = np.rint((flat - SCORE_LO) / SCORE_STEP).astype(np.int64)
    return np.clip(codes, 0, N_BINS - 1)


def class_center(class_id: int) -> float:
    return SCORE_LO + SCORE_STEP * int(class_id)


def shiftmax_1d(values: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """BSA Shiftmax: 2^(x-max) / 2^ceil(log2(sum))."""

    shifted = values - np.max(values)
    numerator = np.power(2.0, shifted)
    total = float(np.clip(numerator.sum(), eps, None))
    denom_power = math.ceil(math.log2(total))
    return numerator / (2.0 ** denom_power)


def _pair_layout(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if codes.size != TOKENS:
        raise ValueError(f"Class File T=2 layout expects {TOKENS} tokens, got {codes.size}")
    return codes[:PAIRS], codes[PAIRS:]


def build_class_file(
    scores: np.ndarray,
    *,
    preserve_mean: bool = True,
    eps: float = 1.0e-6,
) -> ClassFile:
    """Emit the H82 ISA: occupied (class_id, multiplicity, mask, gate_c) + member CSR."""

    codes = q7_codes(scores)
    left, right = _pair_layout(codes)
    occupied_ids = np.unique(codes)
    class_scores = np.full(occupied_ids.shape, -1.0e4, dtype=np.float64)
    records: list[ClassRecord] = []
    for index, class_id in enumerate(occupied_ids):
        class_scores[index] = class_center(int(class_id))
    gates = shiftmax_1d(class_scores, eps=eps)
    if preserve_mean:
        gates = gates * float(TOKENS)
    for index, class_id in enumerate(occupied_ids):
        members: list[MemberRecord] = []
        temporal = 0
        for pair_id in range(PAIRS):
            mask = (1 if left[pair_id] == class_id else 0) | (
                2 if right[pair_id] == class_id else 0
            )
            if mask:
                members.append(MemberRecord(pair_id=pair_id, k_mask=mask))
                temporal |= mask
        multiplicity = int(sum(record.k_mask.bit_count() for record in members))
        records.append(
            ClassRecord(
                class_id=int(class_id),
                multiplicity=multiplicity,
                temporal_mask=temporal,
                class_score=float(class_scores[index]),
                gate_c=float(gates[index]),
                members=tuple(members),
            )
        )
    return ClassFile(
        records=tuple(records),
        codes=tuple(int(code) for code in codes),
        n_tokens=TOKENS,
        preserve_mean=preserve_mean,
    )


def c7_multiplicity_weighted_gates(
    scores: np.ndarray,
    *,
    preserve_mean: bool = True,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """Existing RTL / C7: Shiftmax over tokens that share a Q7 code.

    Equivalent to class Shiftmax with row_sum_c = exp_c * multiplicity_c.
    """

    codes = q7_codes(scores)
    occupied, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
    class_scores = np.asarray([class_center(int(class_id)) for class_id in occupied], dtype=np.float64)
    shifted = class_scores - class_scores.max()
    exp = np.power(2.0, shifted)
    weighted = exp * counts.astype(np.float64)
    total = float(np.clip(weighted.sum(), eps, None))
    denom = 2.0 ** math.ceil(math.log2(total))
    gate_c = exp / denom
    if preserve_mean:
        gate_c = gate_c * float(codes.size)
    return gate_c[inverse]


def token_shiftmax_gates(
    scores: np.ndarray,
    *,
    preserve_mean: bool = True,
    eps: float = 1.0e-6,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    gates = shiftmax_1d(values, eps=eps)
    if preserve_mean:
        gates = gates * float(values.size)
    return gates


def exp2_q8(delta_q7: int) -> int:
    if delta_q7 >= 0:
        return 256
    absolute = -int(delta_q7)
    integer_shift = min(absolute >> 7, 8)
    fraction = absolute & 127
    fraction_index = min((fraction + 7) // 8, 15)
    return LUT_Q8[fraction_index] >> integer_shift


def _round_shift_even(value: int, shift: int) -> int:
    if shift == 0:
        return value
    quotient, remainder = divmod(value, 1 << shift)
    half = 1 << (shift - 1)
    if remainder > half or (remainder == half and quotient & 1):
        quotient += 1
    return quotient


def gate_q17(exp_q8: int, row_sum_q8: int, n_tokens: int, preserve_mean: bool) -> int:
    shift = (row_sum_q8 - 1).bit_length() if row_sum_q8 > 0 else 0
    token_scale = n_tokens if preserve_mean else 1
    code = _round_shift_even(exp_q8 * token_scale * 128, shift) if row_sum_q8 else 0
    return min(256, max(0, code))


def integer_class_major_gates(
    codes: np.ndarray,
    *,
    preserve_mean: bool = True,
) -> np.ndarray:
    """Integer one-vote Class File Shiftmax using the existing exp2 LUT.

    class_score_q7 is the signed Q7 code of the bin center, i.e. class_id - 256
    because bin 256 is 0.0. Denominator does *not* multiply multiplicity.
    """

    occupied = np.unique(codes)
    score_q7 = occupied.astype(np.int64) - 256
    row_max = int(score_q7.max())
    exp = np.asarray([exp2_q8(int(score) - row_max) for score in score_q7], dtype=np.int64)
    row_sum = int(exp.sum())
    gate_c = np.asarray(
        [gate_q17(int(value), row_sum, TOKENS, preserve_mean) for value in exp],
        dtype=np.int64,
    )
    index = {int(class_id): slot for slot, class_id in enumerate(occupied)}
    return np.asarray([gate_c[index[int(code)]] for code in codes], dtype=np.int64)


def integer_c7_gates(
    codes: np.ndarray,
    *,
    preserve_mean: bool = True,
) -> np.ndarray:
    occupied, inverse, counts = np.unique(codes, return_inverse=True, return_counts=True)
    score_q7 = occupied.astype(np.int64) - 256
    row_max = int(score_q7.max())
    exp = np.asarray([exp2_q8(int(score) - row_max) for score in score_q7], dtype=np.int64)
    row_sum = int((exp * counts.astype(np.int64)).sum())
    gate_c = np.asarray(
        [gate_q17(int(value), row_sum, TOKENS, preserve_mean) for value in exp],
        dtype=np.int64,
    )
    return gate_c[inverse]


def class_set(codes: np.ndarray) -> set[int]:
    return {int(code) for code in np.unique(codes)}


def member_sets(codes: np.ndarray) -> dict[int, frozenset[int]]:
    grouped: dict[int, set[int]] = {}
    for index, code in enumerate(codes.tolist()):
        grouped.setdefault(int(code), set()).add(int(index))
    return {class_id: frozenset(members) for class_id, members in grouped.items()}


def jaccard(left: set[int] | frozenset[int], right: set[int] | frozenset[int]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def member_jaccard_surviving(codes_a: np.ndarray, codes_b: np.ndarray) -> dict[str, float]:
    set_a = class_set(codes_a)
    set_b = class_set(codes_b)
    members_a = member_sets(codes_a)
    members_b = member_sets(codes_b)
    surviving = set_a & set_b
    member_scores = [
        jaccard(members_a[class_id], members_b[class_id]) for class_id in surviving
    ]
    return {
        "class_set_jaccard": jaccard(set_a, set_b),
        "class_retention": (len(surviving) / len(set_a)) if set_a else 1.0,
        "member_jaccard_surviving": float(np.mean(member_scores)) if member_scores else 1.0,
        "n_occupied_a": float(len(set_a)),
        "n_occupied_b": float(len(set_b)),
        "n_surviving": float(len(surviving)),
    }


def window_codes(field: np.ndarray, row: int, col: int, spatial: int = 15) -> np.ndarray:
    """field shape (H, W, 2) -> 450 class codes, T0 then T1, matching H82 reshape."""

    patch = field[row:row + spatial, col:col + spatial]
    if patch.shape != (spatial, spatial, 2):
        raise ValueError(f"window {row},{col} out of range for {field.shape}")
    tokens = np.concatenate([patch[:, :, 0].reshape(-1), patch[:, :, 1].reshape(-1)])
    return q7_codes(tokens)


def smooth_field(field: np.ndarray, steps: int = 8, strength: float = 0.35) -> np.ndarray:
    """Intra-window spatial TV proxy: average with 4-neighbors."""

    out = field.astype(np.float64).copy()
    for _ in range(steps):
        padded = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
        neighbor = (
            padded[0:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, 0:-2] + padded[1:-1, 2:]
        ) * 0.25
        out = (1.0 - strength) * out + strength * neighbor
    return out


def sliding_window_study(seed: int = 82, height: int = 20, width: int = 20) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    raw = rng.normal(0.0, 0.45, size=(height, width, 2)).clip(SCORE_LO, SCORE_HI)
    smooth = smooth_field(raw)
    rows = []
    for field, name in ((raw, "raw"), (smooth, "spatial_tv")):
        metrics = []
        for row in range(height - 15):
            for col in range(width - 15 - 1):
                here = window_codes(field, row, col)
                east = window_codes(field, row, col + 1)
                south = window_codes(field, row + 1, col) if row + 1 <= height - 15 else None
                item = member_jaccard_surviving(here, east)
                item["direction"] = "east"
                if south is not None:
                    south_item = member_jaccard_surviving(here, south)
                    item["south_member_jaccard"] = south_item["member_jaccard_surviving"]
                    item["south_class_jaccard"] = south_item["class_set_jaccard"]
                metrics.append(item)
        member = float(np.mean([item["member_jaccard_surviving"] for item in metrics]))
        class_j = float(np.mean([item["class_set_jaccard"] for item in metrics]))
        occupied = float(np.mean([item["n_occupied_a"] for item in metrics]))
        south_m = [item["south_member_jaccard"] for item in metrics if "south_member_jaccard" in item]
        rows.append(
            {
                "field": name,
                "mean_member_jaccard_east": member,
                "mean_class_jaccard_east": class_j,
                "mean_occupied": occupied,
                "mean_member_jaccard_south": float(np.mean(south_m)) if south_m else None,
                "n_pairs": len(metrics),
            }
        )
    return {
        "schema": "h82_c81_sliding_window_proxy_v1",
        "note": (
            "CPU proxy only. Spatial TV on a synthetic score field is what the "
            "live H82 regularizer optimizes. It is not H82-checkpoint evidence."
        ),
        "frozen_t450_member_jaccard": 0.30,
        "fields": rows,
    }


def storage_object_model(n_occupied: int = 10, n_members: int = 450) -> dict[str, Any]:
    """Compare the 450-entry scored directory with a score-less Class File + CSR."""

    old_active = TOKENS * (BITS_PAIR_ID + BITS_SCORE_Q7 + BITS_TMASK + BITS_KMASK)
    old_hist = (H67_MAX_SCORE + 1) * BITS_COUNT
    new_class = n_occupied * (BITS_CLASS_ID + BITS_MULT + BITS_TMASK + BITS_GATE_Q17 + BITS_CSR_PTR)
    new_csr = n_members * (BITS_PAIR_ID + BITS_KMASK)
    new_hist_if_dense = N_BINS * BITS_COUNT
    return {
        "old_h67_h81_bits": {
            "active_scored_list": old_active,
            "hist_163": old_hist,
            "total": old_active + old_hist,
            "note": "PTPX on the frozen pack put 41.1% in the 450-name directory, not the hist",
        },
        "h82_class_file_bits": {
            "occupied_records": new_class,
            "member_csr_scoreless": new_csr,
            "dense_513_hist_optional": new_hist_if_dense,
            "total_compact": new_class + new_csr,
            "saved_vs_old_active": old_active - (new_class + new_csr),
            "saved_is_mostly_score_ram": True,
        },
        "what_4_0_still_needs": (
            "Compact Class File without the 450-score RAM is a real object change "
            "but still stores ~450 memberships. Delta-directory only becomes the "
            "41% story if surviving-class member Jaccard rises well above 0.30."
        ),
        "n_occupied_assumed": n_occupied,
        "n_members_assumed": n_members,
    }


def schedule_contract() -> dict[str, Any]:
    return {
        "old": ["ST_IDLE", "ST_BUILD_scored_active+hist", "ST_CLASS_mult_weighted_denom", "ST_ACTIVE_recompute_token_exp", "ST_DONE"],
        "h82": ["ST_IDLE", "ST_CLASSIFY_into_class_file", "ST_SHIFTMAX_one_vote", "ST_EXPAND_broadcast_gate_c", "ST_DONE"],
        "forbidden_reuse": [
            "class_sum_term = exp_q8 * multiplicity",
            "active_score_store[450] as Shiftmax operand",
            "MAX_SCORE=162 popcount histogram as H82 class_id space",
            "Motion-XOR",
            "Local5 stencil",
        ],
        "c_max": C_MAX_DEFAULT,
        "class_id_space": N_BINS,
        "protocol_error_if_occupied_gt_c_max": True,
        "k_expand": "after class Shiftmax; one gate_c per class; member CSR walks K-store",
        "rtl_start_gate": "H82 rank-1 checkpoint + Class File stats + this SHA frozen",
    }


def file_as_json(class_file: ClassFile) -> dict[str, Any]:
    return {
        "n_occupied": class_file.n_occupied,
        "n_tokens": class_file.n_tokens,
        "preserve_mean": class_file.preserve_mean,
        "records": [
            {
                **{key: value for key, value in asdict(record).items() if key != "members"},
                "n_member_pairs": len(record.members),
                "members": [asdict(member) for member in record.members],
            }
            for record in class_file.records
        ],
    }


def run_self_check() -> dict[str, Any]:
    unequal = np.asarray([0.0] * 3 + [1.0] + [0.0] * (TOKENS - 4), dtype=np.float64)
    equal = np.zeros(TOKENS, dtype=np.float64)
    equal[1] = 1.0
    h82_uneq = build_class_file(unequal, preserve_mean=False)
    c7_uneq = c7_multiplicity_weighted_gates(unequal, preserve_mean=False)
    tok_uneq = token_shiftmax_gates(unequal, preserve_mean=False)
    h82_eq = build_class_file(equal[:2].tolist() + [0.0] * (TOKENS - 2), preserve_mean=False)
    # rebuild equal-multiplicity 2-token case on a padded row: two occupied classes
    # each with multiplicity 1 after taking first two tokens only — use a 450-row
    # with two singleton classes.
    two = np.zeros(TOKENS, dtype=np.float64)
    two[0] = 0.0
    two[1] = 1.0
    h82_two = build_class_file(two, preserve_mean=False).gate_tokens()
    tok_two = token_shiftmax_gates(two, preserve_mean=False)
    # Only the two occupied classes vote; tokens 2..449 share class 0 with token 0,
    # so this is NOT the 2-token unit-test case. Keep the 2-token algebra in tests.
    return {
        "n_bins": N_BINS,
        "unequal_h82_vs_token_maxabs": float(np.max(np.abs(h82_uneq.gate_tokens() - tok_uneq))),
        "unequal_c7_vs_token_maxabs": float(np.max(np.abs(c7_uneq - tok_uneq))),
        "occupied_unequal": h82_uneq.n_occupied,
        "two_occupied_in_padded_row": int(np.unique(q7_codes(two)).size),
        "padded_two_h82_vs_token": float(np.max(np.abs(h82_two - tok_two))),
        "h82_eq_occupied": h82_eq.n_occupied,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "results" / "h82_class_file_isa_20260817"
    out.mkdir(parents=True, exist_ok=True)
    study = sliding_window_study()
    model = storage_object_model()
    schedule = schedule_contract()
    check = run_self_check()
    payload = {
        "schema": "h82_class_file_isa_v1",
        "status": "ISA_FROZEN_PENDING_H82_RANK1",
        "operator_parent": "H81_no_motion",
        "c8": ["C8.3_class_major_projection", "C8.1_class_stability_regularizer"],
        "software_class_id": {
            "lo": SCORE_LO,
            "hi": SCORE_HI,
            "step": SCORE_STEP,
            "n_bins": N_BINS,
        },
        "do_not_inherit": {
            "h67_max_score": H67_MAX_SCORE,
            "reason": "162 is the H67 popcount score range, not the H82 STE grid",
        },
        "schedule": schedule,
        "storage_model": model,
        "c81_proxy": study,
        "self_check": check,
        "innovation_claim": "not_4_0_until_rank1_stats_and_directory_rewrite",
    }
    (out / "isa.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(out / "isa.json"), "c81": study["fields"], "check": check}, indent=2))


if __name__ == "__main__":
    main()
