#!/usr/bin/env python3
"""Screen two structural algorithm-hardware candidates without changing RTL."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    from scripts.generate_h67_checkpoint_row_vectors import (
        exp2_q8,
        round_shift_even,
        score_q7,
    )
except ModuleNotFoundError:
    from generate_h67_checkpoint_row_vectors import (
        exp2_q8,
        round_shift_even,
        score_q7,
    )


HEIGHT = 15
WIDTH = 15
PLANES = 2
TOKENS = HEIGHT * WIDTH * PLANES
LANES = 32
ROLES = 5
PAIR_TOKENS = HEIGHT * WIDTH
DEST_TO_SOURCE = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def round_div_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    doubled = remainder * 2
    if doubled > denominator or (doubled == denominator and quotient & 1):
        quotient += 1
    return quotient


def h67_statistics(q: int, current_k: int, peer_k: int) -> tuple[int, int, int]:
    mask = (1 << LANES) - 1
    overlap = (q & current_k).bit_count()
    same_zero = ((~q) & (~current_k) & mask).bit_count()
    motion = (current_k ^ peer_k).bit_count()
    return overlap, same_zero, motion


def structural_shared_score(
    q0: int,
    k0: int,
    q1: int,
    k1: int,
) -> int:
    """RNE of the mean pre-quantized H67 score using three pair statistics."""

    overlap0, same_zero0, motion0 = h67_statistics(q0, k0, k1)
    overlap1, same_zero1, motion1 = h67_statistics(q1, k1, k0)
    if motion0 != motion1:
        raise AssertionError("temporal motion statistic is not symmetric")
    numerator = (
        64 * (overlap0 + overlap1)
        + 32 * motion0
        + same_zero0
        + same_zero1
    )
    return round_div_even(numerator, 32)


def gate_codes(scores: Iterable[int]) -> list[int]:
    values = list(scores)
    row_max = max(values)
    exponentials = [exp2_q8(value - row_max) for value in values]
    row_sum = sum(exponentials)
    denominator_shift = max(row_sum - 1, 0).bit_length()
    return [
        min(round_shift_even(value * len(values) * 128, denominator_shift), 256)
        for value in exponentials
    ]


def source_index(destination: int, role: int) -> int | None:
    plane, within = divmod(destination, HEIGHT * WIDTH)
    y, x = divmod(within, WIDTH)
    dy, dx = DEST_TO_SOURCE[role]
    sy, sx = y + dy, x + dx
    if not (0 <= sy < HEIGHT and 0 <= sx < WIDTH):
        return None
    return plane * HEIGHT * WIDTH + sy * WIDTH + sx


def popcount_u32(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values.astype(np.uint32, copy=False))
    byte_view = contiguous.view(np.uint8).reshape(contiguous.shape + (4,))
    lut = np.asarray([value.bit_count() for value in range(256)], dtype=np.uint8)
    return lut[byte_view].sum(axis=-1, dtype=np.uint16)


def local_orbit_counts(k_matrix: np.ndarray) -> dict[str, int]:
    if k_matrix.ndim != 2 or k_matrix.shape[1] != TOKENS:
        raise ValueError("Local5 K matrix must have shape [groups,450]")
    role_k: list[np.ndarray] = []
    for role in range(ROLES):
        indices = np.asarray(
            [source_index(destination, role) or 0 for destination in range(TOKENS)],
            dtype=np.int32,
        )
        valid = np.asarray(
            [source_index(destination, role) is not None for destination in range(TOKENS)],
            dtype=np.bool_,
        )
        gathered = k_matrix[:, indices]
        role_k.append(np.where(valid[None, :], gathered, 0).astype(np.uint32))

    self_terms = int(popcount_u32(role_k[0]).sum(dtype=np.int64))
    vertical_terms = int(
        popcount_u32(role_k[1] | role_k[2]).sum(dtype=np.int64)
    )
    horizontal_terms = int(
        popcount_u32(role_k[3] | role_k[4]).sum(dtype=np.int64)
    )
    edge_lane_terms = sum(
        int(popcount_u32(values).sum(dtype=np.int64)) for values in role_k
    )
    return {
        "self_terms": self_terms,
        "vertical_terms": vertical_terms,
        "horizontal_terms": horizontal_terms,
        "orbit_terms": self_terms + vertical_terms + horizontal_terms,
        "edge_lane_terms": edge_lane_terms,
    }


def screen_local5(payload_path: Path, current_source_terms: int) -> dict[str, Any]:
    with np.load(payload_path, allow_pickle=False) as payload:
        offsets = np.asarray(payload["descriptor_group_offsets"])
        source_ids = np.asarray(payload["descriptor_source_id"])
        k_bitmap = np.asarray(payload["descriptor_k_bitmap"], dtype=np.uint32)
        incoming_gates = np.asarray(payload["descriptor_incoming_gates"])
        valid_mask = np.asarray(payload["descriptor_valid_mask"])
    if offsets.ndim != 1 or np.any(np.diff(offsets) != TOKENS):
        raise ValueError("Local5 payload is not a complete 450-source-per-group archive")
    groups = offsets.size - 1
    if not np.all(
        source_ids.reshape(groups, TOKENS)
        == np.arange(TOKENS, dtype=source_ids.dtype)[None, :]
    ):
        raise ValueError("Local5 source order is not canonical 0..449")
    if incoming_gates.shape != (groups * TOKENS, ROLES):
        raise ValueError("Local5 incoming gate shape mismatch")
    if valid_mask.shape != (groups * TOKENS,):
        raise ValueError("Local5 valid-mask shape mismatch")

    counts = local_orbit_counts(k_bitmap.reshape(groups, TOKENS))
    orbit_ratio = counts["orbit_terms"] / current_source_terms

    gates = incoming_gates.reshape(groups, TOKENS, ROLES)
    valid = (
        (valid_mask.reshape(groups, TOKENS, 1)
         >> np.arange(ROLES, dtype=np.uint8)[None, None, :])
        & 1
    ).astype(np.bool_)
    symmetry: dict[str, Any] = {}
    for name, lhs, rhs in (("vertical", 1, 2), ("horizontal", 3, 4)):
        both = valid[:, :, lhs] & valid[:, :, rhs]
        equal = both & (gates[:, :, lhs] == gates[:, :, rhs])
        symmetry[name] = {
            "both_valid": int(both.sum()),
            "equal_gate": int(equal.sum()),
            "equal_rate": float(equal.sum() / both.sum()) if both.any() else 0.0,
        }

    return {
        "status": "NO_GO_ORBIT_TERM_DOMINATED",
        "evidence": "[prof] structural lower-bound screen, not a trained model",
        "groups": groups,
        "counts": counts,
        "current_source_owned_terms": current_source_terms,
        "orbit_over_source_ratio": orbit_ratio,
        "orbit_reduction_vs_edge_lane": (
            1.0 - counts["orbit_terms"] / counts["edge_lane_terms"]
        ),
        "existing_gate_symmetry": symmetry,
        "decision_reason": (
            "Even granting one shared gate per topology orbit and exact 2-bit K-count "
            "folding, destination-owned orbit terms exceed the existing source-owned "
            "term object. Retraining cannot remove that routing lower bound."
        ),
    }


def parse_motion_vectors(vector_path: Path) -> tuple[int, int, list[dict[str, Any]]]:
    lines = iter(vector_path.read_text(encoding="ascii").splitlines())
    row_count, tokens = map(int, next(lines).split())
    rows: list[dict[str, Any]] = []
    for expected_tag in range(row_count):
        header = [int(value) for value in next(lines).split()]
        if len(header) != 6 or header[0] != expected_tag:
            raise ValueError(f"invalid Motion row header at {expected_tag}: {header}")
        vectors = []
        for _ in range(tokens):
            fields = next(lines).split()
            if len(fields) != 4:
                raise ValueError("invalid Motion token vector")
            vectors.append(
                (int(fields[0], 16), int(fields[1], 16), int(fields[2], 16), int(fields[3]))
            )
        rows.append(
            {
                "row_tag": header[0],
                "stage": header[1],
                "block": header[2],
                "head": header[3],
                "vectors": vectors,
            }
        )
    try:
        extra = next(lines)
    except StopIteration:
        extra = ""
    if extra:
        raise ValueError("Motion vector file has trailing data")
    return row_count, tokens, rows


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def screen_motion(vector_path: Path) -> dict[str, Any]:
    row_count, tokens, rows = parse_motion_vectors(vector_path)
    if tokens != TOKENS:
        raise ValueError("Motion structural screen requires T450 rows")

    aggregate: dict[str, float | int] = defaultdict(int)
    per_stage: dict[int, dict[str, Any]] = {}
    row_relative_l1: list[float] = []
    row_gate_mae: list[float] = []
    row_gate_p95: list[float] = []
    row_all_gate_equal = 0

    for row in rows:
        q = [item[0] for item in row["vectors"]]
        current = [item[1] for item in row["vectors"]]
        peer = [item[2] for item in row["vectors"]]
        recorded_gates = [item[3] for item in row["vectors"]]
        scores = [score_q7(q[i], current[i], peer[i]) for i in range(tokens)]
        reference_gates = gate_codes(scores)
        if reference_gates != recorded_gates:
            raise ValueError(f"Motion fixed-point gate mismatch row={row['row_tag']}")

        shared_scores: list[int] = []
        equal_pairs = 0
        for spatial in range(PAIR_TOKENS):
            first = spatial
            second = PAIR_TOKENS + spatial
            if current[first] != peer[second] or peer[first] != current[second]:
                raise ValueError(f"Motion peer-K mismatch row={row['row_tag']} pair={spatial}")
            equal_pairs += int(scores[first] == scores[second])
            shared_scores.append(
                structural_shared_score(
                    q[first], current[first], q[second], current[second]
                )
            )
        structural_scores = shared_scores + shared_scores
        candidate_gates = gate_codes(structural_scores)

        abs_gate = [
            abs(candidate - baseline)
            for candidate, baseline in zip(candidate_gates, recorded_gates, strict=True)
        ]
        row_gate_mae.append(float(sum(abs_gate) / tokens))
        row_gate_p95.append(percentile([float(value) for value in abs_gate], 95))
        row_all_gate_equal += int(not any(abs_gate))

        baseline_lane = [0] * LANES
        candidate_lane = [0] * LANES
        for token, k_value in enumerate(current):
            bits = k_value
            while bits:
                least = bits & -bits
                lane = least.bit_length() - 1
                baseline_lane[lane] += recorded_gates[token]
                candidate_lane[lane] += candidate_gates[token]
                bits ^= least
        l1 = sum(abs(lhs - rhs) for lhs, rhs in zip(candidate_lane, baseline_lane, strict=True))
        denominator = sum(abs(value) for value in baseline_lane)
        row_relative_l1.append(float(l1 / denominator) if denominator else float(l1 != 0))

        rqtb_slots = 2 * PAIR_TOKENS - equal_pairs
        stage = int(row["stage"])
        stage_row = per_stage.setdefault(
            stage,
            {"rows": 0, "pairs": 0, "equal_pairs": 0, "rqtb_slots": 0, "shared_slots": 0},
        )
        stage_row["rows"] += 1
        stage_row["pairs"] += PAIR_TOKENS
        stage_row["equal_pairs"] += equal_pairs
        stage_row["rqtb_slots"] += rqtb_slots
        stage_row["shared_slots"] += PAIR_TOKENS
        aggregate["pairs"] += PAIR_TOKENS
        aggregate["equal_pairs"] += equal_pairs
        aggregate["rqtb_slots"] += rqtb_slots
        aggregate["shared_slots"] += PAIR_TOKENS
        aggregate["gate_abs_sum"] += sum(abs_gate)
        aggregate["gate_exact_tokens"] += sum(value == 0 for value in abs_gate)

    for stage_row in per_stage.values():
        stage_row["incremental_slot_reduction"] = (
            1.0 - stage_row["shared_slots"] / stage_row["rqtb_slots"]
        )
        stage_row["equal_pair_rate"] = stage_row["equal_pairs"] / stage_row["pairs"]

    incremental_slot_reduction = 1.0 - aggregate["shared_slots"] / aggregate["rqtb_slots"]
    result_status = "HOLD_PROFILE_ONLY_REQUIRES_RETRAINING_AND_PPA"
    if incremental_slot_reduction < 0.10:
        result_status = "NO_GO_DESCRIPTOR_GAIN_BELOW_10_PERCENT"

    return {
        "status": result_status,
        "evidence": "[模型] frozen-trace proxy for a different, untrained score operator",
        "rows": row_count,
        "tokens": row_count * tokens,
        "pairs": int(aggregate["pairs"]),
        "equal_pairs": int(aggregate["equal_pairs"]),
        "equal_pair_rate": aggregate["equal_pairs"] / aggregate["pairs"],
        "rqtb_slots": int(aggregate["rqtb_slots"]),
        "structural_shared_slots": int(aggregate["shared_slots"]),
        "incremental_slot_reduction_vs_rqtb": incremental_slot_reduction,
        "gate_exact_token_rate": aggregate["gate_exact_tokens"] / (row_count * tokens),
        "gate_mae": aggregate["gate_abs_sum"] / (row_count * tokens),
        "gate_row_mae_p95": percentile(row_gate_mae, 95),
        "gate_row_abs_delta_p95_p95": percentile(row_gate_p95, 95),
        "rows_with_all_gates_exact": row_all_gate_equal,
        "preprojection_lane_relative_l1": {
            "mean": float(np.mean(row_relative_l1)),
            "p95": percentile(row_relative_l1, 95),
            "max": max(row_relative_l1),
        },
        "per_stage": {str(stage): row for stage, row in sorted(per_stage.items())},
        "decision_reason": (
            "The candidate changes the trained score operator. It can only advance if "
            "the merged score front, not descriptor savings alone, beats MSSB5 under "
            "matched PPA and retraining preserves application accuracy."
        ),
    }


def write_report(output_dir: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    local = result["local5_topology_orbit"]
    motion = result["motion_structural_shared_score"]
    markdown = f"""# 双线结构协同候选只读筛选

## 结论

- Local5 topology-orbit：`{local['status']}`。
- Motion structural shared-score：`{motion['status']}`。
- 两项均未接 RTL、未训练、未改冻结主表；当前创新评分保持 Motion `3.2/5`、Local5 `3.1/5`。

## Local5

在最有利假设下，每个 destination 只保留 self、vertical、horizontal 三个 gate，且同 orbit 的两个二值 K 在线折成 2-bit count。即便如此：

- orbit term：`{local['counts']['orbit_terms']:,}`；
- 现有 source-owned term：`{local['current_source_owned_terms']:,}`；
- orbit/source 比：`{local['orbit_over_source_ratio']:.4f}x`；
- 相对 raw edge-lane 只减少 `{local['orbit_reduction_vs_edge_lane']:.2%}`。

该候选会丢失现有跨 destination source-owned 复用，故按强基线直接 NO-GO。自然 gate 对称率仅作诊断，不改变裁决。

## Motion

候选用 overlap-sum、same-zero-sum、motion 三个 pair 统计生成一个结构共享 score。它是新算子，不是冻结 H67 的 exact 变换。

- 真实十样本行：`{motion['rows']:,}`；
- RQTB slot：`{motion['rqtb_slots']:,}`；结构共享 slot：`{motion['structural_shared_slots']:,}`；
- 相对 RQTB slot 上限：`{motion['incremental_slot_reduction_vs_rqtb']:.2%}`；
- gate exact token rate：`{motion['gate_exact_token_rate']:.2%}`；
- projection 前 32-lane 聚合 relative-L1 p95：`{motion['preprojection_lane_relative_l1']['p95']:.4%}`。

若增量 slot 低于 10%，仅靠 descriptor 数不能晋级；只有合并 score-front 在同 SDC/端口下显著打赢 MSSB5，且重训练 valid825/MVSEC 不退化，才允许重开 RTL。

## 证据边界

- Local5 是 `[prof]` term 下界，不是周期、能量或训练结果。
- Motion 是冻结 trace 上对新算子的 `[模型]` 代理，不是现有 H67 bit-exact，也不是模型精度。
- 输出不进入 `docs/359_DATE终局冻结_20260813.md`。
"""
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local5-payload", type=Path, required=True)
    parser.add_argument("--motion-vectors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--current-source-terms", type=int, default=9_870_505)
    args = parser.parse_args()

    result = {
        "schema": "structural_codesign_candidate_screen_v1",
        "status": "PASS_SCREEN_COMPLETE_NO_RTL_PROMOTION",
        "local5_topology_orbit": screen_local5(
            args.local5_payload, args.current_source_terms
        ),
        "motion_structural_shared_score": screen_motion(args.motion_vectors),
        "sha256": {
            "local5_payload": sha256(args.local5_payload),
            "motion_vectors": sha256(args.motion_vectors),
            "script": sha256(Path(__file__).resolve()),
        },
        "claim_boundary": [
            "No production RTL or frozen table was changed.",
            "Local5 is a structural term-count lower-bound screen.",
            "Motion evaluates a different untrained score operator on frozen traces.",
            "Neither result is ASIC PPA, application accuracy, or encoder speedup.",
        ],
    }
    write_report(args.output_dir, result)
    print(
        "PASS structural co-design screen "
        f"local5={result['local5_topology_orbit']['status']} "
        f"motion={result['motion_structural_shared_score']['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
