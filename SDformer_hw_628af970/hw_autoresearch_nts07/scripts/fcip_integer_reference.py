#!/usr/bin/env python3
"""Motion FCIP class×lane 因子化平面的独立整数参考。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/fcip_integer_reference_20260730"
NUM_CLASSES = 163
GATE_CODES = 257


def baseline_projection(
    score_class: np.ndarray,
    k_event: np.ndarray,
    class_gate: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tokens, lanes = k_event.shape
    outputs = weight.shape[1]
    accumulator = np.zeros((tokens, outputs), dtype=np.int64)
    class_count = np.bincount(score_class, minlength=NUM_CLASSES)
    for destination in range(tokens):
        gate = int(class_gate[int(score_class[destination])])
        if gate == 0:
            continue
        for lane in np.flatnonzero(k_event[destination]):
            accumulator[destination] += gate * weight[lane].astype(np.int64)
    return accumulator, class_count


def fcip_projection(
    score_class: np.ndarray,
    k_event: np.ndarray,
    class_gate: np.ndarray,
    weight: np.ndarray,
    *,
    active_class_slots: int,
    segment_tokens: int = 64,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | bool]]:
    tokens, lanes = k_event.shape
    active_token = k_event.any(axis=1)
    active_classes = np.unique(score_class[active_token])
    if active_classes.size > active_class_slots:
        accumulator, class_count = baseline_projection(
            score_class,
            k_event,
            class_gate,
            weight,
        )
        return accumulator, class_count, {
            "fallback": True,
            "active_classes": int(active_classes.size),
            "class_lane_terms": 0,
            "class_lane_segments": 0,
            "deliveries": int(k_event.sum()),
        }

    class_count = np.bincount(score_class, minlength=NUM_CLASSES)
    class_bitmap = {
        int(class_id): score_class == class_id
        for class_id in active_classes
    }
    k_lane_bitmap = k_event.T.copy()
    outputs = weight.shape[1]
    accumulator = np.zeros((tokens, outputs), dtype=np.int64)
    class_lane_terms = 0
    class_lane_segments = 0
    deliveries = 0

    for class_id in active_classes:
        gate = int(class_gate[int(class_id)])
        if gate == 0:
            continue
        for lane in range(lanes):
            destination_bitmap = class_bitmap[int(class_id)] & k_lane_bitmap[lane]
            if not destination_bitmap.any():
                continue
            class_lane_terms += 1
            product = gate * weight[lane].astype(np.int64)
            for start in range(0, tokens, segment_tokens):
                stop = min(tokens, start + segment_tokens)
                destinations = np.flatnonzero(
                    destination_bitmap[start:stop]
                ) + start
                if destinations.size == 0:
                    continue
                class_lane_segments += 1
                accumulator[destinations] += product
                deliveries += int(destinations.size)

    return accumulator, class_count, {
        "fallback": False,
        "active_classes": int(active_classes.size),
        "class_lane_terms": class_lane_terms,
        "class_lane_segments": class_lane_segments,
        "deliveries": deliveries,
    }


def verify_case(
    rng: np.random.Generator,
    *,
    tokens: int,
    lanes: int,
    outputs: int,
    force_fallback: bool,
) -> dict[str, int | bool]:
    if force_fallback:
        active_class_ids = np.arange(17, dtype=np.int16)
        score_class = np.resize(active_class_ids, tokens)
        k_event = np.zeros((tokens, lanes), dtype=bool)
        k_event[:, 0] = True
    else:
        active_class_count = int(rng.integers(1, 17))
        active_class_ids = rng.choice(
            NUM_CLASSES,
            size=active_class_count,
            replace=False,
        )
        score_class = rng.choice(active_class_ids, size=tokens).astype(np.int16)
        k_event = rng.random((tokens, lanes)) < rng.uniform(0.0, 0.35)

    # 注入仅参与denominator的K-zero score class。
    zero_token_count = min(tokens, max(1, tokens // 11))
    k_event[:zero_token_count] = False
    score_class[:zero_token_count] = int(rng.integers(0, NUM_CLASSES))

    class_gate = rng.integers(0, GATE_CODES, size=NUM_CLASSES, dtype=np.int16)
    # 强制多个score class映射到同一final gate，覆盖late alias。
    class_gate[active_class_ids[: min(4, active_class_ids.size)]] = 64
    weight = rng.integers(-128, 128, size=(lanes, outputs), dtype=np.int16)

    expected, expected_count = baseline_projection(
        score_class,
        k_event,
        class_gate,
        weight,
    )
    actual, actual_count, stats = fcip_projection(
        score_class,
        k_event,
        class_gate,
        weight,
        active_class_slots=16,
    )
    mismatches = int(np.count_nonzero(expected != actual))
    count_mismatches = int(np.count_nonzero(expected_count != actual_count))
    if mismatches or count_mismatches:
        raise AssertionError(
            f"FCIP mismatch: acc={mismatches}, count={count_mismatches}"
        )
    expected_delivery = int(
        (
            k_event
            & class_gate[score_class].reshape(tokens, 1).astype(bool)
        ).sum()
    )
    if not stats["fallback"] and stats["deliveries"] != expected_delivery:
        raise AssertionError("FCIP fast path delivery不守恒")
    return {
        **stats,
        "accumulator_values": int(expected.size),
        "mismatches": mismatches,
        "count_mismatches": count_mismatches,
    }


def run_reference(trials: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    cases = [
        verify_case(
            rng,
            tokens=162,
            lanes=32,
            outputs=8,
            force_fallback=False,
        ),
        verify_case(
            rng,
            tokens=450,
            lanes=32,
            outputs=8,
            force_fallback=False,
        ),
        verify_case(
            rng,
            tokens=162,
            lanes=32,
            outputs=8,
            force_fallback=True,
        ),
        verify_case(
            rng,
            tokens=450,
            lanes=32,
            outputs=8,
            force_fallback=True,
        ),
    ]
    for _ in range(trials):
        cases.append(
            verify_case(
                rng,
                tokens=162 if rng.random() < 0.5 else 450,
                lanes=32,
                outputs=8,
                force_fallback=bool(rng.random() < 0.1),
            )
        )
    result = {
        "schema": "fcip_integer_reference_v1",
        "seed": seed,
        "random_trials": trials,
        "cases": len(cases),
        "fallback_cases": sum(bool(row["fallback"]) for row in cases),
        "accumulator_values": sum(
            int(row["accumulator_values"]) for row in cases
        ),
        "mismatches": sum(int(row["mismatches"]) for row in cases),
        "class_count_mismatches": sum(
            int(row["count_mismatches"]) for row in cases
        ),
        "fast_path_deliveries": sum(
            int(row["deliveries"]) for row in cases if not row["fallback"]
        ),
        "fast_path_class_lane_terms": sum(
            int(row["class_lane_terms"])
            for row in cases
            if not row["fallback"]
        ),
        "fast_path_class_lane_segments": sum(
            int(row["class_lane_segments"])
            for row in cases
            if not row["fallback"]
        ),
        "evidence": (
            "[integer-golden] FCIP representation/fallback；"
            "不是Shiftmax、ordered RTL、cycle或PPA"
        ),
    }
    return result


def render_markdown(result: dict) -> str:
    return "\n".join(
        [
            "# Motion FCIP 整数参考",
            "",
            f"- cases：{result['cases']}",
            f"- fallback cases：{result['fallback_cases']}",
            f"- accumulator values：{result['accumulator_values']}",
            f"- Acc mismatch：{result['mismatches']}",
            f"- class-count mismatch：{result['class_count_mismatches']}",
            f"- fast-path deliveries：{result['fast_path_deliveries']}",
            f"- fast-path class-lane terms：{result['fast_path_class_lane_terms']}",
            (
                "- fast-path class-lane segments："
                f"{result['fast_path_class_lane_segments']}"
            ),
            "",
            "参考覆盖T162/T450、32 lane、K-zero class count、gate alias、",
            "S16 fast path和超过16个active class的whole-row replay。",
            "",
            "该结果只证明给定class与gate后，因子化交集和整行fallback保持",
            "整数Acc与denominator class count；不包含Shiftmax、逐拍协议或PPA。",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0xFC1F)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_reference(args.trials, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.out / "report.md").write_text(
        render_markdown(result) + "\n",
        encoding="utf-8",
    )
    print(args.out / "report.md")
    return int(
        result["mismatches"] != 0
        or result["class_count_mismatches"] != 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
