#!/usr/bin/env python3
"""Local5 DiSEP destination-write/source-read 的独立整数参考。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/local5_disep_reference_20260730"
DIRECTIONS = (
    ("self", 0, 0),
    ("north", -1, 0),
    ("south", 1, 0),
    ("east", 0, 1),
    ("west", 0, -1),
)


def token_id(time: int, y: int, x: int, height: int, width: int) -> int:
    return (time * height + y) * width + x


def topology_mask(times: int, height: int, width: int) -> np.ndarray:
    tokens = times * height * width
    mask = np.zeros((tokens, len(DIRECTIONS)), dtype=bool)
    for time in range(times):
        for y in range(height):
            for x in range(width):
                destination = token_id(time, y, x, height, width)
                for direction, (_, dy, dx) in enumerate(DIRECTIONS):
                    source_y = y + dy
                    source_x = x + dx
                    mask[destination, direction] = (
                        0 <= source_y < height and 0 <= source_x < width
                    )
    return mask


def source_for_edge(
    destination: int,
    direction: int,
    *,
    height: int,
    width: int,
) -> int | None:
    plane = height * width
    time, spatial = divmod(destination, plane)
    y, x = divmod(spatial, width)
    _, dy, dx = DIRECTIONS[direction]
    source_y = y + dy
    source_x = x + dx
    if not (0 <= source_y < height and 0 <= source_x < width):
        return None
    return token_id(time, source_y, source_x, height, width)


def destination_for_source(
    source: int,
    direction: int,
    *,
    height: int,
    width: int,
) -> int | None:
    plane = height * width
    time, spatial = divmod(source, plane)
    y, x = divmod(spatial, width)
    _, dy, dx = DIRECTIONS[direction]
    destination_y = y - dy
    destination_x = x - dx
    if not (0 <= destination_y < height and 0 <= destination_x < width):
        return None
    return token_id(time, destination_y, destination_x, height, width)


def destination_gather_projection(
    k_event: np.ndarray,
    gate_plane: np.ndarray,
    edge_valid: np.ndarray,
    weight: np.ndarray,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, dict[str, int]]:
    tokens, lanes = k_event.shape
    outputs = int(weight.shape[1])
    result = np.zeros((tokens, outputs), dtype=np.int64)
    product_terms = 0
    deliveries = 0
    for destination in range(tokens):
        for direction in range(len(DIRECTIONS)):
            if not edge_valid[destination, direction]:
                continue
            source = source_for_edge(
                destination,
                direction,
                height=height,
                width=width,
            )
            if source is None:
                raise AssertionError("edge_valid 包含越界关系")
            gate = int(gate_plane[destination, direction])
            if gate == 0:
                continue
            for lane in np.flatnonzero(k_event[source]):
                result[destination] += gate * weight[lane].astype(np.int64)
                product_terms += 1
                deliveries += 1
    return result, {
        "edge_lane_products": product_terms,
        "deliveries": deliveries,
    }


def disep_source_projection(
    k_event: np.ndarray,
    gate_plane: np.ndarray,
    edge_valid: np.ndarray,
    weight: np.ndarray,
    *,
    height: int,
    width: int,
) -> tuple[np.ndarray, dict[str, int]]:
    tokens, lanes = k_event.shape
    outputs = int(weight.shape[1])
    result = np.zeros((tokens, outputs), dtype=np.int64)
    source_gate_lane_terms = 0
    deliveries = 0
    max_gate_fanout = 0

    for source in range(tokens):
        destinations_by_gate: dict[int, list[int]] = {}
        for direction in range(len(DIRECTIONS)):
            destination = destination_for_source(
                source,
                direction,
                height=height,
                width=width,
            )
            if destination is None or not edge_valid[destination, direction]:
                continue
            reverse_source = source_for_edge(
                destination,
                direction,
                height=height,
                width=width,
            )
            if reverse_source != source:
                raise AssertionError("inverse-direction 地址不互逆")
            gate = int(gate_plane[destination, direction])
            if gate != 0:
                destinations_by_gate.setdefault(gate, []).append(destination)

        active_lanes = np.flatnonzero(k_event[source])
        for gate, destinations in destinations_by_gate.items():
            max_gate_fanout = max(max_gate_fanout, len(destinations))
            for lane in active_lanes:
                product = gate * weight[lane].astype(np.int64)
                source_gate_lane_terms += 1
                for destination in destinations:
                    result[destination] += product
                    deliveries += 1

    return result, {
        "source_gate_lane_terms": source_gate_lane_terms,
        "deliveries": deliveries,
        "max_gate_fanout": max_gate_fanout,
        "lane_count": lanes,
    }


def verify_case(
    k_event: np.ndarray,
    gate_plane: np.ndarray,
    edge_valid: np.ndarray,
    weight: np.ndarray,
    *,
    times: int,
    height: int,
    width: int,
) -> dict[str, int]:
    expected, gather = destination_gather_projection(
        k_event,
        gate_plane,
        edge_valid,
        weight,
        height=height,
        width=width,
    )
    actual, disep = disep_source_projection(
        k_event,
        gate_plane,
        edge_valid,
        weight,
        height=height,
        width=width,
    )
    mismatches = int(np.count_nonzero(expected != actual))
    if mismatches:
        raise AssertionError(f"DiSEP Acc mismatch: {mismatches}")
    if gather["deliveries"] != disep["deliveries"]:
        raise AssertionError("DiSEP destination delivery 不守恒")
    if expected.shape[0] != times * height * width:
        raise AssertionError("token geometry 不一致")
    return {
        "compared_accumulators": int(expected.size),
        "mismatches": mismatches,
        "gather_edge_lane_products": gather["edge_lane_products"],
        "disep_source_gate_lane_terms": disep["source_gate_lane_terms"],
        "deliveries": disep["deliveries"],
        "max_gate_fanout": disep["max_gate_fanout"],
    }


def random_case(
    rng: np.random.Generator,
    *,
    times: int,
    height: int,
    width: int,
    lanes: int,
    outputs: int,
) -> dict[str, int]:
    tokens = times * height * width
    k_event = rng.random((tokens, lanes)) < rng.uniform(0.0, 0.35)
    gates = rng.integers(0, 257, size=(tokens, 5), dtype=np.int16)
    valid = topology_mask(times, height, width)
    random_keep = rng.random(valid.shape) >= 0.15
    random_keep[:, 0] = True
    valid &= random_keep
    weights = rng.integers(-128, 128, size=(lanes, outputs), dtype=np.int16)
    return verify_case(
        k_event,
        gates,
        valid,
        weights,
        times=times,
        height=height,
        width=width,
    )


def run_reference(trials: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    cases = [
        random_case(
            rng,
            times=1,
            height=1,
            width=1,
            lanes=8,
            outputs=5,
        ),
        random_case(
            rng,
            times=2,
            height=9,
            width=9,
            lanes=8,
            outputs=7,
        ),
        random_case(
            rng,
            times=2,
            height=15,
            width=15,
            lanes=8,
            outputs=7,
        ),
    ]
    for _ in range(trials):
        cases.append(
            random_case(
                rng,
                times=2,
                height=int(rng.integers(2, 9)),
                width=int(rng.integers(2, 9)),
                lanes=8,
                outputs=7,
            )
        )

    totals = {
        key: sum(case[key] for case in cases)
        for key in (
            "compared_accumulators",
            "mismatches",
            "gather_edge_lane_products",
            "disep_source_gate_lane_terms",
            "deliveries",
        )
    }
    totals["max_gate_fanout"] = max(
        case["max_gate_fanout"] for case in cases
    )
    totals.update(
        {
            "schema": "local5_disep_integer_reference_v1",
            "seed": seed,
            "random_trials": trials,
            "directed_geometries": ["T1", "T162=2x9x9", "T450=2x15x15"],
            "cases": len(cases),
            "synthetic_product_reduction": (
                1.0
                - totals["disep_source_gate_lane_terms"]
                / totals["gather_edge_lane_products"]
                if totals["gather_edge_lane_products"]
                else 0.0
            ),
            "evidence": (
                "[integer-golden] projection-only；不是Shiftmax、ordered RTL或PPA"
            ),
        }
    )
    return totals


def render_markdown(result: dict) -> str:
    return "\n".join(
        [
            "# Local5 DiSEP 整数参考",
            "",
            f"- cases：{result['cases']}",
            f"- random trials：{result['random_trials']}",
            f"- geometries：{', '.join(result['directed_geometries'])}",
            f"- compared accumulators：{result['compared_accumulators']}",
            f"- mismatches：{result['mismatches']}",
            f"- destination deliveries：{result['deliveries']}",
            (
                "- synthetic product reduction："
                f"{100.0 * result['synthetic_product_reduction']:.2f}%"
            ),
            f"- max gate fanout：{result['max_gate_fanout']}",
            "",
            "该结果证明给定整数 gate plane 后，destination-major gather 与",
            "DiSEP inverse-direction source-major projection 的最终 Acc 一致。",
            "",
            "随机 product reduction 只验证计数关系，不代表 Local5 真实收益。",
            "本参考不包含 Shiftmax hardware-order、逐拍 ready/valid、SRAM、",
            "backpressure、饱和截断或 PPA。",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=lambda value: int(value, 0), default=0xD15E9)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_reference(args.trials, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    (args.out / "report.md").write_text(render_markdown(result) + "\n")
    print(args.out / "report.md")
    return int(result["mismatches"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
