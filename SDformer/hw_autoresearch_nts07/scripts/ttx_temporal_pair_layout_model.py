"""Model transaction-level benefits of packing a TTX temporal pair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def model(*, timesteps: int, spatial_tokens: int, lanes: int, tensors: int, bus_bits: int) -> dict:
    if min(timesteps, spatial_tokens, lanes, tensors, bus_bits) <= 0:
        raise ValueError("all dimensions must be positive")
    lane_word_bits = lanes
    pair_word_bits = timesteps * lanes
    logical_bits = tensors * spatial_tokens * pair_word_bits

    serial_requests = tensors * spatial_tokens * timesteps
    packed_requests = tensors * spatial_tokens * ceil_div(pair_word_bits, bus_bits)
    serial_bus_bits_without_coalescing = serial_requests * bus_bits
    packed_bus_bits = packed_requests * bus_bits
    minimum_bus_words = tensors * spatial_tokens * ceil_div(pair_word_bits, bus_bits)

    return {
        "assumptions": {
            "timesteps": timesteps,
            "spatial_tokens": spatial_tokens,
            "lanes_per_head": lanes,
            "packed_tensors": tensors,
            "bus_bits": bus_bits,
            "one_request_issued_per_cycle": True,
        },
        "word_bits": {"serial_timestep": lane_word_bits, "packed_temporal_pair": pair_word_bits},
        "logical_storage_bits": logical_bits,
        "serial": {
            "requests": serial_requests,
            "request_cycles": serial_requests,
            "bus_bits_without_cross_time_coalescing": serial_bus_bits_without_coalescing,
        },
        "packed": {
            "requests": packed_requests,
            "request_cycles": packed_requests,
            "bus_bits": packed_bus_bits,
        },
        "ideal_already_coalesced_serial": {
            "requests": minimum_bus_words,
            "bus_bits": minimum_bus_words * bus_bits,
        },
        "deltas": {
            "logical_storage_reduction": 0.0,
            "request_reduction_vs_uncoalesced": 1.0 - packed_requests / serial_requests,
            "request_reduction_vs_already_coalesced": 1.0 - packed_requests / minimum_bus_words,
            "bus_traffic_reduction_vs_uncoalesced": 1.0 - packed_bus_bits / serial_bus_bits_without_coalescing,
            "bus_traffic_reduction_vs_already_coalesced": 0.0,
        },
        "claim_boundary": (
            "Packing is bit-exact and halves request/control cycles only when temporal 32-bit words "
            "were issued separately on a 64-bit interface. It does not reduce logical storage and "
            "does not reduce traffic if the baseline already coalesces both timesteps."
        ),
    }


def write_markdown(path: Path, result: dict) -> None:
    a = result["assumptions"]
    s = result["serial"]
    p = result["packed"]
    c = result["ideal_already_coalesced_serial"]
    d = result["deltas"]
    lines = [
        "# TTX Temporal-Pair Layout Model", "",
        f"Shape: T={a['timesteps']}, spatial tokens={a['spatial_tokens']}, lanes/head={a['lanes_per_head']}, "
        f"tensors={a['packed_tensors']} (Q and K), bus={a['bus_bits']} bit.", "",
        "| layout | requests/window/head | request cycles | transferred bits |",
        "|---|---:|---:|---:|",
        f"| separate timestep, no coalescing | {s['requests']} | {s['request_cycles']} | {s['bus_bits_without_cross_time_coalescing']} |",
        f"| temporal-pair packed | {p['requests']} | {p['request_cycles']} | {p['bus_bits']} |",
        f"| separate timestep, already coalesced | {c['requests']} | {c['requests']} | {c['bus_bits']} |", "",
        f"Request reduction versus uncoalesced baseline: {d['request_reduction_vs_uncoalesced']:.2%}.",
        f"Logical storage reduction: {d['logical_storage_reduction']:.2%}.", "",
        result["claim_boundary"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timesteps", type=int, default=2)
    parser.add_argument("--spatial-tokens", type=int, default=81)
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--tensors", type=int, default=2, help="Q and K by default")
    parser.add_argument("--bus-bits", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = model(
        timesteps=args.timesteps,
        spatial_tokens=args.spatial_tokens,
        lanes=args.lanes,
        tensors=args.tensors,
        bus_bits=args.bus_bits,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.output.with_suffix(".md"), result)
    print(args.output.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
