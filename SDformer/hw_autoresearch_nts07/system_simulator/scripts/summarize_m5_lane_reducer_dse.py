#!/usr/bin/env python3
"""Summarize temporal-fenced M4 lane/reducer physical-interface candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def candidate(payload: dict[str, Any]) -> dict[str, Any]:
    arch = payload["architecture"]
    if arch.get("availability_mode") != "temporal_fenced":
        raise ValueError("M5 physical-port DSE requires temporal-fenced schedules")
    if any(
        identity.get("cross_temporal_batches") != 0
        or identity.get("cross_operator_call_batches") != 0
        or identity.get("cross_sequence_batches") != 0
        for line in payload["variants"].values()
        for identity in line["per_identity"].values()
    ):
        raise ValueError("M5 candidate crossed a dynamic availability fence")
    return {
        "output_lanes": arch["output_lanes"],
        "reduce_slots_per_context": arch["reduce_slots_per_context"],
        "weight_response_width_bits": arch["weight_response_width_bits"],
        "accumulator_output_width_bits": arch["accumulator_output_width_bits"],
        "accumulator_state_bits": arch["accumulator_state_bits"],
        "signed_adder_proxy": arch["shared_reducer_signed_adders"],
        "local": {
            key: payload["variants"]["local"][key]
            for key in (
                "m4_wall_cycles",
                "speedup_vs_p1_sparse_wall",
                "speedup_vs_same_width_dense_wall",
                "same_width_dense_sample_speedup_min",
            )
        },
        "hybrid": {
            key: payload["variants"]["hybrid"][key]
            for key in (
                "m4_wall_cycles",
                "speedup_vs_p1_sparse_wall",
                "speedup_vs_same_width_dense_wall",
                "same_width_dense_sample_speedup_min",
            )
        },
    }


def summarize(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [candidate(payload) for payload in payloads]
    identities = [payload["identities"] for payload in payloads]
    if any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("M5 candidates do not share trace identity")
    keys = [(row["output_lanes"], row["reduce_slots_per_context"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate M5 lane/reducer candidate")
    reference = next(
        (row for row in rows if row["output_lanes"] == 96 and row["reduce_slots_per_context"] == 4),
        None,
    )
    if reference is None:
        raise ValueError("M5 requires the L96/R4 throughput reference")
    for row in rows:
        for line in ("local", "hybrid"):
            relative_throughput = (
                reference[line]["m4_wall_cycles"] / row[line]["m4_wall_cycles"]
            )
            row[line]["throughput_vs_l96_r4"] = relative_throughput
            row[line]["throughput_per_adder_vs_l96_r4"] = relative_throughput / (
                row["signed_adder_proxy"] / reference["signed_adder_proxy"]
            )
            row[line]["throughput_per_weight_bit_vs_l96_r4"] = relative_throughput / (
                row["weight_response_width_bits"]
                / reference["weight_response_width_bits"]
            )
    rows.sort(key=lambda row: (row["output_lanes"], row["reduce_slots_per_context"]))
    return {
        "schema": "m5_temporal_fenced_lane_reducer_dse_v1",
        "status": "PASS_M5_PHYSICAL_PORT_DSE_PREMAPPED",
        "reference": "L96_R4",
        "claim_boundary": (
            "cycle and structural-width sensitivity only; signed adders and interface bits "
            "are RTL proxies, not mapped area/power. Candidate selection requires matched "
            "DC/PrimeTime/PTPX and SRAM port evidence."
        ),
        "candidates": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    result = summarize(payloads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = [
        "# M5 temporal-fenced lane/reducer physical-port DSE\n\n",
        "| lanes | R | weight rsp | Acc out | adders | Local dense speedup | Hybrid dense speedup | Local throughput/ref | Hybrid throughput/ref | Local throughput/adder |\n",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in result["candidates"]:
        report.append(
            f"| {row['output_lanes']} | {row['reduce_slots_per_context']} | "
            f"{row['weight_response_width_bits']} b | {row['accumulator_output_width_bits']} b | "
            f"{row['signed_adder_proxy']} | "
            f"{row['local']['speedup_vs_same_width_dense_wall']:.6f}x | "
            f"{row['hybrid']['speedup_vs_same_width_dense_wall']:.6f}x | "
            f"{row['local']['throughput_vs_l96_r4']:.6f}x | "
            f"{row['hybrid']['throughput_vs_l96_r4']:.6f}x | "
            f"{row['local']['throughput_per_adder_vs_l96_r4']:.6f}x |\n"
        )
    report.append("\nNo physical winner is selected before matched Synopsys PPA.\n")
    args.output.with_suffix(".md").write_text("".join(report), encoding="utf-8")
    print(f"PASS: wrote {args.output} with {len(result['candidates'])} candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
