#!/usr/bin/env python3
"""Correct M26's T2 packing bound and DSE the M30 streaming I/O contract."""

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_M26 = (
    ROOT
    / "hw_autoresearch_nts07/results/"
    "m26_atlif_factor_arithmetic_lower_bound_r5_receipted_20260822/"
    "m26_atlif_factor_arithmetic_lower_bound.json"
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ceil_div(numerator, denominator):
    return (int(numerator) + int(denominator) - 1) // int(denominator)


def select_rank3(m26):
    candidates = [row for row in m26["candidates"] if int(row["rank"]) == 3]
    if len(candidates) != 1:
        raise ValueError("M30 requires exactly one M26 rank-3 candidate")
    return candidates[0]


def port_candidate(
    name, input_bits, rank3, fixed_cycles, non_atlif_local,
    non_atlif_motion, parameter_state_bytes, shared_bitpack_output=False,
):
    t2_vectors = int(rank3["dense_fallback_macs"]) // 4
    if int(rank3["dense_fallback_macs"]) != t2_vectors * 4:
        raise ValueError("T2 dense fallback product count is not divisible by four")
    if shared_bitpack_output:
        # A T2 lane consumes two INT8 inputs and emits two packed bits.
        t2_lanes_per_cycle = min(24, int(input_bits) // 18)
    else:
        t2_lanes_per_cycle = min(24, int(input_bits) // 16)
    if t2_lanes_per_cycle <= 0:
        raise ValueError("M30 port cannot carry one T2 lane")
    t2_cycles = ceil_div(t2_vectors, t2_lanes_per_cycle)
    t10_tile_input_bits = 10 * 16 * 8
    t10_output_bits = 10 * 16 if shared_bitpack_output else 0
    t10_load_cycles = ceil_div(
        t10_tile_input_bits + t10_output_bits, int(input_bits)
    )
    t10_service_cycles = max(10, t10_load_cycles)
    factor_tiles = int(rank3["factor_tiles"])
    t10_cycles = factor_tiles * t10_service_cycles
    parameter_cold_fill_cycles = ceil_div(parameter_state_bytes * 8, input_bits)
    local_cycles = (
        int(non_atlif_local) + t10_cycles + t2_cycles
        + parameter_cold_fill_cycles
    )
    motion_cycles = (
        int(non_atlif_motion) + t10_cycles + t2_cycles
        + parameter_cold_fill_cycles
    )
    local_budget_to_2x = float(fixed_cycles) / 2.0 - local_cycles
    motion_budget_to_2x = float(fixed_cycles) / 2.0 - motion_cycles
    return {
        "name": name,
        "input_payload_bits_per_cycle": int(input_bits),
        "bitpack_output_bus": (
            "shared_with_input" if shared_bitpack_output else "independent"
        ),
        "t10_input_plus_output_load_cycles": t10_load_cycles,
        "t10_compute_cycles_per_tile": 10,
        "t10_service_cycles_per_tile": t10_service_cycles,
        "t10_factor_tiles": factor_tiles,
        "t10_cycles": t10_cycles,
        "t2_vectors": t2_vectors,
        "t2_lanes_per_cycle": t2_lanes_per_cycle,
        "t2_product_slots_used": t2_lanes_per_cycle * 4,
        "t2_product_slot_utilization": t2_lanes_per_cycle * 4 / 96.0,
        "t2_cycles": t2_cycles,
        "parameter_state_bytes": int(parameter_state_bytes),
        "parameter_cold_fill_cycles": parameter_cold_fill_cycles,
        "local_cycles": local_cycles,
        "motion_cycles": motion_cycles,
        "local_speedup_vs_fixed": float(fixed_cycles) / local_cycles,
        "motion_speedup_vs_fixed": float(fixed_cycles) / motion_cycles,
        "local_cycles_budget_before_falling_below_2x": local_budget_to_2x,
        "motion_cycles_budget_before_falling_below_2x": motion_budget_to_2x,
        "local_extra_cycles_budget_per_t10_factor_tile": (
            local_budget_to_2x / factor_tiles
        ),
        "crosses_2x_local": local_cycles * 2 < int(fixed_cycles),
        "crosses_2x_motion": motion_cycles * 2 < int(fixed_cycles),
    }


def build_report(m26_path):
    m26 = json.loads(Path(m26_path).read_text(encoding="utf-8"))
    if (
        m26.get("schema") != "m26_atlif_factor_arithmetic_lower_bound_v2"
        or "NO_SPEEDUP_CLAIM" not in str(m26.get("status"))
    ):
        raise ValueError("unexpected M26 evidence contract")
    rank3 = select_rank3(m26)
    fixed_cycles = int(m26["cycle_contract"]["fixed_compute_cycles"])
    non_atlif_local = int(m26["cycle_contract"]["shared_non_atlif_local_cycles"])
    non_atlif_motion = int(m26["cycle_contract"]["shared_non_atlif_hybrid_cycles"])
    factor_state = rank3["factor_state_contract"]
    parameter_state_bits = (
        8 * int(factor_state["live_temporal_weight_entries"])
        + 24 * int(factor_state["live_bias_entries_not_in_temporal_weight_count"])
        + 24 * int(factor_state["live_threshold_entries_not_in_temporal_weight_count"])
        # Each profitable T10 factor context needs the explicit five-bit
        # Q24-to-INT8 requant shift implemented by the M30A RTL.
        + 5 * int(rank3["factorized_modules"])
    )
    parameter_state_bytes = ceil_div(parameter_state_bits, 8)
    if parameter_state_bits != 37449 or parameter_state_bytes != 4682:
        raise ValueError("M30 live parameter state drift")

    candidates = [
        port_candidate(
            "128b_independent_output", 128, rank3, fixed_cycles,
            non_atlif_local, non_atlif_motion, parameter_state_bytes,
        ),
        port_candidate(
            "256b_independent_output_lanes16", 256, rank3, fixed_cycles,
            non_atlif_local, non_atlif_motion, parameter_state_bytes,
        ),
        port_candidate(
            "256b_shared_with_bitpack_output", 256, rank3, fixed_cycles,
            non_atlif_local, non_atlif_motion, parameter_state_bytes, True,
        ),
        port_candidate(
            "384b_independent_output_packed24", 384, rank3, fixed_cycles,
            non_atlif_local, non_atlif_motion, parameter_state_bytes,
        ),
        port_candidate(
            "dual256b_independent_output_packed24", 512, rank3, fixed_cycles,
            non_atlif_local, non_atlif_motion, parameter_state_bytes,
        ),
    ]
    output_values = 1461240000
    q24_output_bytes = output_values * 3
    bitpack_output_bytes = ceil_div(output_values, 8)
    input_bytes = output_values
    return {
        "schema": "m30_resident_stream_system_dse_v2",
        "status": "PASS_CORRECTED_EXECUTABLE_CYCLE_DSE_NO_ACCURACY_PPA_ENERGY_OR_SYSTEM_CLAIM",
        "identity": {
            "m26": str(Path(m26_path).resolve()),
            "m26_sha256": sha256(m26_path),
            "analyzer_sha256": sha256(Path(__file__).resolve()),
        },
        "frozen_resources": {
            "signed_int8_multipliers": 96,
            "t10_rank": 3,
            "t10_vector_lanes": 16,
            "t10_product_issue_cycles_per_tile": 10,
            "live_parameter_state_bytes": parameter_state_bytes,
            "live_parameter_state_bits": parameter_state_bits,
            "t10_requant_shift_bits": 5 * int(rank3["factorized_modules"]),
            "fixed_compute_cycles": fixed_cycles,
            "shared_non_atlif_local_cycles": non_atlif_local,
            "shared_non_atlif_motion_cycles": non_atlif_motion,
        },
        "m26_t2_correction": {
            "reason": (
                "M26 ceil(total T2 products/96) implicitly packs 24 lanes across "
                "tiles; a fixed lanes16 path uses only 64 of 96 product slots"
            ),
            "t2_dense_products": int(rank3["dense_fallback_macs"]),
            "t2_vectors": int(rank3["dense_fallback_macs"]) // 4,
            "t2_tiles_at_lanes16": ceil_div(
                int(rank3["dense_fallback_macs"]) // 4, 16
            ),
            "m26_ideal_packed24_cycles": ceil_div(
                int(rank3["dense_fallback_macs"]) // 4, 24
            ),
            "minimum_sustained_input_bits_for_packed24": 24 * 2 * 8,
        },
        "port_candidates": candidates,
        "threshold_bitplane_forwarding": {
            "output_values": output_values,
            "q24_materialized_output_bytes": q24_output_bytes,
            "bitpack_output_bytes": bitpack_output_bytes,
            "output_payload_reduction": q24_output_bytes / float(bitpack_output_bytes),
            "input_int8_bytes": input_bytes,
            "input_plus_q24_output_bytes": input_bytes + q24_output_bytes,
            "input_plus_bitpack_output_bytes": input_bytes + bitpack_output_bytes,
            "boundary_payload_reduction": (
                (input_bytes + q24_output_bytes)
                / float(input_bytes + bitpack_output_bytes)
            ),
            "semantic_admission": (
                "PENDING proof that {0,threshold} amplitude is restored or folded "
                "exactly at every downstream consumer"
            ),
        },
        "claim_boundary": {
            "permitted": (
                "corrected compute/I-O DSE and the port width needed to make the "
                "M26 packed-T2 lower bound executable"
            ),
            "forbidden": [
                "claiming M26 2.035x/2.047x for a single 256b lanes16 path",
                "claiming trained rank3 INT8 accuracy",
                "claiming threshold bitplane forwarding preserves network semantics",
                "claiming end-to-end cycles, energy, PPA, FPS, or DATE comparison",
            ],
        },
        "headline_admitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m26", type=Path, default=DEFAULT_M26)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite M30 DSE: {}".format(args.output))
    report = build_report(args.m26.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
