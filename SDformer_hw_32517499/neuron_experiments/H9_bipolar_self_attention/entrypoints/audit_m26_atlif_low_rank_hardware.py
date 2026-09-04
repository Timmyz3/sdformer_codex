#!/usr/bin/env python3
"""Checkpoint-bound DSE for a resource-shared low-rank ATLIF engine.

This audit never treats a truncated SVD as an accuracy-preserving model.  It
only measures the frozen checkpoint matrices and evaluates cycle candidates
that would become legal after a separately trained factorized checkpoint has
passed the normal accuracy and load-identity gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

import profile_nts11_hardware_p0 as profiler


STATUS = (
    "PASS_CHECKPOINT_BOUND_FACTOR_ARITHMETIC_LOWER_BOUND_"
    "TRAINING_REQUIRED_NO_SPEEDUP_CLAIM"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ceil_div(value: int, divisor: int) -> int:
    if value < 0 or divisor <= 0:
        raise ValueError("invalid ceil-div operands")
    return (int(value) + int(divisor) - 1) // int(divisor)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires values")
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, int(math.floor(fraction * len(ordered))))
    return ordered[index]


def matrix_record(name: str, module: torch.nn.Module) -> dict:
    weight = module.weight.detach().float().cpu()
    if weight.ndim != 2 or weight.shape[0] != weight.shape[1]:
        raise ValueError("ATLIF temporal weight must be square: {}".format(name))
    temporal = int(weight.shape[0])
    singular = torch.linalg.svdvals(weight)
    energy = singular.square()
    total = float(energy.sum().item())
    if total <= 0.0:
        raise ValueError("ATLIF temporal weight has zero energy: {}".format(name))
    tolerance = max(float(singular.max().item()) * temporal * 1.0e-6, 1.0e-12)
    record = {
        "name": name,
        "temporal_steps": temporal,
        "dense_parameters": int(weight.numel()),
        "numerical_rank_tolerance": tolerance,
        "numerical_rank": int(singular.gt(tolerance).sum().item()),
        "diagonal_energy_fraction": (
            float(torch.diagonal(weight).square().sum().item()) / total
        ),
        "singular_values": [float(value) for value in singular.tolist()],
    }
    cumulative = torch.cumsum(energy, dim=0)
    for rank in range(1, temporal + 1):
        record["rank{}_energy_fraction".format(rank)] = (
            float(cumulative[rank - 1].item()) / total
        )
    return record


def factor_service_cycles(
    vectors: int,
    temporal: int,
    rank: int,
    multipliers: int,
    vector_tile: int,
) -> tuple[int, str, int]:
    dense_macs = temporal * temporal
    factor_macs = 2 * temporal * rank
    if factor_macs >= dense_macs:
        return ceil_div(vectors * dense_macs, multipliers), "DENSE_FALLBACK", 0
    full_tiles, tail = divmod(int(vectors), int(vector_tile))
    # The two dependent matrix-vector stages are separately scheduled.  No
    # cross-stage overlap or fractional multiplier interpolation is used.
    cycles = full_tiles * 2 * ceil_div(
        vector_tile * temporal * rank, multipliers
    )
    if tail:
        cycles += 2 * ceil_div(tail * temporal * rank, multipliers)
    return cycles, "TWO_STAGE_FACTOR_TILE", full_tiles + int(bool(tail))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--m25", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--multipliers", type=int, default=96)
    parser.add_argument("--vector-tile", type=int, default=16)
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 3, 4, 5])
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite output: {}".format(args.output))
    if args.multipliers <= 0 or args.vector_tile <= 0:
        raise ValueError("resource counts must be positive")
    if not args.ranks or any(rank <= 0 for rank in args.ranks):
        raise ValueError("candidate ranks must be positive")

    profile_payload = json.loads(args.profile.read_text(encoding="utf-8"))
    profile_samples = int(profile_payload.get("samples", 0))
    if profile_samples <= 0:
        raise ValueError("profile sample count is missing")
    profile_rows = profile_payload.get("summary", {}).get("atlif_rows", [])
    if not profile_rows:
        raise ValueError("profile has no ATLIF rows")
    for row in profile_rows:
        if type(row.get("deployment_dead_result")) is not bool:
            raise ValueError(
                "deployment_dead_result must be an explicit JSON boolean: {}".format(
                    row.get("name", "<missing-name>")
                )
            )
    profile_by_name = {row["name"]: row for row in profile_rows}
    if len(profile_by_name) != len(profile_rows):
        raise ValueError("profile ATLIF names are not unique")

    config, _ = profiler.load_config(args.config)
    model = profiler.build_model(config, args.checkpoint, torch.device("cpu"))
    matrices = []
    module_by_name = {}
    for name, module in model.named_modules():
        if module.__class__.__name__ != "ATLIFTernaryPSN":
            continue
        module_by_name[name] = module
        matrices.append(matrix_record(name, module))
    if len(matrices) != 105:
        raise ValueError("expected 105 ATLIF modules, found {}".format(len(matrices)))
    checkpoint_names = set(module_by_name)
    profile_names = set(profile_by_name)
    if not profile_names.issubset(checkpoint_names):
        raise ValueError("profile ATLIF name is absent from checkpoint model")
    live_profile_names = {
        row["name"]
        for row in profile_rows
        if not row["deployment_dead_result"]
    }
    dead_profile_names = profile_names - live_profile_names
    uncalled_names = checkpoint_names - profile_names
    if (
        len(profile_names) != 93
        or len(live_profile_names) != 81
        or len(dead_profile_names) != 12
        or len(uncalled_names) != 12
        or any(
            not name.endswith(".attn.attn_sn.spiking_neuron")
            for name in dead_profile_names
        )
        or any(
            not name.endswith(".sn2_q.spiking_neuron")
            for name in uncalled_names
        )
    ):
        raise ValueError(
            "ATLIF execution partition changed: live={} dead={} uncalled={} "
            "dead_non_attn_sn={} uncalled_non_sn2_q={}".format(
                len(live_profile_names),
                len(dead_profile_names),
                len(uncalled_names),
                sorted(
                    name
                    for name in dead_profile_names
                    if not name.endswith(".attn.attn_sn.spiking_neuron")
                ),
                sorted(
                    name
                    for name in uncalled_names
                    if not name.endswith(".sn2_q.spiking_neuron")
                ),
            )
        )

    m25 = json.loads(args.m25.read_text(encoding="utf-8"))
    if m25.get("status") not in {
        "PASS_RESOURCE_TILING_AND_CYCLE_ENVELOPE_HEADLINE_NO_GO",
        "PASS_FROZEN_C4_TILING_AND_CYCLE_ENVELOPE_HEADLINE_NO_GO",
    }:
        raise ValueError("M25 cycle envelope is not admitted")
    sensitivity = m25.get("bandwidth_sram_sensitivity", [])
    if not sensitivity:
        raise ValueError("M25 lacks its fixed-cycle identity")
    fixed_cycle_values = {int(row["fixed_compute_cycles"]) for row in sensitivity}
    if len(fixed_cycle_values) != 1:
        raise ValueError("M25 fixed-cycle identity is inconsistent")
    fixed_cycles = fixed_cycle_values.pop()
    frequency_mhz = float(m25["uniform_resource_contract"]["frequency_mhz"])
    local_anchor = m25["compute_envelopes"]["local"]["8"]
    hybrid_anchor = m25["compute_envelopes"]["hybrid"]["8"]
    shared_non_atlif_local = (
        int(local_anchor["noneligible_plus_qk_cycles"])
        + int(local_anchor["rqtb_attention_cycles"])
        + int(local_anchor["accelerated_m4_cycles"])
        + int(local_anchor.get("m21_fifo4_phase1_incremental_cycles", 0))
        + int(local_anchor["m21_registered_result_bubble_cycles"])
    )
    shared_non_atlif_hybrid = (
        int(hybrid_anchor["noneligible_plus_qk_cycles"])
        + int(hybrid_anchor["rqtb_attention_cycles"])
        + int(hybrid_anchor["accelerated_m4_cycles"])
        + int(hybrid_anchor.get("m21_fifo4_phase1_incremental_cycles", 0))
        + int(hybrid_anchor["m21_registered_result_bubble_cycles"])
    )

    candidates = []
    live_dense_macs = sum(
        (int(row["elements"]) // profile_samples) * int(row["temporal_steps"])
        for row in profile_rows
        if not row["deployment_dead_result"]
    )
    for rank in sorted(set(args.ranks)):
        service_cycles = 0
        factor_tiles = 0
        candidate_macs = 0
        dense_fallback_macs = 0
        factor_macs = 0
        live_modules = 0
        factor_modules = 0
        intermediate_values = 0
        live_parameter_entries = 0
        live_bias_entries = 0
        live_threshold_entries = 0
        for row in profile_rows:
            if row["deployment_dead_result"]:
                continue
            temporal = int(row["temporal_steps"])
            elements = int(row["elements"])
            if (
                temporal <= 0
                or elements % profile_samples
                or (elements // profile_samples) % temporal
            ):
                raise ValueError("ATLIF profile sample/element/temporal mismatch")
            vectors = (elements // profile_samples) // temporal
            cycles, mode, tiles = factor_service_cycles(
                vectors, temporal, rank, args.multipliers, args.vector_tile
            )
            service_cycles += cycles
            live_modules += 1
            if mode == "TWO_STAGE_FACTOR_TILE":
                macs = vectors * 2 * temporal * rank
                factor_modules += 1
                factor_macs += macs
                factor_tiles += tiles
                intermediate_values += vectors * rank
                live_parameter_entries += 2 * temporal * rank
            else:
                macs = vectors * temporal * temporal
                dense_fallback_macs += macs
                live_parameter_entries += temporal * temporal
            live_bias_entries += temporal
            live_threshold_entries += 1
            candidate_macs += macs
        local_cycles = shared_non_atlif_local + service_cycles
        hybrid_cycles = shared_non_atlif_hybrid + service_cycles
        overhead_sensitivity = []
        for overhead_cycles_per_factor_tile in (0, 1, 2):
            overhead_cycles = factor_tiles * overhead_cycles_per_factor_tile
            sensitivity_local_cycles = local_cycles + overhead_cycles
            sensitivity_hybrid_cycles = hybrid_cycles + overhead_cycles
            overhead_sensitivity.append({
                "overhead_cycles_per_factor_tile": overhead_cycles_per_factor_tile,
                "factor_tile_overhead_cycles": overhead_cycles,
                "local_arithmetic_cycles": sensitivity_local_cycles,
                "motion_arithmetic_cycles": sensitivity_hybrid_cycles,
                "local_speedup_vs_fixed": (
                    float(fixed_cycles) / sensitivity_local_cycles
                ),
                "motion_speedup_vs_fixed": (
                    float(fixed_cycles) / sensitivity_hybrid_cycles
                ),
                "crosses_2x_local": sensitivity_local_cycles * 2 <= fixed_cycles,
                "crosses_2x_motion": sensitivity_hybrid_cycles * 2 <= fixed_cycles,
            })
        model_parameter_entries = sum(
            (
                2 * int(row["temporal_steps"]) * rank
                if 2 * int(row["temporal_steps"]) * rank
                < int(row["temporal_steps"]) ** 2
                else int(row["temporal_steps"]) ** 2
            )
            for row in matrices
        )
        candidates.append({
            "rank": rank,
            "multipliers": args.multipliers,
            "vector_tile": args.vector_tile,
            "live_modules": live_modules,
            "factorized_modules": factor_modules,
            "factor_tiles": factor_tiles,
            "dense_fallback_macs": dense_fallback_macs,
            "factorized_macs": factor_macs,
            "candidate_atlif_macs": candidate_macs,
            "frozen_dense_atlif_macs": live_dense_macs,
            "atlif_arithmetic_reduction_fraction": (
                1.0 - float(candidate_macs) / live_dense_macs
            ),
            "atlif_arithmetic_issue_lower_bound_cycles": service_cycles,
            "local_arithmetic_lower_bound_cycles_if_trained_model_admitted": local_cycles,
            "motion_arithmetic_lower_bound_cycles_if_trained_model_admitted": hybrid_cycles,
            "local_arithmetic_lower_bound_speedup_vs_fixed_if_trained_model_admitted": (
                float(fixed_cycles) / local_cycles
            ),
            "motion_arithmetic_lower_bound_speedup_vs_fixed_if_trained_model_admitted": (
                float(fixed_cycles) / hybrid_cycles
            ),
            "arithmetic_lower_bound_crosses_2x_local_if_trained_model_admitted": (
                local_cycles * 2 <= fixed_cycles
            ),
            "arithmetic_lower_bound_crosses_2x_motion_if_trained_model_admitted": (
                hybrid_cycles * 2 <= fixed_cycles
            ),
            "cycle_per_factor_tile_overhead_sensitivity": overhead_sensitivity,
            "precision_contract": {
                "factor_bits_candidate": 8,
                "stage1_accumulator_bits_minimum": 24,
                "stage2_multiplier_input_bits_required_for_same_96_int8_pool": 8,
                "q24_to_q8_requantization_required": True,
                "requantization_rounding_saturation_scale_frozen": False,
                "same_resource_cycle_point_admitted": False,
            },
            "factor_state_contract": {
                "live_temporal_weight_entries": live_parameter_entries,
                "model_temporal_weight_entries": model_parameter_entries,
                "model_temporal_weight_bytes_if_int8": model_parameter_entries,
                "live_bias_entries_not_in_temporal_weight_count": live_bias_entries,
                "live_threshold_entries_not_in_temporal_weight_count": (
                    live_threshold_entries
                ),
                "model_bias_entries_not_in_temporal_weight_count": sum(
                    int(row["temporal_steps"]) for row in matrices
                ),
                "model_threshold_entries_not_in_temporal_weight_count": len(matrices),
                "unique_live_temporal_weight_stream_bytes_per_frame_if_int8": (
                    live_parameter_entries
                ),
                "model_temporal_weight_cold_fill_bytes_if_int8": (
                    model_parameter_entries
                ),
                "per_factor_tile_external_weight_reload_forbidden": True,
                "parameter_load_cycles_included": False,
                "intermediate_values_per_frame": intermediate_values,
                "intermediate_bytes_per_frame_if_q24_materialized_once": (
                    intermediate_values * 3
                ),
                "external_write_plus_read_bytes_if_q24_materialized": (
                    intermediate_values * 3 * 2
                ),
                "minimum_tile_resident_intermediate_bytes_q24": (
                    args.vector_tile * rank * 3 if factor_modules else 0
                ),
                "minimum_double_buffered_intermediate_bytes_q24": (
                    args.vector_tile * rank * 3 * 2 if factor_modules else 0
                ),
                "tile_resident_bank_port_rtl_frozen": False,
            },
            "headline_admitted": False,
        })

    t10 = [row for row in matrices if int(row["temporal_steps"]) == 10]
    t2 = [row for row in matrices if int(row["temporal_steps"]) == 2]
    if len(t10) != 45 or len(t2) != 60:
        raise ValueError("unexpected T10/T2 checkpoint matrix census")
    rank_summary = {}
    for rank in (1, 2, 3, 4):
        field = "rank{}_energy_fraction".format(rank)
        values = [float(row[field]) for row in t10]
        rank_summary[str(rank)] = {
            "minimum": min(values),
            "median": percentile(values, 0.5),
            "maximum": max(values),
        }

    payload = {
        "schema": "m26_atlif_factor_arithmetic_lower_bound_v2",
        "status": STATUS,
        "identity": {
            "config": str(args.config),
            "config_sha256": sha256(args.config),
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "profile": str(args.profile),
            "profile_sha256": sha256(args.profile),
            "profile_samples": profile_samples,
            "m25": str(args.m25),
            "m25_sha256": sha256(args.m25),
            "profiler_sha256": sha256(Path(profiler.__file__)),
            "generator_sha256": sha256(Path(__file__)),
        },
        "checkpoint_matrix_census": {
            "total": len(matrices),
            "t10": len(t10),
            "t2": len(t2),
            "execution_partition": {
                "profile_forward_modules": len(profile_names),
                "live_modules": len(live_profile_names),
                "deployment_dead_modules": len(dead_profile_names),
                "deployment_dead_names": sorted(dead_profile_names),
                "checkpoint_installed_but_uncalled_modules": len(uncalled_names),
                "uncalled_reason": (
                    "all are attention sn2_q modules bypassed by the frozen "
                    "attention implementation"
                ),
                "uncalled_names": sorted(uncalled_names),
            },
            "t10_numerically_rank_le3": sum(
                int(row["numerical_rank"]) <= 3 for row in t10
            ),
            "t10_diagonal_energy_fraction": {
                "minimum": min(float(row["diagonal_energy_fraction"]) for row in t10),
                "median": percentile(
                    [float(row["diagonal_energy_fraction"]) for row in t10], 0.5
                ),
                "maximum": max(float(row["diagonal_energy_fraction"]) for row in t10),
            },
            "t10_svd_energy_fraction": rank_summary,
        },
        "cycle_contract": {
            "frequency_mhz": frequency_mhz,
            "fixed_compute_cycles": fixed_cycles,
            "shared_non_atlif_local_cycles": shared_non_atlif_local,
            "shared_non_atlif_hybrid_cycles": shared_non_atlif_hybrid,
            "factor_arithmetic_lower_bound": (
                "for each vector tile, issue rank-by-T then T-by-rank products "
                "sequentially onto the declared multiplier pool; pipeline drain, "
                "phase switching, reduction routing and requantization are absent"
            ),
            "no_fractional_lane_interpolation": True,
            "pipeline_and_phase_overhead_included": False,
            "same_int8_resource_precision_closed": False,
            "memory_and_dram_cycles_included": False,
        },
        "candidates": candidates,
        "matrices": matrices,
        "claim_boundary": {
            "permitted": (
                "checkpoint-bound matrix-structure audit and an exactly-96-"
                "multiplier product-issue arithmetic lower bound for a future "
                "trained factorized model"
            ),
            "forbidden": [
                "claiming truncated SVD preserves accuracy",
                "claiming the frozen checkpoint is rank-factorized",
                "claiming system speedup before factorized fine-tuning and valid825",
                "claiming DRAM, energy, PPA, or paper-comparable performance",
            ],
            "admission_required": [
                "factorized Local and H67 checkpoints with strict load identity",
                "valid825 accuracy and equal-rate reporting",
                "bit-exact factor RTL replay using the trained factors",
                "Q24-to-Q8 requantization with frozen scale, RNE, and saturation",
                "tile-resident intermediate bank/port and phase-overhead schedule",
                "same-frequency area-constrained DC and address-timed memory model",
            ],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": STATUS,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "candidates": candidates,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
