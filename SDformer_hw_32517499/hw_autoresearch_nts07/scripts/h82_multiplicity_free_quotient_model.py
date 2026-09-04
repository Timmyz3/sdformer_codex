#!/usr/bin/env python3
"""H82 multiplicity-free quotient-file architecture screening model.

This is a CPU-only analytical go/no-go model. It does not touch the live H82
training process, production RTL, docs/359, or the Synopsys handoff.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


PAIRS = 225
TOKENS = 450
N_CLASSES = 513
CLASS_ID_W = 9
GATE_W = 9
COUNT_W = 9
PAIR_ID_W = 8
K_MASK_W = 2
PAIR_LAST_W = 1
DENOM_SHIFT_W = 5
ROW_MAX_W = 9


def descriptor_count(equal_pairs: int) -> int:
    if not 0 <= equal_pairs <= PAIRS:
        raise ValueError(f"equal_pairs must be in [0,{PAIRS}]")
    return 2 * PAIRS - equal_pairs


def model_point(n_occupied: int, equal_pairs: int) -> dict:
    if not 1 <= n_occupied <= TOKENS:
        raise ValueError(f"n_occupied must be in [1,{TOKENS}]")
    descriptors = descriptor_count(equal_pairs)

    occupancy_bits = N_CLASSES
    compact_gate_bits = n_occupied * GATE_W
    token_code_bits = TOKENS * CLASS_ID_W
    token_gate_bits = TOKENS * GATE_W
    fixed_pair_bits = PAIRS * (2 * CLASS_ID_W + 1)
    quotient_descriptor_bits = descriptors * (
        CLASS_ID_W + K_MASK_W + PAIR_LAST_W
    )
    denominator_certificate_bits = ROW_MAX_W + DENOM_SHIFT_W

    token_gate_materialized = {
        "state_bits": occupancy_bits
        + compact_gate_bits
        + token_code_bits
        + token_gate_bits,
        "normalization_exp2": n_occupied,
        "gate_file_writes": n_occupied,
        "gate_file_reads": TOKENS,
        "token_gate_writes": TOKENS,
        "token_gate_reads": TOKENS,
        "lower_bound_cycles": TOKENS + n_occupied + TOKENS + TOKENS,
        "ports": "one class-gate read plus token-gate scratch",
    }
    pair_order_gate_gather = {
        "state_bits": occupancy_bits + compact_gate_bits + fixed_pair_bits,
        "normalization_exp2": n_occupied,
        "gate_file_writes": n_occupied,
        "gate_file_reads": descriptors,
        "token_gate_writes": 0,
        "token_gate_reads": 0,
        "lower_bound_cycles_1r_gate": TOKENS + n_occupied + descriptors,
        "lower_bound_cycles_2r_gate": TOKENS + n_occupied + PAIRS,
        "ports": "fixed pair record; two-read gate RF for the fastest bound",
    }
    quotient_gate_file = {
        "state_bits": occupancy_bits
        + compact_gate_bits
        + quotient_descriptor_bits,
        "normalization_exp2": n_occupied,
        "expand_exp2": 0,
        "gate_file_writes": n_occupied,
        "gate_file_reads": descriptors,
        "token_gate_writes": 0,
        "token_gate_reads": 0,
        "lower_bound_cycles": TOKENS + n_occupied + descriptors,
        "ports": "one descriptor and one compact-gate read per cycle; k_mask may select both K banks",
    }
    denominator_only_quotient = {
        "state_bits": occupancy_bits
        + denominator_certificate_bits
        + quotient_descriptor_bits,
        "normalization_exp2": n_occupied,
        "expand_exp2": descriptors,
        "gate_file_writes": 0,
        "gate_file_reads": 0,
        "token_gate_writes": 0,
        "token_gate_reads": 0,
        "lower_bound_cycles": TOKENS + n_occupied + descriptors,
        "ports": "one descriptor plus combinational/pipelined exp2 regeneration per cycle",
    }
    class_stationary_csr = {
        "state_bits_steady": occupancy_bits
        + n_occupied * (COUNT_W + GATE_W)
        + descriptors * (PAIR_ID_W + K_MASK_W),
        "normalization_exp2": n_occupied,
        "gate_file_writes": n_occupied,
        "gate_file_reads": n_occupied,
        "token_gate_writes": 0,
        "token_gate_reads": 0,
        "reorder_lower_bound_cycles": descriptors,
        "lower_bound_cycles": TOKENS
        + descriptors
        + n_occupied
        + descriptors,
        "ports": "class-stationary CSR, but arbitrary arrival order requires reorder or linked-list state",
    }

    candidates = {
        "token_gate_materialized": token_gate_materialized,
        "pair_order_gate_gather_strong_baseline": pair_order_gate_gather,
        "multiplicity_free_quotient_gate_file": quotient_gate_file,
        "multiplicity_free_denominator_only_quotient": denominator_only_quotient,
        "class_stationary_csr_with_reorder": class_stationary_csr,
    }
    baseline_bits = pair_order_gate_gather["state_bits"]
    for candidate in (
        quotient_gate_file,
        denominator_only_quotient,
        class_stationary_csr,
    ):
        candidate["state_reduction_vs_pair_gather"] = 1.0 - (
            candidate["state_bits"]
            if "state_bits" in candidate
            else candidate["state_bits_steady"]
        ) / baseline_bits

    q_state_win = quotient_gate_file["state_reduction_vs_pair_gather"]
    d_state_win = denominator_only_quotient["state_reduction_vs_pair_gather"]
    descriptor_ratio = descriptors / TOKENS
    return {
        "n_occupied": n_occupied,
        "equal_pairs": equal_pairs,
        "descriptor_count": descriptors,
        "descriptor_ratio_vs_tokens": descriptor_ratio,
        "candidates": candidates,
        "screen": {
            "quotient_gate_file_state_win_ge_20pct": q_state_win >= 0.20,
            "denominator_only_state_win_ge_25pct": d_state_win >= 0.25,
            "descriptor_reduction_ge_40pct": descriptor_ratio <= 0.60,
            "class_stationary_reorder_is_not_free": True,
            "sidecar_rtl_profile_gate": (
                q_state_win >= 0.20
                and descriptor_ratio <= 0.60
                and n_occupied <= 192
            ),
        },
    }


def sweep() -> dict:
    occupancies = (32, 64, 128, 192, 256)
    equal_rates = (0.84, 0.90, 0.94, 0.98)
    points = []
    for occupied in occupancies:
        for rate in equal_rates:
            equal_pairs = int(round(PAIRS * rate))
            point = model_point(occupied, equal_pairs)
            point["equal_rate"] = equal_pairs / PAIRS
            points.append(point)
    admitted = [
        {
            "n_occupied": point["n_occupied"],
            "equal_rate": point["equal_rate"],
        }
        for point in points
        if point["screen"]["sidecar_rtl_profile_gate"]
    ]
    return {
        "schema": "h82_multiplicity_free_quotient_model_v1",
        "status": "CONDITIONAL_PROFILE_GATE_SUPPORT_ONLY_NO_RTL",
        "exact_contract": {
            "normalization_operand": "513-bit occupied-class bitmap",
            "denominator": "sum exp2(class_id-row_max), no multiplicity term",
            "membership": "temporal quotient descriptor (class_id,k_mask,pair_last)",
            "expand": "descriptor class regenerates or reads gate; k_mask preserves both temporal K destinations",
            "forbidden": [
                "class_sum_term = exp * multiplicity",
                "materialized token_gate[450]",
                "class-wise K folding that drops pair/destination identity",
            ],
        },
        "strong_baselines": [
            "H82 class gate + fixed two-class pair record + direct gate gather",
            "H82 class gate + materialized 450-entry token-gate scratch",
            "class-stationary CSR including mandatory reorder lower bound",
        ],
        "sweep": points,
        "admitted_sensitivity_points": admitted,
        "rank1_profile_gate": {
            "required": [
                "n_occupied p50/p95/max on 513-bin H82 rank-1",
                "temporal equal-pair rate including separate active-pair rate",
                "descriptor count and K-active descriptor count",
            ],
            "thresholds": {
                "p95_n_occupied_lte": 192,
                "descriptor_ratio_lte": 0.60,
                "state_reduction_vs_pair_gather_gte": 0.20,
            },
            "reason": "Without the rank-1 occupancy distribution, H67 C_occ=10 cannot be reused as H82 evidence.",
        },
        "post_profile_production_gate": {
            "required": [
                "bit-exact sidecar against fused token-major H82, including random backpressure",
                "score-to-projection cycles improve by at least 10 percent or component dynamic energy improves by at least 15 percent",
                "all builder metadata and descriptor SRAM included under the same port model",
                "logic plus macro area increase no more than 10 percent and Fmax loss no more than 5 percent",
            ],
            "current_status": "BLOCKED_BY_RANK1_PROFILE_AND_NO_RTL",
        },
        "claim_boundary": {
            "candidate_role": "H82_dataflow_support_not_standalone_contribution",
            "date_4_0": False,
            "production_rtl": False,
            "docs359_update": False,
            "innovation_ceiling_before_rank1_and_rtl": 3.2,
        },
    }


def render_markdown(payload: dict) -> str:
    sample = {
        (point["n_occupied"], round(point["equal_rate"], 2)): point
        for point in payload["sweep"]
    }
    lines = [
        "# H82 multiplicity-free quotient-file screening",
        "",
        f"Status: `{payload['status']}`",
        "",
        "The candidate replaces multiplicity histogram plus token-gate materialization "
        "with an occupied-class bitmap, a row denominator certificate or compact gate "
        "file, and temporal quotient descriptors. This is a model, not RTL or PPA.",
        "",
        "| C_occ | equal pairs | descriptors | pair-gather bits | quotient-gate bits | denom-only bits | RTL gate |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for occupied, rate in ((64, 0.94), (128, 0.94), (192, 0.94), (128, 0.98)):
        point = sample[(occupied, rate)]
        candidates = point["candidates"]
        lines.append(
            f"| {occupied} | {point['equal_pairs']} | {point['descriptor_count']} | "
            f"{candidates['pair_order_gate_gather_strong_baseline']['state_bits']} | "
            f"{candidates['multiplicity_free_quotient_gate_file']['state_bits']} | "
            f"{candidates['multiplicity_free_denominator_only_quotient']['state_bits']} | "
            f"{'PASS' if point['screen']['sidecar_rtl_profile_gate'] else 'DENY'} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "The object is architecturally distinct from H86 member-delta: it does not "
            "depend on cross-row member reuse. It is still conditional because H82 "
            "rank-1 occupancy is unknown and class-stationary reorder is not free.",
            "",
            "Do not write even sidecar RTL until rank-1 shows p95 occupied classes <=192, "
            "descriptor/token ratio <=0.60, and at least 20% state reduction against the "
            "fused direct pair-gather baseline. Production admission additionally requires "
            "at least 10% cycle or 15% dynamic-energy improvement under matched ports.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    payload = sweep()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "model.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "admitted_points": len(payload["admitted_sensitivity_points"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
