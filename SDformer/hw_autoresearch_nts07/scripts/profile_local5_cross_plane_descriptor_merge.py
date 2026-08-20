#!/usr/bin/env python3
"""Profile exact cross-plane Local5 source-descriptor merging."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROLES = 5
K_BITS = 32
GATE_BITS = 9
VALID_BITS = 5
SOURCES_PER_PLANE = 225
RTL_ADMISSION_TERM_REDUCTION = 0.10


def profile_arrays(
    *,
    group_offsets: np.ndarray,
    planes: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    k_bitmaps: np.ndarray,
    valid_masks: np.ndarray,
    gates: np.ndarray,
    terms: np.ndarray,
    sources_per_plane: int = SOURCES_PER_PLANE,
) -> dict:
    arrays = (planes, ys, xs, k_bitmaps, valid_masks, terms)
    descriptor_count = int(group_offsets[-1])
    if any(array.shape != (descriptor_count,) for array in arrays):
        raise ValueError("descriptor scalar array shape mismatch")
    if gates.shape != (descriptor_count, ROLES):
        raise ValueError("descriptor gate array shape mismatch")
    groups = len(group_offsets) - 1
    expected_per_group = 2 * sources_per_plane

    plane_pairs = 0
    any_active = 0
    both_active = 0
    payload_equal = 0
    payload_equal_active = 0
    empty_equal = 0
    saved_terms = 0
    for group in range(groups):
        lo = int(group_offsets[group])
        hi = int(group_offsets[group + 1])
        if hi - lo != expected_per_group:
            raise ValueError(f"group {group} descriptor count is not {expected_per_group}")
        indexes = range(lo, hi)
        left = {
            (int(ys[index]), int(xs[index])): index
            for index in indexes
            if int(planes[index]) == 0
        }
        right = {
            (int(ys[index]), int(xs[index])): index
            for index in indexes
            if int(planes[index]) == 1
        }
        if left.keys() != right.keys() or len(left) != sources_per_plane:
            raise ValueError(f"group {group} plane coordinate map mismatch")
        for coordinate, left_index in left.items():
            right_index = right[coordinate]
            plane_pairs += 1
            left_terms = int(terms[left_index])
            right_terms = int(terms[right_index])
            any_active += int(left_terms > 0 or right_terms > 0)
            both_active += int(left_terms > 0 and right_terms > 0)
            equal = (
                int(k_bitmaps[left_index]) == int(k_bitmaps[right_index])
                and int(valid_masks[left_index]) == int(valid_masks[right_index])
                and np.array_equal(gates[left_index], gates[right_index])
            )
            payload_equal += int(equal)
            if equal and left_terms > 0 and right_terms > 0:
                if left_terms != right_terms:
                    raise ValueError("equal active payload has unequal term counts")
                payload_equal_active += 1
                saved_terms += left_terms
            if equal and left_terms == 0 and right_terms == 0:
                empty_equal += 1

    baseline_terms = int(np.asarray(terms, dtype=np.int64).sum())
    term_reduction = saved_terms / baseline_terms if baseline_terms else 0.0
    buffer_lower_bound = sources_per_plane * (
        K_BITS + ROLES * GATE_BITS + VALID_BITS
    )
    return {
        "groups": groups,
        "descriptors": descriptor_count,
        "plane_pairs": plane_pairs,
        "any_active_pairs": any_active,
        "both_active_pairs": both_active,
        "payload_equal_pairs": payload_equal,
        "payload_equal_active_pairs": payload_equal_active,
        "empty_equal_pairs": empty_equal,
        "baseline_source_terms": baseline_terms,
        "theoretical_saved_source_terms": saved_terms,
        "active_pair_equal_rate": (
            payload_equal_active / both_active if both_active else 0.0
        ),
        "term_reduction_upper_bound": term_reduction,
        "term_speedup_upper_bound": (
            baseline_terms / (baseline_terms - saved_terms)
            if baseline_terms > saved_terms
            else None
        ),
        "empty_fraction_of_equal_pairs": (
            empty_equal / payload_equal if payload_equal else 0.0
        ),
        "one_plane_payload_buffer_lower_bound_bits": buffer_lower_bound,
        "reference_fcsr_ring_bits": 3465,
        "buffer_ratio_vs_fcsr_ring": buffer_lower_bound / 3465,
        "rtl_admission_threshold": RTL_ADMISSION_TERM_REDUCTION,
        "rtl_admitted": term_reduction >= RTL_ADMISSION_TERM_REDUCTION,
    }


def render_markdown(payload: dict) -> str:
    return f"""# Local5 cross-plane source-descriptor merge screening

Status: `{payload['status']}`

This is `[prof]` evidence over sampled post-G0 groups, not RTL cycles or a full workload.

| Metric | Value |
|---|---:|
| groups | {payload['groups']} |
| descriptors | {payload['descriptors']} |
| temporal-plane pairs | {payload['plane_pairs']} |
| both-active pairs | {payload['both_active_pairs']} |
| equal active payloads | {payload['payload_equal_active_pairs']} |
| active-pair equality | {100.0 * payload['active_pair_equal_rate']:.6f}% |
| baseline source terms | {payload['baseline_source_terms']} |
| theoretical saved terms | {payload['theoretical_saved_source_terms']} |
| term reduction upper bound | {100.0 * payload['term_reduction_upper_bound']:.6f}% |
| term-only speedup upper bound | {payload['term_speedup_upper_bound']:.9f}x |
| equal hits that are already empty | {100.0 * payload['empty_fraction_of_equal_pairs']:.6f}% |
| one-plane payload buffer lower bound | {payload['one_plane_payload_buffer_lower_bound_bits']} bit |
| buffer / existing FCSR ring | {payload['buffer_ratio_vs_fcsr_ring']:.3f}x |

The pre-registered RTL gate is at least {100.0 * payload['rtl_admission_threshold']:.1f}% source-term reduction. The measured upper bound is only {100.0 * payload['term_reduction_upper_bound']:.4f}%, so this candidate is closed without RTL.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with np.load(args.npz, allow_pickle=False) as arrays:
        metrics = profile_arrays(
            group_offsets=arrays["descriptor_group_offsets"],
            planes=arrays["descriptor_source_plane"],
            ys=arrays["descriptor_source_y"],
            xs=arrays["descriptor_source_x"],
            k_bitmaps=arrays["descriptor_k_bitmap"],
            valid_masks=arrays["descriptor_valid_mask"],
            gates=arrays["descriptor_incoming_gates"],
            terms=arrays["source_term_count"],
        )
    payload = {
        "schema": "local5_cross_plane_descriptor_merge_profile_v1",
        "status": "NO_GO_NO_RTL" if not metrics["rtl_admitted"] else "CONDITIONAL_RTL",
        "source_npz": str(args.npz),
        "claim_boundary": {
            "evidence": "post_g0_profile",
            "not_rtl_cycles": True,
            "not_full_workload": True,
            "docs359_update": False,
        },
        **metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    print(json.dumps({"status": payload["status"], "term_reduction": payload["term_reduction_upper_bound"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
