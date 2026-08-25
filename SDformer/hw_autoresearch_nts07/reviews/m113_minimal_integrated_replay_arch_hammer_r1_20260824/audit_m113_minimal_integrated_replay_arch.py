#!/usr/bin/env python3
"""Read-only architecture and projection audit for the proposed M113 boundary."""

import hashlib
import json
import math
from pathlib import Path


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
OUTPUT = REVIEW / "m113_minimal_integrated_replay_arch_audit.json"

M109_CONTRACT = HW / "contracts/m109_r2_window_storage_dual_timeline_frontier_contract_r1_20260824.json"
M109_RESULT = HW / "results/m109_r2_window_storage_dual_timeline_frontier_r1_20260824/m109_r2_window_storage_dual_timeline_frontier.json"
M109_RUN = HW / "results/m109_r2_window_storage_dual_timeline_frontier_r1_20260824/RUN_COMPLETE.txt"
M109_REVIEW = HW / "reviews/m109_r2_window_storage_dual_timeline_frontier_independent_hammer_r1_20260824/m109_r2_window_storage_dual_timeline_frontier_independent_hammer_review.json"
M110_CONTRACT = HW / "contracts/m110_w384_full_capacity_transpose_vcs_contract_r1_20260824.json"
M110_RTL = HW / "rtl_m110/m110_w384_bounded_bitmap_transpose_scheduler.sv"
M110_RUN = HW / "dc_handoff/runs/m110_w384_full_capacity_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
M110_REVIEW = HW / "reviews/m110_w384_full_capacity_transpose_independent_hammer_r1_20260824/m110_w384_full_capacity_transpose_independent_hammer_review.json"
M111_CONTRACT = HW / "contracts/m111_w384_signed24_accumulator_vcs_contract_r1_20260824.json"
M111_RTL = HW / "rtl_m111/m111_w384_signed24_accumulator_frontend.sv"
M111_RUN = HW / "dc_handoff/runs/m111_w384_signed24_accumulator_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
M111_REVIEW = HW / "reviews/m111_w384_signed24_accumulator_independent_hammer_r1_20260824/m111_w384_signed24_accumulator_independent_hammer_review.json"
M112_CONTRACT = HW / "contracts/m112_w384_lane_sliced_accumulator_vcs_contract_r1_20260824.json"
M112_RTL = HW / "rtl_m112/m112_w384_lane_sliced_accumulator_adapter.sv"
M112_RUN = HW / "dc_handoff/runs/m112_w384_lane_sliced_accumulator_vcs_r1_sealed_20260824/RUN_COMPLETE.txt"
M108_RESULT = HW / "results/m108_r3_dual_timeline_fused_schedule_r1_20260824/m108_r3_dual_timeline_fused_schedule.json"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_SHA = {
    M109_CONTRACT: "d80efd387bb6b5b01371ca7ed5d07d8e2ec97f3efa93aa1d385cc80281f63b44",
    M109_RESULT: "ee61b90ee894c6e6c778b815a52f1d8b6edc9c877227bc4987e4b135aa16c321",
    M109_RUN: "c0eb8fd06d21da14b496c16f4709c85af55e0eb3bd9c44938c935bbf16c8a6c9",
    M109_REVIEW: "423a53a9d65cc274dad2deedad8e41f28afe08178506f31f234624ccb0e24f9f",
    M110_CONTRACT: "4f2b5c329ea552742c55a362739f032272fb510cc3c659b0c73f52eced9f5253",
    M110_RTL: "61a2c18f3b0a350bfc57193b9573f3d0ed5ea68f68ae4fc982ec1908054dbd6c",
    M110_RUN: "2b73e6e29fcd176ab17d479fa33c0d0d785d3e2b90719ec7047b9513f5acfef7",
    M110_REVIEW: "fcf1eb0dae7f7ac08140094141eef42f73410d5a328d9ae498a57d6a2b4d89ec",
    M111_CONTRACT: "672dbdf2d8eea1c1ef58036a58bf2d3ca14dabb8f5feb5aed8dcbe0e036d22ef",
    M111_RTL: "354e0de95ee4380098c09fac67af3e137b3ab8bb9f88ac706d62fe201179b43a",
    M111_RUN: "9a10f6e25b4451d17ce6849624bdf205d64548e7085986db74b4e75694088bcc",
    M111_REVIEW: "e4b5fbc45ccaf263b7b16393b5b54eb04c7fb9abab342e4eb50257025302204d",
    M112_CONTRACT: "8eb2d82c329bd1612d2808a1edfb13345eddaa770156adf7da172a008f981f44",
    M112_RTL: "ee5a2a84c8c28e113340c73195fc08eec4c975eed27622ea8eee654b3f25226e",
    M112_RUN: "458dc8af156165bf726d36a57813d2d476ec25dded82ffdee077c186f63bba26",
    M108_RESULT: "d5a4d7c27a91a7735ed4481100d0db3640191357e4617e043378bb367a77dacc",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(raw):
        raise ValueError("non-standard JSON constant: " + raw)

    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs_hook, parse_constant=reject)


def ratio(baseline, candidate):
    return baseline / float(candidate)


def main():
    start_sha = sha256(Path(__file__).resolve())
    observed = {}
    for path, expected in EXPECTED_SHA.items():
        actual = sha256(path)
        require(actual == expected, "identity mismatch {} {}".format(path, actual))
        observed[str(path.relative_to(HW))] = actual

    m109 = strict_json(M109_RESULT)
    m108 = strict_json(M108_RESULT)
    m110_contract = strict_json(M110_CONTRACT)
    m111_contract = strict_json(M111_CONTRACT)
    m112_contract = strict_json(M112_CONTRACT)
    frontier = {row["window_rows"]: row for row in m109["frontier"]}
    require({64, 384, 512, 1024, 3000}.issubset(frontier),
            "required frontier points absent")
    w64 = frontier[64]
    w384 = frontier[384]
    w512 = frontier[512]
    w1024 = frontier[1024]
    w3000 = frontier[3000]
    sched = w384["dual_timeline_recurrence"]
    work = w384["exact_work"]
    baseline = sched["fair_fixed8_baseline_cycles"]
    candidate = sched["candidate_cycles"]

    require(candidate == 439708199 and baseline == 1114863448,
            "W384 candidate/baseline drift")
    require(math.isclose(ratio(baseline, candidate), 2.53546204172554,
                         rel_tol=0.0, abs_tol=1e-15), "W384 ratio drift")
    require(work == {"events": 188148490, "groups": 8271296,
                     "pwp_tokens": 226222255}, "W384 work drift")
    require(sched["descriptors"] == 69120
            and sched["accumulator_pipeline_flush_cycles"] == 160
            and sched["accumulator_commit_cycles"] == 480000,
            "W384 descriptor/window tail drift")

    pwp_uses = {int(k): v for k, v in
                m108["work_conservation"]["pwp_uses_by_width"].items()}
    pwp_updates = m108["work_conservation"]["pwp_updates"]
    require(pwp_updates == sum(pwp_uses.values()) == 58969374,
            "PWP update conservation drift")
    beat_count = {8: 3, 9: 4, 10: 4, 11: 5}
    require(sum(pwp_uses[w] * beat_count[w] for w in pwp_uses)
            == work["pwp_tokens"], "PWP token reconstruction drift")
    pwp_payload_bits = sum(pwp_uses[w] * w * 96 for w in pwp_uses)
    ideal_cross_vector_tokens = (pwp_payload_bits + 255) // 256
    pwp_cross_pack_savings = work["pwp_tokens"] - ideal_cross_vector_tokens

    correction_load_tokens = 3 * work["groups"]
    correction_tokens = work["events"] + correction_load_tokens
    update_count = pwp_updates + work["events"]
    require(correction_tokens == sched["correction_service_tokens"],
            "correction token conservation drift")
    require(work["pwp_tokens"] + correction_tokens
            + sched["service_idle_cycles"]
            + sched["accumulator_pipeline_flush_cycles"]
            + sched["accumulator_commit_cycles"] == candidate,
            "W384 candidate conservation drift")

    valid_rows = 3000
    windows_per_record = 8
    records = 20
    fixed_commit_vectors = records * windows_per_record * 8 * 384
    projected_valid_commit_vectors = records * valid_rows * 8
    padded_commit_overhead = fixed_commit_vectors - projected_valid_commit_vectors
    require(projected_valid_commit_vectors == 480000
            and padded_commit_overhead == 11520,
            "partial-window commit accounting drift")

    m110_text = M110_RTL.read_text(encoding="utf-8")
    m111_text = M111_RTL.read_text(encoding="utf-8")
    m112_text = M112_RTL.read_text(encoding="utf-8")
    require("output logic [1:0]               service_load_beat" in m110_text,
            "M110 load-beat interface drift")
    require("output logic [BASE_W-1:0]        service_destination_row" in m110_text,
            "M110 destination interface drift")
    require("update_delta" not in m110_text and "descriptor_done" not in m110_text,
            "M110 bridge/completion gap unexpectedly changed")
    require("input  logic [VECTOR_BITS-1:0]       update_delta" in m111_text,
            "M111 delta interface drift")
    require("commit_pipe_last_q" in m111_text
            and "commit_issue_row_q == WIN_ROWS-1" in m111_text,
            "M111 fixed full-window commit drift")
    require("window_valid_rows" not in m111_text,
            "M111 partial-row interface unexpectedly changed")
    require("flatten_address" in m112_text
            and "DEPTH != 3072" in m112_text,
            "M112 flattened lane macro map drift")

    one_bubble_per_group = work["groups"]
    two_point_five_margin = math.floor(baseline / 2.5) - candidate
    require(one_bubble_per_group > two_point_five_margin,
            "one-bubble risk no longer crosses 2.5")

    options = {
        "integrated_replay_first": {
            "projected_cycle_savings": 0,
            "direct_performance_gain": 0.0,
            "evidence_gain": "closes token-to-delta/address/window/commit executable semantics",
            "risk_exposed": {
                "one_unhidden_payload_read_bubble_per_active_group_cycles":
                    one_bubble_per_group,
                "ratio_if_one_bubble_per_active_group":
                    ratio(baseline, candidate + one_bubble_per_group),
                "cycles_available_before_falling_below_2p5":
                    two_point_five_margin,
                "current_m112_fixed_full_commit_padding_cycles":
                    padded_commit_overhead,
                "ratio_with_only_fixed_full_commit_padding":
                    ratio(baseline, candidate + padded_commit_overhead),
            },
        },
        "w512_before_integration": {
            "projected_cycle_savings_vs_w384": candidate
                - w512["dual_timeline_recurrence"]["candidate_cycles"],
            "projected_ratio": w512["dual_timeline_recurrence"]
                ["same_clock_service_island_ratio"],
            "storage_increase_bytes": w512["storage_lower_bound"]
                ["combined_bytes_ceiling_before_control_ecc_macro_rounding"]
                - w384["storage_lower_bound"]
                ["combined_bytes_ceiling_before_control_ecc_macro_rounding"],
            "controller_geometry_vcs": False,
        },
        "w1024_before_integration": {
            "projected_cycle_savings_vs_w384": candidate
                - w1024["dual_timeline_recurrence"]["candidate_cycles"],
            "projected_ratio": w1024["dual_timeline_recurrence"]
                ["same_clock_service_island_ratio"],
            "storage_increase_bytes": w1024["storage_lower_bound"]
                ["combined_bytes_ceiling_before_control_ecc_macro_rounding"]
                - w384["storage_lower_bound"]
                ["combined_bytes_ceiling_before_control_ecc_macro_rounding"],
            "controller_geometry_vcs": False,
        },
        "w3000_before_integration": {
            "projected_cycle_savings_vs_w384": candidate
                - w3000["dual_timeline_recurrence"]["candidate_cycles"],
            "projected_ratio": w3000["dual_timeline_recurrence"]
                ["same_clock_service_island_ratio"],
            "storage_increase_bytes": w3000["storage_lower_bound"]
                ["combined_bytes_ceiling_before_control_ecc_macro_rounding"]
                - w384["storage_lower_bound"]
                ["combined_bytes_ceiling_before_control_ecc_macro_rounding"],
            "controller_geometry_vcs": False,
        },
        "correction_weight_load_3_to_1_upper_bound": {
            "projected_cycle_savings": 2 * work["groups"],
            "projected_ratio_if_zero_new_stalls":
                ratio(baseline, candidate - 2 * work["groups"]),
            "cost": "triple payload bandwidth or 768-bit path plus new SRAM arbitration",
        },
        "remove_all_group_loads_impossible_upper_bound": {
            "projected_cycle_savings": correction_load_tokens,
            "projected_ratio_if_semantically_possible":
                ratio(baseline, candidate - correction_load_tokens),
            "admitted": False,
        },
        "cross_vector_pwp_bitpacking_upper_bound": {
            "current_pwp_tokens": work["pwp_tokens"],
            "global_bit_ideal_tokens": ideal_cross_vector_tokens,
            "projected_cycle_savings": pwp_cross_pack_savings,
            "projected_ratio_if_zero_unpack_stalls":
                ratio(baseline, candidate - pwp_cross_pack_savings),
            "admitted": False,
            "warning": "global pooling ignores vector/address/descriptor boundaries and unpack backpressure",
        },
    }

    payload = {
        "schema": "m113_minimal_integrated_replay_arch_audit_v1",
        "status": "PASS_READ_ONLY_ARCH_AUDIT_INTEGRATE_BEFORE_MORE_PROJECTION_OPTIMIZATION",
        "identity": observed,
        "frozen_w384_projection": {
            "candidate_cycles": candidate,
            "fair_fixed8_baseline_cycles": baseline,
            "same_clock_precompacted_service_island_ratio":
                ratio(baseline, candidate),
            "headroom_to_two_x_cycles":
                sched["headroom_to_two_x_cycles"],
            "headroom_to_two_point_five_x_cycles": two_point_five_margin,
            "descriptors": sched["descriptors"],
            "global_accumulator_windows": records * windows_per_record,
            "pwp_service_tokens": work["pwp_tokens"],
            "pwp_updates": pwp_updates,
            "correction_load_tokens": correction_load_tokens,
            "correction_event_updates": work["events"],
            "total_accumulator_updates": update_count,
            "service_idle_cycles": sched["service_idle_cycles"],
            "commit_cycles_projected_valid_rows_only":
                sched["accumulator_commit_cycles"],
        },
        "interface_gap_evidence": {
            "m110_has_service_token_identity_and_direction": True,
            "m110_has_weight_or_pwp_payload": False,
            "m110_has_signed24_lane_delta": False,
            "m110_has_descriptor_done_for_nonempty_and_empty": False,
            "m111_requires_block_row_and_2304b_signed24_delta": True,
            "m111_has_global_base_or_context_on_commit": False,
            "m111_accepts_valid_row_count": False,
            "m111_m112_commit_fixed_8_by_384_vectors": True,
            "m112_flattens_block_row_to_96_lane_macros": True,
            "existing_m110_m112_directly_connectable": False,
        },
        "partial_window_accounting": {
            "records": records,
            "windows_per_record": windows_per_record,
            "last_window_valid_rows": 312,
            "projected_valid_commit_vectors": projected_valid_commit_vectors,
            "current_m112_fixed_commit_vectors": fixed_commit_vectors,
            "extra_padded_commit_vectors_if_reused_unchanged":
                padded_commit_overhead,
        },
        "option_comparison": options,
        "source_contract_admission": {
            "m110_controller_geometry_vcs": m110_contract["admission"]
                ["w384_controller_geometry_vcs"],
            "m111_standalone_numeric_vcs": m111_contract["admission"]
                ["full_lane_signed24_numeric_directed_miter"],
            "m112_lane_sliced_mapping_vcs": m112_contract["admission"]
                ["lane_sliced_address_mapping"],
            "m112_exact_heldout_integrated_replay": m112_contract["admission"]
                ["exact_heldout_integrated_replay"],
        },
        "docs_359_sha256_unchanged": sha256(DOC359),
        "production_files_modified": False,
    }
    require(sha256(DOC359) == EXPECTED_SHA[DOC359], "docs/359 drift")
    require(sha256(Path(__file__).resolve()) == start_sha,
            "auditor changed during run")
    require(not OUTPUT.exists(), "refusing output overwrite")
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M113 arch audit W384={:.12f} margin2p5={} bubble_group={} "
          "padded_commit={} output={}".format(
              ratio(baseline, candidate), two_point_five_margin,
              one_bubble_per_group, padded_commit_overhead, OUTPUT), flush=True)


if __name__ == "__main__":
    main()
