#!/usr/bin/env python3
"""Independent closure audit for the M108-r3 dual-timeline schedule."""

import hashlib
import json
import math
from pathlib import Path


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
OUTPUT = REVIEW / "m108_r3_dual_timeline_fused_schedule_independent_audit.json"

R3_ANALYZER = HW / "system_simulator/scripts/analyze_m108_r3_dual_timeline_fused_schedule.py"
R3_DIR = HW / "results/m108_r3_dual_timeline_fused_schedule_r1_20260824"
R3_RESULT = R3_DIR / "m108_r3_dual_timeline_fused_schedule.json"
R3_RUN = R3_DIR / "RUN_COMPLETE.txt"
R3_MANIFEST = R3_DIR / "manifest.sha256"
R3_CONTRACT = HW / "contracts/m108_r3_dual_timeline_fused_schedule_contract_r1_20260824.json"
R2_REVOCATION = HW / "contracts/m108_r2_cycle_schedule_revocation_r1_20260824.json"
R2_CONTRACT = HW / "contracts/m108_r2_rtl_edge_fused_schedule_contract_r1_20260824.json"
R2_REVIEW_DIR = HW / "reviews/m108_r2_rtl_edge_fused_schedule_independent_hammer_r1_20260824"
R2_AUDIT = R2_REVIEW_DIR / "m108_r2_rtl_edge_fused_schedule_independent_audit.json"
R2_REVIEW = R2_REVIEW_DIR / "m108_r2_rtl_edge_fused_schedule_independent_hammer_review.json"
R2_MANIFEST = R2_REVIEW_DIR / "manifest.sha256"
R2_VCS_RUN = R2_REVIEW_DIR / "vcs_m106_controller_serialization_r1/RUN_COMPLETE.txt"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED_SHA = {
    R3_ANALYZER: "39b210c97d31c3574999f33b0fec88f3a6c5d2ef94b646f763b5bc8c276d3f37",
    R3_RESULT: "d5a4d7c27a91a7735ed4481100d0db3640191357e4617e043378bb367a77dacc",
    R3_RUN: "75bf2446fea8deed69014288ebdfe7bc9f2ad22a43249b68a56ee0283c12e32c",
    R3_MANIFEST: "3a14e16d809982e20e8388c9b52d78eec45f5577e6d769e0a1fb9db321b66769",
    R3_CONTRACT: "dbbba8fa1de95f42891afa705cec7efc16b0dc026c29de8c52aa205dbcab06ff",
    R2_REVOCATION: "d2038bcc0a1f9c9d69c1dcf7ffeac3830f96a96ad754775fe708a83e2e586a05",
    R2_CONTRACT: "f2ce9d2ed5b8d2f6f019035f8f25b7ac7edd339a874d15eee748fbb165f0c0ac",
    R2_AUDIT: "7db3ba30936ae505d39ac8bd1134c8877ed3c68234850d94ccdc2d1e65e7cfc7",
    R2_REVIEW: "c6dd22c1ee36b3273fe016e612737ff4e012badf359d3ea456825eaa5b64b504",
    R2_MANIFEST: "8a94acebacf11fba3b0e6531736f4a5b9e8614c4a2d84b3bd3c3751fdf097396",
    R2_VCS_RUN: "a8d634c6805d8b7c2b1b11adaae6f618c958d3c79aa8888ccd95e95f9b8698ec",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

EXPECTED_CANDIDATE = 521264186
EXPECTED_BASELINE = 1114864228
EXPECTED_RATIO = 2.1387700477853278
EXPECTED_HEADROOM = 36167928
EXPECTED_DIGEST = "a011720a36d2c5dee37ddc1bfdd42aa3a22caf18a6ea70cb8b36e1e3f956858d"


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


def verify_manifest(path, base):
    checked = 0
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, name = raw.split(None, 1)
        target = Path(name.strip())
        if not target.is_absolute():
            target = Path(base) / target
        require(target.is_file(), "manifest target missing: " + str(target))
        require(sha256(target) == expected, "manifest mismatch: " + str(target))
        checked += 1
    return checked


def synthetic_dual_timeline(rows):
    bank_free = [0, 0]
    producer_end = 0
    controller_free = 0
    lane_end = 0
    trace = []
    reacquire = 0
    for index, row in enumerate(rows):
        bank = index & 1
        if index == 0 or producer_end > bank_free[bank]:
            fill_start = producer_end
        else:
            fill_start = bank_free[bank] + 1
            reacquire += 1
        fill_end = fill_start + row["events"] + 1
        producer_end = fill_end
        dispatch = max(fill_end, controller_free) + 1
        pwp_start = max(lane_end, fill_start)
        pwp_end = pwp_start + row["pwp"]
        correction_tokens = row["events"] + 3 * row["groups"]
        if correction_tokens:
            correction_start = max(pwp_end, dispatch)
            correction_end = correction_start + correction_tokens
            controller_free = correction_end
            lane_end = correction_end
            bank_free[bank] = correction_end
        else:
            correction_start = None
            controller_free = dispatch
            bank_free[bank] = dispatch
            lane_end = pwp_end
        pre_commit_lane = lane_end
        pre_commit_controller = controller_free
        if row.get("commit", 0) or row.get("flush", False):
            lane_end = max(lane_end, controller_free)
            if row.get("flush", False):
                lane_end += 1
            lane_end += row.get("commit", 0)
        trace.append({
            "index": index,
            "bank": bank,
            "fill_start": fill_start,
            "fill_end": fill_end,
            "dispatch_edge": dispatch,
            "pwp_start": pwp_start,
            "pwp_end": pwp_end,
            "correction_start": correction_start,
            "controller_free_after_descriptor": controller_free,
            "bank_free_after_descriptor": bank_free[bank],
            "lane_before_commit": pre_commit_lane,
            "controller_before_commit": pre_commit_controller,
            "lane_after_commit": lane_end,
        })
    return {
        "trace": trace,
        "producer_end": producer_end,
        "controller_free": controller_free,
        "lane_end": lane_end,
        "makespan": max(controller_free, lane_end),
        "bank_free": bank_free,
        "bank_reacquire_boundaries": reacquire,
    }


def run_attacks():
    prior_nonempty_then_empty = synthetic_dual_timeline([
        {"events": 1, "groups": 1, "pwp": 0},
        {"events": 0, "groups": 0, "pwp": 0},
    ])
    t = prior_nonempty_then_empty["trace"]
    require([row["dispatch_edge"] for row in t] == [3, 8],
            "prior-drain empty dispatch attack failed")
    require(t[1]["bank_free_after_descriptor"] == 8,
            "empty release did not wait prior drain")

    consecutive_empty = synthetic_dual_timeline([
        {"events": 0, "groups": 0, "pwp": 0},
        {"events": 0, "groups": 0, "pwp": 0},
        {"events": 0, "groups": 0, "pwp": 0},
    ])
    require([row["dispatch_edge"] for row in consecutive_empty["trace"]]
            == [2, 3, 5], "consecutive-empty controller attack failed")
    require(consecutive_empty["bank_reacquire_boundaries"] == 1,
            "consecutive-empty bank reacquire attack failed")

    pwp_hides_dispatch = synthetic_dual_timeline([
        {"events": 1, "groups": 1, "pwp": 0},
        {"events": 1, "groups": 1, "pwp": 4},
    ])
    t = pwp_hides_dispatch["trace"]
    require(t[1]["dispatch_edge"] == 8 and t[1]["pwp_end"] == 11
            and t[1]["correction_start"] == 11,
            "PWP dispatch-overlap attack failed")

    commit_overlap = synthetic_dual_timeline([
        {"events": 1, "groups": 1, "pwp": 0, "flush": True, "commit": 3},
        {"events": 0, "groups": 0, "pwp": 0},
        {"events": 0, "groups": 0, "pwp": 0},
    ])
    t = commit_overlap["trace"]
    require(t[0]["lane_before_commit"] == 7 and t[0]["lane_after_commit"] == 11,
            "commit duration attack failed")
    require(t[1]["dispatch_edge"] == 8 and t[1]["lane_after_commit"] == 11,
            "empty dispatch did not overlap prior commit")
    require(t[2]["dispatch_edge"] == 10 and t[2]["lane_after_commit"] == 11,
            "second empty dispatch did not overlap prior commit")

    final_empty_commit_gate = synthetic_dual_timeline([
        {"events": 1, "groups": 1, "pwp": 0},
        {"events": 0, "groups": 0, "pwp": 0, "flush": True, "commit": 3},
    ])
    t = final_empty_commit_gate["trace"]
    require(t[1]["lane_before_commit"] == 7
            and t[1]["controller_before_commit"] == 8
            and t[1]["lane_after_commit"] == 12,
            "final empty commit controller gate attack failed")

    return {
        "prior_nonempty_then_empty": prior_nonempty_then_empty,
        "three_consecutive_empty": consecutive_empty,
        "pwp_hides_dispatch": pwp_hides_dispatch,
        "empty_dispatches_overlap_prior_commit": commit_overlap,
        "final_empty_commit_waits_controller": final_empty_commit_gate,
        "all_passed": True,
    }


def main():
    start_sha = sha256(Path(__file__).resolve())
    observed = {}
    for path, expected in EXPECTED_SHA.items():
        actual = sha256(path)
        require(actual == expected, "identity mismatch {} {}".format(path, actual))
        observed[str(path.relative_to(HW))] = actual
    require(verify_manifest(R3_MANIFEST, R3_DIR) == 5,
            "r3 result manifest extent drift")
    require(verify_manifest(R2_MANIFEST, HW) == 18,
            "r2 independent review manifest extent drift")

    result = strict_json(R3_RESULT)
    contract = strict_json(R3_CONTRACT)
    revocation = strict_json(R2_REVOCATION)
    prior_audit = strict_json(R2_AUDIT)
    prior_review = strict_json(R2_REVIEW)
    schedule = result["dual_timeline_schedule"]
    independent = prior_audit["independent_dual_timeline_recurrence"]

    compare_fields = (
        "descriptors", "ordered_descriptor_sha256", "descriptor_fill_cycles",
        "producer_bank_stall_cycles", "controller_dispatch_edges",
        "bank_reacquire_boundaries", "controller_final_free_cycle",
        "controller_serialization_delay_sum_vs_fill_only_dispatch",
        "empty_release_delay_sum_vs_fill_only_dispatch",
        "dispatch_hidden_by_pwp_or_prior_lane_descriptors",
        "zero_pwp_descriptors",
        "exposed_post_pwp_fill_or_dispatch_wait_cycles", "service_idle_cycles",
        "pwp_service_tokens", "correction_service_tokens",
        "accumulator_pipeline_flush_cycles", "accumulator_commit_cycles",
        "candidate_cycles", "fair_fixed8_baseline_cycles",
        "headroom_to_two_x_cycles",
    )
    for field in compare_fields:
        require(schedule[field] == independent[field],
                "r3/prior-independent mismatch: " + field)
    require(math.isclose(schedule["same_clock_service_island_ratio"],
                         independent["same_clock_service_island_ratio"],
                         rel_tol=0.0, abs_tol=1e-15), "r3 ratio mismatch")

    work = result["work_conservation"]
    require(work == {
        "active_groups": 35140002,
        "events": 188148490,
        "negative_events": 17557357,
        "positive_events": 170591133,
        "pwp_service_tokens": 226222255,
        "pwp_updates": 58969374,
        "pwp_uses_by_width": {"8": 11164284, "9": 32360036,
                              "10": 13936011, "11": 1509043},
        "source_coefficient_checks": 3317760000,
    }, "r3 work ledger drift")
    correction = work["events"] + 3 * work["active_groups"]
    common_tail = (schedule["accumulator_pipeline_flush_cycles"]
                   + schedule["accumulator_commit_cycles"])
    candidate = (work["pwp_service_tokens"] + correction
                 + schedule["service_idle_cycles"] + common_tail)
    baseline = prior_audit["raw_reconstruction"]["fixed8_baseline_tokens"] + common_tail
    ratio = baseline / float(candidate)
    headroom = baseline // 2 - candidate
    require(candidate == EXPECTED_CANDIDATE == schedule["candidate_cycles"],
            "independent candidate arithmetic mismatch")
    require(baseline == EXPECTED_BASELINE == schedule["fair_fixed8_baseline_cycles"],
            "independent baseline arithmetic mismatch")
    require(math.isclose(ratio, EXPECTED_RATIO, rel_tol=0.0, abs_tol=1e-15),
            "independent ratio arithmetic mismatch")
    require(headroom == EXPECTED_HEADROOM == schedule["headroom_to_two_x_cycles"],
            "independent headroom arithmetic mismatch")
    require(schedule["ordered_descriptor_sha256"] == EXPECTED_DIGEST,
            "ordered descriptor digest drift")

    source = R3_ANALYZER.read_text(encoding="utf-8")
    require("dispatch_ready = max(fill_end, controller_free) + 1" in source,
            "corrected dispatch expression missing")
    require(source.count("controller_free = correction_end") == 1
            and source.count("controller_free = dispatch_ready") == 1,
            "controller release branches missing")
    require("window_ready = max(service_end, controller_free)" in source,
            "commit controller gate missing")

    require(revocation["status"] == "REVOKED_M108_R2_SCHEDULED_CYCLE_ADMISSION",
            "r2 revocation status drift")
    require(revocation["revoked_contract_sha256"] == EXPECTED_SHA[R2_CONTRACT]
            and revocation["m108_r2_scheduled_ratio_admitted"] is False,
            "r2 revocation target/admission drift")
    require(revocation["superseding_contract"] == R3_CONTRACT.name,
            "r2 superseding contract drift")
    require(contract["frozen_result"]["candidate_cycles"] == candidate
            and contract["frozen_result"]["ordered_descriptor_sha256"]
            == EXPECTED_DIGEST, "r3 contract result drift")
    require(contract["admission"]["complete_integrated_commercial_vcs_miter"]
            is False and contract["admission"]["physical_speedup"] is False
            and contract["admission"]["system_speedup"] is False
            and contract["admission"]["headline"] is False,
            "r3 prohibited admission drift")
    require(prior_review["m107_p0_closure"]["verdict"] == "NOT_CLOSED",
            "r2 P0 source review drift")
    require(R2_VCS_RUN.read_text(encoding="utf-8").splitlines()[0]
            == "status=PASS_M108_R2_INDEPENDENT_M106_CONTROLLER_SERIALIZATION_VCS",
            "targeted frozen M106 VCS evidence drift")

    attacks = run_attacks()
    payload = {
        "schema": "m108_r3_dual_timeline_fused_schedule_independent_audit_v1",
        "status": "PASS_R3_PRIOR_CONTROLLER_P0_CLOSED_SOFTWARE_BOUND_PORT_CUTS_REMAIN",
        "identity": observed,
        "manifest_entries_verified": {"r3_result": 5, "r2_review": 18},
        "independent_reference": {
            "source": "sealed M108-r2 independent raw reconstruction and dual-timeline correction",
            "r3_exact_match_to_prior_independent_schedule": True,
            "compared_schedule_fields": list(compare_fields),
            "candidate_cycles": candidate,
            "fair_fixed8_baseline_cycles": baseline,
            "same_clock_precompacted_service_island_ratio": ratio,
            "headroom_to_two_x_cycles": headroom,
            "ordered_descriptor_sha256": schedule["ordered_descriptor_sha256"],
        },
        "arithmetic_conservation": {
            "pwp_service_tokens": work["pwp_service_tokens"],
            "correction_service_tokens": correction,
            "service_idle_cycles": schedule["service_idle_cycles"],
            "common_flush_commit_cycles": common_tail,
            "candidate_sum": candidate,
            "raw_fixed8_baseline_tokens":
                prior_audit["raw_reconstruction"]["fixed8_baseline_tokens"],
            "baseline_plus_common_tail": baseline,
        },
        "p0_closure": {
            "prior_drain_or_empty_release_dependency": True,
            "dispatch_expression": "max(fill_end, controller_free)+1",
            "nonempty_release": "controller_free=correction_end",
            "empty_release": "controller_free=dispatch_edge",
            "window_commit_gate": "max(shared_lane_end, controller_free)",
            "r2_scheduled_admission_revoked": True,
            "r3_matches_independent_corrected_cycles": True,
            "targeted_frozen_m106_vcs_evidence_retained": True,
            "verdict": "CLOSED_FOR_SOFTWARE_DUAL_TIMELINE_BOUND",
        },
        "synthetic_attacks": attacks,
        "remaining_port_cuts": {
            "complete_integrated_commercial_vcs_miter": False,
            "accumulator_2304b_rmw_macro_schedule": False,
            "full_lane_signed24_numeric_miter": False,
            "shared_weight_sram_address_port_latency_arbitration": False,
            "precompaction_delivery_and_queue_schedule": False,
            "macro_inclusive_ppa": False,
            "equal_area_or_system_scope": False,
        },
        "admission": {
            "corrected_dual_timeline_software_bound": True,
            "scheduled_precompacted_module_cycle_ratio": True,
            "actual_combined_controller_cycle_miter": False,
            "physical_speedup": False,
            "equal_area": False,
            "macro_inclusive_ppa": False,
            "system_speedup": False,
            "headline": False,
        },
        "docs_359_sha256_unchanged": sha256(DOC359),
        "producer_analyzer_executed": False,
        "production_files_modified": False,
    }
    require(sha256(DOC359) == EXPECTED_SHA[DOC359], "docs/359 drift")
    require(sha256(Path(__file__).resolve()) == start_sha,
            "independent auditor changed during run")
    require(not OUTPUT.exists(), "refusing independent output overwrite")
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M108-r3 independent P0 closure candidate={} ratio={:.12f} "
          "headroom={} attacks=5 output={}".format(
              candidate, ratio, headroom, OUTPUT), flush=True)


if __name__ == "__main__":
    main()
