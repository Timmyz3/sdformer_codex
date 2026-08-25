#!/usr/bin/env python3
"""Independent cross-source audit of the corrected M104 r2 literal token ledger."""

import hashlib
import json
import math
from pathlib import Path


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
CONTRACT = HW / "contracts/m104_r2_literal_serial_token_correction_contract_r1_20260824.json"
ANALYZER = HW / "system_simulator/scripts/analyze_m104_r2_literal_serial_token_correction.py"
RESULT_DIR = HW / "results/m104_r2_literal_serial_token_correction_r1_20260824"
RESULT = RESULT_DIR / "m104_r2_literal_serial_token_correction.json"
RUN_COMPLETE = RESULT_DIR / "RUN_COMPLETE.txt"
RESULT_MANIFEST = RESULT_DIR / "manifest.sha256"
M104_R1_REVIEW_DIR = HW / "reviews/m104_held_weight_correction_broadcaster_independent_hammer_r1_20260824"
M104_R1_REVIEW = M104_R1_REVIEW_DIR / "m104_held_weight_correction_broadcaster_independent_hammer_review.json"
M104_R1_AUDIT = M104_R1_REVIEW_DIR / "m104_independent_hammer_audit.json"
M104_R1_MANIFEST = M104_R1_REVIEW_DIR / "manifest.sha256"
M103 = HW / (
    "reviews/m103_correction_service_reuse_preflight_independent_hammer_r1_20260824/"
    "m103_correction_reuse_preflight_audit.json"
)
M102 = HW / (
    "results/m102_r2_fail_closed_matched_vector_service_islands_vcs_cycle_ledger_r1_20260824/"
    "m102_r2_fail_closed_matched_vector_service_islands.json"
)
RTL = HW / "rtl_m104/m104_held_weight_correction_broadcaster.sv"
DC_FILELIST = HW / "dc_handoff/filelists/date_m104_held_weight_correction_broadcaster_logic_only_dc.f"

EXPECTED_SHA = {
    CONTRACT: "b88ec871b84342a39257497c4803db240f6898b0d5f748bb31d51966deb836c8",
    ANALYZER: "01736afedc74b4f77182931769966ef1657577cedb4916e4d7827a7f593e54d0",
    RESULT: "2c59c7c8836a5f7bf802f6b5eff1ccb8e2d1e3fecc074e307458cd8c08d3538e",
    RUN_COMPLETE: "5ef52d0370ec2f558c34be2b6c2cde5aa390d5efcabf025bc666937fcc031ec9",
    RESULT_MANIFEST: "44ba839026fba21fdd1ab06bd27e31fd39d65ca7b0d01f8f3c51406e3ba73fe3",
    M104_R1_REVIEW: "22ce5342980f53429ab4a3bf1dff8f21df0f874730910556c196e58354e10860",
    M104_R1_AUDIT: "afdcbf92cdbd2514b4afe5f0b6454ee5eb404a269e8f708e6bc540d9ab8bbe3e",
    M104_R1_MANIFEST: "080fdbf1722ccc3271fc7e056b11298b2853d0bf64a8b533af854996d3c162a2",
    M103: "935119fab809e15f49089926550f89b3c84c2b13c0be58c96b0ea8709ed683fe",
    M102: "a5d465b7d3361ed2ff176b4230d9051c29137aee86211cec9c3eb9ee8131aad5",
    RTL: "37f86144563d45ea96f594847828a00c7d872602419d81a070738f12b4417f6a",
    DC_FILELIST: "4507f6af3f41cae8c1c26f6779f3c33803d30e03dcbaeef36348ee905f99fd36",
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
        output = {}
        for key, value in pairs:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject,
    )


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


def main():
    start_sha = sha256(Path(__file__).resolve())
    observed = {}
    for path, expected in EXPECTED_SHA.items():
        actual = sha256(path)
        require(actual == expected, "identity mismatch {} {}".format(path, actual))
        observed[str(path.relative_to(HW))] = actual
    require(verify_manifest(RESULT_MANIFEST, HW) == 4,
            "M104 r2 result manifest count drift")
    require(verify_manifest(M104_R1_MANIFEST, M104_R1_REVIEW_DIR) == 12,
            "M104 r1 review manifest count drift")

    contract = strict_json(CONTRACT)
    result = strict_json(RESULT)
    r1_review = strict_json(M104_R1_REVIEW)
    r1_audit = strict_json(M104_R1_AUDIT)
    m103 = strict_json(M103)
    m102 = strict_json(M102)
    run_lines = dict(
        line.split("=", 1)
        for line in RUN_COMPLETE.read_text(encoding="utf-8").splitlines()
        if "=" in line
    )

    events = m103["order_independent_grouping"]["weight_groups"]["events"]
    groups = m103["order_independent_grouping"]["weight_groups"]["groups"]
    pwp = m102["analytical_service_ledger"]["candidate_pwp_service_cycles"]
    baseline = m102["analytical_service_ledger"]["baseline_service_cycles"]
    correction = events + 3 * groups
    combined = correction + pwp
    ratio = baseline / float(combined)
    fused_correction = events + 2 * groups
    fused_combined = fused_correction + pwp
    fused_ratio = baseline / float(fused_combined)

    require(events == 188148490, "E drift")
    require(groups == 1105920, "G drift")
    require(pwp == 226222255, "PWP drift")
    require(baseline == 1114383288, "baseline drift")
    require(correction == 191466250, "E+3G drift")
    require(combined == 417688505, "combined drift")
    require(math.isclose(ratio, 2.6679769126038075,
                         rel_tol=0.0, abs_tol=1e-15), "ratio drift")
    require(fused_correction == 190360330, "E+2G target drift")
    require(fused_combined == 416582585, "fused combined drift")
    require(math.isclose(fused_ratio, 2.6750597075487446,
                         rel_tol=0.0, abs_tol=1e-15), "fused ratio drift")

    model = result["literal_current_rtl_model"]
    require(model == contract["literal_current_rtl_model"],
            "result literal model differs from contract")
    require(model["events_E"] == events and model["groups_G"] == groups,
            "result E/G differs from M103")
    require(model["existing_pwp_tokens"] == pwp and
            model["fixed8_baseline_tokens"] == baseline,
            "result denominator differs from M102")
    require(model["event_tokens_per_event"] == 1 and
            model["load_tokens_per_group"] == 3 and
            model["formula"] == "E+3G", "literal protocol terms drift")
    require(model["correction_tokens"] == correction and
            model["combined_tokens"] == combined and
            math.isclose(model["conditional_same_clock_token_ratio"], ratio,
                         rel_tol=0.0, abs_tol=1e-15),
            "result literal arithmetic drift")

    target = result["fused_or_overlapped_design_target"]
    require(target == contract["fused_or_overlapped_design_target"],
            "fused target differs from contract")
    require(target["implemented"] is False and target["formula"] == "E+2G",
            "r1 target not downgraded")
    require(target["implicit_free_overlap_tokens_per_group"] == 1 and
            target["combined_tokens"] == fused_combined and
            math.isclose(target["conditional_same_clock_token_ratio"], fused_ratio,
                         rel_tol=0.0, abs_tol=1e-15),
            "r1 target arithmetic drift")
    require(result["correction"]["r1_undercharge_tokens"] == groups,
            "r1 undercharge drift")
    require(result["correction"][
        "r1_ratio_relabelled_as_unimplemented_fused_or_overlapped_target"] is True,
        "r1 ratio downgrade missing")

    require(r1_review["severity_counts"]["P0"] == 1,
            "trigger review P0 drift")
    require(r1_audit["conditional_token_models_not_cycles"]
            ["published_undercharge_tokens"] == groups,
            "trigger audit undercharge drift")
    require(r1_audit["conditional_token_models_not_cycles"]
            ["current_m104_mutually_exclusive_serial_interface"]
            ["correction_tokens"] == correction,
            "trigger audit E+3G drift")

    rtl = RTL.read_text(encoding="utf-8")
    for required in (
        "request_collision = load_valid && event_valid;",
        "event_identity_valid = held_valid_q",
        "assign load_accept = load_valid && load_ready;",
        "assign event_accept = event_valid && event_ready;",
        "if (event_last_for_key)",
    ):
        require(required in rtl, "RTL protocol evidence missing: " + required)
    require("load_ready = !protocol_error && !event_valid" in rtl,
            "load/event mutual exclusion drift")
    require("event_ready = !protocol_error && !load_valid" in rtl,
            "event/load mutual exclusion drift")

    boundary = result["claim_boundary"]
    require(boundary["perfect_phase_key_batching"] is True,
            "perfect batching condition missing")
    for field in (
        "ordered_bounded_schedule", "scheduled_cycles", "physical_speedup",
        "equal_area", "macro_inclusive_ppa", "paper_ppa_ready",
        "system_speedup", "headline",
    ):
        require(boundary[field] is False, "claim boundary promoted: " + field)
    require(run_lines["formula"] == "E+3G" and
            int(run_lines["correction_tokens"]) == correction and
            int(run_lines["combined_tokens"]) == combined and
            run_lines["r1_ratio_is_unimplemented_fused_or_overlapped_target"] == "true",
            "RUN_COMPLETE correction drift")
    for field in ("scheduled_cycles", "physical_speedup", "system_speedup", "headline"):
        require(run_lines[field] == "false", "RUN_COMPLETE promoted: " + field)

    payload = {
        "schema": "m104_r2_literal_serial_token_correction_independent_audit_v1",
        "status": "PASS_R2_LITERAL_E_PLUS_3G_CORRECTION_R1_TARGET_DOWNGRADED",
        "identity": observed,
        "manifest_checks": {
            "m104_r2_result_entries": 4,
            "m104_r1_review_entries": 12,
        },
        "independent_cross_source_recalculation": {
            "events_E_from_m103": events,
            "groups_G_from_m103": groups,
            "pwp_tokens_from_m102": pwp,
            "baseline_tokens_from_m102": baseline,
            "literal_formula": "E+3G",
            "literal_correction_tokens": correction,
            "literal_combined_tokens": combined,
            "literal_conditional_token_ratio": ratio,
            "r1_target_formula": "E+2G",
            "r1_target_correction_tokens": fused_correction,
            "r1_target_combined_tokens": fused_combined,
            "r1_target_conditional_token_ratio": fused_ratio,
            "r1_undercharge_tokens": groups,
        },
        "rtl_protocol_cross_check": {
            "three_load_accepts_per_key": True,
            "one_event_accept_per_destination": True,
            "load_and_event_mutually_exclusive": True,
            "event_requires_already_held_key": True,
            "literal_serial_model_matches_current_rtl": True,
            "fused_or_overlapped_target_implemented": False,
        },
        "analyzer_robustness_notes": {
            "current_output_correct": True,
            "event_tokens_per_event_field_used_in_arithmetic": False,
            "reason": "line 74 adds events directly instead of multiplying the frozen event_tokens_per_event field; the field is one today",
            "fused_target_arithmetic_asserted_by_producer": False,
            "impact": "no numerical impact in r2; strengthen future fail-closed drift detection",
        },
        "admission": {
            "r2_literal_conditional_token_ledger": True,
            "r1_fused_or_overlapped_design_target_only": True,
            "r1_target_as_implemented_rtl": False,
            "ordered_bounded_schedule": False,
            "scheduled_cycles": False,
            "physical_speedup": False,
            "equal_area": False,
            "macro_inclusive_ppa": False,
            "paper_ppa_ready": False,
            "system_speedup": False,
            "headline": False,
        },
        "production_analyzer_executed": False,
        "production_files_modified": False,
    }
    require(sha256(Path(__file__).resolve()) == start_sha,
            "independent audit changed during run")
    output = REVIEW / "m104_r2_literal_serial_token_correction_independent_audit.json"
    require(not output.exists(), "refusing independent audit overwrite")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M104 r2 independent literal correction ratio={:.12f} "
          "scheduled=false physical=false".format(ratio))


if __name__ == "__main__":
    main()
