#!/usr/bin/env python3
"""Independent identity, VCS, protocol-envelope, and DC-admission audit for M104."""

import hashlib
import json
import math
from pathlib import Path


REVIEW = Path(__file__).resolve().parent
HW = REVIEW.parents[1]
SEALED = HW / "dc_handoff/runs/m104_held_weight_correction_broadcaster_vcs_r1_sealed_20260824"
INDEPENDENT = REVIEW / "vcs_adversarial_run_r3"
CONTRACT = HW / "contracts/m104_held_weight_correction_broadcaster_vcs_contract_r1_20260824.json"
M103 = HW / (
    "reviews/m103_correction_service_reuse_preflight_independent_hammer_r1_20260824/"
    "m103_correction_reuse_preflight_audit.json"
)
RESULT_DIR = HW / "results/m104_held_weight_correction_broadcaster_vcs_token_envelope_r1_20260824"
RESULT = RESULT_DIR / "m104_held_weight_correction_broadcaster.json"
ANALYZER = HW / "system_simulator/scripts/analyze_m104_held_weight_correction_broadcaster.py"
RTL = HW / "rtl_m104/m104_held_weight_correction_broadcaster.sv"
SVA = HW / "verif_m104/m104_held_weight_correction_broadcaster_assertions.sv"
TB = HW / "tb_m104/tb_m104_held_weight_correction_broadcaster.sv"
VCS_FILELIST = HW / "dc_handoff/filelists/date_m104_held_weight_correction_broadcaster_directed_vcs.f"
DC_FILELIST = HW / "dc_handoff/filelists/date_m104_held_weight_correction_broadcaster_logic_only_dc.f"

EXPECTED_SHA = {
    RTL: "37f86144563d45ea96f594847828a00c7d872602419d81a070738f12b4417f6a",
    SVA: "ad63c0317b64b5e53aecd037d401669c42f5b4b40409563ed216e4eb776e2f98",
    TB: "7ed7fcf389c49dcc152a002416f6af9198fdb7c770373b6d711c828984529916",
    VCS_FILELIST: "a04e09b3029ee030f53e2cac6146ae13ed6c22bd96e57d86cbfae0adafbe6cbe",
    DC_FILELIST: "4507f6af3f41cae8c1c26f6779f3c33803d30e03dcbaeef36348ee905f99fd36",
    CONTRACT: "bbd086a36719f3682216d39450dfc86db46c9373fc508f65657cfac2277dbdd5",
    M103: "935119fab809e15f49089926550f89b3c84c2b13c0be58c96b0ea8709ed683fe",
    RESULT: "8b00f57d368afe3c80633b0bfdd0770b9200090085204d0ab47c39c36aaaf205",
    ANALYZER: "125e7b4858e7c83207f576172165078a178e4ff22fbf4f2d60e8591137c95a6e",
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
    source_start_sha = sha256(Path(__file__).resolve())
    observed_sha = {}
    for path, expected in EXPECTED_SHA.items():
        observed = sha256(path)
        require(observed == expected, "identity mismatch: {} {}".format(path, observed))
        observed_sha[str(path.relative_to(HW))] = observed

    sealed_input_count = verify_manifest(SEALED / "input_sha256.txt", HW)
    sealed_output_count = verify_manifest(SEALED / "output_sha256.txt", HW)
    independent_input_count = verify_manifest(INDEPENDENT / "input_sha256.txt", HW)
    independent_output_count = verify_manifest(INDEPENDENT / "output_sha256.txt", HW)
    result_manifest_count = verify_manifest(RESULT_DIR / "manifest.sha256", RESULT_DIR)
    require(sealed_input_count == 9 and sealed_output_count == 4,
            "sealed VCS manifest count drift")
    require(independent_input_count == 3 and independent_output_count == 4,
            "independent VCS manifest count drift")
    require(result_manifest_count == 2, "M104 result manifest count drift")

    contract = strict_json(CONTRACT)
    m103 = strict_json(M103)
    result = strict_json(RESULT)
    sealed_sim = (SEALED / "sim.raw.log").read_text(encoding="utf-8")
    sealed_assert = (SEALED / "assert.report").read_text(encoding="utf-8")
    independent_sim = (INDEPENDENT / "sim.raw.log").read_text(encoding="utf-8")
    independent_assert = (INDEPENDENT / "assert.report").read_text(encoding="utf-8")

    require((SEALED / "compile.rc").read_text().strip() == "0", "sealed compile rc")
    require((SEALED / "sim.rc").read_text().strip() == "0", "sealed sim rc")
    require(contract["directed_vcs"]["expected_pass_line"] in sealed_sim,
            "sealed directed PASS missing")
    for cover, matches in contract["directed_vcs"]["required_cover_matches"].items():
        require(", {} match".format(matches) in next(
            line for line in sealed_assert.splitlines() if cover in line),
            "sealed cover mismatch " + cover)

    require((INDEPENDENT / "compile.rc").read_text().strip() == "0",
            "independent compile rc")
    require((INDEPENDENT / "sim.rc").read_text().strip() == "0",
            "independent sim rc")
    independent_pass = (
        "PASS M104 independent adversarial VCS signed_codes=256 lanes=96 signs=2 "
        "ready_release_fault=1 sticky_cycles=3 reset_recovery=1 ii1_turnovers=4 "
        "load_gap=1 last_wait=1"
    )
    require(independent_pass in independent_sim, "independent VCS PASS missing")
    independent_covers = {
        "cp_illegal_plus_ready_release": 1,
        "cp_legal_stalled_last_then_release": 1,
        "cp_reset_recovery": 2,
    }
    for cover, matches in independent_covers.items():
        require(", {} match".format(matches) in next(
            line for line in independent_assert.splitlines() if cover in line),
            "independent cover mismatch " + cover)

    source = RTL.read_text(encoding="utf-8")
    require("request_collision = load_valid && event_valid;" in source,
            "load/event collision policy drift")
    require("event_identity_valid = held_valid_q" in source,
            "event requires already held key drift")
    require("if (event_last_for_key)" in source and
            "held_valid_q <= 1'b0;" in source,
            "last-for-key release drift")
    require(DC_FILELIST.read_text(encoding="utf-8").splitlines() == [
        "rtl_m104/m104_held_weight_correction_broadcaster.sv"
    ], "production-only DC filelist is not RTL-only")

    grouping = m103["order_independent_grouping"]["weight_groups"]
    events = grouping["events"]
    groups = grouping["groups"]
    envelope = result["conditional_service_token_envelope"]
    baseline = envelope["fixed8_baseline_tokens"]
    pwp = envelope["existing_pwp_tokens"]
    frozen_m103_correction = events + 2 * groups
    frozen_m103_combined = pwp + frozen_m103_correction
    frozen_m103_ratio = baseline / float(frozen_m103_combined)
    literal_result_formula = (
        events * envelope["destination_tokens_per_event"]
        + groups * envelope["weight_load_tokens_per_group"]
    )
    current_rtl_serial_correction = events + 3 * groups
    current_rtl_serial_combined = pwp + current_rtl_serial_correction
    current_rtl_serial_ratio = baseline / float(current_rtl_serial_combined)

    require(events == 188148490 and groups == 1105920, "M103 population drift")
    require(frozen_m103_correction == 190360330, "M103 frozen correction drift")
    require(frozen_m103_combined == 416582585, "M103 frozen combined drift")
    require(math.isclose(frozen_m103_ratio, 2.6750597075487446,
                         rel_tol=0.0, abs_tol=1e-15), "M103 ratio drift")
    require(literal_result_formula == current_rtl_serial_correction == 191466250,
            "literal M104 serialized formula drift")
    require(envelope["correction_tokens"] == frozen_m103_correction,
            "published result no longer reflects frozen M103 model")
    require(literal_result_formula - envelope["correction_tokens"] == groups,
            "published arithmetic contradiction delta drift")

    payload = {
        "schema": "m104_held_weight_correction_broadcaster_independent_hammer_audit_v1",
        "status": "FUNCTIONAL_GO_LOGIC_ONLY_DC_GO_TOKEN_LEDGER_P0_SCHEDULE_CLAIMS_NO_GO",
        "identity": {
            "expected_sha_verified": observed_sha,
            "sealed_input_manifest_entries": sealed_input_count,
            "sealed_output_manifest_entries": sealed_output_count,
            "independent_input_manifest_entries": independent_input_count,
            "independent_output_manifest_entries": independent_output_count,
            "m104_result_manifest_entries": result_manifest_count,
        },
        "commercial_vcs_sva": {
            "sealed_directed_pass": True,
            "sealed_cover_counts": contract["directed_vcs"]["required_cover_matches"],
            "independent_adversarial_pass": True,
            "independent_cover_counts": independent_covers,
            "signed_int8_codes_exhausted": 256,
            "lanes_per_vector": 96,
            "positive_and_negative_checked": True,
            "same_cycle_invalid_plus_ready_release_quarantine": True,
            "sticky_fault_reset_only": True,
            "ready_release_turnover": True,
            "unaccepted_last_waits_then_accepted_last_releases": True,
            "three_load_beats_with_idle_gaps": True,
            "counterexample_under_directed_contract": False,
        },
        "conditional_token_models_not_cycles": {
            "events": events,
            "phase_weight_groups": groups,
            "pwp_tokens_unchanged": pwp,
            "baseline_tokens": baseline,
            "frozen_m103_perfect_batching": {
                "formula": "E + 2*G",
                "implicit_extra_assumption": (
                    "one destination descriptor is fused with a load beat or an equivalent "
                    "one-token-per-key overlap is free"
                ),
                "correction_tokens": frozen_m103_correction,
                "combined_tokens": frozen_m103_combined,
                "baseline_ratio": frozen_m103_ratio,
                "ordered_or_scheduled": False,
            },
            "current_m104_mutually_exclusive_serial_interface": {
                "formula": "E + 3*G",
                "reason": (
                    "three accepted load beats are required per key, every event requires a "
                    "separate accepted descriptor, and simultaneous load/event is illegal"
                ),
                "correction_tokens": current_rtl_serial_correction,
                "combined_tokens": current_rtl_serial_combined,
                "baseline_ratio": current_rtl_serial_ratio,
                "ordered_or_scheduled": False,
            },
            "published_result_literal_field_formula": literal_result_formula,
            "published_result_declared_correction_tokens": envelope["correction_tokens"],
            "published_result_internal_arithmetic_consistent": False,
            "published_undercharge_tokens": groups,
        },
        "dc_admission": {
            "production_only_logic_dc_may_start_before_ordered_trace": True,
            "reason": (
                "module function and exact RTL identity are closed enough to characterize "
                "logic area/timing; schedule feasibility is not needed to synthesize this port cut"
            ),
            "rtl_only_filelist_exact_sha": sha256(DC_FILELIST),
            "required_run_seals": [
                "exact RTL and filelist SHA",
                "top m104_held_weight_correction_broadcaster and TAG_W=32",
                "SYNTHESIS define",
                "common library/corners and exact SDC/clock",
                "no macros and logic-only/pre-macro label",
                "DC logs, netlist, SDC, reports, and output SHA manifest",
            ],
            "ordered_transpose_or_acc_bank_trace_required_before_logic_only_dc": False,
            "ordered_transpose_or_acc_bank_trace_required_before_scheduled_performance": True,
            "dc_result_can_admit_physical_or_system_speedup": False,
        },
        "admission": {
            "functional_module": True,
            "production_only_logic_dc": True,
            "frozen_m103_perfect_batch_token_model": True,
            "published_m104_token_result_as_literal_current_rtl_model": False,
            "ordered_schedule": False,
            "actual_record_replay": False,
            "accumulator_equivalence": False,
            "scheduled_cycles": False,
            "physical_speedup": False,
            "equal_area": False,
            "macro_inclusive_ppa": False,
            "system_speedup": False,
            "headline": False,
        },
        "producer_analyzer_executed": False,
        "production_files_modified": False,
    }
    require(sha256(Path(__file__).resolve()) == source_start_sha,
            "independent audit changed during execution")
    output = REVIEW / "m104_independent_hammer_audit.json"
    require(not output.exists(), "refusing to overwrite independent audit output")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS M104 independent hammer audit functional=true dc=true "
          "published_token_arithmetic=false scheduled=false physical=false")


if __name__ == "__main__":
    main()
