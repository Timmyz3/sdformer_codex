#!/usr/bin/env python3
"""Machine-readable exact identity, replay, and independence audit for M86-R3."""

import argparse
import hashlib
import json
import re
from pathlib import Path


EXPECTED_SHA = {
    "rtl_m82/zero_bubble_elastic_pwp_stream.sv":
        "2e8842234917355ee082968487229e83789e1a2f212296168d3a972f83631e1f",
    "rtl_m85/guarded_wordpacked_pwp_stream.sv":
        "ec2680f2fc97500133f3333e063fc268602ad793324a2cf6b8dbc1eb4b5207b0",
    "rtl_m86/sync_banked_guarded_pwp_frontend.sv":
        "edb06b7f4e891d4b00c8b49ace547efdf8daf84dc19716c710a6a343dc97f781",
    "rtl_m86_r3/phase_fsm_sync_banked_guarded_pwp_frontend.sv":
        "bd3d9ea0e4e2a2a98c0403442b9ff589af5f818756528249eec01d3c16333986",
    "verif_m86_r3/phase_fsm_sync_banked_guarded_pwp_frontend_assertions.sv":
        "e4befbfeff9c9e2b30c02f0bd2f48fd260848e4ca0c4c23fa8316f968f4b78c1",
    "tb_m86_r3/tb_phase_fsm_sync_banked_guarded_pwp_frontend.sv":
        "6b8314dbca77f5ff2f8b43b87451b13419f94284fde8e4ad21b6ecfedb593e8b",
    "tb_m86_r3/tb_phase_fsm_sync_bank_actual_records_diff.sv":
        "383b9cd59e5cee9056d1929a2700fda112b276ac826184435024ced9756c0e30",
    "dc_handoff/filelists/date_m86_r3_phase_fsm_sync_bank_vcs.f":
        "0799d78b26f5ada92c4a59c4f9d77d057cd07fc57cfc19bc4b760abb8f63af95",
    "dc_handoff/filelists/date_m86_r3_phase_fsm_actual_records_diff_vcs.f":
        "cf6f22da0522eda0098d735d2562637e09b2108540113182d28878ce39d2c082",
    "contracts/m86_r3_phase_fsm_actual_records_vcs_contract_r1_20260823.json":
        "519fdf647d1016a17cf51e6daeea73d4648e65735966c968bf57eb8ce0689e5f",
    "results/m85_canonical_74b_phase_metadata_r1_20260823/m85_phase_metadata_74b.bin":
        "52b700b1c17172ae5a2d08acacfd9c5bac007893332f9afd9f23c29636e468a0",
    "dc_handoff/scripts/run_vcs_m86_r3_phase_fsm_actual_records_sva.sh":
        "6bbd563195858dc4589bdd9c0f4536dc8f846ddfeef99a6a32b145cb2e921dd7",
}
EXTERNAL_SHA = {
    "records": "6de1521b2ee91281eadd5945d6f69b45df2cf5f1e2cc0c93834df4ec4c87190d",
    "offsets": "1cddfc800ba18569b9c0d7f4c193f4b07ddf9046fa7688a1859f6c3e448bf30c",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def key_values(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def no_failure_signature(text):
    return not re.search(
        r"failed at|Offending|^Error|^Fatal|watchdog timeout", text,
        flags=re.IGNORECASE | re.MULTILINE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hw-root", type=Path, required=True)
    parser.add_argument("--exact-rerun", type=Path, required=True)
    parser.add_argument("--boundary-run", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--offsets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    identities = {}
    for relative, expected in EXPECTED_SHA.items():
        observed = sha256(args.hw_root / relative)
        require(observed == expected, "exact-SHA drift: " + relative)
        identities[relative] = observed
    external = {
        "records": sha256(args.records),
        "offsets": sha256(args.offsets),
    }
    require(external == EXTERNAL_SHA, "external binary identity drift")

    exact_receipt = key_values(args.exact_rerun / "RUN_COMPLETE.txt")
    require(exact_receipt.get("status") ==
            "PASS_M86_R3_PHASE_FSM_ACTUAL_RECORD_DIFFERENTIAL_VCS_SVA",
            "exact rerun receipt mismatch")
    for name in ("compile_directed.rc", "compile_actual.rc",
                 "sim_directed.rc", "sim_actual.rc"):
        require((args.exact_rerun / name).read_text().strip() == "0",
                name + " is nonzero")
    directed_text = (args.exact_rerun / "sim_directed.raw.log").read_text(
        encoding="utf-8")
    actual_text = (args.exact_rerun / "sim_actual.raw.log").read_text(
        encoding="utf-8")
    directed_pass = (
        "PASS M86-R3 phase-fsm triple_contention=3 payload_accepts=461 "
        "phase_accepts=1 descriptor_accepts=128 outputs=128 "
        "bank_issues=384 bank_responses=384 bounded_loader_wait=8 "
        "silent_deadlock=0"
    )
    actual_pass = (
        "PASS M86-R3 actual-record differential phases=1728 "
        "descriptors=221184 outputs=221184 beats=835383 escape=1 "
        "stress_phases=128 backpressure_cycles=5215 "
        "fifo_full_cycles=4900 r1_cycle_mismatches=0"
    )
    require(directed_pass in directed_text and no_failure_signature(directed_text),
            "directed exact rerun failed")
    require(actual_pass in actual_text and no_failure_signature(actual_text),
            "actual exact rerun failed")

    boundary_receipt = key_values(args.boundary_run / "RUN_COMPLETE.txt")
    require(boundary_receipt.get("status") ==
            "PASS_M86_R3_INDEPENDENT_BOUNDARY_ERROR_RESET_HAMMER",
            "boundary receipt mismatch")
    require(boundary_receipt.get("repeated_descriptor_accepts") == "128",
            "repeated descriptor attack missing")
    require((args.boundary_run / "compile.rc").read_text().strip() == "0",
            "boundary compile rc nonzero")
    require((args.boundary_run / "sim.rc").read_text().strip() == "0",
            "boundary sim rc nonzero")
    boundary_text = (args.boundary_run / "sim.raw.log").read_text(
        encoding="utf-8")
    boundary_match = re.search(
        r"PASS M86-R3 independent boundary triple_states=3 "
        r"rows_459_460_461=3 descriptors_127_128_129=3 "
        r"early_commit_wait=3 late_commit_wait=3 drain_stall=6 "
        r"held_loader_wait=(\d+) fault_classes=3 reset_classes=5 "
        r"repeated_descriptor_accepts=128 onehot_checks=(\d+) "
        r"issue=(\d+) response=(\d+)", boundary_text)
    require(boundary_match is not None and no_failure_signature(boundary_text),
            "independent boundary run failed")

    previous_oracle_path = args.hw_root / (
        "reviews/m86_sync_banked_guarded_pwp_independent_hammer_r1_20260823/"
        "m86_independent_oracle.json")
    require(sha256(previous_oracle_path) ==
            "d6ce4267984f7fc710a924823c33ed6857d93af6bd3b258b3b6a2785dac2ff80",
            "prior independent R1 oracle identity drift")
    previous_oracle = json.loads(previous_oracle_path.read_text(encoding="utf-8"))
    require(previous_oracle["status"] ==
            "PASS_M86_SCOPED_FUNCTIONAL_EVIDENCE_WITH_UNCHARGED_LOADER",
            "prior independent R1 oracle not PASS")
    reconstructed = previous_oracle["independent_binary_reconstruction"]
    require(reconstructed["phases"] == 1728 and
            reconstructed["descriptors"] == 221184 and
            reconstructed["bank_read_beats_including_escape_control"] == 835383,
            "prior independent oracle count drift")

    actual_tb = (args.hw_root / (
        "tb_m86_r3/tb_phase_fsm_sync_bank_actual_records_diff.sv"
    )).read_text(encoding="utf-8")
    require("sync_banked_guarded_pwp_frontend reference_r1" in actual_tb,
            "actual TB no longer instantiates same-source R1 reference")
    require("r1_cycle_mismatches=0" in actual_tb,
            "actual TB differential PASS field missing")

    contract = json.loads((args.hw_root / (
        "contracts/m86_r3_phase_fsm_actual_records_vcs_contract_r1_20260823.json"
    )).read_text(encoding="utf-8"))
    boundary = contract["claim_boundary"]
    for field in ("rtl_cycle_speedup", "paper_ppa_ready",
                  "system_speedup", "headline"):
        require(boundary[field] is False, "contract admits " + field)

    result = {
        "schema": "m86_r3_independent_machine_audit_v1",
        "status": "PASS_M86_R3_SCOPED_PHASE_FSM_WITH_NONINDEPENDENT_LOCKSTEP_DIFF",
        "identity": {
            "production_exact_sha256": identities,
            "external_sha256": external,
            "runner_self_identified_in_manifest": True,
        },
        "exact_source_recompile_rerun": {
            "directed": "PASS",
            "actual_records": "PASS",
            "phases": 1728,
            "descriptors": 221184,
            "outputs": 221184,
            "bank_issues": 835383,
            "r1_cycle_mismatches": 0,
            "backpressure_cycles": 5215,
            "fifo_full_cycles": 4900,
        },
        "differential_independence_assessment": {
            "r3_and_reference_use_same_r1_rtl_source": True,
            "same_input_and_output_ready_stream": True,
            "independent_output_decoder_inside_r3_actual_tb": False,
            "independent_cycle_oracle": False,
            "valid_interpretation": (
                "lockstep transparency regression for the R3 control wrapper; "
                "it can detect wrapper-induced cycle differences but not a bug "
                "shared by both R1 instances"
            ),
            "separate_prior_r1_binary_address_signed_oracle_pinned": True,
            "prior_oracle_sha256": sha256(previous_oracle_path),
            "prior_oracle_regular_fetches": reconstructed["regular_bank_fetches"],
            "prior_oracle_cross_row_fetches": reconstructed["cross_row_fetches"],
        },
        "independent_boundary_hammer": {
            "status": "PASS",
            "triple_states": 3,
            "row_boundaries": [459, 460, 461],
            "descriptor_boundaries": [127, 128, 129],
            "early_commit_wait_cycles": 3,
            "late_commit_wait_cycles": 3,
            "drain_stall_cycles": 6,
            "held_loader_wait_after_release": int(boundary_match.group(1)),
            "fault_classes": 3,
            "reset_classes": 5,
            "repeated_same_descriptor_accepts": 128,
            "onehot_checks": int(boundary_match.group(2)),
            "issues_before_destructive_reset": int(boundary_match.group(3)),
            "responses_before_destructive_reset": int(boundary_match.group(4)),
        },
        "claim_boundary": {
            "three_channel_fsm_functional": True,
            "loader_wait_bound_requires_eventual_output_ready": True,
            "descriptor_identity_or_order_enforced": False,
            "double_buffer_or_load_execute_overlap": False,
            "real_escape_fallback": False,
            "compiled_sram_macro": False,
            "rtl_cycle_speedup": False,
            "system_speedup": False,
            "paper_ppa_ready": False,
            "headline": False,
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "actual_phases": 1728,
        "boundary_onehot_checks": int(boundary_match.group(2)),
        "diff_independent": False,
        "repeated_descriptor_accepts": 128,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
