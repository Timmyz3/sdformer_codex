#!/usr/bin/env python3
"""Fail-closed identity, receipt, and arbitration truth-table audit for M86-R2."""

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
    "rtl_m86_r2/arbitrated_sync_banked_guarded_pwp_frontend.sv":
        "099abd4d43d49d5ee2b1e6ec90430334ac317bfec2c3766e9ca780ac313fe22c",
    "verif_m86_r2/arbitrated_sync_banked_guarded_pwp_frontend_assertions.sv":
        "6509c7796f9bccd90ce59252fc1e5e908afab940ed1fc41b69efacb643a5f5d6",
    "tb_m86_r2/tb_arbitrated_sync_banked_guarded_pwp_frontend.sv":
        "1651dc8973fcd44c3229a83f393e9c1bc3da9689e51e0b152c995f98f28880a1",
    "dc_handoff/filelists/date_m86_r2_arbitrated_sync_banked_guarded_pwp_vcs.f":
        "8b290f146d922d4432cc69dc18c24d6247f978090699bc777ccb9ab12883a9c5",
    "contracts/m86_r2_arbitrated_sync_bank_vcs_contract_r1_20260823.json":
        "087312d3401af00c981dbda4f36c36a5cd6d3c983b5348fcbcc614926595fcda",
    "dc_handoff/scripts/run_vcs_m86_r2_arbitrated_sync_bank_sva.sh":
        "fb93c22eff402e5181d611253f589763f4ef1384bdc4d53e988a9b768bcb4e6b",
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


def selection(phase_loaded, payload_valid, descriptor_valid):
    payload_selected = payload_valid and (
        (not phase_loaded) or (not descriptor_valid))
    descriptor_selected = descriptor_valid and (
        phase_loaded or (not payload_valid))
    return payload_selected, descriptor_selected


def key_values(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hw-root", type=Path, required=True)
    parser.add_argument("--sealed-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    identities = {}
    for relative, expected in EXPECTED_SHA.items():
        observed = sha256(args.hw_root / relative)
        require(observed == expected, "exact-SHA drift: " + relative)
        identities[relative] = observed

    truth_table = []
    for phase_loaded in (False, True):
        for payload_valid in (False, True):
            for descriptor_valid in (False, True):
                payload_selected, descriptor_selected = selection(
                    phase_loaded, payload_valid, descriptor_valid)
                require(not (payload_selected and descriptor_selected),
                        "selection is not onehot")
                if payload_valid and descriptor_valid:
                    require(payload_selected == (not phase_loaded),
                            "payload contention priority drift")
                    require(descriptor_selected == phase_loaded,
                            "descriptor contention priority drift")
                truth_table.append({
                    "phase_loaded": phase_loaded,
                    "payload_valid": payload_valid,
                    "descriptor_valid": descriptor_valid,
                    "payload_selected": payload_selected,
                    "descriptor_selected": descriptor_selected,
                })

    complete = key_values(args.sealed_run / "RUN_COMPLETE.txt")
    require(complete.get("status") ==
            "PASS_M86_R2_ARBITRATION_DIRECTED_VCS_SVA",
            "sealed run status mismatch")
    require(complete.get("silent_deadlock_cycles") == "0",
            "sealed deadlock counter mismatch")
    for field in ("actual_record_replay", "compiled_sram_macro",
                  "real_escape_fallback", "rtl_cycle_speedup",
                  "paper_ppa_ready", "system_speedup", "headline"):
        require(complete.get(field) == "false",
                "claim boundary drift: " + field)

    sim_text = (args.sealed_run / "sim.raw.log").read_text(encoding="utf-8")
    pass_line = (
        "PASS M86-R2 arbitration loaded_descriptor_wins=1 "
        "unloaded_loader_wins=1 payload_accepts=462 "
        "descriptor_accepts=1 outputs=1 bank_issues=3 "
        "bank_responses=3 silent_deadlock=0"
    )
    require(pass_line in sim_text, "sealed PASS line missing")
    require(not re.search(
        r"failed at|Offending|^Error|^Fatal|watchdog timeout", sim_text,
        flags=re.IGNORECASE | re.MULTILINE),
        "failure signature in sealed simulation")
    covers = {}
    for cover in ("cp_unloaded_contention", "cp_loaded_contention",
                  "cp_legal_output"):
        match = re.search(cover + r",\s+\d+ attempts,\s+(\d+) match",
                          sim_text)
        require(match is not None and int(match.group(1)) > 0,
                "missing cover: " + cover)
        covers[cover] = int(match.group(1))

    contract = json.loads((args.hw_root / (
        "contracts/m86_r2_arbitrated_sync_bank_vcs_contract_r1_20260823.json"
    )).read_text(encoding="utf-8"))
    not_admitted = " ".join(contract["claim_boundary"]["not_admitted"])
    require("starvation freedom" in not_admitted,
            "contract no longer excludes starvation freedom")
    require("full actual-record replay" in not_admitted,
            "contract no longer excludes actual replay")
    boundary = contract["claim_boundary"]
    for field in ("rtl_cycle_speedup", "paper_ppa_ready",
                  "system_speedup", "headline"):
        require(boundary[field] is False, "contract admits " + field)

    result = {
        "schema": "m86_r2_independent_identity_receipt_truth_table_v1",
        "status": "PASS_M86_R2_EXACT_TWO_CHANNEL_CONTENTION_REPAIR_ONLY",
        "exact_sha256": identities,
        "sealed_run": {
            "receipt_sha256": sha256(args.sealed_run / "RUN_COMPLETE.txt"),
            "sim_log_sha256": sha256(args.sealed_run / "sim.raw.log"),
            "simv_sha256": sha256(args.sealed_run / "simv"),
            "pass_line": pass_line,
            "covers": covers,
        },
        "independent_selection_truth_table": truth_table,
        "scope": {
            "r1_payload_descriptor_exact_trigger_closed": True,
            "phase_load_channel_arbitrated": False,
            "starvation_freedom": False,
            "actual_record_replay": False,
            "compiled_sram_macro": False,
            "rtl_cycle_speedup": False,
            "system_speedup": False,
            "headline": False,
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "truth_table_rows": len(truth_table),
        "loaded_both_winner": "descriptor",
        "unloaded_both_winner": "payload",
    }, sort_keys=True))


if __name__ == "__main__":
    main()
